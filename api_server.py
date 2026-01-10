import os
import sys
import json
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import glob

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.pre_market import PreMarketAnalyst
from src.analysis.post_market import PostMarketAnalyst
from src.analysis.sentiment.dashboard import SentimentDashboard
from src.analysis.commodities.gold_silver import GoldSilverAnalyst
from src.analysis.dashboard import DashboardService
from src.data_sources.akshare_api import search_funds
from src.storage.db import init_db, get_active_funds, get_all_funds, upsert_fund, delete_fund, get_fund_by_code
from src.scheduler.manager import scheduler_manager
from src.report_gen import save_report
from src.data_sources.akshare_api import get_all_fund_list
from datetime import datetime

# --- Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    print("Starting Scheduler...")
    scheduler_manager.start()
    yield
    # Shutdown logic if needed

app = FastAPI(title="EastMoney Report API", lifespan=lifespan)

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
ENV_FILE = os.path.join(BASE_DIR, ".env")
MARKET_FUNDS_CACHE = os.path.join(CONFIG_DIR, "market_funds_cache.json")

# --- Models ---

class ReportSummary(BaseModel):
    filename: str
    date: str
    mode: str # 'pre' or 'post'
    fund_code: Optional[str] = None
    fund_name: Optional[str] = None
    is_summary: bool = True # True if it's a run_all report or Summary

class FundItem(BaseModel):
    code: str
    name: str
    style: Optional[str] = "Unknown"
    focus: Optional[List[str]] = []
    pre_market_time: Optional[str] = None
    post_market_time: Optional[str] = None
    pre_market_time: Optional[str] = None # HH:MM
    post_market_time: Optional[str] = None # HH:MM
    is_active: bool = True

class SettingsUpdate(BaseModel):
    llm_provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None

class GenerateRequest(BaseModel):
    fund_code: Optional[str] = None

class CommodityAnalyzeRequest(BaseModel):
    asset: str # "gold" or "silver"

# --- Helpers ---
def sanitize_data(data):
    """Recursively replace NaN/Inf and non-JSON types (like pd.NA) for JSON compliance."""
    import math
    import pandas as pd
    import numpy as np
    
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(v) for v in data]
    elif pd.isna(data): # Handles None, np.nan, pd.NA, pd.NaT
        return None
    elif isinstance(data, (np.float64, np.float32, float)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (np.int64, np.int32, int)):
        return int(data)
    elif isinstance(data, (datetime, pd.Timestamp)):
        return data.strftime('%Y-%m-%d %H:%M:%S')
    return data

def load_env_file():
    env_vars = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

def save_env_file(updates: Dict[str, str]):
    # Read existing lines to preserve comments/order
    lines = []
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    # Map of key -> line_index
    key_map = {}
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("#") and "=" in line:
            k = line.split("=", 1)[0].strip()
            key_map[k] = i
    
    for key, value in updates.items():
        if value is None: continue # Skip if not provided
        
        new_line = f"{key}={value}\n"
        if key in key_map:
            lines[key_map[key]] = new_line
        else:
            lines.append(new_line)
            
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

# --- Endpoints ---

@app.get("/api/dashboard/overview")
async def get_dashboard_overview():
    try:
        service = DashboardService(REPORT_DIR)
        return service.get_full_dashboard()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/commodities/analyze")
async def analyze_commodity(request: CommodityAnalyzeRequest):
    try:
        analyst = GoldSilverAnalyst()
        # This is a synchronous call. For a real app, use background tasks.
        # But for this CLI tool, it's acceptable.
        report = analyst.analyze(request.asset)
        return {"status": "success", "message": f"{request.asset} analysis complete"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports", response_model=List[ReportSummary])
async def list_reports():
    if not os.path.exists(REPORT_DIR):
        return []
    
    # Load funds for name mapping
    fund_map = {}
    try:
        funds = get_all_funds()
        for f in funds:
            fund_map[f['code']] = f['name']
    except:
        pass

    reports = []
    # Match all .md files
    files = glob.glob(os.path.join(REPORT_DIR, "*.md"))
    files.sort(key=os.path.getmtime, reverse=True) # Newest first
    
    for f in files:
        filename = os.path.basename(f)
        try:
            # Remove extension
            name_no_ext = os.path.splitext(filename)[0]
            
            # Old format compatibility: remove _report suffix if strictly present in the middle (unlikely now but good for safety)
            # Actually, let's just split by "_"
            parts = name_no_ext.split("_")
            
            # Expected formats:
            # 1. YYYY-MM-DD_mode_SUMMARY
            # 2. YYYY-MM-DD_mode_CODE_NAME
            # 3. YYYY-MM-DD_mode_report (Old legacy)
            
            if len(parts) < 2: continue
            
            date_str = parts[0]
            mode = parts[1]
            
            if "SUMMARY" in name_no_ext or (len(parts) > 2 and parts[2] == "report"):
                 reports.append(ReportSummary(
                    filename=filename, 
                    date=date_str, 
                    mode=mode,
                    is_summary=True,
                    fund_name="Market Overview"
                ))
            elif len(parts) >= 3:
                 # Format: YYYY-MM-DD_mode_CODE_NAME...
                 code = parts[2]
                 # Join the rest as name (in case name has underscores, though sanitized)
                 extracted_name = "_".join(parts[3:]) if len(parts) > 3 else ""
                 
                 # Use extracted name if available, else fallback to map, else code
                 final_name = extracted_name if extracted_name else fund_map.get(code, code)
                 
                 reports.append(ReportSummary(
                    filename=filename, 
                    date=date_str, 
                    mode=mode, 
                    fund_code=code,
                    fund_name=final_name,
                    is_summary=False
                ))
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            continue
            
    return reports

@app.get("/api/reports/{filename}")
async def get_report(filename: str):
    # Try root report dir first
    filepath = os.path.join(REPORT_DIR, filename)
    if not os.path.exists(filepath):
        # Try sentiment subdir
        filepath = os.path.join(REPORT_DIR, "sentiment", filename)
    
    if not os.path.exists(filepath):
        # Try commodities subdir
        filepath = os.path.join(REPORT_DIR, "commodities", filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report not found")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    return {"content": content}

@app.get("/api/commodities/reports", response_model=List[ReportSummary])
async def list_commodity_reports():
    commodities_dir = os.path.join(REPORT_DIR, "commodities")
    if not os.path.exists(commodities_dir):
        return []

    reports = []
    files = glob.glob(os.path.join(commodities_dir, "*.md"))
    files.sort(key=os.path.getmtime, reverse=True)
    
    for f in files:
        filename = os.path.basename(f)
        try:
            # Format: YYYY-MM-DD_HHMMSS_commodities_CODE_NAME.md (New)
            # Format: YYYY-MM-DD_commodities_CODE_NAME.md (Old)
            name_no_ext = os.path.splitext(filename)[0]
            parts = name_no_ext.split("_")
            
            # Identify format based on parts length and 'commodities' position
            # New format: len >= 5, parts[2] == 'commodities'
            # Old format: len >= 4, parts[1] == 'commodities'
            
            if len(parts) >= 5 and parts[2] == 'commodities':
                date_str = parts[0]
                time_str = parts[1]
                code = parts[3]
                name = "_".join(parts[4:])
                formatted_date = f"{date_str} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                
                reports.append(ReportSummary(
                    filename=filename,
                    date=formatted_date,
                    mode="commodities",
                    fund_code=code,
                    fund_name=name,
                    is_summary=False
                ))
            elif len(parts) >= 4 and parts[1] == 'commodities':
                # Old format fallback
                date_str = parts[0]
                code = parts[2]
                name = "_".join(parts[3:])
                
                reports.append(ReportSummary(
                    filename=filename,
                    date=date_str,
                    mode="commodities",
                    fund_code=code,
                    fund_name=name,
                    is_summary=False
                ))
        except Exception as e:
            print(f"Error parsing commodity report {filename}: {e}")
            continue
            
    return reports

@app.get("/api/market-funds")
async def search_market_funds(query: str = ""):
    funds = []
    
    # Check cache
    if os.path.exists(MARKET_FUNDS_CACHE):
        # Check modified time (e.g. 24h)
        try:
            mtime = os.path.getmtime(MARKET_FUNDS_CACHE)
            if (datetime.now().timestamp() - mtime) < 86400:
                with open(MARKET_FUNDS_CACHE, 'r', encoding='utf-8') as f:
                    funds = json.load(f)
        except Exception as e:
            print(f"Cache read error: {e}")
    
    # If no funds from cache, fetch
    if not funds:
        print("Fetching fresh fund list from AkShare...")
        funds = get_all_fund_list()
        # Cache it
        if funds:
            try:
                if not os.path.exists(CONFIG_DIR):
                    os.makedirs(CONFIG_DIR)
                with open(MARKET_FUNDS_CACHE, 'w', encoding='utf-8') as f:
                    json.dump(funds, f, ensure_ascii=False)
            except Exception as e:
                print(f"Cache write error: {e}")
            
    # Filter
    if not query:
        return funds[:20] # Return top 20 if no query
        
    query = query.lower()
    results = []
    for f in funds:
        # Match code (startswith) or name (contains) or pinyin (contains)
        # Safe access with .get()
        f_code = str(f.get('code', ''))
        f_name = str(f.get('name', ''))
        f_pinyin = str(f.get('pinyin', ''))
        
        if (f_code.startswith(query) or 
            query in f_name.lower() or 
            query in f_pinyin.lower()):
            results.append(f)
            
            if len(results) >= 50: # Limit results
                break
                
    return results

@app.post("/api/generate/{mode}")
async def generate_report_endpoint(mode: str, request: GenerateRequest = None):
    if mode not in ["pre", "post"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'pre' or 'post'.")
    
    fund_code = request.fund_code if request else None

    try:
        print(f"Generating {mode}-market report... (Fund: {fund_code if fund_code else 'ALL'})")
        
        # Use Scheduler's worker logic to execute immediately
        if fund_code:
            scheduler_manager.run_analysis_task(fund_code, mode)
            return {"status": "success", "message": f"Task triggered for {fund_code}"}
        else:
            # If generating for ALL funds (Manual Trigger)
            # We can iterate and run them
            funds = get_active_funds()
            results = []
            for fund in funds:
                try:
                    scheduler_manager.run_analysis_task(fund['code'], mode)
                    results.append(fund['code'])
                except:
                    pass
            return {"status": "success", "message": f"Triggered tasks for {len(results)} funds"}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/indices")
def get_market_indices():
    """获取主要市场指数实时快照 (使用全球指数接口)"""
    try:
        import akshare as ak
        # 获取全球指数行情快照
        indices_df = ak.index_global_spot_em()
        
        target_names = [
            "上证指数", "深证成指", "创业板指",
            "恒生指数", "日经225", "纳斯达克", "标普500"
        ]
        results = []
        
        for name in target_names:
            row = indices_df[indices_df['名称'] == name]
            if not row.empty:
                results.append({
                    "name": name,
                    "code": str(row.iloc[0].get('代码', '')),
                    "price": float(row.iloc[0]['最新价']),
                    "change_pct": float(row.iloc[0]['涨跌幅']),
                    "change_val": float(row.iloc[0]['涨跌额'])
                })
        
        return sanitize_data(results)
    except Exception as e:
        print(f"Error fetching indices via index_global_spot_em: {e}")
        return []

@app.get("/api/funds")
async def get_funds_endpoint():
    try:
        funds = get_all_funds()
        # Convert JSON string focus to list if needed, DB layer handles dict conversion but 'focus' might be string
        result = []
        for f in funds:
            item = dict(f)
            if isinstance(item.get('focus'), str):
                try:
                    item['focus'] = json.loads(item['focus'])
                except:
                    item['focus'] = []
            
            # Convert SQLite Row to dict fully
            result.append(FundItem(
                code=item['code'],
                name=item['name'],
                style=item.get('style'),
                focus=item['focus'],
                pre_market_time=item.get('pre_market_time'),
                post_market_time=item.get('post_market_time'),
                is_active=bool(item.get('is_active', True))
            ))
        return result
    except Exception as e:
        print(f"Error reading funds: {e}")
        return []

@app.post("/api/funds")
async def save_funds(funds: List[FundItem]):
    try:
        # Bulk upsert using DB
        for fund in funds:
            fund_dict = fund.model_dump()
            # Ensure JSON serialization for list fields if needed by DB wrapper, 
            # but src.storage.db.upsert_fund handles dicts. 
            # Ideally db.upsert_fund expects a dict matching the schema.
            # We need to ensure focus is list or string? 
            # In db.py, it handles conversion.
            upsert_fund(fund_dict)
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/funds/{code}")
async def upsert_fund_endpoint(code: str, fund: FundItem):
    try:
        fund_dict = fund.model_dump()
        upsert_fund(fund_dict)
        
        # Also update scheduler!
        scheduler_manager.add_fund_jobs(fund_dict)
        
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/funds/{code}")
async def delete_fund_endpoint(code: str):
    try:
        delete_fund(code)
        # Remove from scheduler
        scheduler_manager.remove_fund_jobs(code)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/funds")
async def search_market_funds(q: str):
    if not q:
        return []
    try:
        results = search_funds(q)
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

@app.get("/api/settings")
async def get_settings():
    env = load_env_file()
    def mask(key):
        val = env.get(key, "")
        if len(val) > 8:
            return val[:4] + "..." + val[-4:]
        return val
    
    return {
        "llm_provider": env.get("LLM_PROVIDER", "gemini"),
        "gemini_api_key_masked": mask("GEMINI_API_KEY"),
        "openai_api_key_masked": mask("OPENAI_API_KEY"),
        "tavily_api_key_masked": mask("TAVILY_API_KEY")
    }

@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    updates = {}
    if settings.llm_provider:
        updates["LLM_PROVIDER"] = settings.llm_provider
    if settings.gemini_api_key:
        updates["GEMINI_API_KEY"] = settings.gemini_api_key
    if settings.openai_api_key:
        updates["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.tavily_api_key:
        updates["TAVILY_API_KEY"] = settings.tavily_api_key
        
    save_env_file(updates)
    return {"status": "success"}

@app.post("/api/sentiment/analyze")
async def analyze_sentiment():
    try:
        dashboard = SentimentDashboard()
        # This might take time, ideally should be async or background task if very slow.
        # For now, synchronous call is acceptable for a single user tool.
        report = dashboard.run_analysis()
        
        # Save to file as well (optional, but good for history)
        sentiment_dir = os.path.join(REPORT_DIR, "sentiment")
        if not os.path.exists(sentiment_dir):
            os.makedirs(sentiment_dir)
            
        filename = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(sentiment_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
            
        return {"report": report, "filename": filename}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiment/reports")
async def list_sentiment_reports():
    sentiment_dir = os.path.join(REPORT_DIR, "sentiment")
    if not os.path.exists(sentiment_dir):
        return []
    
    reports = []
    files = glob.glob(os.path.join(sentiment_dir, "sentiment_*.md"))
    files.sort(key=os.path.getmtime, reverse=True)
    
    for f in files:
        filename = os.path.basename(f)
        try:
            # Format 1: sentiment_YYYYMMDD_HHMMSS.md -> parts len = 3
            # Format 2: sentiment_YYYYMMDD.md -> parts len = 2
            
            parts = filename.replace(".md", "").split("_")
            date_str = ""
            time_str = ""
            
            if len(parts) >= 3:
                date_str = parts[1]
                time_str = parts[2]
                formatted_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}"
            elif len(parts) == 2:
                date_str = parts[1]
                formatted_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            else:
                continue

            reports.append({
                "filename": filename,
                "date": formatted_time
            })
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue
            
    return reports


@app.get("/api/market/funds/{code}/details")
async def get_fund_market_details(code: str):
    """获取基金在市场上的详细信息（经理、规模、业绩、持仓等）"""
    try:
        # 使用 akshare 获取基金基本信息
        import akshare as ak
        # 1. 基础信息
        df_info = ak.fund_individual_basic_info_xq(symbol=code)
        raw_info = dict(zip(df_info.iloc[:, 0], df_info.iloc[:, 1]))
        
        # 模糊匹配键名的辅助函数
        def get_val(d, *keys):
            for k in d.keys():
                for target in keys:
                    if target in str(k):
                        return d[k]
            return "---"

        # 标准化常用键名，确保前端能拿到数据
        info_dict = {
            "manager": get_val(raw_info, "经理"),
            "size": get_val(raw_info, "规模"),
            "est_date": get_val(raw_info, "成立"),
            "type": get_val(raw_info, "类型"),
            "company": get_val(raw_info, "公司"),
            "rating": get_val(raw_info, "评级"),
            "nav": get_val(raw_info, "净值", "价格")
        }
        
        # 如果 basic_info 没拿到 nav，尝试从历史净值中拿最新的
        if info_dict["nav"] == "---":
            try:
                # 使用 em 接口获取历史净值作为兜底
                df_nav = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
                if df_nav is not None and not df_nav.empty:
                    latest_row = df_nav.iloc[-1]
                    # 优先寻找包含'单位净值'的列，其次寻找包含'净值'的列，最后用索引
                    nav_col = None
                    for col in df_nav.columns:
                        if '单位净值' in str(col):
                            nav_col = col
                            break
                    if not nav_col:
                        for col in df_nav.columns:
                            if '净值' in str(col):
                                nav_col = col
                                break
                    if not nav_col and len(df_nav.columns) >= 2:
                        nav_col = df_nav.columns[1]
                    
                    if nav_col:
                        info_dict["nav"] = str(latest_row[nav_col])
            except:
                pass

        # 2. 业绩评价 (使用更全的 雪球 接口)
        perf_list = []
        try:
            df_perf = ak.fund_individual_achievement_xq(symbol=code)
            if df_perf is not None and not df_perf.empty:
                # 映射列名以匹配前端
                # 原列名: ['业绩类型', '周期', '本产品区间收益', '本产品最大回撤', '周期收益同类排名']
                for _, row in df_perf.iterrows():
                    perf_list.append({
                        "时间范围": row.get("周期", "---"),
                        "收益率": row.get("本产品区间收益", 0.0),
                        "同类排名": row.get("周期收益同类排名", "---")
                    })
        except:
            pass

        # 3. 持仓分析 (作为板块信息的参考)
        portfolio = []
        try:
            df_hold = ak.fund_portfolio_hold_em(symbol=code)
            if df_hold is not None and not df_hold.empty:
                portfolio = df_hold.head(10).to_dict(orient='records')
        except:
            pass

        return sanitize_data({
            "info": info_dict,
            "performance": perf_list,
            "portfolio": portfolio
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error fetching fund details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/funds/{code}/nav")
async def get_fund_nav_history(code: str):
    """获取基金历史净值（用于绘图）"""
    try:
        import akshare as ak
        # 获取单位净值走势
        df_nav = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df_nav is not None and not df_nav.empty:
            # 只取最近 100 条数据以减轻传输压力
            df_nav = df_nav.tail(100).copy()
            
            # 统一列名: 优先根据关键词匹配，兜底使用索引
            found_date = False
            found_value = False
            for col in df_nav.columns:
                col_str = str(col)
                if '日期' in col_str or 'date' in col_str.lower():
                    df_nav = df_nav.rename(columns={col: 'date'})
                    found_date = True
                if '净值' in col_str or 'value' in col_str.lower():
                    df_nav = df_nav.rename(columns={col: 'value'})
                    found_value = True
            
            # 兜底逻辑
            if not found_date and len(df_nav.columns) >= 1:
                df_nav.columns.values[0] = 'date'
            if not found_value and len(df_nav.columns) >= 2:
                df_nav.columns.values[1] = 'value'
            
            # 确保 value 是数值型
            df_nav['value'] = pd.to_numeric(df_nav['value'], errors='coerce')
            # 移除 NaN
            df_nav = df_nav.dropna(subset=['value'])
            
            return sanitize_data(df_nav[['date', 'value']].to_dict(orient='records'))
        return []
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error fetching NAV history: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    # Need to run init_db if running directly without lifespan
    init_db()
    scheduler_manager.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)