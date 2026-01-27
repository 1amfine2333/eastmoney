"""
Stock management endpoints.
"""
import json
import asyncio
from typing import List
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
import akshare as ak
import pandas as pd

from app.models.stocks import StockItem, StockAnalyzeRequest
from app.models.auth import User
from app.core.dependencies import get_current_user
from app.core.cache import stock_feature_cache
from app.core.utils import sanitize_for_json, sanitize_data
from src.storage.db import get_all_stocks, upsert_stock, delete_stock
from src.data_sources.akshare_api import get_stock_realtime_quote

router = APIRouter(prefix="/api/stocks", tags=["Stocks"])


@router.get("", response_model=List[StockItem])
async def get_stocks_endpoint(current_user: User = Depends(get_current_user)):
    """Get all stocks for current user."""
    try:
        stocks = get_all_stocks(user_id=current_user.id)
        result = []
        for s in stocks:
            item = dict(s)
            result.append(StockItem(
                code=item['code'],
                name=item['name'],
                market=item.get('market', ''),
                sector=item.get('sector', ''),
                pre_market_time=item.get('pre_market_time'),
                post_market_time=item.get('post_market_time'),
                is_active=bool(item.get('is_active', True))
            ))
        return result
    except Exception as e:
        print(f"Error reading stocks: {e}")
        return []


@router.post("")
async def save_stocks(stocks: List[StockItem], current_user: User = Depends(get_current_user)):
    """Save multiple stocks."""
    try:
        for stock in stocks:
            stock_dict = stock.model_dump()
            upsert_stock(stock_dict, user_id=current_user.id)
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{code}")
async def upsert_stock_endpoint(code: str, stock: StockItem, current_user: User = Depends(get_current_user)):
    """Create or update a stock."""
    try:
        stock_dict = stock.model_dump()
        upsert_stock(stock_dict, user_id=current_user.id)
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{code}")
async def delete_stock_endpoint(code: str, current_user: User = Depends(get_current_user)):
    """Delete a stock."""
    try:
        delete_stock(code, user_id=current_user.id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{code}/quote")
async def get_stock_quote_endpoint(code: str, current_user: User = Depends(get_current_user)):
    """Get real-time stock quote."""
    try:
        data = await asyncio.to_thread(get_stock_realtime_quote, code)
        return sanitize_data(data)
    except Exception as e:
        print(f"Error fetching quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{code}/analyze")
async def analyze_stock_endpoint(
    code: str,
    request: StockAnalyzeRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate AI analysis for a stock."""
    try:
        from src.analysis.stock import StockAnalyst

        analyst = StockAnalyst()
        report = await asyncio.to_thread(
            analyst.analyze,
            stock_code=code,
            mode=request.mode,
            user_id=current_user.id
        )
        return {"status": "success", "report": report}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{code}/financial-summary")
async def get_stock_financial_summary(code: str, current_user: User = Depends(get_current_user)):
    """Get financial summary from financial indicators."""
    try:
        # Check cache first
        cache_key = f"financial_summary_{code}"
        cached = stock_feature_cache.get(cache_key, ttl_minutes=720)
        if cached:
            return cached

        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, lambda: ak.stock_financial_analysis_indicator(symbol=code))

        if df is None or df.empty:
            return {"message": "No financial data available"}

        df_recent = df.head(5)

        result = {
            "code": code,
            "periods": [],
            "metrics": {}
        }

        key_metrics = [
            "净资产收益率(%)", "总资产报酬率(%)", "资产负债率(%)",
            "流动比率", "速动比率", "存货周转天数(天)",
            "应收账款周转天数(天)", "营业利润率(%)", "净利润率(%)"
        ]

        for col in df_recent.columns:
            if col not in ['日期']:
                result["metrics"][col] = []

        for idx, row in df_recent.iterrows():
            result["periods"].append(str(row.get('日期', idx)))
            for col in df_recent.columns:
                if col not in ['日期'] and col in result["metrics"]:
                    val = row[col]
                    if pd.isna(val):
                        result["metrics"][col].append(None)
                    else:
                        result["metrics"][col].append(float(val))

        stock_feature_cache.set(cache_key, result)
        return sanitize_for_json(result)
    except Exception as e:
        print(f"Error fetching financial summary for {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{code}/shareholders")
async def get_stock_shareholders(code: str, current_user: User = Depends(get_current_user)):
    """Get top 10 shareholders from circulating shareholders."""
    try:
        cache_key = f"shareholders_{code}"
        cached = stock_feature_cache.get(cache_key, ttl_minutes=360)
        if cached:
            return cached

        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, lambda: ak.stock_circulate_stock_holder(symbol=code))

        if df is None or df.empty:
            return {"message": "No shareholder data available"}

        latest_date = df['变动日期'].max() if '变动日期' in df.columns else df.iloc[0, 0]
        df_latest = df[df['变动日期'] == latest_date] if '变动日期' in df.columns else df.head(10)

        shareholders = []
        for _, row in df_latest.iterrows():
            shareholders.append({
                "rank": int(row.get('序号', 0)),
                "name": row.get('股东名称', ''),
                "type": row.get('股东性质', ''),
                "shares": float(row.get('持股数量', 0)),
                "ratio": float(row.get('占流通股比例', 0)),
                "change": row.get('变动比例', ''),
            })

        result = {
            "code": code,
            "report_date": str(latest_date),
            "shareholders": shareholders[:10]
        }

        stock_feature_cache.set(cache_key, result)
        return sanitize_for_json(result)
    except Exception as e:
        print(f"Error fetching shareholders for {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{code}/fund-holdings")
async def get_stock_fund_holdings(code: str, current_user: User = Depends(get_current_user)):
    """Get fund holdings for a stock."""
    try:
        cache_key = f"fund_holdings_{code}"
        cached = stock_feature_cache.get(cache_key, ttl_minutes=360)
        if cached:
            return cached

        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, lambda: ak.stock_report_fund_hold_detail(symbol=code))

        if df is None or df.empty:
            return {"message": "No fund holding data available"}

        df_recent = df.head(20)

        holdings = []
        for _, row in df_recent.iterrows():
            holdings.append({
                "fund_code": row.get('基金代码', ''),
                "fund_name": row.get('基金简称', ''),
                "shares": float(row.get('持有股票数', 0)) if row.get('持有股票数') else 0,
                "value": float(row.get('持有股票市值', 0)) if row.get('持有股票市值') else 0,
                "ratio_nav": float(row.get('占基金净值比', 0)) if row.get('占基金净值比') else 0,
                "report_date": str(row.get('报告日期', '')),
            })

        result = {
            "code": code,
            "holdings": holdings
        }

        stock_feature_cache.set(cache_key, result)
        return sanitize_for_json(result)
    except Exception as e:
        print(f"Error fetching fund holdings for {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{code}/quantitative")
async def get_stock_quantitative_indicators(code: str, current_user: User = Depends(get_current_user)):
    """Get quantitative indicators (momentum, volatility, moving averages)."""
    try:
        cache_key = f"quant_{code}"
        cached = stock_feature_cache.get(cache_key, ttl_minutes=30)
        if cached:
            return cached

        loop = asyncio.get_running_loop()

        df = await loop.run_in_executor(
            None,
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date=(datetime.now() - timedelta(days=365)).strftime('%Y%m%d'), adjust="qfq")
        )

        if df is None or df.empty or len(df) < 30:
            return {"message": "Insufficient data for quantitative analysis"}

        # Calculate indicators
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['MA60'] = df['收盘'].rolling(window=60).mean()

        df['RSI'] = _calculate_rsi(df['收盘'], 14)

        df['daily_return'] = df['收盘'].pct_change()
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std() * (252 ** 0.5)

        latest = df.iloc[-1]
        prev_month = df.iloc[-21] if len(df) >= 21 else df.iloc[0]
        prev_3month = df.iloc[-63] if len(df) >= 63 else df.iloc[0]

        result = {
            "code": code,
            "latest_price": float(latest['收盘']),
            "moving_averages": {
                "MA5": float(latest['MA5']) if pd.notna(latest['MA5']) else None,
                "MA20": float(latest['MA20']) if pd.notna(latest['MA20']) else None,
                "MA60": float(latest['MA60']) if pd.notna(latest['MA60']) else None,
            },
            "momentum": {
                "1m_return": float((latest['收盘'] / prev_month['收盘'] - 1) * 100),
                "3m_return": float((latest['收盘'] / prev_3month['收盘'] - 1) * 100),
            },
            "rsi_14": float(latest['RSI']) if pd.notna(latest['RSI']) else None,
            "volatility_20d": float(latest['volatility_20d']) if pd.notna(latest['volatility_20d']) else None,
            "analysis_date": datetime.now().strftime('%Y-%m-%d'),
        }

        # Add signal interpretations
        signals = []
        if result['rsi_14']:
            if result['rsi_14'] > 70:
                signals.append({"type": "warning", "message": "RSI显示超买"})
            elif result['rsi_14'] < 30:
                signals.append({"type": "opportunity", "message": "RSI显示超卖"})

        if result['moving_averages']['MA5'] and result['moving_averages']['MA20']:
            if result['moving_averages']['MA5'] > result['moving_averages']['MA20']:
                signals.append({"type": "bullish", "message": "5日均线在20日均线上方"})
            else:
                signals.append({"type": "bearish", "message": "5日均线在20日均线下方"})

        result['signals'] = signals

        stock_feature_cache.set(cache_key, result)
        return sanitize_for_json(result)
    except Exception as e:
        print(f"Error calculating quant indicators for {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


@router.get("/{code}/ai-diagnosis")
async def get_stock_ai_diagnosis(
    code: str,
    force_refresh: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Get AI-powered stock diagnosis with five-dimension scoring."""
    try:
        cache_key = f"ai_diagnosis_{code}"
        if not force_refresh:
            cached = stock_feature_cache.get(cache_key, ttl_minutes=60)
            if cached:
                return cached

        # Gather all data in parallel
        financial_task = get_stock_financial_summary(code, current_user)
        quant_task = get_stock_quantitative_indicators(code, current_user)

        financial_data = await financial_task
        quant_data = await quant_task

        # Get basic quote
        quote = await asyncio.to_thread(get_stock_realtime_quote, code)

        # Calculate 5-dimension scores
        scores = _calculate_stock_diagnosis_scores(quote, financial_data, quant_data)

        result = {
            "code": code,
            "name": quote.get('name', code) if quote else code,
            "total_score": scores['total'],
            "max_score": 100,
            "grade": _get_grade(scores['total']),
            "dimensions": [
                {"name": "基本面", "score": scores['fundamental'], "max": 20},
                {"name": "技术面", "score": scores['technical'], "max": 20},
                {"name": "资金面", "score": scores['capital'], "max": 20},
                {"name": "估值", "score": scores['valuation'], "max": 20},
                {"name": "动量", "score": scores['momentum'], "max": 20},
            ],
            "recommendations": scores['recommendations'],
            "analyzed_at": datetime.now().isoformat(),
        }

        stock_feature_cache.set(cache_key, result)
        return sanitize_for_json(result)
    except Exception as e:
        print(f"Error generating AI diagnosis for {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_stock_diagnosis_scores(quote, financial_data, quant_data):
    """Calculate diagnosis scores for a stock."""
    scores = {
        'fundamental': 10,
        'technical': 10,
        'capital': 10,
        'valuation': 10,
        'momentum': 10,
        'recommendations': []
    }

    # Fundamental score based on financial data
    if financial_data and 'metrics' in financial_data:
        metrics = financial_data['metrics']
        roe_vals = metrics.get('净资产收益率(%)', [])
        if roe_vals and roe_vals[0] is not None:
            if roe_vals[0] > 15:
                scores['fundamental'] = 18
            elif roe_vals[0] > 10:
                scores['fundamental'] = 15
            elif roe_vals[0] > 5:
                scores['fundamental'] = 12
            else:
                scores['fundamental'] = 8
                scores['recommendations'].append("ROE偏低，盈利能力待提升")

    # Technical score based on quantitative indicators
    if quant_data and 'rsi_14' in quant_data:
        rsi = quant_data.get('rsi_14')
        if rsi:
            if 40 <= rsi <= 60:
                scores['technical'] = 16
            elif 30 <= rsi <= 70:
                scores['technical'] = 14
            else:
                scores['technical'] = 10
                if rsi > 70:
                    scores['recommendations'].append("RSI偏高，注意回调风险")
                elif rsi < 30:
                    scores['recommendations'].append("RSI偏低，可能存在超卖机会")

    # Valuation score based on PE/PB
    if quote:
        pe = quote.get('pe')
        if pe:
            try:
                pe_val = float(pe)
                if 0 < pe_val < 20:
                    scores['valuation'] = 18
                elif 20 <= pe_val < 40:
                    scores['valuation'] = 14
                elif pe_val >= 40:
                    scores['valuation'] = 8
                    scores['recommendations'].append("PE较高，估值偏贵")
            except:
                pass

    # Momentum score based on return
    if quant_data and 'momentum' in quant_data:
        ret_1m = quant_data['momentum'].get('1m_return', 0)
        if ret_1m > 10:
            scores['momentum'] = 17
        elif ret_1m > 0:
            scores['momentum'] = 14
        elif ret_1m > -10:
            scores['momentum'] = 10
        else:
            scores['momentum'] = 7
            scores['recommendations'].append("近期走势偏弱")

    # Capital score (simplified)
    if quote:
        turnover_rate = quote.get('turnover_rate')
        if turnover_rate:
            try:
                tr = float(turnover_rate)
                if 1 < tr < 5:
                    scores['capital'] = 16
                elif 5 <= tr < 10:
                    scores['capital'] = 13
                else:
                    scores['capital'] = 10
            except:
                pass

    scores['total'] = sum([
        scores['fundamental'], scores['technical'],
        scores['capital'], scores['valuation'], scores['momentum']
    ])

    return scores


def _get_grade(score):
    """Convert score to grade."""
    if score >= 80:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 40:
        return "C"
    else:
        return "D"


@router.get("/{code}/money-flow")
async def get_stock_money_flow(code: str, current_user: User = Depends(get_current_user)):
    """Get capital flow data for a stock."""
    try:
        cache_key = f"money_flow_{code}"
        cached = stock_feature_cache.get(cache_key, ttl_minutes=15)
        if cached:
            return cached

        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            None,
            lambda: ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("6") else "sz")
        )

        if df is None or df.empty:
            return {"message": "No money flow data available"}

        df_recent = df.head(20)

        flows = []
        for _, row in df_recent.iterrows():
            flows.append({
                "date": str(row.get('日期', '')),
                "main_in": float(row.get('主力净流入-净额', 0)) if pd.notna(row.get('主力净流入-净额')) else 0,
                "retail_in": float(row.get('小单净流入-净额', 0)) if pd.notna(row.get('小单净流入-净额')) else 0,
                "change_pct": float(row.get('涨跌幅', 0)) if pd.notna(row.get('涨跌幅')) else 0,
            })

        result = {
            "code": code,
            "flows": flows
        }

        stock_feature_cache.set(cache_key, result)
        return sanitize_for_json(result)
    except Exception as e:
        print(f"Error fetching money flow for {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch-quotes")
async def get_batch_stock_quotes(codes: str, current_user: User = Depends(get_current_user)):
    """Get quotes for multiple stocks at once."""
    try:
        code_list = [c.strip() for c in codes.split(",") if c.strip()]
        if not code_list:
            return {"quotes": []}

        # Limit to prevent abuse
        if len(code_list) > 50:
            code_list = code_list[:50]

        results = []
        for code in code_list:
            try:
                quote = await asyncio.to_thread(get_stock_realtime_quote, code)
                if quote:
                    results.append(quote)
            except:
                pass

        return sanitize_data({"quotes": results})
    except Exception as e:
        print(f"Error in batch quotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
