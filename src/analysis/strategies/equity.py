from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
from .base_strategy import AnalysisStrategy
from src.data_sources.akshare_api import (
    get_fund_info,
    get_fund_holdings,
    get_market_indices,
    get_global_macro_summary,
    get_northbound_flow,
    get_industry_capital_flow,
    get_sector_performance,
    get_sector_performance_ths,
    get_stock_realtime_quote
)
from src.llm.prompts import PRE_MARKET_PROMPT_TEMPLATE, POST_MARKET_PROMPT_TEMPLATE

class EquityStrategy(AnalysisStrategy):
    """
    Strategy for Equity/Mixed Funds (Standard A-share Funds).
    """

    def collect_data(self, mode: str) -> Dict[str, Any]:
        data = {}
        if mode == 'pre':
            data['global_macro'] = self._collect_global_macro()
            data['northbound'], data['sector_flow'] = self._collect_capital_flow_pre(self.fund_info.get('focus', []))
            data['holdings_str'], data['holdings_deep'], data['holdings_list'] = self._collect_holdings_data_pre()
            data['policy'] = self._collect_policy_news(self.fund_info.get('focus', []))
        elif mode == 'post':
            data['market_data'] = self._collect_market_performance()
            data['fund_perf'] = self._collect_fund_performance()
            data['holdings_perf'], holdings_list = self._collect_holdings_performance_post()
            data['sector_data'] = self._collect_sector_data_post(self.fund_info.get('focus', []))
            data['capital_flow'] = self._collect_capital_flow_post()
            data['intraday_news'] = self._collect_intraday_news(holdings_list)
        return data

    def generate_report(self, mode: str, data: Dict[str, Any]) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        if mode == 'pre':
            prompt = PRE_MARKET_PROMPT_TEMPLATE.format(
                fund_name=self.fund_name,
                fund_code=self.fund_code,
                fund_style=self.fund_info.get('style', '混合型'),
                fund_focus=", ".join(self.fund_info.get('focus', [])),
                global_macro_data=data.get('global_macro'),
                northbound_data=data.get('northbound'),
                sector_flow_data=data.get('sector_flow'),
                holdings_data=data.get('holdings_str'),
                holdings_deep_dive=data.get('holdings_deep'),
                policy_news=data.get('policy'),
                report_date=today
            )
        else:
            prompt = POST_MARKET_PROMPT_TEMPLATE.format(
                fund_name=self.fund_name,
                fund_code=self.fund_code,
                market_data=data.get('market_data'),
                fund_performance=data.get('fund_perf'),
                holdings_performance=data.get('holdings_perf'),
                sector_data=data.get('sector_data'),
                capital_flow=data.get('capital_flow'),
                intraday_news=data.get('intraday_news'),
                report_date=today
            )
        
        report = self.llm.generate_content(prompt)
        return report + self.get_sources()

    # ==========================
    # Pre-Market Helper Methods
    # ==========================
    def _collect_global_macro(self) -> str:
        print("  📡 Collecting Global Macro Signals...")
        macro_data = get_global_macro_summary()
        output = []
        if macro_data.get("美股市场"):
            output.append("**隔夜美股:**")
            for name, d in macro_data["美股市场"].items():
                output.append(f"- {name}: {d.get('最新价', d.get('收盘'))} ({d.get('涨跌幅', d.get('涨跌'))})")
        
        if macro_data.get("A50期货"):
             # Simplify A50 handling
            a50 = macro_data["A50期货"]
            if isinstance(a50, dict) and '收盘' in a50:
                 output.append(f"\n**A50期货:** {a50.get('收盘')} (夜盘 {a50.get('夜盘涨跌幅')}%)")

        if macro_data.get("汇率"):
            output.append("\n**汇率:**")
            for name, d in macro_data["汇率"].items():
                output.append(f"- {name}: {d.get('买入价', d.get('最新价'))}")
        
        # News
        news = self.web_search.search_macro_events(max_results=3)
        for n in news:
            self._add_source("🌍 宏观新闻", n.get('title'), n.get('url'))
            
        return "\n".join(output) if output else "暂无宏观数据"

    def _collect_capital_flow_pre(self, focus: List[str]) -> tuple:
        print("  💰 Analyzing Capital Flow...")
        nb = get_northbound_flow()
        nb_str = f"最新: {nb.get('最新净流入', 'N/A')}, 5日: {nb.get('5日累计净流入', 'N/A')}亿" if nb else "暂无"
        
        sf = get_industry_capital_flow()
        sf_str = ""
        if sf.get('行业资金流向Top10'):
            top5 = sf['行业资金流向Top10'][:5]
            sf_str = "\n".join([f"- {i.get('名称')}: {i.get('今日主力净流入', i.get('主力净流入'))}" for i in top5])
        
        return nb_str, sf_str

    def _collect_holdings_data_pre(self) -> tuple:
        print("  📊 Analyzing Holdings (Pre)...")
        holdings_df = get_fund_holdings(self.fund_code)
        if holdings_df.empty: return "暂无", "暂无", []
        
        # Logic to extract top holdings... simplified for brevity
        # Assuming standard columns exists
        holdings_list = []
        output = []
        # Need to handle potential column name differences safely
        name_col = next((c for c in holdings_df.columns if '名称' in c), None)
        code_col = next((c for c in holdings_df.columns if '代码' in c), None)
        
        if name_col and code_col:
            for _, row in holdings_df.head(10).iterrows():
                name = row[name_col]
                code = str(row[code_col])
                holdings_list.append({'name': name, 'code': code})
                output.append(f"- {name}: {row.get('持仓市值', 'N/A')}")

        # Deep dive
        deep_out = []
        for h in holdings_list[:5]:
            # Search
            res = self.web_search.comprehensive_stock_search(h['name'])
            # Add sources
            if res.get('announcements'):
                for a in res['announcements']:
                    self._add_source("📢 公告", f"[{h['name']}] {a.get('title')}", a.get('url'))
                    deep_out.append(f"- {h['name']}公告: {a.get('title')}")
            if res.get('analyst_reports'):
                for r in res['analyst_reports']:
                    self._add_source("📊 研报", f"[{h['name']}] {r.get('title')}", r.get('url'))
                    deep_out.append(f"- {h['name']}研报: {r.get('title')}")

        return "\n".join(output), "\n".join(deep_out), holdings_list

    def _collect_policy_news(self, focus: List[str]) -> str:
        print("  📰 Searching Policy News...")
        out = []
        for f in focus[:3]:
            news = self.web_search.search_policy_news(f, max_results=2)
            for n in news:
                self._add_source("📜 政策", f"[{f}] {n.get('title')}", n.get('url'))
                out.append(f"- [{f}] {n.get('title')}")
        return "\n".join(out)

    # ==========================
    # Post-Market Helper Methods
    # ==========================
    def _collect_market_performance(self) -> str:
        print("  📈 Collecting Market Performance...")
        data = get_market_indices()
        return "\n".join([f"- {k}: {v.get('收盘', v.get('close'))} ({v.get('涨跌幅', v.get('change'))}%)" for k, v in data.items() if isinstance(v, dict)])

    def _collect_fund_performance(self) -> str:
        print("  💹 Collecting Fund NAV...")
        df = get_fund_info(self.fund_code)
        if df.empty: return "暂无"
        latest = df.iloc[0]
        return f"净值: {latest.get('单位净值')} (涨跌: {latest.get('日增长率')}%) Date: {latest.get('净值日期')}"

    def _collect_holdings_performance_post(self) -> tuple:
        print("  📊 Analyzing Holdings Performance...")
        holdings_df = get_fund_holdings(self.fund_code)
        if holdings_df.empty: return "暂无", []
        
        name_col = next((c for c in holdings_df.columns if '名称' in c), None)
        code_col = next((c for c in holdings_df.columns if '代码' in c), None)
        
        out = []
        h_list = []
        if name_col and code_col:
            for _, row in holdings_df.head(5).iterrows():
                name = row[name_col]
                code = str(row[code_col])
                h_list.append({'name': name, 'code': code})
                # Realtime quote
                q = get_stock_realtime_quote(code)
                price = q.get('最新价') if q else 'N/A'
                change = q.get('涨跌幅') if q else 'N/A'
                out.append(f"- {name}: {price} ({change}%)")
        return "\n".join(out), h_list

    def _collect_sector_data_post(self, focus: List[str]) -> str:
        print("  🏢 Analyzing Sectors (THS)...")
        out = []
        for f in focus[:3]:
            # Try THS first
            s = get_sector_performance_ths(f)
            if s:
                out.append(f"- {s.get('板块名称')}: 收盘{s.get('收盘价')} (涨跌: {s.get('涨跌幅')}%) [THS]")
            else:
                # Fallback to EM
                s = get_sector_performance(f)
                if s and isinstance(s, dict):
                    out.append(f"- {f}: {s.get('涨跌幅')}% (流入: {s.get('主力净流入')}) [EM]")

        print('板块数据: ',"\n".join(out))
        return "\n".join(out)

    def _collect_capital_flow_post(self) -> str:
        # Reusing pre-market logic roughly but for post context
        return "参见前文资金流向数据 (北向/行业)"

    def _collect_intraday_news(self, holdings: List[Dict]) -> str:
        print("  📰 Searching Intraday News...")
        out = []
        # Fund news
        news = self.web_search.search_news(f"{self.fund_name} 今日", max_results=2)
        for n in news:
            self._add_source("📰 基金新闻", n.get('title'), n.get('url'))
            out.append(f"- {n.get('title')}")
        # Holdings news
        for h in holdings[:3]:
            news = self.web_search.search_news(f"{h['name']} 今日 异动", max_results=1)
            for n in news:
                self._add_source("📊 持仓新闻", f"[{h['name']}] {n.get('title')}", n.get('url'))
                out.append(f"- {h['name']}: {n.get('title')}")
        return "\n".join(out)
