from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
from .base_strategy import AnalysisStrategy
from src.data_sources.yfinance_api import YFinanceAPI
from src.data_sources.akshare_api import get_fund_info, get_fund_holdings
from src.llm.prompts import (
    PRE_MARKET_COMMODITY_PROMPT_TEMPLATE, 
    POST_MARKET_COMMODITY_PROMPT_TEMPLATE
)

class CommodityStrategy(AnalysisStrategy):
    """
    Strategy for Commodity/Precious Metals Funds (Gold, Silver, Oil, etc.)
    """

    def _get_asset_type(self) -> str:
        """Determine if Gold, Silver, or Oil based on name."""
        name = self.fund_name
        if "白银" in name or "银" in name:
            return "silver"
        elif "油" in name:
            return "oil"
        else:
            return "gold" # Default to Gold for most commodity funds

    def collect_data(self, mode: str) -> Dict[str, Any]:
        data = {}
        asset_type = self._get_asset_type()
        
        # Symbol mapping
        symbol_map = {
            "gold": "GC=F",
            "silver": "SI=F",
            "oil": "CL=F"
        }
        symbol = symbol_map.get(asset_type, "GC=F")
        asset_name = asset_type.capitalize()

        print(f"  ⛏️ Executing Commodity Strategy ({asset_name})...")

        if mode == 'pre':
            # 1. Global Macro (Futures, DXY, Yields)
            print("  kb Fetching Global Macro Data...")
            macro_data = YFinanceAPI.get_macro_data()
            price_history = YFinanceAPI.get_price_history(symbol, period="5d")
            
            latest_price = "N/A"
            change = "N/A"
            if not price_history.empty:
                latest = price_history.iloc[-1]
                latest_price = f"{latest['Close']:.2f}"
                # Calculate change from previous close
                if len(price_history) >= 2:
                    prev = price_history.iloc[-2]['Close']
                    pct = ((latest['Close'] - prev) / prev) * 100
                    change = f"{pct:+.2f}%"

            data['global_macro'] = {
                'asset_price': latest_price,
                'asset_change': change,
                'us_10y': macro_data.get('US_10Y_YIELD', 'N/A'),
                'dxy': macro_data.get('DOLLAR_INDEX', 'N/A')
            }

            # 2. News Search
            print("  🌍 Searching Commodity News...")
            query = f"{asset_name} price overnight analysis fed rate cut"
            news = self.web_search.search_news(query, max_results=3)
            data['news'] = news
            for item in news:
                self._add_source("🌍 宏观/商品新闻", item.get('title'), item.get('url'))

        elif mode == 'post':
            # 1. Fund Performance
            print("  💹 Fetching Fund Performance...")
            fund_df = get_fund_info(self.fund_code)
            data['fund_nav'] = "N/A"
            if not fund_df.empty:
                latest = fund_df.iloc[0]
                data['fund_nav'] = f"{latest.get('单位净值', 'N/A')} ({latest.get('日增长率', 'N/A')}%)"

            # 2. Underlying Asset Performance (Intraday)
            # Fetch today's price action for the commodity
            price_history = YFinanceAPI.get_price_history(symbol, period="1d", interval="15m")
            data['underlying_perf'] = "Data Unavailable"
            if not price_history.empty:
                open_p = price_history.iloc[0]['Open']
                close_p = price_history.iloc[-1]['Close']
                pct = ((close_p - open_p) / open_p) * 100
                data['underlying_perf'] = f"Open: {open_p:.2f}, Latest: {close_p:.2f}, Change: {pct:+.2f}%"

            # 3. Post Market News
            print("  📰 Searching Post-Market News...")
            query = f"{asset_name} price movement reason today"
            news = self.web_search.search_news(query, max_results=3)
            data['news'] = news
            for item in news:
                self._add_source("📰 市场动态", item.get('title'), item.get('url'))

        return data

    def generate_report(self, mode: str, data: Dict[str, Any]) -> str:
        asset_type = self._get_asset_type()
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Format News
        news_text = ""
        if data.get('news'):
            news_text = "\n".join([f"- {n.get('title')}" for n in data['news']])
        
        if mode == 'pre':
            macro = data.get('global_macro', {})
            prompt = PRE_MARKET_COMMODITY_PROMPT_TEMPLATE.format(
                fund_name=self.fund_name,
                fund_code=self.fund_code,
                asset_name=asset_type.capitalize(),
                asset_price=macro.get('asset_price'),
                asset_change=macro.get('asset_change'),
                us_10y=macro.get('us_10y'),
                dxy=macro.get('dxy'),
                news_summary=news_text,
                report_date=today
            )
        else:
            prompt = POST_MARKET_COMMODITY_PROMPT_TEMPLATE.format(
                fund_name=self.fund_name,
                fund_code=self.fund_code,
                asset_name=asset_type.capitalize(),
                fund_nav=data.get('fund_nav'),
                underlying_perf=data.get('underlying_perf'),
                news_summary=news_text,
                report_date=today
            )

        report = self.llm.generate_content(prompt)
        return report + self.get_sources()
