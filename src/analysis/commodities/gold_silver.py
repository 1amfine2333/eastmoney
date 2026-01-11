"""
Gold & Silver Analyst - Multi-Agent System
==========================================
Orchestrates the analysis of Gold (XAU) and Silver (XAG) using:
1. Data Agent: YFinance (Global) + Akshare (China)
2. Research Agent: Tavily Search
3. Quantitative Agent: Historical Backtesting
4. Reasoning Agent: LLM Synthesis
"""

import sys
import os
from datetime import datetime
import pandas as pd
import mplfinance as mpf
import tempfile

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_sources.yfinance_api import YFinanceAPI
from src.data_sources.web_search import WebSearch
from src.llm.client import get_llm_client
from src.llm.prompts import GOLD_SILVER_ANALYSIS_PROMPT_TEMPLATE
from src.analysis.commodities.quantitative import QuantitativeAnalyst

class GoldSilverAnalyst:
    def __init__(self):
        self.web_search = WebSearch()
        self.llm = get_llm_client()
        self.quant = QuantitativeAnalyst()
        self.today = datetime.now().strftime("%Y-%m-%d")

    def _generate_chart(self, df: pd.DataFrame, title: str) -> str:
        """
        Generates a candle chart and returns the path.
        """
        try:
            filename = f"{tempfile.gettempdir()}/{title}_{self.today}.png"
            mpf.plot(df, type='candle', style='yahoo', title=title, savefig=filename)
            return filename
        except Exception as e:
            print(f"Chart generation failed: {e}")
            return ""

    def _analyze_technical_indicators(self, df: pd.DataFrame) -> str:
        """
        Calculates richer technical indicators: MA, RSI, Bollinger Bands, Support/Resistance.
        """
        if df.empty:
            return "Technical data unavailable."
        
        latest = df.iloc[-1]
        close = df['Close']
        
        # 1. Moving Averages
        ma20 = close.rolling(window=20).mean().iloc[-1]
        ma60 = close.rolling(window=60).mean().iloc[-1]
        trend = "Bullish (MA20 > MA60)" if ma20 > ma60 else "Bearish (MA20 < MA60)"
        
        # 2. RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 3. Bollinger Bands (20, 2)
        std20 = close.rolling(window=20).std().iloc[-1]
        upper_band = ma20 + (2 * std20)
        lower_band = ma20 - (2 * std20)
        
        # 4. Support/Resistance (Recent High/Low)
        recent_high = close.tail(30).max()
        recent_low = close.tail(30).min()
        
        return f"""
        - Latest Close: {latest['Close']:.2f}
        - Trend: {trend}
        - MA20: {ma20:.2f} | MA60: {ma60:.2f}
        - RSI (14): {rsi:.2f} (Overbought > 70, Oversold < 30)
        - Bollinger Bands: Upper {upper_band:.2f}, Lower {lower_band:.2f}
        - 30-Day Range: {recent_low:.2f} (Support) - {recent_high:.2f} (Resistance)
        """

    def _format_sources(self, news_results: list) -> str:
        """
        Formats the list of sources for the report footer.
        """
        if not news_results:
            return ""
        
        output = ["\n\n## 📚 引用来源 (Reference Sources)"]
        for idx, item in enumerate(news_results, 1):
            title = item.get('title', 'Unknown Title')
            url = item.get('url', '#')
            output.append(f"{idx}. [{title}]({url})")
        return "\n".join(output)

    def analyze(self, asset_type: str = "gold", user_id: int = None) -> str:
        """
        Main analysis pipeline.
        asset_type: 'gold' or 'silver'
        """
        asset_name = "Gold" if asset_type.lower() == "gold" else "Silver"
        symbol = "GC=F" if asset_type.lower() == "gold" else "SI=F"
        
        print(f"\n{'='*60}")
        print(f"🏆 Starting Deep Analysis for {asset_name} (User: {user_id})...")
        print(f"{ '='*60}")
        
        # 1. Data Collection
        print("  📊 Fetching Market Data...")
        price_history = YFinanceAPI.get_price_history(symbol, period="2y")
        macro_data_dict = YFinanceAPI.get_macro_data()
        
        current_price = "N/A"
        if not price_history.empty:
            current_price = f"{price_history['Close'].iloc[-1]:.2f}"
            self._generate_chart(price_history.tail(100), asset_name)
            
        # 2. Macro Analysis
        # Try to estimate Real Yield
        us_10y = macro_data_dict.get('US_10Y_YIELD', 0)
        # Assumed inflation expectation if data missing (e.g. 2.3%) or use fixed for now
        # Ideally we fetch T10YIE
        real_yield_est = us_10y - 2.3 
        
        macro_text = f"""
        - US 10Y Nominal Yield: {us_10y}%
        - Est. Real Yield (using 2.3% inflation anchor): {real_yield_est:.2f}%
        - Dollar Index (DXY): {macro_data_dict.get('DOLLAR_INDEX', 'N/A')}
        - Macro Context: Fed Policy, Inflation Expectations, Central Bank Buying
        """
        
        # 3. Quantitative Backtest
        print("  🧮 Running Quantitative Backtest...")
        quant_signal_text = "Insufficient data."
        backtest_win_rate = "N/A"
        avg_return = "N/A"
        
        ratio_df = YFinanceAPI.get_gold_silver_ratio(period="5y")
        if not ratio_df.empty:
            current_ratio = ratio_df['Ratio'].iloc[-1]
            
            if asset_type.lower() == 'silver':
                threshold = 80
                signal_type = 'above'
                backtest_res = self.quant.backtest_gold_silver_ratio(
                    ratio_df['Ratio'], ratio_df['Silver'], threshold, signal_type
                )
                if backtest_res.get('win_rate'):
                    backtest_win_rate = f"{backtest_res['win_rate']}%"
                    avg_return = f"{backtest_res['avg_return']}%"
                    quant_signal_text = f"""
                    - Strategy: Gold/Silver Ratio > {threshold} (Buy Silver)
                    - Current Ratio: {current_ratio:.2f}
                    - Signal Active: {"YES" if current_ratio > threshold else "NO"}
                    - Win Rate (5y): {backtest_win_rate}
                    - Avg Return (30d): {avg_return}
                    """
            else:
                # Gold
                quant_signal_text = f"- Current Gold/Silver Ratio: {current_ratio:.2f} (Neutral zone 50-80). No extreme signal triggered."
                backtest_win_rate = "N/A (No Signal)"
                avg_return = "N/A"

        # 4. Deep Research
        print("  🌍 Conducting Deep Web Research...")
        search_query = f"{asset_name} price forecast 2025 analysis"
        if asset_type.lower() == 'silver':
            search_query += " solar demand deficit inventory"
        elif asset_type.lower() == 'gold':
            search_query += " geopolitical risk central bank buying real rates"
            
        news_results = self.web_search.search_news(search_query, max_results=6)
        
        # Richer Fundamental Data for LLM
        fundamental_text_parts = []
        for item in news_results:
            title = item.get('title', '')
            content = item.get('content', '')[:200] # First 200 chars of snippet
            fundamental_text_parts.append(f"- **{title}**: {content}...")
        fundamental_text = "\n".join(fundamental_text_parts)
        
        # 5. Technical Analysis
        tech_text = self._analyze_technical_indicators(price_history)

        # 6. LLM Synthesis
        print("  🤖 Synthesizing Report...")
        
        # Values for placeholders in the prompt template that LLM should fill or use
        # For the Output Format example:
        # {recommendation} -> "{recommendation}" (Literal)
        # {confidence_score} -> "{confidence_score}" (Literal)
        # {target_price} -> "{target_price}" (Literal)
        # {risk_level} -> "{risk_level}" (Literal)
        
        # But {win_rate} and {avg_return} are used in the prompt instructions too, so we fill them. 
        
        prompt = GOLD_SILVER_ANALYSIS_PROMPT_TEMPLATE.format(
            asset_name=asset_name,
            symbol=symbol,
            current_price=current_price,
            macro_data=macro_text,
            fundamental_data=fundamental_text,
            technical_analysis=tech_text,
            quant_signal=quant_signal_text,
            win_rate=backtest_win_rate,
            avg_return=avg_return,
            report_date=self.today,
            # Output format placeholders - keep as literals for LLM to fill
            recommendation="{recommendation}",
            confidence_score="{confidence_score}",
            target_price="{target_price}",
            risk_level="{risk_level}"
        )
        
        report = self.llm.generate_content(prompt)
        
        # Append Sources
        report += self._format_sources(news_results)
        
        # Save logic updated for user isolation
        # src/analysis/commodities/gold_silver.py -> src/analysis/commodities -> src/analysis -> src -> root
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if user_id:
            output_dir = os.path.join(base_dir, "reports", str(user_id), "commodities")
        else:
            output_dir = os.path.join(base_dir, "reports", "commodities")
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{output_dir}/{self.today}_{timestamp}_commodities_{asset_type}_{asset_name}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"  ✅ Deep Report saved to {filename}")
        return report

if __name__ == "__main__":
    analyst = GoldSilverAnalyst()
    analyst.analyze("gold")
    analyst.analyze("silver")
