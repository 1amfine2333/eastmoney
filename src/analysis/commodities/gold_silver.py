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
import numpy as np
import akshare as ak

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_sources.yfinance_api import YFinanceAPI
from src.data_sources import tushare_client
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

    def _latest_date_str(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "N/A"
        idx = df.index[-1]
        try:
            return idx.strftime("%Y-%m-%d")
        except Exception:
            return str(idx)

    def _calc_realized_vol(self, close: pd.Series, window: int = 20) -> float:
        if close is None or len(close) < window + 1:
            return 0.0
        returns = close.pct_change().dropna()
        if len(returns) < window:
            return 0.0
        return float(returns.tail(window).std() * np.sqrt(252) * 100)

    def _calc_atr(self, df: pd.DataFrame, window: int = 14) -> float:
        if df is None or df.empty or len(df) < window + 1:
            return 0.0
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else 0.0

    def _calc_corr(self, s1: pd.Series, s2: pd.Series, window: int = 60) -> float:
        if s1 is None or s2 is None:
            return 0.0
        df = pd.DataFrame({"a": s1, "b": s2}).dropna()
        if len(df) < window + 1:
            return 0.0
        r1 = df["a"].pct_change()
        r2 = df["b"].pct_change()
        corr = r1.tail(window).corr(r2.tail(window))
        return float(corr) if pd.notna(corr) else 0.0

    def _get_shfe_main_contract(self, asset_type: str) -> dict:
        """
        Get SHFE main contract snapshot for gold/silver.
        Prefer TuShare fut_daily; fallback to AkShare if unavailable.
        """
        prefix = "AU" if asset_type.lower() == "gold" else "AG"
        result = {"source": "tushare", "date": None}

        def _sf(val) -> float:
            try:
                if val is None or pd.isna(val):
                    return 0.0
                return float(val)
            except Exception:
                return 0.0

        try:
            # 1) Get active contracts from TuShare
            basic = tushare_client.tushare_call_with_retry("fut_basic", exchange="SHFE")
            if basic is None or basic.empty:
                raise ValueError("fut_basic empty")

            basic = basic[basic["symbol"].str.startswith(prefix)].copy()
            basic["list_date"] = pd.to_datetime(basic["list_date"], errors="coerce")
            basic["delist_date"] = pd.to_datetime(basic["delist_date"], errors="coerce")
            today = pd.to_datetime(datetime.now().strftime("%Y%m%d"))
            active = basic[(basic["list_date"] <= today) & (basic["delist_date"] >= today)]
            if active.empty:
                raise ValueError("no active contracts")

            # 2) Sample nearest contracts and pick highest OI as main
            candidates = active.sort_values("delist_date").head(5)
            best_row = None
            best_oi = -1

            start_date = (datetime.now() - pd.Timedelta(days=10)).strftime("%Y%m%d")
            end_date = datetime.now().strftime("%Y%m%d")

            for _, row in candidates.iterrows():
                ts_code = row.get("ts_code")
                if not ts_code:
                    continue
                df = tushare_client.tushare_call_with_retry(
                    "fut_daily",
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is None or df.empty:
                    continue
                latest = df.sort_values("trade_date", ascending=False).iloc[0]
                oi = _sf(latest.get("oi", 0))
                if oi > best_oi:
                    best_oi = oi
                    best_row = latest

            if best_row is not None:
                close_val = _sf(best_row.get("close", 0))
                pre_close = _sf(best_row.get("pre_close", 0))
                pct_chg = _sf(best_row.get("pct_chg", 0))
                if pct_chg == 0.0 and pre_close > 0:
                    pct_chg = (close_val / pre_close - 1) * 100
                result.update({
                    "date": str(best_row.get("trade_date")),
                    "contract": best_row.get("ts_code"),
                    "close": close_val,
                    "pct_chg": pct_chg,
                    "oi": _sf(best_row.get("oi", 0)),
                    "oi_chg": _sf(best_row.get("oi_chg", 0)),
                })
                return result
        except Exception as e:
            print(f"TuShare SHFE snapshot error: {e}")

        # Fallback to AkShare main contract (Sina)
        try:
            result["source"] = "akshare"
            symbol = "AU0" if asset_type.lower() == "gold" else "AG0"
            df = ak.futures_main_sina(
                symbol=symbol,
                start_date=(datetime.now() - pd.Timedelta(days=35)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "??": "date",
                    "???": "close",
                    "???": "open",
                    "???": "high",
                    "???": "low",
                    "???": "vol",
                    "???": "oi",
                })
                df = df.sort_values("date")
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) >= 2 else None
                pct = 0.0
                if prev is not None and prev.get("close"):
                    pct = (float(last.get("close")) / float(prev.get("close")) - 1) * 100
                result.update({
                    "date": str(last.get("date")),
                    "contract": symbol,
                    "close": float(last.get("close", 0)),
                    "pct_chg": float(pct),
                    "oi": float(last.get("oi", 0)),
                    "oi_chg": 0.0,
                })
                return result
        except Exception as e:
            print(f"AkShare SHFE snapshot error: {e}")

        return result
    def _get_comex_inventory_snapshot(self, asset_type: str) -> dict:
        """
        Get COMEX inventory snapshot via AkShare (no TuShare interface).
        """
        result = {"source": "akshare", "date": None}
        try:
            symbol = "\u9ec4\u91d1" if asset_type.lower() == "gold" else "\u767d\u94f6"
            df = ak.futures_comex_inventory(symbol=symbol)
            if df is not None and not df.empty:
                df = df.sort_values("日期")
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) >= 2 else None
                inv_col = f"COMEX{symbol}库存量-吨"
                inv = float(last.get(inv_col, 0))
                delta = 0.0
                if prev is not None:
                    delta = inv - float(prev.get(inv_col, 0))
                result.update({
                    "date": str(last.get("日期")),
                    "inventory_ton": inv,
                    "delta_ton": delta,
                })
        except Exception as e:
            print(f"COMEX inventory error: {e}")
        return result

    def _get_etf_snapshot(self, asset_type: str) -> dict:
        """
        ETF snapshot (GLD/SLV) as a sentiment proxy.
        """
        result = {"source": "yfinance", "date": None}
        ticker = YFinanceAPI.TICKERS["GOLD_ETF"] if asset_type.lower() == "gold" else YFinanceAPI.TICKERS["SILVER_ETF"]
        df = YFinanceAPI.get_price_history(ticker, period="6mo")
        if df is None or df.empty:
            return result
        df = df.dropna()
        if df.empty:
            return result
        last = df.iloc[-1]
        prev_5 = df.iloc[-6] if len(df) >= 6 else None
        prev_20 = df.iloc[-21] if len(df) >= 21 else None
        result["date"] = self._latest_date_str(df)
        result["price"] = float(last.get("Close", 0))
        result["vol_20d"] = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0.0
        result["vol_60d"] = float(df["Volume"].tail(60).mean()) if "Volume" in df.columns else 0.0
        if prev_5 is not None and prev_5.get("Close"):
            result["chg_5d"] = (float(last.get("Close")) / float(prev_5.get("Close")) - 1) * 100
        if prev_20 is not None and prev_20.get("Close"):
            result["chg_20d"] = (float(last.get("Close")) / float(prev_20.get("Close")) - 1) * 100
        return result

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
        
        vol_20d = self._calc_realized_vol(close, window=20)
        atr_14 = self._calc_atr(df, window=14)
        atr_pct = (atr_14 / latest['Close'] * 100) if latest['Close'] else 0.0
        vol_20 = df['Volume'].tail(20).mean() if 'Volume' in df.columns else 0.0
        vol_60 = df['Volume'].tail(60).mean() if 'Volume' in df.columns else 0.0
        vol_trend = "Rising" if vol_20 > vol_60 else "Cooling"

        return f"""
        - Latest Close: {latest['Close']:.2f}
        - Trend: {trend}
        - MA20: {ma20:.2f} | MA60: {ma60:.2f}
        - RSI (14): {rsi:.2f} (Overbought > 70, Oversold < 30)
        - Bollinger Bands: Upper {upper_band:.2f}, Lower {lower_band:.2f}
        - 30-Day Range: {recent_low:.2f} (Support) - {recent_high:.2f} (Resistance)
        - Volatility (20D, ann.): {vol_20d:.2f}%
        - ATR(14): {atr_14:.2f} ({atr_pct:.2f}% of price)
        - Volume Trend (20D vs 60D): {vol_trend}
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
        print("  ?? Fetching Market Data...")
        price_history = YFinanceAPI.get_price_history(symbol, period="2y")
        macro_data_dict = YFinanceAPI.get_macro_data()
        dxy_hist = YFinanceAPI.get_price_history(YFinanceAPI.TICKERS['DOLLAR_INDEX'], period="1y")
        tnx_hist = YFinanceAPI.get_price_history(YFinanceAPI.TICKERS['US_10Y_YIELD'], period="1y")
        spx_hist = YFinanceAPI.get_price_history(YFinanceAPI.TICKERS['SP500'], period="1y")
        shfe_snapshot = self._get_shfe_main_contract(asset_type)
        etf_snapshot = self._get_etf_snapshot(asset_type)
        comex_inventory = self._get_comex_inventory_snapshot(asset_type)

        current_price = "N/A"
        price_date = "N/A"
        if not price_history.empty:
            current_price = f"{price_history['Close'].iloc[-1]:.2f}"
            price_date = self._latest_date_str(price_history)
            self._generate_chart(price_history.tail(100), asset_name)

        # 2. Macro Analysis
        # Try to estimate Real Yield
        us_10y = macro_data_dict.get('US_10Y_YIELD', 0)
        us_3m = macro_data_dict.get('US_3M_YIELD', 0)
        us_5y = macro_data_dict.get('US_5Y_YIELD', 0)
        us_30y = macro_data_dict.get('US_30Y_YIELD', 0)
        dxy = macro_data_dict.get('DOLLAR_INDEX', 0)
        spx = macro_data_dict.get('SP500', 0)
        macro_date = macro_data_dict.get('DATE', 'N/A')
        # Assumed inflation expectation if data missing (e.g. 2.3%) or use fixed for now
        # Ideally we fetch T10YIE
        real_yield_est = us_10y - 2.3
        term_spread = us_10y - us_3m if us_10y and us_3m else 0

        corr_dxy = self._calc_corr(price_history['Close'], dxy_hist['Close']) if not price_history.empty and not dxy_hist.empty else 0.0
        corr_yield = self._calc_corr(price_history['Close'], tnx_hist['Close']) if not price_history.empty and not tnx_hist.empty else 0.0
        corr_spx = self._calc_corr(price_history['Close'], spx_hist['Close']) if not price_history.empty and not spx_hist.empty else 0.0

        macro_text = f"""
        - Macro Data Date: {macro_date}
        - Price Date ({symbol}): {price_date}
        - US 10Y Nominal Yield: {us_10y:.2f}%
        - US 3M Yield: {us_3m:.2f}% | US 5Y: {us_5y:.2f}% | US 30Y: {us_30y:.2f}%
        - Term Spread (10Y-3M): {term_spread:.2f}%
        - Est. Real Yield (10Y - 2.3% anchor): {real_yield_est:.2f}% (proxy)
        - Dollar Index (DXY): {dxy if dxy else 'N/A'}
        - S&P 500: {spx if spx else 'N/A'}
        - 60D Corr vs DXY: {corr_dxy:.2f} | vs 10Y: {corr_yield:.2f} | vs S&P 500: {corr_spx:.2f}
        - SHFE Main ({shfe_snapshot.get('contract', 'N/A')}): {shfe_snapshot.get('close', 'N/A')} ({shfe_snapshot.get('pct_chg', 'N/A')}%) @ {shfe_snapshot.get('date', 'N/A')} [{shfe_snapshot.get('source', 'N/A')}]
        - ETF ({'GLD' if asset_type.lower()=='gold' else 'SLV'}): {etf_snapshot.get('price','N/A')} | 5D {etf_snapshot.get('chg_5d','N/A')}% | 20D {etf_snapshot.get('chg_20d','N/A')}% | Vol 20D/60D {etf_snapshot.get('vol_20d','N/A')}/{etf_snapshot.get('vol_60d','N/A')}
        - Macro Context: Fed Policy, Inflation Expectations, Central Bank Buying
        """

        # 3. Quantitative Backtest
        print("  ?? Running Quantitative Backtest...")
        quant_signal_text = "Insufficient data."
        backtest_win_rate = "N/A"
        avg_return = "N/A"

        ratio_df = YFinanceAPI.get_gold_silver_ratio(period="5y")
        if not ratio_df.empty:
            current_ratio = ratio_df['Ratio'].iloc[-1]
            ratio_mean = ratio_df['Ratio'].tail(252 * 3).mean() if len(ratio_df) >= 252 * 3 else ratio_df['Ratio'].mean()
            ratio_std = ratio_df['Ratio'].tail(252 * 3).std() if len(ratio_df) >= 252 * 3 else ratio_df['Ratio'].std()
            ratio_z = ((current_ratio - ratio_mean) / ratio_std) if ratio_std and ratio_std > 0 else 0.0

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
                    - Ratio Z-Score (3y): {ratio_z:.2f}
                    - Signal Active: {"YES" if current_ratio > threshold else "NO"}
                    - Win Rate (5y): {backtest_win_rate}
                    - Avg Return (30d): {avg_return}
                    """
            else:
                # Gold
                quant_signal_text = f"- Current Gold/Silver Ratio: {current_ratio:.2f} (Z-Score {ratio_z:.2f}). Neutral zone 50-80. No extreme signal triggered."
                backtest_win_rate = "N/A (No Signal)"
                avg_return = "N/A"

        # 4. Deep Research
        print("  ?? Conducting Deep Web Research...")
        search_query = f"{asset_name} price forecast 2025 analysis last 7 days"
        if asset_type.lower() == 'silver':
            search_query += " solar demand deficit inventory"
        elif asset_type.lower() == 'gold':
            search_query += " geopolitical risk central bank buying real rates"

        news_results = self.web_search.search_news(search_query, max_results=6)

        # Richer Fundamental Data for LLM
        fundamental_text_parts = []
        if comex_inventory.get("date"):
            fundamental_text_parts.append(
                f"- **COMEX??({asset_name})**: {comex_inventory.get('inventory_ton', 'N/A'):.2f} ? (? {comex_inventory.get('delta_ton', 0):.2f} ?) ?? {comex_inventory.get('date')} [{comex_inventory.get('source')}]"
            )
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
