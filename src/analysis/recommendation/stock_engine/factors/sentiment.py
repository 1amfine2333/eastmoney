"""
Stock Sentiment/Money Flow Factors - Capital flow indicators for stock recommendation.

Key factors:
- 5-day main inflow trend: Cumulative institutional money flow (not single day)
- Main inflow momentum: Is the inflow accelerating?
- North-bound capital signal: Hong Kong Stock Connect inflow
- Retail outflow ratio: Retail investors selling (institutions accumulating)

Design principle: Look for accumulation patterns, not already-pumped stocks
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

from src.data_sources.tushare_client import (
    normalize_ts_code,
    format_date_yyyymmdd,
    get_moneyflow_hsgt,
    get_latest_trade_date,
    tushare_call_with_retry,
)


class SentimentFactors:
    """
    Sentiment and money flow factor computation for stocks.

    Uses TuShare moneyflow API (2000+ points) for individual stock flows
    and moneyflow_hsgt (free) for northbound capital.
    """

    # Lookback periods
    FLOW_DAYS = 5
    TREND_DAYS = 10

    @classmethod
    def compute(cls, ts_code: str, trade_date: str) -> Dict:
        """
        Compute all sentiment/money flow factors for a stock.

        Args:
            ts_code: Stock code in TuShare format
            trade_date: Trade date in YYYYMMDD format

        Returns:
            Dict with sentiment factors
        """
        ts_code = normalize_ts_code(ts_code)

        factors = {
            'main_inflow_5d': None,
            'main_inflow_trend': None,
            'north_inflow_5d': None,
            'retail_outflow_ratio': None,
        }

        try:
            # Get money flow data for the stock
            flow_df = cls._get_stock_moneyflow(ts_code, trade_date)

            if flow_df is not None and not flow_df.empty:
                factors.update(cls._compute_flow_factors(flow_df))

            # Get northbound capital data (market-wide indicator)
            north_df = cls._get_northbound_data(trade_date)

            if north_df is not None and not north_df.empty:
                factors['north_inflow_5d'] = cls._compute_north_inflow(north_df)

        except Exception as e:
            print(f"Error computing sentiment factors for {ts_code}: {e}")

        return factors

    @classmethod
    def _get_stock_moneyflow(cls, ts_code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        Get stock-level money flow data from TuShare.

        Uses moneyflow API (requires 2000 points).
        """
        end_date = trade_date
        start_date = format_date_yyyymmdd(
            datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=cls.TREND_DAYS + 10)
        )

        df = tushare_call_with_retry(
            'moneyflow',
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and not df.empty:
            df = df.sort_values('trade_date', ascending=False)
            # Ensure numeric columns are numeric (TuShare may return strings)
            numeric_cols = [
                'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
                'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol',
                'net_mf_vol', 'net_mf_amount',
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    @classmethod
    def _get_northbound_data(cls, trade_date: str) -> Optional[pd.DataFrame]:
        """
        Get northbound capital flow data.

        Uses moneyflow_hsgt API (free).
        """
        end_date = trade_date
        start_date = format_date_yyyymmdd(
            datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=cls.TREND_DAYS + 10)
        )

        df = get_moneyflow_hsgt(start_date, end_date)

        if df is not None and not df.empty:
            df = df.sort_values('trade_date', ascending=False)
            # Ensure numeric columns are numeric
            numeric_cols = ['north_money', 'south_money', 'hgt', 'sgt', 'ggt_ss', 'ggt_sz']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    @classmethod
    def _compute_flow_factors(cls, df: pd.DataFrame) -> Dict:
        """
        Compute money flow factors from moneyflow data.

        TuShare moneyflow columns:
        - buy_sm_vol/sell_sm_vol: Small order (retail) buy/sell volume
        - buy_md_vol/sell_md_vol: Medium order volume
        - buy_lg_vol/sell_lg_vol: Large order (institutional) volume
        - buy_elg_vol/sell_elg_vol: Extra large order volume
        - net_mf_vol: Net money flow volume
        - net_mf_amount: Net money flow amount (yuan)
        """
        result = {}

        if len(df) < cls.FLOW_DAYS:
            return result

        recent = df.head(cls.FLOW_DAYS).copy()

        # 1. 5-day cumulative main (institutional) inflow
        # Main = large + extra large orders
        if all(col in df.columns for col in ['buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol']):
            main_inflow = (
                (recent['buy_lg_vol'] + recent['buy_elg_vol']) -
                (recent['sell_lg_vol'] + recent['sell_elg_vol'])
            ).sum()

            # Normalize to relative scale (vs average daily volume)
            avg_vol = recent['buy_lg_vol'].mean() + recent['buy_elg_vol'].mean()
            if avg_vol > 0:
                result['main_inflow_5d'] = round(main_inflow / avg_vol, 4)
            else:
                result['main_inflow_5d'] = 0

        # 2. Main inflow trend (is it accelerating?)
        if len(df) >= cls.TREND_DAYS and all(col in df.columns for col in ['buy_lg_vol', 'sell_lg_vol']):
            first_half = df.iloc[cls.FLOW_DAYS:cls.TREND_DAYS].copy()
            second_half = df.head(cls.FLOW_DAYS).copy()

            first_flow = (
                (first_half['buy_lg_vol'] + first_half.get('buy_elg_vol', 0)) -
                (first_half['sell_lg_vol'] + first_half.get('sell_elg_vol', 0))
            ).sum()

            second_flow = (
                (second_half['buy_lg_vol'] + second_half.get('buy_elg_vol', 0)) -
                (second_half['sell_lg_vol'] + second_half.get('sell_elg_vol', 0))
            ).sum()

            # Trend score: positive = accelerating inflow, negative = decelerating
            if first_flow != 0:
                trend_ratio = (second_flow - first_flow) / abs(first_flow)
                # Normalize to 0-100 scale (50 = neutral)
                result['main_inflow_trend'] = round(50 + (trend_ratio * 25), 2)
            else:
                result['main_inflow_trend'] = 50 if second_flow >= 0 else 40

        # 3. Retail outflow ratio (institutions accumulating while retail sells)
        if all(col in df.columns for col in ['buy_sm_vol', 'sell_sm_vol']):
            retail_buy = recent['buy_sm_vol'].sum()
            retail_sell = recent['sell_sm_vol'].sum()

            if retail_buy + retail_sell > 0:
                # Ratio > 0.5 means more retail selling (good for accumulation thesis)
                outflow_ratio = retail_sell / (retail_buy + retail_sell)
                result['retail_outflow_ratio'] = round(outflow_ratio, 4)

        return result

    @classmethod
    def _compute_north_inflow(cls, df: pd.DataFrame) -> float:
        """
        Compute 5-day northbound capital inflow score.

        Northbound flow is market-wide, so this is an environmental factor
        rather than stock-specific.
        """
        if len(df) < cls.FLOW_DAYS:
            return 50.0  # Neutral

        recent = df.head(cls.FLOW_DAYS).copy()

        # Use north_money (northbound net flow, in million CNY)
        if 'north_money' in recent.columns:
            total_flow = recent['north_money'].sum()

            # Normalize: typical daily flow is +/- 5 billion
            # 5 billion over 5 days = 25 billion, 10 billion = very bullish
            # Convert to 0-100 scale
            normalized = 50 + (total_flow / 1000)  # total_flow in millions
            return round(max(0, min(100, normalized)), 2)

        # Alternative: use hgt + sgt (Shanghai + Shenzhen connect)
        if 'hgt' in recent.columns and 'sgt' in recent.columns:
            total_flow = (recent['hgt'] + recent['sgt']).sum()
            normalized = 50 + (total_flow / 1000)
            return round(max(0, min(100, normalized)), 2)

        return 50.0


def compute_sentiment_score(factors: Dict) -> float:
    """
    Compute overall sentiment score from flow factors.

    Sentiment score components:
    - 5-day main inflow: 40%
    - Inflow trend (momentum): 30%
    - North inflow: 20%
    - Retail outflow (contrarian): 10%

    Returns:
        Score 0-100 (higher = more bullish sentiment)
    """
    score = 0
    weight_sum = 0

    # Main inflow score (40%)
    main_inflow = factors.get('main_inflow_5d')
    if main_inflow is not None:
        # Normalize: -0.5 to 0.5 typical range
        # -0.5 = 0, 0 = 50, 0.5 = 100
        inflow_score = 50 + (main_inflow * 100)
        inflow_score = max(0, min(100, inflow_score))
        score += inflow_score * 0.40
        weight_sum += 0.40

    # Inflow trend score (30%)
    trend = factors.get('main_inflow_trend')
    if trend is not None:
        score += trend * 0.30
        weight_sum += 0.30

    # North inflow score (20%)
    north = factors.get('north_inflow_5d')
    if north is not None:
        score += north * 0.20
        weight_sum += 0.20

    # Retail outflow score (10%) - contrarian indicator
    retail_outflow = factors.get('retail_outflow_ratio')
    if retail_outflow is not None:
        # Higher retail selling = potentially bullish (accumulation)
        outflow_score = retail_outflow * 100
        score += outflow_score * 0.10
        weight_sum += 0.10

    if weight_sum > 0:
        return round(score / weight_sum * 100, 2)

    return 50.0  # Default neutral


def is_accumulation_signal(factors: Dict) -> bool:
    """
    Check if factors indicate institutional accumulation.

    Accumulation signals:
    1. Positive 5-day main inflow
    2. Accelerating inflow (trend > 50)
    3. Retail outflow ratio > 0.5

    Returns:
        True if accumulation pattern detected
    """
    main_inflow = factors.get('main_inflow_5d', 0)
    trend = factors.get('main_inflow_trend', 50)
    retail_outflow = factors.get('retail_outflow_ratio', 0.5)

    accumulation_signals = 0

    if main_inflow and main_inflow > 0.1:
        accumulation_signals += 1

    if trend and trend > 55:
        accumulation_signals += 1

    if retail_outflow and retail_outflow > 0.55:
        accumulation_signals += 1

    return accumulation_signals >= 2
