"""
Stock Fundamental Factors - Quality and valuation factors for stock recommendation.

Key factors:
- ROE quality: Consistency and level of return on equity
- Gross margin stability: Consistent profitability
- Cash flow quality: OCF/profit ratio (earnings quality)
- Debt health: Asset-liability ratio
- Growth metrics: Revenue and profit CAGR
- Valuation: PEG, PE/PB percentile vs history

Design principle: Quality > Market cap, Set ROE<10% as threshold
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from src.data_sources.tushare_client import (
    get_financial_indicators,
    get_income_statement,
    get_balance_sheet,
    get_cashflow_statement,
    normalize_ts_code,
    format_date_yyyymmdd,
    tushare_call_with_retry,
)


class FundamentalFactors:
    """
    Fundamental factor computation for stocks.

    Uses TuShare fina_indicator (2000+ points) and income/balance/cashflow
    APIs (5000+ points) for comprehensive financial analysis.
    """

    # Thresholds
    ROE_MIN_THRESHOLD = 10.0  # ROE < 10% = quality concern
    DEBT_MAX_THRESHOLD = 70.0  # Debt ratio > 70% = high leverage risk
    OCF_MIN_RATIO = 0.8  # OCF/Net profit > 0.8 = good earnings quality

    @classmethod
    def compute(cls, ts_code: str, trade_date: str) -> Dict:
        """
        Compute all fundamental factors for a stock.

        Args:
            ts_code: Stock code in TuShare format
            trade_date: Trade date in YYYYMMDD format

        Returns:
            Dict with fundamental factors
        """
        ts_code = normalize_ts_code(ts_code)

        factors = {
            'roe': None,
            'roe_yoy': None,
            'gross_margin': None,
            'gross_margin_stability': None,
            'ocf_to_profit': None,
            'debt_ratio': None,
            'revenue_growth_yoy': None,
            'profit_growth_yoy': None,
            'revenue_cagr_3y': None,
            'profit_cagr_3y': None,
            'peg_ratio': None,
            'pe_percentile': None,
            'pb_percentile': None,
        }

        try:
            # Get financial indicators (ROE, margins, debt)
            fina_df = get_financial_indicators(ts_code, periods=12)

            if fina_df is not None and not fina_df.empty:
                factors.update(cls._compute_quality_factors(fina_df))

            # Get income statement for growth calculation
            income_df = get_income_statement(ts_code, periods=8)

            if income_df is not None and not income_df.empty:
                factors.update(cls._compute_growth_factors(income_df))

            # Get cash flow for OCF/profit
            cashflow_df = get_cashflow_statement(ts_code, periods=4)

            if cashflow_df is not None and not cashflow_df.empty and income_df is not None:
                factors['ocf_to_profit'] = cls._compute_ocf_ratio(cashflow_df, income_df)

            # Get valuation metrics
            factors.update(cls._compute_valuation_factors(ts_code, trade_date))

        except Exception as e:
            print(f"Error computing fundamental factors for {ts_code}: {e}")

        return factors

    @classmethod
    def _compute_quality_factors(cls, df: pd.DataFrame) -> Dict:
        """
        Compute quality factors from financial indicators.

        Args:
            df: DataFrame from fina_indicator API

        Returns:
            Dict with ROE, margin, and debt factors
        """
        result = {}

        # Sort by end_date descending to ensure latest first
        df = df.sort_values('end_date', ascending=False)

        # Latest ROE
        if 'roe' in df.columns:
            latest_roe = df['roe'].iloc[0]
            if pd.notna(latest_roe):
                result['roe'] = float(latest_roe)

            # ROE YoY change (compare to same quarter last year)
            if len(df) >= 5:  # Need at least 5 quarters for YoY
                current_roe = df['roe'].iloc[0]
                prev_year_roe = df['roe'].iloc[4] if len(df) > 4 else None

                if pd.notna(current_roe) and pd.notna(prev_year_roe) and prev_year_roe != 0:
                    result['roe_yoy'] = float(
                        (current_roe - prev_year_roe) / abs(prev_year_roe) * 100
                    )

        # Gross margin
        if 'grossprofit_margin' in df.columns:
            latest_margin = df['grossprofit_margin'].iloc[0]
            if pd.notna(latest_margin):
                result['gross_margin'] = float(latest_margin)

            # Gross margin stability (std dev over recent quarters)
            recent_margins = df['grossprofit_margin'].head(8).dropna()
            if len(recent_margins) >= 4:
                stability = recent_margins.std()
                # Lower std = more stable, convert to score
                # std < 2 = 100, std > 10 = 0
                result['gross_margin_stability'] = max(0, min(100, 100 - (stability * 10)))

        # Debt ratio
        if 'debt_to_assets' in df.columns:
            latest_debt = df['debt_to_assets'].iloc[0]
            if pd.notna(latest_debt):
                result['debt_ratio'] = float(latest_debt)

        return result

    @classmethod
    def _compute_growth_factors(cls, df: pd.DataFrame) -> Dict:
        """
        Compute growth factors from income statement.

        Args:
            df: DataFrame from income API

        Returns:
            Dict with revenue and profit growth factors
        """
        result = {}

        df = df.sort_values('end_date', ascending=False)

        # YoY growth (compare latest quarter to same quarter last year)
        if len(df) >= 5:
            # Revenue growth
            if 'revenue' in df.columns:
                current_rev = df['revenue'].iloc[0]
                prev_year_rev = df['revenue'].iloc[4] if pd.notna(df['revenue'].iloc[4]) else None

                if pd.notna(current_rev) and prev_year_rev and prev_year_rev > 0:
                    result['revenue_growth_yoy'] = float(
                        (current_rev - prev_year_rev) / prev_year_rev * 100
                    )

            # Profit growth
            if 'n_income' in df.columns:
                current_profit = df['n_income'].iloc[0]
                prev_year_profit = df['n_income'].iloc[4] if pd.notna(df['n_income'].iloc[4]) else None

                if pd.notna(current_profit) and prev_year_profit and prev_year_profit > 0:
                    result['profit_growth_yoy'] = float(
                        (current_profit - prev_year_profit) / prev_year_profit * 100
                    )

        # 3-year CAGR (need at least 3 years = 12 quarters of data)
        # We approximate by using Q4 figures (annual data)
        annual_data = df[df['end_date'].str.endswith('1231')].head(4)

        if len(annual_data) >= 4:
            # Revenue CAGR
            if 'revenue' in annual_data.columns:
                latest_rev = annual_data['revenue'].iloc[0]
                oldest_rev = annual_data['revenue'].iloc[3]

                if pd.notna(latest_rev) and pd.notna(oldest_rev) and oldest_rev > 0:
                    cagr = ((latest_rev / oldest_rev) ** (1/3) - 1) * 100
                    result['revenue_cagr_3y'] = float(cagr)

            # Profit CAGR
            if 'n_income' in annual_data.columns:
                latest_profit = annual_data['n_income'].iloc[0]
                oldest_profit = annual_data['n_income'].iloc[3]

                if (pd.notna(latest_profit) and pd.notna(oldest_profit) and
                    oldest_profit > 0 and latest_profit > 0):
                    cagr = ((latest_profit / oldest_profit) ** (1/3) - 1) * 100
                    result['profit_cagr_3y'] = float(cagr)

        return result

    @classmethod
    def _compute_ocf_ratio(cls, cashflow_df: pd.DataFrame, income_df: pd.DataFrame) -> Optional[float]:
        """
        Compute OCF to net profit ratio.

        High ratio (>0.8) indicates good earnings quality.
        Low ratio suggests earnings manipulation or accounting issues.
        """
        try:
            cashflow_df = cashflow_df.sort_values('end_date', ascending=False)
            income_df = income_df.sort_values('end_date', ascending=False)

            # Get operating cash flow
            if 'n_cashflow_act' in cashflow_df.columns:
                ocf = cashflow_df['n_cashflow_act'].iloc[0]
            else:
                return None

            # Get net income
            if 'n_income' in income_df.columns:
                net_income = income_df['n_income'].iloc[0]
            else:
                return None

            if pd.notna(ocf) and pd.notna(net_income) and net_income != 0:
                return float(ocf / net_income)

        except Exception:
            pass

        return None

    @classmethod
    def _compute_valuation_factors(cls, ts_code: str, trade_date: str) -> Dict:
        """
        Compute valuation factors (PEG, PE/PB percentile).

        Args:
            ts_code: Stock code
            trade_date: Trade date

        Returns:
            Dict with valuation factors
        """
        result = {
            'peg_ratio': None,
            'pe_percentile': None,
            'pb_percentile': None,
        }

        try:
            # Get daily basic data (PE, PB)
            end_date = trade_date
            start_date = format_date_yyyymmdd(
                datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=365 * 3)
            )

            df = tushare_call_with_retry(
                'daily_basic',
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,pe_ttm,pb,total_mv'
            )

            if df is None or df.empty:
                return result

            df = df.sort_values('trade_date', ascending=False)

            # Current PE and PB
            current_pe = df['pe_ttm'].iloc[0] if pd.notna(df['pe_ttm'].iloc[0]) else None
            current_pb = df['pb'].iloc[0] if pd.notna(df['pb'].iloc[0]) else None

            # PE percentile (where current PE sits in 3-year history)
            if current_pe is not None:
                pe_values = df['pe_ttm'].dropna()
                if len(pe_values) > 20:
                    percentile = (pe_values > current_pe).sum() / len(pe_values) * 100
                    result['pe_percentile'] = round(100 - percentile, 2)  # Lower percentile = cheaper

            # PB percentile
            if current_pb is not None:
                pb_values = df['pb'].dropna()
                if len(pb_values) > 20:
                    percentile = (pb_values > current_pb).sum() / len(pb_values) * 100
                    result['pb_percentile'] = round(100 - percentile, 2)

            # PEG ratio (PE / growth rate)
            # Get growth rate from financial indicators
            fina_df = get_financial_indicators(ts_code, periods=4)
            if fina_df is not None and not fina_df.empty and current_pe:
                fina_df = fina_df.sort_values('end_date', ascending=False)

                # Use net profit YoY growth
                if 'netprofit_yoy' in fina_df.columns or 'op_yoy' in fina_df.columns:
                    growth_col = 'netprofit_yoy' if 'netprofit_yoy' in fina_df.columns else 'op_yoy'
                    growth = fina_df[growth_col].iloc[0]

                    if pd.notna(growth) and growth > 0 and current_pe > 0:
                        result['peg_ratio'] = round(current_pe / growth, 2)

        except Exception as e:
            print(f"Error computing valuation factors for {ts_code}: {e}")

        return result


def compute_quality_score(factors: Dict) -> float:
    """
    Compute overall quality score from fundamental factors.

    Quality score components:
    - ROE level and consistency: 35%
    - Margin stability: 20%
    - Cash flow quality: 25%
    - Debt health: 20%

    Returns:
        Score 0-100 (higher = better quality)
    """
    score = 0
    weight_sum = 0

    # ROE score (35%)
    roe = factors.get('roe')
    if roe is not None:
        if roe < FundamentalFactors.ROE_MIN_THRESHOLD:
            roe_score = roe / FundamentalFactors.ROE_MIN_THRESHOLD * 30  # Cap at 30 if below threshold
        else:
            # 10% = 50, 15% = 70, 20%+ = 90
            roe_score = min(100, 30 + (roe - 10) * 4)
        score += roe_score * 0.35
        weight_sum += 0.35

    # Margin stability score (20%)
    margin_stability = factors.get('gross_margin_stability')
    if margin_stability is not None:
        score += margin_stability * 0.20
        weight_sum += 0.20

    # Cash flow quality (25%)
    ocf_ratio = factors.get('ocf_to_profit')
    if ocf_ratio is not None:
        if ocf_ratio >= FundamentalFactors.OCF_MIN_RATIO:
            ocf_score = min(100, ocf_ratio * 80)
        else:
            ocf_score = ocf_ratio / FundamentalFactors.OCF_MIN_RATIO * 50
        score += ocf_score * 0.25
        weight_sum += 0.25

    # Debt health (20%) - lower is better
    debt_ratio = factors.get('debt_ratio')
    if debt_ratio is not None:
        if debt_ratio > FundamentalFactors.DEBT_MAX_THRESHOLD:
            debt_score = max(0, 100 - (debt_ratio - 70) * 2)
        else:
            debt_score = min(100, 100 - debt_ratio)
        score += debt_score * 0.20
        weight_sum += 0.20

    # Normalize by actual weight used
    if weight_sum > 0:
        return round(score / weight_sum * 100, 2)

    return 50.0  # Default neutral score


def compute_growth_score(factors: Dict) -> float:
    """
    Compute overall growth score from fundamental factors.

    Growth score components:
    - Revenue CAGR: 40%
    - Profit CAGR: 40%
    - Growth acceleration (YoY vs CAGR): 20%

    Returns:
        Score 0-100 (higher = better growth)
    """
    score = 0
    weight_sum = 0

    # Revenue CAGR (40%)
    rev_cagr = factors.get('revenue_cagr_3y')
    if rev_cagr is not None:
        # 0% = 30, 10% = 50, 20% = 70, 30%+ = 90
        cagr_score = min(100, max(0, 30 + rev_cagr * 2))
        score += cagr_score * 0.40
        weight_sum += 0.40

    # Profit CAGR (40%)
    profit_cagr = factors.get('profit_cagr_3y')
    if profit_cagr is not None:
        cagr_score = min(100, max(0, 30 + profit_cagr * 2))
        score += cagr_score * 0.40
        weight_sum += 0.40

    # Growth acceleration (20%)
    rev_yoy = factors.get('revenue_growth_yoy')
    if rev_yoy is not None and rev_cagr is not None:
        # Accelerating if YoY > CAGR
        acceleration = rev_yoy - (rev_cagr or 0)
        accel_score = min(100, max(0, 50 + acceleration * 2))
        score += accel_score * 0.20
        weight_sum += 0.20

    if weight_sum > 0:
        return round(score / weight_sum * 100, 2)

    return 50.0


def compute_valuation_score(factors: Dict) -> float:
    """
    Compute overall valuation score.

    Components:
    - PEG ratio: 50% (ideal: 0.5-1.5)
    - PE percentile: 25% (lower = cheaper)
    - PB percentile: 25% (lower = cheaper)

    Returns:
        Score 0-100 (higher = more attractive valuation)
    """
    score = 0
    weight_sum = 0

    # PEG score (50%)
    peg = factors.get('peg_ratio')
    if peg is not None:
        if peg < 0:
            peg_score = 20  # Negative growth, low score
        elif peg < 0.5:
            peg_score = 100  # Very undervalued
        elif peg <= 1:
            peg_score = 90 - (peg - 0.5) * 20  # 0.5-1: 90-80
        elif peg <= 1.5:
            peg_score = 80 - (peg - 1) * 40  # 1-1.5: 80-60
        elif peg <= 2:
            peg_score = 60 - (peg - 1.5) * 40  # 1.5-2: 60-40
        else:
            peg_score = max(0, 40 - (peg - 2) * 20)  # >2: declining
        score += peg_score * 0.50
        weight_sum += 0.50

    # PE percentile (25%) - lower = cheaper = higher score
    pe_pct = factors.get('pe_percentile')
    if pe_pct is not None:
        pe_score = 100 - pe_pct  # Invert: low percentile = high score
        score += pe_score * 0.25
        weight_sum += 0.25

    # PB percentile (25%)
    pb_pct = factors.get('pb_percentile')
    if pb_pct is not None:
        pb_score = 100 - pb_pct
        score += pb_score * 0.25
        weight_sum += 0.25

    if weight_sum > 0:
        return round(score / weight_sum * 100, 2)

    return 50.0
