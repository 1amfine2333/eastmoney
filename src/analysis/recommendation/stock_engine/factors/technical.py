"""
Stock Technical Factors - Technical analysis indicators for stock recommendation.

Key factors:
- Consolidation score: Detect stocks in consolidation pattern near breakout
- Volume precursor: Volume increase without price movement (accumulation signal)
- MA convergence: Moving average convergence score (trend about to change)

Design principle: Predict breakouts, don't chase rallies
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta

from src.data_sources.tushare_client import (
    get_stock_daily,
    get_stock_factors as get_tushare_factors,
    normalize_ts_code,
    format_date_yyyymmdd,
)


class TechnicalFactors:
    """
    Technical factor computation for stocks.

    Uses TuShare stk_factor API (5000+ points) for MACD/KDJ/RSI/BOLL
    plus custom calculations for consolidation and accumulation detection.
    """

    # Lookback periods
    CONSOLIDATION_DAYS = 20
    VOLUME_DAYS = 10
    MA_DAYS = 60

    @classmethod
    def compute(cls, ts_code: str, trade_date: str) -> Dict:
        """
        Compute all technical factors for a stock.

        Args:
            ts_code: Stock code in TuShare format
            trade_date: Trade date in YYYYMMDD format

        Returns:
            Dict with technical factors
        """
        ts_code = normalize_ts_code(ts_code)

        # Initialize with defaults
        factors = {
            'consolidation_score': None,
            'volume_precursor': None,
            'ma_convergence': None,
            'rsi': None,
            'macd_signal': None,
            'bollinger_position': None,
        }

        try:
            # Get price data for custom calculations
            end_date = trade_date
            start_date = format_date_yyyymmdd(
                datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=cls.MA_DAYS + 30)
            )

            price_df = get_stock_daily(ts_code, start_date, end_date)

            if price_df is not None and len(price_df) >= 20:
                # Compute custom factors
                factors['consolidation_score'] = cls._compute_consolidation_score(price_df)
                factors['volume_precursor'] = cls._compute_volume_precursor(price_df)
                factors['ma_convergence'] = cls._compute_ma_convergence(price_df)

            # Get TuShare technical indicators
            tushare_factors = get_tushare_factors(ts_code, days=5)

            if tushare_factors is not None and not tushare_factors.empty:
                latest = tushare_factors.iloc[0]

                # RSI (oversold < 30 = opportunity, overbought > 70 = avoid)
                if 'rsi_6' in latest and pd.notna(latest['rsi_6']):
                    factors['rsi'] = float(latest['rsi_6'])

                # MACD signal (positive = bullish momentum)
                if 'macd' in latest and pd.notna(latest['macd']):
                    macd = float(latest['macd'])
                    # Normalize to 0-100 scale
                    factors['macd_signal'] = cls._normalize_macd(macd)

                # Bollinger position (how close to bands)
                if all(col in latest for col in ['close', 'boll_upper', 'boll_lower', 'boll_mid']):
                    if all(pd.notna(latest[col]) for col in ['close', 'boll_upper', 'boll_lower']):
                        factors['bollinger_position'] = cls._compute_bollinger_position(
                            float(latest['close']),
                            float(latest['boll_upper']),
                            float(latest['boll_lower']),
                            float(latest['boll_mid'])
                        )

        except Exception as e:
            print(f"Error computing technical factors for {ts_code}: {e}")

        return factors

    @classmethod
    def _compute_consolidation_score(cls, df: pd.DataFrame) -> float:
        """
        Compute consolidation score (0-100).

        High score = stock is in tight consolidation, potential for breakout.
        Looks for:
        - Low price volatility over recent period
        - Narrowing range (decreasing high-low spread)
        - Price near resistance level

        Returns:
            Score 0-100 (higher = tighter consolidation, more likely to break out)
        """
        if len(df) < cls.CONSOLIDATION_DAYS:
            return 50.0

        recent = df.tail(cls.CONSOLIDATION_DAYS).copy()

        # 1. Price volatility (lower = more consolidated)
        returns = recent['close'].pct_change().dropna()
        volatility = returns.std()

        # Normalize: 0.01 (1% daily) = 50, lower = higher score
        vol_score = max(0, min(100, 100 - (volatility * 2000)))

        # 2. Range contraction (decreasing high-low spread)
        recent['range'] = (recent['high'] - recent['low']) / recent['close']
        first_half_range = recent['range'].iloc[:len(recent)//2].mean()
        second_half_range = recent['range'].iloc[len(recent)//2:].mean()

        if first_half_range > 0:
            contraction_ratio = second_half_range / first_half_range
            # Ratio < 1 means range is contracting (good)
            contraction_score = max(0, min(100, (2 - contraction_ratio) * 50))
        else:
            contraction_score = 50.0

        # 3. Price near recent high (within 5% of 20-day high = breakout ready)
        current_price = float(recent['close'].iloc[-1])
        recent_high = float(recent['high'].max())
        distance_from_high = (recent_high - current_price) / recent_high

        # Closer to high = higher score
        high_proximity_score = max(0, min(100, 100 - (distance_from_high * 500)))

        # Weighted combination
        score = (
            vol_score * 0.4 +
            contraction_score * 0.35 +
            high_proximity_score * 0.25
        )

        return round(score, 2)

    @classmethod
    def _compute_volume_precursor(cls, df: pd.DataFrame) -> float:
        """
        Compute volume precursor score (0-100).

        High score = volume increasing but price not moving much yet.
        This suggests accumulation (institutional buying).

        Returns:
            Score 0-100 (higher = stronger accumulation signal)
        """
        if len(df) < cls.VOLUME_DAYS * 2:
            return 50.0

        recent = df.tail(cls.VOLUME_DAYS).copy()
        baseline = df.iloc[-(cls.VOLUME_DAYS * 2):-cls.VOLUME_DAYS].copy()

        # 1. Volume increase ratio
        recent_avg_vol = recent['vol'].mean() if 'vol' in recent else recent['volume'].mean()
        baseline_avg_vol = baseline['vol'].mean() if 'vol' in baseline else baseline['volume'].mean()

        if baseline_avg_vol > 0:
            vol_increase = recent_avg_vol / baseline_avg_vol
        else:
            vol_increase = 1.0

        # 2. Price stability (low movement despite volume)
        price_change_pct = abs(
            (float(recent['close'].iloc[-1]) - float(recent['close'].iloc[0]))
            / float(recent['close'].iloc[0])
        ) * 100

        # Ideal: volume up significantly (>1.5x) but price stable (<3% change)
        # Volume score: 1.5x = 50, 2x = 75, 2.5x = 100
        vol_score = min(100, max(0, (vol_increase - 1) * 100))

        # Price stability score: <1% = 100, 3% = 50, >5% = 0
        stability_score = max(0, min(100, 100 - (price_change_pct * 20)))

        # Combined: want high volume increase with low price movement
        # Only give high score if both conditions met
        if vol_increase >= 1.3 and price_change_pct < 5:
            score = (vol_score * 0.6 + stability_score * 0.4)
        else:
            score = (vol_score * 0.4 + stability_score * 0.6) * 0.7  # Penalize if conditions not met

        return round(score, 2)

    @classmethod
    def _compute_ma_convergence(cls, df: pd.DataFrame) -> float:
        """
        Compute moving average convergence score (0-100).

        High score = MAs are converging (5, 10, 20, 60 day MAs getting closer).
        This often precedes a trend change or breakout.

        Returns:
            Score 0-100 (higher = MAs more converged, trend change likely)
        """
        if len(df) < 60:
            return 50.0

        # Calculate moving averages
        df = df.copy()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # Get latest values
        latest = df.iloc[-1]

        ma_values = []
        for ma in ['ma5', 'ma10', 'ma20', 'ma60']:
            if ma in latest and pd.notna(latest[ma]):
                ma_values.append(float(latest[ma]))

        if len(ma_values) < 3:
            return 50.0

        # Calculate spread between MAs
        ma_spread = (max(ma_values) - min(ma_values)) / np.mean(ma_values) * 100

        # Compare to historical spread (20 days ago)
        historical = df.iloc[-20]
        hist_ma_values = []
        for ma in ['ma5', 'ma10', 'ma20']:
            if ma in historical and pd.notna(historical[ma]):
                hist_ma_values.append(float(historical[ma]))

        if len(hist_ma_values) >= 2:
            hist_spread = (max(hist_ma_values) - min(hist_ma_values)) / np.mean(hist_ma_values) * 100

            # Convergence = spread is decreasing
            if hist_spread > 0:
                convergence_ratio = ma_spread / hist_spread
            else:
                convergence_ratio = 1.0
        else:
            convergence_ratio = 1.0

        # Score based on:
        # 1. Current spread (lower = more converged)
        # 2. Convergence ratio (< 1 means converging)

        # Spread score: 1% spread = 80, 5% = 50, 10%+ = 20
        spread_score = max(20, min(100, 100 - (ma_spread * 8)))

        # Convergence score: ratio < 0.7 = 100, 1.0 = 50, > 1.3 = 0
        if convergence_ratio < 1:
            convergence_score = min(100, 50 + (1 - convergence_ratio) * 100)
        else:
            convergence_score = max(0, 50 - (convergence_ratio - 1) * 100)

        score = spread_score * 0.4 + convergence_score * 0.6

        return round(score, 2)

    @classmethod
    def _normalize_macd(cls, macd: float) -> float:
        """
        Normalize MACD value to 0-100 scale.

        Positive MACD = bullish momentum (score > 50)
        Negative MACD = bearish momentum (score < 50)
        """
        # MACD typically ranges from -2 to 2 for most stocks
        # Normalize to 0-100
        normalized = 50 + (macd * 20)
        return max(0, min(100, round(normalized, 2)))

    @classmethod
    def _compute_bollinger_position(
        cls,
        close: float,
        upper: float,
        lower: float,
        mid: float
    ) -> float:
        """
        Compute Bollinger Band position score (0-100).

        Score interpretation:
        - < 20: Near lower band (oversold, potential bounce)
        - 40-60: Middle zone (neutral)
        - > 80: Near upper band (overbought, caution)

        For our strategy (predict breakouts), we want stocks near
        the middle that are about to break out, not already at extremes.
        """
        if upper <= lower:
            return 50.0

        # Position: 0 = at lower, 100 = at upper
        position = (close - lower) / (upper - lower) * 100

        return max(0, min(100, round(position, 2)))


# Utility functions for external use

def get_consolidation_score(stock_code: str, trade_date: str = None) -> Optional[float]:
    """Get consolidation score for a stock."""
    if not trade_date:
        trade_date = format_date_yyyymmdd()
    factors = TechnicalFactors.compute(stock_code, trade_date)
    return factors.get('consolidation_score')


def get_volume_precursor(stock_code: str, trade_date: str = None) -> Optional[float]:
    """Get volume precursor score for a stock."""
    if not trade_date:
        trade_date = format_date_yyyymmdd()
    factors = TechnicalFactors.compute(stock_code, trade_date)
    return factors.get('volume_precursor')


def get_ma_convergence(stock_code: str, trade_date: str = None) -> Optional[float]:
    """Get MA convergence score for a stock."""
    if not trade_date:
        trade_date = format_date_yyyymmdd()
    factors = TechnicalFactors.compute(stock_code, trade_date)
    return factors.get('ma_convergence')
