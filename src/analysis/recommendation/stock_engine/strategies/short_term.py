"""
Short-Term Stock Strategy - Predict breakouts for 7-30 day holding period.

Weight allocation:
- Technical breakthrough signals: 40%
- Institutional accumulation: 25%
- Event catalyst: 20%
- Risk adjustment: 15%

Key principle: Predict breakouts, don't chase rallies
"""
from typing import Dict, Optional
from datetime import datetime, timedelta

from src.data_sources.tushare_client import (
    get_forecast,
    normalize_ts_code,
    format_date_yyyymmdd,
    tushare_call_with_retry,
)


class ShortTermStrategy:
    """
    Short-term stock recommendation strategy (7-30 day horizon).

    Focuses on identifying stocks about to break out based on:
    - Technical consolidation patterns
    - Institutional accumulation signals
    - Upcoming catalysts
    """

    # Weight configuration
    WEIGHTS = {
        'technical': 0.40,
        'accumulation': 0.25,
        'catalyst': 0.20,
        'risk': 0.15,
    }

    # Score thresholds
    MIN_SCORE_THRESHOLD = 60  # Minimum score to recommend
    HIGH_SCORE_THRESHOLD = 80  # High confidence threshold

    @classmethod
    def compute_score(cls, factors: Dict) -> float:
        """
        Compute overall short-term score from all factors.

        Args:
            factors: Dict containing all factor values

        Returns:
            Score 0-100 (higher = better short-term opportunity)
        """
        scores = {
            'technical': cls._compute_technical_score(factors),
            'accumulation': cls._compute_accumulation_score(factors),
            'catalyst': cls._compute_catalyst_score(factors),
            'risk': cls._compute_risk_score(factors),
        }

        # Weighted average
        total_score = sum(
            scores[key] * cls.WEIGHTS[key]
            for key in scores
            if scores[key] is not None
        )

        # Normalize by actual weights used
        weight_sum = sum(
            cls.WEIGHTS[key]
            for key in scores
            if scores[key] is not None
        )

        if weight_sum > 0:
            return round(total_score / weight_sum * 100, 2)

        return 50.0

    @classmethod
    def _compute_technical_score(cls, factors: Dict) -> float:
        """
        Compute technical breakthrough score (40% weight).

        Components:
        - Consolidation score: 40%
        - Volume precursor: 35%
        - MA convergence: 25%
        """
        score = 0
        weights = 0

        consolidation = factors.get('consolidation_score')
        if consolidation is not None:
            score += consolidation * 0.40
            weights += 0.40

        volume = factors.get('volume_precursor')
        if volume is not None:
            score += volume * 0.35
            weights += 0.35

        ma_conv = factors.get('ma_convergence')
        if ma_conv is not None:
            score += ma_conv * 0.25
            weights += 0.25

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def _compute_accumulation_score(cls, factors: Dict) -> float:
        """
        Compute institutional accumulation score (25% weight).

        Components:
        - 5-day main inflow: 45%
        - Inflow trend: 35%
        - Retail outflow: 20%
        """
        score = 0
        weights = 0

        main_inflow = factors.get('main_inflow_5d')
        if main_inflow is not None:
            # Normalize: -0.5 to 0.5 typical range
            inflow_score = 50 + (main_inflow * 100)
            inflow_score = max(0, min(100, inflow_score))
            score += inflow_score * 0.45
            weights += 0.45

        trend = factors.get('main_inflow_trend')
        if trend is not None:
            score += trend * 0.35
            weights += 0.35

        retail_outflow = factors.get('retail_outflow_ratio')
        if retail_outflow is not None:
            # Higher retail selling = potentially bullish
            outflow_score = retail_outflow * 100
            score += outflow_score * 0.20
            weights += 0.20

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def _compute_catalyst_score(cls, factors: Dict) -> float:
        """
        Compute event catalyst score (20% weight).

        This is a placeholder that should be enhanced with:
        - Days to next earnings report
        - Recent analyst upgrades
        - Sector rotation signals
        """
        # Default to neutral - actual implementation needs earnings calendar
        return 50.0

    @classmethod
    def _compute_risk_score(cls, factors: Dict) -> float:
        """
        Compute risk-adjusted score (15% weight).

        Components:
        - Volatility appropriateness: Not too low (no movement) or too high (risky)
        - Liquidity check: Sufficient volume
        - Valuation floor: Not extremely overvalued
        """
        score = 0
        weights = 0

        # RSI check - avoid extremes
        rsi = factors.get('rsi')
        if rsi is not None:
            # Best range: 35-65 (neither overbought nor oversold extreme)
            if 35 <= rsi <= 65:
                rsi_score = 80
            elif 25 <= rsi < 35 or 65 < rsi <= 75:
                rsi_score = 60
            elif rsi < 25:
                rsi_score = 70  # Oversold could be opportunity
            else:
                rsi_score = 30  # Overbought is risky
            score += rsi_score * 0.40
            weights += 0.40

        # Bollinger position - middle zone preferred
        boll = factors.get('bollinger_position')
        if boll is not None:
            # Best: 30-70 (room to move either way)
            if 30 <= boll <= 70:
                boll_score = 80
            elif 20 <= boll < 30:
                boll_score = 70  # Near lower band, potential bounce
            elif 70 < boll <= 80:
                boll_score = 50  # Near upper band, caution
            else:
                boll_score = 40
            score += boll_score * 0.30
            weights += 0.30

        # Debt ratio check
        debt = factors.get('debt_ratio')
        if debt is not None:
            if debt < 50:
                debt_score = 80
            elif debt < 70:
                debt_score = 60
            else:
                debt_score = 40
            score += debt_score * 0.30
            weights += 0.30

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def is_recommended(cls, score: float) -> bool:
        """Check if score meets recommendation threshold."""
        return score >= cls.MIN_SCORE_THRESHOLD

    @classmethod
    def get_confidence_level(cls, score: float) -> str:
        """Get confidence level based on score."""
        if score >= cls.HIGH_SCORE_THRESHOLD:
            return 'high'
        elif score >= cls.MIN_SCORE_THRESHOLD:
            return 'medium'
        else:
            return 'low'


def get_short_term_recommendation(
    factors: Dict,
    include_reasoning: bool = True
) -> Dict:
    """
    Generate short-term recommendation from factors.

    Args:
        factors: All computed factors for a stock
        include_reasoning: Whether to include factor breakdown

    Returns:
        Recommendation dict with score, confidence, and optional reasoning
    """
    score = ShortTermStrategy.compute_score(factors)

    result = {
        'score': score,
        'is_recommended': ShortTermStrategy.is_recommended(score),
        'confidence': ShortTermStrategy.get_confidence_level(score),
        'holding_period': '7-30 days',
        'strategy_type': 'short_term',
    }

    if include_reasoning:
        result['factor_scores'] = {
            'technical': ShortTermStrategy._compute_technical_score(factors),
            'accumulation': ShortTermStrategy._compute_accumulation_score(factors),
            'catalyst': ShortTermStrategy._compute_catalyst_score(factors),
            'risk': ShortTermStrategy._compute_risk_score(factors),
        }
        result['key_factors'] = _identify_key_factors(factors)

    return result


def _identify_key_factors(factors: Dict) -> list:
    """Identify the most influential factors for explanation."""
    key_factors = []

    # Technical signals
    consolidation = factors.get('consolidation_score', 0)
    if consolidation > 70:
        key_factors.append(f"盘整形态良好 (得分: {consolidation:.0f})")
    elif consolidation > 60:
        key_factors.append(f"处于盘整区间 (得分: {consolidation:.0f})")

    volume = factors.get('volume_precursor', 0)
    if volume > 70:
        key_factors.append(f"量能异动，疑似吸筹 (得分: {volume:.0f})")

    ma_conv = factors.get('ma_convergence', 0)
    if ma_conv > 70:
        key_factors.append(f"均线收敛，方向选择临近 (得分: {ma_conv:.0f})")

    # Money flow signals
    main_inflow = factors.get('main_inflow_5d', 0)
    if main_inflow and main_inflow > 0.2:
        key_factors.append(f"主力资金5日净流入")
    elif main_inflow and main_inflow < -0.2:
        key_factors.append(f"主力资金5日净流出 (风险)")

    # Valuation
    peg = factors.get('peg_ratio')
    if peg and peg < 1:
        key_factors.append(f"PEG估值较低 ({peg:.2f})")
    elif peg and peg > 2:
        key_factors.append(f"PEG估值偏高 ({peg:.2f}, 风险)")

    return key_factors[:5]  # Limit to top 5 factors
