"""
Long-Term Stock Strategy - Quality investing for 3+ month holding period.

Weight allocation:
- Quality factors: 35% (ROE, margin stability, cash flow quality)
- Growth factors: 30% (Revenue/profit CAGR, acceleration)
- Valuation factors: 25% (PEG, PE/PB percentile)
- Moat factors: 10% (market position, competitive advantage)

Key principle: Quality > Market cap, Set ROE<10% as threshold
"""
from typing import Dict, Optional, List


class LongTermStrategy:
    """
    Long-term stock recommendation strategy (3+ month horizon).

    Focuses on quality companies with sustainable competitive advantages:
    - Strong and consistent returns on equity
    - Stable margins and earnings quality
    - Reasonable growth with attractive valuation
    """

    # Weight configuration
    WEIGHTS = {
        'quality': 0.35,
        'growth': 0.30,
        'valuation': 0.25,
        'moat': 0.10,
    }

    # Quality thresholds
    ROE_MIN = 10.0  # Minimum ROE threshold (below = disqualify)
    ROE_EXCELLENT = 15.0
    MARGIN_STABILITY_MIN = 60  # Margin stability score threshold
    OCF_RATIO_MIN = 0.7  # OCF/profit minimum for quality

    # Score thresholds
    MIN_SCORE_THRESHOLD = 60
    HIGH_SCORE_THRESHOLD = 75

    @classmethod
    def compute_score(cls, factors: Dict) -> float:
        """
        Compute overall long-term score from all factors.

        Args:
            factors: Dict containing all factor values

        Returns:
            Score 0-100 (higher = better long-term opportunity)
        """
        # Quality gate: Disqualify if ROE < threshold
        roe = factors.get('roe')
        if roe is not None and roe < cls.ROE_MIN:
            return 30.0  # Hard cap for low-ROE companies

        scores = {
            'quality': cls._compute_quality_score(factors),
            'growth': cls._compute_growth_score(factors),
            'valuation': cls._compute_valuation_score(factors),
            'moat': cls._compute_moat_score(factors),
        }

        # Weighted average
        total_score = sum(
            scores[key] * cls.WEIGHTS[key]
            for key in scores
            if scores[key] is not None
        )

        weight_sum = sum(
            cls.WEIGHTS[key]
            for key in scores
            if scores[key] is not None
        )

        if weight_sum > 0:
            return round(total_score / weight_sum * 100, 2)

        return 50.0

    @classmethod
    def _compute_quality_score(cls, factors: Dict) -> float:
        """
        Compute quality score (35% weight).

        Components:
        - ROE level and trend: 40%
        - Gross margin stability: 25%
        - Cash flow quality (OCF/profit): 25%
        - Debt ratio: 10%
        """
        score = 0
        weights = 0

        # ROE score
        roe = factors.get('roe')
        if roe is not None:
            if roe >= 20:
                roe_score = 95
            elif roe >= 15:
                roe_score = 80 + (roe - 15) * 3
            elif roe >= 10:
                roe_score = 50 + (roe - 10) * 6
            else:
                roe_score = roe * 5  # Linear below threshold

            # Bonus for improving ROE
            roe_yoy = factors.get('roe_yoy')
            if roe_yoy is not None and roe_yoy > 0:
                roe_score = min(100, roe_score + roe_yoy * 0.5)

            score += roe_score * 0.40
            weights += 0.40

        # Margin stability
        margin_stability = factors.get('gross_margin_stability')
        if margin_stability is not None:
            score += margin_stability * 0.25
            weights += 0.25

        # Cash flow quality
        ocf_ratio = factors.get('ocf_to_profit')
        if ocf_ratio is not None:
            if ocf_ratio >= 1.0:
                ocf_score = 90
            elif ocf_ratio >= 0.8:
                ocf_score = 70 + (ocf_ratio - 0.8) * 100
            elif ocf_ratio >= 0.5:
                ocf_score = 40 + (ocf_ratio - 0.5) * 100
            else:
                ocf_score = ocf_ratio * 80
            score += ocf_score * 0.25
            weights += 0.25

        # Debt ratio (lower is better)
        debt = factors.get('debt_ratio')
        if debt is not None:
            if debt < 30:
                debt_score = 90
            elif debt < 50:
                debt_score = 70 + (50 - debt)
            elif debt < 70:
                debt_score = 50 + (70 - debt)
            else:
                debt_score = max(0, 50 - (debt - 70) * 2)
            score += debt_score * 0.10
            weights += 0.10

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def _compute_growth_score(cls, factors: Dict) -> float:
        """
        Compute growth score (30% weight).

        Components:
        - Revenue CAGR (3Y): 35%
        - Profit CAGR (3Y): 35%
        - Growth acceleration: 30%
        """
        score = 0
        weights = 0

        # Revenue CAGR
        rev_cagr = factors.get('revenue_cagr_3y')
        if rev_cagr is not None:
            if rev_cagr >= 30:
                cagr_score = 95
            elif rev_cagr >= 20:
                cagr_score = 75 + (rev_cagr - 20)
            elif rev_cagr >= 10:
                cagr_score = 50 + (rev_cagr - 10) * 2.5
            elif rev_cagr >= 0:
                cagr_score = 30 + rev_cagr * 2
            else:
                cagr_score = max(0, 30 + rev_cagr)  # Negative growth
            score += cagr_score * 0.35
            weights += 0.35

        # Profit CAGR
        profit_cagr = factors.get('profit_cagr_3y')
        if profit_cagr is not None:
            if profit_cagr >= 30:
                cagr_score = 95
            elif profit_cagr >= 20:
                cagr_score = 75 + (profit_cagr - 20)
            elif profit_cagr >= 10:
                cagr_score = 50 + (profit_cagr - 10) * 2.5
            elif profit_cagr >= 0:
                cagr_score = 30 + profit_cagr * 2
            else:
                cagr_score = max(0, 30 + profit_cagr)
            score += cagr_score * 0.35
            weights += 0.35

        # Growth acceleration (YoY vs CAGR)
        rev_yoy = factors.get('revenue_growth_yoy')
        if rev_yoy is not None and rev_cagr is not None:
            acceleration = rev_yoy - rev_cagr
            if acceleration > 10:
                accel_score = 90
            elif acceleration > 0:
                accel_score = 60 + acceleration * 3
            elif acceleration > -10:
                accel_score = 50 + acceleration * 2
            else:
                accel_score = max(0, 30 + acceleration)
            score += accel_score * 0.30
            weights += 0.30

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def _compute_valuation_score(cls, factors: Dict) -> float:
        """
        Compute valuation score (25% weight).

        Components:
        - PEG ratio: 50%
        - PE percentile (vs history): 25%
        - PB percentile (vs history): 25%
        """
        score = 0
        weights = 0

        # PEG score
        peg = factors.get('peg_ratio')
        if peg is not None:
            if peg < 0:
                peg_score = 20  # Negative growth
            elif peg < 0.5:
                peg_score = 95  # Very undervalued
            elif peg <= 1:
                peg_score = 80 + (1 - peg) * 30
            elif peg <= 1.5:
                peg_score = 60 + (1.5 - peg) * 40
            elif peg <= 2:
                peg_score = 40 + (2 - peg) * 40
            else:
                peg_score = max(0, 40 - (peg - 2) * 10)
            score += peg_score * 0.50
            weights += 0.50

        # PE percentile (lower = cheaper)
        pe_pct = factors.get('pe_percentile')
        if pe_pct is not None:
            pe_score = max(0, 100 - pe_pct)
            score += pe_score * 0.25
            weights += 0.25

        # PB percentile
        pb_pct = factors.get('pb_percentile')
        if pb_pct is not None:
            pb_score = max(0, 100 - pb_pct)
            score += pb_score * 0.25
            weights += 0.25

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def _compute_moat_score(cls, factors: Dict) -> float:
        """
        Compute moat/competitive advantage score (10% weight).

        This is approximated from financial indicators:
        - Sustained high ROE suggests competitive advantage
        - Stable/improving margins suggest pricing power
        """
        score = 50  # Default neutral

        # High sustained ROE suggests moat
        roe = factors.get('roe')
        if roe and roe > 15:
            score += 15

        # Gross margin > 40% often indicates pricing power
        margin = factors.get('gross_margin')
        if margin and margin > 40:
            score += 10
        elif margin and margin > 30:
            score += 5

        # Improving margins suggest strengthening position
        margin_stability = factors.get('gross_margin_stability')
        if margin_stability and margin_stability > 80:
            score += 10

        # Low debt allows for competitive flexibility
        debt = factors.get('debt_ratio')
        if debt and debt < 30:
            score += 10

        return min(100, score)

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


def get_long_term_recommendation(
    factors: Dict,
    include_reasoning: bool = True
) -> Dict:
    """
    Generate long-term recommendation from factors.

    Args:
        factors: All computed factors for a stock
        include_reasoning: Whether to include factor breakdown

    Returns:
        Recommendation dict with score, confidence, and optional reasoning
    """
    score = LongTermStrategy.compute_score(factors)

    result = {
        'score': score,
        'is_recommended': LongTermStrategy.is_recommended(score),
        'confidence': LongTermStrategy.get_confidence_level(score),
        'holding_period': '3-12 months',
        'strategy_type': 'long_term',
    }

    if include_reasoning:
        result['factor_scores'] = {
            'quality': LongTermStrategy._compute_quality_score(factors),
            'growth': LongTermStrategy._compute_growth_score(factors),
            'valuation': LongTermStrategy._compute_valuation_score(factors),
            'moat': LongTermStrategy._compute_moat_score(factors),
        }
        result['key_factors'] = _identify_key_factors(factors)

    return result


def _identify_key_factors(factors: Dict) -> List[str]:
    """Identify the most influential factors for explanation."""
    key_factors = []

    # Quality signals
    roe = factors.get('roe')
    if roe:
        if roe >= 20:
            key_factors.append(f"ROE优秀 ({roe:.1f}%)")
        elif roe >= 15:
            key_factors.append(f"ROE良好 ({roe:.1f}%)")
        elif roe < 10:
            key_factors.append(f"ROE偏低 ({roe:.1f}%, 风险)")

    ocf_ratio = factors.get('ocf_to_profit')
    if ocf_ratio:
        if ocf_ratio >= 1.0:
            key_factors.append(f"现金流质量优秀 (OCF/利润={ocf_ratio:.2f})")
        elif ocf_ratio < 0.5:
            key_factors.append(f"现金流质量较差 (OCF/利润={ocf_ratio:.2f}, 风险)")

    # Growth signals
    profit_cagr = factors.get('profit_cagr_3y')
    if profit_cagr:
        if profit_cagr >= 20:
            key_factors.append(f"利润高增长 (3年CAGR={profit_cagr:.1f}%)")
        elif profit_cagr < 0:
            key_factors.append(f"利润负增长 (3年CAGR={profit_cagr:.1f}%, 风险)")

    # Valuation signals
    peg = factors.get('peg_ratio')
    if peg:
        if peg < 1:
            key_factors.append(f"估值吸引力强 (PEG={peg:.2f})")
        elif peg > 2:
            key_factors.append(f"估值偏高 (PEG={peg:.2f}, 风险)")

    pe_pct = factors.get('pe_percentile')
    if pe_pct:
        if pe_pct < 30:
            key_factors.append(f"PE处于历史低位 ({pe_pct:.0f}%分位)")
        elif pe_pct > 80:
            key_factors.append(f"PE处于历史高位 ({pe_pct:.0f}%分位, 风险)")

    # Debt signal
    debt = factors.get('debt_ratio')
    if debt and debt > 70:
        key_factors.append(f"负债率偏高 ({debt:.1f}%, 风险)")

    return key_factors[:5]


def passes_quality_gate(factors: Dict) -> bool:
    """
    Check if stock passes basic quality gates for long-term investing.

    Gates:
    1. ROE >= 10%
    2. OCF/Profit >= 0.5
    3. Debt ratio <= 80%
    """
    roe = factors.get('roe')
    if roe is None or roe < LongTermStrategy.ROE_MIN:
        return False

    ocf = factors.get('ocf_to_profit')
    if ocf is not None and ocf < 0.5:
        return False

    debt = factors.get('debt_ratio')
    if debt is not None and debt > 80:
        return False

    return True
