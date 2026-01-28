"""
Short-Term Fund Strategy (Momentum) - For 1-3 month holding period.

Weight allocation:
- Risk-adjusted momentum: 40%
- Sector timing: 30%
- Capital flow signals: 20%
- Manager signal: 10%
"""
from typing import Dict, List


class MomentumStrategy:
    """
    Short-term fund recommendation strategy based on momentum.

    Focuses on funds with strong recent performance adjusted for risk.
    """

    WEIGHTS = {
        'momentum': 0.40,
        'sector': 0.30,
        'flow': 0.20,
        'manager': 0.10,
    }

    MIN_SCORE_THRESHOLD = 60
    HIGH_SCORE_THRESHOLD = 75

    @classmethod
    def compute_score(cls, factors: Dict) -> float:
        """
        Compute overall short-term score for a fund.

        Args:
            factors: Dict containing all factor values

        Returns:
            Score 0-100 (higher = better short-term opportunity)
        """
        scores = {
            'momentum': cls._compute_momentum_score(factors),
            'sector': cls._compute_sector_score(factors),
            'flow': cls._compute_flow_score(factors),
            'manager': cls._compute_manager_score(factors),
        }

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
    def _compute_momentum_score(cls, factors: Dict) -> float:
        """
        Compute risk-adjusted momentum score (40% weight).

        Components:
        - 1-week return rank: 30%
        - 1-month return rank: 40%
        - Sharpe (20-day): 30%
        """
        score = 0
        weights = 0

        # 1-week return
        ret_1w = factors.get('return_1w')
        if ret_1w is not None:
            # Normalize: -5% to 5% -> 0 to 100
            ret_score = 50 + (ret_1w * 10)
            ret_score = max(0, min(100, ret_score))
            score += ret_score * 0.30
            weights += 0.30

        # 1-month return
        ret_1m = factors.get('return_1m')
        if ret_1m is not None:
            ret_score = 50 + (ret_1m * 5)
            ret_score = max(0, min(100, ret_score))
            score += ret_score * 0.40
            weights += 0.40

        # Sharpe ratio (risk adjustment)
        sharpe = factors.get('sharpe_20d')
        if sharpe is not None:
            if sharpe >= 2:
                sharpe_score = 95
            elif sharpe >= 1:
                sharpe_score = 70 + (sharpe - 1) * 25
            elif sharpe >= 0:
                sharpe_score = 40 + sharpe * 30
            else:
                sharpe_score = max(0, 40 + sharpe * 20)
            score += sharpe_score * 0.30
            weights += 0.30

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def _compute_sector_score(cls, factors: Dict) -> float:
        """
        Compute sector timing score (30% weight).

        This is a placeholder - actual implementation needs:
        - Fund holdings vs hot sectors matching
        - Sector rotation timing signals
        """
        # Default to neutral - actual implementation needs holdings data
        return 50.0

    @classmethod
    def _compute_flow_score(cls, factors: Dict) -> float:
        """
        Compute capital flow score (20% weight).

        This is a placeholder - actual implementation needs:
        - Institutional purchase trends
        - ETF premium/discount data
        """
        # Default to neutral
        return 50.0

    @classmethod
    def _compute_manager_score(cls, factors: Dict) -> float:
        """
        Compute manager signal score (10% weight).

        Uses manager consistency as a stability indicator.
        """
        consistency = factors.get('style_consistency')
        if consistency is not None:
            return consistency

        return 50.0

    @classmethod
    def is_recommended(cls, score: float) -> bool:
        return score >= cls.MIN_SCORE_THRESHOLD

    @classmethod
    def get_confidence_level(cls, score: float) -> str:
        if score >= cls.HIGH_SCORE_THRESHOLD:
            return 'high'
        elif score >= cls.MIN_SCORE_THRESHOLD:
            return 'medium'
        else:
            return 'low'


def get_momentum_recommendation(factors: Dict, include_reasoning: bool = True) -> Dict:
    """
    Generate short-term momentum recommendation for a fund.

    Args:
        factors: All computed factors
        include_reasoning: Whether to include breakdown

    Returns:
        Recommendation dict
    """
    score = MomentumStrategy.compute_score(factors)

    result = {
        'score': score,
        'is_recommended': MomentumStrategy.is_recommended(score),
        'confidence': MomentumStrategy.get_confidence_level(score),
        'holding_period': '1-3 months',
        'strategy_type': 'momentum',
    }

    if include_reasoning:
        result['factor_scores'] = {
            'momentum': MomentumStrategy._compute_momentum_score(factors),
            'sector': MomentumStrategy._compute_sector_score(factors),
            'flow': MomentumStrategy._compute_flow_score(factors),
            'manager': MomentumStrategy._compute_manager_score(factors),
        }
        result['key_factors'] = _identify_key_factors(factors)

    return result


def _identify_key_factors(factors: Dict) -> List[str]:
    """Identify key factors for explanation."""
    key_factors = []

    # Performance
    ret_1m = factors.get('return_1m')
    if ret_1m is not None:
        if ret_1m >= 5:
            key_factors.append(f"近1月收益优秀 (+{ret_1m:.2f}%)")
        elif ret_1m >= 2:
            key_factors.append(f"近1月收益良好 (+{ret_1m:.2f}%)")
        elif ret_1m < -5:
            key_factors.append(f"近1月回撤较大 ({ret_1m:.2f}%, 风险)")

    # Risk metrics
    sharpe = factors.get('sharpe_20d')
    if sharpe is not None:
        if sharpe >= 1.5:
            key_factors.append(f"短期夏普比率优秀 ({sharpe:.2f})")
        elif sharpe < 0:
            key_factors.append(f"短期夏普比率为负 ({sharpe:.2f}, 风险)")

    # Volatility
    vol = factors.get('volatility_20d')
    if vol is not None:
        if vol > 30:
            key_factors.append(f"短期波动较大 ({vol:.1f}%)")
        elif vol < 10:
            key_factors.append(f"短期波动较低 ({vol:.1f}%)")

    return key_factors[:4]
