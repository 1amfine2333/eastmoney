"""
Long-Term Fund Strategy (Alpha) - For 6+ month holding period.

Weight allocation:
- Risk-adjusted returns: 35%
- Drawdown characteristics: 25%
- Manager quality: 25%
- Holdings quality: 15%
"""
from typing import Dict, List


class AlphaStrategy:
    """
    Long-term fund recommendation strategy focused on alpha generation.

    Prioritizes consistent risk-adjusted returns and manager quality.
    """

    WEIGHTS = {
        'risk_adjusted': 0.35,
        'drawdown': 0.25,
        'manager': 0.25,
        'holdings': 0.15,
    }

    MIN_SCORE_THRESHOLD = 60
    HIGH_SCORE_THRESHOLD = 75

    @classmethod
    def compute_score(cls, factors: Dict) -> float:
        """
        Compute overall long-term alpha score for a fund.

        Args:
            factors: Dict containing all factor values

        Returns:
            Score 0-100 (higher = better long-term opportunity)
        """
        scores = {
            'risk_adjusted': cls._compute_risk_adjusted_score(factors),
            'drawdown': cls._compute_drawdown_score(factors),
            'manager': cls._compute_manager_score(factors),
            'holdings': cls._compute_holdings_score(factors),
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
    def _compute_risk_adjusted_score(cls, factors: Dict) -> float:
        """
        Compute risk-adjusted return score (35% weight).

        Components:
        - Sharpe (1Y): 40%
        - Sortino (1Y): 30%
        - Calmar (1Y): 30%
        """
        score = 0
        weights = 0

        # Sharpe ratio
        sharpe = factors.get('sharpe_1y')
        if sharpe is not None:
            if sharpe >= 2:
                sharpe_score = 95
            elif sharpe >= 1:
                sharpe_score = 70 + (sharpe - 1) * 25
            elif sharpe >= 0.5:
                sharpe_score = 50 + (sharpe - 0.5) * 40
            elif sharpe >= 0:
                sharpe_score = 30 + sharpe * 40
            else:
                sharpe_score = max(0, 30 + sharpe * 15)
            score += sharpe_score * 0.40
            weights += 0.40

        # Sortino ratio
        sortino = factors.get('sortino_1y')
        if sortino is not None:
            if sortino >= 2:
                sortino_score = 95
            elif sortino >= 1:
                sortino_score = 70 + (sortino - 1) * 25
            elif sortino >= 0:
                sortino_score = 40 + sortino * 30
            else:
                sortino_score = max(0, 40 + sortino * 20)
            score += sortino_score * 0.30
            weights += 0.30

        # Calmar ratio
        calmar = factors.get('calmar_1y')
        if calmar is not None:
            if calmar >= 1:
                calmar_score = 90 + min(10, (calmar - 1) * 10)
            elif calmar >= 0.5:
                calmar_score = 70 + (calmar - 0.5) * 40
            elif calmar >= 0:
                calmar_score = 40 + calmar * 60
            else:
                calmar_score = max(0, 40 + calmar * 20)
            score += calmar_score * 0.30
            weights += 0.30

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def _compute_drawdown_score(cls, factors: Dict) -> float:
        """
        Compute drawdown characteristics score (25% weight).

        Components:
        - Max drawdown (lower = better): 60%
        - Recovery time (shorter = better): 40%
        """
        score = 0
        weights = 0

        # Max drawdown
        max_dd = factors.get('max_drawdown_1y')
        if max_dd is not None:
            # Lower drawdown = higher score
            if max_dd < 5:
                dd_score = 95
            elif max_dd < 10:
                dd_score = 80 + (10 - max_dd) * 3
            elif max_dd < 20:
                dd_score = 50 + (20 - max_dd) * 3
            elif max_dd < 30:
                dd_score = 30 + (30 - max_dd) * 2
            else:
                dd_score = max(0, 30 - (max_dd - 30))
            score += dd_score * 0.60
            weights += 0.60

        # Recovery time
        recovery = factors.get('avg_recovery_days')
        if recovery is not None:
            # Shorter recovery = higher score
            if recovery < 20:
                recovery_score = 90
            elif recovery < 40:
                recovery_score = 70 + (40 - recovery) * 1
            elif recovery < 60:
                recovery_score = 50 + (60 - recovery) * 1
            else:
                recovery_score = max(20, 50 - (recovery - 60) * 0.5)
            score += recovery_score * 0.40
            weights += 0.40

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def _compute_manager_score(cls, factors: Dict) -> float:
        """
        Compute manager quality score (25% weight).

        Components:
        - Tenure: 35%
        - Bull market alpha: 25%
        - Bear market alpha: 25%
        - Style consistency: 15%
        """
        score = 0
        weights = 0

        # Tenure
        tenure = factors.get('manager_tenure_years')
        if tenure is not None:
            if tenure >= 5:
                tenure_score = min(95, 85 + (tenure - 5) * 2)
            elif tenure >= 3:
                tenure_score = 70 + (tenure - 3) * 7.5
            elif tenure >= 1:
                tenure_score = 50 + (tenure - 1) * 10
            else:
                tenure_score = 30 + tenure * 20
            score += tenure_score * 0.35
            weights += 0.35

        # Bull market alpha
        alpha_bull = factors.get('manager_alpha_bull')
        if alpha_bull is not None:
            if alpha_bull >= 2:
                alpha_score = 90
            elif alpha_bull >= 1:
                alpha_score = 70 + (alpha_bull - 1) * 20
            elif alpha_bull >= 0:
                alpha_score = 50 + alpha_bull * 20
            else:
                alpha_score = max(0, 50 + alpha_bull * 25)
            score += alpha_score * 0.25
            weights += 0.25

        # Bear market alpha
        alpha_bear = factors.get('manager_alpha_bear')
        if alpha_bear is not None:
            if alpha_bear >= 1:
                alpha_score = 80
            elif alpha_bear >= 0:
                alpha_score = 60 + alpha_bear * 20
            else:
                alpha_score = max(0, 60 + alpha_bear * 30)
            score += alpha_score * 0.25
            weights += 0.25

        # Style consistency
        consistency = factors.get('style_consistency')
        if consistency is not None:
            score += consistency * 0.15
            weights += 0.15

        if weights > 0:
            return score / weights

        return 50.0

    @classmethod
    def _compute_holdings_score(cls, factors: Dict) -> float:
        """
        Compute holdings quality score (15% weight).

        This is a placeholder - actual implementation needs:
        - Holdings average ROE
        - Diversification level
        - Turnover rate analysis
        """
        # Default to neutral - needs holdings data
        holdings_roe = factors.get('holdings_avg_roe')
        if holdings_roe is not None:
            if holdings_roe >= 15:
                return 80
            elif holdings_roe >= 10:
                return 60
            else:
                return 40

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


def get_alpha_recommendation(factors: Dict, include_reasoning: bool = True) -> Dict:
    """
    Generate long-term alpha recommendation for a fund.

    Args:
        factors: All computed factors
        include_reasoning: Whether to include breakdown

    Returns:
        Recommendation dict
    """
    score = AlphaStrategy.compute_score(factors)

    result = {
        'score': score,
        'is_recommended': AlphaStrategy.is_recommended(score),
        'confidence': AlphaStrategy.get_confidence_level(score),
        'holding_period': '6-12 months',
        'strategy_type': 'alpha',
    }

    if include_reasoning:
        result['factor_scores'] = {
            'risk_adjusted': AlphaStrategy._compute_risk_adjusted_score(factors),
            'drawdown': AlphaStrategy._compute_drawdown_score(factors),
            'manager': AlphaStrategy._compute_manager_score(factors),
            'holdings': AlphaStrategy._compute_holdings_score(factors),
        }
        result['key_factors'] = _identify_key_factors(factors)

    return result


def _identify_key_factors(factors: Dict) -> List[str]:
    """Identify key factors for explanation."""
    key_factors = []

    # Risk-adjusted returns
    sharpe = factors.get('sharpe_1y')
    if sharpe is not None:
        if sharpe >= 1.5:
            key_factors.append(f"年化夏普比率优秀 ({sharpe:.2f})")
        elif sharpe >= 1:
            key_factors.append(f"年化夏普比率良好 ({sharpe:.2f})")
        elif sharpe < 0.5:
            key_factors.append(f"年化夏普比率较低 ({sharpe:.2f})")

    # Drawdown
    max_dd = factors.get('max_drawdown_1y')
    if max_dd is not None:
        if max_dd < 10:
            key_factors.append(f"回撤控制优秀 (最大回撤{max_dd:.1f}%)")
        elif max_dd > 25:
            key_factors.append(f"回撤较大 (最大回撤{max_dd:.1f}%, 风险)")

    # Manager
    tenure = factors.get('manager_tenure_years')
    if tenure is not None:
        if tenure >= 5:
            key_factors.append(f"基金经理经验丰富 ({tenure:.1f}年)")
        elif tenure < 2:
            key_factors.append(f"基金经理任期较短 ({tenure:.1f}年)")

    # Long-term return
    ret_1y = factors.get('return_1y')
    if ret_1y is not None:
        if ret_1y >= 20:
            key_factors.append(f"年收益优秀 (+{ret_1y:.1f}%)")
        elif ret_1y < 0:
            key_factors.append(f"年收益为负 ({ret_1y:.1f}%)")

    return key_factors[:4]
