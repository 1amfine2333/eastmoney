"""
Fund Recommendation Engine - Orchestrates factor computation and strategy scoring.

This engine:
1. Computes all factors (performance, risk, manager)
2. Applies strategy weights (momentum or alpha)
3. Generates ranked recommendations
4. Integrates with factor cache for performance
"""
from typing import Dict, List, Optional
from datetime import datetime

from src.data_sources.tushare_client import (
    get_latest_trade_date,
    format_date_yyyymmdd,
)
from src.storage.db import get_db_connection
from ..factor_store.cache import factor_cache

from .factors.performance import PerformanceFactors
from .factors.risk import RiskFactors
from .factors.manager import ManagerFactors
from .strategies.momentum import MomentumStrategy, get_momentum_recommendation
from .strategies.alpha import AlphaStrategy, get_alpha_recommendation


class FundRecommendationEngine:
    """
    Fund recommendation engine that orchestrates factor computation
    and strategy-based scoring.
    """

    DEFAULT_TOP_N = 20
    MIN_SCORE_SHORT = 55
    MIN_SCORE_LONG = 55

    def __init__(self):
        self._last_compute_time = None

    def compute_factors(
        self,
        fund_code: str,
        trade_date: str = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Compute all factors for a single fund.

        Args:
            fund_code: Fund code
            trade_date: Trade date (default: latest trade date)
            use_cache: Whether to use cached factors

        Returns:
            Dict containing all computed factors
        """
        # Clean fund code (remove suffix if present)
        code = fund_code.split('.')[0] if '.' in fund_code else fund_code

        if not trade_date:
            trade_date = get_latest_trade_date()
            if not trade_date:
                trade_date = format_date_yyyymmdd()

        trade_date_db = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

        # Check cache first
        if use_cache:
            cached = factor_cache.get_fund_factors(code, trade_date_db)
            if cached:
                return cached

        # Compute all factor groups
        performance = PerformanceFactors.compute(code, trade_date)
        risk = RiskFactors.compute(code, trade_date)
        manager = ManagerFactors.compute(code, trade_date)

        # Merge factors
        factors = {
            **performance,
            **risk,
            **manager,
        }

        # Compute composite scores
        factors['short_term_score'] = MomentumStrategy.compute_score(factors)
        factors['long_term_score'] = AlphaStrategy.compute_score(factors)

        # Cache the result
        if use_cache:
            factor_cache.set_fund_factors(code, trade_date_db, factors)

        return factors

    def get_recommendations(
        self,
        strategy: str = 'short_term',
        top_n: int = None,
        trade_date: str = None,
        min_score: float = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Get fund recommendations based on strategy.

        Args:
            strategy: 'short_term' (momentum) or 'long_term' (alpha)
            top_n: Number of top funds to return
            trade_date: Trade date
            min_score: Minimum score threshold
            use_cache: Whether to use cached factors

        Returns:
            List of recommendation dicts sorted by score
        """
        if top_n is None:
            top_n = self.DEFAULT_TOP_N

        if min_score is None:
            min_score = self.MIN_SCORE_SHORT if strategy == 'short_term' else self.MIN_SCORE_LONG

        if not trade_date:
            trade_date = get_latest_trade_date()
            if not trade_date:
                trade_date = format_date_yyyymmdd()

        trade_date_db = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

        # Get top funds from cache/database
        cached_factors = factor_cache.get_top_funds(
            trade_date_db,
            score_type=strategy,
            limit=top_n * 2,
            min_score=min_score
        )

        recommendations = []

        for factors in cached_factors:
            code = factors.get('code', '')
            score = factors.get(f'{strategy}_score', 0)

            if score < min_score:
                continue

            # Get fund info
            fund_info = self._get_fund_info(code)

            if strategy == 'short_term':
                rec = get_momentum_recommendation(factors, include_reasoning=True)
            else:
                rec = get_alpha_recommendation(factors, include_reasoning=True)

            rec.update({
                'code': code,
                'name': fund_info.get('name', ''),
                'type': fund_info.get('type', ''),
                'trade_date': trade_date_db,
                'factors': {
                    'sharpe_1y': factors.get('sharpe_1y'),
                    'max_drawdown_1y': factors.get('max_drawdown_1y'),
                    'return_1y': factors.get('return_1y'),
                    'return_1m': factors.get('return_1m'),
                    'manager_tenure_years': factors.get('manager_tenure_years'),
                }
            })

            recommendations.append(rec)

            if len(recommendations) >= top_n:
                break

        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations[:top_n]

    def get_single_recommendation(
        self,
        fund_code: str,
        strategy: str = 'short_term',
        trade_date: str = None
    ) -> Dict:
        """
        Get recommendation for a single fund.

        Args:
            fund_code: Fund code
            strategy: 'short_term' or 'long_term'
            trade_date: Trade date

        Returns:
            Recommendation dict with full details
        """
        code = fund_code.split('.')[0] if '.' in fund_code else fund_code

        if not trade_date:
            trade_date = get_latest_trade_date()
            if not trade_date:
                trade_date = format_date_yyyymmdd()

        # Compute factors
        factors = self.compute_factors(fund_code, trade_date)

        # Get fund info
        fund_info = self._get_fund_info(code)

        # Generate recommendation
        if strategy == 'short_term':
            rec = get_momentum_recommendation(factors, include_reasoning=True)
        else:
            rec = get_alpha_recommendation(factors, include_reasoning=True)

        rec.update({
            'code': code,
            'name': fund_info.get('name', ''),
            'type': fund_info.get('type', ''),
            'trade_date': f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}",
            'all_factors': factors,
        })

        return rec

    def compare_funds(
        self,
        codes: List[str],
        strategy: str = 'short_term',
        trade_date: str = None
    ) -> List[Dict]:
        """
        Compare multiple funds side by side.

        Args:
            codes: List of fund codes
            strategy: Strategy to use for scoring
            trade_date: Trade date

        Returns:
            List of recommendations sorted by score
        """
        recommendations = []

        for code in codes:
            rec = self.get_single_recommendation(code, strategy, trade_date)
            recommendations.append(rec)

        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations

    def _get_fund_info(self, code: str) -> Dict:
        """Get basic fund information from database."""
        conn = get_db_connection()

        # Try funds table first
        result = conn.execute(
            "SELECT name, style FROM funds WHERE code = ?",
            (code,)
        ).fetchone()

        if result:
            conn.close()
            return {'name': result[0], 'type': result[1] or ''}

        conn.close()
        return {'name': '', 'type': ''}


# Convenience functions

def get_momentum_picks(top_n: int = 20, trade_date: str = None) -> List[Dict]:
    """Get top short-term momentum fund picks."""
    engine = FundRecommendationEngine()
    return engine.get_recommendations(
        strategy='short_term',
        top_n=top_n,
        trade_date=trade_date
    )


def get_alpha_picks(top_n: int = 20, trade_date: str = None) -> List[Dict]:
    """Get top long-term alpha fund picks."""
    engine = FundRecommendationEngine()
    return engine.get_recommendations(
        strategy='long_term',
        top_n=top_n,
        trade_date=trade_date
    )


def analyze_fund(code: str, trade_date: str = None) -> Dict:
    """
    Comprehensive analysis of a single fund.

    Returns both short-term and long-term recommendations.
    """
    engine = FundRecommendationEngine()

    short_term = engine.get_single_recommendation(code, 'short_term', trade_date)
    long_term = engine.get_single_recommendation(code, 'long_term', trade_date)

    return {
        'code': code,
        'name': short_term.get('name', ''),
        'type': short_term.get('type', ''),
        'trade_date': short_term.get('trade_date', ''),
        'short_term': short_term,
        'long_term': long_term,
        'factors': short_term.get('all_factors', {}),
    }
