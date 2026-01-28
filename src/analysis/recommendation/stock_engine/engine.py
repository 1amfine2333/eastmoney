"""
Stock Recommendation Engine - Orchestrates factor computation and strategy scoring.

This engine:
1. Computes all factors (technical, fundamental, sentiment)
2. Applies strategy weights (short-term or long-term)
3. Generates ranked recommendations
4. Integrates with factor cache for performance
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import time

from src.data_sources.tushare_client import (
    normalize_ts_code,
    denormalize_ts_code,
    get_latest_trade_date,
    format_date_yyyymmdd,
)
from src.storage.db import get_db_connection
from ..factor_store.cache import factor_cache

from .factors.technical import TechnicalFactors
from .factors.fundamental import FundamentalFactors
from .factors.sentiment import SentimentFactors
from .strategies.short_term import ShortTermStrategy, get_short_term_recommendation
from .strategies.long_term import LongTermStrategy, get_long_term_recommendation, passes_quality_gate


class StockRecommendationEngine:
    """
    Stock recommendation engine that orchestrates factor computation
    and strategy-based scoring.
    """

    # Default recommendation limits
    DEFAULT_TOP_N = 20
    MIN_SCORE_SHORT = 60
    MIN_SCORE_LONG = 60

    def __init__(self):
        self._last_compute_time = None

    def compute_factors(
        self,
        ts_code: str,
        trade_date: str = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Compute all factors for a single stock.

        Args:
            ts_code: Stock code
            trade_date: Trade date (default: latest trade date)
            use_cache: Whether to use cached factors

        Returns:
            Dict containing all computed factors
        """
        ts_code = normalize_ts_code(ts_code)
        code = denormalize_ts_code(ts_code)

        if not trade_date:
            trade_date = get_latest_trade_date()
            if not trade_date:
                trade_date = format_date_yyyymmdd()

        # Convert to DB format for cache
        trade_date_db = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

        # Check cache first
        if use_cache:
            cached = factor_cache.get_stock_factors(code, trade_date_db)
            if cached:
                return cached

        # Compute all factor groups
        technical = TechnicalFactors.compute(ts_code, trade_date)
        fundamental = FundamentalFactors.compute(ts_code, trade_date)
        sentiment = SentimentFactors.compute(ts_code, trade_date)

        # Merge factors
        factors = {
            **technical,
            **fundamental,
            **sentiment,
        }

        # Compute composite scores
        factors['short_term_score'] = ShortTermStrategy.compute_score(factors)
        factors['long_term_score'] = LongTermStrategy.compute_score(factors)

        # Cache the result
        if use_cache:
            factor_cache.set_stock_factors(code, trade_date_db, factors)

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
        Get stock recommendations based on strategy.

        Args:
            strategy: 'short_term' or 'long_term'
            top_n: Number of top stocks to return
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

        # Get top stocks from cache/database
        cached_factors = factor_cache.get_top_stocks(
            trade_date_db,
            score_type=strategy,
            limit=top_n * 2,  # Get more to filter
            min_score=min_score
        )

        recommendations = []

        for factors in cached_factors:
            code = factors.get('code', '')
            score = factors.get(f'{strategy}_score', 0)

            if score < min_score:
                continue

            # Get stock info
            stock_info = self._get_stock_info(code)

            if strategy == 'short_term':
                rec = get_short_term_recommendation(factors, include_reasoning=True)
            else:
                # Long-term: Apply quality gate
                if not passes_quality_gate(factors):
                    continue
                rec = get_long_term_recommendation(factors, include_reasoning=True)

            rec.update({
                'code': code,
                'name': stock_info.get('name', ''),
                'industry': stock_info.get('industry', ''),
                'trade_date': trade_date_db,
                'factors': {
                    'roe': factors.get('roe'),
                    'peg_ratio': factors.get('peg_ratio'),
                    'pe_percentile': factors.get('pe_percentile'),
                    'consolidation_score': factors.get('consolidation_score'),
                    'main_inflow_5d': factors.get('main_inflow_5d'),
                }
            })

            recommendations.append(rec)

            if len(recommendations) >= top_n:
                break

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations[:top_n]

    def get_single_recommendation(
        self,
        ts_code: str,
        strategy: str = 'short_term',
        trade_date: str = None
    ) -> Dict:
        """
        Get recommendation for a single stock.

        Args:
            ts_code: Stock code
            strategy: 'short_term' or 'long_term'
            trade_date: Trade date

        Returns:
            Recommendation dict with full details
        """
        ts_code = normalize_ts_code(ts_code)
        code = denormalize_ts_code(ts_code)

        if not trade_date:
            trade_date = get_latest_trade_date()
            if not trade_date:
                trade_date = format_date_yyyymmdd()

        # Compute factors
        factors = self.compute_factors(ts_code, trade_date)

        # Get stock info
        stock_info = self._get_stock_info(code)

        # Generate recommendation
        if strategy == 'short_term':
            rec = get_short_term_recommendation(factors, include_reasoning=True)
        else:
            rec = get_long_term_recommendation(factors, include_reasoning=True)
            rec['passes_quality_gate'] = passes_quality_gate(factors)

        rec.update({
            'code': code,
            'name': stock_info.get('name', ''),
            'industry': stock_info.get('industry', ''),
            'trade_date': f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}",
            'all_factors': factors,
        })

        return rec

    def compare_stocks(
        self,
        codes: List[str],
        strategy: str = 'short_term',
        trade_date: str = None
    ) -> List[Dict]:
        """
        Compare multiple stocks side by side.

        Args:
            codes: List of stock codes
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

    def _get_stock_info(self, code: str) -> Dict:
        """Get basic stock information from database."""
        conn = get_db_connection()
        result = conn.execute(
            "SELECT name, industry FROM stock_basic WHERE symbol = ? OR ts_code LIKE ?",
            (code, f"{code}.%")
        ).fetchone()
        conn.close()

        if result:
            return {'name': result[0], 'industry': result[1]}
        return {'name': '', 'industry': ''}


# Convenience functions

def get_short_term_picks(top_n: int = 20, trade_date: str = None) -> List[Dict]:
    """Get top short-term stock picks."""
    engine = StockRecommendationEngine()
    return engine.get_recommendations(
        strategy='short_term',
        top_n=top_n,
        trade_date=trade_date
    )


def get_long_term_picks(top_n: int = 20, trade_date: str = None) -> List[Dict]:
    """Get top long-term stock picks."""
    engine = StockRecommendationEngine()
    return engine.get_recommendations(
        strategy='long_term',
        top_n=top_n,
        trade_date=trade_date
    )


def analyze_stock(code: str, trade_date: str = None) -> Dict:
    """
    Comprehensive analysis of a single stock.

    Returns both short-term and long-term recommendations.
    """
    engine = StockRecommendationEngine()

    short_term = engine.get_single_recommendation(code, 'short_term', trade_date)
    long_term = engine.get_single_recommendation(code, 'long_term', trade_date)

    return {
        'code': code,
        'name': short_term.get('name', ''),
        'industry': short_term.get('industry', ''),
        'trade_date': short_term.get('trade_date', ''),
        'short_term': short_term,
        'long_term': long_term,
        'factors': short_term.get('all_factors', {}),
    }
