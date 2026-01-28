"""
Recommendation Engine v2 - Main orchestrator for the new quantitative recommendation system.

This is a new implementation that:
1. Uses quantitative factor models for selection (not LLM)
2. Only uses LLM for explanation generation
3. Provides both stock and fund recommendations
4. Maintains API compatibility with v1

Key principles:
- Quantitative models select, LLM explains
- Predict breakouts, don't chase rallies
- Quality and risk-adjusted returns over raw performance
"""
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.data_sources.tushare_client import (
    get_latest_trade_date,
    format_date_yyyymmdd,
)
from src.storage.db import (
    insert_recommendation_record,
    get_recommendation_performance_stats,
)

from .stock_engine import StockRecommendationEngine
from .fund_engine import FundRecommendationEngine
from .factor_store.cache import factor_cache
from .llm_synthesis.explainer import RecommendationExplainer, explain_recommendations_sync


class RecommendationEngineV2:
    """
    Recommendation Engine v2 - Quantitative factor-based recommendations.

    This engine uses pre-computed factors and quantitative strategies
    to generate recommendations, with LLM used only for explanations.
    """

    def __init__(self, use_llm_explanations: bool = True):
        """
        Initialize the v2 recommendation engine.

        Args:
            use_llm_explanations: Whether to use LLM for generating explanations
        """
        self.stock_engine = StockRecommendationEngine()
        self.fund_engine = FundRecommendationEngine()
        self.use_llm = use_llm_explanations

    def generate_recommendations(
        self,
        mode: str = "all",
        stock_limit: int = 20,
        fund_limit: int = 20,
        user_preferences: Optional[Dict[str, Any]] = None,
        trade_date: str = None,
    ) -> Dict[str, Any]:
        """
        Generate investment recommendations using quantitative models.

        Args:
            mode: "short", "long", or "all"
            stock_limit: Maximum number of stocks to recommend
            fund_limit: Maximum number of funds to recommend
            user_preferences: User preferences for filtering (optional)
            trade_date: Trade date for factor lookup

        Returns:
            Dict containing recommendations and metadata
        """
        start_time = datetime.now()

        if not trade_date:
            trade_date = get_latest_trade_date()
            if not trade_date:
                trade_date = format_date_yyyymmdd()

        results = {
            "mode": mode,
            "generated_at": start_time.isoformat(),
            "trade_date": trade_date,
            "engine_version": "v2",
            "short_term": None,
            "long_term": None,
            "metadata": {
                "factor_computation_time": 0,
                "explanation_time": 0,
                "total_time": 0,
            }
        }

        # Generate short-term recommendations
        if mode in ["short", "all"]:
            factor_start = datetime.now()

            short_stocks = self.stock_engine.get_recommendations(
                strategy='short_term',
                top_n=stock_limit,
                trade_date=trade_date
            )

            short_funds = self.fund_engine.get_recommendations(
                strategy='short_term',
                top_n=fund_limit,
                trade_date=trade_date
            )

            # Apply user preferences if provided
            if user_preferences:
                short_stocks = self._apply_stock_preferences(short_stocks, user_preferences)
                short_funds = self._apply_fund_preferences(short_funds, user_preferences)

            # Add LLM explanations if enabled
            if self.use_llm:
                short_stocks = explain_recommendations_sync(short_stocks, 'stock', 'short_term')
                short_funds = explain_recommendations_sync(short_funds, 'fund', 'short_term')

            results["short_term"] = {
                "stocks": short_stocks,
                "funds": short_funds,
                "market_view": self._get_market_summary(),
            }

            results["metadata"]["factor_computation_time"] = (
                datetime.now() - factor_start
            ).total_seconds()

        # Generate long-term recommendations
        if mode in ["long", "all"]:
            long_stocks = self.stock_engine.get_recommendations(
                strategy='long_term',
                top_n=stock_limit,
                trade_date=trade_date
            )

            long_funds = self.fund_engine.get_recommendations(
                strategy='long_term',
                top_n=fund_limit,
                trade_date=trade_date
            )

            if user_preferences:
                long_stocks = self._apply_stock_preferences(long_stocks, user_preferences)
                long_funds = self._apply_fund_preferences(long_funds, user_preferences)

            if self.use_llm:
                long_stocks = explain_recommendations_sync(long_stocks, 'stock', 'long_term')
                long_funds = explain_recommendations_sync(long_funds, 'fund', 'long_term')

            results["long_term"] = {
                "stocks": long_stocks,
                "funds": long_funds,
                "macro_view": self._get_macro_summary(),
            }

        results["metadata"]["total_time"] = (datetime.now() - start_time).total_seconds()

        # Record recommendations for performance tracking
        self._record_recommendations(results)

        return results

    def get_stock_analysis(self, code: str, trade_date: str = None) -> Dict:
        """
        Get comprehensive analysis for a single stock.

        Args:
            code: Stock code
            trade_date: Trade date

        Returns:
            Analysis dict with short-term and long-term recommendations
        """
        from .stock_engine.engine import analyze_stock
        return analyze_stock(code, trade_date)

    def get_fund_analysis(self, code: str, trade_date: str = None) -> Dict:
        """
        Get comprehensive analysis for a single fund.

        Args:
            code: Fund code
            trade_date: Trade date

        Returns:
            Analysis dict with short-term and long-term recommendations
        """
        from .fund_engine.engine import analyze_fund
        return analyze_fund(code, trade_date)

    def get_performance_stats(
        self,
        rec_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Get recommendation performance statistics.

        Args:
            rec_type: Filter by recommendation type
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Performance statistics by recommendation type
        """
        return get_recommendation_performance_stats(rec_type, start_date, end_date)

    def _apply_stock_preferences(
        self,
        stocks: List[Dict],
        prefs: Dict[str, Any]
    ) -> List[Dict]:
        """Apply user preferences to filter stocks."""
        filtered = []

        for stock in stocks:
            # Skip ST stocks if preference set
            if prefs.get('avoid_st_stocks', True):
                name = stock.get('name', '')
                if 'ST' in name or '*ST' in name:
                    continue

            # Sector filter
            stock_industry = stock.get('industry', '')
            excluded_sectors = prefs.get('excluded_sectors', [])

            if excluded_sectors and stock_industry:
                if any(exc in stock_industry for exc in excluded_sectors):
                    continue

            # ROE filter for long-term
            roe = stock.get('factors', {}).get('roe')
            min_roe = prefs.get('min_roe')
            if min_roe and roe and roe < min_roe:
                continue

            # Boost score for preferred sectors
            preferred_sectors = prefs.get('preferred_sectors', [])
            if preferred_sectors and stock_industry:
                if any(pref in stock_industry for pref in preferred_sectors):
                    stock = stock.copy()
                    stock['score'] = stock.get('score', 0) * 1.15

            filtered.append(stock)

        filtered.sort(key=lambda x: x.get('score', 0), reverse=True)
        return filtered

    def _apply_fund_preferences(
        self,
        funds: List[Dict],
        prefs: Dict[str, Any]
    ) -> List[Dict]:
        """Apply user preferences to filter funds."""
        filtered = []

        preferred_types = prefs.get('preferred_fund_types', [])
        excluded_types = prefs.get('excluded_fund_types', [])

        for fund in funds:
            fund_type = fund.get('type', '')

            # Type filter
            if preferred_types:
                if not any(pref in fund_type for pref in preferred_types):
                    continue

            if excluded_types:
                if any(exc in fund_type for exc in excluded_types):
                    continue

            # Max drawdown filter
            max_dd = fund.get('factors', {}).get('max_drawdown_1y')
            max_dd_tolerance = prefs.get('max_drawdown_tolerance')
            if max_dd_tolerance and max_dd and max_dd > max_dd_tolerance * 100:
                continue

            filtered.append(fund)

        filtered.sort(key=lambda x: x.get('score', 0), reverse=True)
        return filtered

    def _get_market_summary(self) -> str:
        """Get brief market summary for short-term context."""
        try:
            from src.data_sources.akshare_api import get_market_indices

            indices = get_market_indices()
            parts = []
            for name, data in list(indices.items())[:3]:
                change = data.get('涨跌幅', 'N/A')
                parts.append(f"{name}: {change}%")
            return " | ".join(parts)
        except:
            return "市场数据获取中..."

    def _get_macro_summary(self) -> str:
        """Get brief macro summary for long-term context."""
        return "宏观环境分析需要结合最新经济数据"

    def _record_recommendations(self, results: Dict) -> None:
        """Record recommendations for performance tracking."""
        trade_date = results.get('trade_date', '')

        for term in ['short_term', 'long_term']:
            if results.get(term):
                rec_type_prefix = 'short' if term == 'short_term' else 'long'

                # Record stock recommendations
                for stock in results[term].get('stocks', [])[:10]:
                    try:
                        insert_recommendation_record({
                            'code': stock.get('code'),
                            'rec_type': f'{rec_type_prefix}_stock',
                            'rec_date': trade_date,
                            'rec_price': stock.get('factors', {}).get('price'),
                            'rec_score': stock.get('score'),
                            'target_return_pct': 5.0 if term == 'short_term' else 10.0,
                            'stop_loss_pct': -3.0 if term == 'short_term' else -5.0,
                        })
                    except:
                        pass

                # Record fund recommendations
                for fund in results[term].get('funds', [])[:10]:
                    try:
                        insert_recommendation_record({
                            'code': fund.get('code'),
                            'rec_type': f'{rec_type_prefix}_fund',
                            'rec_date': trade_date,
                            'rec_score': fund.get('score'),
                            'target_return_pct': 3.0 if term == 'short_term' else 8.0,
                            'stop_loss_pct': -2.0 if term == 'short_term' else -4.0,
                        })
                    except:
                        pass


# Convenience function for backward compatibility
def get_v2_recommendations(
    mode: str = "all",
    stock_limit: int = 20,
    fund_limit: int = 20,
    use_llm: bool = True,
    user_preferences: Optional[Dict] = None
) -> Dict:
    """
    Generate recommendations using the v2 engine.

    This is a convenience function for easy access to the new system.
    """
    engine = RecommendationEngineV2(use_llm_explanations=use_llm)
    return engine.generate_recommendations(
        mode=mode,
        stock_limit=stock_limit,
        fund_limit=fund_limit,
        user_preferences=user_preferences
    )
