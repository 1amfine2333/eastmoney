"""
AI Recommendation System v2 - 智能投资推荐系统

Quantitative factor-based stock and fund recommendation engine.
Key principles:
- Quantitative models select, LLM explains
- Predict breakouts, don't chase rallies
- Quality and risk-adjusted returns over raw performance
"""
from .engine import RecommendationEngine
from .engine_v2 import RecommendationEngineV2, get_v2_recommendations

__all__ = ['RecommendationEngine', 'RecommendationEngineV2', 'get_v2_recommendations']
