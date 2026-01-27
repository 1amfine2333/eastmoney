"""
Recommendation-related Pydantic models.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    """Request for generating recommendations."""
    mode: str = "all"  # "short", "long", "all"
    force_refresh: bool = False


class RecommendationResponse(BaseModel):
    """Response with generated recommendations."""
    mode: str
    generated_at: str
    short_term: Optional[Dict[str, Any]] = None
    long_term: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
