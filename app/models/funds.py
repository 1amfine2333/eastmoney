"""
Fund-related Pydantic models.
"""
from typing import Optional, List
from pydantic import BaseModel


class FundItem(BaseModel):
    """Fund item for user's watchlist."""
    code: str
    name: str
    style: Optional[str] = "Unknown"
    focus: Optional[List[str]] = []
    pre_market_time: Optional[str] = None
    post_market_time: Optional[str] = None
    is_active: bool = True


class FundCompareRequest(BaseModel):
    """Request for comparing multiple funds."""
    codes: List[str]
