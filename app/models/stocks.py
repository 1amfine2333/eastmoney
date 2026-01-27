"""
Stock-related Pydantic models.
"""
from typing import Optional
from pydantic import BaseModel


class StockItem(BaseModel):
    """Stock item for user's watchlist."""
    code: str
    name: str
    market: Optional[str] = ""
    sector: Optional[str] = ""
    pre_market_time: Optional[str] = "08:30"
    post_market_time: Optional[str] = "15:30"
    is_active: bool = True
    price: Optional[float] = None
    change_pct: Optional[float] = None
    volume: Optional[float] = None


class StockAnalyzeRequest(BaseModel):
    """Request for stock analysis."""
    mode: str = "pre"  # 'pre' or 'post'
