"""
News-related Pydantic models.
"""
from typing import Optional
from pydantic import BaseModel


class NewsBookmarkRequest(BaseModel):
    """Request model for bookmarking news."""
    news_title: Optional[str] = None
    news_source: Optional[str] = None
    news_url: Optional[str] = None
    news_category: Optional[str] = None
    bookmarked: Optional[bool] = None  # If None, toggle


class NewsReadRequest(BaseModel):
    """Request model for marking news as read."""
    news_title: Optional[str] = None
    news_source: Optional[str] = None
    news_url: Optional[str] = None
    news_category: Optional[str] = None
