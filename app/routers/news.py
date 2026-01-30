"""
News center endpoints.
"""
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from app.models.news import NewsBookmarkRequest, NewsReadRequest
from app.models.auth import User
from app.core.dependencies import get_current_user
from app.core.utils import sanitize_for_json
from src.services.news_service import news_service

router = APIRouter(prefix="/api/news", tags=["News"])


@router.get("/feed")
async def get_news_feed(
    category: str = "all",
    page: int = 1,
    page_size: int = 50,
    since_days: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get personalized news feed.
    Categories: all, flash, fund, announcement, research, hot
    """
    try:
        data = await asyncio.to_thread(
            news_service.get_personalized_feed,
            user_id=current_user.id,
            category=category,
            page=page,
            page_size=page_size,
            since_days=since_days,
        )
        return sanitize_for_json(data)
    except Exception as e:
        print(f"Error fetching news feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bookmarks")
async def get_news_bookmarks(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """Get user's bookmarked news."""
    try:
        bookmarks = await asyncio.to_thread(
            news_service.get_bookmarks,
            user_id=current_user.id,
            limit=limit,
            offset=offset
        )
        return {"bookmarks": bookmarks, "total": len(bookmarks)}
    except Exception as e:
        print(f"Error fetching bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watchlist-summary")
async def get_news_watchlist_summary(current_user: User = Depends(get_current_user)):
    """Get summary of news related to user's watchlist."""
    try:
        summary = await asyncio.to_thread(
            news_service.get_watchlist_news_summary,
            user_id=current_user.id
        )
        return sanitize_for_json(summary)
    except Exception as e:
        print(f"Error fetching watchlist summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/announcements")
async def get_news_announcements(
    stock_code: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get company announcements."""
    try:
        announcements = await asyncio.to_thread(
            news_service.get_announcements,
            stock_code=stock_code,
            limit=limit
        )
        return {"announcements": announcements, "total": len(announcements)}
    except Exception as e:
        print(f"Error fetching announcements: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/research")
async def search_news_research(
    query: str,
    limit: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Search for research reports via Tavily."""
    try:
        results = await asyncio.to_thread(
            news_service.search_research_reports,
            query=query,
            limit=limit
        )
        return {"results": results, "total": len(results)}
    except Exception as e:
        print(f"Error searching research: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hot")
async def get_hot_news(limit: int = 30):
    """Get hot/trending news (no auth required)."""
    try:
        news = await asyncio.to_thread(
            news_service.get_hot_news,
            limit=limit
        )
        return {"news": news, "total": len(news)}
    except Exception as e:
        print(f"Error fetching hot news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{news_id}")
async def get_news_detail(
    news_id: str,
    title: str = "",
    content: str = "",
    current_user: User = Depends(get_current_user)
):
    """Get news detail with AI analysis."""
    try:
        # Mark as read
        await asyncio.to_thread(
            news_service.mark_read,
            user_id=current_user.id,
            news_id=news_id,
            news_title=title
        )

        # Get AI analysis
        analysis = await asyncio.to_thread(
            news_service.analyze_news,
            news_id=news_id,
            title=title,
            content=content
        )

        return sanitize_for_json({
            "news_id": news_id,
            "analysis": analysis,
            "is_read": True,
        })
    except Exception as e:
        print(f"Error fetching news detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{news_id}/bookmark")
async def toggle_news_bookmark(
    news_id: str,
    request: NewsBookmarkRequest,
    current_user: User = Depends(get_current_user)
):
    """Toggle or set bookmark status for a news item."""
    try:
        if request.bookmarked is not None:
            await asyncio.to_thread(
                news_service.set_bookmark,
                user_id=current_user.id,
                news_id=news_id,
                bookmarked=request.bookmarked,
                news_title=request.news_title,
                news_source=request.news_source,
                news_url=request.news_url,
                news_category=request.news_category
            )
            return {"bookmarked": request.bookmarked}
        else:
            new_state = await asyncio.to_thread(
                news_service.toggle_bookmark,
                user_id=current_user.id,
                news_id=news_id,
                news_title=request.news_title,
                news_source=request.news_source,
                news_url=request.news_url,
                news_category=request.news_category
            )
            return {"bookmarked": new_state}
    except Exception as e:
        print(f"Error toggling bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{news_id}/read")
async def mark_news_read(
    news_id: str,
    request: NewsReadRequest,
    current_user: User = Depends(get_current_user)
):
    """Mark a news item as read."""
    try:
        await asyncio.to_thread(
            news_service.mark_read,
            user_id=current_user.id,
            news_id=news_id,
            news_title=request.news_title,
            news_source=request.news_source,
            news_url=request.news_url,
            news_category=request.news_category
        )
        return {"success": True, "is_read": True}
    except Exception as e:
        print(f"Error marking news as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))
