"""
Recommendations endpoints.
"""
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.models.auth import User
from app.core.dependencies import get_current_user
from app.core.utils import sanitize_for_json

router = APIRouter(prefix="/api/recommend", tags=["Recommendations"])

# Thread pool for async generation
_recommendation_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="recommend_")


class RecommendationRequest(BaseModel):
    mode: str = "all"  # "short", "long", "all"
    force_refresh: bool = False


def _run_recommendation_task(task_id: str, mode: str, user_id: int, user_preferences: Optional[Dict] = None):
    """
    Background task worker for generating recommendations.
    Stores progress and results in cache.
    """
    from src.cache import cache_manager

    cache_key = f"recommend_task:{task_id}"

    try:
        # Update status to running
        cache_manager.set(cache_key, {
            "status": "running",
            "progress": "Initializing recommendation engine...",
            "started_at": datetime.now().isoformat(),
            "user_id": user_id,
            "mode": mode
        }, ttl=3600)  # 1 hour TTL

        from src.analysis.recommendation import RecommendationEngine
        from src.llm.client import get_llm_client
        from src.data_sources.web_search import WebSearch

        # Update progress
        cache_manager.set(cache_key, {
            "status": "running",
            "progress": "Screening stocks and funds...",
            "started_at": datetime.now().isoformat(),
            "user_id": user_id,
            "mode": mode
        }, ttl=3600)

        llm_client = get_llm_client()
        web_search = WebSearch()
        engine = RecommendationEngine(
            llm_client=llm_client,
            web_search=web_search,
            cache_manager=cache_manager
        )

        # Generate recommendations
        results = engine.generate_recommendations(
            mode=mode,
            use_llm=True,
            user_preferences=user_preferences
        )

        # Sanitize results
        results = sanitize_for_json(results)

        # Cache the recommendation results
        prefs_hash = "personalized" if user_preferences else "default"
        result_cache_key = f"recommendations:{user_id}:{mode}:{prefs_hash}"
        cache_manager.set(result_cache_key, results, ttl=14400)  # 4 hours

        # Save to database
        from src.storage.db import save_recommendation_report
        save_recommendation_report({
            "mode": mode,
            "recommendations_json": results,
            "market_context": results.get("metadata", {})
        }, user_id=user_id)

        # Update task status to completed
        cache_manager.set(cache_key, {
            "status": "completed",
            "progress": "Recommendations generated successfully!",
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "user_id": user_id,
            "mode": mode,
            "result": results
        }, ttl=3600)

        print(f"[Task {task_id}] Recommendation generation completed for user {user_id}")

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()

        # Update task status to failed
        cache_manager.set(cache_key, {
            "status": "failed",
            "progress": f"Error: {error_msg}",
            "error": error_msg,
            "user_id": user_id,
            "mode": mode
        }, ttl=3600)

        print(f"[Task {task_id}] Recommendation generation failed: {error_msg}")


@router.post("/generate")
async def generate_recommendations_endpoint(
    request: RecommendationRequest = None,
    current_user: User = Depends(get_current_user)
):
    """
    Generate AI investment recommendations (runs in background).

    - mode: "short" (7+ days), "long" (3+ months), or "all"
    - force_refresh: Force regenerate even if cached

    Returns a task_id immediately. Use GET /api/recommend/task/{task_id} to poll status.
    """
    mode = request.mode if request else "all"
    force_refresh = request.force_refresh if request else False

    if mode not in ["short", "long", "all"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'short', 'long', or 'all'.")

    try:
        from src.cache import cache_manager
        from src.storage.db import get_user_preferences

        # Load user preferences (if configured)
        user_preferences = None
        try:
            prefs_data = get_user_preferences(current_user.id)
            if prefs_data and prefs_data.get('preferences'):
                user_preferences = prefs_data.get('preferences')
                print(f"Loaded personalized preferences for user {current_user.id}")
        except Exception as e:
            print(f"No user preferences found: {e}")

        # Check cache first (unless force refresh)
        prefs_hash = "personalized" if user_preferences else "default"
        if not force_refresh:
            cache_key = f"recommendations:{current_user.id}:{mode}:{prefs_hash}"
            cached = cache_manager.get(cache_key)
            if cached:
                print(f"Returning cached recommendations for user {current_user.id}")
                return {
                    "status": "completed",
                    "cached": True,
                    "result": sanitize_for_json(cached)
                }

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Initialize task status
        cache_manager.set(f"recommend_task:{task_id}", {
            "status": "pending",
            "progress": "Task queued...",
            "user_id": current_user.id,
            "mode": mode
        }, ttl=3600)

        # Submit to background executor
        _recommendation_executor.submit(
            _run_recommendation_task,
            task_id,
            mode,
            current_user.id,
            user_preferences
        )

        print(f"Started recommendation task {task_id} for user {current_user.id}")

        return {
            "status": "started",
            "task_id": task_id,
            "message": "Recommendation generation started. Poll /api/recommend/task/{task_id} for status."
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}")
async def get_recommendation_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of a recommendation generation task.

    Returns:
    - status: "pending", "running", "completed", or "failed"
    - progress: Human-readable progress message
    - result: Full recommendation data (only when status is "completed")
    """
    try:
        from src.cache import cache_manager

        cache_key = f"recommend_task:{task_id}"
        task_data = cache_manager.get(cache_key)

        if not task_data:
            raise HTTPException(status_code=404, detail="Task not found or expired")

        # Security check: ensure task belongs to current user
        if task_data.get("user_id") != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Return task status
        response = {
            "task_id": task_id,
            "status": task_data.get("status"),
            "progress": task_data.get("progress"),
            "mode": task_data.get("mode")
        }

        if task_data.get("status") == "completed":
            response["result"] = task_data.get("result")
            response["completed_at"] = task_data.get("completed_at")

        if task_data.get("status") == "failed":
            response["error"] = task_data.get("error")

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stocks/short")
async def get_short_term_stock_recommendations(
    limit: int = 10,
    min_score: int = 60,
    current_user: User = Depends(get_current_user)
):
    """Get short-term stock recommendations (7+ days)."""
    try:
        from src.storage.db import get_latest_recommendation_report

        report = get_latest_recommendation_report(user_id=current_user.id, mode="short")
        if not report and (report := get_latest_recommendation_report(user_id=current_user.id, mode="all")):
            pass

        if not report:
            return {"recommendations": [], "message": "No recommendations available. Please generate first."}

        data = report.get("recommendations_json", {})
        short_term = data.get("short_term", {})
        stocks = short_term.get("short_term_stocks", []) if isinstance(short_term, dict) else []

        # Filter by min_score
        filtered = [s for s in stocks if s.get("recommendation_score", 0) >= min_score]

        return {
            "recommendations": filtered[:limit],
            "market_view": short_term.get("market_view", ""),
            "generated_at": report.get("generated_at"),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"recommendations": [], "error": str(e)}


@router.get("/stocks/long")
async def get_long_term_stock_recommendations(
    limit: int = 10,
    min_score: int = 60,
    current_user: User = Depends(get_current_user)
):
    """Get long-term stock recommendations (3+ months)."""
    try:
        from src.storage.db import get_latest_recommendation_report

        report = get_latest_recommendation_report(user_id=current_user.id, mode="long")
        if not report and (report := get_latest_recommendation_report(user_id=current_user.id, mode="all")):
            pass

        if not report:
            return {"recommendations": [], "message": "No recommendations available. Please generate first."}

        data = report.get("recommendations_json", {})
        long_term = data.get("long_term", {})
        stocks = long_term.get("long_term_stocks", []) if isinstance(long_term, dict) else []

        filtered = [s for s in stocks if s.get("recommendation_score", 0) >= min_score]

        return {
            "recommendations": filtered[:limit],
            "macro_view": long_term.get("macro_view", ""),
            "generated_at": report.get("generated_at"),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"recommendations": [], "error": str(e)}


@router.get("/funds/short")
async def get_short_term_fund_recommendations(
    limit: int = 5,
    min_score: int = 60,
    current_user: User = Depends(get_current_user)
):
    """Get short-term fund recommendations (7+ days)."""
    try:
        from src.storage.db import get_latest_recommendation_report

        report = get_latest_recommendation_report(user_id=current_user.id, mode="short")
        if not report and (report := get_latest_recommendation_report(user_id=current_user.id, mode="all")):
            pass

        if not report:
            return {"recommendations": [], "message": "No recommendations available. Please generate first."}

        data = report.get("recommendations_json", {})
        short_term = data.get("short_term", {})
        funds = short_term.get("short_term_funds", []) if isinstance(short_term, dict) else []

        filtered = [f for f in funds if f.get("recommendation_score", 0) >= min_score]

        return {
            "recommendations": filtered[:limit],
            "generated_at": report.get("generated_at"),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"recommendations": [], "error": str(e)}


@router.get("/funds/long")
async def get_long_term_fund_recommendations(
    limit: int = 5,
    min_score: int = 60,
    current_user: User = Depends(get_current_user)
):
    """Get long-term fund recommendations (3+ months)."""
    try:
        from src.storage.db import get_latest_recommendation_report

        report = get_latest_recommendation_report(user_id=current_user.id, mode="long")
        if not report and (report := get_latest_recommendation_report(user_id=current_user.id, mode="all")):
            pass

        if not report:
            return {"recommendations": [], "message": "No recommendations available. Please generate first."}

        data = report.get("recommendations_json", {})
        long_term = data.get("long_term", {})
        funds = long_term.get("long_term_funds", []) if isinstance(long_term, dict) else []

        filtered = [f for f in funds if f.get("recommendation_score", 0) >= min_score]

        return {
            "recommendations": filtered[:limit],
            "generated_at": report.get("generated_at"),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"recommendations": [], "error": str(e)}


@router.get("/latest")
async def get_latest_recommendations(
    current_user: User = Depends(get_current_user)
):
    """Get the latest recommendation report."""
    try:
        from src.storage.db import get_latest_recommendation_report

        report = get_latest_recommendation_report(user_id=current_user.id)

        if not report:
            return {
                "available": False,
                "message": "No recommendations available. Please generate first using POST /api/recommend/generate"
            }

        return {
            "available": True,
            "data": report.get("recommendations_json", {}),
            "generated_at": report.get("generated_at"),
            "mode": report.get("mode"),
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_recommendation_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    """Get historical recommendation reports."""
    try:
        from src.storage.db import get_recommendation_reports

        reports = get_recommendation_reports(user_id=current_user.id, limit=limit)

        # Return summaries without full content
        summaries = []
        for r in reports:
            data = r.get("recommendations_json", {})
            summaries.append({
                "id": r.get("id"),
                "mode": r.get("mode"),
                "generated_at": r.get("generated_at"),
                "short_term_count": len(data.get("short_term", {}).get("short_term_stocks", [])) if data.get("short_term") else 0,
                "long_term_count": len(data.get("long_term", {}).get("long_term_stocks", [])) if data.get("long_term") else 0,
            })

        return {"reports": summaries}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
