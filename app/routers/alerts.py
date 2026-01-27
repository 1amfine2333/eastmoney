"""
Alerts endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends

from app.models.auth import User
from app.core.dependencies import get_current_user
from src.storage.db import (
    get_portfolio_alerts, get_unread_alert_count,
    mark_alert_read, dismiss_alert
)

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("")
async def get_all_user_alerts(
    unread_only: bool = False,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get all alerts for the current user across all portfolios."""
    try:
        alerts = get_portfolio_alerts(None, current_user.id, unread_only, limit)
        unread_count = get_unread_alert_count(current_user.id)

        return {
            "alerts": alerts,
            "unread_count": unread_count,
        }
    except Exception as e:
        print(f"Error getting user alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/read")
async def mark_alert_as_read(alert_id: int, current_user: User = Depends(get_current_user)):
    """Mark an alert as read."""
    try:
        success = mark_alert_read(alert_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {"message": "Alert marked as read"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error marking alert as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/dismiss")
async def dismiss_alert_api(alert_id: int, current_user: User = Depends(get_current_user)):
    """Dismiss an alert."""
    try:
        success = dismiss_alert(alert_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {"message": "Alert dismissed"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error dismissing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))
