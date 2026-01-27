"""
Health check endpoint.
"""
from datetime import datetime
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
