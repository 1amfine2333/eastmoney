"""
FastAPI dependencies for authentication and common request handling.
"""
import os
from fastapi import Depends
from src.auth import get_current_user as auth_get_current_user, User
from .config import REPORT_DIR

# Re-export get_current_user from src.auth for use in routers
get_current_user = auth_get_current_user


def get_user_report_dir(user_id: int) -> str:
    """
    Get the report directory for a specific user.
    Creates the directory if it doesn't exist.
    """
    user_dir = os.path.join(REPORT_DIR, str(user_id))
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir


async def get_current_user_report_dir(current_user: User = Depends(get_current_user)) -> str:
    """
    FastAPI dependency that returns the current user's report directory.
    """
    return get_user_report_dir(current_user.id)
