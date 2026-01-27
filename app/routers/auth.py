"""
Authentication endpoints.
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from app.models.auth import Token, UserCreate, User
from src.auth import (
    create_access_token, get_password_hash, verify_password,
    get_current_user, create_user, get_user_by_username
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=Token)
async def register(user: UserCreate):
    """Register a new user."""
    existing = get_user_by_username(user.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_pwd = get_password_hash(user.password)
    try:
        user_id = create_user({
            "username": user.username,
            "email": user.email,
            "hashed_password": hashed_pwd,
            "provider": "local"
        })

        # Auto login
        access_token = create_access_token(
            data={"sub": user.username, "id": user_id},
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token."""
    user_dict = get_user_by_username(form_data.username)
    if not user_dict or not verify_password(form_data.password, user_dict['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user_dict['username'], "id": user_dict['id']}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return current_user
