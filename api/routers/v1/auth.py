"""Authentication endpoints."""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    User,
    TokenResponse,
)
from api.config import api_config

router = APIRouter()


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


@router.post("/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Generate JWT access token.
    
    - **username**: Username
    - **password**: Password
    
    Returns JWT token with expiration time.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=api_config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=api_config.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(current_user: User = Depends(get_current_user)):
    """
    Refresh JWT access token.
    
    Requires valid JWT token in Authorization header.
    Returns new token with extended expiration.
    """
    access_token_expires = timedelta(minutes=api_config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user.username, "role": current_user.role},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=api_config.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information.
    
    Requires valid JWT token in Authorization header.
    Returns user profile information.
    """
    return current_user
