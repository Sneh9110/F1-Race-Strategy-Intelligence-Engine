"""Authentication and authorization for the API."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from pydantic import BaseModel
from passlib.context import CryptContext

from api.config import api_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Security schemes
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(BaseModel):
    """User model for authentication."""
    username: str
    email: str
    role: str = "user"  # user, admin
    permissions: list[str] = []
    disabled: bool = False


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    role: Optional[str] = None


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# Simple in-memory user store (production would use database)
USERS_DB: Dict[str, Dict[str, Any]] = {
    "admin": {
        "username": "admin",
        "email": "admin@f1strategy.com",
        "hashed_password": pwd_context.hash("admin123"),  # Change in production!
        "role": "admin",
        "permissions": ["read", "write", "admin"],
        "disabled": False,
    },
    "user": {
        "username": "user",
        "email": "user@f1strategy.com",
        "hashed_password": pwd_context.hash("user123"),
        "role": "user",
        "permissions": ["read"],
        "disabled": False,
    },
}

# Simple API key store (production would use database with hashed keys)
API_KEYS_DB: Dict[str, Dict[str, Any]] = {
    "test-api-key-12345": {
        "name": "Test API Key",
        "user": "user",
        "permissions": ["read"],
    },
    "admin-api-key-67890": {
        "name": "Admin API Key",
        "user": "admin",
        "permissions": ["read", "write", "admin"],
    },
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password."""
    user_data = USERS_DB.get(username)
    if not user_data:
        return None
    if not verify_password(password, user_data["hashed_password"]):
        return None
    return User(**{k: v for k, v in user_data.items() if k != "hashed_password"})


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=api_config.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, api_config.SECRET_KEY, algorithm=api_config.ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, api_config.SECRET_KEY, algorithms=[api_config.ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return TokenData(username=username, role=role)
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> User:
    """Get current user from JWT token."""
    token = credentials.credentials
    token_data = verify_token(token)
    
    user_data = USERS_DB.get(token_data.username)
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    user = User(**{k: v for k, v in user_data.items() if k != "hashed_password"})
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    
    return user


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> User:
    """Verify API key and return associated user."""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    
    key_data = API_KEYS_DB.get(api_key)
    if key_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    username = key_data["user"]
    user_data = USERS_DB.get(username)
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found for API key",
        )
    
    return User(**{k: v for k, v in user_data.items() if k != "hashed_password"})


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security, auto_error=False),
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[User]:
    """Get current user from JWT or API key (optional)."""
    if credentials:
        try:
            return await get_current_user(credentials)
        except HTTPException:
            pass
    
    if api_key:
        try:
            return await verify_api_key(api_key)
        except HTTPException:
            pass
    
    return None


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security, auto_error=False),
    api_key: Optional[str] = Security(api_key_header)
) -> User:
    """Require authentication via JWT or API key."""
    if credentials:
        try:
            return await get_current_user(credentials)
        except HTTPException:
            pass
    
    if api_key:
        try:
            return await verify_api_key(api_key)
        except HTTPException:
            pass
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required (JWT or API key)",
    )


async def require_admin(current_user: User = Depends(require_auth)) -> User:
    """Require admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
