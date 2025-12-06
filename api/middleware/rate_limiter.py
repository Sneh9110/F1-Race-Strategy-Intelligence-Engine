"""Redis-based rate limiting middleware with sliding window algorithm."""

import time
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis

from api.config import api_config
from api.schemas.common import ErrorResponse, ErrorCode


class RateLimiter:
    """Redis-based rate limiter using sliding window algorithm."""
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize rate limiter.
        
        Args:
            redis_client: Redis client for rate limit storage
        """
        self.redis = redis_client
        self.window_size = 60  # 60 seconds window
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int = 60
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Unique identifier for rate limit (e.g., "user:123:predictions")
            limit: Maximum requests allowed in window
            window: Time window in seconds (default: 60)
            
        Returns:
            Tuple of (allowed, remaining, reset_timestamp)
        """
        now = time.time()
        window_start = now - window
        
        # Remove old entries outside the window
        await self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count requests in current window
        current_count = await self.redis.zcard(key)
        
        if current_count < limit:
            # Add current request
            await self.redis.zadd(key, {str(now): now})
            await self.redis.expire(key, window)
            remaining = limit - current_count - 1
            return True, remaining, int(now + window)
        
        # Get oldest request timestamp to calculate reset time
        oldest = await self.redis.zrange(key, 0, 0, withscores=True)
        if oldest:
            reset_time = int(oldest[0][1] + window)
        else:
            reset_time = int(now + window)
        
        return False, 0, reset_time
    
    async def get_rate_limit_key(self, request: Request, category: str) -> str:
        """
        Generate rate limit key from request.
        
        Args:
            request: FastAPI request
            category: Rate limit category (predictions, simulations, admin)
            
        Returns:
            Redis key for rate limiting
        """
        # Try to get user from request state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        
        if user_id:
            return f"rate_limit:{user_id}:{category}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"rate_limit:ip:{client_ip}:{category}"


async def rate_limit_dependency(
    request: Request,
    category: str = "default",
    limit: Optional[int] = None
) -> None:
    """
    FastAPI dependency for rate limiting.
    
    Args:
        request: FastAPI request
        category: Rate limit category
        limit: Override default limit
        
    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    from api.dependencies import get_redis_client
    
    redis_client = await get_redis_client()
    limiter = RateLimiter(redis_client)
    
    # Get limit from config if not specified
    if limit is None:
        limit_map = {
            "predictions": api_config.RATE_LIMIT_PREDICTIONS,
            "simulations": api_config.RATE_LIMIT_SIMULATIONS,
            "admin": api_config.RATE_LIMIT_ADMIN,
            "default": 60
        }
        limit = limit_map.get(category, 60)
    
    # Generate rate limit key
    rate_key = await limiter.get_rate_limit_key(request, category)
    
    # Check rate limit
    allowed, remaining, reset_time = await limiter.check_rate_limit(rate_key, limit)
    
    # Add rate limit headers to response
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_reset = reset_time
    request.state.rate_limit_limit = limit
    
    if not allowed:
        retry_after = reset_time - int(time.time())
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_time),
                "Retry-After": str(retry_after)
            }
        )


def add_rate_limit_headers(response: JSONResponse, request: Request) -> JSONResponse:
    """
    Add rate limit headers to response.
    
    Args:
        response: FastAPI response
        request: FastAPI request
        
    Returns:
        Response with rate limit headers
    """
    if hasattr(request.state, "rate_limit_limit"):
        response.headers["X-RateLimit-Limit"] = str(request.state.rate_limit_limit)
        response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)
        response.headers["X-RateLimit-Reset"] = str(request.state.rate_limit_reset)
    
    return response
