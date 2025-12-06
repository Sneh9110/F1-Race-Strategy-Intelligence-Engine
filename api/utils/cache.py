"""Redis caching utilities."""

import json
import hashlib
from typing import Any, Optional
import redis.asyncio as redis

from api.config import api_config


class CacheManager:
    """Manager for Redis caching operations."""
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
    
    @staticmethod
    def generate_cache_key(prefix: str, data: dict) -> str:
        """
        Generate cache key from data.
        
        Args:
            prefix: Cache key prefix
            data: Data to hash for key
            
        Returns:
            Cache key string
        """
        # Sort dict for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        hash_value = hashlib.md5(sorted_data.encode()).hexdigest()
        return f"{prefix}:{hash_value}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception:
            pass
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            serialized = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        try:
            await self.redis.delete(key)
            return True
        except Exception:
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "predictions:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception:
            return 0
    
    async def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            return await self.redis.ttl(key)
        except Exception:
            return -2
