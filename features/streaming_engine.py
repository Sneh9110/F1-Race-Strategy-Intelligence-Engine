"""
Streaming feature engine for real-time computation.

Provides low-latency feature computation for live race data
with caching, state management, and fault tolerance.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import redis
import pandas as pd

from features.base import BaseFeature
from features.registry import FeatureRegistry
from features.store import FeatureStore
from config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StreamingFeatureEngine:
    """
    Streaming feature engine for real-time computation.
    
    Maintains state for incremental feature computation with
    Redis caching for sub-200ms latency.
    """
    
    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
        registry: Optional[FeatureRegistry] = None,
        redis_client: Optional[redis.Redis] = None,
        cache_ttl: int = 60,
        state_retention_hours: int = 24
    ):
        """
        Initialize streaming engine.
        
        Args:
            feature_store: FeatureStore instance
            registry: FeatureRegistry instance
            redis_client: Redis client for caching
            cache_ttl: Cache TTL in seconds
            state_retention_hours: How long to retain state
        """
        self.feature_store = feature_store or FeatureStore()
        self.registry = registry or FeatureRegistry()
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.state_retention = timedelta(hours=state_retention_hours)
        
        # In-memory state for active sessions
        self.session_states = {}
        
        logger.info(
            "Initialized StreamingFeatureEngine",
            extra={
                'cache_ttl': cache_ttl,
                'state_retention_hours': state_retention_hours
            }
        )
    
    async def compute_realtime(
        self,
        session_id: str,
        lap_data: Any,
        feature_names: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute features in real-time for new lap data.
        
        Args:
            session_id: Session identifier
            lap_data: New lap data
            feature_names: Features to compute
            
        Returns:
            Dictionary of computed features
        """
        start_time = datetime.utcnow()
        
        # Check cache first
        cached_features = {}
        features_to_compute = []
        
        for feature_name in feature_names:
            cached = await self._get_from_cache(session_id, feature_name)
            if cached is not None:
                cached_features[feature_name] = cached
            else:
                features_to_compute.append(feature_name)
        
        # Compute uncached features
        computed_features = {}
        if features_to_compute:
            execution_order = self.registry.compute_execution_order(features_to_compute)
            
            for feature_name in execution_order:
                try:
                    feature_df = await self._compute_feature_async(
                        session_id, feature_name, lap_data
                    )
                    
                    if feature_df is not None:
                        computed_features[feature_name] = feature_df
                        
                        # Cache result
                        await self._cache_feature(
                            session_id, feature_name, feature_df
                        )
                
                except Exception as e:
                    logger.error(
                        f"Failed to compute feature '{feature_name}': {e}",
                        extra={'session_id': session_id}
                    )
        
        # Combine cached and computed
        all_features = {**cached_features, **computed_features}
        
        # Log performance
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.debug(
            f"Real-time computation completed in {duration_ms:.1f}ms",
            extra={
                'session_id': session_id,
                'duration_ms': duration_ms,
                'cache_hits': len(cached_features),
                'computed': len(computed_features)
            }
        )
        
        return all_features
    
    async def _compute_feature_async(
        self,
        session_id: str,
        feature_name: str,
        data: Any
    ) -> Optional[pd.DataFrame]:
        """Compute single feature asynchronously."""
        feature_calculator = self.registry.get_feature_instance(feature_name)
        
        if feature_calculator is None:
            return None
        
        # Get session state
        state = self.session_states.get(session_id, {})
        
        # Compute (run in thread pool for CPU-bound work)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            feature_calculator.compute,
            data
        )
        
        if result.success:
            # Update state
            self._update_state(session_id, feature_name, result.features)
            return result.features
        else:
            logger.warning(
                f"Feature computation failed: {result.errors}",
                extra={'feature_name': feature_name, 'session_id': session_id}
            )
            return None
    
    def _update_state(
        self,
        session_id: str,
        feature_name: str,
        features: pd.DataFrame
    ) -> None:
        """Update streaming state for session."""
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                'last_update': datetime.utcnow(),
                'features': {}
            }
        
        self.session_states[session_id]['features'][feature_name] = features
        self.session_states[session_id]['last_update'] = datetime.utcnow()
    
    async def _get_from_cache(
        self,
        session_id: str,
        feature_name: str
    ) -> Optional[pd.DataFrame]:
        """Retrieve feature from cache."""
        if self.redis_client is None:
            return None
        
        try:
            key = f"feature:stream:{session_id}:{feature_name}"
            cached = self.redis_client.get(key)
            
            if cached:
                return pd.read_json(cached, orient='records')
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_feature(
        self,
        session_id: str,
        feature_name: str,
        features: pd.DataFrame
    ) -> None:
        """Cache feature for fast retrieval."""
        if self.redis_client is None:
            return
        
        try:
            key = f"feature:stream:{session_id}:{feature_name}"
            value = features.to_json(orient='records')
            self.redis_client.setex(key, self.cache_ttl, value)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    async def get_features(
        self,
        session_id: str,
        feature_names: List[str],
        max_age_seconds: int = 60
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Retrieve cached features.
        
        Args:
            session_id: Session identifier
            feature_names: Features to retrieve
            max_age_seconds: Maximum age of cached data
            
        Returns:
            Dictionary of features (None if not found/stale)
        """
        features = {}
        
        for feature_name in feature_names:
            cached = await self._get_from_cache(session_id, feature_name)
            
            # Check freshness
            if cached is not None:
                state = self.session_states.get(session_id, {})
                last_update = state.get('last_update')
                
                if last_update:
                    age = (datetime.utcnow() - last_update).total_seconds()
                    if age > max_age_seconds:
                        cached = None  # Too stale
            
            features[feature_name] = cached
        
        return features
    
    def cleanup_old_states(self) -> int:
        """
        Clean up old session states.
        
        Returns:
            Number of states removed
        """
        cutoff = datetime.utcnow() - self.state_retention
        removed = 0
        
        for session_id in list(self.session_states.keys()):
            state = self.session_states[session_id]
            if state['last_update'] < cutoff:
                del self.session_states[session_id]
                removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old session states")
        
        return removed
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current state for session."""
        return self.session_states.get(session_id)
    
    def clear_session_state(self, session_id: str) -> bool:
        """Clear state for session."""
        if session_id in self.session_states:
            del self.session_states[session_id]
            return True
        return False
