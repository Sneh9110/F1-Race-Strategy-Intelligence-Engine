"""
Fast inference API for tire degradation predictions.

Optimized for <200ms latency with Redis caching and circuit breaker pattern.
"""

import time
import redis
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from circuitbreaker import circuit

from models.tire_degradation.base import (
    BaseDegradationModel,
    PredictionInput,
    PredictionOutput
)
from models.tire_degradation.registry import ModelRegistry
from models.tire_degradation.fallback import FallbackHeuristics
from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class DegradationPredictor:
    """
    High-performance inference API for tire degradation.
    
    Features:
    - <200ms prediction latency
    - Redis caching for repeated queries
    - Circuit breaker for fault tolerance
    - Batch prediction support
    - Automatic fallback to heuristics
    """
    
    def __init__(
        self,
        model_version: str = 'latest',
        use_cache: bool = True,
        cache_ttl: int = 3600
    ):
        """
        Initialize predictor.
        
        Args:
            model_version: Model version to load ('latest' or specific version)
            use_cache: Enable Redis caching
            cache_ttl: Cache TTL in seconds
        """
        self.registry = ModelRegistry()
        self.fallback = FallbackHeuristics()
        
        # Load model
        self.model = self.registry.load_model(model_version)
        self.model_version = model_version
        
        # Cache setup
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.cache_client = None
        
        if use_cache:
            self._init_cache()
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_used': 0,
            'avg_latency_ms': 0.0,
            'latencies': []
        }
    
    def _init_cache(self) -> None:
        """Initialize Redis cache client."""
        try:
            self.cache_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=getattr(settings, 'REDIS_DB_PREDICTIONS', 2),
                decode_responses=True
            )
            # Test connection
            self.cache_client.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis cache unavailable: {e}. Disabling caching.")
            self.use_cache = False
            self.cache_client = None
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    def predict(
        self,
        input_data: Union[PredictionInput, Dict[str, Any]],
        use_cache: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Make single prediction with caching and circuit breaker.
        
        Args:
            input_data: Prediction input (PredictionInput or dict)
            use_cache: Override cache setting for this prediction
        
        Returns:
            Prediction output
        """
        start_time = time.time()
        
        # Convert dict to PredictionInput
        if isinstance(input_data, dict):
            input_data = PredictionInput(**input_data)
        
        # Check cache
        use_cache_flag = use_cache if use_cache is not None else self.use_cache
        if use_cache_flag:
            cached_result = self._get_from_cache(input_data)
            if cached_result:
                self.stats['cache_hits'] += 1
                self._update_latency(start_time)
                return cached_result
            self.stats['cache_misses'] += 1
        
        # Make prediction
        try:
            result = self.model.predict(input_data)
            
            # Cache result
            if use_cache_flag:
                self._save_to_cache(input_data, result)
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}. Using fallback.")
            result = self._fallback_predict(input_data)
            self.stats['fallback_used'] += 1
        
        # Update stats
        self.stats['total_predictions'] += 1
        self._update_latency(start_time)
        
        return result
    
    def predict_batch(
        self,
        inputs: List[Union[PredictionInput, Dict[str, Any]]],
        use_cache: Optional[bool] = None
    ) -> List[PredictionOutput]:
        """
        Make batch predictions efficiently.
        
        Args:
            inputs: List of prediction inputs
            use_cache: Override cache setting
        
        Returns:
            List of prediction outputs
        """
        logger.info(f"Batch prediction: {len(inputs)} samples")
        
        results = []
        for input_data in inputs:
            result = self.predict(input_data, use_cache=use_cache)
            results.append(result)
        
        return results
    
    def _fallback_predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Use fallback heuristics when model fails."""
        logger.info("Using fallback heuristics for prediction")
        return self.fallback.predict(input_data)
    
    def _get_from_cache(self, input_data: PredictionInput) -> Optional[PredictionOutput]:
        """Get prediction from cache."""
        if not self.cache_client:
            return None
        
        cache_key = self._generate_cache_key(input_data)
        
        try:
            cached = self.cache_client.get(cache_key)
            if cached:
                data = json.loads(cached)
                return PredictionOutput(**data)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def _save_to_cache(
        self,
        input_data: PredictionInput,
        output: PredictionOutput
    ) -> None:
        """Save prediction to cache."""
        if not self.cache_client:
            return
        
        cache_key = self._generate_cache_key(input_data)
        
        try:
            # Serialize output
            data = output.dict()
            self.cache_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _generate_cache_key(self, input_data: PredictionInput) -> str:
        """
        Generate cache key from input data.
        
        Args:
            input_data: Prediction input
        
        Returns:
            Cache key string
        """
        # Create deterministic hash of input
        input_dict = input_data.dict()
        input_str = json.dumps(input_dict, sort_keys=True)
        hash_val = hashlib.md5(input_str.encode()).hexdigest()
        
        return f"tire_deg:v{self.model_version}:{hash_val}"
    
    def _update_latency(self, start_time: float) -> None:
        """Update latency statistics."""
        latency_ms = (time.time() - start_time) * 1000
        self.stats['latencies'].append(latency_ms)
        
        # Keep only last 100 latencies
        if len(self.stats['latencies']) > 100:
            self.stats['latencies'] = self.stats['latencies'][-100:]
        
        # Update average
        self.stats['avg_latency_ms'] = sum(self.stats['latencies']) / len(self.stats['latencies'])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        stats = self.stats.copy()
        
        # Calculate cache hit rate
        total_queries = stats['cache_hits'] + stats['cache_misses']
        if total_queries > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_queries
        else:
            stats['cache_hit_rate'] = 0.0
        
        # Latency percentiles
        if stats['latencies']:
            import numpy as np
            stats['p50_latency_ms'] = np.percentile(stats['latencies'], 50)
            stats['p95_latency_ms'] = np.percentile(stats['latencies'], 95)
            stats['p99_latency_ms'] = np.percentile(stats['latencies'], 99)
        
        # Remove raw latencies from output
        del stats['latencies']
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all cached predictions."""
        if not self.cache_client:
            return
        
        try:
            # Clear predictions for this model version
            pattern = f"tire_deg:v{self.model_version}:*"
            keys = self.cache_client.keys(pattern)
            if keys:
                self.cache_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cached predictions")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def reload_model(self, version: str = 'latest') -> None:
        """
        Reload model to new version.
        
        Args:
            version: Model version to load
        """
        logger.info(f"Reloading model to version: {version}")
        
        self.model = self.registry.load_model(version)
        self.model_version = version
        
        # Clear cache when model changes
        self.clear_cache()
        
        # Reset stats
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_used': 0,
            'avg_latency_ms': 0.0,
            'latencies': []
        }
        
        logger.info("Model reloaded successfully")
