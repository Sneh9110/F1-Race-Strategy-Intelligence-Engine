"""
Lap Time Prediction Inference API

Production inference with:
- Redis caching for <200ms latency
- Circuit breaker for fault tolerance
- Performance monitoring
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import json
import time
from datetime import datetime, timedelta

import redis
from pydantic import ValidationError

from .base import BaseLapTimeModel, PredictionInput, PredictionOutput
from .fallback import FallbackHeuristics
from .registry import ModelRegistry
from config.settings import Settings

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing, use fallback
    - HALF_OPEN: Testing recovery
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successful calls needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open
        """
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if (
                self.last_failure_time and
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
            ):
                self.state = "HALF_OPEN"
                self.success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                logger.info("Circuit breaker closed after recovery")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
            logger.warning("Circuit breaker reopened after failed recovery attempt")
        elif self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class LapTimePredictor:
    """
    Production inference API for lap time prediction.
    
    Features:
    - Redis caching (3600s TTL)
    - Circuit breaker protection
    - Fallback to heuristics
    - Performance monitoring
    - Model hot-swapping
    
    Attributes:
        model: Loaded prediction model
        cache: Redis client for caching
        fallback: Fallback heuristics
        circuit_breaker: Circuit breaker for fault tolerance
    """
    
    def __init__(
        self,
        model_version: str = "latest",
        registry_path: Optional[Path] = None,
        use_cache: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        """
        Initialize prediction API.
        
        Args:
            model_version: Version to load ('latest', 'production', or specific version)
            registry_path: Path to model registry
            use_cache: Whether to use Redis caching
            redis_host: Redis server host
            redis_port: Redis server port
        """
        self.registry = ModelRegistry(registry_path)
        self.fallback = FallbackHeuristics()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        
        # Load model
        self.model = self._load_model(model_version)
        
        # Setup cache
        self.use_cache = use_cache
        if use_cache:
            try:
                self.cache = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                self.cache.ping()
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Proceeding without cache.")
                self.use_cache = False
        
        # Performance tracking
        self.latency_stats = {
            'p50': [],
            'p95': [],
            'p99': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_uses': 0,
        }
        self.cache_ttl = 3600  # 1 hour
    
    def predict(
        self,
        input_data: PredictionInput,
        use_fallback_on_error: bool = True
    ) -> PredictionOutput:
        """
        Predict lap time with caching and fault tolerance.
        
        Args:
            input_data: Prediction input
            use_fallback_on_error: Whether to use fallback on model failure
            
        Returns:
            Prediction output
        """
        start_time = time.time()
        
        # Check cache
        if self.use_cache:
            cache_key = self._generate_cache_key(input_data)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.latency_stats['cache_hits'] += 1
                latency = (time.time() - start_time) * 1000
                self._track_latency(latency)
                return cached_result
            self.latency_stats['cache_misses'] += 1
        
        # Predict with circuit breaker
        try:
            prediction = self.circuit_breaker.call(
                self.model.predict,
                input_data
            )
            
            # Cache result
            if self.use_cache:
                self._store_in_cache(cache_key, prediction)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
            if use_fallback_on_error:
                logger.info("Using fallback heuristics")
                prediction = self.fallback.predict(input_data)
                self.latency_stats['fallback_uses'] += 1
            else:
                raise e
        
        # Track latency
        latency = (time.time() - start_time) * 1000
        self._track_latency(latency)
        
        return prediction
    
    def predict_batch(
        self,
        inputs: List[PredictionInput],
        use_fallback_on_error: bool = True
    ) -> List[PredictionOutput]:
        """
        Predict lap times for multiple inputs.
        
        Args:
            inputs: List of prediction inputs
            use_fallback_on_error: Whether to use fallback on model failure
            
        Returns:
            List of prediction outputs
        """
        start_time = time.time()
        
        # Try batch prediction
        try:
            predictions = self.circuit_breaker.call(
                self.model.predict_batch,
                inputs
            )
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            
            if use_fallback_on_error:
                logger.info("Using fallback heuristics for batch")
                predictions = [self.fallback.predict(inp) for inp in inputs]
                self.latency_stats['fallback_uses'] += len(inputs)
            else:
                raise e
        
        # Track latency
        latency = (time.time() - start_time) * 1000
        self._track_latency(latency)
        
        return predictions
    
    def reload_model(self, model_version: str = "latest") -> None:
        """
        Hot-swap to new model version.
        
        Args:
            model_version: Version to load
        """
        logger.info(f"Reloading model to version: {model_version}")
        self.model = self._load_model(model_version)
        
        # Clear cache to avoid stale predictions
        if self.use_cache:
            try:
                self.cache.flushdb()
                logger.info("Cache cleared after model reload")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with latency and cache metrics
        """
        total_requests = self.latency_stats['cache_hits'] + self.latency_stats['cache_misses']
        cache_hit_rate = (
            self.latency_stats['cache_hits'] / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            'total_requests': total_requests,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.latency_stats['cache_hits'],
            'cache_misses': self.latency_stats['cache_misses'],
            'fallback_uses': self.latency_stats['fallback_uses'],
            'latency_p50_ms': np.percentile(self.latency_stats['p50'], 50) if self.latency_stats['p50'] else 0,
            'latency_p95_ms': np.percentile(self.latency_stats['p95'], 95) if self.latency_stats['p95'] else 0,
            'latency_p99_ms': np.percentile(self.latency_stats['p99'], 99) if self.latency_stats['p99'] else 0,
            'circuit_breaker_state': self.circuit_breaker.state,
        }
    
    def _load_model(self, version: str) -> BaseLapTimeModel:
        """
        Load model from registry.
        
        Args:
            version: Model version
            
        Returns:
            Loaded model
        """
        model = self.registry.load_model(version)
        logger.info(f"Loaded model version: {version}")
        return model
    
    def _generate_cache_key(self, input_data: PredictionInput) -> str:
        """
        Generate cache key from input data.
        
        Args:
            input_data: Prediction input
            
        Returns:
            MD5 hash of input
        """
        # Convert input to canonical JSON
        input_dict = input_data.dict()
        input_json = json.dumps(input_dict, sort_keys=True)
        
        # Generate MD5 hash
        hash_obj = hashlib.md5(input_json.encode())
        return f"lap_time:{hash_obj.hexdigest()}"
    
    def _get_from_cache(self, key: str) -> Optional[PredictionOutput]:
        """
        Retrieve prediction from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached prediction or None
        """
        try:
            cached_json = self.cache.get(key)
            if cached_json:
                cached_dict = json.loads(cached_json)
                return PredictionOutput(**cached_dict)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    def _store_in_cache(self, key: str, prediction: PredictionOutput) -> None:
        """
        Store prediction in cache.
        
        Args:
            key: Cache key
            prediction: Prediction to cache
        """
        try:
            prediction_json = prediction.json()
            self.cache.setex(key, self.cache_ttl, prediction_json)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _track_latency(self, latency_ms: float) -> None:
        """
        Track latency for performance monitoring.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self.latency_stats['p50'].append(latency_ms)
        self.latency_stats['p95'].append(latency_ms)
        self.latency_stats['p99'].append(latency_ms)
        
        # Keep only last 1000 samples
        for key in ['p50', 'p95', 'p99']:
            if len(self.latency_stats[key]) > 1000:
                self.latency_stats[key] = self.latency_stats[key][-1000:]


# Import numpy for percentile calculations
import numpy as np
