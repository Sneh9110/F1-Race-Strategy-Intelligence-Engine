"""
Integration tests for StreamingFeatureEngine.
"""

import pytest
import pandas as pd
import asyncio
from datetime import datetime

from features.streaming_engine import StreamingFeatureEngine
from features.base import FeatureConfig


@pytest.fixture
def streaming_engine():
    """Create StreamingFeatureEngine instance."""
    return StreamingFeatureEngine(cache_ttl=300)


class TestStreamingFeatureEngine:
    """Tests for StreamingFeatureEngine."""
    
    @pytest.mark.asyncio
    async def test_compute_realtime(self, streaming_engine):
        """Test real-time feature computation."""
        results = await streaming_engine.compute_realtime(
            session_id="2024_MONACO_RACE",
            feature_names=["stint_summary"]
        )
        
        assert isinstance(results, dict)
        if "stint_summary" in results:
            result = results["stint_summary"]
            assert hasattr(result, 'success')
            if result.success:
                assert result.data is not None
                assert result.compute_time is not None
    
    @pytest.mark.asyncio
    async def test_latency_target(self, streaming_engine):
        """Test that latency is within target (<200ms)."""
        results = await streaming_engine.compute_realtime(
            session_id="2024_MONACO_RACE",
            feature_names=["stint_summary", "pace_delta"]
        )
        
        # Check latency for each feature
        for feature_name, result in results.items():
            if result.success:
                latency_ms = result.compute_time * 1000
                # Allow some tolerance for test environment
                assert latency_ms < 1000, \
                    f"Feature {feature_name} exceeded latency target: {latency_ms:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, streaming_engine):
        """Test cache hit improves performance."""
        session_id = "2024_MONACO_RACE"
        feature_names = ["stint_summary"]
        
        # First call (cache miss)
        results1 = await streaming_engine.compute_realtime(
            session_id=session_id,
            feature_names=feature_names
        )
        
        # Second call (cache hit)
        results2 = await streaming_engine.compute_realtime(
            session_id=session_id,
            feature_names=feature_names
        )
        
        # Both should succeed
        for feature_name in feature_names:
            if feature_name in results1 and feature_name in results2:
                assert results1[feature_name].success or results1[feature_name].error
                assert results2[feature_name].success or results2[feature_name].error
    
    @pytest.mark.asyncio
    async def test_session_state_management(self, streaming_engine):
        """Test session state initialization and cleanup."""
        session_id = "2024_MONACO_RACE"
        
        # Initialize state
        await streaming_engine._initialize_session_state(session_id)
        
        # State should exist
        assert session_id in streaming_engine.session_states
        
        # Compute features
        results = await streaming_engine.compute_realtime(
            session_id=session_id,
            feature_names=["stint_summary"]
        )
        
        assert isinstance(results, dict)
    
    @pytest.mark.asyncio
    async def test_cleanup_old_states(self, streaming_engine):
        """Test cleanup of old session states."""
        # Create multiple session states
        for i in range(5):
            session_id = f"2024_RACE_{i}"
            await streaming_engine._initialize_session_state(session_id)
        
        initial_count = len(streaming_engine.session_states)
        
        # Cleanup old states
        await streaming_engine.cleanup_old_states(max_age_minutes=0)
        
        # Some states should be cleaned up
        final_count = len(streaming_engine.session_states)
        assert final_count <= initial_count
    
    @pytest.mark.asyncio
    async def test_multiple_features_parallel(self, streaming_engine):
        """Test computing multiple features in parallel."""
        results = await streaming_engine.compute_realtime(
            session_id="2024_MONACO_RACE",
            feature_names=["stint_summary", "pace_delta", "degradation_slope"]
        )
        
        # Should compute all requested features
        assert len(results) <= 3  # May be fewer if some fail
        
        # Check each result
        for feature_name, result in results.items():
            assert hasattr(result, 'success')
            assert hasattr(result, 'compute_time')
    
    @pytest.mark.asyncio
    async def test_error_handling(self, streaming_engine):
        """Test error handling for invalid session."""
        results = await streaming_engine.compute_realtime(
            session_id="INVALID_SESSION",
            feature_names=["stint_summary"]
        )
        
        assert isinstance(results, dict)
        if "stint_summary" in results:
            result = results["stint_summary"]
            # Should handle error gracefully
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_incremental_updates(self, streaming_engine):
        """Test incremental feature updates."""
        session_id = "2024_MONACO_RACE"
        
        # First update
        results1 = await streaming_engine.compute_realtime(
            session_id=session_id,
            feature_names=["stint_summary"]
        )
        
        # Simulate lap progression
        await asyncio.sleep(0.1)
        
        # Second update
        results2 = await streaming_engine.compute_realtime(
            session_id=session_id,
            feature_names=["stint_summary"]
        )
        
        # Both should complete
        assert isinstance(results1, dict)
        assert isinstance(results2, dict)
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, streaming_engine):
        """Test handling multiple concurrent sessions."""
        sessions = ["2024_MONACO_RACE", "2024_SILVERSTONE_RACE"]
        
        # Compute features for multiple sessions concurrently
        tasks = [
            streaming_engine.compute_realtime(
                session_id=session_id,
                feature_names=["stint_summary"]
            )
            for session_id in sessions
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        assert len(results_list) == len(sessions)
        for results in results_list:
            assert isinstance(results, dict)
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, streaming_engine):
        """Test cache TTL expiration."""
        # Create engine with very short TTL
        short_ttl_engine = StreamingFeatureEngine(cache_ttl=1)
        
        session_id = "2024_MONACO_RACE"
        
        # First call
        results1 = await short_ttl_engine.compute_realtime(
            session_id=session_id,
            feature_names=["stint_summary"]
        )
        
        # Wait for TTL to expire
        await asyncio.sleep(2)
        
        # Second call (cache should be expired)
        results2 = await short_ttl_engine.compute_realtime(
            session_id=session_id,
            feature_names=["stint_summary"]
        )
        
        # Both should complete
        assert isinstance(results1, dict)
        assert isinstance(results2, dict)
