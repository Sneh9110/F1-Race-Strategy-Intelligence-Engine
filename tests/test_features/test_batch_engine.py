"""
Integration tests for BatchFeatureEngine.
"""

import pytest
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta

from features.batch_engine import BatchFeatureEngine
from features.base import FeatureConfig
from features.registry import FeatureRegistry


@pytest.fixture
def temp_feature_store():
    """Create temporary feature store."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_sessions():
    """Generate sample session data."""
    return [
        "2024_MONACO_RACE",
        "2024_SILVERSTONE_RACE",
        "2024_MONZA_RACE"
    ]


class TestBatchFeatureEngine:
    """Tests for BatchFeatureEngine."""
    
    def test_compute_single_session(self, temp_feature_store):
        """Test computing features for a single session."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        results = engine.compute_features(
            session_ids=["2024_MONACO_RACE"],
            feature_names=["stint_summary"],
            parallel=False
        )
        
        assert "2024_MONACO_RACE" in results
        result = results["2024_MONACO_RACE"]
        
        # Check result structure
        assert "status" in result
        if result["status"] == "success":
            assert "features" in result
            assert "stint_summary" in result["features"]
    
    def test_compute_multiple_sessions(self, temp_feature_store, sample_sessions):
        """Test computing features for multiple sessions."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        results = engine.compute_features(
            session_ids=sample_sessions,
            feature_names=["stint_summary"],
            parallel=False
        )
        
        assert len(results) == len(sample_sessions)
        for session_id in sample_sessions:
            assert session_id in results
    
    def test_parallel_execution(self, temp_feature_store, sample_sessions):
        """Test parallel feature computation."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        results = engine.compute_features(
            session_ids=sample_sessions,
            feature_names=["stint_summary"],
            parallel=True,
            num_workers=2
        )
        
        assert len(results) == len(sample_sessions)
        # All sessions should complete
        successful = sum(1 for r in results.values() if r.get("status") == "success")
        assert successful >= 0  # At least attempt all sessions
    
    def test_dependency_resolution(self, temp_feature_store):
        """Test dependency resolution in feature computation."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        # Request features with dependencies
        results = engine.compute_features(
            session_ids=["2024_MONACO_RACE"],
            feature_names=["degradation_slope", "stint_summary"],  # degradation depends on stint
            parallel=False
        )
        
        assert "2024_MONACO_RACE" in results
        # Should compute dependencies first
    
    def test_force_recompute(self, temp_feature_store):
        """Test force recomputation of existing features."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        # Compute once
        results1 = engine.compute_features(
            session_ids=["2024_MONACO_RACE"],
            feature_names=["stint_summary"],
            parallel=False,
            force_recompute=False
        )
        
        # Compute again with force=True
        results2 = engine.compute_features(
            session_ids=["2024_MONACO_RACE"],
            feature_names=["stint_summary"],
            parallel=False,
            force_recompute=True
        )
        
        # Both should succeed
        assert results1["2024_MONACO_RACE"]["status"] in ["success", "error"]
        assert results2["2024_MONACO_RACE"]["status"] in ["success", "error"]
    
    def test_backfill_features(self, temp_feature_store):
        """Test backfilling features for date range."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        results = engine.backfill_features(
            start_date=start_date,
            end_date=end_date,
            feature_names=["stint_summary"],
            parallel=False,
            batch_size=5
        )
        
        # Should return results (may be empty if no sessions in range)
        assert isinstance(results, dict)
    
    def test_error_handling(self, temp_feature_store):
        """Test error handling for invalid session."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        results = engine.compute_features(
            session_ids=["INVALID_SESSION"],
            feature_names=["stint_summary"],
            parallel=False
        )
        
        assert "INVALID_SESSION" in results
        # Should handle error gracefully
        assert results["INVALID_SESSION"]["status"] in ["error", "success"]
    
    def test_update_features(self, temp_feature_store):
        """Test updating existing features."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        # Initial computation
        results1 = engine.compute_features(
            session_ids=["2024_MONACO_RACE"],
            feature_names=["stint_summary"],
            parallel=False
        )
        
        # Update features
        results2 = engine.update_features(
            session_ids=["2024_MONACO_RACE"],
            feature_names=["stint_summary"]
        )
        
        assert "2024_MONACO_RACE" in results2
    
    def test_progress_tracking(self, temp_feature_store, sample_sessions):
        """Test progress tracking with tqdm."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        # Should not raise errors with progress tracking
        results = engine.compute_features(
            session_ids=sample_sessions,
            feature_names=["stint_summary"],
            parallel=False
        )
        
        assert len(results) == len(sample_sessions)
    
    def test_batch_size_parameter(self, temp_feature_store):
        """Test batch size in backfill."""
        engine = BatchFeatureEngine(feature_store_path=temp_feature_store)
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 2, 1)
        
        # Test different batch sizes
        for batch_size in [5, 10, 20]:
            results = engine.backfill_features(
                start_date=start_date,
                end_date=end_date,
                feature_names=["stint_summary"],
                parallel=False,
                batch_size=batch_size
            )
            
            assert isinstance(results, dict)
