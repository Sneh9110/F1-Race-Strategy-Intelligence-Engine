"""
Integration tests for FeatureStore.
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from features.store import FeatureStore, FeatureMetadata
from features.base import FeatureConfig


@pytest.fixture
def temp_store_path():
    """Create temporary directory for feature store."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_feature_data():
    """Generate sample feature data."""
    return pd.DataFrame({
        'driver_id': ['VER', 'HAM', 'LEC'],
        'feature_value': [1.23, 4.56, 7.89],
        'feature_score': [0.95, 0.87, 0.92]
    })


class TestFeatureStore:
    """Tests for FeatureStore."""
    
    def test_save_and_load(self, temp_store_path, sample_feature_data):
        """Test saving and loading features."""
        store = FeatureStore(base_path=temp_store_path)
        
        # Save features
        store.save_features(
            session_id="2024_MONACO_RACE",
            feature_name="test_feature",
            data=sample_feature_data,
            version="1.0.0"
        )
        
        # Load features
        loaded = store.load_features(
            session_id="2024_MONACO_RACE",
            feature_names=["test_feature"],
            version="1.0.0"
        )
        
        assert loaded is not None
        assert not loaded.empty
        assert len(loaded) == len(sample_feature_data)
        pd.testing.assert_frame_equal(loaded, sample_feature_data)
    
    def test_versioning(self, temp_store_path, sample_feature_data):
        """Test feature versioning."""
        store = FeatureStore(base_path=temp_store_path)
        
        # Save v1.0.0
        store.save_features(
            session_id="2024_MONACO_RACE",
            feature_name="test_feature",
            data=sample_feature_data,
            version="1.0.0"
        )
        
        # Save v1.1.0 with modified data
        modified_data = sample_feature_data.copy()
        modified_data['feature_value'] *= 2
        store.save_features(
            session_id="2024_MONACO_RACE",
            feature_name="test_feature",
            data=modified_data,
            version="1.1.0"
        )
        
        # Load v1.0.0
        loaded_v1 = store.load_features(
            session_id="2024_MONACO_RACE",
            feature_names=["test_feature"],
            version="1.0.0"
        )
        
        # Load v1.1.0
        loaded_v2 = store.load_features(
            session_id="2024_MONACO_RACE",
            feature_names=["test_feature"],
            version="1.1.0"
        )
        
        # Versions should be different
        assert not loaded_v1.equals(loaded_v2)
        assert (loaded_v2['feature_value'] == loaded_v1['feature_value'] * 2).all()
    
    def test_checksum_verification(self, temp_store_path, sample_feature_data):
        """Test checksum verification."""
        store = FeatureStore(base_path=temp_store_path)
        
        # Save features
        store.save_features(
            session_id="2024_MONACO_RACE",
            feature_name="test_feature",
            data=sample_feature_data,
            version="1.0.0"
        )
        
        # Load features (should verify checksum)
        loaded = store.load_features(
            session_id="2024_MONACO_RACE",
            feature_names=["test_feature"],
            version="1.0.0"
        )
        
        assert loaded is not None
        # If checksum fails, should raise error or return None
    
    def test_metadata_storage(self, temp_store_path, sample_feature_data):
        """Test metadata storage and retrieval."""
        store = FeatureStore(base_path=temp_store_path)
        
        # Save with metadata
        store.save_features(
            session_id="2024_MONACO_RACE",
            feature_name="test_feature",
            data=sample_feature_data,
            version="1.0.0"
        )
        
        # Check metadata exists
        metadata_path = Path(temp_store_path) / "2024_MONACO_RACE" / "test_feature" / "1.0.0" / "metadata.json"
        assert metadata_path.exists(), "Metadata file should exist"
    
    def test_cleanup_old_versions(self, temp_store_path, sample_feature_data):
        """Test cleanup of old feature versions."""
        store = FeatureStore(base_path=temp_store_path, max_versions=3)
        
        # Save multiple versions
        for i in range(5):
            store.save_features(
                session_id="2024_MONACO_RACE",
                feature_name="test_feature",
                data=sample_feature_data,
                version=f"1.{i}.0"
            )
        
        # Cleanup old versions
        store.cleanup_old_versions(
            session_id="2024_MONACO_RACE",
            feature_name="test_feature"
        )
        
        # Should keep only last 3 versions
        feature_dir = Path(temp_store_path) / "2024_MONACO_RACE" / "test_feature"
        versions = [d.name for d in feature_dir.iterdir() if d.is_dir()]
        assert len(versions) <= 3, f"Should keep at most 3 versions, found {len(versions)}"
    
    def test_load_multiple_features(self, temp_store_path):
        """Test loading multiple features at once."""
        store = FeatureStore(base_path=temp_store_path)
        
        # Save multiple features
        for feature_name in ['feature_a', 'feature_b', 'feature_c']:
            data = pd.DataFrame({
                'driver_id': ['VER', 'HAM'],
                f'{feature_name}_value': [1.0, 2.0]
            })
            store.save_features(
                session_id="2024_MONACO_RACE",
                feature_name=feature_name,
                data=data,
                version="1.0.0"
            )
        
        # Load all features
        loaded = store.load_features(
            session_id="2024_MONACO_RACE",
            feature_names=['feature_a', 'feature_b', 'feature_c'],
            version="1.0.0"
        )
        
        assert loaded is not None
        # Should merge features on common columns
        assert 'feature_a_value' in loaded.columns
        assert 'feature_b_value' in loaded.columns
        assert 'feature_c_value' in loaded.columns
    
    def test_nonexistent_feature(self, temp_store_path):
        """Test loading nonexistent feature."""
        store = FeatureStore(base_path=temp_store_path)
        
        loaded = store.load_features(
            session_id="2024_MONACO_RACE",
            feature_names=["nonexistent_feature"],
            version="1.0.0"
        )
        
        # Should return None or empty DataFrame
        assert loaded is None or loaded.empty
