"""
Tests for Storage Manager
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from data_pipeline.base.storage_manager import StorageManager


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def storage_manager(temp_storage_dir):
    """Create storage manager with temp directory."""
    config = {
        "base_path": temp_storage_dir,
        "retention_days": 30
    }
    return StorageManager(config)


@pytest.fixture
def sample_data():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'lap_number': [1, 2, 3],
        'lap_time': [72.5, 73.1, 72.8],
        'timestamp': [datetime.utcnow()] * 3
    })


def test_save_raw_data(storage_manager, sample_data):
    """Test saving raw data."""
    result = storage_manager.save_raw(
        source="timing",
        data=sample_data,
        metadata={"test": True}
    )
    
    assert result is not None
    assert result["version"] is not None
    assert result["path"].exists()


def test_save_multiple_formats(storage_manager, sample_data):
    """Test saving in different formats."""
    # Parquet
    result_parquet = storage_manager.save_raw(
        source="timing",
        data=sample_data,
        format="parquet"
    )
    
    # JSON
    result_json = storage_manager.save_raw(
        source="timing",
        data=sample_data,
        format="json"
    )
    
    assert result_parquet["path"].suffix == ".parquet"
    assert result_json["path"].suffix == ".json"


def test_load_latest_data(storage_manager, sample_data):
    """Test loading latest version."""
    # Save data
    storage_manager.save_raw(source="timing", data=sample_data)
    
    # Load latest
    loaded = storage_manager.load_latest(source="timing", storage_type="raw")
    
    assert loaded is not None
    assert len(loaded) == len(sample_data)


def test_versioning(storage_manager, sample_data):
    """Test version management."""
    # Save multiple versions
    v1 = storage_manager.save_raw(source="timing", data=sample_data)
    v2 = storage_manager.save_raw(source="timing", data=sample_data)
    
    # List versions
    versions = storage_manager.list_versions(source="timing", storage_type="raw")
    
    assert len(versions) >= 2
    assert v1["version"] != v2["version"]


def test_metadata_persistence(storage_manager, sample_data):
    """Test metadata saving and loading."""
    metadata = {"session_id": "TEST123", "track": "Monaco"}
    
    result = storage_manager.save_raw(
        source="timing",
        data=sample_data,
        metadata=metadata
    )
    
    # Check metadata file exists
    metadata_file = result["path"].parent / "manifest.json"
    assert metadata_file.exists()
