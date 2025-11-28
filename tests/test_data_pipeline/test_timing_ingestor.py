"""
Tests for Timing Ingestor
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from data_pipeline.ingestors.timing_ingestor import TimingIngestor
from data_pipeline.base.storage_manager import StorageManager
from data_pipeline.base.qa_engine import QAEngine
from data_pipeline.schemas.timing_schema import SessionTiming


@pytest.fixture
def storage_manager():
    """Mock storage manager."""
    return Mock(spec=StorageManager)


@pytest.fixture
def qa_engine():
    """Mock QA engine."""
    mock_qa = Mock(spec=QAEngine)
    mock_qa.run_checks.return_value = Mock(
        passed=True,
        valid_records=100,
        failed_records=0,
        anomalies_detected=0
    )
    return mock_qa


@pytest.fixture
def timing_ingestor(storage_manager, qa_engine):
    """Create timing ingestor instance."""
    config = {"mock_mode": True, "track_name": "Monaco"}
    return TimingIngestor(storage_manager, qa_engine, config)


@pytest.mark.asyncio
async def test_mock_ingestion(timing_ingestor, storage_manager):
    """Test mock data ingestion."""
    result = await timing_ingestor.run()
    
    assert result.success is True
    assert result.records_ingested > 0
    assert result.records_failed == 0
    assert len(result.errors) == 0


@pytest.mark.asyncio
async def test_validation(timing_ingestor):
    """Test timing data validation."""
    # Generate mock data
    raw_data = await timing_ingestor.ingest()
    
    # Validate
    validated = timing_ingestor.validate(raw_data)
    
    assert len(validated) > 0
    # All lap times should be within range
    for timing in validated:
        assert 30 <= timing.lap_data.lap_time <= 150


@pytest.mark.asyncio
async def test_duplicate_lap_prevention(timing_ingestor):
    """Test that duplicate laps are not re-ingested."""
    # First ingestion
    result1 = await timing_ingestor.run()
    first_lap = timing_ingestor.last_lap_number
    
    # Second ingestion should continue from last lap
    result2 = await timing_ingestor.run()
    
    assert timing_ingestor.last_lap_number > first_lap


@pytest.mark.asyncio
async def test_storage_integration(timing_ingestor, storage_manager):
    """Test data storage."""
    await timing_ingestor.run()
    
    # Verify storage methods called
    assert storage_manager.save_raw.called or storage_manager.save_processed.called
