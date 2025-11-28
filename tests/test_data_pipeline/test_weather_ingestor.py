"""
Tests for Weather Ingestor
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from data_pipeline.ingestors.weather_ingestor import WeatherIngestor
from data_pipeline.base.storage_manager import StorageManager
from data_pipeline.base.qa_engine import QAEngine


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
        valid_records=10,
        failed_records=0
    )
    return mock_qa


@pytest.fixture
def weather_ingestor(storage_manager, qa_engine):
    """Create weather ingestor instance."""
    config = {"mock_mode": True, "track_name": "Monaco"}
    return WeatherIngestor(storage_manager, qa_engine, config)


@pytest.mark.asyncio
async def test_mock_weather_generation(weather_ingestor):
    """Test mock weather data generation."""
    result = await weather_ingestor.run()
    
    assert result.success is True
    assert result.records_ingested > 0


@pytest.mark.asyncio
async def test_weather_validation(weather_ingestor):
    """Test weather data validation."""
    raw_data = await weather_ingestor.ingest()
    validated = weather_ingestor.validate(raw_data)
    
    # Check temperature ranges
    for weather in validated:
        assert 10 <= weather.air_temperature <= 50
        assert weather.track_temperature >= weather.air_temperature
        assert 0 <= weather.humidity <= 100


@pytest.mark.asyncio
async def test_forecast_generation(weather_ingestor):
    """Test weather forecast generation."""
    raw_data = await weather_ingestor.ingest()
    
    # Should include forecasts
    assert "forecasts" in raw_data
    assert len(raw_data["forecasts"]) > 0
