"""
Tests for Ingestion Orchestrator
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from data_pipeline.orchestrator import IngestionOrchestrator, SessionState


@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        "storage": {"base_path": "data"},
        "qa": {},
        "timing": {"mock_mode": True},
        "weather": {"mock_mode": True},
        "historical": {"year": 2024},
        "safety_car": {"mock_mode": True},
        "telemetry": {"mock_mode": True}
    }


@pytest.fixture
def orchestrator(mock_config):
    """Create orchestrator instance."""
    return IngestionOrchestrator(mock_config)


def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initializes all ingestors."""
    assert len(orchestrator.ingestors) == 5
    assert 'timing' in orchestrator.ingestors
    assert 'weather' in orchestrator.ingestors
    assert orchestrator.session_state == SessionState.PRE_SESSION


@pytest.mark.asyncio
async def test_historical_batch_ingestion(orchestrator):
    """Test historical batch processing."""
    with patch.object(orchestrator.ingestors['historical'], 'run', new_callable=AsyncMock) as mock_run:
        mock_run.return_value = Mock(
            success=True,
            records_ingested=1000,
            records_failed=0
        )
        
        result = await orchestrator.run_historical_batch(year=2024)
        
        assert result.success is True
        assert result.records_ingested == 1000


def test_health_status(orchestrator):
    """Test health status reporting."""
    health = orchestrator.get_health_status()
    
    assert 'session_state' in health
    assert 'ingestors' in health
    assert len(health['ingestors']) == 5


@pytest.mark.asyncio
async def test_session_lifecycle(orchestrator):
    """Test session state transitions."""
    assert orchestrator.session_state == SessionState.PRE_SESSION
    
    # Mock session (will timeout quickly)
    session_info = {"name": "Test Session"}
    
    with patch.object(orchestrator, '_run_ingestor', new_callable=AsyncMock):
        with patch('asyncio.gather', new_callable=AsyncMock):
            try:
                await orchestrator.run_live_session(session_info)
            except:
                pass
    
    # Should reach completed state after cleanup
    assert orchestrator.session_state in [SessionState.POST_SESSION, SessionState.COMPLETED]
