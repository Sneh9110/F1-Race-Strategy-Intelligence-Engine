"""Pytest configuration and fixtures."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
import redis.asyncio as redis

from api.main import app
from api.auth import User


@pytest.fixture
def test_client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = AsyncMock(spec=redis.Redis)
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True
    mock.delete.return_value = 1
    mock.ping.return_value = True
    mock.zadd.return_value = 1
    mock.zcard.return_value = 0
    mock.zremrangebyscore.return_value = 0
    return mock


@pytest.fixture
def mock_lap_time_predictor():
    """Mock lap time predictor."""
    mock = MagicMock()
    mock.predict.return_value = (90.5, 0.85)
    return mock


@pytest.fixture
def mock_tire_degradation_predictor():
    """Mock tire degradation predictor."""
    mock = MagicMock()
    mock.predict.return_value = (0.8, 12.0, 88.0)
    return mock


@pytest.fixture
def mock_safety_car_predictor():
    """Mock safety car predictor."""
    mock = MagicMock()
    mock.predict.return_value = (0.25, "MEDIUM")
    return mock


@pytest.fixture
def mock_pit_stop_loss_predictor():
    """Mock pit stop loss predictor."""
    mock = MagicMock()
    mock.predict.return_value = (22.5, 20.0, 25.0)
    return mock


@pytest.fixture
def mock_race_simulator():
    """Mock race simulator."""
    mock = MagicMock()
    mock.simulate.return_value = {
        "total_race_time": 5850.5,
        "final_position": 2,
        "lap_times": []
    }
    return mock


@pytest.fixture
def mock_decision_engine():
    """Mock decision engine."""
    mock = MagicMock()
    mock.make_decision.return_value = {
        "recommendation": "Pit for SOFT tires",
        "confidence": 0.85,
        "reasoning": "Track conditions optimal"
    }
    return mock


@pytest.fixture
def admin_user():
    """Admin user for testing."""
    return User(
        username="admin",
        email="admin@f1strategy.com",
        role="admin",
        disabled=False
    )


@pytest.fixture
def regular_user():
    """Regular user for testing."""
    return User(
        username="testuser",
        email="user@f1strategy.com",
        role="user",
        disabled=False
    )


@pytest.fixture
def valid_jwt_token(test_client):
    """Generate valid JWT token for testing."""
    response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"}
    )
    return response.json()["data"]["access_token"]


@pytest.fixture
def auth_headers(valid_jwt_token):
    """Headers with valid JWT token."""
    return {"Authorization": f"Bearer {valid_jwt_token}"}


@pytest.fixture
def api_key_headers():
    """Headers with valid API key."""
    return {"X-API-Key": "test_key_12345"}
