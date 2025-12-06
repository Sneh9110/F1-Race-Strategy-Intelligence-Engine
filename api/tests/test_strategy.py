"""Tests for strategy recommendation endpoints."""

import pytest
from fastapi import status


def test_recommend_strategy(test_client):
    """Test strategy recommendation."""
    payload = {
        "circuit_name": "Spa",
        "current_lap": 20,
        "total_laps": 44,
        "current_position": 3,
        "current_tire": "MEDIUM",
        "tire_age": 18,
        "fuel_remaining": 60.0,
        "gap_to_leader": 8.5,
        "gap_to_next": 2.3,
        "weather_condition": "Dry",
        "safety_car_deployed": False
    }
    
    response = test_client.post("/api/v1/strategy/recommend", json=payload)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "recommendation" in data["data"]
    assert "confidence" in data["data"]
    assert "reasoning" in data["data"]
    assert "alternative_options" in data["data"]


def test_list_decision_modules_requires_auth(test_client):
    """Test that listing modules requires authentication."""
    response = test_client.get("/api/v1/strategy/modules")
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_list_decision_modules_with_auth(test_client, auth_headers):
    """Test listing decision modules with authentication."""
    response = test_client.get(
        "/api/v1/strategy/modules",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    
    # Check module structure
    module = data["data"][0]
    assert "name" in module
    assert "priority" in module
    assert "enabled" in module
    assert "description" in module


def test_recommendation_invalid_circuit(test_client):
    """Test recommendation with invalid circuit."""
    payload = {
        "circuit_name": "InvalidCircuit",
        "current_lap": 20,
        "total_laps": 44,
        "current_position": 3,
        "current_tire": "MEDIUM",
        "tire_age": 18,
        "fuel_remaining": 60.0
    }
    
    response = test_client.post("/api/v1/strategy/recommend", json=payload)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_recommendation_with_safety_car(test_client):
    """Test recommendation during safety car period."""
    payload = {
        "circuit_name": "Singapore",
        "current_lap": 35,
        "total_laps": 61,
        "current_position": 5,
        "current_tire": "SOFT",
        "tire_age": 20,
        "fuel_remaining": 45.0,
        "safety_car_deployed": True
    }
    
    response = test_client.post("/api/v1/strategy/recommend", json=payload)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
