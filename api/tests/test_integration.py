"""Integration tests for end-to-end workflows."""

import pytest
from fastapi import status


def test_full_prediction_workflow(test_client):
    """Test complete prediction workflow."""
    # 1. Get health status
    health_response = test_client.get("/api/v1/health")
    assert health_response.status_code == status.HTTP_200_OK
    
    # 2. Login
    auth_response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"}
    )
    assert auth_response.status_code == status.HTTP_200_OK
    token = auth_response.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Predict lap time
    laptime_response = test_client.post(
        "/api/v1/predict/laptime",
        json={
            "circuit_name": "Monaco",
            "driver": "Max Verstappen",
            "team": "Red Bull Racing",
            "tire_compound": "SOFT",
            "tire_age": 5,
            "fuel_load": 80.0,
            "track_temp": 35.0,
            "air_temp": 25.0
        }
    )
    assert laptime_response.status_code == status.HTTP_200_OK
    
    # 4. Predict tire degradation
    deg_response = test_client.post(
        "/api/v1/predict/degradation",
        json={
            "circuit_name": "Monaco",
            "tire_compound": "SOFT",
            "laps": 15,
            "track_temp": 35.0,
            "fuel_load": 80.0
        }
    )
    assert deg_response.status_code == status.HTTP_200_OK
    
    # 5. Get prediction stats
    stats_response = test_client.get("/api/v1/predict/stats")
    assert stats_response.status_code == status.HTTP_200_OK


def test_full_simulation_workflow(test_client):
    """Test complete simulation workflow."""
    # 1. Login
    auth_response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"}
    )
    token = auth_response.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Simulate single strategy
    sim_response = test_client.post(
        "/api/v1/simulate/strategy",
        json={
            "circuit_name": "Monaco",
            "total_laps": 78,
            "starting_tire": "SOFT",
            "fuel_load": 105.0,
            "pit_stops": [
                {"lap": 25, "tire": "MEDIUM"}
            ]
        }
    )
    assert sim_response.status_code == status.HTTP_200_OK
    
    # 3. Compare multiple strategies
    compare_response = test_client.post(
        "/api/v1/simulate/compare-strategies",
        json={
            "circuit_name": "Monaco",
            "total_laps": 78,
            "strategies": [
                {"name": "One-stop", "pit_stops": [{"lap": 39, "tire": "HARD"}]},
                {"name": "Two-stop", "pit_stops": [{"lap": 26, "tire": "MEDIUM"}, {"lap": 52, "tire": "SOFT"}]}
            ]
        }
    )
    assert compare_response.status_code == status.HTTP_200_OK


def test_strategy_recommendation_workflow(test_client, auth_headers):
    """Test strategy recommendation workflow."""
    # 1. Get current race state recommendations
    rec_response = test_client.post(
        "/api/v1/strategy/recommend",
        json={
            "circuit_name": "Monaco",
            "current_lap": 20,
            "total_laps": 78,
            "current_position": 3,
            "current_tire": "SOFT",
            "tire_age": 18,
            "fuel_remaining": 75.0
        }
    )
    assert rec_response.status_code == status.HTTP_200_OK
    recommendation = rec_response.json()["data"]
    
    assert "recommendation" in recommendation
    assert "confidence" in recommendation
    assert "reasoning" in recommendation
    
    # 2. List available decision modules
    modules_response = test_client.get(
        "/api/v1/strategy/modules",
        headers=auth_headers
    )
    assert modules_response.status_code == status.HTTP_200_OK
    modules = modules_response.json()["data"]
    assert len(modules) > 0


def test_api_versioning(test_client):
    """Test that API versioning works correctly."""
    # All endpoints should be under /api/v1/
    endpoints = [
        "/api/v1/health",
        "/api/v1/auth/token",
        "/api/v1/predict/stats"
    ]
    
    for endpoint in endpoints:
        response = test_client.get(endpoint) if "token" not in endpoint else test_client.post(endpoint, data={"username": "admin", "password": "admin123"})
        # Should not get 404
        assert response.status_code != status.HTTP_404_NOT_FOUND


def test_error_handling_chain(test_client):
    """Test error handling across multiple requests."""
    # Invalid circuit should fail validation
    response1 = test_client.post(
        "/api/v1/predict/laptime",
        json={
            "circuit_name": "InvalidCircuit",
            "driver": "Test",
            "team": "Test",
            "tire_compound": "SOFT",
            "tire_age": 5,
            "fuel_load": 80.0,
            "track_temp": 35.0,
            "air_temp": 25.0
        }
    )
    assert response1.status_code == status.HTTP_400_BAD_REQUEST
    
    # Invalid tire compound should fail
    response2 = test_client.post(
        "/api/v1/predict/laptime",
        json={
            "circuit_name": "Monaco",
            "driver": "Test",
            "team": "Test",
            "tire_compound": "INVALID",
            "tire_age": 5,
            "fuel_load": 80.0,
            "track_temp": 35.0,
            "air_temp": 25.0
        }
    )
    assert response2.status_code == status.HTTP_400_BAD_REQUEST
