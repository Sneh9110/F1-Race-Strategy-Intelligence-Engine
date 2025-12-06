"""Tests for prediction endpoints."""

import pytest
from fastapi import status


def test_predict_lap_time(test_client):
    """Test lap time prediction."""
    payload = {
        "circuit_name": "Monaco",
        "driver": "Max Verstappen",
        "team": "Red Bull Racing",
        "tire_compound": "SOFT",
        "tire_age": 5,
        "fuel_load": 80.0,
        "track_temp": 35.0,
        "air_temp": 25.0,
        "weather_condition": "Dry"
    }
    
    response = test_client.post("/api/v1/predict/laptime", json=payload)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "predicted_lap_time" in data["data"]
    assert "confidence" in data["data"]
    assert "metadata" in data["data"]


def test_predict_tire_degradation(test_client):
    """Test tire degradation prediction."""
    payload = {
        "circuit_name": "Silverstone",
        "tire_compound": "MEDIUM",
        "laps": 15,
        "track_temp": 30.0,
        "fuel_load": 75.0,
        "downforce_level": "Medium"
    }
    
    response = test_client.post("/api/v1/predict/degradation", json=payload)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "degradation_per_lap" in data["data"]
    assert "total_degradation" in data["data"]
    assert "remaining_performance" in data["data"]


def test_predict_safety_car(test_client):
    """Test safety car prediction."""
    payload = {
        "circuit_name": "Baku",
        "lap": 35,
        "total_laps": 51,
        "weather_condition": "Dry",
        "incidents_so_far": 1
    }
    
    response = test_client.post("/api/v1/predict/safety-car", json=payload)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "probability" in data["data"]
    assert "risk_level" in data["data"]


def test_predict_pit_stop_loss(test_client):
    """Test pit stop loss prediction."""
    payload = {
        "circuit_name": "Monza",
        "pit_lane_type": "Standard",
        "traffic_density": 0.3
    }
    
    response = test_client.post("/api/v1/predict/pit-stop-loss", json=payload)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "time_loss" in data["data"]
    assert "range_min" in data["data"]
    assert "range_max" in data["data"]


def test_invalid_circuit_name(test_client):
    """Test prediction with invalid circuit name."""
    payload = {
        "circuit_name": "InvalidCircuit",
        "driver": "Test Driver",
        "team": "Test Team",
        "tire_compound": "SOFT",
        "tire_age": 5,
        "fuel_load": 80.0,
        "track_temp": 35.0,
        "air_temp": 25.0
    }
    
    response = test_client.post("/api/v1/predict/laptime", json=payload)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_invalid_tire_compound(test_client):
    """Test prediction with invalid tire compound."""
    payload = {
        "circuit_name": "Monaco",
        "driver": "Test Driver",
        "team": "Test Team",
        "tire_compound": "INVALID",
        "tire_age": 5,
        "fuel_load": 80.0,
        "track_temp": 35.0,
        "air_temp": 25.0
    }
    
    response = test_client.post("/api/v1/predict/laptime", json=payload)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_get_prediction_stats(test_client):
    """Test getting prediction model statistics."""
    response = test_client.get("/api/v1/predict/stats")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "model_name" in data["data"]
    assert "total_predictions" in data["data"]
