"""Tests for simulation endpoints."""

import pytest
from fastapi import status


def test_simulate_strategy(test_client):
    """Test race strategy simulation."""
    payload = {
        "circuit_name": "Monaco",
        "total_laps": 78,
        "starting_tire": "SOFT",
        "fuel_load": 105.0,
        "weather_condition": "Dry",
        "pit_stops": [
            {"lap": 25, "tire": "MEDIUM"},
            {"lap": 50, "tire": "MEDIUM"}
        ]
    }
    
    response = test_client.post("/api/v1/simulate/strategy", json=payload)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "total_race_time" in data["data"]
    assert "final_position" in data["data"]
    assert "pit_stop_count" in data["data"]
    assert "tire_strategy" in data["data"]


def test_compare_strategies(test_client):
    """Test strategy comparison."""
    payload = {
        "circuit_name": "Silverstone",
        "total_laps": 52,
        "strategies": [
            {"name": "One-stop", "pit_stops": [{"lap": 26, "tire": "HARD"}]},
            {"name": "Two-stop", "pit_stops": [{"lap": 18, "tire": "MEDIUM"}, {"lap": 35, "tire": "SOFT"}]}
        ]
    }
    
    response = test_client.post("/api/v1/simulate/compare-strategies", json=payload)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "best_strategy" in data["data"]
    assert "comparisons" in data["data"]
    assert "time_differences" in data["data"]


def test_monte_carlo_simulation(test_client):
    """Test Monte Carlo simulation."""
    payload = {
        "circuit_name": "Spa",
        "total_laps": 44,
        "starting_tire": "MEDIUM",
        "fuel_load": 100.0,
        "pit_stops": [{"lap": 22, "tire": "SOFT"}]
    }
    
    response = test_client.post(
        "/api/v1/simulate/monte-carlo",
        json=payload,
        params={"iterations": 500}
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "mean_race_time" in data["data"]
    assert "percentiles" in data["data"]


def test_simulation_invalid_circuit(test_client):
    """Test simulation with invalid circuit."""
    payload = {
        "circuit_name": "InvalidCircuit",
        "total_laps": 50,
        "starting_tire": "SOFT",
        "fuel_load": 100.0
    }
    
    response = test_client.post("/api/v1/simulate/strategy", json=payload)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_compare_strategies_too_few(test_client):
    """Test comparison with too few strategies."""
    payload = {
        "circuit_name": "Monza",
        "total_laps": 53,
        "strategies": [
            {"name": "One-stop", "pit_stops": [{"lap": 26, "tire": "HARD"}]}
        ]
    }
    
    response = test_client.post("/api/v1/simulate/compare-strategies", json=payload)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
