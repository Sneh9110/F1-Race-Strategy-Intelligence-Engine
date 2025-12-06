"""API Integration Tests - Tests for FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient
from fastapi import status
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.main import app

# Create test client
client = TestClient(app)


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""
    
    def test_health_check(self):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data
    
    def test_readiness_probe(self):
        """Test Kubernetes readiness probe."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "ready"
    
    def test_liveness_probe(self):
        """Test Kubernetes liveness probe."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "alive"
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == status.HTTP_200_OK
        assert "text/plain" in response.headers["content-type"]
        assert "api_uptime_seconds" in response.text
    
    def test_correlation_id_in_response(self):
        """Test that correlation ID is added to response headers."""
        response = client.get("/api/v1/health")
        assert "X-Correlation-ID" in response.headers
        assert "X-Response-Time" in response.headers


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    def test_login_success(self):
        """Test successful login with valid credentials."""
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "access_token" in data["data"]
        assert data["data"]["token_type"] == "bearer"
        assert "expires_in" in data["data"]
    
    def test_login_invalid_password(self):
        """Test login with invalid password."""
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "wrongpassword"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_login_nonexistent_user(self):
        """Test login with nonexistent user."""
        response = client.post(
            "/api/v1/auth/token",
            data={"username": "nonexistent", "password": "password"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user(self):
        """Test getting current user info with valid token."""
        # First, login to get token
        login_response = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["data"]["access_token"]
        
        # Then, get user info
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"
    
    def test_get_current_user_invalid_token(self):
        """Test getting user info with invalid token."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_api_key_authentication(self):
        """Test API key authentication."""
        response = client.get(
            "/api/v1/predict/stats",
            headers={"X-API-Key": "test_key_12345"}
        )
        assert response.status_code == status.HTTP_200_OK


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_predict_lap_time(self):
        """Test lap time prediction endpoint."""
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
        
        response = client.post("/api/v1/predict/laptime", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "predicted_lap_time" in data["data"]
        assert "confidence" in data["data"]
        assert "metadata" in data["data"]
        
        # Check metadata
        metadata = data["data"]["metadata"]
        assert "request_id" in metadata
        assert "timestamp" in metadata
        assert "latency_ms" in metadata
        assert "cache_hit" in metadata
    
    def test_predict_tire_degradation(self):
        """Test tire degradation prediction endpoint."""
        payload = {
            "circuit_name": "Silverstone",
            "tire_compound": "MEDIUM",
            "laps": 15,
            "track_temp": 30.0,
            "fuel_load": 75.0,
            "downforce_level": "Medium"
        }
        
        response = client.post("/api/v1/predict/degradation", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "degradation_per_lap" in data["data"]
        assert "total_degradation" in data["data"]
        assert "remaining_performance" in data["data"]
    
    def test_predict_safety_car(self):
        """Test safety car probability prediction."""
        payload = {
            "circuit_name": "Baku",
            "lap": 35,
            "total_laps": 51,
            "weather_condition": "Dry",
            "incidents_so_far": 1
        }
        
        response = client.post("/api/v1/predict/safety-car", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "probability" in data["data"]
        assert "risk_level" in data["data"]
        assert 0 <= data["data"]["probability"] <= 1
    
    def test_predict_pit_stop_loss(self):
        """Test pit stop loss prediction."""
        payload = {
            "circuit_name": "Monza",
            "pit_lane_type": "Standard",
            "traffic_density": 0.3
        }
        
        response = client.post("/api/v1/predict/pit-stop-loss", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "time_loss" in data["data"]
        assert "range_min" in data["data"]
        assert "range_max" in data["data"]
    
    def test_invalid_circuit_name(self):
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
        
        response = client.post("/api/v1/predict/laptime", json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_invalid_tire_compound(self):
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
        
        response = client.post("/api/v1/predict/laptime", json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_get_prediction_stats(self):
        """Test getting prediction model statistics."""
        response = client.get("/api/v1/predict/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "model_name" in data["data"]
        assert "total_predictions" in data["data"]


class TestSimulationEndpoints:
    """Test simulation endpoints."""
    
    def test_simulate_strategy(self):
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
        
        response = client.post("/api/v1/simulate/strategy", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "total_race_time" in data["data"]
        assert "final_position" in data["data"]
        assert "pit_stop_count" in data["data"]
        assert "tire_strategy" in data["data"]
    
    def test_compare_strategies(self):
        """Test strategy comparison."""
        payload = {
            "circuit_name": "Silverstone",
            "total_laps": 52,
            "strategies": [
                {"name": "One-stop", "pit_stops": [{"lap": 26, "tire": "HARD"}]},
                {"name": "Two-stop", "pit_stops": [{"lap": 18, "tire": "MEDIUM"}, {"lap": 35, "tire": "SOFT"}]}
            ]
        }
        
        response = client.post("/api/v1/simulate/compare-strategies", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "best_strategy" in data["data"]
        assert "comparisons" in data["data"]
        assert "time_differences" in data["data"]
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        payload = {
            "circuit_name": "Spa",
            "total_laps": 44,
            "starting_tire": "MEDIUM",
            "fuel_load": 100.0,
            "pit_stops": [{"lap": 22, "tire": "SOFT"}]
        }
        
        response = client.post(
            "/api/v1/simulate/monte-carlo",
            json=payload,
            params={"iterations": 500}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "mean_race_time" in data["data"]
        assert "percentiles" in data["data"]
    
    def test_simulation_invalid_circuit(self):
        """Test simulation with invalid circuit."""
        payload = {
            "circuit_name": "InvalidCircuit",
            "total_laps": 50,
            "starting_tire": "SOFT",
            "fuel_load": 100.0
        }
        
        response = client.post("/api/v1/simulate/strategy", json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestStrategyEndpoints:
    """Test strategy recommendation endpoints."""
    
    def test_recommend_strategy(self):
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
        
        response = client.post("/api/v1/strategy/recommend", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "recommendation" in data["data"]
        assert "confidence" in data["data"]
        assert "reasoning" in data["data"]
        assert "alternative_options" in data["data"]
        assert 0 <= data["data"]["confidence"] <= 1
    
    def test_list_decision_modules_requires_auth(self):
        """Test that listing modules requires authentication."""
        response = client.get("/api/v1/strategy/modules")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_list_decision_modules_with_auth(self):
        """Test listing decision modules with authentication."""
        # First, login to get token
        login_response = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["data"]["access_token"]
        
        # Then, list modules
        response = client.get(
            "/api/v1/strategy/modules",
            headers={"Authorization": f"Bearer {token}"}
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


class TestAPIVersioning:
    """Test API versioning."""
    
    def test_api_version_prefix(self):
        """Test that all endpoints use /api/v1 prefix."""
        endpoints = [
            "/api/v1/health",
            "/api/v1/auth/token",
            "/api/v1/predict/stats"
        ]
        
        for endpoint in endpoints:
            if "token" in endpoint:
                response = client.post(endpoint, data={"username": "admin", "password": "admin123"})
            else:
                response = client.get(endpoint)
            
            # Should not get 404
            assert response.status_code != status.HTTP_404_NOT_FOUND


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_validation_error(self):
        """Test validation error response format."""
        # Missing required fields
        payload = {
            "circuit_name": "Monaco",
            # Missing other required fields
        }
        
        response = client.post("/api/v1/predict/laptime", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_404_not_found(self):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
