"""Tests for health and monitoring endpoints."""

import pytest
from fastapi import status


def test_health_check(test_client):
    """Test basic health check."""
    response = test_client.get("/api/v1/health")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "uptime_seconds" in data
    assert "components" in data


def test_readiness_probe(test_client):
    """Test Kubernetes readiness probe."""
    response = test_client.get("/api/v1/health/ready")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"


def test_liveness_probe(test_client):
    """Test Kubernetes liveness probe."""
    response = test_client.get("/api/v1/health/live")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "alive"


def test_metrics_endpoint(test_client):
    """Test Prometheus metrics endpoint."""
    response = test_client.get("/api/v1/metrics")
    
    assert response.status_code == status.HTTP_200_OK
    assert "text/plain" in response.headers["content-type"]
    assert "api_uptime_seconds" in response.text


def test_health_check_no_auth_required(test_client):
    """Test that health check doesn't require authentication."""
    response = test_client.get("/api/v1/health")
    
    assert response.status_code == status.HTTP_200_OK
