"""Tests for authentication endpoints."""

import pytest
from fastapi import status


def test_login_success(test_client):
    """Test successful login."""
    response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"}
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "access_token" in data["data"]
    assert data["data"]["token_type"] == "bearer"


def test_login_invalid_credentials(test_client):
    """Test login with invalid credentials."""
    response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "wrongpassword"}
    )
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_login_nonexistent_user(test_client):
    """Test login with nonexistent user."""
    response = test_client.post(
        "/api/v1/auth/token",
        data={"username": "nonexistent", "password": "password"}
    )
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_current_user(test_client, auth_headers):
    """Test getting current user info."""
    response = test_client.get(
        "/api/v1/auth/me",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["username"] == "admin"
    assert data["role"] == "admin"


def test_get_current_user_invalid_token(test_client):
    """Test getting user with invalid token."""
    response = test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid_token"}
    )
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_api_key_authentication(test_client, api_key_headers):
    """Test API key authentication."""
    response = test_client.get(
        "/api/v1/health",
        headers=api_key_headers
    )
    
    assert response.status_code == status.HTTP_200_OK


def test_invalid_api_key(test_client):
    """Test with invalid API key."""
    response = test_client.get(
        "/api/v1/strategy/modules",
        headers={"X-API-Key": "invalid_key"}
    )
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
