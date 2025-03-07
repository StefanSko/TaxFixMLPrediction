"""
Tests for the security module.
"""

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from core.security import get_api_key, validate_api_key, api_key_auth
from core.config import Settings, get_settings


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    app = FastAPI()

    @app.get("/protected")
    def protected_route(auth: bool = api_key_auth):
        return {"authenticated": True}

    @app.get("/get-key")
    def get_key_route(api_key: str = Depends(get_api_key)):
        return {"api_key": api_key}

    @app.get("/validate-key")
    def validate_key_route(is_valid: bool = Depends(validate_api_key)):
        return {"is_valid": is_valid}

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_settings():
    """Create mock settings with a test API key."""
    # Create a complete Settings object with all required fields
    return Settings(
        API_KEY="test-api-key-12345678901234567890",
        # Include other required fields if needed
        API_TITLE="Test API",
        API_DESCRIPTION="Test Description",
        API_VERSION="0.1.0",
        HOST="localhost",
        PORT=8000,
        DEBUG=True,
        CORS_ORIGINS=["*"],
        CORS_HEADERS=["*"],
        MODEL_PATH="test/path/model.joblib",
        PREPROCESSOR_PATH="test/path/preprocessor.joblib"
    )


def test_get_api_key_missing(client):
    """Test get_api_key with missing API key."""
    response = client.get("/get-key")
    assert response.status_code == 401
    assert "Missing API key" in response.json()["detail"]


def test_get_api_key_invalid_format(client):
    """Test get_api_key with invalid API key format."""
    response = client.get("/get-key", headers={"X-API-Key": "short"})
    assert response.status_code == 401
    assert "Invalid API key format" in response.json()["detail"]


def test_get_api_key_valid(client):
    """Test get_api_key with valid API key."""
    response = client.get("/get-key", headers={"X-API-Key": "valid-api-key-12345678901234"})
    assert response.status_code == 200
    assert response.json()["api_key"] == "valid-api-key-12345678901234"


def test_validate_api_key_missing_config(client, app, monkeypatch):
    """Test validate_api_key with missing API key configuration."""
    # Create a complete Settings object but with API_KEY=None
    mock_settings = Settings(
        API_KEY=None,
        API_TITLE="Test API",
        API_DESCRIPTION="Test Description",
        API_VERSION="0.1.0",
        HOST="localhost",
        PORT=8000,
        DEBUG=False,
        CORS_ORIGINS=["*"],
        CORS_HEADERS=["*"],
        MODEL_PATH="test/path/model.joblib",
        PREPROCESSOR_PATH="test/path/preprocessor.joblib"
    )

    # Use the correct import path for get_settings
    monkeypatch.setattr("core.security.get_settings", lambda: mock_settings)

    response = client.get(
        "/validate-key",
        headers={"X-API-Key": "valid-api-key-12345678901234"}
    )
    assert response.status_code == 500
    assert "API key authentication is not configured" in response.json()["detail"]