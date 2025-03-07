"""
Tests for the prediction endpoints.
"""

from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.prediction import router as prediction_router
from core.models import ErrorCode
from api.dependencies import ensure_prediction_service
from services.predictor import get_prediction_service


@pytest.fixture
def app():
    """Create a test FastAPI app with the prediction router."""
    app = FastAPI()
    app.include_router(prediction_router)
    return app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    client = TestClient(app, base_url="http://test")
    return client


@pytest.fixture
def valid_request_data():
    """Valid prediction request data."""
    return {
        "age": 35,
        "income": 50000.0,
        "employment_type": "employed",
        "marital_status": "married",
        "time_spent_on_platform": 45.5,
        "number_of_sessions": 3,
        "fields_filled_percentage": 75.0,
        "previous_year_filing": True,
        "device_type": "mobile",
        "referral_source": "search"
    }


@pytest.fixture
def mock_prediction_service():
    """Mock prediction service."""
    service = mock.MagicMock()
    service.process_request.return_value = {
        "prediction": True,
        "completion_probability": 0.85
    }
    return service


@pytest.fixture(autouse=True)
def cleanup_dependency_overrides(app):
    """Clean up dependency overrides after each test."""
    yield
    app.dependency_overrides = {}


def test_predict_success(client, app, mock_prediction_service):
    """Test successful prediction."""
    app.dependency_overrides[ensure_prediction_service] = lambda: None
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service
    
    valid_request_data = {
        "age": 35,
        "income": 50000.0,
        "employment_type": "employed",
        "marital_status": "married",
        "time_spent_on_platform": 45.5,
        "number_of_sessions": 3,
        "fields_filled_percentage": 75.0,
        "previous_year_filing": True,
        "device_type": "mobile",
        "referral_source": "search"
    }
    
    response = client.post("/api/v1/predict", json=valid_request_data)
    
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] is True
    assert data["completion_probability"] == 0.85
    assert "request_id" in data
    assert "timestamp" in data
    
    mock_prediction_service.process_request.assert_called_once_with(valid_request_data)


def test_predict_validation_error(client, app, mock_prediction_service):
    """Test prediction with invalid data."""
    app.dependency_overrides[ensure_prediction_service] = lambda: None
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service

    invalid_data = {"age": 35}
    response = client.post("/api/v1/predict", json=invalid_data)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_processing_error(client, app, valid_request_data, mock_prediction_service):
    """Test prediction with processing error."""
    mock_prediction_service.process_request.side_effect = ValueError("Test error")
    app.dependency_overrides[ensure_prediction_service] = lambda: None
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service

    response = client.post("/api/v1/predict", json=valid_request_data)

    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == ErrorCode.PREDICTION_ERROR
    assert "Test error" in data["message"]
    assert "request_id" in data
    assert "timestamp" in data


def test_predict_unexpected_error(client, app, valid_request_data, mock_prediction_service):
    """Test prediction with unexpected error."""
    mock_prediction_service.process_request.side_effect = Exception("Unexpected error")
    app.dependency_overrides[ensure_prediction_service] = lambda: None
    app.dependency_overrides[get_prediction_service] = lambda: mock_prediction_service

    response = client.post("/api/v1/predict", json=valid_request_data)

    assert response.status_code == 500
    data = response.json()
    assert data["error_code"] == ErrorCode.INTERNAL_ERROR
    assert "unexpected error" in data["message"].lower()
    assert "request_id" in data
    assert "timestamp" in data


def test_batch_predict_not_implemented(client):
    """Test batch prediction endpoint."""
    response = client.post("/api/v1/batch-predict")

    assert response.status_code == 501
    data = response.json()
    assert data["error_code"] == ErrorCode.NOT_FOUND
    assert "not implemented" in data["message"].lower()