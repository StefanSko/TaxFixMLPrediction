"""
Tests for the error handling module.
"""

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, validator

from core.errors import (
    PredictionError,
    InvalidInputError,
    ModelLoadingError,
    PreprocessingError,
    InferenceError,
    add_exception_handlers
)
from core.models import ErrorCode


class TestModel(BaseModel):
    """Test model for validation errors."""
    value: int = Field(..., gt=0)

    @validator("value")
    def validate_value(cls, v):
        if v > 100:
            raise ValueError("Value must be less than or equal to 100")
        return v


@pytest.fixture
def app():
    """Create a test FastAPI app with error handlers."""
    app = FastAPI()

    # Add exception handlers
    add_exception_handlers(app)

    # Add test routes
    @app.get("/prediction-error")
    async def prediction_error():
        raise PredictionError("Test prediction error")

    @app.get("/invalid-input-error")
    async def invalid_input_error():
        raise InvalidInputError("Test invalid input error")

    @app.get("/model-loading-error")
    async def model_loading_error():
        raise ModelLoadingError("Test model loading error")

    @app.get("/preprocessing-error")
    async def preprocessing_error():
        raise PreprocessingError("Test preprocessing error")

    @app.get("/inference-error")
    async def inference_error():
        raise InferenceError("Test inference error")

    @app.get("/generic-error")
    async def generic_error():
        raise Exception("Test generic error")

    @app.post("/validation-error")
    async def validation_error(model: TestModel):
        return {"value": model.value}

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


def test_prediction_error(client):
    """Test handling of PredictionError."""
    response = client.get("/prediction-error")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert data["error_code"] == ErrorCode.PREDICTION_ERROR
    assert data["message"] == "Test prediction error"
    assert "request_id" in data
    assert "timestamp" in data


def test_invalid_input_error(client):
    """Test handling of InvalidInputError."""
    response = client.get("/invalid-input-error")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert data["error_code"] == ErrorCode.VALIDATION_ERROR
    assert data["message"] == "Test invalid input error"
    assert "request_id" in data
    assert "timestamp" in data


def test_model_loading_error(client):
    """Test handling of ModelLoadingError."""
    response = client.get("/model-loading-error")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["error_code"] == ErrorCode.MODEL_ERROR
    assert data["message"] == "Test model loading error"
    assert "request_id" in data
    assert "timestamp" in data


def test_preprocessing_error(client):
    """Test handling of PreprocessingError."""
    response = client.get("/preprocessing-error")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert data["error_code"] == ErrorCode.PREDICTION_ERROR
    assert data["message"] == "Test preprocessing error"
    assert "request_id" in data
    assert "timestamp" in data


def test_inference_error(client):
    """Test handling of InferenceError."""
    response = client.get("/inference-error")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()
    assert data["error_code"] == ErrorCode.PREDICTION_ERROR
    assert data["message"] == "Test inference error"
    assert "request_id" in data
    assert "timestamp" in data



def test_validation_error(client):
    """Test handling of validation errors."""
    # Test with invalid type
    response = client.post("/validation-error", json={"value": "not_an_int"})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY