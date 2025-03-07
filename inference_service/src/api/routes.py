"""
API routes for the inference service.

This module defines the API endpoints for the inference service.
"""

from fastapi import APIRouter, Depends, Request, status

from core.config import settings
from core.models import PredictionRequest, PredictionResponse
from services.predictor import process_prediction_request
from api.dependencies import ensure_prediction_service

# Create a router for the API endpoints
router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(request: Request) -> dict:
    """
    Health check endpoint.

    Returns:
        A dictionary with the status of the service.
    """
    # Check if model loader is available
    model_loader_available = hasattr(request.app.state, "model_loader")
    
    # Check if model and preprocessor are loaded
    model_loaded = False
    preprocessor_loaded = False
    
    if model_loader_available:
        model_loader = request.app.state.model_loader
        model_loaded = model_loader.model is not None
        preprocessor_loaded = model_loader.preprocessor is not None
    
    # Determine overall status
    if model_loaded and preprocessor_loaded:
        status_value = "healthy"
    elif model_loader_available:
        status_value = "degraded"
    else:
        status_value = "unhealthy"
    
    return {
        "status": status_value,
        "version": settings.API_VERSION,
        "model_loader_available": model_loader_available,
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(ensure_prediction_service)]
)
async def predict(
    request: PredictionRequest,
    response: PredictionResponse = Depends(process_prediction_request)
) -> PredictionResponse:
    """
    Make a prediction.

    Args:
        request: The prediction request data
        response: The prediction response from the service

    Returns:
        The prediction response
    """
    return response