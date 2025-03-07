"""
API routes for the inference service.

This module defines the API endpoints for the inference service.
"""

from fastapi import APIRouter, Depends, Request, status

from core.config import settings
from api.dependencies import ensure_prediction_service
from api.routes.prediction import router as prediction_router

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


# Include the prediction router
router.include_router(prediction_router)