"""
Health check endpoints for the inference API.

This module defines the API endpoints for health checking.
"""

from fastapi import APIRouter, Request, status

from core.config import settings

# Create a router for health check endpoints
router = APIRouter(
    tags=["health"]
)


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of the service"
)
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
