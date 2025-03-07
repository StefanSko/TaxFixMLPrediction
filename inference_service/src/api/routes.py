"""
API routes for the inference service.

This module defines the API endpoints for the inference service.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from core.config import settings

# Create a router for the API endpoints
router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(request: Request) -> dict:
    """
    Health check endpoint.

    Returns:
        A dictionary with the status of the service.
    """
    # Check if model service is available
    model_service_available = hasattr(request.app.state, "model_service")

    return {
        "status": "healthy" if model_service_available else "degraded",
        "version": settings.API_VERSION,
        "model_loaded": model_service_available
    }