"""
Dependency injection for the API routes.

This module provides dependencies that can be injected into API route handlers.
"""

from fastapi import Depends, HTTPException, Request, status

from inference_service.services.model_service import ModelService


async def get_model_service(request: Request) -> ModelService:
    """
    Get the model service from the application state.

    Args:
        request: The FastAPI request object

    Returns:
        The model service instance

    Raises:
        HTTPException: If the model service is not available
    """
    if not hasattr(request.app.state, "model_service"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service not available"
        )
    return request.app.state.model_service