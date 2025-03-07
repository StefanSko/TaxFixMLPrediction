"""
Dependency injection for the API routes.

This module provides dependencies that can be injected into API route handlers.
"""

from fastapi import HTTPException, Request, status



async def ensure_model_loader(request: Request) -> None:
    """
    Ensure that the model loader is available in the application state.

    Args:
        request: The FastAPI request object

    Raises:
        HTTPException: If the model loader is not available
    """
    if not hasattr(request.app.state, "model_loader"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model loader not available"
        )


async def get_app_model_loader(request: Request):
    """
    Get the model loader from the application state.

    Args:
        request: The FastAPI request object

    Returns:
        The model loader instance

    Raises:
        HTTPException: If the model loader is not available
    """
    await ensure_model_loader(request)
    return request.app.state.model_loader


async def ensure_prediction_service(request: Request) -> None:
    """
    Ensure that the prediction service can be created.

    This checks if the model and preprocessor are available.

    Args:
        request: The FastAPI request object

    Raises:
        HTTPException: If the model or preprocessor is not available
    """
    await ensure_model_loader(request)
    
    model_loader = request.app.state.model_loader
    
    # Check if model is loaded
    if model_loader.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Check if preprocessor is loaded
    if model_loader.preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Preprocessor not loaded"
        )