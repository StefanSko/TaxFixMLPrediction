"""
Server configuration for the inference service.

This module provides utilities for creating and configuring the FastAPI application.
"""

import logging
import os
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from core.config import get_settings
from services.model_loader import initialize_model_loader, start_update_checker
from utils.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config_path: Optional path to a YAML configuration file

    Returns:
        Configured FastAPI application
    """
    # Load settings from YAML if provided, otherwise use defaults
    if config_path is None:
        # Check for default config location
        default_config = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "config.yaml"
        )
        if os.path.exists(default_config):
            config_path = default_config

    settings = get_settings(config_path)

    # Configure logging
    configure_logging(
        level=settings.LOG_LEVEL,
        log_file=settings.LOG_FILE
    )

    # Create FastAPI app
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION,
        debug=settings.DEBUG
    )

    # Store settings in app state
    app.state.settings = settings

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=settings.CORS_HEADERS,
    )

    # Add error handling middleware
    @app.middleware("http")
    async def error_handling_middleware(request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            logger.exception(f"Unhandled exception: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )

    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize resources on startup."""
        logger.info("Starting inference service")

        # Initialize model loader
        try:
            config = {
                'MODEL_PATH': settings.MODEL_PATH,
                'PREPROCESSOR_PATH': settings.PREPROCESSOR_PATH,
                # Add metadata path if available in settings
                'METADATA_PATH': getattr(settings, 'METADATA_PATH', None)
            }
            
            model_loader = initialize_model_loader(config)
            
            # Store model loader in app state for dependency injection
            app.state.model_loader = model_loader
            
            # Start background thread to check for model updates
            update_interval = getattr(settings, 'MODEL_UPDATE_INTERVAL', 300)  # Default: 5 minutes
            app.state.update_thread = start_update_checker(model_loader, update_interval)
            
            logger.info("Model loader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model loader: {str(e)}")
            # Continue running even if model loading fails
            # This allows the health endpoint to report the issue

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Clean up resources on shutdown."""
        logger.info("Shutting down inference service")
        # No need to explicitly stop the update thread as it's a daemon thread

    # Include routers
    app.include_router(router)

    return app


def run_server(config_path: Optional[str] = None) -> None:
    """
    Run the server using uvicorn.

    Args:
        config_path: Optional path to a YAML configuration file
    """
    # Load settings
    settings = get_settings(config_path)

    # Run server
    uvicorn.run(
        "inference_service.api.server:create_app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        factory=True
    )