"""
API routes package.

This package contains all the API route modules.
"""

from fastapi import APIRouter

from api.routes.health import router as health_router
from api.routes.prediction import router as prediction_router

# Create a main router
router = APIRouter()

# Include all route modules
router.include_router(health_router)
router.include_router(prediction_router)
