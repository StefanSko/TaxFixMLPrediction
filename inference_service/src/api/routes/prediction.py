"""
Prediction endpoints for the inference API.

This module defines the API endpoints for making predictions.
"""

import logging
import time
import json
from typing import Dict, Any

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from core.models import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    ErrorCode,
    generate_request_id
)
from core.security import api_key_auth
from services.predictor import get_prediction_service, PredictionService
from api.dependencies import ensure_prediction_service

# Create a router for prediction endpoints
router = APIRouter(
    prefix="/api/v1",
    tags=["prediction"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": ErrorResponse,
            "description": "Authentication failed"
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ErrorResponse,
            "description": "Validation or prediction error"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal server error"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Service unavailable"
        }
    }
)

logger = logging.getLogger(__name__)


# Custom JSON encoder to handle UUID and datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        import uuid
        from datetime import datetime
        
        if isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Make a prediction",
    description="Predict whether a user will complete their tax filing based on user data",
    dependencies=[Depends(ensure_prediction_service), api_key_auth]
)
async def predict(
    request: Request,
    prediction_request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Make a prediction based on user data.

    Args:
        request: FastAPI request object
        prediction_request: Validated prediction request data
        prediction_service: Prediction service instance

    Returns:
        Prediction response with completion prediction and probability

    Raises:
        HTTPException: If prediction processing fails
    """
    # Generate request ID
    request_id = generate_request_id()
    
    # Start timing
    start_time = time.time()
    
    # Log incoming request (excluding sensitive data)
    log_data = {
        "request_id": str(request_id),
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent", "unknown"),
        "endpoint": request.url.path,
        "method": request.method
    }
    logger.info(f"Received prediction request: {log_data}")
    
    try:
        # Convert Pydantic model to dictionary
        input_data = prediction_request.dict()
        
        # Process the request
        result = prediction_service.process_request(input_data)
        
        # Create response
        response = PredictionResponse(
            prediction=result["prediction"],
            completion_probability=result["completion_probability"],
            request_id=request_id
        )
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Prediction completed: request_id={request_id}, "
            f"prediction={response.prediction}, "
            f"probability={response.completion_probability:.4f}, "
            f"processing_time={processing_time:.4f}s"
        )
        
        return response
        
    except ValueError as e:
        # Log error
        logger.error(f"Prediction error: request_id={request_id}, error={str(e)}")
        
        # Create error response
        error_response = ErrorResponse(
            error_code=ErrorCode.PREDICTION_ERROR,
            message=str(e),
            request_id=request_id
        )
        
        # Convert to dict and manually serialize UUID
        error_dict = error_response.dict()
        if error_dict["request_id"]:
            error_dict["request_id"] = str(error_dict["request_id"])
        if error_dict["timestamp"]:
            error_dict["timestamp"] = error_dict["timestamp"].isoformat()
        
        # Return error response
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_dict
        )
    except Exception as e:
        # Log unexpected error
        logger.exception(f"Unexpected error during prediction: request_id={request_id}, error={str(e)}")
        
        # Create error response
        error_response = ErrorResponse(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="An unexpected error occurred during prediction",
            request_id=request_id
        )
        
        # Convert to dict and manually serialize UUID
        error_dict = error_response.dict()
        if error_dict["request_id"]:
            error_dict["request_id"] = str(error_dict["request_id"])
        if error_dict["timestamp"]:
            error_dict["timestamp"] = error_dict["timestamp"].isoformat()
        
        # Return error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_dict
        )


@router.post(
    "/batch-predict",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Batch prediction (not implemented)",
    description="This endpoint will support batch predictions in the future",
    dependencies=[api_key_auth]
)
async def batch_predict() -> Dict[str, Any]:
    """
    Placeholder for batch prediction endpoint.

    Returns:
        Error response indicating the endpoint is not implemented
    """
    return {
        "error_code": ErrorCode.NOT_FOUND,
        "message": "Batch prediction is not implemented yet"
    }
