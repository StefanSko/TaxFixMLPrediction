"""
Error handling for the inference service.

This module provides custom exception classes, exception handlers, and error logging
utilities for the inference service.
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Type, Union

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from core.models import ErrorCode, ErrorResponse, generate_request_id

logger = logging.getLogger(__name__)


# Custom exception classes
class PredictionError(Exception):
    """Base class for prediction-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.PREDICTION_ERROR,
        status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the prediction error.

        Args:
            message: Error message
            error_code: Error code for the response
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class InvalidInputError(PredictionError):
    """Error for invalid input data."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the invalid input error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class ModelLoadingError(PredictionError):
    """Error when model loading fails."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the model loading error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_ERROR,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )


class PreprocessingError(PredictionError):
    """Error during data preprocessing."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the preprocessing error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.PREDICTION_ERROR,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class InferenceError(PredictionError):
    """Error during model inference."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the inference error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.PREDICTION_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


# Error logging functions
def log_error(
    exception: Exception,
    request_id: Optional[Union[str, uuid.UUID]] = None,
    **additional_info
) -> None:
    """
    Log an error with context information.

    Args:
        exception: The exception to log
        request_id: Optional request ID for context
        additional_info: Additional context information
    """
    error_context = {
        "request_id": str(request_id) if request_id else None,
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "traceback": traceback.format_exc(),
        **additional_info
    }
    
    logger.error(
        f"Error: {error_context['error_type']}: {error_context['error_message']}",
        extra={"error_context": error_context}
    )


def log_warning(
    message: str,
    request_id: Optional[Union[str, uuid.UUID]] = None,
    **additional_info
) -> None:
    """
    Log a warning with context information.

    Args:
        message: Warning message
        request_id: Optional request ID for context
        additional_info: Additional context information
    """
    warning_context = {
        "request_id": str(request_id) if request_id else None,
        "warning_message": message,
        **additional_info
    }
    
    logger.warning(
        f"Warning: {message}",
        extra={"warning_context": warning_context}
    )


# Exception handlers
async def prediction_error_handler(
    request: Request,
    exc: PredictionError
) -> JSONResponse:
    """
    Handle PredictionError exceptions.

    Args:
        request: FastAPI request
        exc: PredictionError exception

    Returns:
        JSON response with error details
    """
    # Get or generate request ID
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    
    # Log the error
    log_error(
        exc,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        client_host=request.client.host if request.client else None,
        details=exc.details
    )
    
    # Create error response
    error_response = ErrorResponse(
        error_code=exc.error_code,
        message=exc.message,
        request_id=request_id
    )
    
    # Convert to dict and ensure UUID is serialized as string
    error_dict = error_response.dict()
    error_dict["request_id"] = str(error_dict["request_id"]) if error_dict["request_id"] else None
    if error_dict["timestamp"]:
        error_dict["timestamp"] = error_dict["timestamp"].isoformat()
    
    # Include additional details if available
    if exc.details:
        error_dict["details"] = exc.details
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_dict
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle RequestValidationError exceptions.

    Args:
        request: FastAPI request
        exc: RequestValidationError exception

    Returns:
        JSON response with error details
    """
    # Get or generate request ID
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    
    # Extract validation errors
    errors = exc.errors()
    error_details = []
    
    for error in errors:
        error_details.append({
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
            "type": error.get("type", "")
        })
    
    # Create error message
    if error_details:
        message = f"Validation error: {error_details[0]['msg']}"
    else:
        message = "Validation error"
    
    # Log the error
    log_error(
        exc,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        client_host=request.client.host if request.client else None,
        validation_errors=error_details
    )
    
    # Create error response
    error_response = ErrorResponse(
        error_code=ErrorCode.VALIDATION_ERROR,
        message=message,
        request_id=request_id
    )
    
    # Convert to dict and ensure UUID is serialized as string
    error_dict = error_response.dict()
    error_dict["request_id"] = str(error_dict["request_id"]) if error_dict["request_id"] else None
    if error_dict["timestamp"]:
        error_dict["timestamp"] = error_dict["timestamp"].isoformat()
    
    # Include validation errors
    error_dict["details"] = {"errors": error_details}
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_dict
    )


async def http_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle HTTPException exceptions.

    Args:
        request: FastAPI request
        exc: HTTPException exception

    Returns:
        JSON response with error details
    """
    # Get or generate request ID
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    
    # Determine error code based on status code
    status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    if status_code == status.HTTP_401_UNAUTHORIZED:
        error_code = ErrorCode.UNAUTHORIZED
    elif status_code == status.HTTP_404_NOT_FOUND:
        error_code = ErrorCode.NOT_FOUND
    elif status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
        error_code = ErrorCode.VALIDATION_ERROR
    elif status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        error_code = ErrorCode.MODEL_ERROR
    else:
        error_code = ErrorCode.INTERNAL_ERROR
    
    # Get error message
    message = str(exc)
    
    # Log the error
    log_error(
        exc,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        client_host=request.client.host if request.client else None,
        status_code=status_code
    )
    
    # Create error response
    error_response = ErrorResponse(
        error_code=error_code,
        message=message,
        request_id=request_id
    )
    
    # Convert to dict and ensure UUID is serialized as string
    error_dict = error_response.dict()
    error_dict["request_id"] = str(error_dict["request_id"]) if error_dict["request_id"] else None
    if error_dict["timestamp"]:
        error_dict["timestamp"] = error_dict["timestamp"].isoformat()
    
    # Include headers if available
    headers = getattr(exc, "headers", None)
    
    return JSONResponse(
        status_code=status_code,
        content=error_dict,
        headers=headers
    )


async def pydantic_validation_error_handler(
    request: Request,
    exc: ValidationError
) -> JSONResponse:
    """
    Handle Pydantic ValidationError exceptions.

    Args:
        request: FastAPI request
        exc: ValidationError exception

    Returns:
        JSON response with error details
    """
    # Get or generate request ID
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    
    # Extract validation errors
    errors = exc.errors()
    error_details = []
    
    for error in errors:
        error_details.append({
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
            "type": error.get("type", "")
        })
    
    # Create error message
    if error_details:
        message = f"Validation error: {error_details[0]['msg']}"
    else:
        message = "Validation error"
    
    # Log the error
    log_error(
        exc,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        client_host=request.client.host if request.client else None,
        validation_errors=error_details
    )
    
    # Create error response
    error_response = ErrorResponse(
        error_code=ErrorCode.VALIDATION_ERROR,
        message=message,
        request_id=request_id
    )
    
    # Convert to dict and ensure UUID is serialized as string
    error_dict = error_response.dict()
    error_dict["request_id"] = str(error_dict["request_id"]) if error_dict["request_id"] else None
    if error_dict["timestamp"]:
        error_dict["timestamp"] = error_dict["timestamp"].isoformat()
    
    # Include validation errors
    error_dict["details"] = {"errors": error_details}
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_dict
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle generic exceptions.

    Args:
        request: FastAPI request
        exc: Exception

    Returns:
        JSON response with error details
    """
    # Get or generate request ID
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    
    # Log the error
    log_error(
        exc,
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        client_host=request.client.host if request.client else None
    )
    
    # Create error response
    error_response = ErrorResponse(
        error_code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred",
        request_id=request_id
    )
    
    # Convert to dict and ensure UUID is serialized as string
    error_dict = error_response.dict()
    error_dict["request_id"] = str(error_dict["request_id"]) if error_dict["request_id"] else None
    if error_dict["timestamp"]:
        error_dict["timestamp"] = error_dict["timestamp"].isoformat()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_dict
    )


def add_exception_handlers(app: FastAPI) -> None:
    """
    Register exception handlers with the FastAPI application.

    Args:
        app: FastAPI application
    """
    # Register handlers for custom exceptions
    app.add_exception_handler(PredictionError, prediction_error_handler)
    app.add_exception_handler(InvalidInputError, prediction_error_handler)
    app.add_exception_handler(ModelLoadingError, prediction_error_handler)
    app.add_exception_handler(PreprocessingError, prediction_error_handler)
    app.add_exception_handler(InferenceError, prediction_error_handler)
    
    # Register handlers for built-in exceptions
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_error_handler)
    
    # Register handler for HTTPException
    from fastapi.exceptions import HTTPException
    app.add_exception_handler(HTTPException, http_exception_handler)
    
    # Register handler for generic exceptions
    app.add_exception_handler(Exception, generic_exception_handler) 