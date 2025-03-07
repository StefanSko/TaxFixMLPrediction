"""
Security utilities for the inference service API.

This module provides functionality for API key authentication and validation.
"""

import logging
import re
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import APIKeyHeader

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Define API key header scheme for OpenAPI documentation
API_KEY_HEADER = "X-API-Key"
api_key_header_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def get_api_key(
    api_key_header: Optional[str] = Header(None, alias=API_KEY_HEADER)
) -> Optional[str]:
    """
    Extract and validate the API key from the request header.

    Args:
        api_key_header: The API key from the request header

    Returns:
        The validated API key

    Raises:
        HTTPException: If the API key is missing or has an invalid format
    """
    if api_key_header is None:
        logger.warning("Missing API key in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    # Validate API key format (alphanumeric string with at least 16 characters)
    if not re.match(r"^[a-zA-Z0-9_-]{16,}$", api_key_header):
        logger.warning("Invalid API key format")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    return api_key_header


def validate_api_key(
    api_key: str = Depends(get_api_key),
    settings: Settings = Depends(get_settings)
) -> bool:
    """
    Validate the provided API key against the configured API key.

    Args:
        api_key: The API key to validate
        settings: Application settings

    Returns:
        True if the API key is valid

    Raises:
        HTTPException: If the API key is invalid or API key authentication is not configured
    """
    # Check if API key authentication is enabled
    if not settings.API_KEY:
        logger.warning("API key authentication is not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key authentication is not configured",
        )

    # Compare the provided API key with the configured API key
    if api_key != settings.API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    logger.debug("API key validated successfully")
    return True


# Create a dependency for API key authentication
api_key_auth = Depends(validate_api_key)
