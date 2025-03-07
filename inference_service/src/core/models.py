"""
Pydantic models for the inference service API.

This module defines the request and response models for the API endpoints.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


def generate_request_id() -> uuid.UUID:
    """
    Generate a unique request ID.

    Returns:
        A unique UUID for request tracking
    """
    return uuid.uuid4()


class EmploymentType(str, Enum):
    """Valid employment types."""
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    RETIRED = "retired"
    STUDENT = "student"


class MaritalStatus(str, Enum):
    """Valid marital statuses."""
    SINGLE = "single"
    MARRIED = "married"
    DIVORCED = "divorced"
    WIDOWED = "widowed"
    SEPARATED = "separated"


class DeviceType(str, Enum):
    """Valid device types."""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"


class ReferralSource(str, Enum):
    """Valid referral sources."""
    SEARCH = "search"
    SOCIAL = "social"
    EMAIL = "email"
    FRIEND = "friend"
    ADVERTISEMENT = "advertisement"
    OTHER = "other"


class PredictionRequest(BaseModel):
    """
    Model for prediction request data.

    This model validates the input data for the prediction endpoint.
    """
    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="User's age in years",
        examples=[35]
    )
    income: float = Field(
        ...,
        gt=0,
        description="User's annual income",
        examples=[50000.0]
    )
    employment_type: EmploymentType = Field(
        ...,
        description="User's employment status",
        examples=[EmploymentType.EMPLOYED]
    )
    marital_status: MaritalStatus = Field(
        ...,
        description="User's marital status",
        examples=[MaritalStatus.MARRIED]
    )
    time_spent_on_platform: float = Field(
        ...,
        gt=0,
        description="Time spent on platform in minutes",
        examples=[45.5]
    )
    number_of_sessions: int = Field(
        ...,
        gt=0,
        description="Number of sessions on the platform",
        examples=[3]
    )
    fields_filled_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of form fields completed",
        examples=[75.0]
    )
    previous_year_filing: bool = Field(
        ...,
        description="Whether the user filed taxes last year",
        examples=[True]
    )
    device_type: DeviceType = Field(
        ...,
        description="Type of device used",
        examples=[DeviceType.MOBILE]
    )
    referral_source: ReferralSource = Field(
        ...,
        description="How the user was referred to the platform",
        examples=[ReferralSource.SEARCH]
    )

    class Config:
        """Configuration for the PredictionRequest model."""
        schema_extra = {
            "example": {
                "age": 35,
                "income": 50000.0,
                "employment_type": "employed",
                "marital_status": "married",
                "time_spent_on_platform": 45.5,
                "number_of_sessions": 3,
                "fields_filled_percentage": 75.0,
                "previous_year_filing": True,
                "device_type": "mobile",
                "referral_source": "search"
            }
        }


class PredictionResponse(BaseModel):
    """
    Model for prediction response data.

    This model defines the structure of the prediction endpoint response.
    """
    prediction: bool = Field(
        ...,
        description="Prediction of whether the user will complete filing",
        examples=[True]
    )
    completion_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of completion",
        examples=[0.85]
    )
    request_id: uuid.UUID = Field(
        default_factory=generate_request_id,
        description="Unique identifier for the request"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the prediction"
    )

    @validator("completion_probability")
    def validate_probability(cls, v: float) -> float:
        """Validate that the probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Completion probability must be between 0 and 1")
        return round(v, 4)  # Round to 4 decimal places for readability

    class Config:
        """Configuration for the PredictionResponse model."""
        schema_extra = {
            "example": {
                "prediction": True,
                "completion_probability": 0.85,
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2023-04-01T12:00:00Z"
            }
        }
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class ErrorCode(str, Enum):
    """Error codes for the API."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    """
    Model for error response data.

    This model defines the structure of error responses.
    """
    error_code: ErrorCode = Field(
        ...,
        description="Error code",
        examples=[ErrorCode.VALIDATION_ERROR]
    )
    message: str = Field(
        ...,
        description="Error message",
        examples=["Invalid input data"]
    )
    request_id: Optional[uuid.UUID] = Field(
        None,
        description="Unique identifier for the request"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the error"
    )

    class Config:
        """Configuration for the ErrorResponse model."""
        schema_extra = {
            "example": {
                "error_code": "VALIDATION_ERROR",
                "message": "Invalid input data: age must be at least 18",
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2023-04-01T12:00:00Z"
            }
        }
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }