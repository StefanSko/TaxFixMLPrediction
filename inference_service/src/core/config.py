"""
Configuration settings for the inference service.

This module provides YAML-based configuration using Pydantic's BaseSettings.
"""

import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Configuration settings for the inference service.

    Settings are loaded from YAML configuration files with fallback to environment variables.
    """
    # API Settings
    API_TITLE: str = "Taxfix Completion Prediction API"
    API_DESCRIPTION: str = "API for predicting whether a user will complete their tax filing"
    API_VERSION: str = "0.1.0"
    HOST: str = Field(default="0.0.0.0", env="API_HOST")
    PORT: int = Field(default=8000, env="API_PORT")
    DEBUG: bool = Field(default=False, env="API_DEBUG")

    # CORS Settings
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    CORS_HEADERS: List[str] = Field(default=["*"], env="CORS_HEADERS")

    # Model Settings
    MODEL_PATH: str = Field(
        default="output/models/tax_filing_classifier.joblib",
        env="MODEL_PATH"
    )
    PREPROCESSOR_PATH: str = Field(
        default="output/models/preprocessor.joblib",
        env="PREPROCESSOR_PATH"
    )
    METADATA_PATH: Optional[str] = Field(
        default=None,
        env="METADATA_PATH"
    )
    MODEL_UPDATE_INTERVAL: int = Field(
        default=300,  # 5 minutes
        env="MODEL_UPDATE_INTERVAL"
    )

    # Authentication Settings
    API_KEY: Optional[str] = Field(
        default=None,
        env="API_KEY",
        description="API key for authentication. If not set, API key authentication is disabled."
    )
    API_KEY_HEADER: str = Field(
        default="X-API-Key",
        env="API_KEY_HEADER",
        description="HTTP header name for API key"
    )

    # Logging Settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")

    @validator("MODEL_PATH", "PREPROCESSOR_PATH")
    def validate_paths(cls, v: str) -> str:
        """Validate that the path exists."""
        path = Path(v)
        if not path.exists():
            # Don't raise error here, just warn - we'll handle this during startup
            print(f"Warning: Path {path} does not exist")
        return v

    @validator("API_KEY")
    def generate_default_api_key_if_debug(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Generate a default API key if in debug mode and no API key is provided."""
        if v is None and values.get("DEBUG", False):
            # Generate a random API key for development
            return f"dev-{secrets.token_hex(16)}"
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration values
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Get settings from YAML file or environment variables.

    Args:
        config_path: Optional path to a YAML configuration file

    Returns:
        Settings object with configuration values
    """
    # Start with default settings
    settings_dict = {}

    # Load from YAML file if provided
    if config_path:
        yaml_config = load_yaml_config(config_path)
        settings_dict.update(yaml_config)

    # Create settings object (environment variables will override YAML values)
    return Settings(**settings_dict)


# Default settings object (can be overridden using get_settings with a config path)
settings = Settings()