"""
Logging configuration for the inference service.

This module provides utilities for configuring logging with structured output
and request ID tracking.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Union

from core.config import settings


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    This formatter outputs log records as JSON objects with standardized fields.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON string representation of the log record
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add request_id if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Add error context if available
        if hasattr(record, "error_context"):
            log_data["error_context"] = record.error_context
        
        # Add warning context if available
        if hasattr(record, "warning_context"):
            log_data["warning_context"] = record.warning_context
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_data)


class RequestIdFilter(logging.Filter):
    """
    Filter that adds request ID to log records.
    
    This filter adds the request ID from the current context to log records.
    """
    
    def __init__(self, request_id: Optional[str] = None):
        """
        Initialize the filter with an optional request ID.
        
        Args:
            request_id: Optional default request ID
        """
        super().__init__()
        self.request_id = request_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request ID to the log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to include the record in the log output
        """
        if not hasattr(record, "request_id"):
            record.request_id = self.request_id
        return True


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_json: bool = None,
    request_id: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        use_json: Whether to use JSON formatting (defaults to True in production)
        request_id: Optional request ID to include in all logs
    """
    # Determine log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Determine whether to use JSON formatting
    if use_json is None:
        use_json = not settings.DEBUG
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create file handler if log file is specified
    file_handler = None
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
        except Exception as e:
            print(f"Warning: Could not create log file handler: {str(e)}")
    
    # Configure formatters
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Set formatters on handlers
    console_handler.setFormatter(formatter)
    if file_handler:
        file_handler.setFormatter(formatter)
    
    # Add request ID filter if provided
    if request_id:
        request_id_filter = RequestIdFilter(request_id)
        console_handler.addFilter(request_id_filter)
        if file_handler:
            file_handler.addFilter(request_id_filter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


def get_logger(name: str, request_id: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with request ID context.
    
    Args:
        name: Logger name
        request_id: Optional request ID to include in logs
        
    Returns:
        Logger with request ID context
    """
    logger = logging.getLogger(name)
    
    if request_id:
        # Add request ID filter to the logger
        for handler in logger.handlers:
            handler.addFilter(RequestIdFilter(request_id))
    
    return logger