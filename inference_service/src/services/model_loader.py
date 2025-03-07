"""
Model loader for the inference service.

This module provides functionality for loading ML models and preprocessors
from different storage backends (local filesystem or S3).
"""

import logging
import os
import time
from datetime import datetime
from threading import Thread
from typing import Any, Dict, Optional, Tuple

import joblib
from fastapi import Depends, Request

logger = logging.getLogger(__name__)


class StorageBackend:
    """Base class for storage backends."""

    def exists(self, path: str) -> bool:
        """
        Check if a file exists.

        Args:
            path: Path to the file

        Returns:
            True if the file exists, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_modified_time(self, path: str) -> datetime:
        """
        Get the last modified time of a file.

        Args:
            path: Path to the file

        Returns:
            Datetime object representing the last modified time
        """
        raise NotImplementedError("Subclasses must implement this method")

    def load(self, path: str) -> Any:
        """
        Load an object from a file.

        Args:
            path: Path to the file

        Returns:
            Loaded object
        """
        raise NotImplementedError("Subclasses must implement this method")


class LocalStorageBackend(StorageBackend):
    """Storage backend for local filesystem."""

    def exists(self, path: str) -> bool:
        """
        Check if a file exists in local storage.

        Args:
            path: Path to the file

        Returns:
            True if the file exists, False otherwise
        """
        return os.path.exists(path)

    def get_modified_time(self, path: str) -> datetime:
        """
        Get the last modified time of a file in local storage.

        Args:
            path: Path to the file

        Returns:
            Datetime object representing the last modified time
        """
        return datetime.fromtimestamp(os.path.getmtime(path))

    def load(self, path: str) -> Any:
        """
        Load an object from a file in local storage.

        Args:
            path: Path to the file

        Returns:
            Loaded object

        Raises:
            FileNotFoundError: If the file does not exist
            Exception: If loading fails
        """
        if not self.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Failed to load file from {path}: {str(e)}")
            raise


class S3StorageBackend(StorageBackend):
    """Storage backend for S3."""

    def __init__(self, bucket: Optional[str] = None):
        """
        Initialize the S3 storage backend.

        Args:
            bucket: Optional S3 bucket name (can be extracted from path)
        """
        try:
            import boto3
            self.s3 = boto3.client('s3')
            self.bucket = bucket
        except ImportError:
            logger.error("boto3 is required for S3 storage. Install with 'pip install boto3'")
            raise

    def _parse_s3_path(self, path: str) -> Tuple[str, str]:
        """
        Parse an S3 path into bucket and key.

        Args:
            path: S3 path (s3://bucket/key)

        Returns:
            Tuple of (bucket, key)
        """
        if path.startswith('s3://'):
            path = path[5:]
        
        parts = path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        return bucket, key

    def exists(self, path: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            path: S3 path (s3://bucket/key)

        Returns:
            True if the file exists, False otherwise
        """
        bucket, key = self._parse_s3_path(path)
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def get_modified_time(self, path: str) -> datetime:
        """
        Get the last modified time of a file in S3.

        Args:
            path: S3 path (s3://bucket/key)

        Returns:
            Datetime object representing the last modified time
        """
        bucket, key = self._parse_s3_path(path)
        response = self.s3.head_object(Bucket=bucket, Key=key)
        return response['LastModified']

    def load(self, path: str) -> Any:
        """
        Load an object from a file in S3.

        Args:
            path: S3 path (s3://bucket/key)

        Returns:
            Loaded object

        Raises:
            FileNotFoundError: If the file does not exist
            Exception: If loading fails
        """
        import tempfile
        
        bucket, key = self._parse_s3_path(path)
        
        if not self.exists(path):
            raise FileNotFoundError(f"File not found in S3: {path}")
        
        try:
            with tempfile.NamedTemporaryFile() as temp_file:
                self.s3.download_file(bucket, key, temp_file.name)
                return joblib.load(temp_file.name)
        except Exception as e:
            logger.error(f"Failed to load file from S3 {path}: {str(e)}")
            raise


class ModelLoader:
    """
    Service for loading ML models and preprocessors.

    This class handles loading models and preprocessors from different storage backends,
    with support for caching and reloading.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model loader.

        Args:
            config: Configuration dictionary with model paths and settings
        """
        self.config = config
        self.model_path = config.get('MODEL_PATH', '')
        self.preprocessor_path = config.get('PREPROCESSOR_PATH', '')
        self.metadata_path = config.get('METADATA_PATH', '')
        
        # Initialize storage backends
        self._init_storage_backends()
        
        # Initialize cache
        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.last_loaded = None
        self.last_checked = None

    def _init_storage_backends(self) -> None:
        """Initialize storage backends based on paths."""
        # Determine storage backend for model
        if self.model_path.startswith('s3://'):
            self.model_storage = S3StorageBackend()
        else:
            self.model_storage = LocalStorageBackend()
        
        # Determine storage backend for preprocessor
        if self.preprocessor_path.startswith('s3://'):
            self.preprocessor_storage = S3StorageBackend()
        else:
            self.preprocessor_storage = LocalStorageBackend()
        
        # Determine storage backend for metadata
        if self.metadata_path and self.metadata_path.startswith('s3://'):
            self.metadata_storage = S3StorageBackend()
        elif self.metadata_path:
            self.metadata_storage = LocalStorageBackend()
        else:
            self.metadata_storage = None

    def load_model(self) -> Any:
        """
        Load the trained model.

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If the model file is not found
            ValueError: If the model cannot be loaded
        """
        logger.info(f"Loading model from {self.model_path}")
        try:
            model = self.model_storage.load(self.model_path)
            logger.info("Model loaded successfully")
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")

    def load_preprocessor(self) -> Any:
        """
        Load the preprocessor.

        Returns:
            Loaded preprocessor object

        Raises:
            FileNotFoundError: If the preprocessor file is not found
            ValueError: If the preprocessor cannot be loaded
        """
        logger.info(f"Loading preprocessor from {self.preprocessor_path}")
        try:
            preprocessor = self.preprocessor_storage.load(self.preprocessor_path)
            logger.info("Preprocessor loaded successfully")
            return preprocessor
        except FileNotFoundError:
            logger.error(f"Preprocessor file not found: {self.preprocessor_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {str(e)}")
            raise ValueError(f"Failed to load preprocessor: {str(e)}")

    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary containing model metadata
        """
        if not self.metadata_path or not self.metadata_storage:
            logger.warning("No metadata path configured")
            return {}
        
        try:
            metadata = self.metadata_storage.load(self.metadata_path)
            return metadata if isinstance(metadata, dict) else {}
        except Exception as e:
            logger.warning(f"Failed to load metadata: {str(e)}")
            return {}

    def reload_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Reload model, preprocessor, and metadata.

        Returns:
            Tuple of (model, preprocessor, metadata)
        """
        logger.info("Reloading model and preprocessor")
        model = self.load_model()
        preprocessor = self.load_preprocessor()
        metadata = self.get_model_metadata()
        
        # Update cache
        self.model = model
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.last_loaded = datetime.now()
        self.last_checked = datetime.now()
        
        return model, preprocessor, metadata

    def check_for_updates(self) -> bool:
        """
        Check if model or preprocessor files have been updated.

        Returns:
            True if updates are available, False otherwise
        """
        if not self.last_loaded:
            return True
        
        try:
            model_modified = self.model_storage.get_modified_time(self.model_path)
            preprocessor_modified = self.preprocessor_storage.get_modified_time(self.preprocessor_path)
            
            # Check if either file has been modified since last load
            return (model_modified > self.last_loaded or 
                    preprocessor_modified > self.last_loaded)
        except Exception as e:
            logger.warning(f"Failed to check for updates: {str(e)}")
            return False

    def get_model(self) -> Any:
        """
        Get the cached model, loading it if necessary.

        Returns:
            Loaded model object
        """
        if self.model is None:
            self.reload_model()
        return self.model

    def get_preprocessor(self) -> Any:
        """
        Get the cached preprocessor, loading it if necessary.

        Returns:
            Loaded preprocessor object
        """
        if self.preprocessor is None:
            self.reload_model()
        return self.preprocessor


def initialize_model_loader(config: Dict[str, Any]) -> ModelLoader:
    """
    Initialize the model loader.

    Args:
        config: Configuration dictionary with model paths and settings

    Returns:
        Initialized ModelLoader instance
    """
    model_loader = ModelLoader(config)
    
    try:
        # Load model and preprocessor
        model_loader.reload_model()
        logger.info("Model loader initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model loader: {str(e)}")
        # Don't raise the exception - we'll handle missing models gracefully
    
    return model_loader


def check_model_updates(model_loader: ModelLoader, interval_seconds: int = 300) -> None:
    """
    Background task to check for model updates.

    Args:
        model_loader: ModelLoader instance
        interval_seconds: Interval between checks in seconds
    """
    logger.info(f"Starting model update checker (interval: {interval_seconds}s)")
    
    while True:
        time.sleep(interval_seconds)
        
        try:
            logger.debug("Checking for model updates")
            if model_loader.check_for_updates():
                logger.info("Model updates detected, reloading")
                model_loader.reload_model()
                logger.info("Model reloaded successfully")
            else:
                logger.debug("No model updates detected")
                model_loader.last_checked = datetime.now()
        except Exception as e:
            logger.error(f"Error checking for model updates: {str(e)}")


def start_update_checker(model_loader: ModelLoader, interval_seconds: int = 300) -> Thread:
    """
    Start the model update checker in a background thread.

    Args:
        model_loader: ModelLoader instance
        interval_seconds: Interval between checks in seconds

    Returns:
        Background thread object
    """
    update_thread = Thread(
        target=check_model_updates,
        args=(model_loader, interval_seconds),
        daemon=True
    )
    update_thread.start()
    return update_thread


# FastAPI dependency functions
def get_model_loader(request: Request) -> ModelLoader:
    """
    FastAPI dependency to get the model loader.

    Args:
        request: FastAPI request object

    Returns:
        ModelLoader instance
    """
    if not hasattr(request.app.state, "model_loader"):
        raise RuntimeError("Model loader not initialized")
    return request.app.state.model_loader


def get_model(model_loader: ModelLoader = Depends(get_model_loader)) -> Any:
    """
    FastAPI dependency to get the model.

    Args:
        model_loader: ModelLoader instance

    Returns:
        Loaded model object
    """
    return model_loader.get_model()


def get_preprocessor(model_loader: ModelLoader = Depends(get_model_loader)) -> Any:
    """
    FastAPI dependency to get the preprocessor.

    Args:
        model_loader: ModelLoader instance

    Returns:
        Loaded preprocessor object
    """
    return model_loader.get_preprocessor() 