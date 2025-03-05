"""
Storage module for the ML pipeline.
This module provides functionality to save and load models, preprocessors,
and metadata to/from different storage backends.
"""

import json
from typing import Dict, Any, Union
from pathlib import Path
import joblib
import boto3
from botocore.exceptions import ClientError


class ModelStorage:
    """
    Base class for model storage operations.
    Provides functionality to save and load models, preprocessors, and metadata
    to/from local storage.
    """

    def __init__(self, base_path: str) -> None:
        """
        Initialize the storage with a base path.

        Args:
            base_path: Base directory path for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different types of artifacts
        self.models_path = self.base_path / "models"
        self.preprocessors_path = self.base_path / "preprocessors"
        self.metadata_path = self.base_path / "metadata"

        self.models_path.mkdir(exist_ok=True)
        self.preprocessors_path.mkdir(exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)

    def save_model(self, model: Any, model_name: str) -> str:
        """
        Save a model to a file.

        Args:
            model: The model to save
            model_name: Name to use for the saved model

        Returns:
            Path where the model was saved
        """
        if not model_name.endswith('.joblib'):
            model_name = f"{model_name}.joblib"

        model_path = self.models_path / model_name

        try:
            joblib.dump(model, model_path)
            return str(model_path)
        except Exception as e:
            raise IOError(f"Failed to save model {model_name}: {str(e)}")

    def load_model(self, model_name: str) -> Any:
        """
        Load a model from a file.

        Args:
            model_name: Name of the model to load

        Returns:
            The loaded model
        """
        if not model_name.endswith('.joblib'):
            model_name = f"{model_name}.joblib"

        model_path = self.models_path / model_name

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            return joblib.load(model_path)
        except Exception as e:
            raise IOError(f"Failed to load model {model_name}: {str(e)}")

    def save_preprocessor(self, preprocessor: Any, preprocessor_name: str) -> str:
        """
        Save a preprocessor to a file.

        Args:
            preprocessor: The preprocessor to save
            preprocessor_name: Name to use for the saved preprocessor

        Returns:
            Path where the preprocessor was saved
        """
        if not preprocessor_name.endswith('.joblib'):
            preprocessor_name = f"{preprocessor_name}.joblib"

        preprocessor_path = self.preprocessors_path / preprocessor_name

        try:
            joblib.dump(preprocessor, preprocessor_path)
            return str(preprocessor_path)
        except Exception as e:
            raise IOError(f"Failed to save preprocessor {preprocessor_name}: {str(e)}")

    def load_preprocessor(self, preprocessor_name: str) -> Any:
        """
        Load a preprocessor from a file.

        Args:
            preprocessor_name: Name of the preprocessor to load

        Returns:
            The loaded preprocessor
        """
        if not preprocessor_name.endswith('.joblib'):
            preprocessor_name = f"{preprocessor_name}.joblib"

        preprocessor_path = self.preprocessors_path / preprocessor_name

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        try:
            return joblib.load(preprocessor_path)
        except Exception as e:
            raise IOError(f"Failed to load preprocessor {preprocessor_name}: {str(e)}")

    def save_metadata(self, metadata: Dict[str, Any], metadata_name: str) -> str:
        """
        Save metadata as JSON.

        Args:
            metadata: Dictionary containing metadata
            metadata_name: Name to use for the saved metadata

        Returns:
            Path where the metadata was saved
        """
        if not metadata_name.endswith('.json'):
            metadata_name = f"{metadata_name}.json"

        metadata_path = self.metadata_path / metadata_name

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            return str(metadata_path)
        except Exception as e:
            raise IOError(f"Failed to save metadata {metadata_name}: {str(e)}")

    def load_metadata(self, metadata_name: str) -> Dict[str, Any]:
        """
        Load metadata from JSON.

        Args:
            metadata_name: Name of the metadata to load

        Returns:
            Dictionary containing the loaded metadata
        """
        if not metadata_name.endswith('.json'):
            metadata_name = f"{metadata_name}.json"

        metadata_path = self.metadata_path / metadata_name

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load metadata {metadata_name}: {str(e)}")


class S3ModelStorage(ModelStorage):
    """
    S3 implementation of model storage.
    Provides functionality to save and load models, preprocessors, and metadata
    to/from an S3 bucket.
    """

    def __init__(self, bucket_name: str, base_prefix: str = 'models') -> None:
        """
        Initialize the S3 storage.

        Args:
            bucket_name: Name of the S3 bucket
            base_prefix: Base prefix (folder) in the bucket
        """
        self.bucket_name = bucket_name
        self.base_prefix = base_prefix.rstrip('/')

        # Define prefixes for different types of artifacts
        self.models_prefix = f"{self.base_prefix}/models"
        self.preprocessors_prefix = f"{self.base_prefix}/preprocessors"
        self.metadata_prefix = f"{self.base_prefix}/metadata"

        # Initialize S3 client
        self.s3_client = boto3.client('s3')

        # Create a temporary directory for local operations
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        super().__init__(self.temp_dir)

    def _s3_key(self, prefix: str, name: str) -> str:
        """
        Generate an S3 key from a prefix and name.

        Args:
            prefix: The prefix (folder)
            name: The file name

        Returns:
            The full S3 key
        """
        return f"{prefix}/{name}"

    def save_model(self, model: Any, model_name: str) -> str:
        """
        Save a model to S3.

        Args:
            model: The model to save
            model_name: Name to use for the saved model

        Returns:
            S3 URI where the model was saved
        """
        if not model_name.endswith('.joblib'):
            model_name = f"{model_name}.joblib"

        # First save locally
        local_path = super().save_model(model, model_name)

        # Then upload to S3
        s3_key = self._s3_key(self.models_prefix, model_name)

        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            return f"s3://{self.bucket_name}/{s3_key}"
        except ClientError as e:
            raise IOError(f"Failed to upload model to S3: {str(e)}")

    def load_model(self, model_name: str) -> Any:
        """
        Load a model from S3.

        Args:
            model_name: Name of the model to load

        Returns:
            The loaded model
        """
        if not model_name.endswith('.joblib'):
            model_name = f"{model_name}.joblib"

        # Define local and S3 paths
        local_path = self.models_path / model_name
        s3_key = self._s3_key(self.models_prefix, model_name)

        try:
            # Download from S3
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))

            # Load using parent method
            return super().load_model(model_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Model not found in S3: {s3_key}")
            else:
                raise IOError(f"Failed to download model from S3: {str(e)}")

    def save_preprocessor(self, preprocessor: Any, preprocessor_name: str) -> str:
        """
        Save a preprocessor to S3.

        Args:
            preprocessor: The preprocessor to save
            preprocessor_name: Name to use for the saved preprocessor

        Returns:
            S3 URI where the preprocessor was saved
        """
        if not preprocessor_name.endswith('.joblib'):
            preprocessor_name = f"{preprocessor_name}.joblib"

        # First save locally
        local_path = super().save_preprocessor(preprocessor, preprocessor_name)

        # Then upload to S3
        s3_key = self._s3_key(self.preprocessors_prefix, preprocessor_name)

        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            return f"s3://{self.bucket_name}/{s3_key}"
        except ClientError as e:
            raise IOError(f"Failed to upload preprocessor to S3: {str(e)}")

    def load_preprocessor(self, preprocessor_name: str) -> Any:
        """
        Load a preprocessor from S3.

        Args:
            preprocessor_name: Name of the preprocessor to load

        Returns:
            The loaded preprocessor
        """
        if not preprocessor_name.endswith('.joblib'):
            preprocessor_name = f"{preprocessor_name}.joblib"

        # Define local and S3 paths
        local_path = self.preprocessors_path / preprocessor_name
        s3_key = self._s3_key(self.preprocessors_prefix, preprocessor_name)

        try:
            # Download from S3
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))

            # Load using parent method
            return super().load_preprocessor(preprocessor_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Preprocessor not found in S3: {s3_key}")
            else:
                raise IOError(f"Failed to download preprocessor from S3: {str(e)}")

    def save_metadata(self, metadata: Dict[str, Any], metadata_name: str) -> str:
        """
        Save metadata to S3.

        Args:
            metadata: Dictionary containing metadata
            metadata_name: Name to use for the saved metadata

        Returns:
            S3 URI where the metadata was saved
        """
        if not metadata_name.endswith('.json'):
            metadata_name = f"{metadata_name}.json"

        # First save locally
        local_path = super().save_metadata(metadata, metadata_name)

        # Then upload to S3
        s3_key = self._s3_key(self.metadata_prefix, metadata_name)

        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            return f"s3://{self.bucket_name}/{s3_key}"
        except ClientError as e:
            raise IOError(f"Failed to upload metadata to S3: {str(e)}")

    def load_metadata(self, metadata_name: str) -> Dict[str, Any]:
        """
        Load metadata from S3.

        Args:
            metadata_name: Name of the metadata to load

        Returns:
            Dictionary containing the loaded metadata
        """
        if not metadata_name.endswith('.json'):
            metadata_name = f"{metadata_name}.json"

        # Define local and S3 paths
        local_path = self.metadata_path / metadata_name
        s3_key = self._s3_key(self.metadata_prefix, metadata_name)

        try:
            # Download from S3
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))

            # Load using parent method
            return super().load_metadata(metadata_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Metadata not found in S3: {s3_key}")
            else:
                raise IOError(f"Failed to download metadata from S3: {str(e)}")


def get_storage(storage_type: str, **kwargs) -> Union[ModelStorage, S3ModelStorage]:
    """
    Factory function to get the appropriate storage class.

    Args:
        storage_type: Type of storage ('local' or 's3')
        **kwargs: Arguments to pass to the storage class constructor

    Returns:
        An instance of the appropriate storage class

    Raises:
        ValueError: If the storage type is not supported
    """
    if storage_type.lower() == 'local':
        if 'base_path' not in kwargs:
            raise ValueError("base_path is required for local storage")
        return ModelStorage(**kwargs)
    elif storage_type.lower() == 's3':
        if 'bucket_name' not in kwargs:
            raise ValueError("bucket_name is required for S3 storage")
        return S3ModelStorage(**kwargs)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}. "
                         f"Supported types are: 'local', 's3'")