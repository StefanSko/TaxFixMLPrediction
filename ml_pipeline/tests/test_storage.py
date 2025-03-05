"""
Tests for the storage module.
"""

import os
import tempfile

import boto3
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from utils.storage import ModelStorage, S3ModelStorage, get_storage


class TestModelStorage:

    def setup_method(self):
        """Set up test data and storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ModelStorage(self.temp_dir)

        # Create test data
        self.test_model = {"model_type": "test_model"}
        self.test_preprocessor = {"preprocessor_type": "test_preprocessor"}
        self.test_metadata = {"version": "1.0", "created_at": "2023-01-01"}

    def test_init(self):
        """Test initialization creates directories."""
        assert Path(self.temp_dir, "models").exists()
        assert Path(self.temp_dir, "preprocessors").exists()
        assert Path(self.temp_dir, "metadata").exists()

    def test_save_load_model(self):
        """Test saving and loading a model."""
        # Save model
        model_path = self.storage.save_model(self.test_model, "test_model")
        assert os.path.exists(model_path)

        # Load model
        loaded_model = self.storage.load_model("test_model")
        assert loaded_model == self.test_model

        # Test with .joblib extension already included
        model_path = self.storage.save_model(self.test_model, "test_model.joblib")
        loaded_model = self.storage.load_model("test_model.joblib")
        assert loaded_model == self.test_model

    def test_save_load_preprocessor(self):
        """Test saving and loading a preprocessor."""
        # Save preprocessor
        preprocessor_path = self.storage.save_preprocessor(
            self.test_preprocessor, "test_preprocessor")
        assert os.path.exists(preprocessor_path)

        # Load preprocessor
        loaded_preprocessor = self.storage.load_preprocessor("test_preprocessor")
        assert loaded_preprocessor == self.test_preprocessor

    def test_save_load_metadata(self):
        """Test saving and loading metadata."""
        # Save metadata
        metadata_path = self.storage.save_metadata(self.test_metadata, "test_metadata")
        assert os.path.exists(metadata_path)

        # Load metadata
        loaded_metadata = self.storage.load_metadata("test_metadata")
        assert loaded_metadata == self.test_metadata

    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.storage.load_model("nonexistent_model")

        with pytest.raises(FileNotFoundError):
            self.storage.load_preprocessor("nonexistent_preprocessor")

        with pytest.raises(FileNotFoundError):
            self.storage.load_metadata("nonexistent_metadata")


class TestS3ModelStorage:

    def setup_method(self):
        """Set up test data and mock S3 client."""
        self.bucket_name = "test-bucket"
        self.base_prefix = "test-models"

        # Create test data
        self.test_model = {"model_type": "test_model"}
        self.test_preprocessor = {"preprocessor_type": "test_preprocessor"}
        self.test_metadata = {"version": "1.0", "created_at": "2023-01-01"}

        # Create patcher for boto3 client
        self.s3_client_mock = MagicMock()
        self.boto3_patcher = patch('boto3.client', return_value=self.s3_client_mock)
        self.boto3_mock = self.boto3_patcher.start()

    def teardown_method(self):
        """Clean up patchers."""
        self.boto3_patcher.stop()

    def test_init(self):
        """Test initialization."""
        storage = S3ModelStorage(self.bucket_name, self.base_prefix)
        assert storage.bucket_name == self.bucket_name
        assert storage.base_prefix == self.base_prefix
        assert storage.models_prefix == f"{self.base_prefix}/models"
        assert storage.preprocessors_prefix == f"{self.base_prefix}/preprocessors"
        assert storage.metadata_prefix == f"{self.base_prefix}/metadata"

    def test_save_model(self):
        """Test saving a model to S3."""
        storage = S3ModelStorage(self.bucket_name, self.base_prefix)

        # Test saving
        s3_uri = storage.save_model(self.test_model, "test_model")

        # Check S3 URI format
        expected_uri = f"s3://{self.bucket_name}/{self.base_prefix}/models/test_model.joblib"
        assert s3_uri == expected_uri

        # Verify S3 client was called correctly
        self.s3_client_mock.upload_file.assert_called_once()
        args = self.s3_client_mock.upload_file.call_args[0]
        assert args[1] == self.bucket_name
        assert args[2] == f"{self.base_prefix}/models/test_model.joblib"

    def test_load_model(self):
        """Test loading a model from S3."""
        storage = S3ModelStorage(self.bucket_name, self.base_prefix)

        # Mock download_file to write test data to the local file
        def side_effect(bucket, key, filename):
            import joblib
            joblib.dump(self.test_model, filename)

        self.s3_client_mock.download_file.side_effect = side_effect

        # Test loading
        loaded_model = storage.load_model("test_model")

        # Verify model was loaded correctly
        assert loaded_model == self.test_model

        # Verify S3 client was called correctly
        self.s3_client_mock.download_file.assert_called_once()
        args = self.s3_client_mock.download_file.call_args[0]
        assert args[0] == self.bucket_name
        assert args[1] == f"{self.base_prefix}/models/test_model.joblib"

    def test_s3_file_not_found(self):
        """Test handling of non-existent S3 files."""
        storage = S3ModelStorage(self.bucket_name, self.base_prefix)

        # Mock ClientError for NoSuchKey
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        self.s3_client_mock.download_file.side_effect = \
            boto3.exceptions.botocore.exceptions.ClientError(error_response, 'GetObject')

        with pytest.raises(FileNotFoundError):
            storage.load_model("nonexistent_model")

    def test_s3_client_error(self):
        """Test handling of S3 client errors."""
        storage = S3ModelStorage(self.bucket_name, self.base_prefix)

        # Mock ClientError for other errors
        error_response = {'Error': {'Code': 'AccessDenied'}}
        self.s3_client_mock.download_file.side_effect = \
            boto3.exceptions.botocore.exceptions.ClientError(error_response, 'GetObject')

        with pytest.raises(IOError):
            storage.load_model("test_model")


class TestGetStorage:

    def test_get_local_storage(self):
        """Test getting local storage."""
        storage = get_storage('local', base_path='/tmp/models')
        assert isinstance(storage, ModelStorage)
        assert storage.base_path == Path('/tmp/models')

    def test_get_s3_storage(self):
        """Test getting S3 storage."""
        with patch('boto3.client'):
            storage = get_storage('s3', bucket_name='test-bucket')
            assert isinstance(storage, S3ModelStorage)
            assert storage.bucket_name == 'test-bucket'

    def test_missing_required_args(self):
        """Test error handling for missing required arguments."""
        with pytest.raises(ValueError, match="base_path is required"):
            get_storage('local')

        with pytest.raises(ValueError, match="bucket_name is required"):
            get_storage('s3')

    def test_unsupported_storage_type(self):
        """Test error handling for unsupported storage type."""
        with pytest.raises(ValueError, match="Unsupported storage type"):
            get_storage('unsupported')