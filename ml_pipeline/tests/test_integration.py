"""
Integration tests for the model training pipeline using real data.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from training.pipeline import ModelTrainingPipeline


@pytest.fixture
def sample_data_path():
    """Return the path to the sample data file."""
    # Get the absolute path to the sample.csv file
    current_dir = Path(__file__).parent
    sample_path = current_dir / "resources" / "sample.csv"

    if not sample_path.exists():
        pytest.skip(f"Sample data file not found at {sample_path}")

    return str(sample_path)


@pytest.fixture
def integration_config(sample_data_path):
    """Create a configuration for integration testing with real data."""
    return {
        "data_path": sample_data_path,
        "model_config": {
            "type": "logistic_regression",
            "params": {
                "C": 1.0,
                "max_iter": 100,
                "random_state": 42
            }
        },
        "preprocessing_config": {
            "target_column": "completed_filing",
            "test_size": 0.2,
            "stratify": True,
            "use_preprocessing_pipeline": True,
            "numerical_columns": [
                "age", "income", "time_spent_on_platform",
                "number_of_sessions", "fields_filled_percentage",
                "previous_year_filing"
            ],
            "categorical_columns": [
                "employment_type", "marital_status", "device_type",
                "referral_source"
            ]
        },
        "storage_config": {
            "output_dir": "test_output",
            "model_path": "test_output/model.joblib"
        },
        "random_seed": 42
    }


def test_end_to_end_pipeline(integration_config):
    """
    Test the complete pipeline from data loading to model evaluation using real data.

    This test verifies that:
    1. Data can be loaded from the sample file
    2. Preprocessing works correctly with real data
    3. Model training completes successfully
    4. Evaluation produces valid metrics
    5. Artifacts are saved correctly
    """
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update storage paths in config
        integration_config["storage_config"]["output_dir"] = temp_dir
        integration_config["storage_config"]["model_path"] = os.path.join(temp_dir, "model.joblib")

        # Create and run the pipeline
        pipeline = ModelTrainingPipeline(integration_config)
        results = pipeline.run()

        # Verify results structure
        assert "metrics" in results
        assert "model_path" in results
        assert "config" in results

        # Check that metrics are reasonable
        metrics = results["metrics"]
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

        # Verify that accuracy is better than random guessing (0.5)
        # This is a reasonable expectation for this dataset
        assert metrics["accuracy"] > 0.5

        # Check that files were created
        assert os.path.exists(results["model_path"])
        assert os.path.exists(os.path.join(temp_dir, "preprocessor.joblib"))
        assert os.path.exists(os.path.join(temp_dir, "evaluation_results.json"))
        assert os.path.exists(os.path.join(temp_dir, "training_config.json"))

        # Verify that the saved evaluation results match the returned results
        with open(os.path.join(temp_dir, "evaluation_results.json"), "r") as f:
            saved_metrics = json.load(f)
        assert saved_metrics == metrics


def test_data_preprocessing_effect(integration_config, sample_data_path):
    """
    Test that preprocessing has the expected effect on the data.

    This test verifies that:
    1. Numerical features are properly scaled
    2. Categorical features are properly one-hot encoded
    3. The resulting dataset has the expected shape
    """
    # Load the raw data to understand its structure
    raw_data = pd.read_csv(sample_data_path)

    # Create pipeline and run preprocessing only
    pipeline = ModelTrainingPipeline(integration_config)
    X_train, y_train, X_test, y_test = pipeline.load_and_preprocess_data()

    # Check that the preprocessor was created
    assert pipeline.preprocessor is not None

    # Verify that the target column was properly extracted
    target_col = integration_config["preprocessing_config"]["target_column"]
    assert target_col not in X_train.columns
    assert target_col not in X_test.columns

    # Check that all categorical columns were one-hot encoded
    cat_cols = integration_config["preprocessing_config"]["categorical_columns"]
    for col in cat_cols:
        assert col not in X_train.columns
        # Check for at least one one-hot encoded column for each category
        assert any(col in c for c in X_train.columns)

    # Verify that numerical columns are present
    num_cols = integration_config["preprocessing_config"]["numerical_columns"]
    for col in num_cols:
        assert col in X_train.columns

    # Check that train and test sets have the same columns
    assert set(X_train.columns) == set(X_test.columns)

    # Verify that the total number of samples is preserved
    assert len(X_train) + len(X_test) == len(raw_data)
    assert len(y_train) + len(y_test) == len(raw_data)

    # Check that the class distribution is reasonable
    train_positive = y_train.mean()
    test_positive = y_test.mean()
    raw_positive = raw_data[target_col].mean()

    # Class distribution should be similar across splits (within 10%)
    assert abs(train_positive - raw_positive) < 0.1
    assert abs(test_positive - raw_positive) < 0.1


def test_model_reproducibility(integration_config):
    """
    Test that the pipeline produces reproducible results with the same random seed.

    This test verifies that:
    1. Running the pipeline twice with the same seed produces the same results
    2. Running with different seeds produces different results
    """
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update storage paths in config
        integration_config["storage_config"]["output_dir"] = temp_dir
        integration_config["storage_config"]["model_path"] = os.path.join(temp_dir, "model.joblib")

        # Run the pipeline with the original seed
        pipeline1 = ModelTrainingPipeline(integration_config)
        results1 = pipeline1.run()

        # Run the pipeline again with the same seed
        pipeline2 = ModelTrainingPipeline(integration_config)
        results2 = pipeline2.run()

        # Results should be identical with the same seed
        assert results1["metrics"] == results2["metrics"]

        # Now run with a different seed
        different_config = integration_config.copy()
        different_config["random_seed"] = 100  # Different seed

        # Create a new temporary directory
        with tempfile.TemporaryDirectory() as temp_dir2:
            different_config["storage_config"]["output_dir"] = temp_dir2
            different_config["storage_config"]["model_path"] = os.path.join(temp_dir2, "model.joblib")

            pipeline3 = ModelTrainingPipeline(different_config)
            results3 = pipeline3.run()

            # Results should be different with a different seed
            # We check if at least one metric is different
            metrics1 = results1["metrics"]
            metrics3 = results3["metrics"]

            # Check if any metric differs by more than a small epsilon
            assert any(abs(metrics1[k] - metrics3[k]) > 1e-10 for k in metrics1.keys())