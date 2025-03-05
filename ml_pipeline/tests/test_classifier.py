"""
Tests for the TaxfixClassifier class.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile

from training.classifier import TaxfixClassifier


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a simple dataset with 100 samples and 5 features
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # Binary classification problem
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Convert to pandas
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')

    return X, y, X_df, y_series


def test_init_with_valid_model_type():
    """Test initialization with valid model types."""
    # Test default initialization
    clf = TaxfixClassifier()
    assert clf.model_type == 'logistic_regression'

    # Test with specific model type
    clf = TaxfixClassifier(model_type='logistic_regression')
    assert clf.model_type == 'logistic_regression'


def test_init_with_invalid_model_type():
    """Test initialization with invalid model type."""
    with pytest.raises(ValueError):
        TaxfixClassifier(model_type='invalid_model')


def test_init_with_custom_params():
    """Test initialization with custom parameters."""
    custom_params = {'C': 0.5, 'max_iter': 200}
    clf = TaxfixClassifier(model_type='logistic_regression', **custom_params)

    # Check that custom parameters were applied
    assert clf.model_params['C'] == 0.5
    assert clf.model_params['max_iter'] == 200
    # Check that default parameters were preserved
    assert clf.model_params['random_state'] == 42


def test_fit_with_numpy_arrays(sample_data):
    """Test fitting with numpy arrays."""
    X, y, _, _ = sample_data
    clf = TaxfixClassifier()

    # Should not raise any exceptions
    clf.fit(X, y)
    assert clf.fitted is True
    assert clf.feature_names is None  # No feature names with numpy arrays


def test_fit_with_pandas_dataframes(sample_data):
    """Test fitting with pandas DataFrames."""
    _, _, X_df, y_series = sample_data
    clf = TaxfixClassifier()

    # Should not raise any exceptions
    clf.fit(X_df, y_series)
    assert clf.fitted is True
    assert clf.feature_names == X_df.columns.tolist()


def test_predict_without_fitting(sample_data):
    """Test prediction without fitting first."""
    X, _, _, _ = sample_data
    clf = TaxfixClassifier()

    with pytest.raises(ValueError, match="Model has not been fitted"):
        clf.predict(X)


def test_predict_with_numpy_arrays(sample_data):
    """Test prediction with numpy arrays."""
    X, y, _, _ = sample_data
    clf = TaxfixClassifier()
    clf.fit(X, y)

    predictions = clf.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (100,)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_predict_with_pandas_dataframes(sample_data):
    """Test prediction with pandas DataFrames."""
    _, _, X_df, y_series = sample_data
    clf = TaxfixClassifier()
    clf.fit(X_df, y_series)

    predictions = clf.predict(X_df)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (100,)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_predict_proba(sample_data):
    """Test probability prediction."""
    X, y, _, _ = sample_data
    clf = TaxfixClassifier()
    clf.fit(X, y)

    probabilities = clf.predict_proba(X)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (100, 2)  # Binary classification
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_feature_name_mismatch(sample_data):
    """Test error when feature names don't match."""
    _, _, X_df, y_series = sample_data
    clf = TaxfixClassifier()
    clf.fit(X_df, y_series)

    # Create DataFrame with different column names
    X_different = X_df.copy()
    X_different.columns = ['a', 'b', 'c', 'd', 'e']

    with pytest.raises(ValueError, match="Feature names in prediction data do not match"):
        clf.predict(X_different)


def test_save_and_load(sample_data):
    """Test saving and loading the model."""
    _, _, X_df, y_series = sample_data

    # Create and fit a model
    clf = TaxfixClassifier()
    clf.fit(X_df, y_series)

    # Get predictions from the original model
    original_predictions = clf.predict(X_df)

    # Save the model to a temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.joblib")
        clf.save(model_path)

        # Load the model
        loaded_clf = TaxfixClassifier.load(model_path)

        # Check that the loaded model has the same attributes
        assert loaded_clf.model_type == clf.model_type
        assert loaded_clf.model_params == clf.model_params
        assert loaded_clf.feature_names == clf.feature_names
        assert loaded_clf.fitted == clf.fitted

        # Check that the loaded model makes the same predictions
        loaded_predictions = loaded_clf.predict(X_df)
        assert np.array_equal(loaded_predictions, original_predictions)