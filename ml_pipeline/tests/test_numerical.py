"""
Tests for the numerical preprocessor module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from preprocessing.numerical import NumericalPreprocessor, DEFAULT_NUMERICAL_COLUMNS


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    data = {
        'age': np.random.normal(35, 10, 100),
        'income': np.random.normal(50000, 15000, 100),
        'time_spent_on_platform': np.random.normal(120, 30, 100),
        'number_of_sessions': np.random.normal(25, 10, 100),
        'fields_filled_percentage': np.random.normal(75, 15, 100),
        'target': np.random.randint(0, 2, 100)
    }
    return pd.DataFrame(data)


def test_init():
    """Test initialization of NumericalPreprocessor."""
    preprocessor = NumericalPreprocessor(DEFAULT_NUMERICAL_COLUMNS)
    assert preprocessor.numerical_columns == DEFAULT_NUMERICAL_COLUMNS
    assert preprocessor.means is None
    assert preprocessor.stds is None


def test_fit(sample_data):
    """Test fitting the preprocessor."""
    preprocessor = NumericalPreprocessor(DEFAULT_NUMERICAL_COLUMNS)
    preprocessor.fit(sample_data)

    # Check that means and stds were calculated
    for col in DEFAULT_NUMERICAL_COLUMNS:
        assert col in preprocessor.means
        assert col in preprocessor.stds
        assert abs(preprocessor.means[col] - sample_data[col].mean()) < 1e-10
        assert abs(preprocessor.stds[col] - sample_data[col].std()) < 1e-10


def test_transform(sample_data):
    """Test transforming data with the preprocessor."""
    preprocessor = NumericalPreprocessor(DEFAULT_NUMERICAL_COLUMNS)
    preprocessor.fit(sample_data)

    transformed_data = preprocessor.transform(sample_data)

    # Check that transformation was applied correctly
    for col in DEFAULT_NUMERICAL_COLUMNS:
        # Transformed data should have mean close to 0 and std close to 1
        assert abs(transformed_data[col].mean()) < 1e-10
        assert abs(transformed_data[col].std() - 1.0) < 1e-10

    # Check that non-numerical columns were not modified
    assert 'target' in transformed_data.columns
    assert (transformed_data['target'] == sample_data['target']).all()


def test_fit_transform(sample_data):
    """Test fit_transform method."""
    preprocessor = NumericalPreprocessor(DEFAULT_NUMERICAL_COLUMNS)
    transformed_data = preprocessor.fit_transform(sample_data)

    # Check that transformation was applied correctly
    for col in DEFAULT_NUMERICAL_COLUMNS:
        # Transformed data should have mean close to 0 and std close to 1
        assert abs(transformed_data[col].mean()) < 1e-10
        assert abs(transformed_data[col].std() - 1.0) < 1e-10


def test_save_load():
    """Test saving and loading the preprocessor."""
    # Create a preprocessor
    preprocessor = NumericalPreprocessor(['age', 'income'])
    preprocessor.means = {'age': 35.0, 'income': 50000.0}
    preprocessor.stds = {'age': 10.0, 'income': 15000.0}

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'preprocessor.joblib')
        preprocessor.save(save_path)

        # Load the preprocessor
        loaded_preprocessor = NumericalPreprocessor.load(save_path)

        # Check that the loaded preprocessor has the same attributes
        assert loaded_preprocessor.numerical_columns == preprocessor.numerical_columns
        assert loaded_preprocessor.means == preprocessor.means
        assert loaded_preprocessor.stds == preprocessor.stds


def test_transform_without_fit():
    """Test that transform raises an error if called before fit."""
    preprocessor = NumericalPreprocessor(DEFAULT_NUMERICAL_COLUMNS)
    with pytest.raises(ValueError, match="Preprocessor has not been fitted"):
        preprocessor.transform(pd.DataFrame())


def test_column_not_found():
    """Test that appropriate errors are raised when columns are missing."""
    preprocessor = NumericalPreprocessor(['nonexistent_column'])
    with pytest.raises(ValueError, match="Column 'nonexistent_column' not found"):
        preprocessor.fit(pd.DataFrame({'other_column': [1, 2, 3]}))


def test_zero_std_handling(sample_data):
    """Test handling of zero standard deviation."""
    # Create data with a constant column
    data = sample_data.copy()
    data['constant_column'] = 5.0

    preprocessor = NumericalPreprocessor(['constant_column'])
    preprocessor.fit(data)

    # Check that std was set to 1.0 to avoid division by zero
    assert preprocessor.stds['constant_column'] == 1.0

    # Transform should not raise an error
    transformed = preprocessor.transform(data)
    assert (transformed['constant_column'] == 0.0).all()  # (5-5)/1 = 0