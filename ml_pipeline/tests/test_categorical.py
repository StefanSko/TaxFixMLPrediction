"""
Tests for the categorical preprocessor module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from preprocessing.categorical import CategoricalPreprocessor, DEFAULT_CATEGORICAL_COLUMNS


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    data = {
        'employment_type': np.random.choice(['full-time', 'part-time', 'self-employed', 'unemployed'], 100),
        'marital_status': np.random.choice(['single', 'married', 'divorced', 'widowed'], 100),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], 100),
        'referral_source': np.random.choice(['search', 'direct', 'social', 'email'], 100),
        'previous_year_filing': np.random.choice(['yes', 'no'], 100),
        'age': np.random.normal(35, 10, 100),
        'target': np.random.randint(0, 2, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_data_with_new_categories():
    """Create test data with new categories not seen in training."""
    np.random.seed(43)
    data = {
        'employment_type': np.random.choice(['full-time', 'part-time', 'self-employed', 'unemployed', 'contract'], 50),
        'marital_status': np.random.choice(['single', 'married', 'divorced', 'widowed', 'separated'], 50),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet', 'smart-tv'], 50),
        'referral_source': np.random.choice(['search', 'direct', 'social', 'email', 'partner'], 50),
        'previous_year_filing': np.random.choice(['yes', 'no', 'unknown'], 50),
        'age': np.random.normal(35, 10, 50),
        'target': np.random.randint(0, 2, 50)
    }
    return pd.DataFrame(data)


def test_init():
    """Test initialization of CategoricalPreprocessor."""
    # Test with default encoding method
    preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS)
    assert preprocessor.categorical_columns == DEFAULT_CATEGORICAL_COLUMNS
    assert preprocessor.encoding_method == 'onehot'
    assert not preprocessor.fitted

    # Test with label encoding
    preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS, encoding_method='label')
    assert preprocessor.encoding_method == 'label'

    # Test with invalid encoding method
    with pytest.raises(ValueError, match="encoding_method must be either 'onehot' or 'label'"):
        CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS, encoding_method='invalid')


def test_fit(sample_data):
    """Test fitting the preprocessor."""
    # Test one-hot encoding
    preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS)
    preprocessor.fit(sample_data)

    assert preprocessor.fitted

    # Check that categories were learned
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        assert col in preprocessor.categories
        assert set(preprocessor.categories[col]) == set(sample_data[col].unique())

    # Test label encoding
    preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS, encoding_method='label')
    preprocessor.fit(sample_data)

    assert preprocessor.fitted

    # Check that label encodings were created
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        assert col in preprocessor.label_encodings
        assert len(preprocessor.label_encodings[col]) == len(sample_data[col].unique())


def test_transform_onehot(sample_data):
    """Test transforming data with one-hot encoding."""
    preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS)
    preprocessor.fit(sample_data)

    transformed_data = preprocessor.transform(sample_data)

    # Check that original categorical columns were removed
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        assert col not in transformed_data.columns

    # Check that one-hot encoded columns were created
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        for category in preprocessor.categories[col]:
            one_hot_col = f"{col}_{category}"
            assert one_hot_col in transformed_data.columns
            # Values should be 0 or 1
            assert set(transformed_data[one_hot_col].unique()).issubset({0, 1})

    # Check that non-categorical columns were not modified
    assert 'age' in transformed_data.columns
    assert 'target' in transformed_data.columns
    assert (transformed_data['age'] == sample_data['age']).all()
    assert (transformed_data['target'] == sample_data['target']).all()


def test_transform_label(sample_data):
    """Test transforming data with label encoding."""
    preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS, encoding_method='label')
    preprocessor.fit(sample_data)

    transformed_data = preprocessor.transform(sample_data)

    # Check that categorical columns were encoded
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        assert col in transformed_data.columns
        # Values should be integers
        assert all(isinstance(val, (int, np.integer)) for val in transformed_data[col])
        # Check range of values
        assert transformed_data[col].min() >= 0
        assert transformed_data[col].max() < len(preprocessor.categories[col])

    # Check that non-categorical columns were not modified
    assert 'age' in transformed_data.columns
    assert 'target' in transformed_data.columns
    assert (transformed_data['age'] == sample_data['age']).all()
    assert (transformed_data['target'] == sample_data['target']).all()


def test_fit_transform(sample_data):
    """Test fit_transform method."""
    # Test one-hot encoding
    preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS)
    transformed_data = preprocessor.fit_transform(sample_data)

    assert preprocessor.fitted

    # Check that original categorical columns were removed
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        assert col not in transformed_data.columns

    # Check that one-hot encoded columns were created
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        for category in preprocessor.categories[col]:
            one_hot_col = f"{col}_{category}"
            assert one_hot_col in transformed_data.columns


def test_save_load():
    """Test saving and loading the preprocessor."""
    # Create a preprocessor
    preprocessor = CategoricalPreprocessor(['employment_type', 'marital_status'])
    preprocessor.categories = {
        'employment_type': ['full-time', 'part-time'],
        'marital_status': ['single', 'married']
    }
    preprocessor.fitted = True

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'preprocessor.joblib')
        preprocessor.save(save_path)

        # Load the preprocessor
        loaded_preprocessor = CategoricalPreprocessor.load(save_path)

        # Check that the loaded preprocessor has the same attributes
        assert loaded_preprocessor.categorical_columns == preprocessor.categorical_columns
        assert loaded_preprocessor.encoding_method == preprocessor.encoding_method
        assert loaded_preprocessor.categories == preprocessor.categories
        assert loaded_preprocessor.fitted == preprocessor.fitted


def test_transform_without_fit():
    """Test that transform raises an error if called before fit."""
    preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS)
    with pytest.raises(ValueError, match="Preprocessor has not been fitted"):
        preprocessor.transform(pd.DataFrame())


def test_column_not_found():
    """Test that appropriate errors are raised when columns are missing."""
    preprocessor = CategoricalPreprocessor(['nonexistent_column'])
    with pytest.raises(ValueError, match="Column 'nonexistent_column' not found"):
        preprocessor.fit(pd.DataFrame({'other_column': [1, 2, 3]}))


def test_handle_unseen_categories(sample_data, test_data_with_new_categories):
    """Test handling of unseen categories in test data."""
    # Test one-hot encoding
    onehot_preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS)
    onehot_preprocessor.fit(sample_data)

    transformed_test = onehot_preprocessor.transform(test_data_with_new_categories)

    # Check that only columns for categories seen during training were created
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        for category in onehot_preprocessor.categories[col]:
            one_hot_col = f"{col}_{category}"
            assert one_hot_col in transformed_test.columns

        # New categories should not have columns
        for category in set(test_data_with_new_categories[col].unique()) - set(onehot_preprocessor.categories[col]):
            one_hot_col = f"{col}_{category}"
            assert one_hot_col not in transformed_test.columns

    # Test label encoding
    label_preprocessor = CategoricalPreprocessor(DEFAULT_CATEGORICAL_COLUMNS, encoding_method='label')
    label_preprocessor.fit(sample_data)

    transformed_test = label_preprocessor.transform(test_data_with_new_categories)

    # Check that unseen categories are encoded as -1
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        # Find rows with unseen categories
        unseen_mask = ~test_data_with_new_categories[col].isin(label_preprocessor.categories[col])
        if unseen_mask.any():
            # These should be encoded as -1
            assert (transformed_test.loc[unseen_mask, col] == -1).all()