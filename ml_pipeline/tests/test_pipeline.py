"""
Tests for the integrated preprocessing pipeline module.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.numerical import DEFAULT_NUMERICAL_COLUMNS
from preprocessing.categorical import DEFAULT_CATEGORICAL_COLUMNS


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    data = {
        # Numerical features
        'age': np.random.normal(35, 10, 100),
        'income': np.random.normal(50000, 15000, 100),
        'time_spent_on_platform': np.random.normal(120, 30, 100),
        'number_of_sessions': np.random.normal(25, 10, 100),
        'fields_filled_percentage': np.random.normal(75, 15, 100),

        # Categorical features
        'employment_type': np.random.choice(['full-time', 'part-time', 'self-employed', 'unemployed'], 100),
        'marital_status': np.random.choice(['single', 'married', 'divorced', 'widowed'], 100),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], 100),
        'referral_source': np.random.choice(['search', 'direct', 'social', 'email'], 100),
        'previous_year_filing': np.random.choice(['yes', 'no'], 100),

        # Target variable
        'target': np.random.randint(0, 2, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_data_with_new_categories():
    """Create test data with new categories not seen in training."""
    np.random.seed(43)
    data = {
        # Numerical features
        'age': np.random.normal(35, 10, 50),
        'income': np.random.normal(50000, 15000, 50),
        'time_spent_on_platform': np.random.normal(120, 30, 50),
        'number_of_sessions': np.random.normal(25, 10, 50),
        'fields_filled_percentage': np.random.normal(75, 15, 50),

        # Categorical features with new categories
        'employment_type': np.random.choice(['full-time', 'part-time', 'self-employed', 'unemployed', 'contract'], 50),
        'marital_status': np.random.choice(['single', 'married', 'divorced', 'widowed', 'separated'], 50),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet', 'smart-tv'], 50),
        'referral_source': np.random.choice(['search', 'direct', 'social', 'email', 'partner'], 50),
        'previous_year_filing': np.random.choice(['yes', 'no', 'unknown'], 50),

        # Target variable
        'target': np.random.randint(0, 2, 50)
    }
    return pd.DataFrame(data)


def test_init():
    """Test initialization of PreprocessingPipeline."""
    # Test with default parameters
    pipeline = PreprocessingPipeline()
    assert pipeline.numerical_columns == DEFAULT_NUMERICAL_COLUMNS
    assert pipeline.categorical_columns == DEFAULT_CATEGORICAL_COLUMNS
    assert pipeline.encoding_method == 'onehot'
    assert not pipeline.fitted

    # Test with custom parameters
    custom_num_cols = ['age', 'income']
    custom_cat_cols = ['employment_type', 'marital_status']
    pipeline = PreprocessingPipeline(
        numerical_columns=custom_num_cols,
        categorical_columns=custom_cat_cols,
    )
    assert pipeline.numerical_columns == custom_num_cols
    assert pipeline.categorical_columns == custom_cat_cols


def test_fit(sample_data):
    """Test fitting the pipeline."""
    pipeline = PreprocessingPipeline()
    pipeline.fit(sample_data)

    assert pipeline.fitted
    assert pipeline._feature_names is not None

    # Check that both preprocessors were fitted
    assert pipeline.numerical_preprocessor.means is not None
    assert pipeline.numerical_preprocessor.stds is not None
    assert pipeline.categorical_preprocessor.fitted


def test_transform_onehot(sample_data):

    """Test transforming data with one-hot encoding."""
    pipeline = PreprocessingPipeline()
    pipeline.fit(sample_data)

    transformed_data = pipeline.transform(sample_data)

    # Check that numerical features were standardized
    for col in DEFAULT_NUMERICAL_COLUMNS:
        assert col in transformed_data.columns
        assert abs(transformed_data[col].mean()) < 1e-10
        assert abs(transformed_data[col].std() - 1.0) < 1e-10

    # Check that categorical features were one-hot encoded
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        assert col not in transformed_data.columns

        # Check that one-hot encoded columns exist
        for category in pipeline.categorical_preprocessor.categories[col]:
            one_hot_col = f"{col}_{category}"
            assert one_hot_col in transformed_data.columns
            # Values should be 0 or 1
            assert set(transformed_data[one_hot_col].unique()).issubset({0, 1})

    # Check that target column is preserved
    assert 'target' in transformed_data.columns
    assert (transformed_data['target'] == sample_data['target']).all()


def test_fit_transform(sample_data):
    """Test fit_transform method."""
    pipeline = PreprocessingPipeline()
    transformed_data = pipeline.fit_transform(sample_data)

    assert pipeline.fitted

    # Check dimensions
    expected_cols = len(DEFAULT_NUMERICAL_COLUMNS)
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        expected_cols += len(pipeline.categorical_preprocessor.categories[col])

    # Add 1 for target column
    expected_cols += 1

    assert transformed_data.shape[1] == expected_cols
    assert transformed_data.shape[0] == sample_data.shape[0]


def test_get_feature_names(sample_data):
    """Test get_feature_names method."""
    # Test with one-hot encoding
    onehot_pipeline = PreprocessingPipeline()
    onehot_pipeline.fit(sample_data)

    feature_names = onehot_pipeline.get_feature_names()

    # Check numerical features
    for col in DEFAULT_NUMERICAL_COLUMNS:
        assert col in feature_names

    # Check categorical features
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        assert col not in feature_names  # Original column names should not be present
        for category in onehot_pipeline.categorical_preprocessor.categories[col]:
            assert f"{col}_{category}" in feature_names


def test_save_load(sample_data):
    """Test saving and loading the pipeline."""
    # Create and fit a pipeline
    pipeline = PreprocessingPipeline()
    pipeline.fit(sample_data)

    transformed_original = pipeline.transform(sample_data)

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline.save(tmpdir)

        # Load the pipeline
        loaded_pipeline = PreprocessingPipeline.load(tmpdir)

        # Check that the loaded pipeline has the same attributes
        assert loaded_pipeline.numerical_columns == pipeline.numerical_columns
        assert loaded_pipeline.categorical_columns == pipeline.categorical_columns
        assert loaded_pipeline._feature_names == pipeline._feature_names
        assert loaded_pipeline.fitted == pipeline.fitted

        # Check that the loaded pipeline produces the same transformations
        transformed_loaded = loaded_pipeline.transform(sample_data)
        pd.testing.assert_frame_equal(transformed_original, transformed_loaded)


def test_transform_without_fit():
    """Test that transform raises an error if called before fit."""
    pipeline = PreprocessingPipeline()
    with pytest.raises(ValueError, match="Pipeline has not been fitted"):
        pipeline.transform(pd.DataFrame())


def test_get_feature_names_without_fit():
    """Test that get_feature_names raises an error if called before fit."""
    pipeline = PreprocessingPipeline()
    with pytest.raises(ValueError, match="Pipeline has not been fitted"):
        pipeline.get_feature_names()


def test_feature_order_consistency(sample_data):
    """Test that feature order is consistent between training and inference."""
    pipeline = PreprocessingPipeline()
    pipeline.fit(sample_data)

    # Get feature names
    feature_names = pipeline.get_feature_names()

    # Transform data
    transformed_data = pipeline.transform(sample_data)

    # Check that columns in transformed data match feature names
    feature_cols = [col for col in transformed_data.columns if col != 'target']
    assert feature_cols == feature_names


def test_handle_unseen_categories(sample_data, test_data_with_new_categories):
    """Test handling of unseen categories in test data."""
    # Test with one-hot encoding
    onehot_pipeline = PreprocessingPipeline()
    onehot_pipeline.fit(sample_data)

    transformed_test = onehot_pipeline.transform(test_data_with_new_categories)

    # Check that only columns for categories seen during training were created
    for col in DEFAULT_CATEGORICAL_COLUMNS:
        for category in onehot_pipeline.categorical_preprocessor.categories[col]:
            one_hot_col = f"{col}_{category}"
            assert one_hot_col in transformed_test.columns

        # New categories should not have columns
        for category in set(test_data_with_new_categories[col].unique()) - set(
                onehot_pipeline.categorical_preprocessor.categories[col]):
            one_hot_col = f"{col}_{category}"
            assert one_hot_col not in transformed_test.columns


def test_subset_columns(sample_data):
    """Test that the pipeline works with a subset of columns."""
    # Create a pipeline with a subset of columns
    num_cols = ['age', 'income']
    cat_cols = ['employment_type', 'marital_status']

    pipeline = PreprocessingPipeline(numerical_columns=num_cols, categorical_columns=cat_cols)
    pipeline.fit(sample_data)

    transformed_data = pipeline.transform(sample_data)

    # Check that only the specified columns were transformed
    for col in num_cols:
        assert col in transformed_data.columns

    for col in cat_cols:
        assert col not in transformed_data.columns
        for category in pipeline.categorical_preprocessor.categories[col]:
            assert f"{col}_{category}" in transformed_data.columns

    # Columns not specified should not be in the result
    for col in set(DEFAULT_NUMERICAL_COLUMNS) - set(num_cols):
        assert col not in transformed_data.columns

    for col in set(DEFAULT_CATEGORICAL_COLUMNS) - set(cat_cols):
        assert col not in transformed_data.columns
        for category in sample_data[col].unique():
            assert f"{col}_{category}" not in transformed_data.columns