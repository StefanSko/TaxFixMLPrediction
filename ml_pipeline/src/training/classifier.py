"""
Classifier module for the ML pipeline.
This module provides functionality to train and use classification models
for predicting user behavior.
"""

from typing import Union, Any, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression


class TaxfixClassifier:
    """
    Classifier for predicting user behavior.

    This class provides a unified interface for training and using different
    classification models, with initial support for logistic regression.
    """

    # Default parameters for each supported model type
    DEFAULT_PARAMS = {
        'logistic_regression': {
            'C': 1.0,
            'max_iter': 100,
            'random_state': 42
        }
    }

    SUPPORTED_MODELS = list(DEFAULT_PARAMS.keys())

    def __init__(self, model_type: str = 'logistic_regression', **model_params) -> None:
        """
        Initialize the classifier.

        Args:
            model_type: Type of model to use ('logistic_regression', 'random_forest', or 'svm')
            **model_params: Parameters to pass to the model constructor
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                             f"Supported types are: {', '.join(self.SUPPORTED_MODELS)}")

        self.model_type = model_type

        # Combine default parameters with user-provided parameters
        self.model_params = self.DEFAULT_PARAMS[model_type].copy()
        self.model_params.update(model_params)

        # Initialize the model based on the type
        self.model = self._create_model()

        # Track feature names used during training
        self.feature_names: Optional[list] = None
        self.fitted = False

    def _create_model(self) -> Any:
        """
        Create a new model instance based on the model type.

        Returns:
            A new model instance
        """
        if self.model_type == 'logistic_regression':
            return LogisticRegression(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the model on the provided data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X

        # Convert y to numpy array if it's a pandas Series
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Train the model
        self.model.fit(X_array, y_array)
        self.fitted = True

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make binary predictions using the trained model.

        Args:
            X: Feature matrix

        Returns:
            Array of binary predictions
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Convert to numpy array if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            # Check if feature names match those used during training
            if self.feature_names and set(X.columns) != set(self.feature_names):
                raise ValueError("Feature names in prediction data do not match those used during training")
            X_array = X.values
        else:
            X_array = X

        return self.model.predict(X_array)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using the trained model.

        Args:
            X: Feature matrix

        Returns:
            Array of class probabilities
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Convert to numpy array if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            # Check if feature names match those used during training
            if self.feature_names and set(X.columns) != set(self.feature_names):
                raise ValueError("Feature names in prediction data do not match those used during training")
            X_array = X.values
        else:
            X_array = X

        return self.model.predict_proba(X_array)

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path: Path where the model should be saved
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare metadata to save
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'fitted': self.fitted
        }

        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: str) -> "TaxfixClassifier":
        """
        Load a model from a file.

        Args:
            path: Path from which to load the model

        Returns:
            Loaded TaxfixClassifier instance
        """
        # Load model data
        model_data = joblib.load(path)

        # Create a new instance with the same model type and parameters
        classifier = cls(model_type=model_data['model_type'], **model_data['model_params'])

        # Restore the trained model and metadata
        classifier.model = model_data['model']
        classifier.feature_names = model_data['feature_names']
        classifier.fitted = model_data['fitted']

        return classifier
