"""
Model service for the inference service.

This module provides functionality for loading and using ML models for prediction.
"""

import joblib
import logging
import pandas as pd
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service for loading and using ML models for prediction.

    This class handles loading the model and preprocessor, and provides methods
    for making predictions.
    """

    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize the model service.

        Args:
            model_path: Path to the trained model file
            preprocessor_path: Path to the preprocessor file

        Raises:
            FileNotFoundError: If the model or preprocessor file is not found
            ValueError: If the model or preprocessor cannot be loaded
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

        # Load model and preprocessor
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Preprocessor loaded from {preprocessor_path}")

    def _load_model(self, model_path: str) -> Any:
        """
        Load the model from a file.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If the model file is not found
            ValueError: If the model cannot be loaded
        """
        try:
            return joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    def _load_preprocessor(self, preprocessor_path: str) -> Any:
        """
        Load the preprocessor from a file.

        Args:
            preprocessor_path: Path to the preprocessor file

        Returns:
            Loaded preprocessor object

        Raises:
            FileNotFoundError: If the preprocessor file is not found
            ValueError: If the preprocessor cannot be loaded
        """
        try:
            return joblib.load(preprocessor_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        except Exception as e:
            raise ValueError(f"Failed to load preprocessor: {str(e)}")

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction using the loaded model.

        Args:
            data: Dictionary containing the input features

        Returns:
            Dictionary containing the prediction results

        Raises:
            ValueError: If the input data is invalid or the prediction fails
        """
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([data])

            # Preprocess the input data
            preprocessed_data = self.preprocessor.transform(input_df)

            # Make prediction
            prediction = self.model.predict(preprocessed_data)[0]

            # Get prediction probability if available
            probability = None
            if hasattr(self.model, 'predict_proba'):
                probability = float(self.model.predict_proba(preprocessed_data)[0, 1])

            # Return prediction results
            result = {
                "prediction": bool(prediction),
                "probability": probability
            }

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Failed to make prediction: {str(e)}")