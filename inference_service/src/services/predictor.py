"""
Prediction service for the inference API.

This module provides functionality for preprocessing inputs, making predictions,
and formatting outputs using the loaded ML model and preprocessor.
"""

import logging
import time
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, HTTPException, status

from core.models import PredictionRequest, PredictionResponse, ErrorCode
from services.model_loader import get_model, get_preprocessor

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for making predictions using the loaded ML model and preprocessor.

    This class handles the entire prediction pipeline, from input validation
    to output formatting.
    """

    def __init__(self, model: Any, preprocessor: Any):
        """
        Initialize the prediction service.

        Args:
            model: Trained ML model
            preprocessor: Data preprocessor
        """
        self.model = model
        self.preprocessor = preprocessor

        # Extract feature names from preprocessor if available
        self.feature_names = self._extract_feature_names()

        logger.info("Prediction service initialized")

    def _extract_feature_names(self) -> Optional[List[str]]:
        """
        Extract feature names from the preprocessor if available.

        Returns:
            List of feature names or None if not available
        """
        try:
            # Try to get feature names from different types of preprocessors
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                return list(self.preprocessor.get_feature_names_out())
            elif hasattr(self.preprocessor, 'feature_names_in_'):
                return list(self.preprocessor.feature_names_in_)
            elif hasattr(self.preprocessor, 'feature_names_'):
                return list(self.preprocessor.feature_names_)
            else:
                logger.warning("Could not extract feature names from preprocessor")
                return None
        except Exception as e:
            logger.warning(f"Error extracting feature names: {str(e)}")
            return None

    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert API input to DataFrame and preprocess.

        Args:
            input_data: Dictionary containing input features

        Returns:
            Preprocessed DataFrame ready for prediction

        Raises:
            ValueError: If preprocessing fails
        """
        start_time = time.time()

        try:
            # Convert input dictionary to DataFrame
            input_df = pd.DataFrame([input_data])

            # Log input data
            logger.debug(f"Input data: {input_df.to_dict(orient='records')[0]}")

            # Validate input features if feature names are available
            if self.feature_names:
                missing_features = [f for f in self.feature_names if f not in input_df.columns]
                if missing_features:
                    raise ValueError(f"Missing required features: {', '.join(missing_features)}")

            # Apply preprocessor
            processed_data = self.preprocessor.transform(input_df)

            # Convert to DataFrame if the preprocessor returns a numpy array
            if isinstance(processed_data, np.ndarray):
                if self.feature_names and len(self.feature_names) == processed_data.shape[1]:
                    processed_df = pd.DataFrame(processed_data, columns=self.feature_names)
                else:
                    processed_df = pd.DataFrame(processed_data)
            else:
                processed_df = processed_data

            preprocessing_time = time.time() - start_time
            logger.info(f"Preprocessing completed in {preprocessing_time:.4f} seconds")

            return processed_df

        except Exception as e:
            preprocessing_time = time.time() - start_time
            logger.error(f"Preprocessing failed in {preprocessing_time:.4f} seconds: {str(e)}")
            raise ValueError(f"Failed to preprocess input data: {str(e)}")

    def predict(self, processed_input: pd.DataFrame) -> Tuple[bool, float]:
        """
        Generate prediction and probability using the model.

        Args:
            processed_input: Preprocessed DataFrame

        Returns:
            Tuple of (prediction, probability)

        Raises:
            ValueError: If prediction fails
        """
        start_time = time.time()

        try:
            # Make prediction
            prediction = bool(self.model.predict(processed_input)[0])

            # Get prediction probability if available
            probability = 0.5  # Default if predict_proba is not available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_input)[0]
                # Get probability of positive class (index 1 for binary classification)
                probability = float(probabilities[1] if len(probabilities) > 1 else probabilities[0])

            prediction_time = time.time() - start_time
            logger.info(f"Prediction completed in {prediction_time:.4f} seconds")

            return prediction, probability

        except Exception as e:
            prediction_time = time.time() - start_time
            logger.error(f"Prediction failed in {prediction_time:.4f} seconds: {str(e)}")
            raise ValueError(f"Failed to generate prediction: {str(e)}")

    def format_output(self, prediction: bool, probability: float) -> Dict[str, Any]:
        """
        Format prediction results for API response.

        Args:
            prediction: Boolean prediction (True/False)
            probability: Prediction probability

        Returns:
            Dictionary containing formatted prediction results
        """
        # Create response dictionary
        response = {
            "prediction": prediction,
            "completion_probability": probability
        }

        logger.debug(f"Prediction output: {response}")
        return response

    def process_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a prediction request through the full pipeline.

        Args:
            input_data: Dictionary containing input features

        Returns:
            Dictionary containing prediction results

        Raises:
            ValueError: If any step in the pipeline fails
        """
        overall_start_time = time.time()

        try:
            # Preprocess input data
            processed_input = self.preprocess_input(input_data)

            # Generate prediction
            prediction, probability = self.predict(processed_input)

            # Format output
            result = self.format_output(prediction, probability)

            overall_time = time.time() - overall_start_time
            logger.info(f"Request processed successfully in {overall_time:.4f} seconds")

            return result

        except Exception as e:
            overall_time = time.time() - overall_start_time
            logger.error(f"Request processing failed in {overall_time:.4f} seconds: {str(e)}")
            raise ValueError(f"Failed to process prediction request: {str(e)}")


def get_prediction_service(
        model: Any = Depends(get_model),
        preprocessor: Any = Depends(get_preprocessor)
) -> PredictionService:
    """
    FastAPI dependency to get the prediction service.

    Args:
        model: ML model from dependency injection
        preprocessor: Data preprocessor from dependency injection

    Returns:
        PredictionService instance
    """
    return PredictionService(model=model, preprocessor=preprocessor)


def process_prediction_request(
        request_data: PredictionRequest,
        prediction_service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Process a prediction request and return a formatted response.

    Args:
        request_data: Validated request data
        prediction_service: PredictionService instance

    Returns:
        Formatted prediction response

    Raises:
        HTTPException: If prediction processing fails
    """
    try:
        # Convert Pydantic model to dictionary
        input_data = request_data.dict()

        # Process the request
        result = prediction_service.process_request(input_data)

        # Create response
        response = PredictionResponse(
            prediction=result["prediction"],
            completion_probability=result["completion_probability"]
        )

        return response

    except ValueError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": ErrorCode.PREDICTION_ERROR,
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR,
                "message": "An unexpected error occurred during prediction"
            }
        )