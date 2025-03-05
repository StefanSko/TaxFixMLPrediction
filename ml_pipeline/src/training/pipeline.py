"""
Model Training Pipeline for the ML pipeline.

This module provides a complete pipeline for training, evaluating, and saving
machine learning models for predicting user behavior.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from training.classifier import TaxfixClassifier
from preprocessing.pipeline import PreprocessingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    End-to-end pipeline for training and evaluating classification models.

    This class handles data loading, preprocessing, model training, evaluation,
    and artifact storage.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the training pipeline with configuration parameters.

        Args:
            config: Dictionary containing configuration parameters for the pipeline
        """
        self.config = config
        self.validate_config()

        # Set random seed for reproducibility
        np.random.seed(self.config.get('random_seed', 42))

        # Initialize preprocessor
        self.preprocessor = None

    def validate_config(self) -> None:
        """
        Validate that the configuration has all required fields.

        Raises:
            ValueError: If required configuration is missing
        """
        required_fields = ['data_path', 'model_config', 'preprocessing_config', 'storage_config']

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        # Validate model config
        model_config = self.config['model_config']
        if 'type' not in model_config:
            raise ValueError("Model configuration must include 'type'")

        # Validate storage config
        storage_config = self.config['storage_config']
        if 'model_path' not in storage_config:
            raise ValueError("Storage configuration must include 'model_path'")

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load data from the specified path and preprocess it for training.

        Returns:
            Tuple containing training and testing data splits (X_train, y_train, X_test, y_test)

        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data doesn't contain the target column
        """
        logger.info("Loading data from %s", self.config['data_path'])

        # Load data
        data_path = Path(self.config['data_path'])
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Determine file type and load accordingly
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        logger.info("Loaded data with shape: %s", df.shape)

        # Extract target variable
        target_col = self.config['preprocessing_config'].get('target_column', 'target')
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Handle feature selection if specified
        feature_cols = self.config['preprocessing_config'].get('feature_columns')
        if feature_cols:
            missing_cols = set(feature_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Feature columns not found in data: {missing_cols}")
            X = X[feature_cols]

        logger.info("Using %d features for training", X.shape[1])

        # Split data
        test_size = self.config['preprocessing_config'].get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.config.get('random_seed', 42),
            stratify=y if self.config['preprocessing_config'].get('stratify', True) else None
        )

        # Apply preprocessing
        preprocessing_config = self.config['preprocessing_config']
        if preprocessing_config.get('use_preprocessing_pipeline', False):
            logger.info("Applying preprocessing pipeline")
            numerical_cols = preprocessing_config.get('numerical_columns', [])
            categorical_cols = preprocessing_config.get('categorical_columns', [])
            
            # Initialize and fit the preprocessing pipeline
            self.preprocessor = PreprocessingPipeline(
                numerical_columns=numerical_cols,
                categorical_columns=categorical_cols
            )
            
            # Add target column temporarily for preprocessing
            X_train_with_target = X_train.copy()
            X_train_with_target[target_col] = y_train
            
            # Fit and transform training data
            X_train_processed = self.preprocessor.fit_transform(X_train_with_target)
            
            # Remove target column from processed data
            if target_col in X_train_processed.columns:
                X_train_processed = X_train_processed.drop(columns=[target_col])
            
            # Transform test data
            X_test_with_target = X_test.copy()
            X_test_with_target[target_col] = y_test
            X_test_processed = self.preprocessor.transform(X_test_with_target)
            
            # Remove target column from processed data
            if target_col in X_test_processed.columns:
                X_test_processed = X_test_processed.drop(columns=[target_col])
            
            X_train, X_test = X_train_processed, X_test_processed
        
        logger.info("Data preprocessing complete. Train shape: %s, Test shape: %s",
                    X_train.shape, X_test.shape)

        return X_train, y_train, X_test, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train a model using the provided training data.

        Args:
            X_train: Training features
            y_train: Training target values

        Returns:
            Trained model object
        """
        model_config = self.config['model_config']
        model_type = model_config['type']
        model_params = model_config.get('params', {})

        logger.info("Training %s model with parameters: %s", model_type, model_params)

        # Initialize and train the model
        model = TaxfixClassifier(model_type=model_type, **model_params)
        model.fit(X_train, y_train)

        logger.info("Model training complete")
        return model

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.

        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test target values

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance")

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
        }

        # Add ROC AUC if probability predictions are available
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)

        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def save_artifacts(self, model: Any, preprocessor: Optional[Any],
                       evaluation_results: Dict[str, Any]) -> None:
        """
        Save model, preprocessor, and evaluation results.

        Args:
            model: Trained model object
            preprocessor: Fitted preprocessor object (if any)
            evaluation_results: Dictionary of evaluation metrics
        """
        storage_config = self.config['storage_config']

        # Create output directory if it doesn't exist
        output_dir = Path(storage_config.get('output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = Path(storage_config['model_path'])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving model to %s", model_path)
        model.save(str(model_path))

        # Save preprocessor if it exists
        if preprocessor is not None:
            preprocessor_path = output_dir / 'preprocessor.joblib'
            logger.info("Saving preprocessor to %s", preprocessor_path)
            import joblib
            joblib.dump(preprocessor, preprocessor_path)

        # Save evaluation results
        results_path = output_dir / 'evaluation_results.json'
        logger.info("Saving evaluation results to %s", results_path)
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        # Save configuration
        config_path = output_dir / 'training_config.json'
        logger.info("Saving configuration to %s", config_path)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary containing evaluation metrics and paths to saved artifacts
        """
        logger.info("Starting model training pipeline")

        try:
            # Load and preprocess data
            X_train, y_train, X_test, y_test = self.load_and_preprocess_data()

            # Train model
            model = self.train_model(X_train, y_train)

            # Evaluate model
            evaluation_results = self.evaluate_model(model, X_test, y_test)

            # Save artifacts
            self.save_artifacts(model, self.preprocessor, evaluation_results)

            # Return results
            results = {
                'metrics': evaluation_results,
                'model_path': self.config['storage_config']['model_path'],
                'config': self.config
            }

            logger.info("Model training pipeline completed successfully")
            return results

        except Exception as e:
            logger.error("Error in training pipeline: %s", str(e), exc_info=True)
            raise


def train_model_from_config(config_path: str) -> Dict[str, Any]:
    """
    Train a model using configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Dictionary containing training results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create and run pipeline
    pipeline = ModelTrainingPipeline(config)
    results = pipeline.run()

    return results