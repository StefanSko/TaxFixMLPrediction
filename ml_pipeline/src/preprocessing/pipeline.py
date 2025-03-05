"""
Integrated preprocessing pipeline module for the ML pipeline.
This module provides functionality to combine numerical and categorical
preprocessing into a single pipeline.
"""

from typing import List, Optional
import pandas as pd
import os
import joblib
from pathlib import Path

from preprocessing.numerical import NumericalPreprocessor, DEFAULT_NUMERICAL_COLUMNS
from preprocessing.categorical import CategoricalPreprocessor, DEFAULT_CATEGORICAL_COLUMNS


class PreprocessingPipeline:
    """
    Integrated preprocessing pipeline that combines numerical and categorical preprocessing.

    This class handles the preprocessing of both numerical and categorical features
    and combines them into a single output suitable for model training.
    """

    def __init__(
            self,
            numerical_columns: List[str] = DEFAULT_NUMERICAL_COLUMNS,
            categorical_columns: List[str] = DEFAULT_CATEGORICAL_COLUMNS
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            numerical_columns: List of column names containing numerical features
            categorical_columns: List of column names containing categorical features
        """
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
        # Initialize preprocessors
        self.numerical_preprocessor = NumericalPreprocessor(numerical_columns)
        self.categorical_preprocessor = CategoricalPreprocessor(categorical_columns)
        
        # Track feature names after preprocessing
        self._feature_names: Optional[List[str]] = None
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit both preprocessors on the data.

        Args:
            df: DataFrame containing both numerical and categorical columns to fit
        """
        # Fit numerical preprocessor
        self.numerical_preprocessor.fit(df)

        # Fit categorical preprocessor
        self.categorical_preprocessor.fit(df)

        # Generate and store feature names
        numerical_features = self.numerical_columns.copy()
        categorical_features = []
        for col in self.categorical_columns:
            for category in self.categorical_preprocessor.categories[col]:
                categorical_features.append(f"{col}_{category}")
        
        self._feature_names = numerical_features + categorical_features
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply both preprocessing steps to data.

        Args:
            df: DataFrame containing both numerical and categorical columns to transform

        Returns:
            DataFrame with transformed features
        """
        if not self.fitted:
            raise ValueError("Pipeline has not been fitted. Call fit() first.")

        # Transform numerical features
        numerical_df = self.numerical_preprocessor.transform(df)

        # Transform categorical features
        categorical_df = self.categorical_preprocessor.transform(df)

        # Start with a fresh DataFrame containing only numerical features
        result_df = pd.DataFrame()
        
        # Add numerical features
        for col in self.numerical_columns:
            result_df[col] = numerical_df[col]
        
        # Add one-hot encoded categorical features
        for col in self.categorical_columns:
            for category in self.categorical_preprocessor.categories[col]:
                encoded_col = f"{col}_{category}"
                if encoded_col in categorical_df.columns:
                    result_df[encoded_col] = categorical_df[encoded_col]
        
        # Preserve the target column if it exists
        if 'target' in df.columns:
            result_df['target'] = df['target']
        
        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine fit and transform operations.

        Args:
            df: DataFrame containing both numerical and categorical columns to fit and transform

        Returns:
            DataFrame with transformed features
        """
        self.fit(df)
        return self.transform(df)

    def save(self, directory: str) -> None:
        """
        Save the entire pipeline to a directory.

        Args:
            directory: Directory where the pipeline components should be saved
        """
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Save numerical preprocessor
        self.numerical_preprocessor.save(os.path.join(directory, "numerical_preprocessor.joblib"))

        # Save categorical preprocessor
        self.categorical_preprocessor.save(os.path.join(directory, "categorical_preprocessor.joblib"))

        # Save pipeline metadata
        metadata = {
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "feature_names": self._feature_names,
            "fitted": self.fitted
        }
        joblib.dump(metadata, os.path.join(directory, "pipeline_metadata.joblib"))

    @classmethod
    def load(cls, directory: str) -> "PreprocessingPipeline":
        """
        Load a pipeline from a directory.

        Args:
            directory: Directory from which to load the pipeline components

        Returns:
            Loaded PreprocessingPipeline instance
        """
        # Load pipeline metadata
        metadata_path = os.path.join(directory, "pipeline_metadata.joblib")
        metadata = joblib.load(metadata_path)

        # Create pipeline instance
        pipeline = cls(
            numerical_columns=metadata["numerical_columns"],
            categorical_columns=metadata["categorical_columns"]
        )

        # Load numerical preprocessor
        numerical_path = os.path.join(directory, "numerical_preprocessor.joblib")
        pipeline.numerical_preprocessor = NumericalPreprocessor.load(numerical_path)

        # Load categorical preprocessor
        categorical_path = os.path.join(directory, "categorical_preprocessor.joblib")
        pipeline.categorical_preprocessor = CategoricalPreprocessor.load(categorical_path)

        # Restore feature names and fitted status
        pipeline._feature_names = metadata["feature_names"]
        pipeline.fitted = metadata["fitted"]

        return pipeline

    def get_feature_names(self) -> List[str]:
        """
        Get the names of features after preprocessing.

        Returns:
            List of feature names after preprocessing
        """
        if not self.fitted:
            raise ValueError("Pipeline has not been fitted. Call fit() first.")

        return self._feature_names if self._feature_names else []