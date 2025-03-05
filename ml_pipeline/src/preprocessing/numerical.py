"""
Numerical feature preprocessing module for the ML pipeline.
This module provides functionality to preprocess numerical features
using standard scaling techniques.
"""

from typing import List, Dict
import pandas as pd
import joblib
from pathlib import Path


class NumericalPreprocessor:
    """
    Preprocessor for numerical features that applies standard scaling.

    This class handles the preprocessing of numerical features by standardizing
    them to have zero mean and unit variance.
    """

    def __init__(self, numerical_columns: List[str]):
        """
        Initialize the numerical preprocessor.

        Args:
            numerical_columns: List of column names containing numerical features
        """
        self.numerical_columns = numerical_columns
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Learn scaling parameters from training data.

        Args:
            df: DataFrame containing the numerical columns to fit
        """
        for col in self.numerical_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame")

            self.means[col] = df[col].mean()
            self.stds[col] = df[col].std()

            # Handle zero standard deviation
            if self.stds[col] == 0:
                self.stds[col] = 1.0

        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling to data.

        Args:
            df: DataFrame containing the numerical columns to transform

        Returns:
            DataFrame with transformed numerical features
        """
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted. Call fit() first.")

        result = df.copy()

        for col in self.numerical_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame")

            result[col] = (df[col] - self.means[col]) / self.stds[col]

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine fit and transform operations.

        Args:
            df: DataFrame containing the numerical columns to fit and transform

        Returns:
            DataFrame with transformed numerical features
        """
        self.fit(df)
        return self.transform(df)

    def save(self, path: str) -> None:
        """
        Save the preprocessor to a file.

        Args:
            path: Path where the preprocessor should be saved
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "NumericalPreprocessor":
        """
        Load a preprocessor from a file.

        Args:
            path: Path from which to load the preprocessor

        Returns:
            Loaded NumericalPreprocessor instance
        """
        return joblib.load(path)


# Default numerical columns to use if not specified
DEFAULT_NUMERICAL_COLUMNS = [
    'age',
    'income',
    'time_spent_on_platform',
    'number_of_sessions',
    'fields_filled_percentage'
]