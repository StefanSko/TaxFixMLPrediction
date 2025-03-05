"""
Categorical feature preprocessing module for the ML pipeline.
This module provides functionality to preprocess categorical features
using one-hot encoding.
"""

from typing import List, Dict, Any
import pandas as pd
import joblib
from pathlib import Path


class CategoricalPreprocessor:
    """
    Preprocessor for categorical features that applies one-hot encoding.

    This class handles the preprocessing of categorical features by applying
    one-hot encoding.
    """

    def __init__(self, categorical_columns: List[str]):
        """
        Initialize the categorical preprocessor.

        Args:
            categorical_columns: List of column names containing categorical features
        """
        self.categorical_columns = categorical_columns
        self.categories: Dict[str, List[Any]] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Learn categories from training data.

        Args:
            df: DataFrame containing the categorical columns to fit
        """
        for col in self.categorical_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame")

            # Get unique categories and sort them for deterministic behavior
            unique_values = sorted(df[col].dropna().unique().tolist())
            self.categories[col] = unique_values

        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to data.

        Args:
            df: DataFrame containing the categorical columns to transform

        Returns:
            DataFrame with one-hot encoded categorical features
        """
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted. Call fit() first.")

        result = df.copy()

        for col in self.categorical_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame")

            # Apply one-hot encoding
            for category in self.categories[col]:
                col_name = f"{col}_{category}"
                result[col_name] = (df[col] == category).astype(int)

            # Drop the original column
            result = result.drop(col, axis=1)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine fit and transform operations.

        Args:
            df: DataFrame containing the categorical columns to fit and transform

        Returns:
            DataFrame with one-hot encoded categorical features
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
    def load(cls, path: str) -> "CategoricalPreprocessor":
        """
        Load a preprocessor from a file.

        Args:
            path: Path from which to load the preprocessor

        Returns:
            Loaded CategoricalPreprocessor instance
        """
        return joblib.load(path)


# Default categorical columns to use if not specified
DEFAULT_CATEGORICAL_COLUMNS = [
    'employment_type',
    'marital_status',
    'device_type',
    'referral_source',
    'previous_year_filing'
]