"""
Data loading utilities for the ML pipeline.

"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a CSV file or has invalid format
    """
    file_path = Path(data_path)
    
    # Check if file exists
    if not file_path.exists():
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Check if file is a CSV
    if file_path.suffix.lower() != '.csv':
        logger.error(f"File is not a CSV: {data_path}")
        raise ValueError(f"File must be a CSV, got: {file_path.suffix}")
    
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {data_path}")
        raise ValueError(f"CSV file is empty: {data_path}")
    except pd.errors.ParserError:
        logger.error(f"Invalid CSV format: {data_path}")
        raise ValueError(f"Invalid CSV format: {data_path}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def load_and_split_data(
    data_path: str, 
    test_size: float = 0.2, 
    validation_size: Optional[float] = 0.1,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load data and split it into training, testing, and optionally validation sets.
    
    Args:
        data_path: Path to the CSV file
        test_size: Proportion of data to use for testing (default: 0.2)
        validation_size: Proportion of data to use for validation (default: 0.1)
                         If None, no validation set is created
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df, val_df) DataFrames
        If validation_size is None, val_df will be None
    """
    from sklearn.model_selection import train_test_split
    
    df = load_data(data_path)
    
    # First split: separate test set
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Second split: create validation set from training data if requested
    if validation_size is not None:
        # Adjust validation size relative to the remaining training data
        adjusted_val_size = validation_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_df, test_size=adjusted_val_size, random_state=random_state
        )
        logger.info(f"Data split into {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
        return train_df, test_df, val_df
    
    logger.info(f"Data split into {len(train_df)} training and {len(test_df)} test samples")
    return train_df, test_df, None