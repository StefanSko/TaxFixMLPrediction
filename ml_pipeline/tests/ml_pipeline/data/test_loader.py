"""
Tests for the data loader module.
"""

import os
import tempfile
import pandas as pd
import pytest
from pathlib import Path

from ml_pipeline.data.loader import load_data, load_and_split_data

# Create test data
@pytest.fixture
def sample_csv_file():
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        f.write(b"age,income,employment_type,marital_status,time_spent_on_platform,number_of_sessions,fields_filled_percentage,previous_year_filing,device_type,referral_source,completed_filing\n")
        f.write(b"30,50000,full-time,single,120,5,80,True,mobile,search,True\n")
        f.write(b"45,70000,part-time,married,90,3,60,False,desktop,referral,False\n")
        f.write(b"25,40000,freelance,single,150,8,90,True,mobile,social,True\n")
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def empty_csv_file():
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        pass
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def invalid_csv_file():
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        f.write(b"This is not a valid CSV file")
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def non_csv_file():
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"This is a text file, not a CSV")
    yield f.name
    os.unlink(f.name)

def test_load_data_success(sample_csv_file):
    df = load_data(sample_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == [
        'age', 'income', 'employment_type', 'marital_status', 
        'time_spent_on_platform', 'number_of_sessions', 
        'fields_filled_percentage', 'previous_year_filing', 
        'device_type', 'referral_source', 'completed_filing'
    ]

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")

def test_load_data_empty_file(empty_csv_file):
    with pytest.raises(ValueError, match="CSV file is empty"):
        load_data(empty_csv_file)

def test_load_data_invalid_csv(invalid_csv_file):
    with pytest.raises(ValueError, match="Invalid CSV format"):
        load_data(invalid_csv_file)

def test_load_data_not_csv(non_csv_file):
    with pytest.raises(ValueError, match="File must be a CSV"):
        load_data(non_csv_file)

def test_load_and_split_data(sample_csv_file):
    # Test with validation set
    train_df, test_df, val_df = load_and_split_data(
        sample_csv_file, test_size=0.33, validation_size=0.2
    )
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert len(train_df) + len(test_df) + len(val_df) == 3
    
    # Test without validation set
    train_df, test_df, val_df = load_and_split_data(
        sample_csv_file, test_size=0.33, validation_size=None
    )
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert val_df is None
    assert len(train_df) + len(test_df) == 3 