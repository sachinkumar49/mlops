import pytest
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.dataprocessing import load_data, preprocess_data
from src.model_train import (
    train_model,
    save_model,
    main
)
from unittest.mock import patch


@pytest.fixture
def mock_data():
    """Fixture to return mock training and test data."""
    X_train = np.array([[23, 1, 25.3, 80, 1, 0, 1, 0, 0, 1],
                        [45, 0, 30.2, 70, 0, 1, 0, 1, 0, 0],
                        [50, 1, 22.5, 60, 1, 1, 1, 0, 1, 1]])
    X_test = np.array([[60, 0, 28.5, 75, 1, 0, 1, 1, 0, 0],
                       [30, 1, 27.1, 85, 0, 1, 0, 0, 1, 1]])
    y_train = np.array([150, 130, 160])
    y_test = np.array([140, 135])
    return X_train, X_test, y_train, y_test


def test_load_data():
    """Test loading data from a CSV file."""
    data = {
        "age": [23, 45],
        "sex": [1, 0],
        "bmi": [25.3, 30.2],
        "bp": [80, 70],
        "s1": [1, 0],
        "s2": [0, 1],
        "s3": [1, 0],
        "s4": [0, 1],
        "s5": [0, 0],
        "s6": [1, 0],
        "target": [150, 130]
    }
    mock_data = pd.DataFrame(data)
    mock_data.to_csv("test_data.csv", index=False)
    loaded_data = load_data("test_data.csv")
    assert isinstance(loaded_data, pd.DataFrame)
    assert loaded_data.shape == (2, 11)
    os.remove("test_data.csv")  # Clean up after test


def test_preprocess_data():
    """Test preprocessing of the data."""
    data = {
        "age": [23, 45],
        "sex": [1, 0],
        "bmi": [25.3, 30.2],
        "bp": [80, 70],
        "s1": [1, 0],
        "s2": [0, 1],
        "s3": [1, 0],
        "s4": [0, 1],
        "s5": [0, 0],
        "s6": [1, 0],
        "target": [150, 130]
    }
    df = pd.DataFrame(data)
    processed_data = preprocess_data(df)
    assert "target" in processed_data.columns
    assert processed_data.isnull().sum().sum() == 0


def test_train_model(mock_data):
    """Test the model training function."""
    X_train, X_test, y_train, y_test = mock_data
    model, rmse = train_model(X_train, X_test, y_train, y_test)
    assert isinstance(model, RandomForestRegressor)
    assert rmse >= 0


def test_save_model():
    """Test the saving of the model."""
    model = RandomForestRegressor()
    save_model(model, "test_model.pkl")
    # Check if the model file is created
    assert os.path.exists("test_model.pkl")
    # Clean up after test
    os.remove("test_model.pkl")


@patch("src.model_train.load", return_value=(np.array(
    [[23, 1, 25.3, 80, 1, 0, 1, 0, 0, 1]]), np.array(
        [[60, 0, 28.5, 75, 1, 0, 1, 1, 0, 0]]),
    np.array([150]), np.array([140])))
@patch("src.model_train.save_model")
@patch("builtins.print")
def test_main(mock_print, mock_save_model, mock_load):
    """Test the main function for running the model training and saving."""
    main()

    # Ensure that the print statement was called
    mock_print.assert_any_call("Model saved as 'model.pkl'.")
    # Ensure that the model is saved using the save_model function
    mock_save_model.assert_called_once()
