import pytest
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.model_train import (
    train_model,
    save_model,
    main
)
from unittest.mock import patch


@pytest.fixture
def mock_dataset_path(tmp_path):
    # Create a temporary mock dataset
    mock_file = tmp_path / "diabetes_dataset.csv"
    mock_data = pd.DataFrame({
        "age": [25, 30],
        "sex": [1, 2],
        "bmi": [22.0, 25.5],
        "bp": [80, 85],
        "s1": [150, 160],
        "s2": [100, 110],
        "s3": [20, 30],
        "s4": [0.5, 0.6],
        "s5": [4.0, 5.0],
        "s6": [90, 95],
        "target": [100, 200],
    })
    mock_data.to_csv(mock_file, index=False)
    return str(mock_file)


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
