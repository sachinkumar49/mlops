import numpy as np
from unittest.mock import patch
from sklearn.ensemble import RandomForestRegressor
from src.multi_training import train_and_log_model


@patch("src.dataprocessing.load")
@patch("mlflow.start_run")
@patch("mlflow.log_param")
@patch("mlflow.log_metric")
@patch("mlflow.sklearn.log_model")
def test_train_and_log_model(
    mock_log_model, mock_log_metric, mock_log_param,
    mock_start_run, mock_load
):
    # Prepare mock data
    X_train = np.random.rand(10, 5)
    X_test = np.random.rand(5, 5)
    y_train = np.random.rand(10)
    y_test = np.random.rand(5)

    # Mock the load function to return mock data
    mock_load.return_value = X_train, X_test, y_train, y_test

    # Mock the start_run function to prevent actual run initiation
    # Mock context manager behavior
    mock_start_run.return_value.__enter__.return_value = None

    # Call the function
    model, rmse = train_and_log_model(
        100, 10, X_train, X_test, y_train, y_test)

    # Assertions
    assert isinstance(model, RandomForestRegressor)
    assert isinstance(rmse, float)
    assert rmse > 0

    # Check if MLflow methods were called
    mock_log_param.assert_any_call("n_estimators", 100)
    mock_log_param.assert_any_call("max_depth", 10)
    mock_log_metric.assert_any_call("rmse", rmse)
    mock_log_model.assert_any_call(model, "model")
