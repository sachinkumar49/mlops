import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.hyper_parameter import objective, main


@pytest.fixture
def mock_data():
    """Fixture to return mock training and test data."""
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    X_test = np.array([[10, 11, 12], [13, 14, 15]])
    y_train = np.array([1, 2, 3])
    y_test = np.array([4, 5])
    return X_train, X_test, y_train, y_test


@patch("src.hyper_parameter.load", return_value=(np.array(
    [[23, 1, 25.3, 80, 1, 0, 1, 0, 0, 1]]), np.array(
        [[60, 0, 28.5, 75, 1, 0, 1, 1, 0, 0]]),
    np.array([150]), np.array([140])))
def test_objective(mock_load, mock_data):
    """Test the objective function with mock data."""
    mock_load.return_value = mock_data

    # Mock Optuna's trial object
    mock_trial = MagicMock()
    mock_trial.suggest_int.side_effect = [100, 20, 5]

    rmse = objective(mock_trial)

    # Verify RMSE is calculated correctly
    assert isinstance(rmse, float), "RMSE should be a float"
    assert rmse > 0, "RMSE should be greater than 0"

    # Verify hyperparameter suggestions
    mock_trial.suggest_int.assert_any_call("n_estimators", 50, 200)
    mock_trial.suggest_int.assert_any_call("max_depth", 10, 30)
    mock_trial.suggest_int.assert_any_call("min_samples_split", 2, 10)


@patch("src.hyper_parameter.objective")
@patch("optuna.create_study")
def test_main(mock_create_study, mock_objective):
    """Test the main function for hyperparameter optimization."""
    # Mock the study object
    mock_study = MagicMock()
    mock_study.best_trial.params = {"n_estimators": 150, "max_depth": 25,
                                    "min_samples_split": 4}
    mock_study.best_value = 1.5
    mock_create_study.return_value = mock_study

    main()

    # Verify the study is created and optimized
    mock_create_study.assert_called_once_with(direction="minimize")
    mock_study.optimize.assert_called_once_with(mock_objective, n_trials=100)

    # Verify best parameters and RMSE
    assert mock_study.best_trial.params["n_estimators"] == 150
    assert mock_study.best_value == 1.5
