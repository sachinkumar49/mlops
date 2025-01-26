import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.hyper_parameter import objective, main
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


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


@patch("src.hyper_parameter.load")
@patch("src.hyper_parameter.save_model")
@patch("optuna.create_study")
@patch("src.hyper_parameter.objective")
def test_main(mock_objective, mock_create_study, mock_save_model, mock_load):
    """Test the main function for hyperparameter optimization."""

    # Mocking the study object
    mock_study = MagicMock()
    mock_study.best_trial.params = {"n_estimators": 150, "max_depth": 25,
                                    "min_samples_split": 4}
    mock_study.best_value = 1.5
    mock_create_study.return_value = mock_study

    # Mock the objective function
    mock_objective.return_value = 1.5

    # Mock the dataset (X_train, X_test, y_train, y_test)
    X_train = np.random.rand(10, 5)
    X_test = np.random.rand(5, 5)
    y_train = np.random.rand(10)
    y_test = np.random.rand(5)

    # Mock the load function to return the dataset
    mock_load.return_value = (X_train, X_test, y_train, y_test)

    # Create a mock model (RandomForestRegressor in this case)
    best_model = RandomForestRegressor(n_estimators=150, max_depth=25,
                                       min_samples_split=4)
    mock_study.best_trial.user_attrs = {"model": best_model}

    # Mock fit and predict
    best_model.fit = MagicMock()
    best_model.predict = MagicMock(return_value=np.random.rand(5))

    # Run the main function
    with patch("builtins.print") as mock_print:
        main()

        # Verify that the study was created and optimized
        mock_create_study.assert_called_once_with(direction="minimize")
        mock_study.optimize.assert_called_once_with(mock_objective,
                                                    n_trials=100)

        # Check that the best model was fitted
        best_model.fit.assert_called_once_with(X_train, y_train)

        # Verify the final RMSE calculation
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mock_print.assert_any_call(f"Final RMSE: {rmse}")

        # Check if the model was saved
        mock_save_model.assert_called_once_with(best_model, "model.pkl")

        # Check the printed messages for correct output
        mock_print.assert_any_call("Model saved as 'model.pkl'.")
        mock_print.assert_any_call(f"Best hyperparameters: {
            mock_study.best_trial.params}")
        mock_print.assert_any_call(f"Best RMSE: {mock_study.best_value}")
