import pandas as pd
from unittest.mock import patch
from src.dataprocessing import (
    load_data, define_category_column,
    preprocess_data, load
)


@patch("pandas.read_csv")
def test_load_data(mock_read_csv):
    # Create a mock DataFrame
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
    mock_read_csv.return_value = mock_data

    # Call the function
    data = load_data("./data/diabetes_dataset.csv")

    # Assert the DataFrame is returned
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2, 11)
    mock_read_csv.assert_called_once_with("./data/diabetes_dataset.csv")


def test_define_category_column():
    # Create a mock dataset similar to the columns mentioned
    data = {
        "age": [25, 30],
        "sex": [1, 0],
        "bmi": [22.5, 27.3],
        "bp": [80, 90],
        "s1": [0.1, 0.2],
        "s2": [0.2, 0.3],
        "s3": [0.3, 0.4],
        "s4": [0.4, 0.5],
        "s5": [0.5, 0.6],
        "s6": [0.6, 0.7],
        "target": [1, 0]
    }
    df = pd.DataFrame(data)
    # Call the function to define the target and features
    X, y = define_category_column(df)
    # Assert that the target is correctly separated and
    # does not appear in features (X)
    assert "target" not in X.columns
    assert y.name == "target"


def test_preprocess_data():
    # Create mock dataset
    data = {
        "age": [25, 30],
        "sex": [1, 0],
        "bmi": [22.5, 27.3],
        "bp": [80, 90],
        "s1": [0.1, 0.2],
        "s2": [0.2, 0.3],
        "s3": [0.3, 0.4],
        "s4": [0.4, 0.5],
        "s5": [0.5, 0.6],
        "s6": [0.6, 0.7],
        "target": [1, 0]
    }
    df = pd.DataFrame(data)
    # Preprocess data (you can customize the function as needed)
    processed_data = preprocess_data(df)
    # Assuming 'bmi' and other numerical features were processed,
    # we can assert:
    assert "target" in processed_data.columns  # Target column should remain
    assert processed_data.isnull().sum().sum() == 0  # No missing values


def test_load():
    # Create a mock dataset
    data = {
        "age": [25, 30],
        "sex": [1, 0],
        "bmi": [22.5, 27.3],
        "bp": [80, 90],
        "s1": [0.1, 0.2],
        "s2": [0.2, 0.3],
        "s3": [0.3, 0.4],
        "s4": [0.4, 0.5],
        "s5": [0.5, 0.6],
        "s6": [0.6, 0.7],
        "target": [1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv("test_data.csv", index=False)
    # Load data using the load function from src.dataprocessing
    X_train, X_test, y_train, y_test = load('test_data.csv')
    # Assertions
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0
