
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.model_train import (
    load_data,
    preprocess_data,
    train_model,
    save_model,
    main
)
from unittest.mock import patch


def test_load_data():
    # Create a mock dataset
    data = {
        "InvoiceNo": ["1", "2"],
        "StockCode": ["A", "B"],
        "Description": ["Item1", "Item2"],
        "Quantity": [10, 5],
        "InvoiceDate": ["2023-01-01", "2023-01-02"],
        "UnitPrice": [2.5, 5.0],
        "CustomerID": ["C1", "C2"],
        "Country": ["USA", "UK"]
    }
    mock_data = pd.DataFrame(data)
    mock_data.to_csv("test_data.csv", index=False)
    # Load the data
    loaded_data = load_data("test_data.csv")
    assert isinstance(loaded_data, pd.DataFrame)
    assert loaded_data.shape == (2, 8)


def test_preprocess_data():
    data = {
        "InvoiceNo": ["1", "2"],
        "StockCode": ["A", "B"],
        "Description": ["Item1", "Item2"],
        "Quantity": [10, 5],
        "InvoiceDate": ["2023-01-01", "2023-01-02"],
        "UnitPrice": [2.5, 5.0],
        "CustomerID": ["C1", "C2"],
        "Country": ["USA", "UK"]
    }
    df = pd.DataFrame(data)
    processed_data = preprocess_data(df)
    assert "TotalSales" in processed_data.columns
    assert processed_data.isnull().sum().sum() == 0


def test_train_model():
    data = {
        "Quantity": [10, 5],
        "UnitPrice": [2.5, 5.0],
        "StockCode": [1, 2],
        "CustomerID": [1, 2],
        "Country": [1, 2],
        "Year": [2023, 2023],
        "Month": [1, 1],
        "Day": [1, 2],
        "Hour": [0, 0],
        "TotalSales": [25.0, 25.0]
    }
    df = pd.DataFrame(data)
    X = df.drop(columns=['TotalSales'])
    y = df['TotalSales']
    model, rmse = train_model(X, y)
    assert isinstance(model, RandomForestRegressor)
    assert rmse >= 0


def test_save_model():
    model = RandomForestRegressor()
    save_model(model, "test_model.pkl")
    assert open("test_model.pkl", "rb") is not None


def test_main(mocker):
    mocker.patch("src.model_train.load_data", return_value=pd.DataFrame({
        'Quantity': [1, 2],
        'UnitPrice': [10.0, 20.0],
        'InvoiceDate': ['2023-01-01 10:00:00', '2023-01-02 11:00:00'],
        'StockCode': ['S001', 'S002'],
        'CustomerID': ['C001', 'C002'],
        'Country': ['UK', 'USA'],
        'InvoiceNo': ['INV001', 'INV002'],  # Added InvoiceNo
        'Description': ['Item1', 'Item2']
    }))
    mocker.patch("src.model_train.save_model")
    with patch("builtins.print") as mock_print:
        main()
    mock_print.assert_any_call("Model saved as 'model.pkl'.")
