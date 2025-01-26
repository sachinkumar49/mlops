from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    data['TotalSales'] = data['Quantity'] * data['UnitPrice']
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['Year'] = data['InvoiceDate'].dt.year
    data['Month'] = data['InvoiceDate'].dt.month
    data['Day'] = data['InvoiceDate'].dt.day
    data['Hour'] = data['InvoiceDate'].dt.hour
    data = data.drop(columns=['InvoiceNo', 'Description', 'InvoiceDate'])
    data = data.dropna()

    le_stock = LabelEncoder()
    le_customer = LabelEncoder()
    le_country = LabelEncoder()

    data['StockCode'] = le_stock.fit_transform(data['StockCode'])
    data['CustomerID'] = le_customer.fit_transform(data['CustomerID'])
    data['Country'] = le_country.fit_transform(data['Country'])

    return data


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, rmse


def save_model(model, file_name):
    joblib.dump(model, file_name)


def main():
    data = load_data('./data/online_retail.csv')
    data = preprocess_data(data)
    X = data.drop(columns=['TotalSales'])
    y = data['TotalSales']
    model, rmse = train_model(X, y)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    save_model(model, 'model.pkl')
    print("Model saved as 'model.pkl'.")


if __name__ == "__main__":
    main()
