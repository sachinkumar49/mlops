import unittest
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import LabelEncoder


class TestRetailDataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the dataset for testing
        cls.file_path = './data/online_retail.csv'
        cls.model_path = 'model.pkl'
        if not os.path.exists(cls.file_path):
            raise FileNotFoundError(f"Dataset not found at {cls.file_path}")
        cls.data = pd.read_csv(cls.file_path)
        print("Dataset loaded successfully for testing.")

    def test_data_loading(self):
        # Check if the data is loaded correctly
        self.assertFalse(self.data.empty, "The dataset should not be empty.")
        self.assertIn(
            'Quantity', self.data.columns,
            "Dataset should have a Quantity column."
        )
        self.assertIn(
            'UnitPrice', self.data.columns,
            "Dataset should have a 'UnitPrice' column."
            )

    def test_total_sales_creation(self):
        # Test if TotalSales column is created correctly
        self.data['TotalSales'] = self.data['Quantity'] * \
            self.data['UnitPrice']
        self.assertIn(
            'TotalSales', self.data.columns,
            "TotalSales column is not created."
            )
        self.assertIn(
            'TotalSales', self.data.columns,
            "Dataset should have a 'TotalSales' column."
            )

    def test_data_preprocessing(self):
        # Convert InvoiceDate to datetime and test
        self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'])
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                self.data['InvoiceDate']),
            "InvoiceDate should be datetime."
            )

        # Extract features and drop unnecessary columns
        self.data['Year'] = self.data['InvoiceDate'].dt.year
        self.data['Month'] = self.data['InvoiceDate'].dt.month
        self.data['Day'] = self.data['InvoiceDate'].dt.day
        self.data['Hour'] = self.data['InvoiceDate'].dt.hour
        self.data = self.data.drop(
            columns=['InvoiceNo', 'Description', 'InvoiceDate']
            )

        # Check missing value handling
        self.data = self.data.dropna()
        self.assertFalse(
            self.data.isnull().any().any(),
            "Data should not contain missing values after preprocessing."
            )

    def test_train_test_split(self):
        # Define features and target
        self.data['TotalSales'] = self.data['Quantity'] * \
                                self.data['UnitPrice']
        X = self.data.drop(columns=['TotalSales'])
        y = self.data['TotalSales']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.assertEqual(
            len(X_train) + len(X_test), len(X),
            "Train and test split should cover all rows."
        )
        self.assertEqual(
            len(y_train) + len(y_test), len(y),
            "Train and test split should cover all targets."
        )

    def test_model_training(self):
        # Train a RandomForestRegressor and check
        self.data = self.data.drop(
            columns=['InvoiceNo', 'Description', 'InvoiceDate']
        )
        # Use Label Encoding for Categorical Variables
        le_stock = LabelEncoder()
        le_customer = LabelEncoder()
        le_country = LabelEncoder()

        self.data['StockCode'] = le_stock.fit_transform(self.data['StockCode'])
        self.data['CustomerID'] = le_customer.fit_transform(
            self.data['CustomerID']
        )
        self.data['Country'] = le_country.fit_transform(self.data['Country'])

        self.data['TotalSales'] = \
            self.data['Quantity'] * self.data['UnitPrice']
        X = self.data.drop(columns=['TotalSales'])
        y = self.data['TotalSales']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        self.assertGreater(rmse, 0, "RMSE should be greater than 0.")
        print(f"RMSE: {rmse}")

        # Save the model
        joblib.dump(model, self.model_path)
        self.assertTrue(
            os.path.exists(self.model_path), "Model file should be saved."
        )

    def test_model_loading(self):
        # Test loading of saved model
        model = joblib.load(self.model_path)
        self.assertIsInstance(
            model, RandomForestRegressor,
            "Loaded model should be a RandomForestRegressor."
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up model file after testing
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)
            print("Cleaned up model file.")


if __name__ == '__main__':
    unittest.main()
