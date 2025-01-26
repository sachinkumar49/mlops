from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Load the Dataset in Chunks

data = pd.read_csv('./data/online_retail.csv')

# Step 2: Data Preprocessing
# Convert InvoiceDate to datetime
data['TotalSales'] = data['Quantity'] * data['UnitPrice']
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
print('InvoiceDate converted to datetime', data.head())
# Extract features from InvoiceDate
data['Year'] = data['InvoiceDate'].dt.year
data['Month'] = data['InvoiceDate'].dt.month
data['Day'] = data['InvoiceDate'].dt.day
data['Hour'] = data['InvoiceDate'].dt.hour

# Drop unnecessary columns
data = data.drop(columns=['InvoiceNo', 'Description', 'InvoiceDate'])

# Handle missing values
data = data.dropna()

# Use Label Encoding for Categorical Variables
le_stock = LabelEncoder()
le_customer = LabelEncoder()
le_country = LabelEncoder()

data['StockCode'] = le_stock.fit_transform(data['StockCode'])
data['CustomerID'] = le_customer.fit_transform(data['CustomerID'])
data['Country'] = le_country.fit_transform(data['Country'])

# Step 3: Define Features and Target
X = data.drop(columns=['TotalSales'])
y = data['TotalSales']
print('dropping total sales from x', data.head())
# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Step 5: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 7: Save the Model
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'.")
