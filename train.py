import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Load the Dataset
# Replace 'retail_data.csv' with the path to your dataset
data = pd.read_csv('./data/online_retail.csv')
print("data loaded", data.head())
# Step 2: Data Preprocessing
# Create a new column for Total Sales
data['TotalSales'] = data['Quantity'] * data['UnitPrice']
print("Total Sales created", data.head())	
# Convert InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Extract features from InvoiceDate
data['Year'] = data['InvoiceDate'].dt.year
data['Month'] = data['InvoiceDate'].dt.month
data['Day'] = data['InvoiceDate'].dt.day
data['Hour'] = data['InvoiceDate'].dt.hour

# Drop unnecessary columns
data = data.drop(columns=['InvoiceNo', 'Description', 'InvoiceDate'])
print('drop unnecessary columns', data.head())
# Handle missing values
data = data.dropna()
print('missing values handled', data.head())
# Convert categorical variables to dummy variables
data = pd.get_dummies(
    data, columns=['StockCode', 'CustomerID', 'Country'],
    drop_first=True)

# Step 3: Define Features and Target
X = data.drop(columns=['TotalSales'])
y = data['TotalSales']

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
print('data splitted', X_train.head())
# Step 5: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print('model trained', model)
# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 7: Save the Model
joblib.dump(model, 'sales_prediction_model.pkl')
print("Model saved as 'sales_prediction_model.pkl'.")

# Save the model
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'")
