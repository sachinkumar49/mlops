import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.dataprocessing import load


def train_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, rmse


def save_model(model, file_name):
    joblib.dump(model, file_name)


def main():
    print("Training model...")
    X_train, X_test, y_train, y_test = load()
    model, rmse = train_model(X_train, X_test, y_train, y_test)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    save_model(model, 'model.pkl')
    print("Model saved as 'model.pkl'.")


if __name__ == "__main__":
    main()
