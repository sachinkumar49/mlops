
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from src.dataprocessing import load
from src.model_train import save_model

mlflow.set_tracking_uri('http://localhost:5000')


# Function to train model and log metrics with MLflow
def train_and_log_model(n_estimators, max_depth,
                        X_train, X_test, y_train, y_test):
    with mlflow.start_run():  # Start a new MLflow run
        # Create the model with the given parameters
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # Log model parameters, metrics and the model itself
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        print(f"Run with n_estimators={n_estimators}, "
              f"max_depth={max_depth}, RMSE={rmse}")
        return model, rmse


def multi_train():
    X_train, X_test, y_train, y_test = load()
    for i in range(3):
        model, rmse = train_and_log_model(
            50*(i+1), 10*(i+1), X_train, X_test, y_train, y_test
            )
        print(f"Model ${i} trained successfully with RMSE={rmse}.")
        save_model(model, f'model_{i}.pkl')


if __name__ == "__main__":
    multi_train()
