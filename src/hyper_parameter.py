import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import optuna
from src.dataprocessing import load
from src.model_train import save_model


def objective(trial):
    # Hyperparameter search space
    X_train, X_test, y_train, y_test = load()
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 10, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    trial.set_user_attr("model", model)

    return rmse


def main():
    print("Model saved as 'model.pkl'.")
    # Create study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    # Print best hyperparameters and RMSE
    print(f"Best hyperparameters: {study.best_trial.params}")
    print(f"Best RMSE: {study.best_value}")
    best_trial = study.best_trial
    best_model = best_trial.user_attrs["model"]
    X_train, X_test, y_train, y_test = load()
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Final RMSE: {rmse}")
    save_model(best_model, "model.pkl")
    print("Model saved as 'model.pkl'.")


if __name__ == "__main__":
    main()
