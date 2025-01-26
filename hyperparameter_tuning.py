import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data = load_iris()
X, y = data.data, data.target


def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    return cross_val_score(clf, X, y, cv=3).mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best parameters:", study.best_params)
