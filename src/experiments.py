"""Step 10: try several model settings and record every run in MLflow."""
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from train_baseline import load_test_last_flight, mae, nasa_score, rmse

# Where MLflow keeps its records: a small database file in the project folder.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("engine-rul")

train = pd.read_csv("data/train_features.csv")
val = pd.read_csv("data/val_features.csv")
test = load_test_last_flight(joblib.load("models/feature_prep.joblib"))
feature_cols = [c for c in train.columns if c not in ("unit", "cycle", "rul")]

# Three different "settings" (hyperparameters) to compare.
settings = [
    {"learning_rate": 0.1,  "max_depth": 3, "max_iter": 200},
    {"learning_rate": 0.05, "max_depth": 6, "max_iter": 400},
    {"learning_rate": 0.02, "max_depth": 8, "max_iter": 800},
]

for i, params in enumerate(settings, start=1):
    with mlflow.start_run(run_name=f"gboost-{i}"):
        mlflow.log_params(params)  # record the settings

        model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(train[feature_cols], train["rul"])
        val_pred = np.clip(model.predict(val[feature_cols]), 0, None)
        test_pred = np.clip(model.predict(test[feature_cols]), 0, None)

        scores = {
            "val_rmse": rmse(val["rul"], val_pred),
            "test_rmse": rmse(test["rul"], test_pred),
            "test_mae": mae(test["rul"], test_pred),
            "test_nasa_score": nasa_score(test["rul"], test_pred),
        }
        mlflow.log_metrics(scores)  # record the scores

        path = f"models/gboost-{i}.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(path)  # keep the model file with the run

        print(f"Run gboost-{i} {params}  ->  val RMSE {scores['val_rmse']:.2f}, "
              f"test RMSE {scores['test_rmse']:.2f}")

print("\nDone. Start the MLflow UI to compare the runs.")
