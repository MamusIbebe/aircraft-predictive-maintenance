"""Step 8: train our first ML model (gradient boosting) and score it."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from data import check, load
from features import build_features


def rmse(true, pred):
    """Typical size of the error, in flights. Big misses count extra."""
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(true, pred):
    """Average size of the error, in flights."""
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def nasa_score(true, pred):
    """NASA's score: predicting failure too LATE is punished more than too early."""
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def load_test_last_flight(prep):
    """NASA test engines: build features, keep each engine's LATEST flight, attach true RUL."""
    test = check(load("data/raw/test_FD001.txt"), "test")
    feats = build_features(test, prep["sensors"], prep["scaler"])
    last = feats.groupby("unit").tail(1).reset_index(drop=True)
    true_rul = pd.read_csv("data/raw/RUL_FD001.txt", header=None)[0]
    last["rul"] = true_rul.clip(upper=125).values
    return last


if __name__ == "__main__":
    train = pd.read_csv("data/train_features.csv")
    val = pd.read_csv("data/val_features.csv")
    feature_cols = [c for c in train.columns if c not in ("unit", "cycle", "rul")]

    # 0. A "dumb" model to beat: always guess the average RUL.
    guess = train["rul"].mean()
    print(f"Dumb guess (always {guess:.0f}):  validation RMSE {rmse(val['rul'], [guess] * len(val)):.1f}")

    # 1. Train on the 80 engines.
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(train[feature_cols], train["rul"])

    # 2. Check on the 20 hidden engines.
    val_pred = np.clip(model.predict(val[feature_cols]), 0, None)
    print(f"Gradient boosting:     validation RMSE {rmse(val['rul'], val_pred):.1f}, "
          f"MAE {mae(val['rul'], val_pred):.1f}")

    # 3. Final exam: NASA's 100 test engines.
    prep = joblib.load("models/feature_prep.joblib")
    test = load_test_last_flight(prep)
    test_pred = np.clip(model.predict(test[feature_cols]), 0, None)
    print(f"\nNASA test set (100 engines): RMSE {rmse(test['rul'], test_pred):.1f}, "
          f"MAE {mae(test['rul'], test_pred):.1f}, NASA score {nasa_score(test['rul'], test_pred):.0f}")

    print("\nFirst 5 test engines:")
    print(pd.DataFrame({"engine": test["unit"], "true_rul": test["rul"],
                        "predicted_rul": test_pred.round(1)}).head().to_string(index=False))

    joblib.dump({"model": model, "features": feature_cols}, "models/baseline.joblib")
    print("\nSaved models/baseline.joblib")
