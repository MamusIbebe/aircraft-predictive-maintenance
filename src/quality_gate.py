"""Step 11: quality gate. Exit with an error if the model isn't good enough to deploy."""
import sys

import joblib
import pandas as pd
import torch

from data import check, load
from train_baseline import mae, rmse
from train_lstm import LSTMModel, make_windows, predict

MAX_TEST_RMSE = 22.0  # the bar a model must clear (in flights)

prep = joblib.load("models/feature_prep.joblib")
saved = torch.load("models/lstm.pt")
model = LSTMModel(saved["n_sensors"])
model.load_state_dict(saved["weights"])

test = check(load("data/raw/test_FD001.txt"), "test")
X_test, _ = make_windows(test, prep["sensors"], prep["scaler"], last_only=True)
true_rul = pd.read_csv("data/raw/RUL_FD001.txt", header=None)[0].clip(upper=125).values
pred = predict(model, X_test)

score = rmse(true_rul, pred)
print(f"LSTM test RMSE {score:.1f}, MAE {mae(true_rul, pred):.1f}  (limit {MAX_TEST_RMSE})")
if score > MAX_TEST_RMSE:
    print("FAIL: model is not good enough to deploy.")
    sys.exit(1)
print("PASS: model can be deployed.")
