"""Step 9: a deep learning model (LSTM) in PyTorch that reads the last 30 flights in order."""
import copy
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from data import check, load
from train_baseline import mae, nasa_score, rmse

WINDOW = 30         # the model looks at the last 30 flights of an engine
torch.manual_seed(42)  # same random start every run, so results are repeatable
np.random.seed(42)


def make_windows(df, sensors, scaler, last_only=False):
    """Turn each engine's flights into 30-flight windows: shape (samples, 30, sensors)."""
    X, y = [], []
    for _, engine in df.groupby("unit"):
        values = scaler.transform(engine[sensors]).astype(np.float32)
        # Engines with fewer than 30 flights: repeat the first reading at the front.
        if len(values) < WINDOW:
            values = np.vstack([np.repeat(values[:1], WINDOW - len(values), axis=0), values])
            engine = pd.concat([engine.iloc[[0]]] * (WINDOW - len(engine)) + [engine])
        ends = [len(values)] if last_only else range(WINDOW, len(values) + 1)
        for end in ends:
            X.append(values[end - WINDOW:end])
            if "rul" in engine:
                y.append(engine["rul"].iloc[end - 1])
    return torch.tensor(np.array(X)), torch.tensor(np.array(y, dtype=np.float32))


class LSTMModel(nn.Module):
    def __init__(self, n_sensors):
        super().__init__()
        # The LSTM reads the 30 flights one by one and keeps a "memory" of what it saw.
        self.lstm = nn.LSTM(input_size=n_sensors, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.2)
        # A small layer that turns that memory into one number: predicted RUL.
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def predict(model, X):
    model.eval()  # switch off dropout for predicting
    with torch.no_grad():
        return model(X).clamp(min=0).numpy()


if __name__ == "__main__":
    torch.set_num_threads(2)
    prep = joblib.load("models/feature_prep.joblib")
    sensors, scaler = prep["sensors"], prep["scaler"]

    X_train, y_train = make_windows(pd.read_csv("data/train_part.csv"), sensors, scaler)
    X_val, y_val = make_windows(pd.read_csv("data/val_part.csv"), sensors, scaler)
    print(f"Training windows: {tuple(X_train.shape)}  (samples, flights, sensors)")

    model = LSTMModel(len(sensors))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_rmse, best_weights, patience_left = float("inf"), None, 5
    start = time.time()
    for epoch in range(1, 31):
        model.train()
        order = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 256):
            batch = order[i:i + 256]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_rmse = rmse(y_val, predict(model, X_val))
        print(f"epoch {epoch:2d}  validation RMSE {val_rmse:6.2f}  ({time.time() - start:4.0f}s)")
        if val_rmse < best_rmse - 0.01:
            best_rmse, best_weights, patience_left = val_rmse, copy.deepcopy(model.state_dict()), 5
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping: validation stopped improving.")
                break

    model.load_state_dict(best_weights)
    print(f"\nBest validation RMSE: {best_rmse:.1f}")

    # Final exam: NASA's 100 test engines, using each engine's last 30 flights.
    test = check(load("data/raw/test_FD001.txt"), "test")
    X_test, _ = make_windows(test, sensors, scaler, last_only=True)
    true_rul = pd.read_csv("data/raw/RUL_FD001.txt", header=None)[0].clip(upper=125).values
    test_pred = predict(model, X_test)
    print(f"NASA test set (100 engines): RMSE {rmse(true_rul, test_pred):.1f}, "
          f"MAE {mae(true_rul, test_pred):.1f}, NASA score {nasa_score(true_rul, test_pred):.0f}")

    torch.save({"weights": model.state_dict(), "n_sensors": len(sensors)}, "models/lstm.pt")
    print("Saved models/lstm.pt")
