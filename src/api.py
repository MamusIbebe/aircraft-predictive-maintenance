"""Step 12: a web API that serves remaining-life predictions from the LSTM."""
import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from data import check, load
from train_lstm import LSTMModel, make_windows, predict

# Load everything ONCE when the server starts.
prep = joblib.load("models/feature_prep.joblib")
saved = torch.load("models/lstm.pt")
model = LSTMModel(saved["n_sensors"])
model.load_state_dict(saved["weights"])
fleet = check(load("data/raw/test_FD001.txt"), "in-service fleet")

app = FastAPI(title="Engine Remaining Life API")


def status_for(rul):
    """Turn a number into a maintenance decision."""
    if rul <= 20:
        return "CRITICAL: ground and schedule maintenance now"
    if rul <= 50:
        return "WARNING: plan maintenance, order parts"
    return "OK: no action"


class Flight(BaseModel):
    cycle: int
    sensors: dict[str, float]


class EngineHistory(BaseModel):
    engine_id: str
    flights: list[Flight]


def score(df):
    X, _ = make_windows(df, prep["sensors"], prep["scaler"], last_only=True)
    return round(float(predict(model, X)[0]), 1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/fleet-engine/{engine_id}")
def predict_fleet_engine(engine_id: int):
    engine = fleet[fleet["unit"] == engine_id]
    if engine.empty:
        raise HTTPException(status_code=404, detail="No engine with that id (use 1-100)")
    rul = score(engine)
    return {"engine_id": engine_id, "flights_so_far": len(engine),
            "predicted_flights_left": rul, "status": status_for(rul)}


@app.post("/predict")
def predict_engine(history: EngineHistory):
    rows = [{"unit": 1, "cycle": f.cycle, **f.sensors} for f in history.flights]
    df = pd.DataFrame(rows)
    missing = [s for s in prep["sensors"] if s not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing sensors: {missing}")
    rul = score(df)
    return {"engine_id": history.engine_id, "predicted_flights_left": rul, "status": status_for(rul)}
