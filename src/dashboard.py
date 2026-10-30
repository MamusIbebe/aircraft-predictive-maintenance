"""Step 16: fleet readiness dashboard. Which engines need maintenance first?"""
import joblib
import pandas as pd
import streamlit as st
import torch

from api import status_for
from data import check, load
from train_lstm import LSTMModel, make_windows, predict


@st.cache_resource  # load the model once, not on every click
def load_model():
    prep = joblib.load("models/feature_prep.joblib")
    saved = torch.load("models/lstm.pt")
    model = LSTMModel(saved["n_sensors"])
    model.load_state_dict(saved["weights"])
    return prep, model


@st.cache_data
def predict_fleet():
    prep, model = load_model()
    fleet = check(load("data/raw/test_FD001.txt"), "fleet")
    rows = []
    for engine_id, engine in fleet.groupby("unit"):
        X, _ = make_windows(engine, prep["sensors"], prep["scaler"], last_only=True)
        rul = round(float(predict(model, X)[0]), 1)
        rows.append({"engine": engine_id, "flights_so_far": len(engine),
                     "predicted_flights_left": rul, "status": status_for(rul)})
    ranked = pd.DataFrame(rows).sort_values("predicted_flights_left").reset_index(drop=True)
    ranked.insert(0, "priority", range(1, len(ranked) + 1))
    return fleet, ranked


st.set_page_config(page_title="Fleet Readiness", layout="wide")
st.title("Engine fleet readiness")
st.caption("Predicted flights left before maintenance, from the PyTorch LSTM model.")

fleet, ranked = predict_fleet()
critical = ranked["status"].str.startswith("CRITICAL").sum()
warning = ranked["status"].str.startswith("WARNING").sum()
ready_pct = 100 * (len(ranked) - critical) / len(ranked)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fleet readiness", f"{ready_pct:.0f}%")
c2.metric("Critical (20 or fewer flights)", int(critical))
c3.metric("Warning (21-50 flights)", int(warning))
c4.metric("Engines", len(ranked))

st.subheader("Maintenance priority list")
st.dataframe(ranked.head(25), hide_index=True)

st.subheader("Engine detail")
engine_id = st.selectbox("Pick an engine", ranked["engine"])
engine = fleet[fleet["unit"] == engine_id].set_index("cycle")
st.line_chart(engine[["s4", "s11"]])
st.write(ranked[ranked["engine"] == engine_id].iloc[0]["status"])
