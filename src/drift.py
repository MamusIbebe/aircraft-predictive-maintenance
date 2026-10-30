"""Step 15: drift monitoring. Has the live sensor data changed since the model was trained?"""
import sys

import joblib
import numpy as np
import pandas as pd

from data import check, load

EARLY_FLIGHTS = 30  # compare engines at the same life stage (their first 30 flights)


def psi(reference, current, bins=10):
    """Population Stability Index: how much a sensor's distribution has shifted."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share = np.clip(ref_share, 1e-6, None)  # avoid dividing by zero
    cur_share = np.clip(cur_share, 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def status(value):
    if value > 0.25:
        return "ALERT"
    if value > 0.10:
        return "WARNING"
    return "OK"


if __name__ == "__main__":
    sensors = joblib.load("models/feature_prep.joblib")["sensors"]
    reference = pd.read_csv("data/train_part.csv")                  # what the model learned from
    current = check(load("data/raw/test_FD001.txt"), "in-service")  # what it sees now

    if "--fault" in sys.argv:  # pretend sensor s11 was badly re-calibrated
        current["s11"] = current["s11"] + 1.5 * current["s11"].std()
        print("Simulating a faulty s11 sensor (calibration shift)\n")

    reference = reference[reference["cycle"] <= EARLY_FLIGHTS]
    current = current[current["cycle"] <= EARLY_FLIGHTS]

    rows = [{"sensor": s, "psi": round(psi(reference[s], current[s]), 3)} for s in sensors]
    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    report["status"] = report["psi"].apply(status)
    print(report.to_string(index=False))

    overall = "ALERT" if (report["status"] == "ALERT").any() else \
              "WARNING" if (report["status"] == "WARNING").any() else "OK"
    print(f"\nOverall drift status: {overall}")
    report.to_csv("reports/drift_report.csv", index=False)
