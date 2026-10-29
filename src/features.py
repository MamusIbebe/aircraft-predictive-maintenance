"""Step 6: turn raw sensor readings into features the model can learn from."""
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

SENSORS = [f"s{i}" for i in range(1, 22)]
WINDOWS = [5, 10, 20]  # how many recent flights each rolling average looks at


def pick_sensors(train):
    """Keep only sensors that actually change."""
    spread = train[SENSORS].std()
    keep = [s for s in SENSORS if spread[s] > 0.01]
    dropped = [s for s in SENSORS if s not in keep]
    print(f"Dropped {len(dropped)} flat sensors: {dropped}")
    print(f"Keeping {len(keep)} sensors: {keep}")
    return keep


def build_features(df, sensors, scaler):
    """Scale the sensors, then add rolling averages and rolling spread per engine."""
    out = df[["unit", "cycle"]].copy()
    out[sensors] = scaler.transform(df[sensors])
    by_engine = out.groupby("unit")[sensors]
    for w in WINDOWS:
        mean = by_engine.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        std = by_engine.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        out = out.join(mean.add_suffix(f"_mean{w}")).join(std.add_suffix(f"_std{w}"))
    if "rul" in df:
        out["rul"] = df["rul"].values
    return out


if __name__ == "__main__":
    train_part = pd.read_csv("data/train_part.csv")
    val_part = pd.read_csv("data/val_part.csv")

    sensors = pick_sensors(train_part)

    # Learn each sensor's "normal" from TRAINING engines only.
    scaler = StandardScaler().fit(train_part[sensors])

    train_feat = build_features(train_part, sensors, scaler)
    val_feat = build_features(val_part, sensors, scaler)
    print(f"\nEach row now has {train_feat.shape[1] - 3} features (was 21 raw sensors).")

    print("\nEngine 1, sensor s4: raw vs scaled vs 10-flight average (last 3 flights):")
    e1 = train_feat[train_feat["unit"] == 1].tail(3)
    raw = train_part[train_part["unit"] == 1].tail(3)
    print(pd.DataFrame({"cycle": e1["cycle"].values, "s4_raw": raw["s4"].values,
                        "s4_scaled": e1["s4"].round(2).values,
                        "s4_mean10": e1["s4_mean10"].round(2).values}).to_string(index=False))

    train_feat.to_csv("data/train_features.csv", index=False)
    val_feat.to_csv("data/val_features.csv", index=False)
    joblib.dump({"sensors": sensors, "scaler": scaler}, "models/feature_prep.joblib")
    print("\nSaved data/train_features.csv, data/val_features.csv, models/feature_prep.joblib")
