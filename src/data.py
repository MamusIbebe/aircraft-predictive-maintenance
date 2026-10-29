"""Step 4: load, check and label the NASA engine data."""
import pandas as pd

# The files have no header row, so we name the 26 columns ourselves:
# engine id, flight number, 3 flight settings, 21 sensors.
COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load(path):
    """Read one NASA text file into a table (a pandas DataFrame)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = COLUMNS
    return df


def check(df, name):
    """Stop with an error if the data is broken."""
    assert not df.isna().any().any(), f"{name}: has missing values"
    assert not df.duplicated(["unit", "cycle"]).any(), f"{name}: has duplicate rows"
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    gaps = df.groupby("unit")["cycle"].diff().dropna()
    assert (gaps == 1).all(), f"{name}: an engine is missing flights"
    print(f"{name}: OK - {df['unit'].nunique()} engines, {len(df)} rows")
    return df


def add_rul(df, cap=125):
    """RUL = flights left before failure."""
    last_flight = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (last_flight - df["cycle"]).clip(upper=cap)
    return df


if __name__ == "__main__":
    train = check(load("data/raw/train_FD001.txt"), "train")
    test = check(load("data/raw/test_FD001.txt"), "test")
    train = add_rul(train)

    engine1 = train[train["unit"] == 1]
    print("\nEngine 1, first flights:")
    print(engine1[["unit", "cycle", "s4", "rul"]].head(3).to_string(index=False))
    print("\nEngine 1, last flights before failure:")
    print(engine1[["unit", "cycle", "s4", "rul"]].tail(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    top.plot(engine1["cycle"], engine1["s4"])
    top.set_ylabel("Sensor s4 (temperature)")
    top.set_title("Engine 1: temperature rises as it wears out")
    bottom.plot(engine1["cycle"], engine1["rul"], color="red")
    bottom.set_ylabel("Flights left (RUL)")
    bottom.set_xlabel("Flight number")
    fig.savefig("reports/engine1.png")
    print("\nChart saved to reports/engine1.png")
