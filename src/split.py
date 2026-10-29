"""Step 5: split engines into a training group and a validation group."""
import numpy as np

from data import add_rul, check, load

train = add_rul(check(load("data/raw/train_FD001.txt"), "train"))

# Pick 20 of the 100 engines at random to hold back.
# seed=42 makes the "random" pick the same every time.
rng = np.random.default_rng(seed=42)
all_engines = train["unit"].unique()
val_engines = rng.choice(all_engines, size=20, replace=False)

# Every row of an engine goes to the same side.
is_val = train["unit"].isin(val_engines)
train_part = train[~is_val]
val_part = train[is_val]

print(f"Training engines:   {train_part['unit'].nunique()}  ({len(train_part)} rows)")
print(f"Validation engines: {val_part['unit'].nunique()}  ({len(val_part)} rows)")
print(f"Validation engine ids: {sorted(val_engines.tolist())}")

# Safety check: no engine may be in both groups.
overlap = set(train_part["unit"]) & set(val_part["unit"])
assert not overlap, f"Leak! Engines on both sides: {overlap}"
print("Check passed: no engine is in both groups.")

# Save both groups for the next steps.
train_part.to_csv("data/train_part.csv", index=False)
val_part.to_csv("data/val_part.csv", index=False)
print("Saved data/train_part.csv and data/val_part.csv")
