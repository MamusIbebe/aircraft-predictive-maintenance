"""Step 11: automated tests. Each test checks one promise our code makes."""
import numpy as np
import pandas as pd
import pytest
import torch

from data import add_rul, check
from train_baseline import nasa_score, rmse
from train_lstm import LSTMModel


def tiny_fleet():
    """Fake data: 2 engines, 5 flights each."""
    rows = []
    for unit in (1, 2):
        for cycle in range(1, 6):
            rows.append({"unit": unit, "cycle": cycle, "s4": 1400 + cycle})
    return pd.DataFrame(rows)


def test_rul_counts_down_to_zero():
    df = add_rul(tiny_fleet())
    engine1 = df[df["unit"] == 1]
    assert list(engine1["rul"]) == [4, 3, 2, 1, 0]


def test_rul_is_capped():
    df = add_rul(tiny_fleet(), cap=2)
    assert df["rul"].max() == 2


def test_check_catches_missing_values():
    df = tiny_fleet()
    df.loc[0, "s4"] = np.nan
    with pytest.raises(AssertionError):
        check(df, "broken")


def test_check_catches_missing_flight():
    df = tiny_fleet()
    df = df[df["cycle"] != 3]  # engine skips flight 3
    with pytest.raises(AssertionError):
        check(df, "gap")


def test_rmse_perfect_prediction_is_zero():
    assert rmse([10, 20, 30], [10, 20, 30]) == 0


def test_nasa_score_punishes_late_more_than_early():
    late = nasa_score([50], [60])
    early = nasa_score([50], [40])
    assert late > early


def test_lstm_gives_one_prediction_per_engine():
    model = LSTMModel(n_sensors=14)
    fake_windows = torch.randn(3, 30, 14)
    assert model(fake_windows).shape == (3,)
