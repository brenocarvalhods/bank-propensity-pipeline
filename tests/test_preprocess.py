"""Unit tests for the preprocessing step, using synthetic data only."""

import pandas as pd

from bank_propensity_pipeline.data.preprocess import clean_dataset, split_dataset
from bank_propensity_pipeline.data.schema import NEVER_CONTACTED_CODE, TARGET_COLUMN


def make_raw_sample(n_rows: int = 100) -> pd.DataFrame:
    """Build a small dataset shaped like the output of load_raw_dataset."""
    n_positive = n_rows // 10
    return pd.DataFrame(
        {
            "age": range(n_rows),
            "last_call_duration_sec": [120] * n_rows,
            "days_since_previous_contact": [NEVER_CONTACTED_CODE, 5] * (n_rows // 2),
            TARGET_COLUMN: ["yes"] * n_positive + ["no"] * (n_rows - n_positive),
        }
    )


def test_clean_dataset_drops_leakage_column() -> None:
    clean = clean_dataset(make_raw_sample())
    assert "last_call_duration_sec" not in clean.columns


def test_clean_dataset_encodes_target_as_binary() -> None:
    clean = clean_dataset(make_raw_sample())
    assert set(clean[TARGET_COLUMN].unique()) == {0, 1}


def test_clean_dataset_replaces_sentinel_with_flag() -> None:
    clean = clean_dataset(make_raw_sample())
    assert "days_since_previous_contact" not in clean.columns
    assert clean["previously_contacted"].tolist()[:2] == [0, 1]


def test_split_dataset_keeps_target_proportion() -> None:
    clean = clean_dataset(make_raw_sample(n_rows=200))
    train, test = split_dataset(clean, test_size=0.25, seed=42)
    assert len(train) == 150
    assert len(test) == 50
    assert train[TARGET_COLUMN].mean() == test[TARGET_COLUMN].mean()
