"""Clean the raw dataset and split it into stratified train and test sets."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from bank_propensity_pipeline.config import PROJECT_ROOT, load_params, load_settings
from bank_propensity_pipeline.data.schema import (
    LEAKAGE_COLUMNS,
    NEVER_CONTACTED_CODE,
    TARGET_COLUMN,
    load_raw_dataset,
)
from bank_propensity_pipeline.logging_config import setup_logging

logger = logging.getLogger(__name__)


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are only known after the outcome (data leakage)."""
    return df.drop(columns=list(LEAKAGE_COLUMNS))


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the target from 'yes'/'no' to 1/0."""
    encoded = df.copy()
    encoded[TARGET_COLUMN] = (encoded[TARGET_COLUMN] == "yes").astype(int)
    return encoded


def add_contact_history_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the 999 sentinel in days_since_previous_contact with a binary flag."""
    flagged = df.copy()
    previously_contacted = flagged["days_since_previous_contact"] != NEVER_CONTACTED_CODE
    flagged["previously_contacted"] = previously_contacted.astype(int)
    return flagged.drop(columns=["days_since_previous_contact"])


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps decided in the exploratory analysis."""
    return df.pipe(drop_leakage_columns).pipe(encode_target).pipe(add_contact_history_flag)


def split_dataset(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test sets, keeping the target proportion."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET_COLUMN]
    )
    logger.info("Split done: %d train rows, %d test rows", len(train), len(test))
    return train, test


def save_splits(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> None:
    """Save the train and test sets as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    logger.info("Saved train and test sets to %s", output_dir)


def main() -> None:
    """Entry point used by the DVC preprocess stage."""
    setup_logging(load_settings().log_level)
    params = load_params()
    raw = load_raw_dataset(PROJECT_ROOT / params["data"]["raw_path"])
    clean = clean_dataset(raw)
    train, test = split_dataset(clean, params["data"]["test_size"], params["seed"])
    save_splits(train, test, PROJECT_ROOT / params["data"]["processed_dir"])


if __name__ == "__main__":
    main()
