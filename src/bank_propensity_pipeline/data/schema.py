"""Dataset schema: descriptive column names and column roles used across the pipeline."""

from pathlib import Path

import pandas as pd

RAW_SEPARATOR = ";"

COLUMN_RENAMES: dict[str, str] = {
    "marital": "marital_status",
    "default": "has_credit_default",
    "housing": "has_housing_loan",
    "loan": "has_personal_loan",
    "contact": "contact_channel",
    "month": "last_contact_month",
    "day_of_week": "last_contact_weekday",
    "duration": "last_call_duration_sec",
    "campaign": "contacts_this_campaign",
    "pdays": "days_since_previous_contact",
    "previous": "previous_contacts",
    "poutcome": "previous_outcome",
    "emp.var.rate": "employment_variation_rate",
    "cons.price.idx": "consumer_price_index",
    "cons.conf.idx": "consumer_confidence_index",
    "euribor3m": "euribor_3m",
    "nr.employed": "employees_thousands",
    "y": "subscribed",
}

TARGET_COLUMN = "subscribed"
LEAKAGE_COLUMNS: tuple[str, ...] = ("last_call_duration_sec",)
NEVER_CONTACTED_CODE = 999


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Read the raw UCI CSV and rename its columns to descriptive snake_case names."""
    raw = pd.read_csv(path, sep=RAW_SEPARATOR)
    return raw.rename(columns=COLUMN_RENAMES)
