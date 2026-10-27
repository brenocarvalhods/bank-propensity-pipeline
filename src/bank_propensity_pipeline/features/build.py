"""Feature definitions and the Scikit-Learn preprocessor shared by training and inference."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_propensity_pipeline.data.schema import TARGET_COLUMN

NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "contacts_this_campaign",
    "previous_contacts",
    "previously_contacted",
    "employment_variation_rate",
    "consumer_price_index",
    "consumer_confidence_index",
    "euribor_3m",
    "employees_thousands",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "job",
    "marital_status",
    "education",
    "has_credit_default",
    "has_housing_loan",
    "has_personal_loan",
    "contact_channel",
    "last_contact_month",
    "last_contact_weekday",
    "previous_outcome",
)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the model input features from the target column."""
    features = df[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
    return features, df[TARGET_COLUMN]


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode categorical features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
        ],
        sparse_threshold=0.0,
    )
