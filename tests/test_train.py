"""Unit tests for the training step, using synthetic data only."""

import random

import pandas as pd

from bank_propensity_pipeline.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from bank_propensity_pipeline.models.train import build_model, cross_validate_model

MODEL_PARAMS = {"C": 1.0, "max_iter": 200}


def make_feature_sample(n_rows: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Build random features with the same columns the model expects."""
    rng = random.Random(seed)
    numeric = {name: [rng.gauss(0, 1) for _ in range(n_rows)] for name in NUMERIC_FEATURES}
    categories = ["a", "b", "unknown"]
    categorical = {
        name: [rng.choice(categories) for _ in range(n_rows)] for name in CATEGORICAL_FEATURES
    }
    target = pd.Series([rng.randint(0, 1) for _ in range(n_rows)])
    return pd.DataFrame({**numeric, **categorical}), target


def test_build_model_outputs_valid_probabilities() -> None:
    features, target = make_feature_sample()
    model = build_model(MODEL_PARAMS, seed=42).fit(features, target)
    probabilities = model.predict_proba(features)[:, 1]
    assert probabilities.min() >= 0
    assert probabilities.max() <= 1


def test_cross_validate_model_returns_expected_metrics() -> None:
    features, target = make_feature_sample()
    model = build_model(MODEL_PARAMS, seed=42)
    metrics = cross_validate_model(model, features, target, n_splits=3, seed=42)
    assert set(metrics) == {"cv_pr_auc", "cv_roc_auc", "cv_f1"}
