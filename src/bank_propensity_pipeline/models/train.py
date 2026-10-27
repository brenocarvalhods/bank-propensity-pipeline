"""Train the classification pipeline with cross-validation and save the fitted model."""

import json
import logging
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from bank_propensity_pipeline.config import PROJECT_ROOT, load_params, load_settings
from bank_propensity_pipeline.features.build import build_preprocessor, split_features_target
from bank_propensity_pipeline.logging_config import setup_logging
from bank_propensity_pipeline.models.tracking import configure_mlflow, log_training_run

logger = logging.getLogger(__name__)

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "metrics" / "train.json"
CV_SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1"}


def build_model(model_params: dict[str, Any], seed: int) -> Pipeline:
    """Combine the preprocessor and a balanced logistic regression in one pipeline."""
    classifier = LogisticRegression(
        C=model_params["C"],
        max_iter=model_params["max_iter"],
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])


def cross_validate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series, n_splits: int, seed: int
) -> dict[str, float]:
    """Run stratified k-fold cross-validation and return the mean of each metric."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_validate(model, features, target, cv=folds, scoring=CV_SCORING)
    metrics = {f"cv_{name}": round(float(scores[f"test_{name}"].mean()), 4) for name in CV_SCORING}
    logger.info("Cross-validation metrics: %s", metrics)
    return metrics


def save_artifacts(model: Pipeline, metrics: dict[str, float]) -> None:
    """Save the fitted model and the cross-validation metrics to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved model to %s and metrics to %s", MODEL_PATH, METRICS_PATH)


def main() -> None:
    """Entry point used by the DVC train stage."""
    settings = load_settings()
    setup_logging(settings.log_level)
    configure_mlflow(settings)
    params = load_params()
    train = pd.read_csv(PROJECT_ROOT / params["data"]["processed_dir"] / "train.csv")
    features, target = split_features_target(train)
    model = build_model(params["model"]["logreg"], params["seed"])
    metrics = cross_validate_model(
        model, features, target, params["train"]["cv_folds"], params["seed"]
    )
    model.fit(features, target)
    save_artifacts(model, metrics)
    log_training_run(model, params, metrics, features)

if __name__ == "__main__":
    main()
