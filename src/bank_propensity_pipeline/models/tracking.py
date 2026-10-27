"""MLflow tracking helpers: configure the tracking backend and log training runs."""

import logging
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from bank_propensity_pipeline.config import Settings

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_NAME = "model"
INPUT_EXAMPLE_ROWS = 5
# Types created by our own tree-based models; safe to load because we produce the files ourselves.
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def configure_mlflow(settings: Settings) -> None:
    """Point MLflow to the tracking backend and select the experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Turn nested params such as {'model': {'C': 1}} into flat keys such as 'model.C'."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def log_training_run(
    model: Pipeline, params: dict[str, Any], metrics: dict[str, float], features: pd.DataFrame
) -> str:
    """Log params, metrics and the fitted model as one MLflow run and return its id."""
    with mlflow.start_run(run_name=params["model"]["type"]) as run:
        mlflow.log_params(flatten_params(params))
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", params["model"]["type"])
        signature = infer_signature(features, model.predict(features))
        mlflow.sklearn.log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            signature=signature,
            input_example=features.head(INPUT_EXAMPLE_ROWS),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    logger.info("Logged MLflow run %s", run.info.run_id)
    return run.info.run_id
