"""Score clients with the champion model loaded from the MLflow Model Registry."""

import logging

import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from bank_propensity_pipeline.config import PROJECT_ROOT, load_params, load_settings
from bank_propensity_pipeline.features.build import split_features_target
from bank_propensity_pipeline.logging_config import setup_logging
from bank_propensity_pipeline.models.registry import CHAMPION_ALIAS
from bank_propensity_pipeline.models.tracking import configure_mlflow

logger = logging.getLogger(__name__)

TOP_SHARE = 0.10


def champion_uri(model_name: str) -> str:
    """Build the registry URI that always points to the current champion version."""
    return f"models:/{model_name}@{CHAMPION_ALIAS}"


def load_champion(model_name: str) -> Pipeline:
    """Load the champion model from the MLflow Model Registry."""
    uri = champion_uri(model_name)
    logger.info("Loading model from %s", uri)
    return mlflow.sklearn.load_model(uri)


def score_clients(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Return the probability of subscribing for each client."""
    probabilities = model.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=features.index, name="propensity")


def top_share_conversion(scores: pd.Series, target: pd.Series, share: float = TOP_SHARE) -> float:
    """Return the conversion rate among the clients with the highest scores."""
    top_clients = scores >= scores.quantile(1 - share)
    return float(target[top_clients].mean())


def main() -> None:
    """Score the test set with the champion and compare top-scored conversion to the base rate."""
    settings = load_settings()
    setup_logging(settings.log_level)
    configure_mlflow(settings)
    params = load_params()
    test = pd.read_csv(PROJECT_ROOT / params["data"]["processed_dir"] / "test.csv")
    features, target = split_features_target(test)
    scores = score_clients(load_champion(settings.mlflow_model_name), features)
    logger.info("Scored %d clients", len(scores))
    logger.info("Base conversion rate: %.1f%%", target.mean() * 100)
    logger.info("Top 10%% scored conversion rate: %.1f%%", top_share_conversion(scores, target) * 100)


if __name__ == "__main__":
    main()
