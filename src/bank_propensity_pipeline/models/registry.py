"""Register the model produced by the pipeline and promote it with the champion alias."""

import json
import logging
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from bank_propensity_pipeline.config import PROJECT_ROOT, load_settings
from bank_propensity_pipeline.logging_config import setup_logging
from bank_propensity_pipeline.models.tracking import configure_mlflow

logger = logging.getLogger(__name__)

RUN_INFO_PATH = PROJECT_ROOT / "models" / "run_info.json"
CHAMPION_ALIAS = "champion"


def load_run_info(path: Path = RUN_INFO_PATH) -> dict[str, str]:
    """Read which MLflow run and model URI the training stage produced."""
    return json.loads(path.read_text(encoding="utf-8"))


def register_model(model_uri: str, model_name: str) -> str:
    """Register the logged model as a new version of the registered model."""
    version = mlflow.register_model(model_uri, model_name)
    logger.info("Registered %s version %s", model_name, version.version)
    return str(version.version)


def promote_to_champion(model_name: str, version: str) -> None:
    """Point the champion alias to the given model version."""
    MlflowClient().set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    logger.info("Alias @%s now points to %s version %s", CHAMPION_ALIAS, model_name, version)


def main() -> None:
    """Register the latest trained model and promote it to champion."""
    settings = load_settings()
    setup_logging(settings.log_level)
    configure_mlflow(settings)
    run_info = load_run_info()
    version = register_model(run_info["model_uri"], settings.mlflow_model_name)
    promote_to_champion(settings.mlflow_model_name, version)


if __name__ == "__main__":
    main()
