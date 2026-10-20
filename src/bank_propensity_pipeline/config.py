"""Project configuration loaded from params.yaml and environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@dataclass(frozen=True)
class Settings:
    """Runtime settings that depend on the environment (.env)."""

    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_model_name: str
    log_level: str


def load_settings() -> Settings:
    """Load environment settings from the .env file, with safe defaults."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "bank-propensity"),
        mlflow_model_name=os.getenv("MLFLOW_MODEL_NAME", "bank-propensity-classifier"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Load experiment parameters tracked by DVC from params.yaml."""
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)