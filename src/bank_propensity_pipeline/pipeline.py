"""Run every pipeline step in order: fetch, preprocess, train, evaluate, register and score."""

import logging
from collections.abc import Callable

from bank_propensity_pipeline.config import load_settings
from bank_propensity_pipeline.data import fetch, preprocess
from bank_propensity_pipeline.logging_config import setup_logging
from bank_propensity_pipeline.models import evaluate, predict, registry, train

logger = logging.getLogger(__name__)

PIPELINE_STEPS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("fetch", fetch.main),
    ("preprocess", preprocess.main),
    ("train", train.main),
    ("evaluate", evaluate.main),
    ("register", registry.main),
    ("score", predict.main),
)


def run_pipeline() -> None:
    """Run each pipeline step in sequence, logging when each one starts."""
    for name, step in PIPELINE_STEPS:
        logger.info("Starting step: %s", name)
        step()
    logger.info("Pipeline finished: data, model, registry and scoring are ready")


def main() -> None:
    """Entry point used by the Docker image."""
    setup_logging(load_settings().log_level)
    run_pipeline()


if __name__ == "__main__":
    main()
