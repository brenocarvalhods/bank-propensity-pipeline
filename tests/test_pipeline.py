"""Unit tests for the pipeline entry point."""

from bank_propensity_pipeline.pipeline import PIPELINE_STEPS


def test_pipeline_steps_follow_the_dvc_order_then_register_and_score() -> None:
    names = [name for name, _ in PIPELINE_STEPS]
    assert names == ["fetch", "preprocess", "train", "evaluate", "register", "score"]


def test_pipeline_steps_are_callables() -> None:
    assert all(callable(step) for _, step in PIPELINE_STEPS)
