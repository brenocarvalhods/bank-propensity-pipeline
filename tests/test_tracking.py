"""Unit tests for the MLflow tracking helpers."""

from bank_propensity_pipeline.models.tracking import flatten_params


def test_flatten_params_joins_nested_keys_with_dots() -> None:
    params = {"seed": 42, "model": {"type": "logreg", "logreg": {"C": 1.0}}}
    expected = {"seed": 42, "model.type": "logreg", "model.logreg.C": 1.0}
    assert flatten_params(params) == expected


def test_flatten_params_keeps_flat_dict_unchanged() -> None:
    params = {"seed": 42, "cv_folds": 5}
    assert flatten_params(params) == params
