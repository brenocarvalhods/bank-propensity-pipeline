"""Unit tests for scoring helpers, using synthetic data only."""

import json

import pandas as pd

from bank_propensity_pipeline.models.predict import champion_uri, top_share_conversion
from bank_propensity_pipeline.models.registry import load_run_info


def test_champion_uri_uses_registry_alias() -> None:
    assert champion_uri("my-model") == "models:/my-model@champion"


def test_top_share_conversion_measures_best_scored_clients() -> None:
    scores = pd.Series([0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05])
    target = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert top_share_conversion(scores, target, share=0.2) == 1.0


def test_load_run_info_reads_json_file(tmp_path) -> None:
    run_info_path = tmp_path / "run_info.json"
    run_info_path.write_text(json.dumps({"run_id": "abc", "model_uri": "models:/m-1"}))
    assert load_run_info(run_info_path)["model_uri"] == "models:/m-1"
