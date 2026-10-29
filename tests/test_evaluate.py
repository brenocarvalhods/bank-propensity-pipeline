"""Unit tests for the business evaluation helpers, using synthetic data only."""

import pandas as pd

from bank_propensity_pipeline.models.evaluate import (
    campaign_profit,
    find_best_threshold,
    lift_by_decile,
)

TARGET = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
SCORES = pd.Series([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])
COSTS = {"cost_per_call": 8.0, "margin_per_deposit": 50.0}


def test_campaign_profit_counts_margins_and_call_costs() -> None:
    # Threshold 0.8 calls the 2 best clients and both subscribe: 2 * 50 - 2 * 8 = 84.
    assert campaign_profit(TARGET, SCORES, 0.8, **COSTS) == 84.0


def test_campaign_profit_calling_everyone_pays_for_all_calls() -> None:
    # Threshold 0 calls all 10 clients and 2 subscribe: 2 * 50 - 10 * 8 = 20.
    assert campaign_profit(TARGET, SCORES, 0.0, **COSTS) == 20.0


def test_find_best_threshold_calls_only_profitable_clients() -> None:
    threshold, profit = find_best_threshold(TARGET, SCORES, **COSTS)
    assert profit == 84.0
    assert 0.75 < threshold <= 0.85


def test_lift_by_decile_ranks_best_clients_first() -> None:
    table = lift_by_decile(TARGET, SCORES)
    assert len(table) == 10
    assert table.loc[1, "lift"] == 5.0
    assert table["cumulative_capture"].iloc[-1] == 1.0
