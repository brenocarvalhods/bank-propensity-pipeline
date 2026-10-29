"""Evaluate the trained model on the test set with ranking, decision and business metrics."""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from bank_propensity_pipeline.config import PROJECT_ROOT, load_params, load_settings
from bank_propensity_pipeline.features.build import split_features_target
from bank_propensity_pipeline.logging_config import setup_logging
from bank_propensity_pipeline.models.predict import score_clients
from bank_propensity_pipeline.models.registry import load_run_info
from bank_propensity_pipeline.models.tracking import configure_mlflow
from bank_propensity_pipeline.models.train import MODEL_PATH

logger = logging.getLogger(__name__)

METRICS_PATH = PROJECT_ROOT / "metrics" / "eval.json"
LIFT_TABLE_PATH = PROJECT_ROOT / "reports" / "lift_by_decile.csv"
LIFT_PLOT_PATH = PROJECT_ROOT / "reports" / "lift_by_decile.png"
THRESHOLDS = [step / 100 for step in range(1, 100)]
N_DECILES = 10


def campaign_profit(
    target: pd.Series,
    scores: pd.Series,
    threshold: float,
    cost_per_call: float,
    margin_per_deposit: float,
) -> float:
    """Return the campaign profit when calling every client scored at or above the threshold."""
    called = scores >= threshold
    conversions = int(target[called].sum())
    return conversions * margin_per_deposit - int(called.sum()) * cost_per_call


def find_best_threshold(
    target: pd.Series, scores: pd.Series, cost_per_call: float, margin_per_deposit: float
) -> tuple[float, float]:
    """Search the decision threshold that maximizes the campaign profit."""
    profits = {
        threshold: campaign_profit(target, scores, threshold, cost_per_call, margin_per_deposit)
        for threshold in THRESHOLDS
    }
    best_threshold = max(profits, key=profits.get)
    return best_threshold, profits[best_threshold]


def business_metrics(
    target: pd.Series, scores: pd.Series, cost_per_call: float, margin_per_deposit: float
) -> dict[str, float]:
    """Compare the profit of calling everyone with the profit of the model at its best threshold."""
    threshold, model_profit = find_best_threshold(target, scores, cost_per_call, margin_per_deposit)
    call_all_profit = campaign_profit(target, scores, 0.0, cost_per_call, margin_per_deposit)
    return {
        "best_threshold": threshold,
        "profit_model": model_profit,
        "profit_call_all": call_all_profit,
        "share_called": float((scores >= threshold).mean()),
    }


def classification_metrics(
    target: pd.Series, scores: pd.Series, threshold: float
) -> dict[str, float]:
    """Compute ranking metrics and decision metrics at the chosen threshold."""
    predicted = (scores >= threshold).astype(int)
    return {
        "test_pr_auc": average_precision_score(target, scores),
        "test_roc_auc": roc_auc_score(target, scores),
        "test_precision": precision_score(target, predicted, zero_division=0),
        "test_recall": recall_score(target, predicted),
        "test_f1": f1_score(target, predicted),
    }


def lift_by_decile(target: pd.Series, scores: pd.Series) -> pd.DataFrame:
    """Group clients into score deciles (1 = highest scores) and measure conversion and lift."""
    ranks = scores.rank(method="first", ascending=False)
    decile = pd.qcut(ranks, N_DECILES, labels=range(1, N_DECILES + 1)).rename("decile")
    table = target.groupby(decile, observed=True).agg(clients="count", conversions="sum")
    table["conversion_rate"] = table["conversions"] / table["clients"]
    table["lift"] = table["conversion_rate"] / target.mean()
    table["cumulative_capture"] = table["conversions"].cumsum() / target.sum()
    return table.round(4)


def save_lift_plot(table: pd.DataFrame, path: Path) -> None:
    """Save a bar chart of lift by score decile, with random targeting as reference."""
    figure = Figure(figsize=(8, 4.5))
    axis = figure.subplots()
    axis.bar(table.index.astype(str), table["lift"], color="#2E7D5B")
    axis.axhline(1.0, color="grey", linestyle="--", label="Random targeting")
    axis.set_title("Lift by score decile (test set)")
    axis.set_xlabel("Score decile (1 = highest scores)")
    axis.set_ylabel("Lift")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def save_reports(metrics: dict[str, float], lift_table: pd.DataFrame) -> None:
    """Save the evaluation metrics, the lift table and the lift chart to disk."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIFT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lift_table.to_csv(LIFT_TABLE_PATH)
    save_lift_plot(lift_table, LIFT_PLOT_PATH)


def log_evaluation_to_run(run_id: str, metrics: dict[str, float]) -> None:
    """Attach the test metrics and evaluation reports to the training run in MLflow."""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(LIFT_TABLE_PATH), artifact_path="evaluation")
        mlflow.log_artifact(str(LIFT_PLOT_PATH), artifact_path="evaluation")
    logger.info("Logged evaluation to MLflow run %s", run_id)


def main() -> None:
    """Entry point used by the DVC evaluate stage."""
    settings = load_settings()
    setup_logging(settings.log_level)
    configure_mlflow(settings)
    params = load_params()
    test = pd.read_csv(PROJECT_ROOT / params["data"]["processed_dir"] / "test.csv")
    features, target = split_features_target(test)
    scores = score_clients(joblib.load(MODEL_PATH), features)
    business = business_metrics(target, scores, **params["business"])
    metrics = {**classification_metrics(target, scores, business["best_threshold"]), **business}
    metrics = {name: round(float(value), 4) for name, value in metrics.items()}
    save_reports(metrics, lift_by_decile(target, scores))
    log_evaluation_to_run(load_run_info()["run_id"], metrics)
    logger.info("Test metrics: %s", metrics)


if __name__ == "__main__":
    main()
