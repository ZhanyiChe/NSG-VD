"""Offline threshold scanning for NSG-VD per-query score files."""

from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REQUIRED_QUERY_SCORE_COLUMNS = frozenset({
    "generator",
    "query_label",
    "mmd_score",
})

THRESHOLD_METRICS = (
    "recall",
    "accuracy",
    "f1",
    "precision",
    "tnr",
    "fpr",
    "auroc",
)


def build_thresholds(start=0.4, stop=1.3, step=0.1):
    """Build an inclusive, decimal-stable threshold grid."""

    start_decimal = Decimal(str(start))
    stop_decimal = Decimal(str(stop))
    step_decimal = Decimal(str(step))
    if step_decimal <= 0:
        raise ValueError("threshold step must be positive")
    if stop_decimal < start_decimal:
        raise ValueError("threshold stop must be greater than or equal to start")

    thresholds = []
    value = start_decimal
    while value <= stop_decimal:
        thresholds.append(float(value))
        value += step_decimal
    return thresholds


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _binary_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores > threshold).astype(np.int64)

    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))

    recall = _safe_ratio(true_positive, true_positive + false_negative)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    accuracy = _safe_ratio(
        true_positive + true_negative,
        labels.size,
    )
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    tnr = _safe_ratio(true_negative, true_negative + false_positive)
    fpr = _safe_ratio(false_positive, true_negative + false_positive)

    return {
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "tnr": tnr,
        "fpr": fpr,
        "auroc": float(roc_auc_score(labels, scores)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def validate_query_scores(query_scores):
    missing = REQUIRED_QUERY_SCORE_COLUMNS.difference(query_scores.columns)
    if missing:
        raise ValueError(
            "query score CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if query_scores.empty:
        raise ValueError("query score CSV is empty")
    if query_scores[list(REQUIRED_QUERY_SCORE_COLUMNS)].isnull().any().any():
        raise ValueError("query score CSV contains null generator, label, or score values")

    labels = set(pd.to_numeric(query_scores["query_label"]).astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError("query_label must contain only 0 (real) and 1 (fake)")

    for generator, group in query_scores.groupby("generator", sort=False):
        generator_labels = set(
            pd.to_numeric(group["query_label"]).astype(int).unique()
        )
        if generator_labels != {0, 1}:
            raise ValueError(
                f"generator {generator!r} must contain both real and fake queries"
            )


def scan_query_scores(query_scores, thresholds):
    """Compute threshold-dependent metrics independently for each generator."""

    validate_query_scores(query_scores)
    thresholds = list(thresholds)
    if not thresholds:
        raise ValueError("at least one threshold is required")

    rows = []
    for generator, group in query_scores.groupby("generator", sort=False):
        labels = pd.to_numeric(group["query_label"]).to_numpy(dtype=np.int64)
        scores = pd.to_numeric(group["mmd_score"]).to_numpy(dtype=np.float64)
        num_real = int(np.sum(labels == 0))
        num_fake = int(np.sum(labels == 1))
        for threshold in thresholds:
            metrics = _binary_metrics(labels, scores, float(threshold))
            rows.append({
                "generator": str(generator),
                "threshold": float(threshold),
                "num_real": num_real,
                "num_fake": num_fake,
                **metrics,
            })
    return pd.DataFrame(rows)


def macro_average_scan(per_generator_scan):
    """Match Table 1 by averaging every metric equally across generators."""

    required = {"threshold", "generator", *THRESHOLD_METRICS}
    missing = required.difference(per_generator_scan.columns)
    if missing:
        raise ValueError(
            "per-generator scan is missing required columns: "
            + ", ".join(sorted(missing))
        )

    aggregations = {
        metric: (metric, "mean") for metric in THRESHOLD_METRICS
    }
    aggregations["num_generators"] = ("generator", "nunique")
    return (
        per_generator_scan.groupby("threshold", as_index=False)
        .agg(**aggregations)
        .sort_values("threshold")
        .reset_index(drop=True)
    )


def aggregate_run_macros(run_macros):
    """Aggregate repeated runs, using sample standard deviation for uncertainty."""

    if not run_macros:
        raise ValueError("at least one run macro is required")
    combined = pd.concat(run_macros, ignore_index=True)
    rows = []
    for threshold, group in combined.groupby("threshold", sort=True):
        row = {
            "threshold": float(threshold),
            "num_runs": int(group["run_index"].nunique()),
            "num_generators": int(group["num_generators"].iloc[0]),
        }
        for metric in THRESHOLD_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
