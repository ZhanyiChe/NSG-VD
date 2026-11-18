#!/usr/bin/env python3
"""Scan MMD decision thresholds and draw NSG-VD Figure 3 style curves."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.threshold_scan import (  # noqa: E402
    THRESHOLD_METRICS,
    aggregate_real_only_runs,
    aggregate_run_macros,
    build_thresholds,
    macro_average_scan,
    scan_real_only_query_scores,
    scan_query_scores,
)


FIGURE_METRICS = (
    ("recall", "Recall", "#F28E2B", "s"),
    ("accuracy", "Accuracy", "#4E79A7", "^"),
    ("f1", "F1", "#E15759", "o"),
    ("precision", "Precision", "#59A14F", "D"),
    ("real_accuracy", "Real Accuracy (TNR)", "#B07AA1", "v"),
)


def _parse_label_and_paths(value, allow_multiple):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=CSV_PATH")
    label, raw_paths = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("experiment label cannot be empty")
    paths = [Path(item.strip()).expanduser().resolve() for item in raw_paths.split(",")]
    if not allow_multiple and len(paths) != 1:
        raise argparse.ArgumentTypeError("--run accepts exactly one CSV path")
    if any(not str(path) for path in paths):
        raise argparse.ArgumentTypeError("CSV path cannot be empty")
    for path in paths:
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"query score CSV does not exist: {path}")
    return label, paths


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute Table-1-style macro metrics from query_scores.csv over "
            "a threshold grid and save Figure-3-style plots."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="add one reference strategy; may be repeated",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="LABEL=CSV1,CSV2,...",
        help="add repeated runs such as five random seeds",
    )
    parser.add_argument(
        "--real-only-run",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help=(
            "add one unique-query real-only score file for a --run strategy; "
            "when omitted, Real Accuracy falls back to the paired score file"
        ),
    )
    parser.add_argument(
        "--real-only-group",
        action="append",
        default=[],
        metavar="LABEL=CSV1,CSV2,...",
        help="add repeated real-only runs matching a --group strategy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-start", type=float, default=0.4)
    parser.add_argument("--threshold-stop", type=float, default=1.3)
    parser.add_argument("--threshold-step", type=float, default=0.1)
    parser.add_argument(
        "--title",
        default="Impact of decision threshold",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    args = parser.parse_args()
    if not args.run and not args.group:
        parser.error("at least one --run or --group is required")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def _load_experiments(run_specs, group_specs):
    experiments = []
    labels = set()
    for raw_spec in run_specs:
        label, paths = _parse_label_and_paths(raw_spec, allow_multiple=False)
        if label in labels:
            raise ValueError(f"duplicate experiment label: {label}")
        labels.add(label)
        experiments.append((label, paths))
    for raw_spec in group_specs:
        label, paths = _parse_label_and_paths(raw_spec, allow_multiple=True)
        if label in labels:
            raise ValueError(f"duplicate experiment label: {label}")
        labels.add(label)
        experiments.append((label, paths))
    return experiments


def _scan_experiments(experiments, thresholds):
    per_generator_outputs = []
    per_run_outputs = []
    strategy_outputs = []

    for strategy, paths in experiments:
        run_macros = []
        for run_index, path in enumerate(paths):
            query_scores = pd.read_csv(path)
            per_generator = scan_query_scores(query_scores, thresholds)
            per_generator.insert(0, "source_csv", str(path))
            per_generator.insert(0, "run_index", run_index)
            per_generator.insert(0, "strategy", strategy)
            per_generator_outputs.append(per_generator)

            run_macro = macro_average_scan(per_generator)
            run_macro.insert(0, "source_csv", str(path))
            run_macro.insert(0, "run_index", run_index)
            run_macro.insert(0, "strategy", strategy)
            per_run_outputs.append(run_macro)
            run_macros.append(run_macro)

        strategy_summary = aggregate_run_macros(run_macros)
        strategy_summary.insert(0, "strategy", strategy)
        strategy_outputs.append(strategy_summary)

    return (
        pd.concat(per_generator_outputs, ignore_index=True),
        pd.concat(per_run_outputs, ignore_index=True),
        pd.concat(strategy_outputs, ignore_index=True),
    )


def _scan_real_only_experiments(experiments, thresholds):
    per_run_outputs = []
    strategy_outputs = []

    for strategy, paths in experiments:
        run_scans = []
        for run_index, path in enumerate(paths):
            query_scores = pd.read_csv(path)
            run_scan = scan_real_only_query_scores(query_scores, thresholds)
            run_scan.insert(0, "source_csv", str(path))
            run_scan.insert(0, "run_index", run_index)
            run_scan.insert(0, "strategy", strategy)
            per_run_outputs.append(run_scan)
            run_scans.append(run_scan)

        strategy_summary = aggregate_real_only_runs(run_scans)
        strategy_summary.insert(0, "strategy", strategy)
        strategy_outputs.append(strategy_summary)

    return (
        pd.concat(per_run_outputs, ignore_index=True),
        pd.concat(strategy_outputs, ignore_index=True),
    )


def _attach_real_accuracy(summary, standard_experiments, real_only_experiments):
    result = summary.copy()
    if not real_only_experiments:
        result["real_accuracy_mean"] = result["tnr_mean"]
        result["real_accuracy_std"] = result["tnr_std"]
        return result, None, None

    standard_labels = [label for label, _ in standard_experiments]
    real_only_labels = [label for label, _ in real_only_experiments]
    if set(standard_labels) != set(real_only_labels):
        missing = sorted(set(standard_labels).difference(real_only_labels))
        extra = sorted(set(real_only_labels).difference(standard_labels))
        raise ValueError(
            "real-only strategy labels must exactly match standard strategy "
            f"labels; missing={missing}, extra={extra}"
        )

    real_only_runs, real_only_summary = _scan_real_only_experiments(
        real_only_experiments,
        sorted(result["threshold"].unique()),
    )
    real_only_for_merge = real_only_summary.rename(columns={
        "num_runs": "real_only_num_runs",
        "num_real": "real_only_num_queries",
    })
    result = result.merge(
        real_only_for_merge,
        on=["strategy", "threshold"],
        how="left",
        validate="one_to_one",
    )
    if result["real_accuracy_mean"].isnull().any():
        raise ValueError("failed to match every strategy and threshold to real-only data")
    return result, real_only_runs, real_only_summary


def _resolve_y_limits(summary, y_min, y_max):
    values = []
    for metric, _, _, _ in FIGURE_METRICS:
        values.extend(summary[f"{metric}_mean"].tolist())
    if y_min is None:
        y_min = max(0.0, math.floor((min(values) - 0.05) * 10) / 10)
    if y_max is None:
        y_max = min(1.02, math.ceil((max(values) + 0.02) * 10) / 10)
    if y_max <= y_min:
        raise ValueError("y-axis maximum must be greater than minimum")
    return y_min, y_max


def _plot_figure3(summary, strategy_order, output_dir, title, dpi, y_min, y_max):
    columns = 2 if len(strategy_order) > 1 else 1
    rows = math.ceil(len(strategy_order) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.0 * columns, 4.8 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    y_min, y_max = _resolve_y_limits(summary, y_min, y_max)
    threshold_min = float(summary["threshold"].min())
    threshold_max = float(summary["threshold"].max())
    x_tick_start = math.ceil((threshold_min - 1e-9) * 10) / 10
    x_tick_stop = math.floor((threshold_max + 1e-9) * 10) / 10
    x_ticks = np.arange(x_tick_start, x_tick_stop + 0.05, 0.1)
    y_tick_start = math.ceil((y_min - 1e-9) * 10) / 10
    y_tick_stop = math.floor((y_max + 1e-9) * 10) / 10
    y_ticks = np.arange(y_tick_start, y_tick_stop + 0.05, 0.1)

    for axis, strategy in zip(axes.flat, strategy_order):
        current = summary[summary["strategy"] == strategy].sort_values("threshold")
        thresholds = current["threshold"].to_numpy(dtype=float)
        for metric, display_name, color, marker in FIGURE_METRICS:
            means = current[f"{metric}_mean"].to_numpy(dtype=float)
            stds = current[f"{metric}_std"].to_numpy(dtype=float)
            axis.plot(
                thresholds,
                means,
                label=display_name,
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=6.5,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.6,
            )
            if np.any(stds > 0):
                axis.fill_between(
                    thresholds,
                    np.clip(means - stds, 0.0, 1.0),
                    np.clip(means + stds, 0.0, 1.0),
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
        axis.axvline(1.0, color="#666666", linestyle="--", linewidth=1.2)
        axis.set_title(strategy)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks(x_ticks)
        axis.set_yticks(y_ticks)
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.tick_params(axis="x", which="both", labelbottom=True)
        axis.tick_params(axis="y", which="both", labelleft=True)
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.set_xlabel(r"Decision threshold $\tau$")
        axis.set_ylabel("Macro-average value")
        axis.legend(loc="best")

    for axis in axes.flat[len(strategy_order):]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_dir / "threshold_scan_figure3.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "threshold_scan_figure3.pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    args = _parse_args()
    experiments = _load_experiments(args.run, args.group)
    real_only_experiments = _load_experiments(
        args.real_only_run,
        args.real_only_group,
    ) if (args.real_only_run or args.real_only_group) else []
    thresholds = build_thresholds(
        args.threshold_start,
        args.threshold_stop,
        args.threshold_step,
    )
    per_generator, per_run, summary = _scan_experiments(experiments, thresholds)
    summary, real_only_runs, real_only_summary = _attach_real_accuracy(
        summary,
        experiments,
        real_only_experiments,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_generator.to_csv(output_dir / "threshold_scan_by_generator.csv", index=False)
    per_run.to_csv(output_dir / "threshold_scan_by_run_macro.csv", index=False)
    summary.to_csv(output_dir / "threshold_scan_macro.csv", index=False)
    if real_only_runs is not None:
        real_only_runs.to_csv(
            output_dir / "threshold_scan_real_only_by_run.csv",
            index=False,
        )
        real_only_summary.to_csv(
            output_dir / "threshold_scan_real_only.csv",
            index=False,
        )
    _plot_figure3(
        summary,
        [label for label, _ in experiments],
        output_dir,
        args.title,
        args.dpi,
        args.y_min,
        args.y_max,
    )

    print(f"Thresholds: {thresholds}")
    print(f"Saved: {output_dir / 'threshold_scan_by_generator.csv'}")
    print(f"Saved: {output_dir / 'threshold_scan_by_run_macro.csv'}")
    print(f"Saved: {output_dir / 'threshold_scan_macro.csv'}")
    if real_only_runs is not None:
        print(f"Saved: {output_dir / 'threshold_scan_real_only_by_run.csv'}")
        print(f"Saved: {output_dir / 'threshold_scan_real_only.csv'}")
    print(f"Saved: {output_dir / 'threshold_scan_figure3.png'}")
    print(f"Saved: {output_dir / 'threshold_scan_figure3.pdf'}")
    at_default = summary[np.isclose(summary["threshold"], 1.0)]
    if not at_default.empty:
        columns = [
            "strategy",
            "recall_mean",
            "accuracy_mean",
            "f1_mean",
            "real_accuracy_mean",
            "auroc_mean",
        ]
        print("\nMetrics at threshold 1.0:")
        print(at_default[columns].to_string(index=False))


if __name__ == "__main__":
    main()
