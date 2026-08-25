#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import os
from datetime import datetime
from collections import defaultdict
from typing import Iterable


def _safe_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_limited_label_train(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if not stripped.startswith("limited_label_train:"):
                    continue
                value = stripped.split(":", 1)[1].split("#", 1)[0].strip()
                if not value:
                    return None
                try:
                    return float(value)
                except ValueError:
                    return None
    except FileNotFoundError:
        return None
    return None


def _read_config_encoder(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if not stripped.startswith("encoder:"):
                    continue
                value = stripped.split(":", 1)[1].split("#", 1)[0].strip()
                return value
    except FileNotFoundError:
        return ""
    return ""


def _label_split(value: float | None) -> str:
    if value is None:
        return "unknown"
    if abs(value - 0.1) < 1e-6:
        return "10%"
    if abs(value - 1.0) < 1e-6:
        return "full"
    return str(value)



def _parse_experiment_date(exp_dir: str, experiment: str) -> str:
    for candidate in (os.path.basename(exp_dir), experiment):
        if not candidate:
            continue
        parts = candidate.split("_")
        if len(parts) < 2:
            continue
        date_part, time_part = parts[0], parts[1]
        if len(date_part) != 8 or len(time_part) != 6:
            continue
        try:
            dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
        except ValueError:
            continue
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return ""

def _load_rows_csv(paths: Iterable[str]):
    rows = []
    config_cache: dict[str, str] = {}
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    row["_source"] = path
                    exp_dir = row.get("exp_dir", "")
                    if exp_dir and not os.path.exists(exp_dir):
                        print(f"[warn] missing exp_dir: {exp_dir}")
                    if exp_dir not in config_cache:
                        config_path = os.path.join(exp_dir, "configs", "config.yaml") if exp_dir else ""
                        value = _read_limited_label_train(config_path) if config_path else None
                        config_cache[exp_dir] = _label_split(value)
                        encoder_name = _read_config_encoder(config_path) if config_path else ""
                        if encoder_name and row.get("model") and encoder_name != row.get("model"):
                            print("[warn] model mismatch for {0}: csv={1} config={2}".format(exp_dir, row.get("model"), encoder_name))
                    row["label_split"] = config_cache.get(exp_dir, "unknown")
                    if row["label_split"] == "unknown" and exp_dir:
                        print(f"[warn] label_split unknown for {exp_dir}")
                    row["experiment_date"] = _parse_experiment_date(exp_dir, row.get("experiment", ""))
                    rows.append(row)
        except FileNotFoundError:
            continue
    return rows


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _compute_summary(rows: list[dict[str, str]]):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("normalization", ""),
            row.get("preprocessing", ""),
            row.get("label_split", ""),
        )
        grouped[key].append(row)

    summary = []
    for key, items in grouped.items():
        preprocessing = key[3]
        if "ssl4eo" in preprocessing:
            downstream_norm = "SSL4EO Mean/Std"
        elif "divideby10000" in preprocessing:
            downstream_norm = "Scaled"
        else:
            downstream_norm = "Mean/Std"

        metrics = {"mIoU": [], "mF1": [], "mAcc": []}
        dates = []
        for item in items:
            for metric in metrics:
                value = _safe_float(item.get(metric, ""))
                if value is not None:
                    metrics[metric].append(value)
            date_val = item.get("experiment_date", "")
            if date_val:
                dates.append(date_val)

        def mean_std(values: list[float]):
            if not values:
                return None, None
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            return mean, var ** 0.5

        mIoU_mean, mIoU_std = mean_std(metrics["mIoU"])
        mF1_mean, mF1_std = mean_std(metrics["mF1"])
        mAcc_mean, mAcc_std = mean_std(metrics["mAcc"])
        summary.append(
            {
                "dataset": key[0],
                "model": key[1],
                "normalization": key[2],
                "preprocessing": key[3],
                "label_split": key[4],
                "experiment_date": min(dates) if dates else "",
                "downstream_preprocessing": downstream_norm,
                "runs": len(items),
                "mIoU_mean": mIoU_mean,
                "mIoU_std": mIoU_std,
                "mF1_mean": mF1_mean,
                "mF1_std": mF1_std,
                "mAcc_mean": mAcc_mean,
                "mAcc_std": mAcc_std,
            }
        )
    return summary


def _write_csv(path: str, rows: list[dict[str, str]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: str, summary: list[dict[str, str]]):
    lines = []
    header = [
        "dataset",
        "model",
        "normalization",
        "preprocessing",
        "label_split",
        "experiment_date",
        "downstream_preprocessing",
        "runs",
        "mIoU_mean",
        "mIoU_std",
        "mF1_mean",
        "mF1_std",
        "mAcc_mean",
        "mAcc_std",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["model"]),
                    str(row["normalization"]),
                    str(row["preprocessing"]),
                    str(row["label_split"]),
                    str(row["experiment_date"]),
                    str(row["downstream_preprocessing"]),
                    str(row["runs"]),
                    _format_float(row["mIoU_mean"]),
                    _format_float(row["mIoU_std"]),
                    _format_float(row["mF1_mean"]),
                    _format_float(row["mF1_std"]),
                    _format_float(row["mAcc_mean"]),
                    _format_float(row["mAcc_std"]),
                ]
            )
            + " |"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")



def _plot_bars(summary: list[dict[str, str]], output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    by_dataset = defaultdict(list)
    for row in summary:
        by_dataset[row["dataset"]].append(row)

    for dataset, rows in by_dataset.items():
        stats = defaultdict(list)
        for row in rows:
            if row["mIoU_mean"] is None:
                continue
            stats[row["model"]].append(row["mIoU_mean"])
        if not stats:
            continue

        models = sorted(stats.keys())
        means = [sum(stats[m]) / len(stats[m]) for m in models]

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 4))
        ax.bar(models, means)
        ax.set_ylabel("mIoU mean")
        ax.set_title(f"mIoU mean across normalization/preprocessing ({dataset})")
        ax.set_xticklabels(models, rotation=30, ha="right")
        fig.tight_layout()
        fig_path = os.path.join(output_dir, f"bar_mIoU_{dataset}.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)


def _write_split_outputs(rows: list[dict[str, str]], output_dir: str, label: str) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: (r.get("experiment_date") or "9999-99-99 99:99:99"))

    combined_path = os.path.join(output_dir, "combined_metrics.csv")
    _write_csv(combined_path, rows_sorted)

    summary = _compute_summary(rows_sorted)
    summary_sorted = sorted(
        summary,
        key=lambda r: (
            (r["experiment_date"] or "9999-99-99 99:99:99"),
            r["dataset"],
            r["model"],
            r["normalization"],
            r["preprocessing"],
            r["label_split"],
        ),
    )

    summary_rows = [
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "normalization": row["normalization"],
            "preprocessing": row["preprocessing"],
            "label_split": row["label_split"],
            "experiment_date": row["experiment_date"],
            "downstream_preprocessing": row["downstream_preprocessing"],
            "runs": row["runs"],
            "mIoU_mean": _format_float(row["mIoU_mean"]),
            "mIoU_std": _format_float(row["mIoU_std"]),
            "mF1_mean": _format_float(row["mF1_mean"]),
            "mF1_std": _format_float(row["mF1_std"]),
            "mAcc_mean": _format_float(row["mAcc_mean"]),
            "mAcc_std": _format_float(row["mAcc_std"]),
        }
        for row in summary_sorted
    ]

    summary_csv = os.path.join(output_dir, "summary_metrics.csv")
    _write_csv(summary_csv, summary_rows)

    summary_md = os.path.join(output_dir, "summary_metrics.md")
    _write_markdown(summary_md, summary_sorted)

    _plot_bars(summary_sorted, output_dir)

    print(f"Wrote {combined_path} ({label})")
    print(f"Wrote {summary_csv} ({label})")
    print(f"Wrote {summary_md} ({label})")
    print(f"Figures saved to {output_dir} ({label})")




def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pangaea batch evaluation results.")
    parser.add_argument(
        "--work-dir",
        default="/local/ms-data/pangaea-bench/batch_runs",
        help="Directory with batch run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write summary tables and figures.",
    )
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    output_dir = args.output_dir or os.path.join(work_dir, "metrics_summary")
    _ensure_dir(output_dir)

    metric_paths = glob.glob(os.path.join(work_dir, "metrics", "*_metrics.csv"))
    metric_paths += glob.glob(os.path.join(work_dir, "metrics", "fewshot10", "*_metrics.csv"))
    metric_paths = sorted(set(metric_paths))

    rows = _load_rows_csv(metric_paths)
    if not rows:
        raise SystemExit(f"No metrics.csv files found under {work_dir}")

    rows_sorted = sorted(rows, key=lambda r: (r.get("experiment_date") or "9999-99-99 99:99:99"))

    rows_by_split = {"full": [], "10%": []}
    for row in rows_sorted:
        label = (row.get("label_split") or "").strip().lower()
        if label == "full":
            rows_by_split["full"].append(row)
        elif label == "10%":
            rows_by_split["10%"].append(row)

    full_rows = rows_by_split["full"]
    ten_rows = rows_by_split["10%"]

    if full_rows:
        _write_split_outputs(full_rows, output_dir, "full")

    if ten_rows:
        ten_dir = os.path.join(output_dir, "10pct")
        _ensure_dir(ten_dir)
        _write_split_outputs(ten_rows, ten_dir, "10%")

    if not full_rows and not ten_rows:
        print("No rows found for label_split full or 10%")


if __name__ == "__main__":
    main()
