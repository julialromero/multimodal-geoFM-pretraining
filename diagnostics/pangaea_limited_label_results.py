#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _load_results(index_path: Path, limited_labels: float) -> List[dict]:
    with index_path.open() as f:
        data = json.load(f)
    results = []
    for entry in data.get("results", []):
        if entry.get("benchmark") != "pangaea":
            continue
        if entry.get("limited_labels") != limited_labels:
            continue
        results.append(entry)
    return results


def _normalize_by_task(rows: List[dict]) -> Dict[Tuple[str, str], float]:
    task_values: Dict[str, List[float]] = {}
    for row in rows:
        task = row["dataset"]
        value = row["metrics"]["mIoU"]
        task_values.setdefault(task, []).append(value)

    task_minmax: Dict[str, Tuple[float, float]] = {}
    for task, values in task_values.items():
        task_minmax[task] = (min(values), max(values))

    normalized: Dict[Tuple[str, str], float] = {}
    for row in rows:
        task = row["dataset"]
        model_id = row["model_id"]
        value = row["metrics"]["mIoU"]
        vmin, vmax = task_minmax[task]
        if vmax == vmin:
            norm = 1.0
        else:
            norm = (value - vmin) / (vmax - vmin)
        normalized[(task, model_id)] = norm
    return normalized


def _boxplot(
    model_order: List[str],
    normalized_by_task: Dict[Tuple[str, str], float],
    output_path: Path,
) -> None:
    values_by_model: Dict[str, List[float]] = {m: [] for m in model_order}
    for (task, model_id), value in normalized_by_task.items():
        if model_id in values_by_model:
            values_by_model[model_id].append(value)

    data = [values_by_model[m] for m in model_order]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    colors = plt.get_cmap("Set1").colors
    boxplot_kwargs = {
        "showfliers": True,
        "patch_artist": True,
        "medianprops": {"linewidth": 2.0, "color": "black"},
    }
    try:
        boxes = ax.boxplot(data, tick_labels=model_order, **boxplot_kwargs)
    except TypeError:
        boxes = ax.boxplot(data, labels=model_order, **boxplot_kwargs)
    for i, box in enumerate(boxes["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
    ax.set_ylabel("Normalized mIoU (per-task min-max)")
    ax.set_xlabel("Model")
    ax.set_title("Pangaea Few-Shot (10% labels) Normalized Task Performances")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Pangaea limited-label results from diagnostics/index.json."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("diagnostics/index.json"),
        help="Path to diagnostics/index.json",
    )
    parser.add_argument(
        "--limited-labels",
        type=float,
        default=0.1,
        help="Limited label fraction to filter on",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/pangaea_limited_labels"),
        help="Output directory for plots and tables",
    )
    args = parser.parse_args()

    rows = _load_results(args.index, args.limited_labels)
    if not rows:
        raise SystemExit("No matching pangaea limited-label results found.")

    normalized = _normalize_by_task(rows)

    raw_df = pd.DataFrame(
        [
            {
                "model_id": row["model_id"],
                "dataset": row["dataset"],
                "mIoU": row["metrics"]["mIoU"],
                "mF1": row["metrics"]["mF1"],
                "mAcc": row["metrics"]["mAcc"],
            }
            for row in rows
        ]
    )
    raw_mean = raw_df.groupby("model_id")["mIoU"].mean().rename("mean_mIoU")
    raw_df = raw_df.merge(raw_mean, on="model_id")
    raw_df = raw_df.sort_values(["mean_mIoU", "model_id", "dataset"], ascending=[False, True, True])

    norm_df = pd.DataFrame(
        [
            {
                "model_id": row["model_id"],
                "dataset": row["dataset"],
                "normalized_mIoU": normalized[(row["dataset"], row["model_id"])],
            }
            for row in rows
        ]
    )
    norm_mean = norm_df.groupby("model_id")["normalized_mIoU"].mean().rename("mean_normalized_mIoU")
    norm_df = norm_df.merge(norm_mean, on="model_id")
    norm_df = norm_df.sort_values(
        ["mean_normalized_mIoU", "model_id", "dataset"], ascending=[False, True, True]
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_dir / "pangaea_fewshot10_raw_results.csv", index=False)
    norm_df.to_csv(output_dir / "pangaea_fewshot10_normalized_results.csv", index=False)

    model_order = (
        norm_df.groupby("model_id")["mean_normalized_mIoU"]
        .first()
        .sort_values(ascending=False)
        .index.tolist()
    )
    plot_path = output_dir / "pangaea_fewshot10_normalized_boxplot.png"
    _boxplot(model_order, normalized, plot_path)
    print(f"Saved boxplot: {plot_path}")


if __name__ == "__main__":
    main()
