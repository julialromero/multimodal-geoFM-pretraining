"""Plot downstream evaluation outputs from run_downstream tasks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ciip.evaluation.output_utils import build_model_tag


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


NEUCO_TASKS = [
    "biomass_mean",
    "biomass_std",
    "clouds_reg",
    "crops",
    "heatisland_mean",
    "heatisland_std",
    "landcover_agriculture",
    "landcover_forest",
]


@dataclass(frozen=True)
class FewshotRecord:
    model_label: str
    task: str
    x_value: float
    score: float


@dataclass(frozen=True)
class UnifiedRecord:
    model_label: str
    task: str
    score: float


def _read_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON: {path}") from exc


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_score(payload: Mapping[str, object]) -> Optional[float]:
    for key in ("raw_score", "r2", "accuracy_mean", "score"):
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return None


def _format_model_label(payload: Mapping[str, object]) -> str:
    model_label = build_model_tag(
        model_type=payload.get("model_type"),
        model_weights=payload.get("model_weights"),
        model_path=payload.get("model_path"),
        ciip_epoch=payload.get("ciip_epoch"),
    )
    normalization = payload.get("normalization_method") or payload.get("normalization_label")
    model_label = _clean_model_label(model_label)
    if normalization:
        return f"{model_label} ({_clean_model_label(str(normalization))})"
    return model_label


def _clean_model_label(label: str) -> str:
    cleaned = label.replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(r"\bciip checkpoint\b", "CIIP", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bclip\b", "CLIP", cleaned, flags=re.IGNORECASE)
    return cleaned


def _format_tick(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _iter_eurosat_fewshot_results(root: Path, *, knn_k: Optional[int]) -> Iterable[Tuple[str, float, float]]:
    if not root.exists():
        return
    for path in root.rglob("results.json"):
        if "eurosat_fewshot" not in path.parts:
            continue
        payload = _read_json(path)
        if payload.get("dataset") != "EuroSAT":
            continue
        if knn_k is not None and payload.get("knn_k") != knn_k:
            continue
        k_shot = _safe_float(payload.get("k_shot"))
        score = _safe_float(payload.get("accuracy_mean"))
        if k_shot is None or score is None:
            continue
        yield _format_model_label(payload), k_shot, score


def plot_eurosat_fewshot(root: Path, output_dir: Path, *, knn_k: Optional[int]) -> Optional[Path]:
    records = list(_iter_eurosat_fewshot_results(root, knn_k=knn_k))
    if not records:
        print("[plot_downstream] No EuroSAT few-shot results found.")
        return None

    data: Dict[str, List[Tuple[float, float]]] = {}
    x_values: set[float] = set()
    for model, k_shot, score in records:
        data.setdefault(model, []).append((k_shot, score))
        x_values.add(k_shot)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, points in sorted(data.items()):
        points = sorted(points, key=lambda item: item[0])
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=model,
        )
    ax.set_xlabel("k-shot")
    ax.set_ylabel("Accuracy")
    title = "EuroSAT few-shot (1-NN)"
    if knn_k is not None:
        title += f" (kNN k={knn_k})"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    tick_values = sorted(x_values)
    ax.set_xticks(tick_values)
    ax.set_xticklabels([_format_tick(value) for value in tick_values])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)
    fig.tight_layout(rect=[0, 0, 0.78, 1])

    out_path = output_dir / "eurosat_fewshot_lineplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_downstream] Saved {out_path}")
    return out_path


def _iter_neuco_fewshot_results(root: Path) -> Iterable[FewshotRecord]:
    if not root.exists():
        return
    for path in root.rglob("task_*.json"):
        if "neuco_fewshot" not in path.parts:
            continue
        payload = _read_json(path)
        task = payload.get("task")
        if task not in NEUCO_TASKS:
            continue
        x_value = _safe_float(payload.get("limited_label_train"))
        score = _extract_score(payload)
        if x_value is None or score is None:
            continue
        yield FewshotRecord(
            model_label=_format_model_label(payload),
            task=task,
            x_value=x_value,
            score=score,
        )


def _group_fewshot(records: Iterable[FewshotRecord]) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    grouped: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    for record in records:
        grouped.setdefault(record.task, {}).setdefault(record.model_label, []).append(
            (record.x_value, record.score)
        )
    return grouped


def plot_neuco_fewshot_tasks(root: Path, output_dir: Path) -> Optional[Path]:
    records = list(_iter_neuco_fewshot_results(root))
    if not records:
        print("[plot_downstream] No NeuCo few-shot results found.")
        return None

    grouped = _group_fewshot(records)
    tick_values = sorted({record.x_value for record in records})
    n_tasks = len(NEUCO_TASKS)
    cols = 2
    rows = int(np.ceil(n_tasks / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True)
    axes = np.atleast_2d(axes)

    for idx, task in enumerate(NEUCO_TASKS):
        ax = axes[idx // cols, idx % cols]
        task_data = grouped.get(task, {})
        for model, points in sorted(task_data.items()):
            points = sorted(points, key=lambda item: item[0])
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                label=model,
            )
        ax.set_title(task)
        ax.set_ylabel("Raw score")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(tick_values)
        ax.set_xticklabels([_format_tick(value) for value in tick_values])
        if idx // cols == rows - 1:
            ax.set_xlabel("Limited-label train fraction")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("NeuCo few-shot (limited-label)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = output_dir / "neuco_fewshot_tasks.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_downstream] Saved {out_path}")
    return out_path


def _aggregate_neuco_fewshot(records: Iterable[FewshotRecord]) -> Tuple[Dict[str, Dict[float, float]], List[float]]:
    by_model: Dict[str, Dict[float, List[float]]] = {}
    x_values: set[float] = set()
    for record in records:
        by_model.setdefault(record.model_label, {}).setdefault(record.x_value, []).append(record.score)
        x_values.add(record.x_value)

    averaged: Dict[str, Dict[float, float]] = {}
    for model, by_x in by_model.items():
        averaged[model] = {x: float(np.mean(scores)) for x, scores in by_x.items() if scores}
    return averaged, sorted(x_values)


def _task_normalized_scores(
    records: Iterable[FewshotRecord],
) -> Tuple[Dict[str, Dict[float, float]], List[float]]:
    by_task: Dict[str, Dict[float, Dict[str, float]]] = {}
    for record in records:
        by_task.setdefault(record.task, {}).setdefault(record.x_value, {})[record.model_label] = record.score

    models = {record.model_label for record in records}
    x_values = sorted({record.x_value for record in records})
    normalized: Dict[str, Dict[float, List[float]]] = {model: {x: [] for x in x_values} for model in models}

    for task, by_x in by_task.items():
        for x_value, model_scores in by_x.items():
            scores = [score for score in model_scores.values() if score is not None]
            if not scores:
                continue
            min_score, max_score = min(scores), max(scores)
            denom = max_score - min_score
            for model in models:
                score = model_scores.get(model)
                if score is None:
                    continue
                if denom == 0:
                    normalized_score = 1.0
                else:
                    normalized_score = (score - min_score) / denom
                normalized[model][x_value].append(normalized_score)

    averaged: Dict[str, Dict[float, float]] = {}
    for model, by_x in normalized.items():
        averaged[model] = {
            x: float(np.mean(values)) if values else float("nan") for x, values in by_x.items()
        }
    return averaged, x_values


def plot_neuco_fewshot_aggregates(root: Path, output_dir: Path) -> Optional[Path]:
    records = list(_iter_neuco_fewshot_results(root))
    if not records:
        print("[plot_downstream] No NeuCo few-shot results found for aggregation.")
        return None

    avg_scores, x_values = _aggregate_neuco_fewshot(records)
    norm_scores, _ = _task_normalized_scores(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    for model, by_x in sorted(avg_scores.items()):
        points = [(x, by_x.get(x)) for x in x_values if x in by_x]
        if not points:
            continue
        points = sorted(points, key=lambda item: item[0])
        axes[0].plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=model,
        )
    axes[0].set_title("Average raw score")
    axes[0].set_ylabel("Raw score")
    axes[0].set_xlabel("Limited-label train fraction")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].set_xticks(x_values)
    axes[0].set_xticklabels([_format_tick(value) for value in x_values])

    for model, by_x in sorted(norm_scores.items()):
        points = [(x, by_x.get(x)) for x in x_values if x in by_x]
        if not points:
            continue
        points = sorted(points, key=lambda item: item[0])
        axes[1].plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=model,
        )
    axes[1].set_title("Task-normalized score")
    axes[1].set_ylabel("Normalized score")
    axes[1].set_xlabel("Limited-label train fraction")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels([_format_tick(value) for value in x_values])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("NeuCo few-shot aggregates")
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    out_path = output_dir / "neuco_fewshot_aggregates.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_downstream] Saved {out_path}")
    return out_path


def _parse_fraction_key(key: str) -> Optional[float]:
    try:
        return float(str(key).replace(",", "."))
    except ValueError:
        return None


def _collect_unified_eurosat_knn(root: Path) -> Dict[str, Dict[float, float]]:
    results: Dict[str, Dict[float, float]] = {}
    if not root.exists():
        return results
    for path in root.rglob("*metrics.json"):
        name = path.name.lower()
        if "eurosat" not in name or "knn" not in name:
            continue
        model_label = _clean_model_label(path.relative_to(root).parts[0])
        payload = _read_json(path)
        for key, metrics in payload.items():
            frac = _parse_fraction_key(key)
            if frac is None or not isinstance(metrics, dict):
                continue
            score = _safe_float(metrics.get("test_accuracy"))
            if score is None:
                continue
            results.setdefault(model_label, {})[frac] = score
    return results


def plot_eurosat_unified_eval(root: Path, output_dir: Path) -> Optional[Path]:
    results = _collect_unified_eurosat_knn(root)
    if not results:
        print("[plot_downstream] No EuroSAT unified-eval kNN results found.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, scores in sorted(results.items()):
        points = sorted(scores.items(), key=lambda item: item[0])
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=model,
        )
    ax.set_xlabel("Training split fraction")
    ax.set_ylabel("Accuracy")
    ax.set_title("EuroSAT unified evaluation (kNN)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = output_dir / "eurosat_unified_eval_knn.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_downstream] Saved {out_path}")
    return out_path


def _iter_unified_neuco_results(root: Path) -> Iterable[UnifiedRecord]:
    if not root.exists():
        return
    for path in root.rglob("results_summary.json"):
        payload = _read_json(path)
        task_results = payload.get("task_results")
        if not isinstance(task_results, dict):
            continue
        model_label = _clean_model_label(path.relative_to(root).parts[0])
        for task, task_payload in task_results.items():
            if task not in NEUCO_TASKS:
                continue
            if isinstance(task_payload, dict):
                score = _safe_float(task_payload.get("raw_score"))
                if score is None:
                    score = _safe_float(task_payload.get("score"))
            else:
                score = _safe_float(task_payload)
            if score is None:
                continue
            yield UnifiedRecord(model_label=model_label, task=task, score=score)


def _write_neuco_table(
    records: Iterable[UnifiedRecord],
    output_dir: Path,
) -> Optional[Tuple[Path, Path, List[str]]]:
    table: Dict[str, Dict[str, float]] = {}
    for record in records:
        table.setdefault(record.model_label, {})[record.task] = record.score
    if not table:
        return None

    rows: List[Tuple[str, Dict[str, float], float]] = []
    for model, scores in table.items():
        values = [scores.get(task) for task in NEUCO_TASKS if scores.get(task) is not None]
        mean_val = float(np.mean(values)) if values else float("nan")
        rows.append((model, scores, mean_val))
    rows.sort(key=lambda item: item[2], reverse=True)

    csv_path = output_dir / "neuco_unified_eval_table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *NEUCO_TASKS, "mean"])
        for model, scores, mean_val in rows:
            writer.writerow([model, *[scores.get(task, "") for task in NEUCO_TASKS], mean_val])

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(rows) + 1.5))
    ax.axis("off")
    table_data = [
        [model, *[f"{scores.get(task, float('nan')):.4f}" for task in NEUCO_TASKS], f"{mean_val:.4f}"]
        for model, scores, mean_val in rows
    ]
    column_labels = ["Model", *[task.replace("_", " ") for task in NEUCO_TASKS], "Mean"]
    table_artist = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        cellLoc="center",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(10)
    table_artist.scale(1, 1.4)
    fig.tight_layout()

    png_path = output_dir / "neuco_unified_eval_table.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[plot_downstream] Saved {png_path} and {csv_path}")
    return png_path, csv_path, [row[0] for row in rows]


def plot_neuco_unified_eval_boxplot(
    records: Iterable[UnifiedRecord],
    output_dir: Path,
    model_order: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    table: Dict[str, Dict[str, float]] = {}
    for record in records:
        table.setdefault(record.model_label, {})[record.task] = record.score
    if not table:
        print("[plot_downstream] No NeuCo unified-eval results found.")
        return None

    models = list(model_order) if model_order else sorted(table.keys())
    task_minmax: Dict[str, Tuple[float, float]] = {}
    for task in NEUCO_TASKS:
        scores = [table[model].get(task) for model in table if table[model].get(task) is not None]
        if scores:
            task_minmax[task] = (min(scores), max(scores))

    normalized_scores: List[List[float]] = []
    for model in models:
        scores = []
        for task in NEUCO_TASKS:
            value = table.get(model, {}).get(task)
            if value is None:
                continue
            min_score, max_score = task_minmax.get(task, (value, value))
            denom = max_score - min_score
            scores.append(1.0 if denom == 0 else (value - min_score) / denom)
        normalized_scores.append(scores)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(normalized_scores, labels=models, vert=True, patch_artist=True)
    ax.set_ylabel("Task-normalized score")
    ax.set_title("NeuCo unified evaluation (task-normalized)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    out_path = output_dir / "neuco_unified_eval_boxplot.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_downstream] Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot downstream evaluation outputs.")
    parser.add_argument(
        "--eurosat-fewshot-root",
        type=Path,
        default=Path("diagnostics/fewshot_eval"),
        help="Root directory for EuroSAT few-shot outputs.",
    )
    parser.add_argument(
        "--neuco-fewshot-root",
        type=Path,
        default=Path("diagnostics/neuco_fewshot"),
        help="Root directory for NeuCo few-shot outputs.",
    )
    parser.add_argument(
        "--unified-eval-root",
        type=Path,
        default=Path("diagnostics/unified_eval"),
        help="Root directory for unified evaluation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/downstream_plots"),
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--eurosat-knn-k",
        type=int,
        default=None,
        help="Optional kNN k value to filter EuroSAT few-shot results.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_eurosat_fewshot(args.eurosat_fewshot_root, output_dir, knn_k=args.eurosat_knn_k)
    plot_neuco_fewshot_tasks(args.neuco_fewshot_root, output_dir)
    plot_neuco_fewshot_aggregates(args.neuco_fewshot_root, output_dir)
    plot_eurosat_unified_eval(args.unified_eval_root, output_dir)

    neuco_records = list(_iter_unified_neuco_results(args.unified_eval_root))
    table_result = _write_neuco_table(neuco_records, output_dir)
    model_order = table_result[2] if table_result else None
    plot_neuco_unified_eval_boxplot(neuco_records, output_dir, model_order=model_order)


if __name__ == "__main__":
    main()
