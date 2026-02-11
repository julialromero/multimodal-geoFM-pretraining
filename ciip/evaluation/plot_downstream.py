"""Plot downstream evaluation outputs from run_downstream tasks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
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

INTRINSIC_METRICS: Sequence[Tuple[str, str]] = (
    ("fishers", "FisherS"),
    ("mle", "MLE"),
    ("effective_rank", "Effective Rank"),
)

EUROSAT_LIMITED_K_SHOTS = (1.0, 10.0)
LIMITED_LABEL_NEUCO_FRACTION = 0.1
PANGAEA_EXPECTED_TASK_COUNT = 5

# Map benchmark-specific naming variants onto the intrinsic-dimension labels.
MODEL_LABEL_ALIASES = {
    "croma optical": "croma croma",
    "ciip s2 vit": "CIIP ciip 10kscale 1 28 ViT DAI epoch300",
    "llama3 ms clip": "llama3 ms clip base",
    "scalemae": "scalemae large rgb",
    "ssl4eo dino": "vitsmall16 s2 all dino",
    "ssl4eo moco": "vitsmall16 s2 all moco",
    "torchgeo resnet50 s2 all dino": "torchgeo resnet50 dino",
    "torchgeo resnet50 s2 all moco": "torchgeo resnet50 moco",
}

MODEL_DISPLAY_ALIASES = {
    "ciip ciip 10kscale 1 28 vit dai epoch300": "ViT-CIIP",
}

# Current model labels seen in downstream outputs (after normalization).
# Edit this list to customize --models-include values.
MODELS_INCLUDE_CHOICES = [
    "CIIP ciip 10kscale 1 28 ViT DAI epoch300",
    # "CIIP ciip bandwise 2 8 2026 vit bandwise 2dsin epoch300",
    "CIIP ciip text s2 2025 12 2 text s2 epoch70",
    # "CIIP ciip text s2 2026 02 01 vit ciip dai bandwise epoch300",
    # "croma croma",
    "llama3 ms CLIP base",
    "rcf 13ch",
    "remoteclip", 
    # "scalemae large rgb",
    # "torchgeo resnet50 dino",
    # "torchgeo resnet50 moco",
    # "vitsmall16 s2 all dino",
    # "vitsmall16 s2 all moco",

    #### PANGAEA NAMES
    "ciip s2 vit",
    "croma optical",
    "llama3 ms CLIP",
    # "remoteclip",
    "scalemae",
    "ssl4eo dino",
    "ssl4eo mae optical",
    "ssl4eo moco",
    "terramind large",
    "torchgeo_resnet50_s2_all_moco",
    "torchgeo_resnet50_s2_all_dino",
    "ciip_s2_vit_sin2d_epoch300",
]

# Models used for task-normalized score min/max reference.
# This can include extra models beyond MODELS_INCLUDE_CHOICES.
# Leave empty to default to MODELS_INCLUDE_CHOICES.
MODELS_TASK_NORMALIZE_CHOICES = [
    *MODELS_INCLUDE_CHOICES,
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


@dataclass(frozen=True)
class PangaeaRecord:
    model_label: str
    task: str
    score: float


@dataclass(frozen=True)
class AggregatePoint:
    mean: float
    std: Optional[float]


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
    # normalization = payload.get("normalization_method") or payload.get("normalization_label")
    model_label = _clean_model_label(model_label)
    # if normalization:
    #     return f"{model_label} {_clean_model_label(str(normalization))}".strip()
    return model_label


def _clean_model_label(label: str) -> str:
    cleaned = label.replace("backbone_only_", "").replace("backbone_only", "")
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(r"\bciip checkpoint\b", "CIIP", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bclip\b", "CLIP", cleaned, flags=re.IGNORECASE)
    
    return cleaned


def _canonical_model_label(label: str) -> str:
    cleaned = _clean_model_label(label)
    alias = MODEL_LABEL_ALIASES.get(cleaned.lower())
    return _clean_model_label(alias) if alias else cleaned


def _display_model_label(label: str) -> str:
    cleaned = _clean_model_label(label)
    return MODEL_DISPLAY_ALIASES.get(cleaned.lower(), cleaned)


def _normalize_model_filters(models: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not models:
        return None
    return {_clean_model_label(str(model)) for model in models}


def _normalize_plot_filters(models: Optional[Sequence[str]]) -> Optional[set[str]]:
    allowed = _normalize_model_filters(models)
    if allowed is None:
        return None
    expanded = set(allowed)
    expanded.update({_canonical_model_label(model) for model in allowed})
    return expanded


def _model_allowed(model_label: str, allowed: Optional[set[str]]) -> bool:
    return allowed is None or model_label in allowed


def _format_pangaea_model_label(payload: Mapping[str, object]) -> str:
    if payload.get("model_type"):
        return _format_model_label(payload)
    encoder = payload.get("encoder")
    if isinstance(encoder, str) and encoder:
        return _clean_model_label(encoder.split(".")[-1])
    return _clean_model_label("model")


def _find_run_manifest_config(path: Path, root: Path) -> Optional[Dict[str, object]]:
    for parent in (path, *path.parents):
        manifest = parent / "run_manifest.json"
        if manifest.exists():
            payload = _read_json(manifest)
            config = payload.get("config")
            if isinstance(config, dict):
                return config
            return None
        if parent == root:
            break
    return None


def _format_unified_model_label(path: Path, root: Path) -> str:
    root_name = path.relative_to(root).parts[0]
    config = _find_run_manifest_config(path, root)
    if isinstance(config, dict):
        model_type = config.get("model_type")
        model_weights = config.get("model_weights")
        model_path = config.get("model_path")
        ciip_epoch = config.get("ciip_epoch")
        if model_type == "ciip_checkpoint":
            epoch_match = re.search(r"epoch(\d+)", root_name)
            if ciip_epoch is None and epoch_match:
                ciip_epoch = int(epoch_match.group(1))
            if model_path is None and model_type and model_weights:
                prefix = f"{model_type}_{model_weights}_"
                if root_name.startswith(prefix):
                    remainder = root_name[len(prefix):]
                    if epoch_match:
                        remainder = remainder.replace(f"_epoch{epoch_match.group(1)}", "")
                    remainder = remainder.replace("_meanpool", "").strip("_")
                    if remainder:
                        model_path = remainder
        return _format_model_label(
            {
                "model_type": model_type,
                "model_weights": model_weights,
                "model_path": model_path,
                "ciip_epoch": ciip_epoch,
            }
        )
    return _clean_model_label(root_name.replace("_meanpool", ""))


def _format_tick(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _load_intrinsic_dimension_metrics(
    root: Path,
    *,
    allowed_models: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    metrics_by_model: Dict[str, Dict[str, float]] = {}
    if not root.exists():
        print(f"[plot_downstream] Intrinsic-dimension root not found: {root}")
        return metrics_by_model

    allowed = _normalize_model_filters(allowed_models)
    for path in root.rglob("results.json"):
        payload = _read_json(path)
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        model_label = _format_model_label(payload)
        canonical_label = _canonical_model_label(model_label)
        if allowed is not None and canonical_label not in allowed and model_label not in allowed:
            continue

        metric_values: Dict[str, float] = {}
        for metric_key, _ in INTRINSIC_METRICS:
            value = _safe_float(metrics.get(metric_key))
            if value is None:
                metric_values = {}
                break
            metric_values[metric_key] = value
        if metric_values:
            metrics_by_model[canonical_label] = metric_values
    return metrics_by_model


def _mean_by_task_and_model(
    values: Mapping[str, Mapping[str, List[float]]],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for task_name, by_model in values.items():
        task_scores: Dict[str, float] = {}
        for model_label, scores in by_model.items():
            if scores:
                task_scores[model_label] = float(np.mean(scores))
        if task_scores:
            result[task_name] = task_scores
    return result


def _task_normalized_model_aggregates(
    task_scores: Mapping[str, Mapping[str, float]],
) -> Dict[str, AggregatePoint]:
    normalized_by_model: Dict[str, List[float]] = defaultdict(list)
    for by_model in task_scores.values():
        if not by_model:
            continue
        min_score = min(by_model.values())
        max_score = max(by_model.values())
        denom = max_score - min_score
        for model_label, score in by_model.items():
            normalized = 1.0 if denom == 0 else (score - min_score) / denom
            normalized_by_model[model_label].append(normalized)

    aggregates: Dict[str, AggregatePoint] = {}
    for model_label, scores in normalized_by_model.items():
        if not scores:
            continue
        aggregates[model_label] = AggregatePoint(
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
        )
    return aggregates


def _collect_limited_label_task_scores(
    *,
    eurosat_root: Path,
    neuco_root: Path,
    pangaea_root: Path,
    pangaea_label_split: str,
    eurosat_knn_k: Optional[int],
    allowed_models: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    by_task: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for model_label, k_shot, score in _iter_eurosat_fewshot_results(
        eurosat_root,
        knn_k=eurosat_knn_k,
        allowed_models=allowed_models,
    ):
        if k_shot not in EUROSAT_LIMITED_K_SHOTS:
            continue
        task_name = f"EuroSAT {int(k_shot)}-shot"
        by_task[task_name][_canonical_model_label(model_label)].append(score)

    for record in _iter_neuco_fewshot_results(neuco_root, allowed_models=allowed_models):
        if abs(record.x_value - LIMITED_LABEL_NEUCO_FRACTION) > 1e-9:
            continue
        task_name = f"NeuCo {record.task}"
        by_task[task_name][_canonical_model_label(record.model_label)].append(record.score)

    for record in _iter_pangaea_fewshot_results(
        pangaea_root,
        label_split=pangaea_label_split,
        allowed_models=allowed_models,
    ):
        task_name = f"Pangaea {record.task}"
        by_task[task_name][_canonical_model_label(record.model_label)].append(record.score)

    return _mean_by_task_and_model(by_task)


def _collect_limited_label_benchmark_aggregates(
    *,
    eurosat_root: Path,
    neuco_root: Path,
    pangaea_root: Path,
    pangaea_label_split: str,
    eurosat_knn_k: Optional[int],
    allowed_models: Optional[Sequence[str]] = None,
    normalization_models: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, AggregatePoint]]:
    benchmark_scores: Dict[str, Dict[str, AggregatePoint]] = {}
    plot_allowed = _normalize_plot_filters(allowed_models)

    eurosat_by_model_shot: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for model_label, k_shot, score in _iter_eurosat_fewshot_results(
        eurosat_root,
        knn_k=eurosat_knn_k,
        allowed_models=allowed_models,
    ):
        if k_shot in EUROSAT_LIMITED_K_SHOTS:
            eurosat_by_model_shot[_canonical_model_label(model_label)][k_shot].append(score)

    eurosat_agg: Dict[str, AggregatePoint] = {}
    for model_label, by_shot in eurosat_by_model_shot.items():
        available = [k for k in EUROSAT_LIMITED_K_SHOTS if by_shot.get(k)]
        if len(available) < len(EUROSAT_LIMITED_K_SHOTS):
            missing = [str(int(k)) for k in EUROSAT_LIMITED_K_SHOTS if k not in available]
            print(
                f"[plot_downstream] EuroSAT aggregate missing k-shot values for {model_label}: {', '.join(missing)}"
            )
            continue
        shot_means = [float(np.mean(by_shot[k])) for k in EUROSAT_LIMITED_K_SHOTS]
        eurosat_agg[model_label] = AggregatePoint(mean=float(np.mean(shot_means)), std=None)
    if eurosat_agg:
        benchmark_scores["EuroSAT (avg 1-shot and 10-shot)"] = eurosat_agg

    neuco_task_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for record in _iter_neuco_fewshot_results(neuco_root, allowed_models=normalization_models):
        if abs(record.x_value - LIMITED_LABEL_NEUCO_FRACTION) > 1e-9:
            continue
        neuco_task_values[record.task][_canonical_model_label(record.model_label)].append(record.score)

    neuco_task_scores = _mean_by_task_and_model(neuco_task_values)
    neuco_agg = _task_normalized_model_aggregates(neuco_task_scores)
    if plot_allowed is not None:
        neuco_agg = {model: point for model, point in neuco_agg.items() if model in plot_allowed}
    if neuco_agg:
        benchmark_scores["NeuCo-Bench (10% labels)"] = neuco_agg

    pangaea_task_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for record in _iter_pangaea_fewshot_results(
        pangaea_root,
        label_split=pangaea_label_split,
        allowed_models=normalization_models,
    ):
        pangaea_task_values[record.task][_canonical_model_label(record.model_label)].append(record.score)

    pangaea_task_scores = _mean_by_task_and_model(pangaea_task_values)
    pangaea_task_counts: Dict[str, int] = defaultdict(int)
    for by_model in pangaea_task_scores.values():
        for model_label in by_model:
            pangaea_task_counts[model_label] += 1
    for model_label, task_count in sorted(pangaea_task_counts.items()):
        if task_count < PANGAEA_EXPECTED_TASK_COUNT:
            print(
                f"[plot_downstream] Pangaea aggregate model {model_label} has "
                f"{task_count}/{PANGAEA_EXPECTED_TASK_COUNT} tasks."
            )

    pangaea_agg = _task_normalized_model_aggregates(pangaea_task_scores)
    if plot_allowed is not None:
        pangaea_agg = {model: point for model, point in pangaea_agg.items() if model in plot_allowed}
    if pangaea_agg:
        benchmark_scores["Pangaea-Bench (10% labels)"] = pangaea_agg

    return benchmark_scores


def _collect_global_task_normalized_limited_label_scores(
    *,
    eurosat_root: Path,
    neuco_root: Path,
    pangaea_root: Path,
    pangaea_label_split: str,
    allowed_models: Optional[Sequence[str]] = None,
    normalization_models: Optional[Sequence[str]] = None,
) -> Dict[str, AggregatePoint]:
    task_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    plot_allowed = _normalize_plot_filters(allowed_models)

    # EuroSAT: use only kNN k=1, and keep all available few-shot settings as separate tasks.
    for model_label, k_shot, score in _iter_eurosat_fewshot_results(
        eurosat_root,
        knn_k=1,
        allowed_models=normalization_models,
    ):
        task_name = f"EuroSAT kNN1 {int(k_shot)}-shot"
        task_values[task_name][_canonical_model_label(model_label)].append(score)

    # NeuCo-Bench: use limited_label_train = 0.1.
    for record in _iter_neuco_fewshot_results(neuco_root, allowed_models=normalization_models):
        if abs(record.x_value - LIMITED_LABEL_NEUCO_FRACTION) > 1e-9:
            continue
        task_name = f"NeuCo {record.task}"
        task_values[task_name][_canonical_model_label(record.model_label)].append(record.score)

    # Pangaea-Bench: use the selected label split (default 10%).
    for record in _iter_pangaea_fewshot_results(
        pangaea_root,
        label_split=pangaea_label_split,
        allowed_models=normalization_models,
    ):
        task_name = f"Pangaea {record.task}"
        task_values[task_name][_canonical_model_label(record.model_label)].append(record.score)

    task_scores = _mean_by_task_and_model(task_values)
    aggregated = _task_normalized_model_aggregates(task_scores)
    if plot_allowed is not None:
        aggregated = {model: point for model, point in aggregated.items() if model in plot_allowed}
    return aggregated


def _annotate_scatter_points(
    ax: plt.Axes,
    x_values: Sequence[float],
    y_values: Sequence[float],
    labels: Sequence[str],
) -> None:
    for x_value, y_value, label in zip(x_values, y_values, labels):
        ax.annotate(
            _display_model_label(label),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            alpha=0.9,
        )


def plot_intrinsic_vs_downstream_tasks(
    intrinsic_metrics: Mapping[str, Mapping[str, float]],
    task_scores: Mapping[str, Mapping[str, float]],
    output_dir: Path,
    *,
    metric_key: str,
    metric_title: str,
) -> Optional[Path]:
    if not intrinsic_metrics:
        print("[plot_downstream] No intrinsic-dimension metrics found.")
        return None
    if not task_scores:
        print("[plot_downstream] No limited-label downstream task scores found.")
        return None

    tasks = sorted(task_scores.keys())
    cols = 3
    rows = int(np.ceil(len(tasks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.8 * rows), squeeze=False)

    for idx, task_name in enumerate(tasks):
        ax = axes[idx // cols, idx % cols]
        scores = task_scores[task_name]
        models = sorted(set(scores.keys()) & set(intrinsic_metrics.keys()))
        x_values = [intrinsic_metrics[model][metric_key] for model in models]
        y_values = [scores[model] for model in models]

        if x_values and y_values:
            ax.scatter(x_values, y_values, s=46, alpha=0.85)
            _annotate_scatter_points(ax, x_values, y_values, models)
        else:
            ax.text(
                0.5,
                0.5,
                "No overlapping models",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

        ax.set_title(task_name)
        ax.set_xlabel(metric_title)
        ax.set_ylabel("Raw score")
        ax.grid(True, linestyle="--", alpha=0.3)

    for idx in range(len(tasks), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(f"{metric_title} vs raw downstream task performance (limited-label regime)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = output_dir / f"intrinsic_vs_downstream_tasks_{metric_key}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_downstream] Saved {out_path}")
    return out_path


def plot_intrinsic_vs_limited_label_benchmarks(
    intrinsic_metrics: Mapping[str, Mapping[str, float]],
    benchmark_scores: Mapping[str, Mapping[str, AggregatePoint]],
    output_dir: Path,
    *,
    metric_key: str,
    metric_title: str,
) -> Optional[Path]:
    if not intrinsic_metrics:
        print("[plot_downstream] No intrinsic-dimension metrics found.")
        return None
    if not benchmark_scores:
        print("[plot_downstream] No benchmark aggregates found.")
        return None

    benchmark_order = [
        "EuroSAT (avg 1-shot and 10-shot)",
        "NeuCo-Bench (10% labels)",
        "Pangaea-Bench (10% labels)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), squeeze=False)
    for idx, benchmark_name in enumerate(benchmark_order):
        ax = axes[0, idx]
        points = benchmark_scores.get(benchmark_name, {})
        models = sorted(set(points.keys()) & set(intrinsic_metrics.keys()))
        x_values = [intrinsic_metrics[model][metric_key] for model in models]
        y_values = [points[model].mean for model in models]
        y_errors = [points[model].std for model in models]

        if x_values and y_values:
            if any(error is not None for error in y_errors):
                for x_value, y_value, y_error in zip(x_values, y_values, y_errors):
                    if y_error is not None:
                        ax.errorbar(
                            [x_value],
                            [y_value],
                            yerr=[y_error],
                            fmt="o",
                            markersize=6,
                            alpha=0.85,
                            capsize=3,
                            color="#1f77b4",
                        )
                    else:
                        ax.scatter([x_value], [y_value], s=44, alpha=0.85, color="#1f77b4")
            else:
                ax.scatter(x_values, y_values, s=44, alpha=0.85, color="#1f77b4")
            _annotate_scatter_points(ax, x_values, y_values, models)
        else:
            ax.text(
                0.5,
                0.5,
                "No overlapping models",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

        ax.set_title(benchmark_name)
        ax.set_xlabel(metric_title)
        if benchmark_name.startswith("EuroSAT"):
            ax.set_ylabel("Raw score")
        else:
            ax.set_ylabel("Task-normalized score")
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        f"{metric_title} vs performance in the limited-label regime "
        "(EuroSAT raw few-shot; NeuCo/Pangaea task-normalized)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = output_dir / f"intrinsic_vs_limited_label_benchmarks_{metric_key}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_downstream] Saved {out_path}")
    return out_path


def plot_fishers_vs_global_limited_label_average(
    intrinsic_metrics: Mapping[str, Mapping[str, float]],
    global_scores: Mapping[str, AggregatePoint],
    output_dir: Path,
) -> Optional[Path]:
    if not intrinsic_metrics:
        print("[plot_downstream] No intrinsic-dimension metrics found.")
        return None
    if not global_scores:
        print("[plot_downstream] No global limited-label task-normalized scores found.")
        return None

    models = sorted(set(global_scores.keys()) & set(intrinsic_metrics.keys()))
    valid_models = [model for model in models if "fishers" in intrinsic_metrics[model]]
    x_values = [intrinsic_metrics[model]["fishers"] for model in valid_models]
    y_values = [global_scores[model].mean for model in valid_models]
    y_errors = [global_scores[model].std for model in valid_models]
    labels = valid_models
    if not x_values or not y_values:
        print("[plot_downstream] No overlapping models for FisherS global limited-label plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for x_value, y_value, y_error in zip(x_values, y_values, y_errors):
        if y_error is not None:
            ax.errorbar(
                [x_value],
                [y_value],
                yerr=[y_error],
                fmt="o",
                markersize=6.5,
                alpha=0.88,
                capsize=3,
                color="#1f77b4",
            )
        else:
            ax.scatter([x_value], [y_value], s=54, alpha=0.88, color="#1f77b4")
    _annotate_scatter_points(ax, x_values, y_values, labels)
    ax.set_xlabel("FisherS")
    ax.set_ylabel("Mean task-normalized score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(
        "FisherS vs average task-normalized performance\n"
        "(EuroSAT kNN=1, NeuCo-Bench 10%, Pangaea-Bench 10%)"
    )
    fig.tight_layout()

    out_path = output_dir / "fishers_vs_global_limited_label_task_normalized_mean.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot_downstream] Saved {out_path}")
    return out_path


def _has_run_manifest(path: Path, root: Path) -> bool:
    for parent in (path, *path.parents):
        if (parent / "run_manifest.json").exists():
            return True
        if parent == root:
            break
    return False


def _iter_eurosat_fewshot_results(
    root: Path,
    *,
    knn_k: Optional[int],
    allowed_models: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, float, float]]:
    if not root.exists():
        return
    allowed = _normalize_model_filters(allowed_models)
    for path in root.rglob("results.json"):
        if not _has_run_manifest(path.parent, root):
            continue
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
        model_label = _format_model_label(payload)
        if not _model_allowed(model_label, allowed):
            continue
        yield model_label, k_shot, score


def plot_eurosat_fewshot(
    root: Path,
    output_dir: Path,
    *,
    knn_k: Optional[int],
    allowed_models: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    records = list(_iter_eurosat_fewshot_results(root, knn_k=knn_k, allowed_models=allowed_models))
    if not records:
        print("[plot_downstream] No EuroSAT few-shot results found.")
        return None

    data: Dict[str, List[Tuple[float, float]]] = {}
    x_values: set[float] = set()
    for model, k_shot, score in records:
        data.setdefault(model, []).append((k_shot, score))
        x_values.add(k_shot)

    fig, ax = plt.subplots(figsize=(20, 10))
    for model, points in sorted(data.items()):
        points = sorted(points, key=lambda item: item[0])
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=_display_model_label(model),
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


def _iter_neuco_fewshot_results(
    root: Path,
    *,
    allowed_models: Optional[Sequence[str]] = None,
) -> Iterable[FewshotRecord]:
    if not root.exists():
        return
    allowed = _normalize_model_filters(allowed_models)
    for path in root.rglob("task_*.json"):
        if not _has_run_manifest(path.parent, root):
            continue
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
        model_label = _format_model_label(payload)
        if not _model_allowed(model_label, allowed):
            continue
        yield FewshotRecord(model_label=model_label, task=task, x_value=x_value, score=score)


def _group_fewshot(records: Iterable[FewshotRecord]) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    grouped: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    for record in records:
        grouped.setdefault(record.task, {}).setdefault(record.model_label, []).append(
            (record.x_value, record.score)
        )
    return grouped


def plot_neuco_fewshot_tasks(
    root: Path,
    output_dir: Path,
    *,
    allowed_models: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    records = list(_iter_neuco_fewshot_results(root, allowed_models=allowed_models))
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
                label=_display_model_label(model),
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
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=2,
            frameon=False,
        )
    fig.suptitle("NeuCo-Bench with limited-labels", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.89])

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


def plot_neuco_fewshot_aggregates(
    root: Path,
    output_dir: Path,
    *,
    allowed_models: Optional[Sequence[str]] = None,
    normalization_models: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    records = list(_iter_neuco_fewshot_results(root, allowed_models=allowed_models))
    if not records:
        print("[plot_downstream] No NeuCo few-shot results found for aggregation.")
        return None

    avg_scores, x_values = _aggregate_neuco_fewshot(records)
    norm_records = list(_iter_neuco_fewshot_results(root, allowed_models=normalization_models))
    norm_scores, _ = _task_normalized_scores(norm_records)
    plot_allowed = _normalize_plot_filters(allowed_models)
    if plot_allowed is not None:
        norm_scores = {model: by_x for model, by_x in norm_scores.items() if model in plot_allowed}

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

    for model, by_x in sorted(avg_scores.items()):
        points = [(x, by_x.get(x)) for x in x_values if x in by_x]
        if not points:
            continue
        points = sorted(points, key=lambda item: item[0])
        axes[0].plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=_display_model_label(model),
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
            label=_display_model_label(model),
        )
    axes[1].set_title("Task-normalized score")
    axes[1].set_ylabel("Normalized score")
    axes[1].set_xlabel("Limited-label train fraction")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels([_format_tick(value) for value in x_values])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=2,
            frameon=False,
        )
    fig.suptitle("NeuCo-Bench - limited label aggregates", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.87])

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


def _collect_unified_eurosat_knn(
    root: Path,
    *,
    allowed_models: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[float, float]]:
    results: Dict[str, Dict[float, float]] = {}
    if not root.exists():
        return results
    allowed = _normalize_model_filters(allowed_models)
    for path in root.rglob("*metrics.json"):
        if not _has_run_manifest(path.parent, root):
            continue
        name = path.name.lower()
        if "eurosat" not in name or "knn" not in name:
            continue
        model_label = _clean_model_label(path.relative_to(root).parts[0])
        if not _model_allowed(model_label, allowed):
            continue
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


def plot_eurosat_unified_eval(
    root: Path,
    output_dir: Path,
    *,
    allowed_models: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    results = _collect_unified_eurosat_knn(root, allowed_models=allowed_models)
    if not results:
        print("[plot_downstream] No EuroSAT unified-eval kNN results found.")
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    for model, scores in sorted(results.items()):
        points = sorted(scores.items(), key=lambda item: item[0])
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=_display_model_label(model),
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


def _iter_unified_neuco_results(
    root: Path,
    *,
    allowed_models: Optional[Sequence[str]] = None,
) -> Iterable[UnifiedRecord]:
    if not root.exists():
        return
    allowed = _normalize_model_filters(allowed_models)
    for path in root.rglob("results_summary.json"):
        if not _has_run_manifest(path.parent, root):
            continue
        payload = _read_json(path)
        task_results = payload.get("task_results")
        if not isinstance(task_results, dict):
            continue
        model_label = _format_unified_model_label(path, root)
        if not _model_allowed(model_label, allowed):
            continue
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
        [_display_model_label(model), *[f"{scores.get(task, float('nan')):.4f}" for task in NEUCO_TASKS], f"{mean_val:.4f}"]
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

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.boxplot(
        normalized_scores,
        labels=[_display_model_label(model) for model in models],
        vert=True,
        patch_artist=True,
    )
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


def _iter_pangaea_fewshot_results(
    root: Path,
    *,
    label_split: Optional[str],
    allowed_models: Optional[Sequence[str]] = None,
) -> Iterable[PangaeaRecord]:
    if not root.exists():
        return
    allowed = _normalize_model_filters(allowed_models)
    for path in root.rglob("results.json"):
        payload = _read_json(path)
        if payload.get("task") != "segmentation":
            continue
        if label_split and payload.get("label_split") != label_split:
            continue
        dataset = payload.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            continue
        score = _safe_float(payload.get("mIoU"))
        if score is None:
            continue
        raw_model_label = _format_pangaea_model_label(payload)
        model_label = _canonical_model_label(raw_model_label)
        if allowed is not None and model_label not in allowed and raw_model_label not in allowed:
            continue
        yield PangaeaRecord(model_label=model_label, task=dataset, score=score)


def _pangaea_table(
    records: Iterable[PangaeaRecord],
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    table: Dict[str, Dict[str, float]] = {}
    tasks: set[str] = set()
    for record in records:
        table.setdefault(record.model_label, {})[record.task] = record.score
        tasks.add(record.task)
    return table, sorted(tasks)


def _pangaea_normalized_scores(
    table: Dict[str, Dict[str, float]],
    tasks: Sequence[str],
) -> Dict[str, List[float]]:
    normalized: Dict[str, List[float]] = {model: [] for model in table}
    for task in tasks:
        scores = {model: values.get(task) for model, values in table.items() if values.get(task) is not None}
        if not scores:
            continue
        min_score = min(scores.values())
        max_score = max(scores.values())
        denom = max_score - min_score
        for model, score in scores.items():
            normalized_score = 1.0 if denom == 0 else (score - min_score) / denom
            normalized[model].append(normalized_score)
    return normalized


def _write_pangaea_table(
    table: Dict[str, Dict[str, float]],
    tasks: Sequence[str],
    output_dir: Path,
    *,
    label_split: str,
) -> Optional[Path]:
    if not table:
        return None

    best_per_task: Dict[str, float] = {}
    for task in tasks:
        scores = [values.get(task) for values in table.values() if values.get(task) is not None]
        if scores:
            best_per_task[task] = max(scores)

    models = sorted(table.keys())
    csv_path = output_dir / f"pangaea_fewshot_{label_split.replace('%', 'pct')}_table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *tasks])
        for model in models:
            writer.writerow([model, *[table[model].get(task, "") for task in tasks]])

    md_path = output_dir / f"pangaea_fewshot_{label_split.replace('%', 'pct')}_table.md"
    lines = []
    lines.append("| Model | " + " | ".join(tasks) + " |")
    lines.append("| " + " | ".join(["---"] * (len(tasks) + 1)) + " |")
    for model in models:
        row = [model]
        for task in tasks:
            value = table[model].get(task)
            if value is None:
                row.append("")
                continue
            formatted = f"{value:.3f}"
            best = best_per_task.get(task)
            if best is not None and abs(value - best) < 1e-9:
                formatted = f"**{formatted}**"
            row.append(formatted)
        lines.append("| " + " | ".join(row) + " |")
    md_path.write_text("\n".join(lines))
    print(f"[plot_downstream] Saved {md_path} and {csv_path}")
    return md_path


def plot_pangaea_fewshot_boxplot(
    records: Iterable[PangaeaRecord],
    output_dir: Path,
    *,
    label_split: str,
) -> Optional[Path]:
    table, tasks = _pangaea_table(records)
    if not table:
        print("[plot_downstream] No Pangaea few-shot results found.")
        return None

    normalized = _pangaea_normalized_scores(table, tasks)
    models_with_norm = [model for model, scores in normalized.items() if scores]
    model_order = sorted(
        models_with_norm,
        key=lambda model: float(np.mean(normalized[model])),
        reverse=True,
    )
    if not model_order:
        model_order = sorted(table.keys())
    if not model_order:
        print("[plot_downstream] No Pangaea models available to plot.")
        return None

    data_norm = [normalized.get(model, []) for model in model_order]
    data_raw = [
        [table[model][task] for task in tasks if task in table[model]]
        for model in model_order
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].boxplot(
        data_norm,
        labels=[_display_model_label(model) for model in model_order],
        vert=True,
        patch_artist=True,
    )
    axes[0].set_ylabel("Task-normalized mIoU")
    axes[0].set_title(f"Pangaea few-shot ({label_split}) task-normalized mIoU")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[1].boxplot(
        data_raw,
        labels=[_display_model_label(model) for model in model_order],
        vert=True,
        patch_artist=True,
    )
    axes[1].set_ylabel("mIoU")
    axes[1].set_title(f"Pangaea few-shot ({label_split}) raw mIoU")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    out_path = output_dir / f"pangaea_fewshot_{label_split.replace('%', 'pct')}_boxplot.png"
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
    parser.add_argument(
        "--pangaea-fewshot-root",
        type=Path,
        default=Path("/local/ms-data/pangaea-bench/batch_runs/fewshot10/evaluation_outputs/pangaea_segmentation/10%"),
        help="Root directory for Pangaea few-shot results.json outputs.",
    )
    parser.add_argument(
        "--pangaea-label-split",
        type=str,
        default="10%",
        help="Label split to filter Pangaea few-shot results.",
    )
    parser.add_argument(
        "--intrinsic-dim-root",
        type=Path,
        default=Path("diagnostics/global_id_table1/intrinsic_dimension"),
        help="Root directory for intrinsic-dimension results.json outputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_models = MODELS_INCLUDE_CHOICES or None
    normalization_models = MODELS_TASK_NORMALIZE_CHOICES or allowed_models
    plot_eurosat_fewshot(
        args.eurosat_fewshot_root,
        output_dir,
        knn_k=args.eurosat_knn_k,
        allowed_models=allowed_models,
    )
    plot_neuco_fewshot_tasks(args.neuco_fewshot_root, output_dir, allowed_models=allowed_models)
    plot_neuco_fewshot_aggregates(
        args.neuco_fewshot_root,
        output_dir,
        allowed_models=allowed_models,
        normalization_models=normalization_models,
    )
    plot_eurosat_unified_eval(args.unified_eval_root, output_dir, allowed_models=allowed_models)

    neuco_records = list(
        _iter_unified_neuco_results(args.unified_eval_root, allowed_models=allowed_models)
    )
    table_result = _write_neuco_table(neuco_records, output_dir)
    model_order = table_result[2] if table_result else None
    plot_neuco_unified_eval_boxplot(neuco_records, output_dir, model_order=model_order)

    pangaea_records = list(
        _iter_pangaea_fewshot_results(
            args.pangaea_fewshot_root,
            label_split=args.pangaea_label_split,
            allowed_models=allowed_models,
        )
    )
    if pangaea_records:
        table, tasks = _pangaea_table(pangaea_records)
        _write_pangaea_table(table, tasks, output_dir, label_split=args.pangaea_label_split)
        plot_pangaea_fewshot_boxplot(pangaea_records, output_dir, label_split=args.pangaea_label_split)
    else:
        print("[plot_downstream] No Pangaea few-shot records to plot.")

    intrinsic_metrics = _load_intrinsic_dimension_metrics(
        args.intrinsic_dim_root,
        allowed_models=allowed_models,
    )
    task_level_scores = _collect_limited_label_task_scores(
        eurosat_root=args.eurosat_fewshot_root,
        neuco_root=args.neuco_fewshot_root,
        pangaea_root=args.pangaea_fewshot_root,
        pangaea_label_split=args.pangaea_label_split,
        eurosat_knn_k=args.eurosat_knn_k,
        allowed_models=allowed_models,
    )
    benchmark_aggregates = _collect_limited_label_benchmark_aggregates(
        eurosat_root=args.eurosat_fewshot_root,
        neuco_root=args.neuco_fewshot_root,
        pangaea_root=args.pangaea_fewshot_root,
        pangaea_label_split=args.pangaea_label_split,
        eurosat_knn_k=args.eurosat_knn_k,
        allowed_models=allowed_models,
        normalization_models=normalization_models,
    )
    global_limited_label_scores = _collect_global_task_normalized_limited_label_scores(
        eurosat_root=args.eurosat_fewshot_root,
        neuco_root=args.neuco_fewshot_root,
        pangaea_root=args.pangaea_fewshot_root,
        pangaea_label_split=args.pangaea_label_split,
        allowed_models=allowed_models,
        normalization_models=normalization_models,
    )
    plot_fishers_vs_global_limited_label_average(
        intrinsic_metrics,
        global_limited_label_scores,
        output_dir,
    )

    for metric_key, metric_title in INTRINSIC_METRICS:
        plot_intrinsic_vs_downstream_tasks(
            intrinsic_metrics,
            task_level_scores,
            output_dir,
            metric_key=metric_key,
            metric_title=metric_title,
        )
        plot_intrinsic_vs_limited_label_benchmarks(
            intrinsic_metrics,
            benchmark_aggregates,
            output_dir,
            metric_key=metric_key,
            metric_title=metric_title,
        )


if __name__ == "__main__":
    main()
