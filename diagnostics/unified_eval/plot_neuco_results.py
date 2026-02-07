#!/usr/bin/env python3
"""Summarize NeuCo unified-eval results and plot task performance."""

from __future__ import annotations

import argparse
import csv
import math
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import sys

import matplotlib.pyplot as plt
import numpy as np

FIG_DPI = 300

NEUCO_BENCH_EVAL_PATH = Path("/local/ms-data/NeuCo-Bench/benchmark")
if NEUCO_BENCH_EVAL_PATH.exists():
    sys.path.append(str(NEUCO_BENCH_EVAL_PATH))
    try:
        from evaluation.results import plot_matryoshka_task_lineplots
    except ImportError:
        plot_matryoshka_task_lineplots = None
else:
    plot_matryoshka_task_lineplots = None

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

# Map run directories to concise base labels (mirrors eurosat script).

# python /home/juro4948/ciip/diagnostics/unified_eval/plot_neuco_results.py \
#     --root /home/juro4948/ciip/diagnostics/unified_eval \
#     --output /home/juro4948/ciip/diagnostics/unified_eval/neuco_results.png \
#     --models "Matryoshka (BS=12k) (scaled)" "Alpha Earth Uniformity (BS=12k) (scaled)" \
#         "Vanilla CIIP (BS=12k) (scaled), ls<=35" "ViT CIIP (BS=12k) (scaled)" \
#             "Alpha Earth Uniformity (BS=12k) (scaled)" "rcf_13ch" "croma" \
#                 "dofa_base_s2_13ch" "scalemae_large_rgb" "moco" "dino" \
#                     "vitsmall16_s2_all_moco"   "Vanilla CIIP (BS=8k) (band-norm), ls<40"     "Vanilla CIIP (BS=8k) (scaled)" \
#     --probe-subdir linear_probe_bandwisenorm linear_probe_divideby10000 \
#     --best-epoch-only


RUN_LABELS: Dict[str, str] = {
    "2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=8k) (band-norm)",
    "2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16": "Hyperbolic CIIP (BS=8k) (band-norm), ls<40",
    "2025_12_2-text-s2": "Text-S2 alignment (BS=8k) (band-norm)",
    "12_14_2025_joint_ciip_s1s2_text": "S1-S2-Text CIIP",
    "2025_12_08-10_01_18-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16": "Hyperbolic CIIP (BS=10k) (band-norm), ls<40",
    "2025_12_19-11_23_33-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=12k) (scaled), ls<=18",
    "2025_12_22-23_48_54-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=12k) (scaled), ls<=35",
    "2025_12_28-20_16_37-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16": "Matryoshka (BS=12k) (scaled)",
    "2026_01_01-09_37_25-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16": "VCReg Loss (BS=12k) (scaled)",
    "2026_01_03-00_02_16-model_resnet50-lr_0.002-b_2-j_6-p_amp_bfloat16": "Alpha Earth Uniformity (BS=12k) (scaled)",
    "1_8_2026": "ViT CIIP (BS=12k) (scaled)",
    "2026_01_12-14_13_43-model_resnet50-lr_0.002-b_6-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=8k) (scaled)",
    "2026_01_14-18_01_06-model_resnet50-lr_0.002-b_2-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=20k) (scaled)",
    "2026_01_20-08_11_45-model_transformer-lr_0.002-b_2-j_6-p_amp_bfloat16": "Masked ViT CIIP (BS=12k, scaled)",
    '1_28-ViT-DAI': "ViT DAI CIIP (BS=12k) (scaled)",
    "2026_01_29_matryoshka_vit": "Matryoshka ViT CIIP (BS=12k) (scaled)",
    "1_28-ViT-DAI___projected": "ViT DAI CIIP PROJECTED (BS=12k) (scaled)",
    "2026_02_01_vit_ciip_dai_bandwise": "ViT DAI CIIP BANDWISE (BS=12k) (band-norm)",

}
# RUN_LABELS: Dict[str, str] = {}
MATRYOSHKA_ONLY = True

MODEL_ALIASES: Dict[str, str] = {
    "Vanilla CIIP (bs=8k, band-norm)": "Vanilla CIIP (BS=8k) (band-norm)",
    "Hyperbolic CIIP (S2A)": "Hyperbolic CIIP (BS=8k) (band-norm), ls<40",
    "CIIP (S2-text)": "Text-S2 alignment (BS=8k) (band-norm)",
    "Hyperbolic (BS=10k)": "Hyperbolic CIIP (BS=10k) (band-norm)",
    "Vanilla CIIP (ls<18, bs=12k)": "Vanilla CIIP (BS=12k) (scaled), ls<=18",
    "Vanilla CIIP (ls<35, bs=12k)": "Vanilla CIIP (BS=12k) (scaled), ls<=35",
    "Matryoshka CIIP": "Matryoshka (BS=12k) (scaled)",
    "VCReg CIIP": "VCReg Loss (BS=12k) (scaled)",
    "AE Uniformity CIIP": "Alpha Earth Uniformity (BS=12k) (scaled)",
    "ViT CIIP": "ViT CIIP (BS=12k) (scaled)",
    "Vanilla CIIP (bs=8k, scaled)": "Vanilla CIIP (BS=8k) (scaled)",
    "1_28-ViT-DAI" : "ViT DAI CIIP (BS=12k) (scaled)",
    "2026_01_29_matryoshka_vit": "Matryoshka ViT CIIP (BS=12k) (scaled)",
    "ViT DAI CIIP PROJECTED (BS=12k) (scaled)": "ViT DAI CIIP PROJECTED (BS=12k) (scaled)",
     "ViT DAI CIIP BANDWISE (BS=12k) (band-norm)": "ViT DAI CIIP BANDWISE (BS=12k) (band-norm)",
}

# python /home/juro4948/ciip/diagnostics/unified_eval/plot_neuco_results.py   \
#     --root /home/juro4948/ciip/diagnostics/unified_eval  \
#     --output /home/juro4948/ciip/diagnostics/unified_eval/neuco_results.png \
#     --models "1_28-ViT-DAI" "Vanilla CIIP (bs=8k, band-norm)" "ViT CIIP" \
#     --probe-subdir linear_probe_bandwisenorm linear_probe_divideby10000

EXCLUDED_TASKS = {"nodata", "random_cls", "random_reg"}

EUROSAT_PROBE_SUBDIRS = ["linear_probe_bandwisenorm", "linear_probe_divideby10000"]

def _normalize_model_name(model: str) -> str:
    model = model.strip().rstrip(",")
    match = re.search(r"(epoch_\d+)", model)
    epoch_token = match.group(1) if match else None
    base = re.sub(r"\(?epoch_\d+\)?", "", model).strip()
    base = re.sub(r"\s{2,}", " ", base)
    base = MODEL_ALIASES.get(base, base)
    return f"{base} {epoch_token}" if epoch_token else base


def _discover_matryoshka_runs(root: Path) -> Dict[str, str]:
    runs: Dict[str, str] = {}
    for path in root.rglob("matryoshka_dim_*"):
        parts = path.parts
        if "ciip_checkpoint" not in parts:
            continue
        idx = parts.index("ciip_checkpoint")
        if idx + 1 >= len(parts):
            continue
        run_dir = parts[idx + 1]
        runs.setdefault(run_dir, run_dir)
    return runs


def _matryoshka_tag(parts: tuple[str, ...]) -> Optional[str]:
    return next((p for p in parts if p.startswith("matryoshka_dim_")), None)


def _matryoshka_global_dim(base_label: str) -> int:
    label = base_label.lower()
    if "matryoshka vit" in label:
        return 768
    return 2048


def _format_matryoshka_label(
    epoch_token: Optional[str],
    mat_tag: Optional[str],
    base_label: Optional[str] = None,
) -> Optional[str]:
    if epoch_token is None or mat_tag is None:
        return None
    dim = mat_tag.split("_")[-1]
    prefix = base_label or "Matryoshka (BS=12k, scaled)"
    return f"{prefix} {epoch_token} ndim={dim}"

PROBE_SUBDIR_TO_NEUCO_DIRS: Dict[str, Dict[str, List[str]]] = {
    "linear_probe": {"s2": ["neuco"], "s1": ["neuco_s1"]},
    "linear_probe_s1": {"s2": [], "s1": ["neuco_s1"]},
    "linear_probe_bandwisenorm": {"s2": ["neuco_bandwisenorm"], "s1": []},
    "linear_probe_divideby10000": {"s2": ["neuco_divideby10000"], "s1": []},
}
DEFAULT_PROBE_SUBDIRS: List[str] = ["linear_probe"]
VIT_RESULTS = "auto"


def _unique_ordered(items: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _backbone_timestamp(path: Path) -> Tuple:
    """Extract sortable timestamp tuple from backbone_YYYYMMDD_HHMMSS."""
    m = re.search(r"backbone_(\d{8})_(\d{6})", path.name)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def _normalize_epoch_token(token: Optional[str]) -> Optional[str]:
    if token and token.endswith("_CLS"):
        return token[:-4]
    if token and token.endswith("_meanpool"):
        return token[:-9]
    return token


def _vit_epoch_variant(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    if token.endswith("_CLS"):
        return "CLS"
    if token.endswith("_meanpool"):
        return "meanpool"
    return None


def _resolve_vit_mode(root: Path, base_run: str, vit_results: str) -> Optional[str]:
    if "vit" not in base_run.lower():
        return None
    meanpool_exists = (root / "ciip_checkpoint" / f"{base_run}_meanpool").exists()
    if vit_results == "meanpool":
        return "meanpool"
    if vit_results == "cls":
        return "cls"
    return "meanpool" if meanpool_exists else "cls"


def _parse_fraction_key(key: str) -> Optional[float]:
    try:
        return float(str(key).replace(",", "."))
    except ValueError:
        return None


def _eurosat_preprocess_kind_from_path(path: Path) -> Optional[str]:
    parts = set(path.parts)
    if "linear_probe_bandwisenorm" in parts:
        return "bandwise"
    if "linear_probe_divideby10000" in parts:
        return "scaled"
    return None


def load_eurosat_epoch_scores(
    root: Path,
    *,
    base_model: str,
    allowed_models: Optional[Tuple[str, ...]],
    method: str,
    fraction: float = 1.0,
    probe_subdirs: Optional[List[str]] = None,
) -> Dict[str, Dict[int, float]]:
    pattern = "eurosat_backbone*_metrics.json"
    probe_subdirs = probe_subdirs or EUROSAT_PROBE_SUBDIRS
    results: Dict[str, Dict[int, float]] = {}

    for path in root.rglob(pattern):
        if not any(subdir in path.parts for subdir in probe_subdirs):
            continue
        is_knn = "knn" in path.name
        if method == "knn" and not is_knn:
            continue
        if method == "linear" and is_knn:
            continue

        preprocess_kind = _eurosat_preprocess_kind_from_path(path)
        if preprocess_kind is None:
            continue

        model_label = extract_model_label(path, allowed_models)
        if model_label is None:
            continue
        epoch = _epoch_from_label(model_label)
        if epoch is None:
            continue
        base_label = _strip_epoch_token(model_label)
        if base_label != base_model:
            continue

        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue

        value = None
        for key, metrics in raw.items():
            frac = _parse_fraction_key(key)
            if frac is None or not isinstance(metrics, dict):
                continue
            if math.isclose(frac, fraction, rel_tol=1e-6):
                value = metrics.get("test_accuracy")
                break
        if value is None:
            continue
        results.setdefault(preprocess_kind, {})[epoch] = float(value)

    return results


@lru_cache(maxsize=None)
def _allowed_epochs_for_label(base_label: str, allowed_models: Optional[Tuple[str, ...]]) -> Optional[set[str]]:
    if allowed_models is None:
        return None
    epochs: set[str] = set()
    base_label_allowed = base_label in allowed_models
    for name in allowed_models:
        if not name.startswith(base_label):
            continue
        match = re.search(r"(epoch_\d+)", name)
        if match:
            epochs.add(match.group(1))
    if epochs:
        print(f'Base label "{base_label}" allowed epochs: {epochs}')
        return epochs
    if base_label_allowed:
        print(f'Base label "{base_label}" allowed epochs: all')
        return None
    # print(f'Base label "{base_label}" allowed epochs: none')
    return set()


def extract_model_label(path: Path, allowed_models: Optional[Tuple[str, ...]]) -> Optional[str]:
    try:
        idx = path.parts.index("unified_eval")
        after_root = path.parts[idx + 1 :]
    except ValueError:
        after_root = path.parts

    if not after_root:
        return None

    model_root = after_root[0]
    mat_tag = _matryoshka_tag(after_root)
    # Special handling for ciip checkpoints (epoch + run dir)
    if model_root == "ciip_checkpoint":
        epoch_token = next((p for p in after_root if p.startswith("epoch_")), None)
        raw_epoch_token = epoch_token
        run_dir = after_root[1] if len(after_root) > 1 else None
        if epoch_token is None or run_dir is None:
            return None
        base_run = run_dir[:-9] if run_dir.endswith("_meanpool") else run_dir
        vit_mode = _resolve_vit_mode(Path("/home/juro4948/ciip/diagnostics/unified_eval"), base_run, VIT_RESULTS)
        if VIT_RESULTS != "auto" and vit_mode is not None:
            if vit_mode == "meanpool" and (raw_epoch_token is None or not raw_epoch_token.endswith("_meanpool")):
                return None
            if vit_mode == "cls" and (raw_epoch_token is None or not raw_epoch_token.endswith("_CLS")):
                return None
        epoch_token = _normalize_epoch_token(epoch_token)
        base_label = RUN_LABELS.get(base_run, base_run)

        if mat_tag is not None and mat_tag != "matryoshka_dim_2048" and allowed_models:
            allowed_set = set(allowed_models)
            base_allowed = base_label in allowed_set
            allow_all_matryoshka = any(
                name.startswith("Matryoshka")
                and "epoch_" not in name
                and "ndim=" not in name
                for name in allowed_set
            )
            if not base_allowed and not allow_all_matryoshka:
                return None

        if mat_tag is None and ("Matroyshka" in base_label or "Matryoshka" in base_label):
            mat_tag = f"matryoshka_dim_{_matryoshka_global_dim(base_label)}"
        mat_label = _format_matryoshka_label(epoch_token, mat_tag, base_label)
        if mat_label is not None:
            if allowed_models:
                allowed_set = set(allowed_models)
                base_allowed = base_label in allowed_set
                allow_all_matryoshka = any(
                    name.startswith("Matryoshka")
                    and "epoch_" not in name
                    and "ndim=" not in name
                    for name in allowed_set
                )
                if not base_allowed and not allow_all_matryoshka:
                    return None
            label = mat_label
            if VIT_RESULTS == "auto" and vit_mode is not None:
                variant = _vit_epoch_variant(raw_epoch_token)
                if variant is not None:
                    label = f"{label} [{variant}]"
            return label

        allowed_epochs = _allowed_epochs_for_label(base_label, allowed_models)
        if allowed_epochs is not None and epoch_token not in allowed_epochs:
            return None
        if "Matroyshka" in base_label or "Matryoshka" in base_label:
            label = f"{base_label} {epoch_token} ndim={_matryoshka_global_dim(base_label)}"
        else:
            label = f"{base_label} {epoch_token}"
        if VIT_RESULTS == "auto" and vit_mode is not None:
            variant = _vit_epoch_variant(raw_epoch_token)
            if variant is not None:
                label = f"{label} [{variant}]"
        return label

    label = model_root
    if allowed_models:
        allowed_set = set(allowed_models)
        if label not in allowed_set:
            return None
    if "Matroyshka" in label or "Matryoshka" in label:
        if "ndim=" not in label:
            match = re.search(r"(epoch_\d+)", label)
            if match:
                base_label = _strip_epoch_token(label)
                return f"{base_label} {match.group(1)} ndim={_matryoshka_global_dim(base_label)}"
    return label


def _sorted_models(models: List[str], preferred_order: Optional[Tuple[str, ...]]) -> List[str]:
    if preferred_order:
        ordered = [m for m in preferred_order if m in models]
        remaining = sorted(set(models) - set(ordered))
        return ordered + remaining
    return sorted(models)


def _rankdata_min_desc(values: List[float]) -> List[int]:
    unique_vals = sorted(set(values), reverse=True)
    rank_map: Dict[float, int] = {}
    seen = 0
    for val in unique_vals:
        rank_map[val] = seen + 1
        seen += values.count(val)
    return [rank_map[v] for v in values]


def compute_weighted_scores(
    task_order: List[str],
    model_scores: Dict[str, Dict[str, float]],
    models: Optional[List[str]] = None,
) -> Dict[str, float]:
    base_models = models or list(model_scores.keys())
    complete_models = [
        m for m in base_models
        if all(
            model_scores[m].get(t) is not None and not math.isnan(model_scores[m].get(t))
            for t in task_order
        )
    ]
    if not complete_models:
        return {m: float("nan") for m in base_models}

    scores = np.array(
        [[model_scores[m][t] for t in task_order] for m in complete_models],
        dtype=float,
    )
    if len(complete_models) > 1:
        stds = np.std(scores, axis=0, ddof=1)
        total = float(stds.sum())
        weights = (stds / total) if total > 0 else np.ones(scores.shape[1]) / scores.shape[1]
    else:
        weights = np.ones(scores.shape[1]) / scores.shape[1]

    ranks = np.zeros_like(scores)
    for idx in range(scores.shape[1]):
        col = scores[:, idx].tolist()
        ranks[:, idx] = np.array(_rankdata_min_desc(col), dtype=float)

    weighted = (ranks * weights).sum(axis=1)
    out = {m: float("nan") for m in base_models}
    for model, val in zip(complete_models, weighted):
        out[model] = float(val)
    return out


def select_best_epoch_per_model(
    task_order: List[str],
    model_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    weighted_scores = compute_weighted_scores(task_order, model_scores, list(model_scores.keys()))
    chosen: Dict[str, str] = {}
    for model in model_scores.keys():
        base = re.sub(r"\s*epoch_\d+", "", model).strip()
        base = re.sub(r"\s{2,}", " ", base)
        current = chosen.get(base)
        if current is None:
            chosen[base] = model
            continue
        cur_weighted = weighted_scores.get(current, float("nan"))
        new_weighted = weighted_scores.get(model, float("nan"))
        cur_overall = model_scores[current].get("overall_score", float("-inf"))
        new_overall = model_scores[model].get("overall_score", float("-inf"))

        cur_weighted_ok = not math.isnan(cur_weighted)
        new_weighted_ok = not math.isnan(new_weighted)
        if new_weighted_ok and not cur_weighted_ok:
            chosen[base] = model
        elif new_weighted_ok and cur_weighted_ok:
            if new_weighted < cur_weighted or (
                math.isclose(new_weighted, cur_weighted) and new_overall > cur_overall
            ):
                chosen[base] = model
        elif not new_weighted_ok and not cur_weighted_ok:
            if new_overall > cur_overall:
                chosen[base] = model

    return {model: model_scores[model] for model in chosen.values()}


def select_best_variant_per_base_model_by_overall(
    model_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    chosen: Dict[str, str] = {}
    for model, scores in model_scores.items():
        base_with_modality, _ = _split_model_preprocess_label(model)
        base = _strip_epoch_token(base_with_modality)
        current = chosen.get(base)
        overall = scores.get("overall_score", float("-inf"))
        if overall is None or math.isnan(overall):
            continue
        if current is None:
            chosen[base] = model
            continue
        cur_overall = model_scores[current].get("overall_score", float("-inf"))
        if overall > cur_overall:
            chosen[base] = model
    return {model: model_scores[model] for model in chosen.values()}


def _matryoshka_group_key(model: str) -> Optional[Tuple[str, str, str, str]]:
    if "Matryoshka" not in model:
        return None
    dim_match = re.search(r"ndim=(\d+)", model)
    if not dim_match:
        return None
    dim = dim_match.group(1)
    base_label = _strip_epoch_token(_split_model_preprocess_label(model)[0])
    modality = "unknown"
    preprocess = "unknown"
    match = re.search(r"\((S[12]),\s*([^)]+)\)\s*$", model)
    if match:
        modality = match.group(1)
        suffix = match.group(2).lower()
        if "bandwise" in suffix or "bandwisenorm" in suffix:
            preprocess = "bandwise"
        elif "divideby10000" in suffix or "1/10000" in suffix:
            preprocess = "scaled"
    return (base_label, dim, modality, preprocess)


def select_best_epoch_per_matryoshka_dim(
    task_order: List[str],
    model_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    groups: Dict[Tuple[str, str, str], List[str]] = {}
    for model in model_scores.keys():
        key = _matryoshka_group_key(model)
        if key is None:
            continue
        groups.setdefault(key, []).append(model)

    keep: set[str] = set()
    for key, models in groups.items():
        weighted_scores = compute_weighted_scores(task_order, model_scores, models)
        best = None
        for model in models:
            if best is None:
                best = model
                continue
            cur_weighted = weighted_scores.get(best, float("nan"))
            new_weighted = weighted_scores.get(model, float("nan"))
            cur_overall = model_scores[best].get("overall_score", float("-inf"))
            new_overall = model_scores[model].get("overall_score", float("-inf"))

            cur_weighted_ok = not math.isnan(cur_weighted)
            new_weighted_ok = not math.isnan(new_weighted)
            if new_weighted_ok and not cur_weighted_ok:
                best = model
            elif new_weighted_ok and cur_weighted_ok:
                if new_weighted < cur_weighted or (
                    math.isclose(new_weighted, cur_weighted) and new_overall > cur_overall
                ):
                    best = model
            elif not new_weighted_ok and not cur_weighted_ok:
                if new_overall > cur_overall:
                    best = model
        if best is not None:
            keep.add(best)

    out: Dict[str, Dict[str, float]] = {}
    for model, scores in model_scores.items():
        if "Matryoshka" in model:
            if model in keep:
                out[model] = scores
        else:
            out[model] = scores
    return out


def find_latest_results(
    root: Path,
    allowed_models: Optional[Tuple[str, ...]],
    modality: str = "s2",
    neuco_subdirs: Optional[List[str]] = None,
    select_best_preprocess: bool = True,
    matryoshka_only: bool = False,
    epoch_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Path]:
    """Return the best results_summary.json per model label for a modality."""
    if modality == "s1":
        default_subdirs = ["neuco_s1"]
    else:
        default_subdirs = ["neuco"]
    selected_subdirs = _unique_ordered(neuco_subdirs or [])
    if not selected_subdirs:
        selected_subdirs = default_subdirs

    patterns = [f"{subdir}/testing/backbone_*/results_summary.json" for subdir in selected_subdirs]
    patterns.extend(
        f"{subdir}/matryoshka_dim_*/testing/backbone_*/results_summary.json"
        for subdir in selected_subdirs
    )
    patterns.extend(
        f"matryoshka_dim_*/{subdir}/testing/backbone_*/results_summary.json"
        for subdir in selected_subdirs
    )

    preprocess_label_map = {
        "neuco": "1/10000",
        "neuco_divideby10000": "1/10000",
        "neuco_bandwisenorm": "neuco_bandwisenorm",
        "neuco-ssl4eonormalize": "ssl4eonormalize",
        "neuco_s1": "neuco_s1",
    }

    def preprocess_label_from_path(path: Path) -> Optional[str]:
        parts = {p.lower() for p in path.parts}
        for key, label in preprocess_label_map.items():
            if key in parts:
                return label
        return None

    def overall_score(p: Path) -> float:
        data = json.loads(p.read_text())
        return float(data.get("overall_score", float("-inf")))


    results: Dict[str, Path] = {}
    for pattern in patterns:
        # Collect latest-by-timestamp per model within this directory
        latest_in_dir: Dict[str, Path] = {}
        for path in root.rglob(pattern):
            if matryoshka_only and not any(p.startswith("matryoshka_dim_") for p in path.parts):
                continue
            preprocess_label = preprocess_label_from_path(path)
            if preprocess_label is None:
                continue
            model_label = extract_model_label(path, allowed_models)
            if model_label is None:
                continue
            if epoch_map:
                base_label = _strip_epoch_token(_split_model_preprocess_label(model_label)[0])
                desired = epoch_map.get(base_label)
                if desired is not None:
                    epoch = _epoch_from_label(model_label)
                    if epoch != desired:
                        continue
            cur = latest_in_dir.get(model_label)
            if cur is None or _backbone_timestamp(path.parent) > _backbone_timestamp(cur.parent):
                latest_in_dir[model_label] = path

        # Merge with global results: compare scores and update key to include dir_pattern when this pattern wins
        for model_label, path in latest_in_dir.items():
            preprocess_label = preprocess_label_from_path(path)
            if preprocess_label is None:
                continue
            candidate_key = f"{model_label} ({modality.upper()}, {preprocess_label})"
            if not select_best_preprocess:
                results[candidate_key] = path
                continue

            existing_key = next(
                (k for k in results.keys()
                 if k.startswith(f"{model_label} ({modality.upper()}, ")),
                None
            )

            def score(p: Path) -> float:
                try:
                    data = json.loads(p.read_text())
                    return float(data.get("overall_score", float("-inf")))
                except Exception:
                    return float("-inf")

            if existing_key is None:
                results[candidate_key] = path
            else:
                cur_path = results[existing_key]
                new_score = score(path)
                cur_score = score(cur_path)
                wins = (new_score > cur_score) or (
                    math.isclose(new_score, cur_score)
                    and _backbone_timestamp(path.parent) > _backbone_timestamp(cur_path.parent)
                )
                if wins:
                    del results[existing_key]
                    results[candidate_key] = path


    # print(results)
    print('---')
    return results




def load_task_scores(path: Path, metric_field: str = "raw_score") -> Dict[str, Tuple[float, str]]:
    data = json.loads(path.read_text())
    task_results = data.get("task_results", {})
    out: Dict[str, Tuple[float, str]] = {}

    def _extract_raw_score(value):
        if isinstance(value, dict):
            return value.get("raw_score")
        return value

    raw_scores = [
        score
        for key, value in task_results.items()
        for score in (_extract_raw_score(value),)
        if key not in EXCLUDED_TASKS and score is not None
    ]
    overall_score = float(np.mean(raw_scores)) if raw_scores else float("nan")
    out["overall_score"] = (overall_score, "overall")
    for task, payload in task_results.items():
        if task in EXCLUDED_TASKS:
            continue

        if isinstance(payload, dict):
            raw = payload.get(metric_field)
            metric = payload.get("metric_name", "")
            if raw is not None:
                out[task] = (float(raw), metric)
        else:
            try:
                out[task] = (float(payload), "")
            except Exception:
                continue
    return out


def build_table(
    latest: Dict[str, Path],
    metric_field: str = "raw_score",
) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, str]]:
    tasks: Dict[str, str] = {}
    model_scores: Dict[str, Dict[str, float]] = {}
    print(latest)
    for model, path in latest.items():
        scores = load_task_scores(path, metric_field=metric_field)
        if scores is None:
            continue
        model_scores[model] = {}
        for task, (val, metric) in scores.items():
            tasks[task] = metric_field if metric_field != "raw_score" else metric
            model_scores[model][task] = val
    tasks.pop("overall_score", None)
    task_order = sorted(tasks.keys())
    return task_order, model_scores, tasks


def _format_score(value: Optional[float], precision: int = 3) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{value:.{precision}f}"


def print_table(
    task_order: List[str],
    model_scores: Dict[str, Dict[str, float]],
    task_metrics: Dict[str, str],
    include_overall_score: bool = False,
) -> None:
    headers = ["Model"]
    if include_overall_score:
        headers.append("Overall Score")
    headers += [f"{t} ({task_metrics[t]})" for t in task_order]
    rows = [headers]
    for model in sorted(model_scores.keys()):
        row = [model]
        if include_overall_score:
            row.append(_format_score(model_scores[model].get("overall_score")))
        for task in task_order:
            val = model_scores[model].get(task)
            row.append(_format_score(val))
        rows.append(row)
    col_widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    for row in rows:
        print(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))


def render_table_png(
    task_order: List[str],
    model_scores: Dict[str, Dict[str, float]],
    task_metrics: Dict[str, str],
    output_path: Path,
    model_order: Optional[Tuple[str, ...]],
    include_overall_score: bool = False,
) -> None:
    # Base ordering (used as tie-breaker)
    base_models = _sorted_models(list(model_scores.keys()), model_order)
    weighted_scores = compute_weighted_scores(task_order, model_scores, base_models)
    base_order = {model: idx for idx, model in enumerate(base_models)}

    if include_overall_score:
        # Sort by overall score (descending), tie-break by base_models order.
        def overall_val(model: str) -> float:
            v = model_scores[model].get("overall_score")
            return v if v is not None and not math.isnan(v) else float("-inf")

        models = sorted(
            base_models,
            key=lambda m: (-overall_val(m), base_order[m]),
        )
    else:
        # Sort by weighted score (ascending), tie-break by base_models order.
        def weighted_val(model: str) -> float:
            v = weighted_scores.get(model)
            return v if v is not None and not math.isnan(v) else float("inf")

        models = sorted(
            base_models,
            key=lambda m: (weighted_val(m), base_order[m]),
        )

    def parse_preprocess(model: str) -> Tuple[str, str]:
        match = re.search(r"\(([^)]*)\)\s*$", model)
        if not match:
            return model, "bandwise"
        suffix = match.group(1)
        cleaned_model = model[: match.start()].strip()
        cleaned_model = re.sub(r"(\s*\[meanpool\]){2,}", " [meanpool]", cleaned_model, flags=re.IGNORECASE)
        cleaned_model = re.sub(r"(\s*\[cls\]){2,}", " [CLS]", cleaned_model, flags=re.IGNORECASE)
        cleaned_model = re.sub(r"(meanpool\s*){2,}", "meanpool ", cleaned_model, flags=re.IGNORECASE)
        cleaned_model = re.sub(r"(cls\s*){2,}", "CLS ", cleaned_model, flags=re.IGNORECASE)
        cleaned_model = re.sub(r"\s{2,}", " ", cleaned_model).strip()
        preprocess = "bandwise"
        if "bandwise" in suffix or "bandwisenorm" in suffix:
            preprocess = "bandwise"
        elif "divideby10000" in suffix or "1/10000" in suffix:
            preprocess = "scaled"
        return cleaned_model, preprocess

    # Headers: Model, Overall (optional), Preprocess, Weighted, then per-task columns
    headers = ["Model"]
    if include_overall_score:
        headers.append("Overall Score")
    headers += ["Downstream preproc", "Weighted Score (1/rank)"] + [
        f"{t} ({task_metrics[t]})" for t in task_order
    ]

    rows: List[List[str]] = []
    for model in models:
        model_name, preprocess = parse_preprocess(model)
        row = [model_name]
        if include_overall_score:
            row.append(_format_score(model_scores[model].get("overall_score")))
        row.append(preprocess)
        weighted_val = weighted_scores.get(model)
        if weighted_val is not None and not math.isnan(weighted_val) and weighted_val > 0:
            weighted_display = 1.0 / weighted_val
            row.append(f"{weighted_display:.3f}")
        else:
            row.append("N/A")
        for task in task_order:
            val = model_scores[model].get(task)
            row.append(f"{val:.3f}" if val is not None else "N/A")
        rows.append(row)

    # Figure / table layout
    num_task_cols = len(task_order)
    fig_width = max(14, 5 + 0.7 * num_task_cols) + 10.0
    fig, ax = plt.subplots(figsize=(fig_width, 0.5 * len(rows) + 0.6))
    ax.axis("off")
    fig.suptitle("NeuCo task scores", fontsize=14, y=0.93)

    # Column widths: Model, Overall (optional), Preprocess, Weighted, then tasks
    model_col_width = 0.40 if include_overall_score else 0.46
    overall_col_width = 0.12
    preprocess_col_width = 0.14
    weighted_col_width = 0.12
    other_col_width = 0.11
    heat_land_col_width = 0.16
    crops_col_width = 0.08
    task_col_widths: List[float] = []
    for task in task_order:
        task_lower = task.lower()
        if "heatisland" in task_lower or "landcover" in task_lower:
            task_col_widths.append(heat_land_col_width)
        elif "crops" in task_lower:
            task_col_widths.append(crops_col_width)
        else:
            task_col_widths.append(other_col_width)
    col_widths = [model_col_width]
    if include_overall_score:
        col_widths.append(overall_col_width)
    col_widths += [preprocess_col_width, weighted_col_width] + task_col_widths

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.15)

    # Style header row
    for col in range(len(headers)):
        header_cell = table[(0, col)]
        header_cell._text.set_fontweight("bold")
        header_cell.set_facecolor("#E6E6E6")
        header_cell.set_linewidth(2)

    # Bold best value per task column only (skip Model/Overall/Preproc/Weighted columns).
    start_task_idx = 4 if include_overall_score else 3
    for col_idx in range(start_task_idx, len(headers)):
        vals = []
        for row_idx in range(1, len(rows) + 1):
            text = table[(row_idx, col_idx)].get_text().get_text()
            try:
                vals.append(float(text))
            except ValueError:
                continue
        if not vals:
            continue
        best = max(vals)
        for row_idx in range(1, len(rows) + 1):
            text = table[(row_idx, col_idx)].get_text().get_text()
            try:
                val = float(text)
            except ValueError:
                continue
            if abs(val - best) < 1e-9:
                table[(row_idx, col_idx)]._text.set_fontweight("bold")

    fig.subplots_adjust(top=0.9, bottom=0.02)
    fig.tight_layout(pad=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)


def plot_lines(task_order: List[str], model_scores: Dict[str, Dict[str, float]], output: Path) -> None:
    x = np.arange(len(task_order))
    fig, ax = plt.subplots(figsize=(max(8, len(task_order) * 0.8), 6))
    for model in sorted(model_scores.keys()):
        y = [model_scores[model].get(t, np.nan) for t in task_order]
        ax.plot(x, y, marker="o", label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(task_order, rotation=40, ha="right")
    ax.set_ylabel("Raw score")
    ax.set_title("NeuCo task performance")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=FIG_DPI)
    plt.close(fig)


def _split_model_preprocess_label(label: str) -> Tuple[str, Optional[str]]:
    match = re.search(r"\((S[12]),\s*([^)]+)\)\s*$", label)
    if not match:
        return label.strip(), None
    return label[: match.start()].strip(), match.group(2).strip().lower()


def _strip_epoch_token(label: str) -> str:
    base = re.sub(r"\s*epoch_\d+(?:_meanpool|_CLS)?", "", label).strip()
    base = re.sub(r"\s*ndim=\d+", "", base).strip()
    base = re.sub(r"\s\[(CLS|meanpool)\]", "", base).strip()
    return re.sub(r"\s{2,}", " ", base)


def _matryoshka_preprocess_from_label(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    lowered = raw.lower()
    if lowered in {"neuco_bandwisenorm", "bandwisenorm"}:
        return "bandwise"
    if lowered in {"neuco_divideby10000", "divideby10000"}:
        return "scaled"
    if "bandwise" in lowered or "bandwisenorm" in lowered:
        return "bandwise"
    if "divideby10000" in lowered or "1/10000" in lowered or "scaled" in lowered:
        return "scaled"
    return lowered


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "matryoshka"


def _probe_subdirs_to_preprocesses(probe_subdirs: Optional[List[str]]) -> List[str]:
    if not probe_subdirs:
        return []
    mapping = {
        "linear_probe_bandwisenorm": "bandwise",
        "linear_probe_divideby10000": "scaled",
    }
    ordered: List[str] = []
    for subdir in probe_subdirs:
        mapped = mapping.get(subdir)
        if mapped and mapped not in ordered:
            ordered.append(mapped)
    return ordered


def plot_matryoshka_task_lineplots(
    csv_path: Path,
    output_dir: Path,
    probe_subdirs: Optional[List[str]] = None,
):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        headers = next(reader, None)
        if not headers or len(headers) < 3:
            return None
        task_headers = headers[2:]
        task_names = [h.split(" (")[0] for h in task_headers]

        rows = []
        for row in reader:
            if not row or len(row) < 2:
                continue
            if len(row) > len(headers):
                extra = len(row) - len(headers)
                model_field = ",".join(row[: 1 + extra])
                row = [model_field] + row[1 + extra :]
            model_label = row[0].strip()
            if "matryoshka" not in model_label.lower() or "ndim=" not in model_label:
                continue
            dim_match = re.search(r"ndim=(\d+)", model_label)
            if not dim_match:
                continue
            dim = int(dim_match.group(1))
            base_label_raw, preprocess_label = _split_model_preprocess_label(model_label)
            base_label = _strip_epoch_token(base_label_raw)
            preprocess = _matryoshka_preprocess_from_label(preprocess_label)
            if preprocess == "unknown":
                desired_preprocesses = _probe_subdirs_to_preprocesses(probe_subdirs)
                if len(desired_preprocesses) == 1:
                    preprocess = desired_preprocesses[0]
            overall = row[1].strip()
            try:
                overall_val = float(overall)
            except Exception:
                overall_val = float("nan")
            task_vals: Dict[str, float] = {}
            for idx, task in enumerate(task_names, start=2):
                if idx >= len(row):
                    continue
                val = row[idx].strip()
                if not val:
                    continue
                try:
                    task_vals[task] = float(val)
                except Exception:
                    continue
            rows.append((base_label, preprocess, dim, overall_val, task_vals))

    if not rows:
        return None

    grouped: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    overall_map: Dict[str, Dict[str, Dict[int, float]]] = {}
    for base_label, preprocess, dim, overall_val, task_vals in rows:
        grouped.setdefault(base_label, {}).setdefault(preprocess, {})[dim] = task_vals
        overall_map.setdefault(base_label, {}).setdefault(preprocess, {})[dim] = overall_val

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    preprocess_order = ("bandwise", "scaled")
    desired_preprocesses = _probe_subdirs_to_preprocesses(probe_subdirs)

    for base_label, preprocess_data in grouped.items():
        preprocesses = [p for p in preprocess_order if p in preprocess_data]
        for p in preprocess_data.keys():
            if p not in preprocesses:
                preprocesses.append(p)
        if desired_preprocesses:
            filtered = [p for p in preprocesses if p in desired_preprocesses]
            if filtered:
                preprocesses = filtered
            else:
                preprocesses = list(preprocess_data.keys())
        if not preprocesses:
            continue

        # If multiple preprocesses are requested, emit one figure per preprocess.
        requested = desired_preprocesses or preprocesses
        for preprocess in requested:
            if preprocess not in preprocess_data:
                fig, axes = plt.subplots(1, 2, figsize=(12, 7), squeeze=False)
                fig.suptitle(base_label, fontsize=14)
                for ax in axes[0]:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                suffix = f"_{preprocess}" if desired_preprocesses else ""
                out_path = output_dir / f"matryoshka_task_lineplot_{_slugify(base_label)}{suffix}.png"
                fig.savefig(out_path, dpi=FIG_DPI)
                plt.close(fig)
                outputs.append(out_path)
                continue

            dims = sorted(preprocess_data[preprocess].keys())

            fig, axes = plt.subplots(1, 2, figsize=(12, 7), squeeze=False)
            fig.suptitle(base_label, fontsize=14)

            ax_overall = axes[0][0]
            for dim in dims:
                y_vals = [preprocess_data[preprocess][dim].get(task, float("nan")) for task in task_names]
                ax_overall.plot(task_names, y_vals, marker="o", label=f"dim={dim}")
            ax_overall.set_title(f"Downstream preprocess: {preprocess} (lines=dim)")
            ax_overall.set_xlabel("Task")
            ax_overall.set_ylabel("Task score")
            ax_overall.tick_params(axis="x", rotation=40)
            ax_overall.grid(alpha=0.3)
            ax_overall.legend(ncol=2, fontsize=8)

            ax_tasks = axes[0][1]
            for task in task_names:
                y_vals = [preprocess_data[preprocess][dim].get(task, float("nan")) for dim in dims]
                ax_tasks.plot(dims, y_vals, marker="o", label=task)
            ax_tasks.set_title(f"Downstream preprocess: {preprocess} (lines=task)")
            ax_tasks.set_xlabel("Matryoshka dim")
            ax_tasks.set_ylabel("Task score")
            ax_tasks.grid(alpha=0.3)
            ax_tasks.legend(ncol=2, fontsize=8)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            suffix = f"_{preprocess}" if desired_preprocesses else ""
            out_path = output_dir / f"matryoshka_task_lineplot_{_slugify(base_label)}{suffix}.png"
            fig.savefig(out_path, dpi=FIG_DPI)
            plt.close(fig)
            outputs.append(out_path)

    return outputs


def _epoch_from_label(label: str) -> Optional[int]:
    match = re.search(r"epoch_(\d+)", label)
    return int(match.group(1)) if match else None


def _preprocess_kind_from_path(path: Path) -> Optional[str]:
    parts = {p.lower() for p in path.parts}
    if "neuco_divideby10000" in parts:
        return "scaled"
    if "neuco_bandwisenorm" in parts:
        return "bandwise"
    return None


def collect_epoch_overall_scores(
    root: Path,
    allowed_models: Optional[Tuple[str, ...]],
    modality: str,
    preprocess_kinds: Tuple[str, ...] = ("scaled", "bandwise"),
) -> Dict[str, Dict[str, Dict[int, float]]]:
    subdirs = []
    if "scaled" in preprocess_kinds:
        subdirs.append("neuco_divideby10000")
    if "bandwise" in preprocess_kinds:
        subdirs.append("neuco_bandwisenorm")

    patterns = [f"{subdir}/testing/backbone_*/results_summary.json" for subdir in subdirs]
    patterns.extend(f"{subdir}/matryoshka_dim_*/testing/backbone_*/results_summary.json" for subdir in subdirs)
    patterns.extend(f"matryoshka_dim_*/{subdir}/testing/backbone_*/results_summary.json" for subdir in subdirs)

    results: Dict[str, Dict[str, Dict[int, float]]] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            if modality == "s1" and "neuco_s1" not in path.parts:
                continue
            if modality == "s2" and "neuco_s1" in path.parts:
                continue
            preprocess_kind = _preprocess_kind_from_path(path)
            if preprocess_kind is None:
                continue

            model_label = extract_model_label(path, allowed_models)
            if model_label is None:
                continue
            epoch = _epoch_from_label(model_label)
            if epoch is None:
                continue
            base_label = _strip_epoch_token(model_label)

            scores = load_task_scores(path, metric_field="raw_score")
            if not scores:
                continue
            overall_score = scores.get("overall_score", (float("nan"), ""))[0]
            if overall_score is None or math.isnan(overall_score):
                continue
            results.setdefault(base_label, {}).setdefault(preprocess_kind, {})[epoch] = float(overall_score)
    return results


def _select_best_preprocess_per_epoch(
    preprocess_data: Dict[str, Dict[int, float]],
) -> Dict[int, float]:
    best: Dict[int, float] = {}
    for epoch_scores in preprocess_data.values():
        for epoch, score in epoch_scores.items():
            current = best.get(epoch)
            if current is None or score > current:
                best[epoch] = score
    return best


def plot_overall_score_epochs(
    epoch_overall_scores: Dict[str, Dict[str, Dict[int, float]]],
    selected_models: List[str],
    output_dir: Path,
    *,
    best_preprocess_only: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_colors = {"scaled": "#1f77b4", "bandwise": "#ff7f0e"}
    base_models = sorted({_strip_epoch_token(_split_model_preprocess_label(m)[0]) for m in selected_models})

    def matryoshka_dim(label: str) -> Optional[int]:
        match = re.search(r"ndim=(\d+)", label)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    matryoshka_items: List[Tuple[int, str, Dict[str, Dict[int, float]]]] = []

    allowed_models = tuple(selected_models) if selected_models else None
    allowed_models_eurosat = tuple(base_models)
    for base_model in base_models:
        preprocess_data = epoch_overall_scores.get(base_model)
        if not preprocess_data:
            continue
        if "Matryoshka" in base_model:
            dim = matryoshka_dim(base_model)
            if dim is not None:
                matryoshka_items.append((dim, base_model, preprocess_data))
                continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax = axes[0]
        if best_preprocess_only:
            epoch_scores = _select_best_preprocess_per_epoch(preprocess_data)
            if epoch_scores:
                epochs_sorted = sorted(epoch_scores.keys())
                scores = [epoch_scores[e] for e in epochs_sorted]
                ax.plot(
                    epochs_sorted,
                    scores,
                    marker="o",
                    linewidth=2,
                    color="#2ca02c",
                    label="best_preprocess",
                )
        else:
            for preprocess, epoch_scores in preprocess_data.items():
                if not epoch_scores:
                    continue
                epochs_sorted = sorted(epoch_scores.keys())
                scores = [epoch_scores[e] for e in epochs_sorted]
                color = preprocess_colors.get(preprocess, None)
                ax.plot(
                    epochs_sorted,
                    scores,
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=preprocess,
                )

        ax.set_title(f"{base_model} overall score")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Overall Score")
        ax.grid(alpha=0.3)
        ax.legend()

        knn_scores = load_eurosat_epoch_scores(
            root=output_dir.parent,
            base_model=base_model,
            allowed_models=allowed_models_eurosat,
            method="knn",
            fraction=1.0,
        )
        ax_knn = axes[1]
        if best_preprocess_only:
            epoch_scores = _select_best_preprocess_per_epoch(knn_scores)
            if epoch_scores:
                epochs_sorted = sorted(epoch_scores.keys())
                scores = [epoch_scores[e] for e in epochs_sorted]
                ax_knn.plot(
                    epochs_sorted,
                    scores,
                    marker="o",
                    linewidth=2,
                    color="#2ca02c",
                    label="best_preprocess",
                )
        else:
            for preprocess, epoch_scores in knn_scores.items():
                if not epoch_scores:
                    continue
                epochs_sorted = sorted(epoch_scores.keys())
                scores = [epoch_scores[e] for e in epochs_sorted]
                color = preprocess_colors.get(preprocess, None)
                ax_knn.plot(
                    epochs_sorted,
                    scores,
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=preprocess,
                )
        ax_knn.set_title(f"{base_model} EuroSAT kNN")
        ax_knn.set_xlabel("Epoch")
        ax_knn.set_ylabel("Accuracy")
        ax_knn.grid(alpha=0.3)
        ax_knn.legend()

        lin_scores = load_eurosat_epoch_scores(
            root=output_dir.parent,
            base_model=base_model,
            allowed_models=allowed_models_eurosat,
            method="linear",
            fraction=1.0,
        )
        ax_lin = axes[2]
        if best_preprocess_only:
            epoch_scores = _select_best_preprocess_per_epoch(lin_scores)
            if epoch_scores:
                epochs_sorted = sorted(epoch_scores.keys())
                scores = [epoch_scores[e] for e in epochs_sorted]
                ax_lin.plot(
                    epochs_sorted,
                    scores,
                    marker="o",
                    linewidth=2,
                    color="#2ca02c",
                    label="best_preprocess",
                )
        else:
            for preprocess, epoch_scores in lin_scores.items():
                if not epoch_scores:
                    continue
                epochs_sorted = sorted(epoch_scores.keys())
                scores = [epoch_scores[e] for e in epochs_sorted]
                color = preprocess_colors.get(preprocess, None)
                ax_lin.plot(
                    epochs_sorted,
                    scores,
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=preprocess,
                )
        ax_lin.set_title(f"{base_model} EuroSAT linear probe")
        ax_lin.set_xlabel("Epoch")
        ax_lin.set_ylabel("Accuracy")
        ax_lin.grid(alpha=0.3)
        ax_lin.legend()

        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_model).strip("_")
        fig.savefig(output_dir / f"{safe_name}_weighted_score_epochs.png", dpi=FIG_DPI)
        plt.close(fig)

    if matryoshka_items:
        matryoshka_items.sort(key=lambda item: item[0])
        n = len(matryoshka_items)
        ncols = min(3, n)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.2 * ncols, 3.4 * nrows),
            sharex=False,
            sharey=True,
        )
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        for idx, (dim, base_model, preprocess_data) in enumerate(matryoshka_items):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            if best_preprocess_only:
                epoch_scores = _select_best_preprocess_per_epoch(preprocess_data)
                if epoch_scores:
                    epochs_sorted = sorted(epoch_scores.keys())
                    scores = [epoch_scores[e] for e in epochs_sorted]
                    ax.plot(
                        epochs_sorted,
                        scores,
                        marker="o",
                        linewidth=2,
                        color="#2ca02c",
                        label="best_preprocess",
                    )
            else:
                for preprocess, epoch_scores in preprocess_data.items():
                    if not epoch_scores:
                        continue
                    epochs_sorted = sorted(epoch_scores.keys())
                    scores = [epoch_scores[e] for e in epochs_sorted]
                    color = preprocess_colors.get(preprocess, None)
                    ax.plot(
                        epochs_sorted,
                        scores,
                        marker="o",
                        linewidth=2,
                        color=color,
                        label=preprocess,
                    )
            ax.set_title(f"ndim={dim}")
            ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel("Overall Score")
            ax.grid(alpha=0.3)
            ax.legend()

        # Hide any unused axes
        total_axes = nrows * ncols
        for idx in range(n, total_axes):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis("off")

        fig.suptitle("Matryoshka overall score by epoch", fontsize=14, y=0.98)
        fig.tight_layout()
        fig.savefig(output_dir / "Matryoshka_overall_score_epochs.png", dpi=FIG_DPI)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NeuCo unified-eval results.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing model subfolders (default: diagnostics/unified_eval)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neuco_results.png"),
        help="Path to save the line plot.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("neuco_results.csv"),
        help="Optional path to save the table as CSV.",
    )
    parser.add_argument(
        "--table-image",
        type=Path,
        default=Path("neuco_results_table.png"),
        help="Optional path to save the table as an image.",
    )
    parser.add_argument(
        "--qstat-line-output",
        type=Path,
        default=None,
        help="Optional path to save a line plot using q_stat values.",
    )
    parser.add_argument(
        "--qstat-table-image",
        type=Path,
        default=None,
        help="Optional path to save a table image using q_stat values.",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of model names to include.",
    )
    model_group.add_argument(
        "--all-models",
        action="store_true",
        help="Include all discovered models (ignores --models).",
    )
    parser.add_argument(
        "--best-epoch-only",
        action="store_true",
        help="Only keep the best epoch per model (default is to include all epochs).",
    )
    parser.add_argument(
        "--best-preprocess-and-epoch",
        action="store_true",
        help="Keep the best variant (preprocess + epoch) per base model by overall score.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="*",
        default=["s2"],
        choices=["s1", "s2", "both"],
        help="Modalities to include: s1, s2, or both (default: s2)",
    )
    parser.add_argument(
        "--probe-subdir",
        nargs="+",
        choices=list(PROBE_SUBDIR_TO_NEUCO_DIRS.keys()),
        default=DEFAULT_PROBE_SUBDIRS,
        help="Which linear probe variants to include when reading NeuCo results.",
    )
    parser.add_argument(
        "--best-preprocess-only",
        action="store_true",
        help="Only keep the best preprocessing method per model (default is to include all).",
    )
    parser.add_argument(
        "--vit-results",
        choices=["auto", "cls", "meanpool"],
        default="auto",
        help="Which ViT unified-eval results to read (auto prefers _meanpool if present).",
    )
    parser.add_argument(
        "--epoch-map",
        type=str,
        default=None,
        help="Comma-separated model=epoch filters (e.g., '1_28-ViT-DAI=200,2026_01_29_matryoshka_vit=200').",
    )
    args = parser.parse_args()

    global VIT_RESULTS
    VIT_RESULTS = args.vit_results
    epoch_map: Dict[str, int] = {}
    if args.epoch_map:
        entries: List[str] = []
        buf: List[str] = []
        depth = 0
        for ch in args.epoch_map:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            if ch == "," and depth == 0:
                entry = "".join(buf).strip()
                if entry:
                    entries.append(entry)
                buf = []
                continue
            buf.append(ch)
        tail = "".join(buf).strip()
        if tail:
            entries.append(tail)
        for entry in entries:
            if not entry.strip():
                continue
            if "=" not in entry:
                raise ValueError(f"Invalid --epoch-map entry: {entry}")
            name, value = entry.rsplit("=", 1)
            name = _strip_epoch_token(_normalize_model_name(name.strip()))
            value = value.strip()
            value = value.strip().strip('"').strip("'")
            try:
                epoch_map[name] = int(value)
            except ValueError as exc:
                raise ValueError(f"Invalid epoch in --epoch-map: {entry}") from exc
    matryoshka_only = MATRYOSHKA_ONLY
    if args.models:
        allowed_tuple = tuple(_normalize_model_name(m) for m in args.models)
        matryoshka_only = False
    else:
        allowed_tuple = None
        RUN_LABELS.update(_discover_matryoshka_runs(args.root))

    selected_probe_subdirs: List[str] = args.probe_subdir or DEFAULT_PROBE_SUBDIRS
    s2_neuco_subdirs: List[str] = []
    s1_neuco_subdirs: List[str] = []
    for probe in selected_probe_subdirs:
        mapping = PROBE_SUBDIR_TO_NEUCO_DIRS[probe]
        s2_neuco_subdirs.extend(mapping["s2"])
        s1_neuco_subdirs.extend(mapping["s1"])
    s2_neuco_subdirs = _unique_ordered(s2_neuco_subdirs)
    s1_neuco_subdirs = _unique_ordered(s1_neuco_subdirs)
    if not s2_neuco_subdirs:
        s2_neuco_subdirs = ["neuco"]
    if not s1_neuco_subdirs:
        s1_neuco_subdirs = ["neuco_s1"]

    # Determine which modalities to process
    modalities_to_process = []
    if "both" in args.modalities:
        modalities_to_process = ["s2", "s1"]
    else:
        modalities_to_process = [m for m in args.modalities if m in ["s1", "s2"]]

    # Collect results from all requested modalities
    all_latest = {}
    for modality in modalities_to_process:
        neuco_subdirs = s2_neuco_subdirs if modality == "s2" else s1_neuco_subdirs
        latest = find_latest_results(
            args.root,
            allowed_tuple,
            modality=modality,
            neuco_subdirs=neuco_subdirs,
            select_best_preprocess=args.best_preprocess_only,
            matryoshka_only=matryoshka_only,
            epoch_map=epoch_map if epoch_map else None,
        )
        if epoch_map:
            filtered = {}
            for model_label, path in latest.items():
                base_model = _strip_epoch_token(_split_model_preprocess_label(model_label)[0])
                desired = epoch_map.get(base_model)
                if desired is None:
                    filtered[model_label] = path
                    continue
                epoch = _epoch_from_label(model_label)
                if epoch == desired:
                    filtered[model_label] = path
            latest = filtered
        all_latest.update(latest)
    
    if not all_latest:
        raise SystemExit("No NeuCo results found under root.")

    task_order, model_scores, task_metrics = build_table(all_latest, metric_field="raw_score")
    full_task_order = task_order
    full_model_scores = model_scores
    full_task_metrics = task_metrics
    if args.best_preprocess_and_epoch:
        model_scores = select_best_variant_per_base_model_by_overall(model_scores)
        selected_models = set(model_scores.keys())
    elif args.best_epoch_only:
        model_scores = select_best_epoch_per_model(task_order, model_scores)
        model_scores = select_best_epoch_per_matryoshka_dim(task_order, model_scores)
        selected_models = set(model_scores.keys())
    else:
        selected_models = set(model_scores.keys())
    print_table(task_order, model_scores, task_metrics, include_overall_score=True)
    plot_lines(task_order, model_scores, args.output)
    print(f"Saved line plot to {args.output.resolve()}")
    
    if args.table_output:
        # Save CSV with proper quoting (model names include commas)
        headers = ["Model", "Overall Score"] + [f"{t} ({task_metrics[t]})" for t in task_order]
        with args.table_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for model in sorted(model_scores.keys()):
                overall = model_scores[model].get("overall_score")
                overall_str = f"{overall:.6f}" if overall is not None and not math.isnan(overall) else ""
                row = [model, overall_str] + [
                    f"{model_scores[model].get(t):.6f}" if t in model_scores[model] else "" for t in task_order
                ]
                writer.writerow(row)
        print(f"Saved table to {args.table_output.resolve()}")
    
    if args.table_image:
        render_table_png(
            task_order,
            model_scores,
            task_metrics,
            args.table_image,
            allowed_tuple,
            include_overall_score=True,
        )
        print(f"Saved table image to {args.table_image.resolve()}")

    epoch_scores = collect_epoch_overall_scores(
        args.root,
        allowed_tuple,
        modality="s2",
        preprocess_kinds=("scaled", "bandwise"),
    )
    if epoch_scores:
        weighted_dir_name = (
            "neuco_weighted_score_epochs_best_preprocc"
            if args.best_preprocess_only
            else "neuco_weighted_score_epochs"
        )
        weighted_output_dir = args.root / weighted_dir_name
        plot_overall_score_epochs(
            epoch_scores,
            sorted(model_scores.keys()),
            weighted_output_dir,
            best_preprocess_only=args.best_preprocess_only,
        )
        print(f"Saved weighted-score epoch plots to {weighted_output_dir.resolve()}")

    if plot_matryoshka_task_lineplots and args.table_output:
        matryoshka_output_dir = Path("/home/juro4948/ciip/diagnostics/matryoshka")
        matryoshka_output_dir.mkdir(parents=True, exist_ok=True)
        matryoshka_csv = args.table_output.with_name(f"{args.table_output.stem}_matryoshka_source.csv")
        headers = ["Model", "Overall Score"] + [f"{t} ({full_task_metrics[t]})" for t in full_task_order]
        with matryoshka_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for model in sorted(full_model_scores.keys()):
                overall = full_model_scores[model].get("overall_score")
                overall_str = f"{overall:.6f}" if overall is not None and not math.isnan(overall) else ""
                row = [model, overall_str] + [
                    f"{full_model_scores[model].get(t):.6f}" if t in full_model_scores[model] else ""
                    for t in full_task_order
                ]
                writer.writerow(row)
        matryoshka_plot = plot_matryoshka_task_lineplots(
            matryoshka_csv,
            matryoshka_output_dir,
            probe_subdirs=selected_probe_subdirs,
        )
        if matryoshka_plot:
            if isinstance(matryoshka_plot, list):
                for path in matryoshka_plot:
                    print(f"Saved matryoshka line plot to {path.resolve()}")
            else:
                print(f"Saved matryoshka line plot to {matryoshka_plot.resolve()}")
        else:
            print(f"No matryoshka plots produced from {matryoshka_csv.resolve()}")

    if args.qstat_line_output or args.qstat_table_image:
        t_order_q, scores_q, metrics_q = build_table(all_latest, metric_field="q_stat")
        if args.best_epoch_only:
            scores_q = {m: scores_q[m] for m in selected_models if m in scores_q}
        if args.qstat_line_output:
            plot_lines(t_order_q, scores_q, args.qstat_line_output)
            print(f"Saved q_stat line plot to {args.qstat_line_output.resolve()}")
        if args.qstat_table_image:
            render_table_png(t_order_q, scores_q, metrics_q, args.qstat_table_image, allowed_tuple)
            print(f"Saved q_stat table image to {args.qstat_table_image.resolve()}")


if __name__ == "__main__":
    main()

# python /home/juro4948/ciip/diagnostics/unified_eval/plot_neuco_results.py       --root /home/juro4948/ciip/diagnostics/unified_eval      --output /home/juro4948/ciip/diagnostics/unified_eval/neuco_results.png     --models "1_28-ViT-DAI" "ViT CIIP (BS=12k) (scaled)" "rcf_13ch" "croma" "vitsmall16_s2_all_moco"     --probe-subdir linear_probe_bandwisenorm linear_probe_divideby10000 --vit-results auto


#   "rcf_13ch",
#     "croma",
#     "dofa_base_s2_13ch",
#     "vitsmall16_s2_all_moco",
#     "scalemae_large_rgb",
#     "resnet18_s2_all_moco",
#     "moco",  # resnet50 all moco
#     "dino",
#     "resnet18_s2_rgb_moco",
#     "resnet50_s2_rgb_moco",
#     "resnet152_imagenet_rgb",
#     "random_resnet50_s2_12",

# python  /home/juro4948/ciip/diagnostics/unified_eval/plot_neuco_results.py \
#     --root /home/juro4948/ciip/diagnostics/unified_eval \
#     --output /home/juro4948/ciip/diagnostics/unified_eval/neuco_results.png \
#     --models "Vanilla CIIP (S2A) epoch_50" "CIIP (S2-text) epoch_70" "Hyperbolic CIIP (S2A) epoch_70" "croma" "rcf_13ch"  "vitsmall16_s2_all_moco" "scalemae_large_rgb" "moco" "dino" "resnet152_imagenet_rgb" "llama3_ms_clip_base" "dofa_base_s2_13ch" "resnet18_s2_rgb_moco" "resnet50_s2_rgb_moco"  "resnet18_s2_all_moco" "random_resnet50_s2_12"

# "dofa_base_s2_13ch" "resnet18_s2_rgb_moco" "resnet50_s2_rgb_moco"  "resnet18_s2_all_moco" "random_resnet50_s2_12"
# "CIIP (S2-text) epoch_70" "Hyperbolic CIIP (S2A) epoch_70" 
