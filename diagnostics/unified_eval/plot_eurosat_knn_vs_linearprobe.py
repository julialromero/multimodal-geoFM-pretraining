import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    from imageio import v2 as iio
except ImportError:
    iio = None
import math

# Make plot text comfortably readable across all generated figures.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
})

# Root directory that holds all unified evaluation outputs.
UNIFIED_EVAL_ROOT = Path(__file__).resolve().parent
print(f"Using unified eval root: {UNIFIED_EVAL_ROOT}")
# Subdirectory that contains linear probe outputs (switch to linear_probe_s1 for S1 runs).
# LINEAR_PROBE_SUBDIR = "linear_probe"
LINEAR_PROBE_SUBDIRS: List[str] = []  # Will be populated from args
PROBE_LABELS: Dict[str, str] = {
    "linear_probe": "default",
    "linear_probe_s1": "S1",
    "linear_probe_bandwisenorm": "bandwise-norm",
    "linear_probe_divideby10000": "divideby10000",
    "linear_probe_ssl4eonorm": "ssl4eonorm",
}

DATASET = "eurosat"
DATASET_LABEL = "EuroSAT"
MATRYOSHKA_ONLY = False

# Fractions of the dataset we want to visualize in bar/table plots.
TARGET_FRACTIONS: List[float] = [0.01, 0.1, 1.0]
# Fractions used for the line plot (keep early/late splits visible).
LINE_PLOT_FRACTIONS: List[float] = [0.01, 0.1, 1.0]
# Collect at least the union so all downstream plots have the values they need.
FRACTIONS_FOR_COLLECTION: List[float] = sorted(set(TARGET_FRACTIONS) | set(LINE_PLOT_FRACTIONS))

# #### TEMPORARY SET OF MODELS #####
# PREFERRED_MODEL_ORDER: List[str] = [
#     "random rcf_13ch",
#     "rcf_13ch",
#     "croma",
#     # "dofa_base_s2_13ch",
#     "scalemae_large_rgb",
#     "moco",
#     "dino",
#     "resnet50_s2_rgb_moco",
#     "ciip_resnet50_epoch20",
#     "Vanilla CIIP (S2A) epoch_50",
#     "Hyperbolic CIIP (S2A) epoch_70"
# ]

# ALLOWED_MODELS: List[str] = [
#     "random rcf_13ch",
#     "rcf_13ch",
#     "croma",
#     # "dofa_base_s2_13ch",
#     "scalemae_large_rgb",
#     "moco",
#     "dino",
#     "resnet50_s2_rgb_moco",
#     "ciip_resnet50_epoch20",
#     "Vanilla CIIP (S2A) epoch_50",
#     "Hyperbolic CIIP (S2A) epoch_70"
# ]


# #### FULL SET OF MODELS #####
# ALLOWED_MODELS: List[str] = [
#     "rcf_13ch",
#     "croma",
#     "dofa_base_s2_13ch",
#     "vitsmall16_s2_all_moco",
#     "scalemae_large_rgb",
#     "resnet18_s2_all_moco",
#     "resnet50_s2_all_moco",
#     "moco",  # resnet50 all moco
#     "dino",
#     "resnet18_s2_rgb_moco",
#     "resnet50_s2_rgb_moco",
#     "resnet152_imagenet_rgb",
#     "random_resnet50_s2_12",
#     "ciip_resnet50_epoch20",
#     "Vanilla CIIP (S2A) epoch_50",
#     "Hyperbolic CIIP (S2A) epoch_70",
#     "CIIP (S2-text) epoch_70",
#     "llama3_ms_clip_base",
#     # "Hyperbolic (BS=10k, clamped-18) epoch_70",
#     "S1-S2-Text CIIP epoch_135",
#     "S1-S2-Text CIIP epoch_215",
#     "S1-S2-Text CIIP epoch_265"
# ]

# PREFERRED_MODEL_ORDER: List[str] = [
#     # "rcf_13ch",
#     "croma",
#     # "dofa_base_s2_13ch",
#     # "vitsmall16_s2_all_moco",
#     # "scalemae_large_rgb",
#     "resnet18_s2_all_moco",
#     "moco",  # resnet50 all moco
#     "resnet50_s2_all_moco",
#     # "dino",
#     "resnet18_s2_rgb_moco",
#     "resnet50_s2_rgb_moco",
#     "resnet152_imagenet_rgb",
#     # "random_resnet50_s2_12",
#     # "ciip_resnet50_epoch20",
#     "Vanilla CIIP (S2A) epoch_50",
#     "Hyperbolic CIIP (S2A) epoch_70",
#     "CIIP (S2-text) epoch_70",
#     "llama3_ms_clip_base"
#     # "Hyperbolic (BS=10k, clamped-18) epoch_70",
#     "S1-S2-Text CIIP epoch_70",
#     "S1-S2-Text CIIP epoch_150",
#     "S1-S2-Text CIIP epoch_200"
# ]

# # ##### JUST CIIP AND RCF #####
# PREFERRED_MODEL_ORDER: List[str] = [
#     "rcf_13ch",
#     "Vanilla CIIP (S2A) epoch_50",
#     "Hyperbolic CIIP (S2A) epoch_70",
#     "CIIP (S2-text) epoch_70",
#     "Hyperbolic (BS=10k) epoch_70"
# ]

# ALLOWED_MODELS: List[str] = [
#     "rcf_13ch",
#     "Vanilla CIIP (S2A) epoch_50",
#     "Hyperbolic CIIP (S2A) epoch_70",
#     "CIIP (S2-text) epoch_70",
#     "Hyperbolic (BS=10k) epoch_70"
# ]

# PREFERRED_MODEL_ORDER: List[str] = [
#     "Hyperbolic CIIP (S2A) epoch_10",
#     "Hyperbolic CIIP (S2A) epoch_20",
#     "Hyperbolic CIIP (S2A) epoch_30",
#     "Hyperbolic CIIP (S2A) epoch_40",
#     "Hyperbolic CIIP (S2A) epoch_50",
#     "Hyperbolic CIIP (S2A) epoch_60",
#     "Hyperbolic CIIP (S2A) epoch_70",
#     "Hyperbolic CIIP (S2A) epoch_80",
    
# ]

# ALLOWED_MODELS: List[str] = [
#     "Hyperbolic CIIP (S2A) epoch_10",
#     "Hyperbolic CIIP (S2A) epoch_20",
#     "Hyperbolic CIIP (S2A) epoch_30",
#     "Hyperbolic CIIP (S2A) epoch_40",
#     "Hyperbolic CIIP (S2A) epoch_50",
#     "Hyperbolic CIIP (S2A) epoch_60",
#     "Hyperbolic CIIP (S2A) epoch_70",
#     "Hyperbolic CIIP (S2A) epoch_80",
# ]

# PREFERRED_MODEL_ORDER: List[str] = [
#     "Vanilla CIIP (S2A) epoch_10",
#     "Vanilla CIIP (S2A) epoch_20",
#     "Vanilla CIIP (S2A) epoch_30",
#     "Vanilla CIIP (S2A) epoch_40",
#     "Vanilla CIIP (S2A) epoch_50",
#     "Vanilla CIIP (S2A) epoch_60",
#     "Vanilla CIIP (S2A) epoch_70",
#     "Vanilla CIIP (S2A) epoch_80",
#     "Vanilla CIIP (S2A) epoch_90",
#     # "Vanilla CIIP (S2A) epoch_115",
# ]
# ALLOWED_MODELS: List[str] = [
#     "Vanilla CIIP (S2A) epoch_10",
#     "Vanilla CIIP (S2A) epoch_20",
#     "Vanilla CIIP (S2A) epoch_30",
#     "Vanilla CIIP (S2A) epoch_40",
#     "Vanilla CIIP (S2A) epoch_50",
#     "Vanilla CIIP (S2A) epoch_60",
#     "Vanilla CIIP (S2A) epoch_70",
#     "Vanilla CIIP (S2A) epoch_80",
#     "Vanilla CIIP (S2A) epoch_90",
#     # "Vanilla CIIP (S2A) epoch_115",
# ]

# PREFERRED_MODEL_ORDER: List[str] = [
#     "CIIP (S2-text) epoch_5",
#     "CIIP (S2-text) epoch_10",
#     "CIIP (S2-text) epoch_20",
#     "CIIP (S2-text) epoch_30",
#     "CIIP (S2-text) epoch_40",
#     "CIIP (S2-text) epoch_50",
#     "CIIP (S2-text) epoch_60",
#     "CIIP (S2-text) epoch_70",
# ]

# ALLOWED_MODELS: List[str] = [
#     "CIIP (S2-text) epoch_5",
#     "CIIP (S2-text) epoch_10",
#     "CIIP (S2-text) epoch_20",
#     "CIIP (S2-text) epoch_30",
#     "CIIP (S2-text) epoch_40",
#     "CIIP (S2-text) epoch_50",
#     "CIIP (S2-text) epoch_60",
#     "CIIP (S2-text) epoch_70",
# ]

# "1_8_2026": "Vanilla CIIP (BS=12k) (scaled)",
#     "1_12_2026": "Vanilla CIIP (BS=8k) (scaled)",
#     "1_14_2026": "Vanilla CIIP (BS=20k) (scaled)",

PREFERRED_MODEL_ORDER: List[str] = [
    # "Vanilla CIIP (BS=12k) (scaled)",
    "Vanilla CIIP (BS=8k) (scaled)",
    "Vanilla CIIP (BS=8k) (band-norm)",
    # "Vanilla CIIP (BS=20k) (scaled)"
    # "VCReg Loss (BS=12k) (scaled)",
    # "Alpha Earth Uniformity (BS=12k) (scaled)",
    # "Vanilla CIIP (BS=12k) (scaled), ls<=18",
    # "Vanilla CIIP (BS=12k) (scaled), ls<=35",
    # # "Matroyshka v2",
    # "ViT CIIP (BS=12k) (scaled)",
]
PREFERRED_MODEL_ORDER: List[str] = []

ALLOWED_MODELS: List[str] = [
    # "Vanilla CIIP (BS=12k) (scaled)",
    "Vanilla CIIP (BS=8k) (scaled)",
    "Vanilla CIIP (BS=8k) (band-norm)"
    # "Vanilla CIIP (BS=20k) (scaled)"
    # "VCReg Loss (BS=12k) (scaled)",
    # "Alpha Earth Uniformity (BS=12k) (scaled)",
    # "Vanilla CIIP (BS=12k) (scaled), ls<=18",
    # "Vanilla CIIP (BS=12k) (scaled), ls<=35",
    # # "Matroyshka v2",
    # "ViT CIIP (BS=12k) (scaled)",
]
# ALLOWED_MODELS: List[str] = []


@lru_cache(maxsize=None)
def _allowed_epochs_for_label(base_label: str) -> Optional[set[str]]:
    """Return the set of epoch tokens permitted for a given base label.

    If the allowed models list does not specify any epochs for the label, None
    is returned so that all discovered epochs are accepted.
    """
    epochs: set[str] = set()
    for name in ALLOWED_MODELS:
        if not name.startswith(base_label):
            continue
        match = re.search(r"(epoch_\d+)", name)
        if match:
            epochs.add(match.group(1))
    return epochs or None


RUN_LABELS: Dict[str, str] = {
    "2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=8k) (band-norm)",
    "2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16": "Hyperbolic CIIP (BS=8k) (band-norm), ls<40",
    "2025_12_2-text-s2": "Text-S2 alignment (BS=8k) (band-norm)",
    "12_14_2025_joint_ciip_s1s2_text": "S1-S2-Text CIIP",
    "2025_12_19-11_23_33-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=12k) (scaled), ls<=18",
    "2025_12_22-23_48_54-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=12k) (scaled)",
    "2025_12_28-20_16_37-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16": "Matryoshka (BS=12k) (scaled)",
    "2026_01_01-09_37_25-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16": "VCReg Loss (BS=12k) (scaled)",
    "2026_01_03-00_02_16-model_resnet50-lr_0.002-b_2-j_6-p_amp_bfloat16": "Alpha Earth Uniformity (BS=12k) (scaled)",
    "1_8_2026": "ViT CIIP (BS=12k) (scaled)",
    "2026_01_12-14_13_43-model_resnet50-lr_0.002-b_6-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=8k) (scaled)",
    "2026_01_14-18_01_06-model_resnet50-lr_0.002-b_2-j_6-p_amp_bfloat16": "Vanilla CIIP (BS=20k) (scaled)",
}
GIF_FRAME_DURATION_SECONDS = 3  # increase to slow down playback

def parse_fraction_key(key: str) -> Optional[float]:
    """Convert JSON fraction keys to floats, handling comma decimals."""
    try:
        return float(key.replace(",", "."))
    except ValueError:
        return None


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


def _format_matryoshka_label(epoch_token: Optional[str], mat_tag: Optional[str]) -> Optional[str]:
    if epoch_token is None or mat_tag is None:
        return None
    dim = mat_tag.split("_")[-1]
    return f"Matryoshka {epoch_token} ndim={dim}"


def select_fraction_accuracies(path: Path) -> Dict[float, Optional[float]]:
    """Load a metrics file and pull test_accuracy for the needed fractions."""
    with path.open() as f:
        raw = json.load(f)

    fraction_map: Dict[float, float] = {}
    for key, metrics in raw.items():
        frac = parse_fraction_key(key)
        if frac is None:
            continue
        if not isinstance(metrics, dict):
            continue
        test_acc = metrics.get("test_accuracy")
        if test_acc is None:
            continue
        fraction_map[round(frac, 3)] = test_acc

    selected: Dict[float, Optional[float]] = {}
    for target in FRACTIONS_FOR_COLLECTION:
        selected[target] = fraction_map.get(round(target, 3))
    return selected


def extract_model_label(path: Path, probe_subdir: str) -> Optional[str]:
    """Create a concise model label from the path, applying ciip rules."""
    try:
        idx = path.parts.index("unified_eval")
        after_root = path.parts[idx + 1 :]
    except ValueError:
        after_root = path.parts

    if not after_root:
        return None

    model_root = after_root[0]
    mat_tag = _matryoshka_tag(after_root)
    if model_root == "moco":
        model_root = "resnet50_s2_all_moco"
    if model_root == "ciip_checkpoint":
        epoch_token = next((p for p in after_root if p.startswith("epoch_")), None)
        run_dir = after_root[1] if len(after_root) > 1 else None
        base_label = RUN_LABELS.get(run_dir) if run_dir else None
        if base_label is None and run_dir is not None:
            base_label = run_dir
        if base_label is None:
            return None
        if mat_tag is None and ("Matroyshka" in base_label or "Matryoshka" in base_label):
            mat_tag = "matryoshka_dim_2048"
        mat_label = _format_matryoshka_label(epoch_token, mat_tag)
        if mat_label is not None:
            label = mat_label
        elif epoch_token is None:
            label = base_label
        else:
            allowed_epochs = _allowed_epochs_for_label(base_label)
            if allowed_epochs is not None and epoch_token not in allowed_epochs:
                return None
            if "Matroyshka" in base_label or "Matryoshka" in base_label:
                label = f"Matryoshka {epoch_token} ndim=2048"
            else:
                label = f"{base_label} {epoch_token}"
    else:
        label = model_root
    
    # Append probe subdir label
    probe_label = PROBE_LABELS.get(probe_subdir, probe_subdir)
    return f"{label} ({probe_label})"


def parse_preprocess_label(model: str) -> tuple[str, str]:
    match = re.search(r"\(([^)]*)\)\s*$", model)
    if not match:
        return model, "unknown"
    suffix = match.group(1).strip().lower()
    cleaned_model = model[: match.start()].strip()
    preprocess = "unknown"
    if "bandwise" in suffix:
        preprocess = "bandwise"
    elif "ssl4eonorm" in suffix:
        preprocess = "ssl4eonorm"
    elif "divideby10000" in suffix or "1/10000" in suffix:
        preprocess = "scaled"
    elif suffix == "default":
        preprocess = "default"
    elif suffix == "s1":
        preprocess = "s1"
    return cleaned_model, preprocess


def collect_results() -> Dict[str, Dict[str, Dict[float, Optional[float]]]]:
    """Gather knn and linear probe accuracies, picking best of raw vs batchnorm."""

    def method_and_variant(filename: str) -> Optional[tuple[str, str]]:
        if "backbone" not in filename:
            return None
        method = "knn" if "knn" in filename else "linear"
        variant = "normalized" if "batchnorm" in filename else "raw"
        return method, variant

    # base_label -> variant -> method -> fractions
    by_model: Dict[str, Dict[str, Dict[str, Dict[float, Optional[float]]]]] = {}

    pattern = f"{DATASET}_backbone*_metrics.json"
    
    for probe_subdir in LINEAR_PROBE_SUBDIRS:
        for path in UNIFIED_EVAL_ROOT.rglob(pattern):
            if probe_subdir not in path.parts:
                continue
            if MATRYOSHKA_ONLY and not any(p.startswith("matryoshka_dim_") for p in path.parts):
                continue
            
            # print(f"Found probe path in {probe_subdir}: {path}")
            
            parsed = method_and_variant(path.name)
            if parsed is None:
                continue
            method, variant = parsed

            model_label = extract_model_label(path, probe_subdir)
            # print(f"  Model label: {model_label}")
            if model_label is None:
                continue
            
            # Check if base model (without probe label or epoch) is allowed
            base_model = model_label.rsplit(" (", 1)[0]
            if " epoch_" in base_model:
                base_model = base_model.split(" epoch_")[0]
            if ALLOWED_MODELS and base_model not in ALLOWED_MODELS:
                continue

            fraction_accs = select_fraction_accuracies(path)
            model_entry = by_model.setdefault(model_label, {})
            variant_entry = model_entry.setdefault(variant, {})
            variant_entry[method] = fraction_accs

    def variant_score(variant_data: Dict[str, Dict[float, Optional[float]]]) -> float:
        scores: List[float] = []
        for method in ("linear", "knn"):
            frac_map = variant_data.get(method, {})
            for val in frac_map.values():
                if val is not None:
                    scores.append(val)
        return sum(scores) / len(scores) if scores else float("-inf")

    results: Dict[str, Dict[str, Dict[float, Optional[float]]]] = {}
    for base_label, variants in by_model.items():
        if not variants:
            continue
        best_variant = None
        best_score = float("-inf")
        for variant_name, data in variants.items():
            score = variant_score(data)
            if score > best_score or (score == best_score and variant_name == "normalized"):
                best_score = score
                best_variant = variant_name
        if best_variant is None:
            continue
        chosen = variants[best_variant]
        # Don't add variant suffix here since probe label is already in base_label
        results[base_label] = chosen

    return results


def _rankdata_min_desc(values: List[float]) -> List[int]:
    unique_vals = sorted(set(values), reverse=True)
    rank_map: Dict[float, int] = {}
    seen = 0
    for val in unique_vals:
        rank_map[val] = seen + 1
        seen += values.count(val)
    return [rank_map[v] for v in values]


def compute_weighted_scores(
    results: Dict[str, Dict[str, Dict[float, Optional[float]]]],
    *,
    methods: List[str],
    fractions: List[float],
) -> Dict[str, float]:
    models = list(results.keys())
    metric_keys = [(method, frac) for method in methods for frac in fractions]
    complete_models = []
    for model in models:
        if all(results.get(model, {}).get(method, {}).get(frac) is not None for method, frac in metric_keys):
            complete_models.append(model)

    if not complete_models:
        return {m: float("nan") for m in models}

    scores = np.array(
        [
            [results[m][method][frac] for method, frac in metric_keys]
            for m in complete_models
        ],
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
    out = {m: float("nan") for m in models}
    for model, val in zip(complete_models, weighted):
        out[model] = float(val)
    return out


def sorted_models(
    models: List[str],
    results: Optional[Dict[str, Dict[str, Dict[float, Optional[float]]]]] = None,
    *,
    weight_methods: Optional[List[str]] = None,
    weight_fractions: Optional[List[float]] = None,
) -> List[str]:
    """Sort models by weighted score (asc), then preferred order, then alphabetically."""
    order_index = {name: idx for idx, name in enumerate(PREFERRED_MODEL_ORDER)}
    weight_methods = weight_methods or ["linear", "knn"]
    weight_fractions = weight_fractions or FRACTIONS_FOR_COLLECTION

    def base_label(model: str) -> str:
        if " epoch_" in model:
            return model.split(" epoch_")[0]
        return model.split(" (")[0]

    weights = (
        compute_weighted_scores(results, methods=weight_methods, fractions=weight_fractions)
        if results is not None
        else {}
    )

    def weighted_score(model: str) -> float:
        if results is None:
            return float("inf")
        val = weights.get(model)
        return val if val is not None and not math.isnan(val) else float("inf")

    return sorted(
        models,
        key=lambda m: (
            weighted_score(m),
            order_index.get(base_label(m), len(order_index)),
            base_label(m),
            m,
        ),
    )


def select_best_epoch_per_model(
    results: Dict[str, Dict[str, Dict[float, Optional[float]]]],
) -> Dict[str, Dict[str, Dict[float, Optional[float]]]]:
    weighted_scores = compute_weighted_scores(
        results,
        methods=["linear", "knn"],
        fractions=FRACTIONS_FOR_COLLECTION,
    )

    def base_label(model: str) -> str:
        # Preserve preprocess suffix, drop epoch token for grouping.
        prefix, suffix = model.rsplit(" (", 1) if " (" in model else (model, "")
        suffix = f" ({suffix}" if suffix else ""
        if " epoch_" in prefix:
            prefix = prefix.split(" epoch_")[0]
        return f"{prefix}{suffix}"

    best_for_base: Dict[str, str] = {}
    for model in results.keys():
        base = base_label(model)
        current = best_for_base.get(base)
        if current is None:
            best_for_base[base] = model
            continue
        cur_weighted = weighted_scores.get(current, float("nan"))
        new_weighted = weighted_scores.get(model, float("nan"))
        cur_overall = results[current]
        new_overall = results[model]

        def overall_mean(entry: Dict[str, Dict[float, Optional[float]]]) -> float:
            vals: List[float] = []
            for method in ("linear", "knn"):
                for val in entry.get(method, {}).values():
                    if val is not None:
                        vals.append(val)
            return float(sum(vals) / len(vals)) if vals else float("-inf")

        cur_mean = overall_mean(cur_overall)
        new_mean = overall_mean(new_overall)

        cur_weighted_ok = not math.isnan(cur_weighted)
        new_weighted_ok = not math.isnan(new_weighted)
        if new_weighted_ok and not cur_weighted_ok:
            best_for_base[base] = model
        elif new_weighted_ok and cur_weighted_ok:
            if new_weighted < cur_weighted or (
                math.isclose(new_weighted, cur_weighted) and new_mean > cur_mean
            ):
                best_for_base[base] = model
        elif not new_weighted_ok and not cur_weighted_ok:
            if new_mean > cur_mean:
                best_for_base[base] = model

    return {model: results[model] for model in best_for_base.values()}


def compute_model_colors(models: List[str]) -> Dict[str, str]:
    """Assign a stable color per model."""
    color_cycle = plt.cm.tab20.colors
    return {model: color_cycle[idx % len(color_cycle)] for idx, model in enumerate(models)}


def _ciip_epoch_sequence(label: str) -> List[str]:
    """Return epoch tokens for a CIIP label, preferring discovered PCA/SVD outputs."""
    # Discover epochs that have either PCA projections or SVD plots on disk.
    discovered: List[str] = []
    run_dirs = [run_dir for run_dir, base in RUN_LABELS.items() if base == label]
    for run_dir in run_dirs:
        search_root = UNIFIED_EVAL_ROOT / "ciip_checkpoint"
        for path in search_root.rglob("*pca_projected*.png"):
            if run_dir not in path.parts:
                continue
            match = next((p for p in path.parts if p.startswith("epoch_")), None)
            if match:
                discovered.append(match)
        for path in search_root.rglob("singular_values_all.png"):
            if run_dir not in path.parts:
                continue
            match = next((p for p in path.parts if p.startswith("epoch_")), None)
            if match:
                discovered.append(match)
    # If nothing was found, fall back to the configured order.
    if not discovered:
        for name in PREFERRED_MODEL_ORDER:
            if not name.startswith(label):
                continue
            match = re.search(r"(epoch_\d+)", name)
            if match:
                discovered.append(match.group(1))
    # Deduplicate and sort numerically.
    seen = set()
    ordered = []
    for ep in sorted(discovered, key=lambda x: int(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else x):
        if ep not in seen:
            seen.add(ep)
            ordered.append(ep)
    return ordered


def _find_pca_png(run_dir: str, epoch_token: str) -> Optional[Path]:
    """Locate a PCA projection image for a given run/epoch."""
    roots = [
        UNIFIED_EVAL_ROOT / "ciip_checkpoint" / epoch_token / run_dir,
        UNIFIED_EVAL_ROOT / "ciip_checkpoint" / run_dir / epoch_token,
        UNIFIED_EVAL_ROOT / "ciip_checkpoint" / run_dir,
    ]
    filenames = [
        "pca_projected_raw.png",
        "eurosat_pca_projected.png",  # seen under linear_probe outputs
    ]
    for root in roots:
        for subdir in ("embedding_diagnostics", LINEAR_PROBE_SUBDIR, ""):
            for fname in filenames:
                candidate = root / subdir / fname if subdir else root / fname
                if candidate.exists():
                    return candidate
        # As a last resort, look for any pca projection under the root.
        glob_match = next(root.rglob("*pca_projected*.png"), None) if root.exists() else None
        if glob_match is not None:
            return glob_match
    return None


def _find_svd_png(run_dir: str, epoch_token: str) -> Optional[Path]:
    """Locate a singular values plot for a given run/epoch."""
    roots = [
        UNIFIED_EVAL_ROOT / "ciip_checkpoint" / epoch_token / run_dir,
        UNIFIED_EVAL_ROOT / "ciip_checkpoint" / run_dir / epoch_token,
        UNIFIED_EVAL_ROOT / "ciip_checkpoint" / run_dir,
    ]
    for root in roots:
        candidates = [
            root / "embedding_diagnostics" / "singular_values_all.png",
            root / "singular_values_all.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Last resort: any singular values plot under the root.
        glob_match = next(root.rglob("singular_values_all.png"), None) if root.exists() else None
        if glob_match is not None:
            return glob_match
    return None


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _annotate_epoch_on_frame(frame: np.ndarray, epoch_token: str) -> np.ndarray:
    """Overlay an epoch label on the PCA frame for clarity."""
    # Prefer showing just the numeric part if available.
    match = re.search(r"\d+", epoch_token)
    epoch_label = f"Epoch {match.group(0)}" if match else epoch_token
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    padding = 6
    x0, y0 = 20, 10
    # textbbox is available on recent Pillow; returns (left, top, right, bottom).
    bbox = draw.textbbox((0, 0), epoch_label, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x1, y1 = x0 + text_w + 2 * padding, y0 + text_h + 2 * padding
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    draw.text((x0 + padding, y0 + padding), epoch_label, fill=(255, 255, 255), font=font, fontsize=50)
    return np.array(img)


def make_ciip_pca_gifs() -> List[Path]:
    """Create slow GIFs of PCA projections across epochs for CIIP runs."""
    if iio is None:
        raise RuntimeError("imageio is required to build GIFs but is not installed.")
    outputs: List[Path] = []
    for run_dir, base_label in RUN_LABELS.items():
        if "CIIP" not in base_label:
            continue
        epochs = _ciip_epoch_sequence(base_label)
        if not epochs:
            continue
        frames: List[object] = []
        found_epochs: List[str] = []
        target_size: Optional[tuple[int, int]] = None  # (width, height)
        for epoch_token in epochs:
            png_path = _find_pca_png(run_dir, epoch_token)
            if png_path is None:
                print(f"Missing PCA projection for {base_label} {epoch_token} in {run_dir}")
                continue
            frame = iio.imread(png_path)
            frame = _annotate_epoch_on_frame(frame, epoch_token)
            if target_size is None:
                target_size = (frame.shape[1], frame.shape[0])  # width, height
            else:
                if (frame.shape[1], frame.shape[0]) != target_size:
                    frame = np.array(Image.fromarray(frame).resize(target_size, Image.LANCZOS))
            frames.append(frame)
            found_epochs.append(epoch_token)
        if not frames:
            print(f"No PCA projections found for {base_label}; skipping GIF.")
            continue
        gif_path = UNIFIED_EVAL_ROOT / f"{_slugify(base_label)}_pca_projected_raw.gif"
        iio.mimsave(gif_path, frames, duration=GIF_FRAME_DURATION_SECONDS)
        print(f"Saved PCA GIF for {base_label} ({', '.join(found_epochs)}) to {gif_path}")
        outputs.append(gif_path)
    return outputs


def make_ciip_svd_gifs() -> List[Path]:
    """Create GIFs of singular value plots across epochs for CIIP runs."""
    if iio is None:
        raise RuntimeError("imageio is required to build GIFs but is not installed.")
    outputs: List[Path] = []
    for run_dir, base_label in RUN_LABELS.items():
        if "CIIP" not in base_label:
            continue
        epochs = _ciip_epoch_sequence(base_label)
        if not epochs:
            continue
        frames: List[object] = []
        found_epochs: List[str] = []
        target_size: Optional[tuple[int, int]] = None
        for epoch_token in epochs:
            png_path = _find_svd_png(run_dir, epoch_token)
            if png_path is None:
                print(f"Missing SVD plot for {base_label} {epoch_token} in {run_dir}")
                continue
            frame = iio.imread(png_path)
            frame = _annotate_epoch_on_frame(frame, epoch_token)
            if target_size is None:
                target_size = (frame.shape[1], frame.shape[0])
            else:
                if (frame.shape[1], frame.shape[0]) != target_size:
                    frame = np.array(Image.fromarray(frame).resize(target_size, Image.LANCZOS))
            frames.append(frame)
            found_epochs.append(epoch_token)
        if not frames:
            print(f"No SVD plots found for {base_label}; skipping SVD GIF.")
            continue
        gif_path = UNIFIED_EVAL_ROOT / f"{_slugify(base_label)}_singular_values.gif"
        iio.mimsave(gif_path, frames, duration=GIF_FRAME_DURATION_SECONDS)
        print(f"Saved SVD GIF for {base_label} ({', '.join(found_epochs)}) to {gif_path}")
        outputs.append(gif_path)
    return outputs


def plot_results(results: Dict[str, Dict[str, Dict[float, Optional[float]]]]) -> Path:
    """Create subplots comparing knn vs linear for each fraction."""
    if not results:
        raise RuntimeError("No results found for the requested criteria.")

    models = sorted_models(list(results.keys()), results=results)
    fig, axes = plt.subplots(1, len(TARGET_FRACTIONS), figsize=(16, 6), sharey=True)

    bar_width = 0.35
    colors = {"linear": "#1f77b4", "knn": "#ff7f0e"}

    for ax, frac in zip(axes, TARGET_FRACTIONS):
        rows = []
        for model in models:
            model_data = results[model]
            lin = model_data.get("linear", {}).get(frac)
            knn = model_data.get("knn", {}).get(frac)
            if lin is None and knn is None:
                continue
            rows.append((model, lin, knn))

        if not rows:
            ax.set_title(f"Fraction {frac}")
            ax.set_xticks([])
            ax.set_ylabel("Test Accuracy")
            continue

        x = list(range(len(rows)))
        linear_vals = [0 if r[1] is None else r[1] for r in rows]
        knn_vals = [0 if r[2] is None else r[2] for r in rows]

        ax.bar(
            [v - bar_width / 2 for v in x],
            linear_vals,
            width=bar_width,
            label="Linear probe",
            color=colors["linear"],
        )
        ax.bar(
            [v + bar_width / 2 for v in x],
            knn_vals,
            width=bar_width,
            label="k-NN",
            color=colors["knn"],
        )

        ax.set_title(f"Fraction {frac}")
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ax is axes[0]:
            ax.set_ylabel("Test Accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()

    output_path = UNIFIED_EVAL_ROOT / f"{DATASET}_knn_vs_linear_probe.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def render_table(
    method: str,
    results: Dict[str, Dict[str, Dict[float, Optional[float]]]],
    output_name: str,
) -> Path:
    """Render a LaTeX-style table for a given method and save as PNG."""
    models = sorted_models(
        list(results.keys()),
        results=results,
        weight_methods=[method],
        weight_fractions=TARGET_FRACTIONS,
    )

    headers = ["Model", "Downstream preproc", "Weighted Score"] + [str(f) for f in TARGET_FRACTIONS]
    rows: List[List[str]] = []
    best_by_frac: Dict[float, float] = {}
    weighted_scores = compute_weighted_scores(
        results,
        methods=[method],
        fractions=TARGET_FRACTIONS,
    )
    best_weighted = min(
        (v for v in weighted_scores.values() if v is not None and not math.isnan(v)),
        default=None,
    )

    for frac in TARGET_FRACTIONS:
        vals = [
            results.get(model, {}).get(method, {}).get(frac)
            for model in models
            if results.get(model, {}).get(method, {}).get(frac) is not None
        ]
        if vals:
            best_by_frac[frac] = max(vals)

    for model in models:
        model_name, preprocess = parse_preprocess_label(model)
        row = [model_name, preprocess]
        weighted_val = weighted_scores.get(model)
        row.append(f"{weighted_val:.3f}" if weighted_val is not None and not math.isnan(weighted_val) else "N/A")
        for frac in TARGET_FRACTIONS:
            val = results.get(model, {}).get(method, {}).get(frac)
            row.append(f"{val:.3f}" if val is not None else "N/A")
        rows.append(row)

    max_label_len = max((len(m) for m in models), default=10)
    model_col_width = max(0.30, min(0.52, 0.013 * max_label_len))
    preprocess_col_width = 0.22
    weighted_col_width = 0.16
    other_col_width = 0.16
    col_widths = [model_col_width, preprocess_col_width, weighted_col_width] + [other_col_width] * len(TARGET_FRACTIONS)

    fig_width = max(9, 3.5 + model_col_width * 6 + len(TARGET_FRACTIONS) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, 0.55 * len(rows) + 1.2))
    ax.axis("off")

    ax.set_title(
        f"{DATASET_LABEL} backbone: {'Linear Probe' if method == 'linear' else 'k-NN'}",
        fontsize=14,
        pad=4,
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.3)

    # Style the header row
    num_cols = len(headers)
    for col in range(num_cols):
        header_cell = table[(0, col)]  # Header row is row 0
        header_cell._text.set_fontweight("bold")
        header_cell.set_facecolor("#E6E6E6")
        header_cell._text.set_color("black")
        header_cell.set_linewidth(2)

    # Bold the best value for weighted score (lower is better)
    if best_weighted is not None:
        for row_idx, model in enumerate(models, start=1):
            val = weighted_scores.get(model)
            if val is not None and not math.isnan(val) and abs(val - best_weighted) < 1e-6:
                table[(row_idx, 2)]._text.set_fontweight("bold")

    # Bold the best value per fraction column (data rows only)
    for col_idx, frac in enumerate(TARGET_FRACTIONS, start=3):
        best = best_by_frac.get(frac)
        if best is None:
            continue
        for row_idx, model in enumerate(models, start=1):  # +1 to skip header row
            val = results.get(model, {}).get(method, {}).get(frac)
            if val is not None and abs(val - best) < 1e-6:
                table[(row_idx, col_idx)]._text.set_fontweight("bold")

    fig.tight_layout(pad=0.5)
    output_path = UNIFIED_EVAL_ROOT / output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_bump_chart(
    results: Dict[str, Dict[str, Dict[float, Optional[float]]]],
    model_colors: Dict[str, str],
) -> Path:
    """Render a bump chart ranking models across linear/kNN fractions."""
    models = sorted_models(list(results.keys()), results=results)
    conditions = [
        ("linear", 0.01, "LP 0.01"),
        ("linear", 0.1, "LP 0.1"),
        ("linear", 1.0, "LP 1.0"),
        ("knn", 0.01, "kNN 0.01"),
        ("knn", 0.1, "kNN 0.1"),
        ("knn", 1.0, "kNN 1.0"),
    ]

    # Compute ranks per condition.
    ranks: Dict[str, Dict[str, int]] = {m: {} for m in models}
    for method, frac, label in conditions:
        vals = []
        for model in models:
            val = results.get(model, {}).get(method, {}).get(frac)
            if val is not None:
                vals.append((model, val))
        vals.sort(key=lambda x: x[1], reverse=True)
        for rank, (model, _) in enumerate(vals, start=1):
            ranks[model][label] = rank

    fig, ax = plt.subplots(figsize=(13, 8))
    x_positions = list(range(len(conditions)))
    x_labels = [label for _, _, label in conditions]

    for model in models:
        y = []
        x = []
        for idx, (_, _, label) in enumerate(conditions):
            rank = ranks[model].get(label)
            if rank is None:
                continue
            x.append(idx)
            y.append(rank)
        if not x:
            continue
        color = model_colors[model]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=6,
            label=model,
            color=color,
            alpha=0.85,
        )
        # Annotate at the first and last available points to avoid misalignment.
        ax.text(
            x[0] - 0.12,
            y[0],
            model,
            ha="right",
            va="center",
            fontsize=12,
            color=color,
        )
        ax.text(
            x[-1] + 0.12,
            y[-1],
            model,
            ha="left",
            va="center",
            fontsize=12,
            color=color,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.set_ylim(len(models) + 1, 0.5)  # invert so rank 1 is at the top
    ax.set_ylabel("") #Rank (top to bottom, top is Rank 1)")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(f"Model rank (Top is best) across Linear Probe (LP) and k-NN on {DATASET_LABEL}", fontsize=14)

    fig.tight_layout()
    output_path = UNIFIED_EVAL_ROOT / f"{DATASET}_bump_chart.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_line_plots(
    results: Dict[str, Dict[str, Dict[float, Optional[float]]]],
    model_colors: Dict[str, str],
) -> Path:
    """Render per-model line plots across fractions for linear and kNN."""
    models = sorted_models(list(results.keys()), results=results)
    x_positions = list(range(len(LINE_PLOT_FRACTIONS)))
    x_labels = [str(f) for f in LINE_PLOT_FRACTIONS]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    method_specs = [
        ("linear", "Linear probe", axes[0]),
        ("knn", "k-NN", axes[1]),
    ]

    for method, title, ax in method_specs:
        for model in models:
            xs = []
            ys = []
            for idx, frac in enumerate(LINE_PLOT_FRACTIONS):
                val = results.get(model, {}).get(method, {}).get(frac)
                if val is None:
                    continue
                xs.append(idx)
                ys.append(val)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                markersize=6,
                color=model_colors[model],
                label=model,
                alpha=0.9,
            )
        ax.set_title(f"{title} test accuracy")
        ax.set_ylabel("Accuracy")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_labels)
    axes[-1].set_xlabel("Training fraction")

    # Single legend for clarity.
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    output_path = UNIFIED_EVAL_ROOT / f"{DATASET}_fraction_lineplots.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    global DATASET, DATASET_LABEL, LINEAR_PROBE_SUBDIRS, ALLOWED_MODELS, PREFERRED_MODEL_ORDER, RUN_LABELS

    parser = argparse.ArgumentParser(description="Plot kNN vs linear probe results.")
    parser.add_argument(
        "--dataset",
        choices=["eurosat", "bigearthnet"],
        default="eurosat",
        help="Which dataset results to plot.",
    )
    parser.add_argument(
        "--probe-subdir",
        nargs="+",
        choices=[
            "linear_probe",
            "linear_probe_s1",
            "linear_probe_bandwisenorm",
            "linear_probe_divideby10000",
            "linear_probe_ssl4eonorm",
        ],
        default=["linear_probe"],
        help="Which linear probe subdirectories to read metrics from (can specify multiple).",
    )
    parser.add_argument(
        "--best-epoch-only",
        action="store_true",
        help="Only keep the best epoch per model (default is to include all epochs).",
    )
    args = parser.parse_args()

    DATASET = args.dataset.lower()
    DATASET_LABEL = "EuroSAT" if DATASET == "eurosat" else "BigEarthNet"
    LINEAR_PROBE_SUBDIRS = args.probe_subdir
    if MATRYOSHKA_ONLY and LINEAR_PROBE_SUBDIRS == ["linear_probe"]:
        LINEAR_PROBE_SUBDIRS = list(PROBE_LABELS.keys())
    RUN_LABELS.update(_discover_matryoshka_runs(UNIFIED_EVAL_ROOT))

    results = collect_results()
    if not results and ALLOWED_MODELS:
        print("No results found with ALLOWED_MODELS; retrying without model filter.")
        ALLOWED_MODELS = []
        PREFERRED_MODEL_ORDER = []
        _allowed_epochs_for_label.cache_clear()
        results = collect_results()
    if args.best_epoch_only:
        results = select_best_epoch_per_model(results)
    output_path = plot_results(results)
    linear_table_path = render_table("linear", results, f"{DATASET}_linear_probe_table.png")
    knn_table_path = render_table("knn", results, f"{DATASET}_knn_table.png")
    model_list = sorted_models(list(results.keys()), results=results)
    model_colors = compute_model_colors(model_list)
    bump_chart_path = render_bump_chart(results, model_colors)
    line_plot_path = render_line_plots(results, model_colors)
    # gif_paths = make_ciip_pca_gifs()
    # svd_gif_paths = make_ciip_svd_gifs()

    print("Collected results (test_accuracy):")
    for model in sorted_models(list(results.keys())):
        methods = results[model]
        print(f"- {model}")
        for method in ("linear", "knn"):
            frac_values = methods.get(method)
            if not frac_values:
                continue
            formatted = ", ".join(
                f"{frac}: {acc:.3f}" if acc is not None else f"{frac}: N/A"
                for frac, acc in sorted(frac_values.items())
            )
            print(f"  {method}: {formatted}")

    print(f"\nPlot saved to: {output_path}")
    print(f"Linear probe table saved to: {linear_table_path}")
    print(f"k-NN table saved to: {knn_table_path}")
    print(f"Bump chart saved to: {bump_chart_path}")
    print(f"Line plots saved to: {line_plot_path}")
    # for gif in gif_paths:
    #     print(f"PCA GIF saved to: {gif}")
    # for gif in svd_gif_paths:
    #     print(f"SVD GIF saved to: {gif}")


if __name__ == "__main__":
    main()
