#!/usr/bin/env python3
"""Utilities for visualizing embedding collapse and training instability.

This script loads CIIP checkpoints, extracts embeddings for a shared subset of
samples, and produces a suite of plots that are commonly used to diagnose
representation collapse:

* Positive vs. negative cosine similarity statistics.
* Cosine similarity histograms per epoch.
* Per-dimension embedding variance heatmaps and summaries.
* Covariance spectrum (singular values) and condition numbers.
* Variance-covariance geometry diagnostics (std minima, γ gap, off-diagonal Frobenius norm, participation ratio).
* Linear probe accuracy overlays (optional, from CSV exports).
* t-SNE / UMAP projections of the embedding space across epochs.

The implementation follows the data loading patterns used in
``visualizations/ssl4eo/initialization_evaluation.py`` and
``ciip/evaluation/linearprobe_comparison.py`` so it can load models,
instantiate the SSL4EO dataset, and compute embeddings on-the-fly without
pre-extracted features.

Example usage::

    # python -m visualizations.ssl4eo.embedding_collapse_diagnostics \
    #     --checkpoint-root /home/juro4948/ciip/logs/2025_09_05-13_28_50-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints \
    #     --output-dir diagnostics/random_init \
    #     --dataset-root /local/ms-data/SSL4EO/ \
    #     --subset-size 2048 \
    #     --negative-samples 2000 \
    #     --linear-probe-csv results.csv \
    #     --linear-probe-pattern "RandomInit-hal-epoch(\\d+)" \
    #     --tsne-samples 800 \
    #     --umap-samples 800 \
    #     --use-orthogonal-mapping  # opt-in to applying W.pt to s*_features_vc

All plots are saved in the provided output directory; intermediate statistics
are exported as JSON/NumPy files for downstream analysis. The VC diagnostics are
also written to ``vc_metrics.csv`` and summarized in ``vc_metrics_timeseries.png``.

By default, embeddings are exported directly from the encoders (raw ``s*_features_vc``).
Pass ``--use-orthogonal-mapping`` to apply the optional ``W.pt`` alignment when present.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.manifold import TSNE

import sys
parent_dir = '/home/juro4948/ciip/ciip/open_clip_train/'
sys.path.insert(0, parent_dir)

# from ciip.open_clip_train.data import get_data
# from ciip.open_clip_train.precision import get_autocast
# from ciip.open_clip_train.utils import create_model
from data import get_data
from utils import create_model
from open_clip_train.precision import get_autocast

import os
try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    umap = None

try:  # pragma: no cover - optional dependency
    from open_clip import get_input_dtype  # type: ignore
except ImportError:  # pragma: no cover - open_clip is optional for docs/tests
    get_input_dtype = None  # type: ignore[assignment]


_LOGGER = logging.getLogger("embedding_collapse")


@dataclass
class EpochEmbeddings:
    """Container for the embeddings associated with a single epoch."""

    label: str
    epoch_index: int
    path: Path
    s1: Optional[torch.Tensor]
    s2: Optional[torch.Tensor]
    uids: List[str]
    s1_normalized: Optional[torch.Tensor] = None
    s2_normalized: Optional[torch.Tensor] = None
    s1_pre_projection: Optional[torch.Tensor] = None
    s2_pre_projection: Optional[torch.Tensor] = None


@dataclass
class EpochMetrics:
    """Aggregated metrics computed for a single epoch."""

    label: str
    epoch_index: int
    sample_count: int
    cosine_positive: Optional[np.ndarray] = None
    cosine_negative: Optional[np.ndarray] = None
    s1_variance: Optional[np.ndarray] = None
    s2_variance: Optional[np.ndarray] = None
    s1_spectrum: Optional[np.ndarray] = None
    s2_spectrum: Optional[np.ndarray] = None
    s1_singular_values: Optional[np.ndarray] = None
    s2_singular_values: Optional[np.ndarray] = None
    s1_pre_singular_values: Optional[np.ndarray] = None
    s2_pre_singular_values: Optional[np.ndarray] = None
    s1_condition_number: Optional[float] = None
    s2_condition_number: Optional[float] = None
    s1_std_min: Optional[float] = None
    s2_std_min: Optional[float] = None
    s1_cov_fro: Optional[float] = None
    s2_cov_fro: Optional[float] = None
    s1_participation_ratio: Optional[float] = None
    s2_participation_ratio: Optional[float] = None
    s1_correlation_spectrum: Optional[np.ndarray] = None
    s2_correlation_spectrum: Optional[np.ndarray] = None
    s1_corr_participation_ratio: Optional[float] = None
    s2_corr_participation_ratio: Optional[float] = None
    s1_corr_spectral_entropy: Optional[float] = None
    s2_corr_spectral_entropy: Optional[float] = None
    cca_spectrum: Optional[np.ndarray] = None
    cca_rho_max: Optional[float] = None
    cca_rho_topk_mean: Optional[float] = None
    s1_pre_std_min: Optional[float] = None
    s2_pre_std_min: Optional[float] = None
    s1_pre_cov_fro: Optional[float] = None
    s2_pre_cov_fro: Optional[float] = None
    s1_pre_participation_ratio: Optional[float] = None
    s2_pre_participation_ratio: Optional[float] = None
    s1_pre_corr_participation_ratio: Optional[float] = None
    s2_pre_corr_participation_ratio: Optional[float] = None
    s1_pre_corr_spectral_entropy: Optional[float] = None
    s2_pre_corr_spectral_entropy: Optional[float] = None

    def to_summary_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly summary of the metrics."""

        def _safe_stats(values: Optional[np.ndarray]) -> Dict[str, float]:
            if values is None or values.size == 0:
                return {}
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        vc_metrics = {
            "s1": {
                "std_min": self.s1_std_min,
                "cov_fro": self.s1_cov_fro,
                "participation_ratio": self.s1_participation_ratio,
                "correlation_participation_ratio": self.s1_corr_participation_ratio,
                "correlation_spectral_entropy": self.s1_corr_spectral_entropy,
            },
            "s2": {
                "std_min": self.s2_std_min,
                "cov_fro": self.s2_cov_fro,
                "participation_ratio": self.s2_participation_ratio,
                "correlation_participation_ratio": self.s2_corr_participation_ratio,
                "correlation_spectral_entropy": self.s2_corr_spectral_entropy,
            },
        }

        def _has_pre_metrics(*values: Optional[float]) -> bool:
            for value in values:
                if value is None:
                    continue
                value_float = float(value)
                if not math.isnan(value_float):
                    return True
            return False

        if _has_pre_metrics(
            self.s1_pre_std_min,
            self.s1_pre_cov_fro,
            self.s1_pre_participation_ratio,
            self.s1_pre_corr_participation_ratio,
            self.s1_pre_corr_spectral_entropy,
        ):
            vc_metrics["s1_pre_projection"] = {
                "std_min": self.s1_pre_std_min,
                "cov_fro": self.s1_pre_cov_fro,
                "participation_ratio": self.s1_pre_participation_ratio,
                "correlation_participation_ratio": self.s1_pre_corr_participation_ratio,
                "correlation_spectral_entropy": self.s1_pre_corr_spectral_entropy,
            }

        if _has_pre_metrics(
            self.s2_pre_std_min,
            self.s2_pre_cov_fro,
            self.s2_pre_participation_ratio,
            self.s2_pre_corr_participation_ratio,
            self.s2_pre_corr_spectral_entropy,
        ):
            vc_metrics["s2_pre_projection"] = {
                "std_min": self.s2_pre_std_min,
                "cov_fro": self.s2_pre_cov_fro,
                "participation_ratio": self.s2_pre_participation_ratio,
                "correlation_participation_ratio": self.s2_pre_corr_participation_ratio,
                "correlation_spectral_entropy": self.s2_pre_corr_spectral_entropy,
            }

        return {
            "label": self.label,
            "epoch_index": self.epoch_index,
            "sample_count": self.sample_count,
            "cosine_positive": _safe_stats(self.cosine_positive),
            "cosine_negative": _safe_stats(self.cosine_negative),
            "s1_variance": _safe_stats(self.s1_variance),
            "s2_variance": _safe_stats(self.s2_variance),
            "s1_condition_number": self.s1_condition_number,
            "s2_condition_number": self.s2_condition_number,
            "vc_metrics": vc_metrics,
            "cca": {
                "rho_max": self.cca_rho_max,
                "rho_topk_mean": self.cca_rho_topk_mean,
            },
        }

def _sanitize_label(label: str) -> str:
    """Return a filesystem-friendly variant of ``label``."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return safe or "epoch"


def _natural_key(value: str) -> List[object]:
    """Return a key for natural sorting of strings containing numbers."""

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\\d+)", value)]


def infer_epoch_index(name: str, fallback: int) -> int:
    """Infer an integer epoch index from a directory or model name."""

    for pattern in (r"epoch[_-]?(\\d+)", r"(\\d+)"):
        match = re.search(pattern, name)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return fallback


def resolve_input_dtype(precision: str) -> torch.dtype:
    """Resolve the tensor dtype used for model inputs."""

    dtype: Optional[torch.dtype] = None
    if get_input_dtype is not None:
        dtype = get_input_dtype(precision)
    if dtype is not None:
        return dtype

    precision = precision.lower()
    if "bf16" in precision:
        return torch.bfloat16
    if "fp16" in precision or "amp" in precision:
        return torch.float16
    return torch.float32


def ensure_hydra_original_cwd() -> None:
    """Ensure ``hydra.utils.get_original_cwd`` is defined when not using Hydra."""

    try:
        from hydra import utils as hydra_utils  # type: ignore
    except ImportError as exc:  # pragma: no cover - hydra is required for dataset loading
        raise ImportError(
            "hydra-core is required to instantiate the SSL4EO dataset; install it with `pip install hydra-core`."
        ) from exc

    try:
        hydra_utils.get_original_cwd()
    except Exception:
        repo_root = Path(__file__).resolve().parents[3]
        default_cwd = repo_root / "ciip" / "open_clip_train"
        hydra_utils.get_original_cwd = lambda: str(default_cwd)


def compare_state_keys(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[set, set]:
    """Return missing and unexpected keys between a model and checkpoint."""

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    return model_keys - checkpoint_keys, checkpoint_keys - model_keys


def load_model_from_checkpoint(
    config: DictConfig,
    checkpoint_path: Path,
    *,
    device: torch.device,
    input_dtype: torch.dtype,
    w_path: Optional[Path] = None,
    skip_final_fc: bool = False,
    use_orthogonal_mapping: bool = False,
) -> torch.nn.Module:
    """Instantiate a CIIP model and load weights from ``checkpoint_path``."""

    model = create_model(config, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(cleaned, strict=True)
        except Exception:
            if "encoder_s1.W" in cleaned:
                del cleaned["encoder_s1.W"]
            missing, unexpected = compare_state_keys(model, cleaned)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} is incompatible with the CIIP architecture."
                )
            model.load_state_dict(cleaned, strict=True)
        else:
            state_dict = cleaned

    use_w_mapping = bool(use_orthogonal_mapping and w_path is not None and w_path.exists())
    if use_w_mapping:
        try:
            orthogonal = torch.load(w_path, map_location=device)
        except Exception as exc:  # pragma: no cover - defensive branch
            _LOGGER.warning("Failed to load orthogonal mapping from %s: %s", w_path, exc)
            use_w_mapping = False
        else:
            if hasattr(model, "encoder_s1"):
                model.encoder_s1.register_buffer("W", orthogonal)
    if hasattr(model, "encoder_s1") and hasattr(model.encoder_s1, "apply_orthogonal_matrix"):
        model.encoder_s1.apply_orthogonal_matrix = use_w_mapping

    if input_dtype is not None:
        model = model.to(device, dtype=input_dtype, non_blocking=True)
    else:
        model = model.to(device, non_blocking=True)

    if skip_final_fc:
        if hasattr(model, "encoder_s1") and hasattr(model.encoder_s1, "fc"):
            model.encoder_s1.fc = nn.Identity()
        if hasattr(model, "encoder_s2") and hasattr(model.encoder_s2, "fc"):
            model.encoder_s2.fc = nn.Identity()

    model.eval()
    return model


def _unwrap_subset(dataset: torch.utils.data.Dataset) -> Tuple[torch.utils.data.Dataset, Sequence[int]]:
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset, dataset.indices  # type: ignore[return-value]
    return dataset, range(len(dataset))


def _register_pre_projection_hooks(
    model: torch.nn.Module,
) -> Tuple[Dict[str, List[torch.Tensor]], List[torch.utils.hooks.RemovableHandle]]:
    """Register hooks that capture inputs to the projection heads when available."""

    caches: Dict[str, List[torch.Tensor]] = {"s1": [], "s2": []}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(key: str):
        cache = caches[key]

        def hook(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):  # type: ignore[override]
            if not inputs:
                return
            tensor = inputs[0]
            if tensor is None:
                return
            cache.append(tensor.detach().to(dtype=torch.float32, device="cpu"))

        return hook

    def _maybe_register(module: Optional[torch.nn.Module], key: str) -> None:
        if module is None:
            return
        try:
            handle = module.register_forward_hook(_make_hook(key))
        except Exception:
            return
        handles.append(handle)

    encoder_s1 = getattr(model, "encoder_s1", None)
    if encoder_s1 is not None:
        for attr in ("fc", "proj", "projection_head"):
            module = getattr(encoder_s1, attr, None)
            if isinstance(module, torch.nn.Module):
                _maybe_register(module, "s1")
                break

    encoder_s2 = getattr(model, "encoder_s2", None)
    if encoder_s2 is not None:
        for attr in ("fc", "proj", "projection_head"):
            module = getattr(encoder_s2, attr, None)
            if isinstance(module, torch.nn.Module):
                _maybe_register(module, "s2")
                break

    return caches, handles


def extract_embeddings_for_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    input_dtype: torch.dtype,
    device: torch.device,
    autocast,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    List[str],
]:
    """Run the encoders over ``dataset`` and return raw & normalized embeddings."""

    base_dataset, index_map = _unwrap_subset(dataset)
    s1_vectors: List[torch.Tensor] = []
    s2_vectors: List[torch.Tensor] = []
    s1_normalized_vectors: List[torch.Tensor] = []
    s2_normalized_vectors: List[torch.Tensor] = []
    pre_projection_cache, handles = _register_pre_projection_hooks(model)
    uids: List[str] = []

    try:
        with torch.no_grad():
            for local_idx, base_idx in enumerate(index_map):
                sample = dataset[local_idx]
                if isinstance(sample, dict):
                    s1_img = sample.get("s1")
                    s2_img = sample.get("s2")
                else:
                    s1_img, s2_img = sample  # type: ignore[misc]

                if s1_img is None or s2_img is None:
                    continue

                s1_tensor = torch.as_tensor(s1_img).unsqueeze(0)
                s2_tensor = torch.as_tensor(s2_img).unsqueeze(0)
                if input_dtype is not None:
                    s1_tensor = s1_tensor.to(device=device, dtype=input_dtype, non_blocking=True)
                    s2_tensor = s2_tensor.to(device=device, dtype=input_dtype, non_blocking=True)
                else:
                    s1_tensor = s1_tensor.to(device=device, non_blocking=True)
                    s2_tensor = s2_tensor.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(s1_tensor, s2_tensor)

                if isinstance(model_out, dict):
                    s1_embed = model_out.get("s1_features_vc")
                    s2_embed = model_out.get("s2_features_vc")
                    s1_norm = model_out.get("s1_features")
                    s2_norm = model_out.get("s2_features")
                else:
                    s1_embed = s2_embed = None
                    s1_norm = s2_norm = None

                if s1_embed is None or s2_embed is None:
                    with autocast():
                        s1_embed = model.encode_s1(s1_tensor, normalize=False)
                        s2_embed = model.encode_s2(s2_tensor, normalize=False)
                    s1_norm = None
                    s2_norm = None

                if s1_norm is None:
                    s1_norm = F.normalize(s1_embed, dim=-1)
                if s2_norm is None:
                    s2_norm = F.normalize(s2_embed, dim=-1)

                s1_embed = s1_embed.squeeze(0).detach()
                s2_embed = s2_embed.squeeze(0).detach()
                s1_norm = s1_norm.squeeze(0).detach()
                s2_norm = s2_norm.squeeze(0).detach()
                del model_out

                s1_cpu = s1_embed.cpu()
                s2_cpu = s2_embed.cpu()
                s1_vectors.append(s1_cpu)
                s2_vectors.append(s2_cpu)

                s1_normalized_vectors.append(s1_norm.to(dtype=torch.float32).cpu())
                s2_normalized_vectors.append(s2_norm.to(dtype=torch.float32).cpu())

                if hasattr(base_dataset, "get_sample_uid"):
                    uid, _ = base_dataset.get_sample_uid(base_idx)
                else:
                    uid = base_idx
                uids.append(str(uid))
    finally:
        for handle in handles:
            handle.remove()

    if not s1_vectors or not s2_vectors:
        raise RuntimeError("No embeddings were extracted from the provided dataset subset.")

    def _stack_pre_features(key: str, expected: int) -> Optional[torch.Tensor]:
        cache = pre_projection_cache.get(key, [])
        if expected <= 0 or not cache:
            return None
        if len(cache) < expected:
            return None
        selected = cache[-expected:]
        tensors: List[torch.Tensor] = []
        for tensor in selected:
            if tensor.dim() > 1 and tensor.shape[0] == 1:
                tensors.append(tensor.squeeze(0))
            else:
                tensors.append(tensor.clone())
        try:
            return torch.stack(tensors, dim=0)
        except Exception:
            return None

    s1_stack = torch.stack(s1_vectors, dim=0)
    s2_stack = torch.stack(s2_vectors, dim=0)
    s1_norm_stack = torch.stack(s1_normalized_vectors, dim=0)
    s2_norm_stack = torch.stack(s2_normalized_vectors, dim=0)
    s1_pre_stack = _stack_pre_features("s1", len(s1_vectors))
    s2_pre_stack = _stack_pre_features("s2", len(s2_vectors))
    return s1_stack, s2_stack, s1_norm_stack, s2_norm_stack, s1_pre_stack, s2_pre_stack, uids


def discover_checkpoints(
    checkpoint_root: Path,
    *,
    pattern: Optional[str],
    include_init: bool,
    max_checkpoints: Optional[int],
) -> List[Path]:
    """Return checkpoint files sorted using natural ordering."""

    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root '{checkpoint_root}' does not exist")

    regex = re.compile(pattern) if pattern else None
    candidates: List[Path] = []
    for extension in ("*.pt", "*.pth"):
        candidates.extend(checkpoint_root.glob(extension))

    if regex is not None:
        candidates = [path for path in candidates if regex.search(path.name)]
    if not include_init:
        candidates = [path for path in candidates if "epoch_init" not in path.name]

    # candidates.sort(key=lambda p: _natural_key(p.name))
    # sort candidates
    # candidates = sorted(
    #     candidates,
    #     key=lambda p: int(p.stem.split('_')[1])
    # )
    # print(candidates)


    # if epoch_stride > 1:
    #     candidates = candidates[::epoch_stride]
    epochs = [1,2,3,4,5,6]

    # filter and keep only those epochs
    selected_paths = [
        p for p in candidates
        if int(p.stem.split('_')[1]) in epochs
    ]

    # (optional) sort them by epoch number
    candidates = sorted(
        selected_paths,
        key=lambda p: int(p.stem.split('_')[1])
    )

    print(candidates)

    # candidates = [candidates[i] for i in range(len(candidates)) if i in epochs or i == 0 or i == len(candidates)-1]
    if max_checkpoints is not None:
        candidates = candidates[:max_checkpoints]

    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching the provided criteria were found in '{checkpoint_root}'")

    return candidates


def collect_epoch_embeddings(
    checkpoint_root: Path,
    config: DictConfig,
    dataset: torch.utils.data.Dataset,
    *,
    input_dtype: torch.dtype,
    device: torch.device,
    autocast,
    pattern: Optional[str],
    include_init: bool,
    max_checkpoints: Optional[int],
    skip_final_fc: bool = False,
    use_orthogonal_mapping: bool = False,
) -> List[EpochEmbeddings]:
    """Extract embeddings for each checkpoint under ``checkpoint_root``.

    Raw encoder outputs are returned by default. Set ``use_orthogonal_mapping`` to ``True``
    to apply the optional ``W.pt`` alignment when present.
    """

    checkpoints = discover_checkpoints(
        checkpoint_root,
        pattern=pattern,
        include_init=include_init,
        max_checkpoints=max_checkpoints,
    )

    w_path = checkpoint_root / "W.pt"
    if not w_path.exists():
        w_path = None
        print("No orthogonal mapping W.pt found in checkpoint root")

    epochs: List[EpochEmbeddings] = []
    for idx, checkpoint_path in enumerate(checkpoints):
        label = checkpoint_path.stem
        epoch_index = infer_epoch_index(label, fallback=idx)
        _LOGGER.info("Extracting embeddings for %s (epoch index %d)", checkpoint_path.name, epoch_index)

        model = load_model_from_checkpoint(
            config,
            checkpoint_path,
            device=device,
            input_dtype=input_dtype,
            w_path=w_path,
            skip_final_fc=skip_final_fc,
            use_orthogonal_mapping=use_orthogonal_mapping,
        )

        try:
            (
                s1,
                s2,
                s1_normalized,
                s2_normalized,
                s1_pre,
                s2_pre,
                uids,
            ) = extract_embeddings_for_dataset(
                model,
                dataset,
                input_dtype=input_dtype,
                device=device,
                autocast=autocast,
            )
        except RuntimeError as exc:
            _LOGGER.error("Skipping %s due to extraction failure: %s", checkpoint_path.name, exc)
            continue

        epochs.append(
            EpochEmbeddings(
                label,
                epoch_index,
                checkpoint_path,
                s1,
                s2,
                uids,
                s1_normalized,
                s2_normalized,
                s1_pre,
                s2_pre,
            )
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    epochs.sort(key=lambda e: (e.epoch_index, e.label))
    if not epochs:
        raise RuntimeError(f"No embeddings were extracted from checkpoints in '{checkpoint_root}'")

    _LOGGER.info("Extracted embeddings for %d checkpoints from %s", len(epochs), checkpoint_root)
    return epochs




def build_subset(
    dataset: torch.utils.data.Dataset,
    subset_size: Optional[int],
    seed: int,
) -> torch.utils.data.Dataset:
    """Return a subset of ``dataset`` containing ``subset_size`` samples."""

    total = len(dataset)
    if subset_size is None or subset_size <= 0 or subset_size >= total:
        indices = list(range(total))
    else:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(total, size=subset_size, replace=False).tolist())
    return torch.utils.data.Subset(dataset, indices)


def compute_covariance(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Return the covariance matrix for ``tensor`` (N x D)."""

    if tensor.dim() != 2 or tensor.shape[0] < 2:
        return None
    centered = tensor.to(torch.float64) - tensor.mean(dim=0, keepdim=True).to(torch.float64)
    cov = centered.t().matmul(centered) / max(centered.shape[0] - 1, 1)
    return cov


def compute_singular_values(features: torch.Tensor) -> Optional[np.ndarray]:
    """Return singular values for ``features`` after mean-centering."""

    if features.dim() != 2:
        return None
    batch, dims = features.shape
    if batch < 2 or dims < 1:
        return None

    feats = features.to(torch.float64)
    feats = feats - feats.mean(dim=0, keepdim=True)

    try:
        singular = torch.linalg.svdvals(feats)
    except RuntimeError:
        return None

    if singular.numel() == 0:
        return None
    return singular.cpu().numpy()


def compute_vc_geometry(
    features: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """Return variance-covariance diagnostics for a batch of ``features``."""

    if features.dim() != 2 or features.shape[0] < 2:
        nan = float("nan")
        return nan, nan, nan

    feats = features.to(torch.float64)
    variances = torch.var(feats, dim=0, unbiased=False)
    std = torch.sqrt(torch.clamp(variances, min=0.0) + 1e-4)
    std_min = float(std.min().item())

    cov = compute_covariance(feats)
    if cov is None:
        nan = float("nan")
        return nan, nan, nan

    diag = torch.diag(torch.diagonal(cov))
    off_diag = (cov - diag).to(torch.float64)
    cov_fro = float(torch.linalg.norm(off_diag, ord="fro").item())

    eigvals = torch.linalg.eigvalsh(cov).real
    eigvals = torch.clamp(eigvals, min=0.0)
    participation = float(((eigvals.sum() ** 2) / (eigvals.pow(2).sum() + eps)).item())

    return std_min, cov_fro, participation


def compute_correlation_metrics(
    features: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[Optional[np.ndarray], float, float]:
    """Return correlation spectrum and scale-invariant summaries for ``features``."""

    if features.dim() != 2 or features.shape[0] < 2:
        nan = float("nan")
        return None, nan, nan

    feats = features.to(torch.float64)
    cov = compute_covariance(feats)
    if cov is None:
        nan = float("nan")
        return None, nan, nan

    variances = torch.diagonal(cov)
    std = torch.sqrt(torch.clamp(variances, min=0.0))
    inv_std = torch.where(std > eps, 1.0 / std, torch.zeros_like(std))
    inv_std_mat = torch.diag(inv_std)

    corr = inv_std_mat @ cov @ inv_std_mat
    corr = (corr + corr.t()) / 2

    try:
        eigvals = torch.linalg.eigvalsh(corr).flip(0)
    except RuntimeError:
        nan = float("nan")
        return None, nan, nan

    eigvals_np = eigvals.cpu().numpy().astype(np.float64)
    if eigvals_np.size == 0:
        nan = float("nan")
        return None, nan, nan

    eigvals_np = np.clip(eigvals_np, a_min=0.0, a_max=None)
    participation = float(((eigvals_np.sum() ** 2) / (np.sum(np.square(eigvals_np)) + eps)))

    total = float(np.sum(eigvals_np))
    if total <= eps:
        spectral_entropy = float("nan")
    else:
        probs = eigvals_np / total
        entropy = -float(np.sum(probs * np.log(probs + eps)))
        if eigvals_np.size <= 1:
            spectral_entropy = 0.0
        else:
            norm = math.log(eigvals_np.size)
            spectral_entropy = float(entropy / norm) if norm > 0 else float("nan")

    return eigvals_np, participation, spectral_entropy


def _symmetric_matrix_inverse_sqrt(
    matrix: torch.Tensor,
    eps: float = 1e-12,
) -> Optional[torch.Tensor]:
    """Return ``matrix^{-1/2}`` for a symmetric positive semi-definite matrix."""

    try:
        eigvals, eigvecs = torch.linalg.eigh((matrix + matrix.t()) / 2)
    except RuntimeError:
        return None

    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.t()
    return inv_sqrt


def compute_cca_spectrum(
    s1: torch.Tensor,
    s2: torch.Tensor,
    eps: float = 1e-12,
) -> Optional[np.ndarray]:
    """Return the canonical correlation spectrum between ``s1`` and ``s2``."""

    if s1.dim() != 2 or s2.dim() != 2:
        return None

    count = min(s1.shape[0], s2.shape[0])
    if count < 2:
        return None

    x = s1[:count].to(torch.float64)
    y = s2[:count].to(torch.float64)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    denom = max(count - 1, 1)
    cov_xx = (x.t().matmul(x)) / denom
    cov_yy = (y.t().matmul(y)) / denom
    cov_xy = (x.t().matmul(y)) / denom

    inv_sqrt_xx = _symmetric_matrix_inverse_sqrt(cov_xx, eps=eps)
    inv_sqrt_yy = _symmetric_matrix_inverse_sqrt(cov_yy, eps=eps)
    if inv_sqrt_xx is None or inv_sqrt_yy is None:
        return None

    t_matrix = inv_sqrt_xx @ cov_xy @ inv_sqrt_yy
    try:
        singular_values = torch.linalg.svdvals(t_matrix)
    except RuntimeError:
        return None

    singular_values = torch.clamp(singular_values, min=0.0, max=1.0)
    return singular_values.cpu().numpy()

def compute_condition_number(spectrum: np.ndarray, eps: float = 1e-12) -> float:
    """Compute a safe condition number from a descending spectrum."""

    if spectrum.size == 0:
        return float("nan")
    finite = spectrum[spectrum > eps]
    if finite.size == 0:
        return float("inf")
    smallest = float(finite[-1])
    largest = float(spectrum[0])
    if smallest <= eps:
        return float("inf")
    return largest / smallest


def compute_cosine_statistics(
    s1: torch.Tensor,
    s2: torch.Tensor,
    *,
    negative_samples: Optional[int],
    torch_generator: Optional[torch.Generator],
    numpy_generator: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute positive and (sampled) negative cosine similarities."""

    if s1.shape[0] != s2.shape[0]:
        count = min(s1.shape[0], s2.shape[0])
        s1 = s1[:count]
        s2 = s2[:count]

    s1_norm = F.normalize(s1, dim=1)
    s2_norm = F.normalize(s2, dim=1)
    positive = torch.sum(s1_norm * s2_norm, dim=1).cpu().numpy()

    n = s1_norm.shape[0]
    if n < 2:
        return positive, np.empty(0, dtype=np.float32)

    total_pairs = n * (n - 1)
    if negative_samples is None or negative_samples >= total_pairs:
        sim_matrix = s1_norm @ s2_norm.t()
        mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        mask.fill_diagonal_(False)
        negatives = sim_matrix[mask].cpu().numpy()
        if negative_samples is not None and negatives.size > negative_samples:
            indices = numpy_generator.choice(negatives.size, size=negative_samples, replace=False)
            negatives = negatives[indices]
        return positive, negatives

    if torch_generator is None:
        torch_generator = torch.Generator(device=s1.device)
    idx_i = torch.randint(0, n, (negative_samples,), generator=torch_generator)
    idx_j = torch.randint(0, n - 1, (negative_samples,), generator=torch_generator)
    idx_j = idx_j + (idx_j >= idx_i).long()
    negatives = torch.sum(s1_norm[idx_i] * s2_norm[idx_j], dim=1).cpu().numpy()
    return positive, negatives


def compute_epoch_metrics(
    epochs: Sequence[EpochEmbeddings],
    *,
    negative_samples: Optional[int],
    random_seed: int,
    cca_top_k: int,
) -> List[EpochMetrics]:
    """Compute diagnostic metrics for each epoch."""

    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(random_seed)
    np_gen = np.random.default_rng(random_seed)

    results: List[EpochMetrics] = []
    for epoch in epochs:
        if epoch.s1 is None or epoch.s2 is None:
            _LOGGER.warning("Skipping epoch %s due to missing embeddings", epoch.label)
            continue

        s1_tensor = epoch.s1.to(dtype=torch.float32)
        s2_tensor = epoch.s2.to(dtype=torch.float32)
        sample_count = s1_tensor.shape[0]
        s1_var = torch.var(s1_tensor, dim=0, unbiased=False).cpu().numpy()
        s2_var = torch.var(s2_tensor, dim=0, unbiased=False).cpu().numpy()

        metrics = EpochMetrics(label=epoch.label, epoch_index=epoch.epoch_index, sample_count=sample_count)
        metrics.s1_variance = s1_var
        metrics.s2_variance = s2_var
        metrics.s1_singular_values = compute_singular_values(s1_tensor)
        metrics.s2_singular_values = compute_singular_values(s2_tensor)

        cov_s1 = compute_covariance(s1_tensor)
        cov_s2 = compute_covariance(s2_tensor)

        s1_std_min, s1_cov_fro, s1_participation = compute_vc_geometry(epoch.s1)
        s2_std_min, s2_cov_fro, s2_participation = compute_vc_geometry(epoch.s2)

        metrics.s1_std_min = s1_std_min
        metrics.s2_std_min = s2_std_min
        metrics.s1_cov_fro = s1_cov_fro
        metrics.s2_cov_fro = s2_cov_fro
        metrics.s1_participation_ratio = s1_participation
        metrics.s2_participation_ratio = s2_participation

        (
            s1_corr_spec,
            s1_corr_pr,
            s1_corr_entropy,
        ) = compute_correlation_metrics(epoch.s1)
        (
            s2_corr_spec,
            s2_corr_pr,
            s2_corr_entropy,
        ) = compute_correlation_metrics(epoch.s2)

        metrics.s1_correlation_spectrum = s1_corr_spec
        metrics.s2_correlation_spectrum = s2_corr_spec
        metrics.s1_corr_participation_ratio = s1_corr_pr
        metrics.s2_corr_participation_ratio = s2_corr_pr
        metrics.s1_corr_spectral_entropy = s1_corr_entropy
        metrics.s2_corr_spectral_entropy = s2_corr_entropy

        if epoch.s1_pre_projection is not None:
            s1_pre_tensor = epoch.s1_pre_projection.to(dtype=torch.float32)
            metrics.s1_pre_singular_values = compute_singular_values(s1_pre_tensor)
            (
                s1_pre_std_min,
                s1_pre_cov_fro,
                s1_pre_participation,
            ) = compute_vc_geometry(s1_pre_tensor)
            (
                _s1_pre_corr_spec,
                s1_pre_corr_pr,
                s1_pre_corr_entropy,
            ) = compute_correlation_metrics(s1_pre_tensor)
            metrics.s1_pre_std_min = s1_pre_std_min
            metrics.s1_pre_cov_fro = s1_pre_cov_fro
            metrics.s1_pre_participation_ratio = s1_pre_participation
            metrics.s1_pre_corr_participation_ratio = s1_pre_corr_pr
            metrics.s1_pre_corr_spectral_entropy = s1_pre_corr_entropy

        if epoch.s2_pre_projection is not None:
            s2_pre_tensor = epoch.s2_pre_projection.to(dtype=torch.float32)
            metrics.s2_pre_singular_values = compute_singular_values(s2_pre_tensor)
            (
                s2_pre_std_min,
                s2_pre_cov_fro,
                s2_pre_participation,
            ) = compute_vc_geometry(s2_pre_tensor)
            (
                _s2_pre_corr_spec,
                s2_pre_corr_pr,
                s2_pre_corr_entropy,
            ) = compute_correlation_metrics(s2_pre_tensor)
            metrics.s2_pre_std_min = s2_pre_std_min
            metrics.s2_pre_cov_fro = s2_pre_cov_fro
            metrics.s2_pre_participation_ratio = s2_pre_participation
            metrics.s2_pre_corr_participation_ratio = s2_pre_corr_pr
            metrics.s2_pre_corr_spectral_entropy = s2_pre_corr_entropy

        cca_spectrum = compute_cca_spectrum(epoch.s1, epoch.s2)
        metrics.cca_spectrum = cca_spectrum
        if cca_spectrum is not None and cca_spectrum.size:
            metrics.cca_rho_max = float(cca_spectrum[0])
            top_k = cca_spectrum.size if cca_top_k <= 0 else min(cca_top_k, cca_spectrum.size)
            metrics.cca_rho_topk_mean = float(np.mean(cca_spectrum[:top_k]))

        if cov_s1 is not None:
            eigvals1 = torch.linalg.eigvalsh(cov_s1).flip(0).cpu().numpy()
            metrics.s1_spectrum = eigvals1
            metrics.s1_condition_number = compute_condition_number(eigvals1)
        if cov_s2 is not None:
            eigvals2 = torch.linalg.eigvalsh(cov_s2).flip(0).cpu().numpy()
            metrics.s2_spectrum = eigvals2
            metrics.s2_condition_number = compute_condition_number(eigvals2)

        s1_for_cosine = epoch.s1_normalized if epoch.s1_normalized is not None else epoch.s1
        s2_for_cosine = epoch.s2_normalized if epoch.s2_normalized is not None else epoch.s2

        positive, negative = compute_cosine_statistics(
            s1_for_cosine,
            s2_for_cosine,
            negative_samples=negative_samples,
            torch_generator=torch_gen,
            numpy_generator=np_gen,
        )
        metrics.cosine_positive = positive
        metrics.cosine_negative = negative

        results.append(metrics)

    return results



def _mark_axis_no_data(ax, title: str, message: str = "No data available") -> None:
    """Annotate ``ax`` to indicate that the desired plot could not be produced."""

    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_cosine_summary(metrics: Sequence[EpochMetrics], output_dir: Path) -> None:
    x = [m.epoch_index for m in metrics]
    labels = [m.label for m in metrics]

    pos_mean = [float(np.mean(m.cosine_positive)) if m.cosine_positive is not None else math.nan for m in metrics]
    pos_std = [float(np.std(m.cosine_positive)) if m.cosine_positive is not None else math.nan for m in metrics]
    neg_mean = [float(np.mean(m.cosine_negative)) if m.cosine_negative is not None and m.cosine_negative.size else math.nan for m in metrics]
    neg_std = [float(np.std(m.cosine_negative)) if m.cosine_negative is not None and m.cosine_negative.size else math.nan for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, pos_mean, yerr=pos_std, label="Positive pairs", marker="o", capsize=4)
    ax.errorbar(x, neg_mean, yerr=neg_std, label="Negative pairs", marker="s", capsize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine similarity statistics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cosine_similarity_summary.png", dpi=200)
    plt.close(fig)


def plot_cosine_histograms(metrics: Sequence[EpochMetrics], output_dir: Path, *, bins: int = 50) -> None:
    for metric in metrics:
        positives = metric.cosine_positive
        negatives = metric.cosine_negative

        has_positive = positives is not None and getattr(positives, "size", 0)
        has_negative = negatives is not None and getattr(negatives, "size", 0)
        if not (has_positive or has_negative):
            continue
        # fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        # axes[0].hist(metric.cosine_positive, bins=bins, color="#1f77b4", alpha=0.8)
        # axes[0].set_title(f"Positive pairs — {metric.label}")
        # axes[0].set_xlabel("Cosine similarity")
        # axes[0].set_ylabel("Count")

        # negatives = metric.cosine_negative if metric.cosine_negative is not None else np.empty(0)
        # axes[1].hist(negatives, bins=bins, color="#ff7f0e", alpha=0.8)
        # axes[1].set_title(f"Negative pairs — {metric.label}")
        # axes[1].set_xlabel("Cosine similarity")
        fig, ax = plt.subplots(figsize=(8, 4))
        if has_positive:
            ax.hist(
                positives,
                bins=bins,
                color="#1f77b4",
                alpha=0.6,
                label="Positive pairs",
            )
        if has_negative:
            ax.hist(
                negatives,
                bins=bins,
                color="#ff7f0e",
                alpha=0.6,
                label="Negative pairs",
            )

        ax.set_title(f"Cosine similarity — {metric.label}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        if has_positive or has_negative:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"cosine_hist_{metric.label}.png", dpi=200)
        plt.close(fig)


def plot_variance_heatmap(
    metrics: Sequence[EpochMetrics],
    output_dir: Path,
    *,
    modality: str,
) -> None:
    variances = [m.s1_variance if modality == "s1" else m.s2_variance for m in metrics]
    if any(v is None for v in variances):
        _LOGGER.warning("Skipping variance heatmap for %s (missing data)", modality)
        return
    variance_matrix = np.stack(variances)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(variance_matrix, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Epoch")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.label for m in metrics])
    ax.set_title(f"{modality.upper()} variance per dimension")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Variance")
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_variance_heatmap.png", dpi=200)
    plt.close(fig)

    # Summary statistics per epoch
    stats = {
        "mean": np.mean(variance_matrix, axis=1),
        "median": np.median(variance_matrix, axis=1),
        "min": np.min(variance_matrix, axis=1),
        "max": np.max(variance_matrix, axis=1),
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [m.epoch_index for m in metrics]
    for name, values in stats.items():
        ax.plot(x, values, marker="o", label=name.capitalize())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Variance")
    ax.set_title(f"{modality.upper()} variance statistics")
    ax.set_xticks(x)
    ax.set_xticklabels([m.label for m in metrics], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_variance_summary.png", dpi=200)
    plt.close(fig)


def _extract_metric_series(metrics: Sequence[EpochMetrics], attr: str) -> List[float]:
    values: List[float] = []
    for metric in metrics:
        value = getattr(metric, attr, None)
        if value is None:
            values.append(math.nan)
        else:
            values.append(float(value))
    return values


def _plot_vc_timeseries_panel(
    ax,
    metrics: Sequence[EpochMetrics],
    *,
    attr_s1: str,
    attr_s2: str,
    attr_s1_pre: Optional[str] = None,
    attr_s2_pre: Optional[str] = None,
    title: str,
    ylabel: str,
    reference: Optional[Tuple[float, Optional[str]]] = None,
) -> None:
    x = [m.epoch_index for m in metrics]
    labels = [m.label for m in metrics]
    series_plotted = False

    series_definitions = [
        (attr_s1, "S1 (post)", "o", "-", "#2ca02c"),
        (attr_s1_pre, "S1 (pre)", "o", "--", "#2ca02c"),
        (attr_s2, "S2 (post)", "s", "-", "#17becf"),
        (attr_s2_pre, "S2 (pre)", "s", "--", "#17becf"),
    ]

    for attr, label, marker, linestyle, color in series_definitions:
        if attr is None:
            continue
        values = _extract_metric_series(metrics, attr)
        if any(not math.isnan(v) for v in values):
            ax.plot(x, values, marker=marker, linestyle=linestyle, color=color, label=label)
            series_plotted = True

    if not series_plotted:
        _mark_axis_no_data(ax, title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Epoch")
        return

    if reference is not None:
        ref_value, ref_label = reference
        if ref_value is not None:
            line = ax.axhline(ref_value, color="k", linestyle="--", linewidth=1)
            if ref_label:
                line.set_label(ref_label)
            else:
                line.set_label("_nolegend_")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    handles, legends = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, legends) if l != "_nolegend_"]
    if filtered:
        handles, legends = zip(*filtered)
        ax.legend(handles, legends)


def _plot_spectrum_panel(
    ax,
    metrics: Sequence[EpochMetrics],
    *,
    attr_s1: Optional[str] = None,
    attr_s2: Optional[str] = None,
    attr_s1_pre: Optional[str] = None,
    attr_s2_pre: Optional[str] = None,
    title: str,
    ylabel: str,
    top_k: int,
    log_scale: bool,
    component_symbol: str = "λ",
) -> None:
    x = [m.epoch_index for m in metrics]
    labels = [m.label for m in metrics]

    def _collect(attr: Optional[str]) -> List[Optional[np.ndarray]]:
        if attr is None:
            return [None for _ in metrics]
        return [getattr(m, attr) for m in metrics]

    spectra_s1 = _collect(attr_s1)
    spectra_s2 = _collect(attr_s2)
    spectra_s1_pre = _collect(attr_s1_pre)
    spectra_s2_pre = _collect(attr_s2_pre)
    max_dims = max(
        [
            max((spec.size for spec in spectra if spec is not None), default=0)
            for spectra in (spectra_s1, spectra_s2, spectra_s1_pre, spectra_s2_pre)
        ]
    )

    if max_dims == 0:
        _mark_axis_no_data(ax, title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Epoch")
        return

    if top_k <= 0 or top_k > max_dims:
        top_k = max_dims

    colors = plt.cm.viridis(np.linspace(0.2, 0.95, top_k))
    plotted = False
    for idx in range(top_k):
        s1_values: List[float] = []
        s2_values: List[float] = []
        s1_pre_values: List[float] = []
        s2_pre_values: List[float] = []
        for spec in spectra_s1:
            if spec is None or spec.size <= idx:
                s1_values.append(math.nan)
            else:
                value = float(spec[idx])
                if log_scale and value <= 0:
                    s1_values.append(math.nan)
                else:
                    s1_values.append(value)
        for spec in spectra_s2:
            if spec is None or spec.size <= idx:
                s2_values.append(math.nan)
            else:
                value = float(spec[idx])
                if log_scale and value <= 0:
                    s2_values.append(math.nan)
                else:
                    s2_values.append(value)
        for spec in spectra_s1_pre:
            if spec is None or spec.size <= idx:
                s1_pre_values.append(math.nan)
            else:
                value = float(spec[idx])
                if log_scale and value <= 0:
                    s1_pre_values.append(math.nan)
                else:
                    s1_pre_values.append(value)
        for spec in spectra_s2_pre:
            if spec is None or spec.size <= idx:
                s2_pre_values.append(math.nan)
            else:
                value = float(spec[idx])
                if log_scale and value <= 0:
                    s2_pre_values.append(math.nan)
                else:
                    s2_pre_values.append(value)

        if any(not math.isnan(v) for v in s1_values):
            ax.plot(
                x,
                s1_values,
                marker="o",
                linestyle="-",
                color=colors[idx],
                label=f"S1 post {component_symbol}{idx + 1}",
            )
            plotted = True
        if any(not math.isnan(v) for v in s1_pre_values):
            ax.plot(
                x,
                s1_pre_values,
                marker="o",
                linestyle="--",
                color=colors[idx],
                alpha=0.8,
                label=f"S1 pre {component_symbol}{idx + 1}",
            )
            plotted = True
        if any(not math.isnan(v) for v in s2_values):
            ax.plot(
                x,
                s2_values,
                marker="s",
                linestyle="--",
                color=colors[idx],
                label=f"S2 post {component_symbol}{idx + 1}",
            )
            plotted = True
        if any(not math.isnan(v) for v in s2_pre_values):
            ax.plot(
                x,
                s2_pre_values,
                marker="s",
                linestyle=":",
                color=colors[idx],
                alpha=0.8,
                label=f"S2 pre {component_symbol}{idx + 1}",
            )
            plotted = True

    if not plotted:
        _mark_axis_no_data(ax, title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Epoch")
        return

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    else:
        ax.grid(True, alpha=0.3)

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, legend_labels, ncol=2, fontsize="small")


def _plot_correlation_panel(
    ax,
    metrics: Sequence[EpochMetrics],
    *,
    top_k: int,
) -> None:
    x = [m.epoch_index for m in metrics]
    labels = [m.label for m in metrics]

    spectra_s1 = [m.s1_correlation_spectrum for m in metrics]
    spectra_s2 = [m.s2_correlation_spectrum for m in metrics]
    max_dims = max(
        [
            max((spec.size for spec in spectra if spec is not None), default=0)
            for spectra in (spectra_s1, spectra_s2)
        ]
    )

    corr_pr_s1 = _extract_metric_series(metrics, "s1_corr_participation_ratio")
    corr_pr_s2 = _extract_metric_series(metrics, "s2_corr_participation_ratio")
    corr_pr_s1_pre = _extract_metric_series(metrics, "s1_pre_corr_participation_ratio")
    corr_pr_s2_pre = _extract_metric_series(metrics, "s2_pre_corr_participation_ratio")
    has_pr_s1 = any(not math.isnan(v) for v in corr_pr_s1)
    has_pr_s2 = any(not math.isnan(v) for v in corr_pr_s2)
    has_pr_s1_pre = any(not math.isnan(v) for v in corr_pr_s1_pre)
    has_pr_s2_pre = any(not math.isnan(v) for v in corr_pr_s2_pre)

    if max_dims == 0:
        if not (has_pr_s1 or has_pr_s2):
            _mark_axis_no_data(ax, "Correlation spectrum and participation ratio")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel("Epoch")
            return

        ax.set_title("Correlation participation ratio")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Correlation PR")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        if has_pr_s1:
            ax.plot(
                x,
                corr_pr_s1,
                marker="^",
                linestyle="-.",
                color="#2ca02c",
                label="S1 Corr PR",
            )
        if has_pr_s1_pre:
            ax.plot(
                x,
                corr_pr_s1_pre,
                marker="^",
                linestyle="--",
                color="#98df8a",
                label="S1 Corr PR (pre)",
            )
        if has_pr_s2:
            ax.plot(
                x,
                corr_pr_s2,
                marker="v",
                linestyle=":",
                color="#d62728",
                label="S2 Corr PR",
            )
        if has_pr_s2_pre:
            ax.plot(
                x,
                corr_pr_s2_pre,
                marker="v",
                linestyle="-",
                color="#ff9896",
                label="S2 Corr PR (pre)",
            )
        ax.grid(True, alpha=0.3)
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, loc="upper right", fontsize="small")
        return

    if top_k <= 0 or top_k > max_dims:
        top_k = max_dims

    colors = plt.cm.plasma(np.linspace(0.2, 0.95, top_k))
    plotted = False
    for idx in range(top_k):
        s1_values: List[float] = []
        s2_values: List[float] = []
        for spec in spectra_s1:
            if spec is None or spec.size <= idx:
                s1_values.append(math.nan)
            else:
                s1_values.append(float(spec[idx]))
        for spec in spectra_s2:
            if spec is None or spec.size <= idx:
                s2_values.append(math.nan)
            else:
                s2_values.append(float(spec[idx]))

        if any(not math.isnan(v) for v in s1_values):
            ax.plot(
                x,
                s1_values,
                marker="o",
                linestyle="-",
                color=colors[idx],
                label=f"S1 ρ{idx + 1}",
            )
            plotted = True
        if any(not math.isnan(v) for v in s2_values):
            ax.plot(
                x,
                s2_values,
                marker="s",
                linestyle="--",
                color=colors[idx],
                label=f"S2 ρ{idx + 1}",
            )
            plotted = True

    ax.set_title("Correlation spectrum and participation ratio")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation eigenvalue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    handles1, labels1 = ax.get_legend_handles_labels()

    if has_pr_s1 or has_pr_s2 or has_pr_s1_pre or has_pr_s2_pre:
        ax2 = ax.twinx()
        ax2.set_ylabel("Correlation PR")
        ax2.grid(False)
        if has_pr_s1:
            ax2.plot(
                x,
                corr_pr_s1,
                marker="^",
                linestyle="-.",
                color="#2ca02c",
                label="S1 Corr PR",
            )
        if has_pr_s1_pre:
            ax2.plot(
                x,
                corr_pr_s1_pre,
                marker="^",
                linestyle="--",
                color="#98df8a",
                label="S1 Corr PR (pre)",
            )
        if has_pr_s2:
            ax2.plot(
                x,
                corr_pr_s2,
                marker="v",
                linestyle=":",
                color="#d62728",
                label="S2 Corr PR",
            )
        if has_pr_s2_pre:
            ax2.plot(
                x,
                corr_pr_s2_pre,
                marker="v",
                linestyle="-",
                color="#ff9896",
                label="S2 Corr PR (pre)",
            )
        handles2, labels2 = ax2.get_legend_handles_labels()
    else:
        handles2, labels2 = [], []

    if plotted or handles2:
        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize="small")
    else:
        _mark_axis_no_data(ax, "Correlation spectrum and participation ratio")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Epoch")


def _plot_cca_panel(
    ax,
    metrics: Sequence[EpochMetrics],
    *,
    top_k: int,
) -> None:
    x = [m.epoch_index for m in metrics]
    labels = [m.label for m in metrics]

    spectra = [m.cca_spectrum for m in metrics]
    max_dims = max((spec.size for spec in spectra if spec is not None), default=0)

    if max_dims == 0:
        _mark_axis_no_data(ax, "CCA canonical correlations")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Epoch")
        return

    requested_top_k = top_k
    if top_k <= 0 or top_k > max_dims:
        top_k = max_dims

    colors = plt.cm.magma(np.linspace(0.2, 0.95, top_k))
    plotted = False
    for idx in range(top_k):
        values: List[float] = []
        for spec in spectra:
            if spec is None or spec.size <= idx:
                values.append(math.nan)
            else:
                values.append(float(spec[idx]))
        if any(not math.isnan(v) for v in values):
            ax.plot(
                x,
                values,
                marker="o",
                linestyle="-",
                color=colors[idx],
                label=f"ρ{idx + 1}",
            )
            plotted = True

    topk_mean = _extract_metric_series(metrics, "cca_rho_topk_mean")
    if any(not math.isnan(v) for v in topk_mean):
        if requested_top_k <= 0 or requested_top_k > max_dims:
            mean_label = "Mean (all)"
        else:
            mean_label = f"Mean top-{requested_top_k}"
        ax.plot(
            x,
            topk_mean,
            marker="s",
            linestyle="--",
            color="black",
            label=mean_label,
        )
        plotted = True

    if not plotted:
        _mark_axis_no_data(ax, "CCA canonical correlations")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Epoch")
        return

    ax.set_title("CCA canonical correlations")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Canonical correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, legend_labels, ncol=2, fontsize="small")


def plot_vc_timeseries(
    metrics: Sequence[EpochMetrics],
    output_dir: Path,
    *,
    vc_gamma: float,
    spectrum_top_k: int,
    cca_top_k: int,
) -> None:
    if not metrics:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_list = axes.flatten()

    _plot_vc_timeseries_panel(
        axes_list[0],
        metrics,
        attr_s1="s1_std_min",
        attr_s2="s2_std_min",
        attr_s1_pre="s1_pre_std_min",
        attr_s2_pre="s2_pre_std_min",
        title="Minimum per-dimension std. deviation",
        ylabel="Std. deviation",
        reference=(vc_gamma, r"γ"),
    )
    _plot_vc_timeseries_panel(
        axes_list[1],
        metrics,
        attr_s1="s1_cov_fro",
        attr_s2="s2_cov_fro",
        attr_s1_pre="s1_pre_cov_fro",
        attr_s2_pre="s2_pre_cov_fro",
        title="Off-diagonal covariance Frobenius norm",
        ylabel="‖Cov_off‖₍fro₎",
        reference=None,
    )
    _plot_vc_timeseries_panel(
        axes_list[2],
        metrics,
        attr_s1="s1_participation_ratio",
        attr_s2="s2_participation_ratio",
        attr_s1_pre="s1_pre_participation_ratio",
        attr_s2_pre="s2_pre_participation_ratio",
        title="Covariance participation ratio",
        ylabel="Participation ratio",
        reference=None,
    )

    if spectrum_top_k > 0:
        cov_title = f"Covariance spectrum (top {spectrum_top_k})"
    else:
        cov_title = "Covariance spectrum (all)"

    _plot_spectrum_panel(
        axes_list[3],
        metrics,
        attr_s1="s1_spectrum",
        attr_s2="s2_spectrum",
        title=cov_title,
        ylabel="Eigenvalue",
        top_k=spectrum_top_k,
        log_scale=True,
    )
    _plot_correlation_panel(
        axes_list[4],
        metrics,
        top_k=spectrum_top_k,
    )
    _plot_cca_panel(
        axes_list[5],
        metrics,
        top_k=cca_top_k,
    )

    fig.suptitle("Variance, correlation, and CCA diagnostics across epochs")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "vc_metrics_timeseries.png", dpi=200)
    plt.close(fig)


def plot_singular_value_timeseries(
    metrics: Sequence[EpochMetrics],
    output_dir: Path,
    *,
    top_k: int,
) -> None:
    """Plot singular value trajectories for pre/post-projection features."""

    if not metrics:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    _plot_spectrum_panel(
        axes[0],
        metrics,
        attr_s1="s1_singular_values",
        attr_s1_pre="s1_pre_singular_values",
        title="S1 singular values (pre vs. post)",
        ylabel="Singular value",
        top_k=top_k,
        log_scale=True,
        component_symbol="σ",
    )

    _plot_spectrum_panel(
        axes[1],
        metrics,
        attr_s2="s2_singular_values",
        attr_s2_pre="s2_pre_singular_values",
        title="S2 singular values (pre vs. post)",
        ylabel="Singular value",
        top_k=top_k,
        log_scale=True,
        component_symbol="σ",
    )

    fig.suptitle("Singular values across epochs")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "singular_values_timeseries.png", dpi=200)
    plt.close(fig)


def plot_spectrum(
    metrics: Sequence[EpochMetrics],
    output_dir: Path,
    *,
    modality: str,
    top_k: int,
) -> None:
    spectra = [m.s1_spectrum if modality == "s1" else m.s2_spectrum for m in metrics]
    valid_spectra = [s for s in spectra if s is not None]
    if not valid_spectra:
        _LOGGER.warning("Skipping spectrum plot for %s (missing data)", modality)
        return

    max_len = min(top_k, min(len(s) for s in valid_spectra))
    if max_len == 0:
        _LOGGER.warning("No eigenvalues available to plot for %s", modality)
        return

    x = [m.epoch_index for m in metrics]
    labels = [m.label for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx in range(max_len):
        values = [s[idx] if s is not None and len(s) > idx else math.nan for s in spectra]
        ax.plot(x, values, marker="o", label=f"λ{idx + 1}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"{modality.upper()} covariance spectrum (top {max_len})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_spectrum.png", dpi=200)
    plt.close(fig)

    condition_numbers = [m.s1_condition_number if modality == "s1" else m.s2_condition_number for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, condition_numbers, marker="o", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Condition number")
    ax.set_title(f"{modality.upper()} covariance condition number")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_condition_number.png", dpi=200)
    plt.close(fig)


def plot_epoch_dashboards(
    metrics: Sequence[EpochMetrics],
    output_dir: Path,
    *,
    cosine_bins: int,
    spectrum_top_k: int,
) -> None:
    """Generate a multi-panel diagnostic figure for each epoch."""

    for metric in metrics:
        fig = plt.figure(figsize=(14, 12))
        # fig.suptitle(f"Diagnostics — {metric.label}")
        grid = fig.add_gridspec(3, 2)
        hist_ax = fig.add_subplot(grid[0, :])
        s1_var_ax = fig.add_subplot(grid[1, 0])
        s2_var_ax = fig.add_subplot(grid[1, 1])
        s1_spec_ax = fig.add_subplot(grid[2, 0])
        s2_spec_ax = fig.add_subplot(grid[2, 1])

        def _format_line(modality: str) -> str:
            std_min = getattr(metric, f"{modality}_std_min")
            cov_fro = getattr(metric, f"{modality}_cov_fro")
            participation = getattr(metric, f"{modality}_participation_ratio")
            corr_pr = getattr(metric, f"{modality}_corr_participation_ratio")
            corr_entropy = getattr(metric, f"{modality}_corr_spectral_entropy")
            pre_std_min = getattr(metric, f"{modality}_pre_std_min")
            pre_cov_fro = getattr(metric, f"{modality}_pre_cov_fro")
            pre_participation = getattr(metric, f"{modality}_pre_participation_ratio")
            pre_corr_pr = getattr(metric, f"{modality}_pre_corr_participation_ratio")
            pre_corr_entropy = getattr(metric, f"{modality}_pre_corr_spectral_entropy")

            def _fmt(value: Optional[float], fmt: str) -> str:
                if value is None:
                    return "NA"
                value_float = float(value)
                if math.isnan(value_float):
                    return "NA"
                return format(value_float, fmt)

            def _fmt_pair(value: Optional[float], pre_value: Optional[float], fmt: str) -> str:
                formatted = _fmt(value, fmt)
                if pre_value is None:
                    return formatted
                pre_float = float(pre_value)
                if math.isnan(pre_float):
                    return formatted
                return f"{formatted} (pre={_fmt(pre_value, fmt)})"

            return (
                f"{modality.upper()} std_min={_fmt_pair(std_min, pre_std_min, '.4f')}, "
                f"‖Cov_off‖={_fmt_pair(cov_fro, pre_cov_fro, '.2e')}, "
                f"CovPR={_fmt_pair(participation, pre_participation, '.2f')}, "
                f"CorrPR={_fmt_pair(corr_pr, pre_corr_pr, '.2f')}, "
                f"CorrH={_fmt_pair(corr_entropy, pre_corr_entropy, '.3f')}"
            )

        fig.suptitle(
            "\n".join(
                [
                    f"Diagnostics — {metric.label}",
                    _format_line("s1"),
                    _format_line("s2"),
                ]
            )
        )

        positives = metric.cosine_positive
        negatives = metric.cosine_negative
        has_positive = positives is not None and getattr(positives, "size", 0)
        has_negative = negatives is not None and getattr(negatives, "size", 0)

        if has_positive or has_negative:
            if has_positive:
                hist_ax.hist(
                    positives,
                    bins=cosine_bins,
                    color="#1f77b4",
                    alpha=0.6,
                    label="Positive pairs",
                )
            if has_negative:
                hist_ax.hist(
                    negatives,
                    bins=cosine_bins,
                    color="#ff7f0e",
                    alpha=0.6,
                    label="Negative pairs",
                )
            hist_ax.set_title("Cosine similarity distribution")
            hist_ax.set_xlabel("Cosine similarity")
            hist_ax.set_ylabel("Count")
            hist_ax.legend()
            hist_ax.grid(True, alpha=0.3)
        else:
            _mark_axis_no_data(hist_ax, "Cosine similarity distribution")

        # Variance per modality
        if metric.s1_variance is not None and metric.s1_variance.size:
            dims = np.arange(1, metric.s1_variance.size + 1)
            s1_var_ax.plot(dims, metric.s1_variance, color="#2ca02c")
            s1_var_ax.set_title("S1 variance by dimension")
            s1_var_ax.set_xlabel("Dimension")
            s1_var_ax.set_ylabel("Variance")
        else:
            _mark_axis_no_data(s1_var_ax, "S1 variance by dimension")

        if metric.s2_variance is not None and metric.s2_variance.size:
            dims = np.arange(1, metric.s2_variance.size + 1)
            s2_var_ax.plot(dims, metric.s2_variance, color="#17becf")
            s2_var_ax.set_title("S2 variance by dimension")
            s2_var_ax.set_xlabel("Dimension")
            s2_var_ax.set_ylabel("Variance")
        else:
            _mark_axis_no_data(s2_var_ax, "S2 variance by dimension")

        # Covariance spectrum per modality
        if metric.s1_spectrum is not None and metric.s1_spectrum.size:
            take = min(spectrum_top_k, metric.s1_spectrum.size)
            idx = np.arange(1, take + 1)
            s1_spec_ax.plot(idx, metric.s1_spectrum[:take], marker="o", color="#9467bd")
            s1_spec_ax.set_yscale("log")
            s1_spec_ax.set_title(f"S1 covariance spectrum (top {take})")
            s1_spec_ax.set_xlabel("Component")
            s1_spec_ax.set_ylabel("Eigenvalue")
            cond = metric.s1_condition_number
            if cond is not None:
                if math.isnan(cond):
                    text = "cond = nan"
                elif math.isinf(cond):
                    text = "cond = inf"
                else:
                    text = f"cond = {cond:.2e}"
                s1_spec_ax.text(
                    0.95,
                    0.05,
                    text,
                    transform=s1_spec_ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6},
                )
        else:
            _mark_axis_no_data(s1_spec_ax, "S1 covariance spectrum")

        if metric.s2_spectrum is not None and metric.s2_spectrum.size:
            take = min(spectrum_top_k, metric.s2_spectrum.size)
            idx = np.arange(1, take + 1)
            s2_spec_ax.plot(idx, metric.s2_spectrum[:take], marker="o", color="#8c564b")
            s2_spec_ax.set_yscale("log")
            s2_spec_ax.set_title(f"S2 covariance spectrum (top {take})")
            s2_spec_ax.set_xlabel("Component")
            s2_spec_ax.set_ylabel("Eigenvalue")
            cond = metric.s2_condition_number
            if cond is not None:
                if math.isnan(cond):
                    text = "cond = nan"
                elif math.isinf(cond):
                    text = "cond = inf"
                else:
                    text = f"cond = {cond:.2e}"
                s2_spec_ax.text(
                    0.95,
                    0.05,
                    text,
                    transform=s2_spec_ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.6},
                )
        else:
            _mark_axis_no_data(s2_spec_ax, "S2 covariance spectrum")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        safe_label = _sanitize_label(metric.label)
        filename = output_dir / f"epoch_{metric.epoch_index:04d}_{safe_label}_diagnostics.png"
        fig.savefig(filename, dpi=200)
        plt.close(fig)



def _plot_epoch_svd_panel(
    ax,
    *,
    post: Optional[np.ndarray],
    pre: Optional[np.ndarray],
    top_k: int,
    title: str,
    post_color: str,
    pre_color: str,
) -> None:
    if post is None and pre is None:
        _mark_axis_no_data(ax, title)
        return

    lengths = [len(arr) for arr in (post, pre) if arr is not None]
    if not lengths:
        _mark_axis_no_data(ax, title)
        return

    max_len = max(lengths)
    take = max_len if top_k <= 0 else min(top_k, max_len)
    dims = np.arange(1, take + 1)

    if post is not None and post.size:
        ax.plot(
            dims,
            post[:take],
            marker="o",
            linestyle="-",
            color=post_color,
            label="Post projection",
        )
    if pre is not None and pre.size:
        ax.plot(
            dims,
            pre[:take],
            marker="s",
            linestyle="--",
            color=pre_color,
            label="Pre projection",
        )

    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular value")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)


def plot_epoch_singular_values(
    metrics: Sequence[EpochMetrics],
    output_dir: Path,
    *,
    top_k: int,
) -> None:
    '''Write per-epoch singular value comparisons for pre/post features.'''

    for metric in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Singular values — {metric.label}")

        _plot_epoch_svd_panel(
            axes[0],
            post=metric.s1_singular_values,
            pre=metric.s1_pre_singular_values,
            top_k=top_k,
            title="S1 singular values",
            post_color="#2ca02c",
            pre_color="#98df8a",
        )

        _plot_epoch_svd_panel(
            axes[1],
            post=metric.s2_singular_values,
            pre=metric.s2_pre_singular_values,
            top_k=top_k,
            title="S2 singular values",
            post_color="#17becf",
            pre_color="#9edae5",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        safe_label = _sanitize_label(metric.label)
        output_path = output_dir / f"epoch_{metric.epoch_index:04d}_{safe_label}_singular_values.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)





def parse_linear_probe_csv(
    csv_path: Path,
    *,
    model_pattern: Optional[str],
    metric: str,
    k_value: Optional[float],
) -> Dict[int, Tuple[float, Optional[float]]]:
    """Parse a CSV produced by ``output_results_to_csv``.

    Returns a mapping from epoch index to (mean, std) for the requested
    configuration. When multiple rows match, later values override earlier ones.
    """

    pattern = re.compile(model_pattern) if model_pattern else None
    results: Dict[int, Tuple[float, Optional[float]]] = {}

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        _LOGGER.warning("Linear probe CSV '%s' not found; skipping", csv_path)
        return {}


    os.makedirs(csv_path.parent, exist_ok=True)
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        total_cols = len(header)
        split = (total_cols - 1) // 2
        k_headers = header[1 : 1 + split]
        std_headers = header[1 + split : 1 + 2 * split]
        k_values = [float(h.split("=")[1]) for h in k_headers]

        std_map = {float(h.split("=")[1]): idx for idx, h in enumerate(std_headers)}

        for row in reader:
            model_name = row[0]
            if pattern and not pattern.search(model_name):
                continue
            epoch_idx = infer_epoch_index(model_name, fallback=-1)
            if epoch_idx < 0:
                continue

            metrics_mean = [float(v) for v in row[1 : 1 + split]]
            metrics_std = [float(v) for v in row[1 + split : 1 + 2 * split]]

            if k_value is None and metrics_mean:
                results[epoch_idx] = (metrics_mean[0], metrics_std[0] if metrics_std else None)
                continue

            try:
                idx = min(range(len(k_values)), key=lambda i: abs(k_values[i] - k_value))
            except ValueError:  # pragma: no cover - defensive
                continue
            results[epoch_idx] = (metrics_mean[idx], metrics_std[idx] if metrics_std else None)

    return results


def plot_linear_probe_curve(
    linear_probe: Dict[int, Tuple[float, Optional[float]]],
    output_dir: Path,
    *,
    metric: str,
) -> None:
    if not linear_probe:
        _LOGGER.warning("No linear probe results to plot")
        return

    epochs = sorted(linear_probe.keys())
    means = [linear_probe[e][0] for e in epochs]
    stds = [linear_probe[e][1] for e in epochs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(epochs, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Linear probe {metric} vs. epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"linear_probe_{metric}.png", dpi=200)
    plt.close(fig)


def sample_for_projection(
    s1: torch.Tensor,
    s2: torch.Tensor,
    *,
    per_modality: int,
    generator: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a stacked array of embeddings and modality labels for projection."""

    if per_modality <= 0:
        raise ValueError("per_modality must be positive")

    count = min(s1.shape[0], s2.shape[0])
    if count == 0:
        raise ValueError("No paired embeddings available for projection")

    take = min(per_modality, count)
    indices = generator.choice(count, size=take, replace=False)
    stacked = torch.cat([s1[indices], s2[indices]], dim=0)
    labels = np.array(["S1"] * take + ["S2"] * take)
    return stacked.cpu().numpy(), labels


def compute_projection(
    data: np.ndarray,
    *,
    method: str,
    random_state: int,
) -> Optional[np.ndarray]:
    if data.shape[0] < 3:
        return None

    if method == "tsne":
        perplexity = min(30, data.shape[0] - 1)
        if perplexity < 1:
            return None
        reducer = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state)
    elif method == "umap":
        if umap is None:
            _LOGGER.warning("UMAP is not installed; skipping projection")
            return None
        reducer = umap.UMAP(n_components=2, random_state=random_state, init="spectral")
    else:
        raise ValueError(f"Unknown projection method: {method}")

    return reducer.fit_transform(data)


# def plot_projection(
def _plot_projection_panel(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    # output_path: Path,
    *,
    title: str,
    ):
# ) -> None:
    # fig, ax = plt.subplots(figsize=(6, 5))
    unique_labels = np.unique(labels)
    handles = []
    for label in unique_labels:
        mask = labels == label
        handle = ax.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.7, s=18)
        handles.append(handle)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    # ax.legend()
    ax.grid(True, alpha=0.2)
    return handles


def plot_projection(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    handles = _plot_projection_panel(ax, coords, labels, title=title)
    if handles:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def _flatten_axes(axes) -> List:
    if hasattr(axes, "flat"):
        return list(axes.flat)
    if isinstance(axes, (list, tuple)):
        flattened: List = []
        for item in axes:
            flattened.extend(_flatten_axes(item))
        return flattened
    return [axes]


def plot_projection_grid(
    panels: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    method: str,
) -> None:
    if not panels:
        _LOGGER.warning("No %s projections to plot", method)
        return

    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_list = _flatten_axes(axes)

    legend_handles: List = []
    legend_labels: List[str] = []

    for idx, (ax, (epoch_label, coords, modalities)) in enumerate(zip(axes_list, panels)):
        handles = _plot_projection_panel(ax, coords, modalities, title=f"{method} — {epoch_label}")
        if idx == 0:
            legend_handles = handles
            legend_labels = [h.get_label() for h in handles]

    for ax in axes_list[len(panels) :]:
        fig.delaxes(ax)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper right", frameon=True)

    fig.suptitle(f"{method} projections across epochs")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def export_metrics(metrics: Sequence[EpochMetrics], output_dir: Path) -> None:
    summary = [m.to_summary_dict() for m in metrics]
    with (output_dir / "metrics_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


def _csv_value(value: Optional[float]):
    if value is None:
        return ""
    value_float = float(value)
    if math.isnan(value_float):
        return ""
    return value_float


def export_vc_metrics_csv(
    metrics: Sequence[EpochMetrics],
    output_dir: Path,
    *,
    cca_top_k: int,
) -> None:
    if not metrics:
        return

    cca_label = f"cca_rho_mean_top_{cca_top_k}" if cca_top_k > 0 else "cca_rho_mean_top_all"
    fieldnames = [
        "epoch_index",
        "label",
        "sample_count",
        "vc_std_min_s1",
        "vc_std_min_s2",
        "vc_std_min_s1_pre",
        "vc_std_min_s2_pre",
        "vc_cov_fro_s1",
        "vc_cov_fro_s2",
        "vc_cov_fro_s1_pre",
        "vc_cov_fro_s2_pre",
        "vc_participation_ratio_s1",
        "vc_participation_ratio_s2",
        "vc_participation_ratio_s1_pre",
        "vc_participation_ratio_s2_pre",
        "corr_participation_ratio_s1",
        "corr_participation_ratio_s2",
        "corr_participation_ratio_s1_pre",
        "corr_participation_ratio_s2_pre",
        "corr_spectral_entropy_s1",
        "corr_spectral_entropy_s2",
        "corr_spectral_entropy_s1_pre",
        "corr_spectral_entropy_s2_pre",
        "cca_rho_max",
        cca_label,
    ]

    with (output_dir / "vc_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "epoch_index": metric.epoch_index,
                    "label": metric.label,
                    "sample_count": metric.sample_count,
                    "vc_std_min_s1": _csv_value(metric.s1_std_min),
                    "vc_std_min_s2": _csv_value(metric.s2_std_min),
                    "vc_std_min_s1_pre": _csv_value(metric.s1_pre_std_min),
                    "vc_std_min_s2_pre": _csv_value(metric.s2_pre_std_min),
                    "vc_cov_fro_s1": _csv_value(metric.s1_cov_fro),
                    "vc_cov_fro_s2": _csv_value(metric.s2_cov_fro),
                    "vc_cov_fro_s1_pre": _csv_value(metric.s1_pre_cov_fro),
                    "vc_cov_fro_s2_pre": _csv_value(metric.s2_pre_cov_fro),
                    "vc_participation_ratio_s1": _csv_value(metric.s1_participation_ratio),
                    "vc_participation_ratio_s2": _csv_value(metric.s2_participation_ratio),
                    "vc_participation_ratio_s1_pre": _csv_value(metric.s1_pre_participation_ratio),
                    "vc_participation_ratio_s2_pre": _csv_value(metric.s2_pre_participation_ratio),
                    "corr_participation_ratio_s1": _csv_value(metric.s1_corr_participation_ratio),
                    "corr_participation_ratio_s2": _csv_value(metric.s2_corr_participation_ratio),
                    "corr_participation_ratio_s1_pre": _csv_value(metric.s1_pre_corr_participation_ratio),
                    "corr_participation_ratio_s2_pre": _csv_value(metric.s2_pre_corr_participation_ratio),
                    "corr_spectral_entropy_s1": _csv_value(metric.s1_corr_spectral_entropy),
                    "corr_spectral_entropy_s2": _csv_value(metric.s2_corr_spectral_entropy),
                    "corr_spectral_entropy_s1_pre": _csv_value(metric.s1_pre_corr_spectral_entropy),
                    "corr_spectral_entropy_s2_pre": _csv_value(metric.s2_pre_corr_spectral_entropy),
                    "cca_rho_max": _csv_value(metric.cca_rho_max),
                    cca_label: _csv_value(metric.cca_rho_topk_mean),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedding collapse diagnostic visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:

              python -m visualizations.ssl4eo.embedding_collapse_diagnostics \
                --checkpoint-root '/home/juro4948/ciip/logs/2025_09_22-16_02_28-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints' \
                --output-dir diagnostics/random_init/9-22-2025-vcregstats \
                --dataset-root /local/ms-data/SSL4EO/ \
                --vc-gamma 1 
            """
            # '' \
        ).strip(),
        # '/home/juro4948/ciip/logs/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints'

        #          /home/juro4948/ciip/logs/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints' \
        # /home/juro4948/ciip/logs/2025_09_20-22_35_45-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Directory containing CIIP checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to store plots and metrics")
    parser.add_argument(
        "--config-name",
        type=str,
        default="prod_default",
        help="Name of the Hydra config to load (without .yaml)",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional path to the Hydra config directory (defaults to ciip/open_clip_train/configs)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset.root value from the config",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=2048,
        help="Number of samples to evaluate per epoch (0 or negative = full dataset)",
    )
    parser.add_argument("--subset-seed", type=int, default=42, help="Random seed for subset sampling")
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default=r"epoch",
        help="Regex used to select checkpoints for evaluation",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Optional cap on the number of checkpoints to process",
    )
    parser.add_argument(
        "--include-init",
        action="store_true",
        help="Include epoch_init checkpoints when present",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run extraction on (defaults to CUDA if available else CPU)",
    )
    parser.add_argument(
        "--skip-final-fc",
        action="store_true",
        help=(
            "Replace the encoders' final projection layers with identity mappings "
            "when extracting embeddings (defaults to keeping the trained heads)."
        ),
    )
    parser.add_argument(
        "--use-orthogonal-mapping",
        action="store_true",
        help=(
            "Apply the optional W.pt orthogonal alignment when present so s*_features_vc "
            "match the rotated training space (defaults to raw encoder outputs)."
        ),
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=2000,
        help="Number of negative pairs to sample for cosine similarity (None = all)",
    )
    parser.add_argument(
        "--vc-gamma",
        type=float,
        default=None,
        help="Override the variance floor γ used when computing VC diagnostics (defaults to config)",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--spectrum-top-k", type=int, default=10, help="Number of leading eigenvalues to plot")
    parser.add_argument(
        "--cca-top-k",
        type=int,
        default=5,
        help="Number of leading canonical correlations to track (0 = all available)",
    )
    parser.add_argument(
        "--linear-probe-csv",
        type=Path,
        default=None,
        help="Optional CSV from output_results_to_csv containing linear probe metrics",
    )
    parser.add_argument(
        "--linear-probe-pattern",
        type=str,
        default=None,
        help="Regex identifying rows that correspond to the embeddings under study",
    )
    parser.add_argument(
        "--linear-probe-k",
        type=float,
        default=None,
        help="Desired k/percent column from the linear probe CSV (defaults to the first column)",
    )
    parser.add_argument(
        "--linear-probe-metric",
        type=str,
        default="accuracy",
        choices=("accuracy", "f1"),
        help="Metric name used in the CSV file",
    )
    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=800,
        help="Number of samples per modality for t-SNE projections (0 = skip)",
    )
    parser.add_argument(
        "--umap-samples",
        type=int,
        default=800,
        help="Number of samples per modality for UMAP projections (0 = skip)",
    )
    parser.add_argument(
        "--cosine-hist-bins",
        type=int,
        default=50,
        help="Number of bins for cosine similarity histograms",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = args.config_path or Path(__file__).resolve().parents[3] / "ciip" / "ciip" / "open_clip_train" / "configs"
    config_file = (config_dir / f"{args.config_name}.yaml").resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Could not find config file '{config_file}'")

    _LOGGER.info("Loading config from %s", config_file)
    config = OmegaConf.load(config_file)
    OmegaConf.set_struct(config, False)

    if args.dataset_root is not None:
        config.dataset.root = str(args.dataset_root)
    else:
        config.dataset.root = str(config.dataset.root)

    config.io.checkpoint_path = str(args.checkpoint_root)
    config.datamodule.distributed = False
    if "horovod" in config.datamodule:
        config.datamodule.horovod = False

    ensure_hydra_original_cwd()
    data = get_data(config)
    train_dataset = data["train"].dataloader.dataset
    subset = build_subset(train_dataset, args.subset_size, args.subset_seed)
    _LOGGER.info(
        "Using %d samples per epoch from dataset root %s",
        len(subset),
        config.dataset.root,
    )

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    config.datamodule.device = device_str

    input_dtype = resolve_input_dtype(config.model.precision)
    if device.type != "cuda" and input_dtype in {torch.float16, torch.bfloat16}:
        input_dtype = torch.float32
    _LOGGER.info("Running extraction on %s with input dtype %s", device, input_dtype)

    autocast_fn = get_autocast(config.model.precision)
    if device.type != "cuda":
        autocast_fn = contextlib.nullcontext

    epochs = collect_epoch_embeddings(
        args.checkpoint_root,
        config,
        subset,
        input_dtype=input_dtype,
        device=device,
        autocast=autocast_fn,
        pattern=args.checkpoint_pattern,
        include_init=args.include_init,
        max_checkpoints=args.max_checkpoints,
        skip_final_fc=args.skip_final_fc,
        use_orthogonal_mapping=args.use_orthogonal_mapping,
    )

    # metrics = compute_epoch_metrics(epochs, negative_samples=args.negative_samples, random_seed=args.random_seed)
    try:
        config_gamma = float(config.loss.vc_gamma)
    except Exception:
        config_gamma = None

    if args.vc_gamma is not None:
        vc_gamma = float(args.vc_gamma)
        gamma_source = "CLI override"
    elif config_gamma is not None:
        vc_gamma = config_gamma
        gamma_source = "config"
    else:
        vc_gamma = 1.0
        gamma_source = "default"

    _LOGGER.info("Using VC γ target of %.4f (%s)", vc_gamma, gamma_source)

    cca_top_k = max(0, int(args.cca_top_k))

    metrics = compute_epoch_metrics(
        epochs,
        negative_samples=args.negative_samples,
        random_seed=args.random_seed,
        cca_top_k=cca_top_k,
    )

    export_metrics(metrics, output_dir)
    export_vc_metrics_csv(metrics, output_dir, cca_top_k=cca_top_k)
    plot_cosine_summary(metrics, output_dir)
    plot_cosine_histograms(metrics, output_dir, bins=args.cosine_hist_bins)
    plot_variance_heatmap(metrics, output_dir, modality="s1")
    plot_variance_heatmap(metrics, output_dir, modality="s2")
    plot_spectrum(metrics, output_dir, modality="s1", top_k=args.spectrum_top_k)
    plot_spectrum(metrics, output_dir, modality="s2", top_k=args.spectrum_top_k)
    plot_vc_timeseries(
        metrics,
        output_dir,
        vc_gamma=vc_gamma,
        spectrum_top_k=args.spectrum_top_k,
        cca_top_k=cca_top_k,
    )
    plot_singular_value_timeseries(
        metrics,
        output_dir,
        top_k=args.spectrum_top_k,
    )
    plot_epoch_dashboards(
        metrics,
        output_dir,
        cosine_bins=args.cosine_hist_bins,
        spectrum_top_k=args.spectrum_top_k,
    )
    plot_epoch_singular_values(
        metrics,
        output_dir,
        top_k=args.spectrum_top_k,
    )
    

    if args.linear_probe_csv:
        linear_probe = parse_linear_probe_csv(
            args.linear_probe_csv,
            model_pattern=args.linear_probe_pattern,
            metric=args.linear_probe_metric,
            k_value=args.linear_probe_k,
        )
        plot_linear_probe_curve(linear_probe, output_dir, metric=args.linear_probe_metric)

    np_gen = np.random.default_rng(args.random_seed)
    if args.tsne_samples > 0:
        tsne_panels: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for epoch in epochs:
            if epoch.s1 is None or epoch.s2 is None:
                continue
            s1_proj = epoch.s1_normalized if epoch.s1_normalized is not None else epoch.s1
            s2_proj = epoch.s2_normalized if epoch.s2_normalized is not None else epoch.s2
            if s1_proj is None or s2_proj is None:
                continue
            try:
                data, labels = sample_for_projection(
                    s1_proj,
                    s2_proj,
                    per_modality=args.tsne_samples,
                    generator=np_gen,
                )
            except ValueError:
                continue
            coords = compute_projection(data, method="tsne", random_state=args.random_seed)
            if coords is None:
                continue
            # plot_projection(coords, labels, output_dir / f"tsne_{epoch.label}.png", title=f"t-SNE — {epoch.label}")
            tsne_panels.append((epoch.label, coords, labels))

        if tsne_panels:
            plot_projection_grid(tsne_panels, output_dir / "tsne_epochs.png", method="t-SNE")
        else:
            _LOGGER.warning("Skipping t-SNE projections (insufficient data)")

    if args.umap_samples > 0:
        if umap is None:
            _LOGGER.warning("UMAP requested but not installed; skipping")
        else:
            umap_panels: List[Tuple[str, np.ndarray, np.ndarray]] = []
            for epoch in epochs:
                if epoch.s1 is None or epoch.s2 is None:
                    continue
                s1_proj = epoch.s1_normalized if epoch.s1_normalized is not None else epoch.s1
                s2_proj = epoch.s2_normalized if epoch.s2_normalized is not None else epoch.s2
                if s1_proj is None or s2_proj is None:
                    continue
                try:
                    data, labels = sample_for_projection(
                        s1_proj,
                        s2_proj,
                        per_modality=args.umap_samples,
                        generator=np_gen,
                    )
                except ValueError:
                    continue
                coords = compute_projection(data, method="umap", random_state=args.random_seed)
                if coords is None:
                    continue
                umap_panels.append((epoch.label, coords, labels))

            if umap_panels:
                plot_projection_grid(umap_panels, output_dir / "umap_epochs.png", method="UMAP")
            else:
                _LOGGER.warning("Skipping UMAP projections (insufficient data)")
    

    _LOGGER.info("Saved plots and metrics to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
