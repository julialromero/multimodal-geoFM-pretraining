#!/usr/bin/env python3
"""Single-epoch embedding diagnostics utilities.

This module intentionally favours clarity over feature completeness.  It exposes
helpers that are consumed by other research scripts (``unified_evaluation.py``
and ``eval_collapse_diagnostics.py``) while providing a compact CLI for
producing diagnostics for a single checkpoint epoch.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ciip.open_clip_train.data import get_data
from ciip.open_clip_train.precision import get_autocast
from ciip.open_clip_train.utils import create_model

try:  # optional dependency
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional in CI
    umap = None

_LOGGER = logging.getLogger("embedding_collapse")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModalityEmbeddings:
    """Embeddings and layer activations for one modality."""

    raw: torch.Tensor
    normalized: torch.Tensor
    layer_activations: Dict[str, torch.Tensor]


@dataclass
class EpochDiagnostics:
    """Diagnostics computed for a single checkpoint epoch."""

    label: str
    epoch: int
    ids: List[str]
    s1: ModalityEmbeddings
    s2: ModalityEmbeddings
    s1_singular_values: np.ndarray
    s2_singular_values: np.ndarray
    s1_layers: List[str]
    s2_layers: List[str]
    s1_within_cka: Optional[np.ndarray]
    s2_within_cka: Optional[np.ndarray]
    cross_cka: Optional[np.ndarray]
    cross_s1_layers: List[str]
    cross_s2_layers: List[str]


# ---------------------------------------------------------------------------
# Compatibility helpers imported by other modules
# ---------------------------------------------------------------------------


def ensure_hydra_original_cwd() -> None:
    """Ensure Hydra provides a deterministic ``get_original_cwd``."""

    try:
        from hydra.core import utils as hydra_utils  # type: ignore

        hydra_utils.get_original_cwd()
    except Exception:
        try:
            from hydra.core import utils as hydra_utils  # type: ignore

            repo_root = Path(__file__).resolve().parents[2]
            default_cwd = repo_root / "ciip" / "open_clip_train"
            hydra_utils.get_original_cwd = lambda: str(default_cwd)
        except Exception:
            _LOGGER.debug("Hydra not available; skipping original cwd override")


def resolve_input_dtype(precision: str) -> torch.dtype:
    """Map a precision string from the config to a torch dtype."""

    precision = precision.lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16", "amp", "half"}:
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Public helper functions (used by other modules)
# ---------------------------------------------------------------------------


def compute_singular_values(tensor: torch.Tensor) -> np.ndarray:
    """Return singular values of ``tensor`` treated as (N, D)."""

    if tensor.dim() != 2:
        tensor = tensor.flatten(start_dim=1)
    if tensor.shape[0] < 2:
        return np.empty(0, dtype=np.float32)
    with torch.no_grad():
        centered = tensor.to(torch.float64)
        centered = centered - centered.mean(dim=0, keepdim=True)
        u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    return s.to(torch.float32).cpu().numpy()


def preprocess_projection_data(
    data: np.ndarray,
    *,
    mode: str,
    random_state: int,
    pca_components: int = 50,
) -> np.ndarray:
    """Prepare data prior to dimensionality reduction."""

    data = np.asarray(data, dtype=np.float64)
    if mode == "zscore":
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        data = (data - mean) / std
    elif mode == "l2":
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        data = data / norms
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    if data.shape[1] > pca_components and min(data.shape[0], data.shape[1]) > 1:
        components = min(pca_components, data.shape[0], data.shape[1])
        if components < data.shape[1]:
            pca = PCA(n_components=components, random_state=random_state)
            data = pca.fit_transform(data)
    return data


def compute_projection(
    data: np.ndarray,
    *,
    method: str,
    random_state: int,
) -> Optional[np.ndarray]:
    """Compute a 2-D embedding using t-SNE or UMAP."""

    if data.shape[0] < 3:
        return None
    if method == "tsne":
        perplexity = min(30, data.shape[0] - 1)
        if perplexity < 1:
            return None
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            random_state=random_state,
            metric="cosine",
        )
    elif method == "umap":
        if umap is None:
            _LOGGER.warning("UMAP requested but not installed")
            return None
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            init="spectral",
            metric="cosine",
        )
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return reducer.fit_transform(data)


def plot_projection(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    """Plot a scatter projection."""

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], label=str(label), alpha=0.7, s=18)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    if labels.size:
        ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Linear CKA utilities
# ---------------------------------------------------------------------------


def _prepare_cka_tensor(tensor: torch.Tensor, take: Optional[int] = None) -> Optional[torch.Tensor]:
    if tensor.dim() != 2:
        tensor = tensor.flatten(start_dim=1)
    if tensor.shape[0] < 2:
        return None
    if take is not None and take > 0:
        tensor = tensor[:take]
    return tensor.to(dtype=torch.float64)


def compute_linear_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> Optional[float]:
    """Compute linear CKA similarity between two activation matrices."""

    count = min(x.shape[0], y.shape[0])
    if count < 2:
        return None
    x = x[:count] - x[:count].mean(dim=0, keepdim=True)
    y = y[:count] - y[:count].mean(dim=0, keepdim=True)
    cov_xy = x.t().matmul(y)
    cov_xx = x.t().matmul(x)
    cov_yy = y.t().matmul(y)
    numerator = torch.linalg.norm(cov_xy, ord="fro") ** 2
    denom = torch.linalg.norm(cov_xx, ord="fro") * torch.linalg.norm(cov_yy, ord="fro")
    denom_value = float(denom.item()) if isinstance(denom, torch.Tensor) else float(denom)
    if denom_value <= eps:
        return None
    return float((numerator / (denom + eps)).item())


def _layer_sort_key(name: str) -> Tuple[int, int, str]:
    numbers = [int(match) for match in re.findall(r"\d+", name)]
    if not numbers:
        return (0, 0, name)
    return (numbers[0], numbers[1] if len(numbers) > 1 else 0, name)


def _order_layers(layer_dict: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(layer_dict.keys(), key=_layer_sort_key)


def compute_within_encoder_cka(layer_features: Dict[str, torch.Tensor]) -> Tuple[List[str], Optional[np.ndarray]]:
    ordered = _order_layers(layer_features)
    prepared: Dict[str, torch.Tensor] = {}
    for name in ordered:
        tensor = _prepare_cka_tensor(layer_features[name])
        if tensor is not None:
            prepared[name] = tensor
    names = [name for name in ordered if name in prepared]
    size = len(names)
    if size == 0:
        return [], None
    matrix = np.full((size, size), np.nan, dtype=np.float32)
    for i, name_i in enumerate(names):
        x = prepared[name_i]
        matrix[i, i] = 1.0
        for j in range(i + 1, size):
            value = compute_linear_cka(x, prepared[names[j]])
            if value is None:
                continue
            matrix[i, j] = matrix[j, i] = np.clip(value, 0.0, 1.0)
    return names, matrix


def compute_cross_encoder_cka(
    s1_layers: Dict[str, torch.Tensor],
    s2_layers: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str], Optional[np.ndarray]]:
    names_s1 = _order_layers(s1_layers)
    names_s2 = _order_layers(s2_layers)
    prepared_s1 = {name: _prepare_cka_tensor(s1_layers[name]) for name in names_s1}
    prepared_s1 = {k: v for k, v in prepared_s1.items() if v is not None}
    prepared_s2 = {name: _prepare_cka_tensor(s2_layers[name]) for name in names_s2}
    prepared_s2 = {k: v for k, v in prepared_s2.items() if v is not None}
    names_1 = [name for name in names_s1 if name in prepared_s1]
    names_2 = [name for name in names_s2 if name in prepared_s2]
    if not names_1 or not names_2:
        return [], [], None
    matrix = np.full((len(names_1), len(names_2)), np.nan, dtype=np.float32)
    for i, name_i in enumerate(names_1):
        for j, name_j in enumerate(names_2):
            value = compute_linear_cka(prepared_s1[name_i], prepared_s2[name_j])
            if value is None:
                continue
            matrix[i, j] = np.clip(value, 0.0, 1.0)
    return names_1, names_2, matrix


# ---------------------------------------------------------------------------
# Model and data helpers
# ---------------------------------------------------------------------------


def _resolve_config(args: argparse.Namespace) -> DictConfig:
    repo_root = Path(__file__).resolve().parents[2]
    default_config_dir = repo_root / "ciip" / "open_clip_train" / "configs"
    config_dir = args.config_path or default_config_dir
    config_file = (Path(config_dir) / f"{args.config_name}.yaml").resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Could not find config file '{config_file}'")
    config = OmegaConf.load(config_file)
    OmegaConf.set_struct(config, False)
    if args.dataset_root is not None:
        config.dataset.root = str(args.dataset_root)
    config.io.checkpoint_path = str(args.checkpoint_root)
    config.datamodule.distributed = False
    if "horovod" in config.datamodule:
        config.datamodule.horovod = False
    return config


# def load_model_from_checkpoint(
#     config: DictConfig,
#     checkpoint_path: Path,
#     *,
#     device: torch.device,
#     input_dtype: torch.dtype,
#     w_path: Optional[Path] = None,  # retained for API compatibility
#     skip_final_fc: bool = False,
#     use_orthogonal_mapping: bool = False,
# ) -> nn.Module:
#     model = create_model(config, device=device)
#     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     state_dict = checkpoint.get("state_dict", checkpoint)
#     cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     missing, unexpected = model.load_state_dict(cleaned, strict=False)
#     allowed_missing = {
#         "encoder_s1.fc.weight",
#         "encoder_s1.fc.bias",
#         "encoder_s2.fc.weight",
#         "encoder_s2.fc.bias",
#     }
#     remaining_missing = {key for key in missing if key not in allowed_missing}
#     if remaining_missing:
#         _LOGGER.warning("Missing keys when loading %s: %s", checkpoint_path.name, sorted(remaining_missing))
#     if unexpected:
#         _LOGGER.warning("Unexpected keys when loading %s: %s", checkpoint_path.name, sorted(unexpected))
#     model = model.to(device=device, dtype=input_dtype, non_blocking=True)
#     if skip_final_fc:
#         for encoder_name in ["encoder_s1", "encoder_s2"]:
#             encoder = getattr(model, encoder_name, None)
#             if encoder is not None and hasattr(encoder, "fc"):
#                 setattr(encoder, "fc", nn.Identity())
#     model.eval()
#     return model


def _unwrap_subset(dataset: torch.utils.data.Dataset) -> Tuple[torch.utils.data.Dataset, Sequence[int]]:
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset, dataset.indices  # type: ignore[return-value]
    return dataset, range(len(dataset))


def _register_layer_hooks(model: nn.Module) -> Tuple[Dict[str, Dict[str, List[torch.Tensor]]], List]:
    layer_caches: Dict[str, Dict[str, List[torch.Tensor]]] = {"s1": {}, "s2": {}}
    handles: List = []

    def _make_layer_hook(key: str, name: str):
        cache = layer_caches[key].setdefault(name, [])

        def hook(module: nn.Module, inputs, output):  # type: ignore[override]
            tensor = output[0] if isinstance(output, (list, tuple)) else output
            if not isinstance(tensor, torch.Tensor):
                return
            tensor = tensor.detach()
            if tensor.dim() > 2:
                tensor = F.adaptive_avg_pool2d(tensor, (1, 1))
            tensor = tensor.flatten(start_dim=1).to(device="cpu", dtype=torch.float32)
            cache.append(tensor)

        return hook

    def _attach_layers(encoder: Optional[nn.Module], key: str) -> None:
        if encoder is None:
            return
        layer_names = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
            "fc",
        ]
        for name in layer_names:
            module = getattr(encoder, name, None)
            if not isinstance(module, nn.Module):
                continue
            try:
                handles.append(module.register_forward_hook(_make_layer_hook(key, name)))
            except Exception:
                continue

    encoder_s1 = getattr(model, "encoder_s1", None)
    if encoder_s1 is not None:
        for attr in ("proj", "projection_head", "fc"):
            module = getattr(encoder_s1, attr, None)
            if isinstance(module, nn.Module):
                break
        _attach_layers(encoder_s1, "s1")

    encoder_s2 = getattr(model, "encoder_s2", None)
    if encoder_s2 is not None:
        for attr in ("proj", "projection_head", "fc"):
            module = getattr(encoder_s2, attr, None)
            if isinstance(module, nn.Module):
                break
        _attach_layers(encoder_s2, "s2")

    return layer_caches, handles


def _encoder_accepts_lorentz(method) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return "lorentz" in signature.parameters


def _run_encoder_method(
    method,
    tensor: torch.Tensor,
    *,
    project_hyperbolic: bool,
    normalize: bool,
) -> torch.Tensor:
    accepts_lorentz = _encoder_accepts_lorentz(method)
    try:
        if accepts_lorentz:
            return method(tensor, lorentz=project_hyperbolic, normalize=normalize)
        return method(tensor, normalize=normalize)
    except TypeError:
        # Some historical encoder signatures take positional arguments only.
        if accepts_lorentz:
            return method(tensor, project_hyperbolic, normalize)
        return method(tensor, normalize)


def extract_embeddings_for_dataset(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    input_dtype: torch.dtype,
    device: torch.device,
    autocast,
) -> Tuple[ModalityEmbeddings, ModalityEmbeddings, List[str]]:
    base_dataset, indices = _unwrap_subset(dataset)
    layer_cache, handles = _register_layer_hooks(model)
    s1_vectors: List[torch.Tensor] = []
    s2_vectors: List[torch.Tensor] = []
    s1_norm_vectors: List[torch.Tensor] = []
    s2_norm_vectors: List[torch.Tensor] = []
    sample_ids: List[str] = []

    def _to_tensor(array) -> torch.Tensor:
        tensor = torch.as_tensor(array)
        if tensor.ndim == 3:
            return tensor
        if tensor.ndim == 4:
            return tensor.squeeze(0)
        raise ValueError("Unsupported sample shape for embedding extraction")

    with torch.no_grad():
        for dataset_idx in indices:
            sample = base_dataset[dataset_idx]
            if isinstance(sample, dict):
                s1_img = sample.get("s1")
                s2_img = sample.get("s2")
                uid = sample.get("uid") or sample.get("id") or sample.get("file_name")
            else:
                s1_img, s2_img = sample  # type: ignore[misc]
                uid = str(dataset_idx)
            if s1_img is None or s2_img is None:
                continue
            s1_tensor = _to_tensor(s1_img).unsqueeze(0).to(device=device, dtype=input_dtype)
            s2_tensor = _to_tensor(s2_img).unsqueeze(0).to(device=device, dtype=input_dtype)
            with autocast():
                output = model(s1_tensor, s2_tensor)
            if isinstance(output, dict):
                s1_raw = output.get("s1_features_vc")
                s2_raw = output.get("s2_features_vc")
                s1_norm = output.get("s1_features")
                s2_norm = output.get("s2_features")
            else:
                s1_raw = s2_raw = None
                s1_norm = s2_norm = None

            encode_s1 = getattr(model, "encode_s1", None)
            encode_s2 = getattr(model, "encode_s2", None)
            if (s1_raw is None or s2_raw is None) and (encode_s1 is None or encode_s2 is None):
                raise AttributeError("Model does not expose encode_s1/encode_s2 methods for embedding extraction")

            if s1_raw is None and encode_s1 is not None:
                with autocast():
                    s1_raw = _run_encoder_method(
                        encode_s1,
                        s1_tensor,
                        project_hyperbolic=False,
                        normalize=False,
                    )
            if s2_raw is None and encode_s2 is not None:
                with autocast():
                    s2_raw = _run_encoder_method(
                        encode_s2,
                        s2_tensor,
                        project_hyperbolic=False,
                        normalize=False,
                    )

            def _select_normalized(
                raw_tensor: torch.Tensor,
                encoder_method,
                input_tensor: torch.Tensor,
            ) -> torch.Tensor:
                if encoder_method is None:
                    return raw_tensor
                try:
                    with autocast():
                        candidate = _run_encoder_method(
                            encoder_method,
                            input_tensor,
                            project_hyperbolic=False,
                            normalize=True,
                        )
                except Exception:  # pragma: no cover - encoder mismatch fallback
                    candidate = None

                if candidate is None:
                    return raw_tensor

                norms = candidate.flatten(start_dim=1).norm(dim=-1)
                if torch.allclose(norms, torch.ones_like(norms), atol=1e-3, rtol=1e-3):
                    return candidate
                return raw_tensor

            if s1_norm is None:
                s1_norm = _select_normalized(s1_raw, encode_s1, s1_tensor)
            if s2_norm is None:
                s2_norm = _select_normalized(s2_raw, encode_s2, s2_tensor)
            s1_vectors.append(s1_raw.squeeze(0).cpu().to(torch.float32))
            s2_vectors.append(s2_raw.squeeze(0).cpu().to(torch.float32))
            s1_norm_vectors.append(s1_norm.squeeze(0).cpu().to(torch.float32))
            s2_norm_vectors.append(s2_norm.squeeze(0).cpu().to(torch.float32))
            sample_ids.append(str(uid))

    for handle in handles:
        handle.remove()

    def _stack(list_tensors: List[torch.Tensor]) -> torch.Tensor:
        if not list_tensors:
            return torch.empty(0, 0)
        return torch.stack(list_tensors, dim=0)

    s1_layers = {name: torch.cat(tensors, dim=0) for name, tensors in layer_cache["s1"].items() if tensors}
    s2_layers = {name: torch.cat(tensors, dim=0) for name, tensors in layer_cache["s2"].items() if tensors}

    s1_embeddings = ModalityEmbeddings(_stack(s1_vectors), _stack(s1_norm_vectors), s1_layers)
    s2_embeddings = ModalityEmbeddings(_stack(s2_vectors), _stack(s2_norm_vectors), s2_layers)
    return s1_embeddings, s2_embeddings, sample_ids


# ---------------------------------------------------------------------------
# Diagnostics computation and plotting
# ---------------------------------------------------------------------------


def _discover_checkpoint(checkpoint_root: Path, epoch: int) -> Path:
    patterns = [
        f"epoch_{epoch}.pt",
        f"epoch_{epoch:02d}.pt",
        f"epoch_{epoch:03d}.pt",
    ]
    for pattern in patterns:
        candidate = checkpoint_root / pattern
        if candidate.exists():
            return candidate
    matches = sorted(checkpoint_root.glob(f"*{epoch}*.pt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint for epoch {epoch} under {checkpoint_root}")
    return matches[0]


def _compute_epoch_diagnostics(
    label: str,
    epoch: int,
    s1: ModalityEmbeddings,
    s2: ModalityEmbeddings,
    ids: List[str],
) -> EpochDiagnostics:
    s1_sv = compute_singular_values(s1.raw)
    s2_sv = compute_singular_values(s2.raw)
    s1_layers, s1_cka = compute_within_encoder_cka(s1.layer_activations)
    s2_layers, s2_cka = compute_within_encoder_cka(s2.layer_activations)
    cross_s1, cross_s2, cross_cka = compute_cross_encoder_cka(s1.layer_activations, s2.layer_activations)
    return EpochDiagnostics(
        label=label,
        epoch=epoch,
        ids=ids,
        s1=s1,
        s2=s2,
        s1_singular_values=s1_sv,
        s2_singular_values=s2_sv,
        s1_layers=s1_layers,
        s2_layers=s2_layers,
        s1_within_cka=s1_cka,
        s2_within_cka=s2_cka,
        cross_cka=cross_cka,
        cross_s1_layers=cross_s1,
        cross_s2_layers=cross_s2,
    )


def _plot_singular_values(ax, values: np.ndarray, *, modality: str, embedding_dim: int) -> None:
    if values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{modality} singular values")
        return
    ax.plot(np.arange(1, values.size + 1), values, marker="o")
    ax.set_xlim(0, 50)
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Singular value")
    ax.set_title(f"{modality} SVD (dim={embedding_dim})")
    ax.grid(True, alpha=0.2)


def _plot_cka(ax, matrix: Optional[np.ndarray], x_labels: List[str], y_labels: List[str], *, title: str, xlabel: str, ylabel: str) -> None:
    if matrix is None or matrix.size == 0 or np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, "CKA unavailable", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Linear CKA")


def plot_epoch_diagnostics(epoch_diag: EpochDiagnostics, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    embedding_dim_s1 = epoch_diag.s1.raw.shape[1] if epoch_diag.s1.raw.ndim == 2 else epoch_diag.s1.raw.view(epoch_diag.s1.raw.shape[0], -1).shape[1]
    embedding_dim_s2 = epoch_diag.s2.raw.shape[1] if epoch_diag.s2.raw.ndim == 2 else epoch_diag.s2.raw.view(epoch_diag.s2.raw.shape[0], -1).shape[1]

    _plot_singular_values(axes[0, 0], epoch_diag.s1_singular_values, modality="S1", embedding_dim=embedding_dim_s1)
    _plot_singular_values(axes[0, 1], epoch_diag.s2_singular_values, modality="S2", embedding_dim=embedding_dim_s2)
    axes[0, 2].axis("off")
    axes[0, 2].text(0.5, 0.5, f"Epoch {epoch_diag.epoch}\n{epoch_diag.label}\nSamples: {len(epoch_diag.ids)}", ha="center", va="center", transform=axes[0, 2].transAxes)

    _plot_cka(
        axes[1, 0],
        epoch_diag.s1_within_cka,
        epoch_diag.s1_layers,
        epoch_diag.s1_layers,
        title="S1 within-encoder",
        xlabel="Layer",
        ylabel="Layer",
    )
    _plot_cka(
        axes[1, 1],
        epoch_diag.s2_within_cka,
        epoch_diag.s2_layers,
        epoch_diag.s2_layers,
        title="S2 within-encoder",
        xlabel="Layer",
        ylabel="Layer",
    )
    _plot_cka(
        axes[1, 2],
        epoch_diag.cross_cka,
        epoch_diag.cross_s2_layers,
        epoch_diag.cross_s1_layers,
        title="Cross encoder",
        xlabel="S2 layer",
        ylabel="S1 layer",
    )

    fig.suptitle(f"Embedding diagnostics — {epoch_diag.label}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "diagnostics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-epoch embedding diagnostics")
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Directory containing checkpoint files")
    parser.add_argument("--epoch", type=int, default=20, help="Epoch number to analyse")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diagnostics outputs")
    parser.add_argument("--config-name", type=str, default="ssl4eo_clip", help="Name of the training config to load")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional directory that stores config files")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override dataset root from config")
    parser.add_argument("--subset-size", type=int, default=2048, help="Number of samples to evaluate (0 = full dataset)")
    parser.add_argument("--subset-seed", type=int, default=0, help="Random seed for subset sampling")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available else cpu)")
    parser.add_argument("--skip-final-fc", action="store_true", help="Replace encoder FC layers with identity")
    return parser


def _build_subset(dataset: torch.utils.data.Dataset, subset_size: int, seed: int) -> torch.utils.data.Dataset:
    total = len(dataset)
    if subset_size <= 0 or subset_size >= total:
        indices = list(range(total))
    else:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(total, size=subset_size, replace=False).tolist())
    return torch.utils.data.Subset(dataset, indices)


def run_single_epoch_diagnostics(args: argparse.Namespace) -> EpochDiagnostics:
    config = _resolve_config(args)
    ensure_hydra_original_cwd()
    data = get_data(config)
    dataset = data["train"].dataloader.dataset
    subset = _build_subset(dataset, args.subset_size, args.subset_seed)
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    # config.datamodule.device = device_str
    # input_dtype = resolve_input_dtype(str(config.model.precision))
    # if device.type != "cuda" and input_dtype in {torch.float16, torch.bfloat16}:
    #     input_dtype = torch.float32
    # autocast_fn = get_autocast(config.model.precision)
    # if device.type != "cuda":
    #     autocast_fn = contextlib.nullcontext
    checkpoint_path = _discover_checkpoint(args.checkpoint_root, args.epoch)
    # model = load_model_from_checkpoint(
    #     config,
    #     checkpoint_path,
    #     device=device,
    #     input_dtype=input_dtype,
    #     skip_final_fc=args.skip_final_fc,
    # )
    s1_embeddings, s2_embeddings, sample_ids = extract_embeddings_for_dataset(
        model,
        subset,
        input_dtype=input_dtype,
        device=device,
        autocast=autocast_fn,
    )
    label = checkpoint_path.stem
    epoch_diag = _compute_epoch_diagnostics(label, args.epoch, s1_embeddings, s2_embeddings, sample_ids)
    epoch_dir = args.output_dir / f"epoch_{args.epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    plot_epoch_diagnostics(epoch_diag, epoch_dir)
    metrics_path = epoch_dir / "metrics.json"
    metrics_payload = {
        "label": epoch_diag.label,
        "epoch": epoch_diag.epoch,
        "num_samples": len(epoch_diag.ids),
        "s1_singular_values": epoch_diag.s1_singular_values.tolist(),
        "s2_singular_values": epoch_diag.s2_singular_values.tolist(),
        "s1_layers": epoch_diag.s1_layers,
        "s2_layers": epoch_diag.s2_layers,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    return epoch_diag


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_single_epoch_diagnostics(args)


if __name__ == "__main__":  # pragma: no cover
    main()
