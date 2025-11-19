#!/usr/bin/env python3
"""Single-epoch embedding diagnostics utilities.

This module intentionally favours clarity over feature completeness.  It exposes
helpers that are consumed by other research scripts (``unified_evaluation.py``
and ``eval_collapse_diagnostics.py``) while providing a compact CLI for
producing diagnostics for a single checkpoint epoch.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ciip.open_clip_train.data import SSL4EODataset
from ciip.open_clip_train.precision import get_autocast
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ciip.evaluation.unified_evaluation import ModelEvalConfig

try:  # optional dependency
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional in CI
    umap = None

_LOGGER = logging.getLogger("embedding_collapse")


class _LayerCaptureController:
    """Control whether forward hooks capture activations."""

    def __init__(self) -> None:
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextlib.contextmanager
    def suspend(self):
        previous = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = previous

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModalityEmbeddings:
    """Embeddings and layer activations for one modality."""

    raw: torch.Tensor
    projected: torch.Tensor
    layer_activations: Dict[str, torch.Tensor]
    # add optional field for backbone embeddings
    backbone: Optional[torch.Tensor] = None


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
# Runtime configuration
# ---------------------------------------------------------------------------


DEFAULT_S2_BANDS: Sequence[str] = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "8A",
    "9",
    "10",
    "11",
    "12",
)


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
    elif mode == "none":
        pass
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
    unique_labels = list(np.unique(labels))
    if "S1" in unique_labels and "S2" in unique_labels:
        unique_labels = [lbl for lbl in unique_labels if lbl != "S1"] + ["S1"]
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=str(label),
            alpha=0.7,
            s=18,
            zorder=3 if label == "S1" else 2,
        )
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
    # print dim before
    # print("Preparing CKA tensor with shape:", tensor.shape)
    if tensor.dim() != 2:
        tensor = tensor.flatten(start_dim=1)
    if tensor.shape[0] < 2:
        return None
    if take is not None and take > 0:
        tensor = tensor[:take]
    # print("Prepared CKA tensor with shape:", tensor.shape)
    return tensor.to(dtype=torch.float64)


def compute_linear_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> Optional[float]:
    """Compute linear CKA similarity between two activation matrices."""

    # print shape of x and y
    # print("Computing CKA between shapes:", x.shape, y.shape)
    # if there are 3 dims:


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


# def _order_layers(layer_dict: Dict[str, torch.Tensor]) -> List[str]:
#     return sorted(layer_dict.keys(), key=_layer_sort_key)


def compute_within_encoder_cka(layer_features: Dict[str, torch.Tensor]) -> Tuple[List[str], Optional[np.ndarray]]:
    ordered = layer_features #_order_layers(layer_features)
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
    names_s1 = s1_layers #_order_layers(s1_layers)
    names_s2 = s2_layers #_order_layers(s2_layers)
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


def _build_ssl4eo_dataset(
    dataset_root: Path,
    *,
    s2_tier: str,
    s2_bands: Sequence[str],
    image_dimension: int,
) -> SSL4EODataset:
    ensure_hydra_original_cwd()
    return SSL4EODataset(
        root=str(dataset_root),
        s2_tier=str(s2_tier),
        s2_bands=list(s2_bands),
        transforms=None,
        target_image_dimension=(image_dimension, image_dimension),
    )


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


def _register_layer_hooks(
    model: nn.Module,
) -> Tuple[Dict[str, Dict[str, List[torch.Tensor]]], List, _LayerCaptureController]:
    layer_caches: Dict[str, Dict[str, List[torch.Tensor]]] = {"s1": {}, "s2": {}}
    handles: List = []
    controller = _LayerCaptureController()

    print('REGISTERING LAYER HOOKS')

    def _make_layer_hook(key: str, name: str):
        cache = layer_caches[key].setdefault(name, [])

        def hook(module: nn.Module, inputs, output):  # type: ignore[override]
            if not controller.enabled:
                return
            tensor = output[0] if isinstance(output, (list, tuple)) else output
            if not isinstance(tensor, torch.Tensor):
                return
            tensor = tensor.detach()
            if tensor.dim() > 2:
                tensor = F.adaptive_avg_pool2d(tensor, (1, 1))
            tensor = tensor.flatten(start_dim=1).to(device="cpu", dtype=torch.float32)
            cache.append(tensor)

        return hook

    # def _attach_layers(encoder: Optional[nn.Module], key: str) -> None:
    #     if encoder is None:
    #         return
    #     layer_names = [
    #         "conv1",
    #         "bn1",
    #         "relu",
    #         "maxpool",
    #         "layer1",
    #         "layer2",
    #         "layer3",
    #         "layer4",
    #         "avgpool",
    #         "fc",
    #     ]
    #     for name in layer_names:
    #         module = getattr(encoder, name, None)
    #         if not isinstance(module, nn.Module):
    #             continue
    #         # try:
    #         handles.append(module.register_forward_hook(_make_layer_hook(key, name)))
    #         # except Exception:
    #         #     continue

    def _attach_layers(encoder: Optional[nn.Module], key: str) -> None:
        if encoder is None:
            return

        # Pick which kinds of modules you want to capture.
        # This covers all useful points: conv/bn/relu/pools/linear + block activations.
        WHITELIST = (
            nn.Conv2d,
            nn.BatchNorm2d,
            nn.ReLU,
            nn.MaxPool2d,
            nn.AdaptiveAvgPool2d,
            nn.AvgPool2d,
            nn.Linear,
        )

        for name, module in encoder.named_modules():
            # skip the top-level module itself (empty name)
            if not name:
                continue

            # Option A: capture every whitelisted submodule (includes block relus like layer3.5.relu)
            if isinstance(module, WHITELIST):
                handles.append(module.register_forward_hook(_make_layer_hook(key, name)))
    
    # if ciip model
    if hasattr(model, "encoder_s1") and hasattr(model, "encoder_s2") and model.encoder_s1 is not None:
        encoder_s1 = getattr(model, "encoder_s1", None)
        assert encoder_s1 is not None, f"Model must have an S1 encoder: {model}"
        if encoder_s1 is not None:
            for attr in ("proj", "projection_head", "fc"):
                module = getattr(encoder_s1, attr, None)
                if isinstance(module, nn.Module):
                    break
            # print("Attaching layers for S1 encoder")
            # print(encoder_s1)
            _attach_layers(encoder_s1, "s1")

        encoder_s2 = getattr(model, "encoder_s2", None)
        assert encoder_s2 is not None, f"Model must have an S2 encoder: {model}"
        if encoder_s2 is not None:
            for attr in ("proj", "projection_head", "fc"):
                module = getattr(encoder_s2, attr, None)
                if isinstance(module, nn.Module):
                    break
            _attach_layers(encoder_s2, "s2")

    # if model is torchgeo resnet, attach layers differently
    elif model.__class__.__name__.lower() == "torchgeoresnetadapter":
        # print("Attaching layers for TorchGeoResNetAdapter")
        encoder = getattr(model, "encoder_s2", None)
        _attach_layers(encoder, "s2")

    # print("Registered layer hooks for embedding extraction.")
    # print("Layers for S1 encoder:", list(layer_caches["s1"].keys()))
    # print("Layers for S2 encoder:", list(layer_caches["s2"].keys()))
    # print("Total hooks registered:", len(handles))
    # print("Hook handles:", handles)

    return layer_caches, handles, controller


def _encoder_accepts_lorentz(method) -> bool:
    # print(type(method))
    inst = getattr(method, "__self__", None)   # -> CiipEvaluationAdapter instance
    # print is lorent
    # print(f'is lorentz={getattr(inst, "is_lorentz", False)}')
    return getattr(inst, "is_lorentz", False)
    



    # # if model class of method is LorentzCIIP, return true
    # model_class = method.__self__.__class__.__name__
    # print("Encoder model class:", model_class)
    # return model_class.lower() == "lorentzciip"


def _run_encoder_method(
    method,
    tensor: torch.Tensor,
    *,
    project_hyperbolic: Optional[bool] = None,
    normalize: Optional[bool] = None,
    post_head: bool = True,
) -> torch.Tensor:
    accepts_lorentz = _encoder_accepts_lorentz(method)
    print(f'Encoder accepts lorentz: {accepts_lorentz}')
    

    # squeeze tensor if it has 5 dim
    tensor = tensor.squeeze(0) if tensor.dim() == 5 else tensor

    if accepts_lorentz and post_head:
        # normalize is not used in LorentzCIIP
        print(f"Running encoder with lorentz={project_hyperbolic}")
        return method(tensor, lorentz=project_hyperbolic, normalize=normalize, post_head=post_head)
    
    # if method does not have noramlize or lorentz arguments, just call it with tensor
    sig = inspect.signature(method)
    # print(sig.parameters)
    if "normalize" not in sig.parameters:
        method.fc = nn.Identity()
        print('Dropping final fc layer for encoder method without normalize argument')

        # # drop fc layer if exists
        # # todo: modify this so it is temporary only within this scope
        # method_fc = getattr(method.__self__, "fc", None)
        # if method_fc is not None and isinstance(method_fc, nn.Identity):
        #     print("Dropping final fc layer for encoder method")
        #     setattr(method.__self__, "fc", nn.Identity())

            

        t = method(tensor)
        print("Embedding shape from encoder backbone:", t.shape)
        assert t.shape[1] == 2048, f"Unexpected embedding shape from encoder: {t.shape}, backbone should be 2048"
        return method(tensor)
    
    # return method(tensor, lorentz=project_hyperbolic, normalize=normalize)
    return method(tensor, normalize=normalize, post_head=post_head)


def extract_embeddings_for_dataset(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    input_dtype: torch.dtype,
    device: torch.device,
    autocast,
) -> Tuple[ModalityEmbeddings, ModalityEmbeddings, List[str]]:
    base_dataset, indices = _unwrap_subset(dataset)
    layer_cache, handles, capture_controller = _register_layer_hooks(model)
    s1_vectors: List[torch.Tensor] = []
    s2_vectors: List[torch.Tensor] = []
    s1_norm_vectors: List[torch.Tensor] = []
    s2_norm_vectors: List[torch.Tensor] = []
    s1_backbone_vectors: List[torch.Tensor] = []
    s2_backbone_vectors: List[torch.Tensor] = []
    sample_ids: List[str] = []

    def _to_tensor(array) -> torch.Tensor:
        tensor = torch.as_tensor(array)
        if tensor.ndim == 3:
            return tensor
        if tensor.ndim == 4:
            return tensor.squeeze(0)
        raise ValueError("Unsupported sample shape for embedding extraction")

    # print len of dataset
    print("Extracting embeddings for num files (64 images per file)", len(indices))
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

            # S1 tensor shape with unsqueeze(0): torch.Size([1, 64, 2, 224, 224])
            # without unsqueeze(0): S1 shape=torch.Size([64, 2, 224, 224])
            s1_tensor = _to_tensor(s1_img).to(device=device, dtype=input_dtype)
            s2_tensor = _to_tensor(s2_img).to(device=device, dtype=input_dtype)

            # print shapes
            print(f"Processing sample UID={uid}: S1 shape={s1_tensor.shape}, S2 shape={s2_tensor.shape}")

            with autocast():
                print('Extractoring raw embeddings')
                if model.encoder_s1 is not None:
                    s1_raw = model.compute_posthead(s1_tensor, modality='s1')

                s2_raw = model.compute_posthead(s2_tensor, modality='s2')
                    # s1_raw = _run_encoder_method(
                    #     model.encode_s1,
                    #     s1_tensor,
                    #     project_hyperbolic=False,
                    #     normalize=False,
                    #     post_head=True
                    # )
                
                # s2_raw = _run_encoder_method(
                #     model.encode_s2,
                #     s2_tensor,
                #     project_hyperbolic=False,
                #     normalize=False,
                #     post_head=True
                #     )
                assert s2_raw is not None, "S2 raw embeddings must not be None"
                assert (model.encoder_s1 is None) or (s1_raw.shape == s2_raw.shape), "S1 and S2 raw embeddings must have the same shape {s1_raw.shape} vs {s2_raw.shape}"


                with capture_controller.suspend():
                
                # if lorentzciip model, project hyperbolic is true
                    print('Extracting norm embeddings')
                    # if _encoder_accepts_lorentz(model.encode_s2):
                    if model.encoder_s1 is not None:
                        s1_norm = model. compute_projected(s1_tensor, modality='s1')

                    s2_norm = model.compute_projected(s2_tensor, modality='s2')
                        #     s1_norm = _run_encoder_method(
                        #         model.encode_s1,
                        #         s1_tensor,
                        #         project_hyperbolic=True,
                        #         normalize=False, # not used in lorentzciip
                        #         post_head=True
                        #     # post_head=True
                        #     )
                        # # print("S1 norm shape:", s1_norm.shape)
                        # s2_norm = _run_encoder_method(
                        #         model.encode_s2,
                        #         s2_tensor,
                        #         project_hyperbolic=True,
                        #         normalize=False, # not used in lorentzciip
                        #         post_head=True
                        #     )
                        # print("S2 norm shape:", s2_norm.shape)
                    assert s2_norm is not None, "S2 norm embeddings must not be None"
                    # # else take L2 norm
                    # else:
                    #     if model.encoder_s1 is not None:
                        #     s1_norm = _run_encoder_method(
                        #         model.encode_s1,
                        #         s1_tensor,
                        #         project_hyperbolic=False,
                        #         normalize=True,
                        #         post_head=True
                        #     )
                        # s2_norm = _run_encoder_method(
                        #     model.encode_s2,
                        #     s2_tensor,
                        #     project_hyperbolic=False,
                        #     normalize=True,
                        #     post_head=True
                        # )
                        # assert s2_norm is not None, "S2 norm embeddings must not be None"

                    # print("S2 norm shape:", s2_norm.shape)
                    
                
                    # now extract backbone embeddings 
                    print('Extracting backbone embeddings')

                    if model.encoder_s1 is not None:
                        s1_backbone = model.compute_backbone(s1_tensor, modality='s1')
                    s2_backbone = model.compute_backbone(s2_tensor, modality='s2')
                    #         s1_backbone = _run_encoder_method(
                    #             model.encoder_s1,
                    #             s1_tensor,
                    #             project_hyperbolic=False,
                    #             normalize=False,
                    #             post_head=False

                    #         )
                    # s2_backbone = _run_encoder_method(
                    #     model.encoder_s2,
                    #     s2_tensor,
                    #     project_hyperbolic=False,
                    #     normalize=False,
                    #     post_head=False
                    # )
                    assert s2_backbone is not None, "S2 norm embeddings must not be None"
                    # assert thatt there are no non finite values in s2_backbone
                    # print(s2_backbone)
                    # count the number of non finite values in s2_backbone
                    num_non_finite = torch.sum(~torch.isfinite(s2_backbone)).item()
                    print(f"Number of non-finite values in S2 backbone embeddings: {num_non_finite}")
                    # how many rows in s2_backbone have non finite values
                    num_rows_non_finite = torch.sum(~torch.isfinite(s2_backbone).any(dim=1)).item()
                    print(f"Number of rows with non-finite values in S2 backbone embeddings: {num_rows_non_finite}")
                    # total number of rows
                    print(f"shape of S2 backbone embeddings: {s2_backbone.shape}")
                    assert torch.isfinite(s2_backbone).all(), "S2 backbone embeddings contain non-finite values"

                    # print("S2 backbone shape:", s2_backbone.shape)
                    # print("S1 backbone shape:", s1_backbone.shape)
                    # raise NotImplementedError("Stop here for debugging")



            for handle in handles:
                handle.remove()

            if model.encoder_s1 is not None:
                if s1_raw.squeeze(0).dim() != 2 or s2_raw.squeeze(0).dim() != 2 or s1_norm.squeeze(0).dim() != 2 or s2_norm.squeeze(0).dim() != 2:
                    raise ValueError("Extracted embeddings must be 2D after squeezing")
                s1_vectors.append(s1_raw.squeeze(0).cpu().to(torch.float32))
                s2_vectors.append(s2_raw.squeeze(0).cpu().to(torch.float32))
                s1_norm_vectors.append(s1_norm.squeeze(0).cpu().to(torch.float32))
                s2_norm_vectors.append(s2_norm.squeeze(0).cpu().to(torch.float32))
                s1_backbone_vectors.append(s1_backbone.squeeze(0).cpu().to(torch.float32))
                s2_backbone_vectors.append(s2_backbone.squeeze(0).cpu().to(torch.float32))

            else:
                if s2_raw.squeeze(0).dim() != 2 or s2_norm.squeeze(0).dim() != 2:
                    raise ValueError("Extracted embeddings must be 2D after squeezing")
                s2_vectors.append(s2_raw.squeeze(0).cpu().to(torch.float32))
                s2_norm_vectors.append(s2_norm.squeeze(0).cpu().to(torch.float32))
                s2_backbone_vectors.append(s2_backbone.squeeze(0).cpu().to(torch.float32))
            sample_ids.append(str(uid))

    

    def _stack(list_tensors: List[torch.Tensor]) -> torch.Tensor:
        if not list_tensors:
            raise ValueError("No tensors to stack")
        # if there are 2 dims, stack on first dim
        if list_tensors[0].dim() == 2:
            # if the first dim is 64
            assert list_tensors[0].shape[0] == 64, "Expected first dimension to be 64"
            return torch.cat(list_tensors, dim=0)
        return torch.stack(list_tensors, dim=0)
    
    # print(layer_cache["s1"].items())
    # print(layer_cache["s2"].items())

    s1_layers = {name: torch.cat(tensors, dim=0) for name, tensors in layer_cache["s1"].items() if tensors}
    s2_layers = {name: torch.cat(tensors, dim=0) for name, tensors in layer_cache["s2"].items() if tensors}

    if model.encoder_s1 is not None:
        s1_embeddings = ModalityEmbeddings(_stack(s1_vectors), _stack(s1_norm_vectors), s1_layers, backbone=_stack(s1_backbone_vectors))
    else:
        s1_embeddings = None
    s2_embeddings = ModalityEmbeddings(_stack(s2_vectors), _stack(s2_norm_vectors), s2_layers, backbone=_stack(s2_backbone_vectors))
    return s1_embeddings, s2_embeddings, sample_ids


# ---------------------------------------------------------------------------
# Diagnostics computation and plotting
# ---------------------------------------------------------------------------


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


def _plot_singular_values(ax, values: np.ndarray, *, modality: str, embedding_dim: int, label: Optional[str]) -> None:
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
    ax.set_title(f"{modality} SVD (dim={embedding_dim}) {label}")
    ax.grid(True, alpha=0.2)


def _plot_cka(ax, matrix: Optional[np.ndarray], x_labels: List[str], y_labels: List[str], *, title: str, xlabel: str, ylabel: str) -> None:
    if matrix is None or matrix.size == 0 or np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, "CKA unavailable", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis", origin='lower')
    ax.set_title(title)
    # assert contents of
    assert [a == b for (a, b) in zip(x_labels, y_labels)], f"{x_labels} \n\n {y_labels}"
    
    ax.set_xlabel('Layer #')
    ax.set_ylabel('Layer #')

    n = len(x_labels)
    label_nums = list(range(n))
    downsampled = label_nums[::10] if n > 0 else []

    ax.set_xticks(downsampled)
    ax.set_xticklabels(downsampled, rotation=45, ha="right")

    ax.set_yticks(downsampled)
    ax.set_yticklabels(downsampled)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Linear CKA")


def plot_epoch_diagnostics_s2only(epoch_diag: EpochDiagnostics, output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2x2 layout: [S2 singular values | summary], [S2 within-CKA | empty]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Embedding dim for S2
    if epoch_diag.s2.raw.ndim == 2:
        embedding_dim_s2 = epoch_diag.s2.raw.shape[1]
    else:
        embedding_dim_s2 = epoch_diag.s2.raw.view(epoch_diag.s2.raw.shape[0], -1).shape[1]

    # Top-left: S2 singular values
    _plot_singular_values(
        axes[0, 0],
        epoch_diag.s2_singular_values,
        modality="S2",
        embedding_dim=embedding_dim_s2,
        label='posthead-raw'
    )

    # Top-right: summary text
    axes[0, 1].axis("off")
    axes[0, 1].text(
        0.5,
        0.5,
        f"{label}:\nEpoch {epoch_diag.epoch}\n"
        f"Samples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}",
        ha="center",
        va="center",
        transform=axes[0, 1].transAxes,
    )

    # Bottom-left: S2 within-encoder CKA
    _plot_cka(
        axes[1, 0],
        epoch_diag.s2_within_cka,
        epoch_diag.s2_layers,
        epoch_diag.s2_layers,
        title="S2 within-encoder",
        xlabel="Layer",
        ylabel="Layer",
    )

    # Bottom-right: empty
    axes[1, 1].axis("off")

    fig.suptitle(f"Embedding diagnostics — {epoch_diag.label}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / "embedding_diagnostics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def plot_epoch_diagnostics(epoch_diag: EpochDiagnostics, output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    embedding_dim_s1 = epoch_diag.s1.raw.shape[1] if epoch_diag.s1.raw.ndim == 2 else epoch_diag.s1.raw.view(epoch_diag.s1.raw.shape[0], -1).shape[1]
    embedding_dim_s2 = epoch_diag.s2.raw.shape[1] if epoch_diag.s2.raw.ndim == 2 else epoch_diag.s2.raw.view(epoch_diag.s2.raw.shape[0], -1).shape[1]


    _plot_singular_values(axes[0, 0], epoch_diag.s1_singular_values, modality="S1", embedding_dim=embedding_dim_s1, label='posthead-raw')
    _plot_singular_values(axes[0, 1], epoch_diag.s2_singular_values, modality="S2", embedding_dim=embedding_dim_s2, label='posthead-raw')
    axes[0, 2].axis("off")
    axes[0, 2].text(0.5, 0.5, f"{label}:/nEpoch {epoch_diag.epoch}\n{epoch_diag.label}\nSamples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}", ha="center", va="center", transform=axes[0, 2].transAxes)

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
    output_path = output_dir / "embedding_diagnostics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _build_subset(dataset: torch.utils.data.Dataset, subset_size: int, seed: int) -> torch.utils.data.Dataset:
    total = len(dataset)
    if subset_size <= 0 or subset_size >= total:
        indices = list(range(total))
    else:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(total, size=subset_size, replace=False).tolist())
    return torch.utils.data.Subset(dataset, indices)


# def run_single_epoch_diagnostics(args: argparse.Namespace) -> EpochDiagnostics:
#     config = _resolve_config(args)
#     ensure_hydra_original_cwd()
#     data = get_data(config)
#     dataset = data["train"].dataloader.dataset
#     subset = _build_subset(dataset, args.subset_size, args.subset_seed)
#     device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device(device_str)
#     # config.datamodule.device = device_str
#     # input_dtype = resolve_input_dtype(str(config.model.precision))
#     # if device.type != "cuda" and input_dtype in {torch.float16, torch.bfloat16}:
#     #     input_dtype = torch.float32
#     # autocast_fn = get_autocast(config.model.precision)
#     # if device.type != "cuda":
#     #     autocast_fn = contextlib.nullcontext
#     checkpoint_path = _discover_checkpoint(args.checkpoint_root, args.epoch)
#     # model = load_model_from_checkpoint(
#     #     config,
#     #     checkpoint_path,
#     #     device=device,
#     #     input_dtype=input_dtype,
#     #     skip_final_fc=args.skip_final_fc,
#     # )
#     autocast_fn = (lambda: torch.autocast("cuda"))
#     s1_embeddings, s2_embeddings, sample_ids = extract_embeddings_for_dataset(
#         model,
#         subset,
#         input_dtype=input_dtype,
#         device=device_obj,
#         autocast=autocast_fn,
#     )

#     label = checkpoint_path.stem
#     resolved_epoch = epoch if epoch is not None else _infer_epoch_from_checkpoint(checkpoint_path)
#     diagnostics_epoch = resolved_epoch if resolved_epoch is not None else -1

#     epoch_diag = _compute_epoch_diagnostics(label, diagnostics_epoch, s1_embeddings, s2_embeddings, sample_ids)

#     output_dir = config.output_dir.expanduser()
#     if resolved_epoch is not None and resolved_epoch >= 0:
#         epoch_dir = output_dir / f"epoch_{resolved_epoch:04d}"
#     else:
#         epoch_dir = output_dir / label
#     epoch_dir.mkdir(parents=True, exist_ok=True)

#     plot_epoch_diagnostics(epoch_diag, epoch_dir)

#     metrics_path = epoch_dir / "metrics.json"
#     metrics_payload = {
#         "label": epoch_diag.label,
#         "epoch": epoch_diag.epoch,
#         "num_samples": len(epoch_diag.ids),
#         "s1_singular_values": epoch_diag.s1_singular_values.tolist(),
#         "s2_singular_values": epoch_diag.s2_singular_values.tolist(),
#         "s1_layers": epoch_diag.s1_layers,
#         "s2_layers": epoch_diag.s2_layers,
#     }
#     with metrics_path.open("w", encoding="utf-8") as handle:
#         json.dump(metrics_payload, handle, indent=2)

#     return epoch_diag


__all__ = ["EpochDiagnostics", "run_single_epoch_diagnostics"]


if __name__ == "__main__":  # pragma: no cover
    from pathlib import Path
    from ciip.evaluation.unified_evaluation import ModelEvalConfig, run_full_evaluation

    model_root = '/local/ms-data/SSL4EO/model/'
    model_path = '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/'
    checkpoint_root = Path(model_root) / model_path / "checkpoints"
    output_dir = Path("diagnostics/output")

    cfg = ModelEvalConfig(
        checkpoint=Path(f"{checkpoint_root}/epoch_10.pt"),
        eurosat_root=Path("/local/ms-data/EuroSAT/"),
        neuco_root=Path("/local/ms-data/SSL4EO-S12-downstream/data"),
        output_dir=Path("/home/juro4948/ciip/diagnostics/unified_eval/curv_init_1")
    )

    run_single_epoch_diagnostics(cfg, epoch=10)
