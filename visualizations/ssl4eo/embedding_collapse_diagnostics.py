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

    raw: Optional[torch.Tensor]
    projected: Optional[torch.Tensor]
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
    # # print dim before
    # print("Preparing CKA tensor with shape:", tensor.shape)
    # if tensor.dim() != 2:
    #     tensor = tensor.flatten(start_dim=1)
    # if tensor.shape[0] < 2:
    #     return None
    # if take is not None and take > 0:
    #     tensor = tensor[:take]
    # print("Prepared CKA tensor with shape:", tensor.shape)
    return tensor.to(dtype=torch.float32)


# def compute_linear_cka(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     eps: float = 1e-12,
# ) -> Optional[float]:
#     """Compute linear CKA similarity between two activation matrices."""

#     # print shape of x and y
#     # print("Computing CKA between shapes:", x.shape, y.shape)
#     # if there are 3 dims:


#     count = min(x.shape[0], y.shape[0])
#     if count < 2:
#         return None
#     x = x[:count] - x[:count].mean(dim=0, keepdim=True)
#     y = y[:count] - y[:count].mean(dim=0, keepdim=True)
#     cov_xy = x.t().matmul(y)
#     cov_xx = x.t().matmul(x)
#     cov_yy = y.t().matmul(y)
#     numerator = torch.linalg.norm(cov_xy, ord="fro") ** 2
#     denom = torch.linalg.norm(cov_xx, ord="fro") * torch.linalg.norm(cov_yy, ord="fro")
#     denom_value = float(denom.item()) if isinstance(denom, torch.Tensor) else float(denom)
#     if denom_value <= eps:
#         return None
#     return float((numerator / (denom + eps)).item())


def _layer_sort_key(name: str) -> Tuple[int, int, str]:
    numbers = [int(match) for match in re.findall(r"\d+", name)]
    if not numbers:
        return (0, 0, name)
    return (numbers[0], numbers[1] if len(numbers) > 1 else 0, name)


# def _order_layers(layer_dict: Dict[str, torch.Tensor]) -> List[str]:
#     return sorted(layer_dict.keys(), key=_layer_sort_key)


def compute_within_encoder_cka(layer_features: Dict[str, torch.Tensor]) -> Tuple[List[str], Optional[np.ndarray]]:
    import CKA
    cuda_cka = CKA.CudaCKA(device='cpu')

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
            # assert that dtype matches
            assert x.dtype == prepared[names[j]].dtype, "Mismatched dtypes in CKA computation: {} vs {}".format(x.dtype, prepared[names[j]].dtype)  
            value = cuda_cka.linear_CKA(x, prepared[names[j]])
            if value is None:
                raise ValueError("CKA computation returned None")
            matrix[i, j] = matrix[j, i] = np.clip(value.cpu(), 0.0, 1.0)
    return names, matrix


def compute_cross_encoder_cka(
    s1_layers: Dict[str, torch.Tensor],
    s2_layers: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str], Optional[np.ndarray]]:
    import CKA
    cuda_cka = CKA.CudaCKA(device='cpu')
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
            value = cuda_cka.linear_CKA(prepared_s1[name_i], prepared_s2[name_j])
            if value is None:
                raise ValueError("CKA computation returned None")
            matrix[i, j] = np.clip(value.cpu(), 0.0, 1.0)
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

        def hook(module, inputs, output):
            if not controller.enabled:
                return
            tensor = output[0] if isinstance(output, (list, tuple)) else output
            if not isinstance(tensor, torch.Tensor):
                return
            tensor = tensor.detach()
            tensor = tensor.flatten(start_dim=1)
            tensor = tensor.to(dtype=torch.float32)
            cache.append(tensor)

        return hook

    def _make_block_hook(layer_key: str, layer_name: str):
        base = _make_layer_hook(layer_key, layer_name)

        def block_hook(module, inputs, output):
            # Capture block output
            return base(module, inputs, output)

        return block_hook

    def _attach_layers_resnet(encoder: nn.Module, key: str):
        for name, module in encoder.named_modules():
            # Treat any 'Bottleneck' with a bn3 as a bottleneck block
            if module.__class__.__name__ == "Bottleneck" and hasattr(module, "bn3"):
                bn3_name = f"{name}.bn3"
                handles.append(
                    module.bn3.register_forward_hook(
                        _make_layer_hook(key, bn3_name)
                    )
                )

                block_out_name = f"{name}.block_out"
                handles.append(
                    module.register_forward_hook(
                        _make_block_hook(key, block_out_name)
                    )
                )


    def _attach_layers_croma_vit(vit: nn.Module, key: str):
        """
        Collect activations at the 3 relevant points per sublayer:
        1. after LayerNorm
        2. after SelfAttention or FFN core
        3. after residual addition
        """

        transformer = vit.transformer

        for layer_idx, (attn, ffn) in enumerate(transformer.layers):

            # ===== Attention sublayer =====

            # 1. LN
            handles.append(
                attn.input_norm.register_forward_hook(
                    _make_layer_hook(key, f"layer{layer_idx}.attn.ln")
                )
            )

            # 2. Core attention output (after projection)
            handles.append(
                attn.to_out.register_forward_hook(
                    _make_layer_hook(key, f"layer{layer_idx}.attn.core")
                )
            )

            # 3. Residual after attention
            def make_attn_residual_hook(idx):
                def attn_res_hook(module, inputs, output):
                    # output is the tensor returned by self_attn(...)+x inside BaseTransformer
                    return _make_layer_hook(key, f"layer{idx}.attn.residual")(module, inputs, output)
                return attn_res_hook

            handles.append(
                attn.register_forward_hook(make_attn_residual_hook(layer_idx))
            )

            # ===== FFN sublayer =====

            # 1. LN before FFN
            handles.append(
                ffn.input_norm.register_forward_hook(
                    _make_layer_hook(key, f"layer{layer_idx}.ffn.ln")
                )
            )

            # 2. Core MLP output
            handles.append(
                ffn.net.register_forward_hook(
                    _make_layer_hook(key, f"layer{layer_idx}.ffn.core")
                )
            )

            # 3. Residual after FFN
            def make_ffn_residual_hook(idx):
                def ffn_res_hook(module, inputs, output):
                    return _make_layer_hook(key, f"layer{idx}.ffn.residual")(module, inputs, output)
                return ffn_res_hook

            handles.append(
                ffn.register_forward_hook(make_ffn_residual_hook(layer_idx))
            )

        
    real_model = model
    encoder_s1 = getattr(real_model, "encoder_s1", None)
    encoder_s2 = getattr(real_model, "encoder_s2")


    print('ENCODER S1:', encoder_s1)
    print('ENCODER S2:', encoder_s2)

    print('ENCODER S1 TYPE:', type(encoder_s1))
    print('ENCODER S2 TYPE:', type(encoder_s2))

    # if the encoders are resnets
    if encoder_s1 is not None:
        # if type is resnet
        if 'ResNet' in str(type(encoder_s1)):
            print('Attaching ResNet S1 layers')
            _attach_layers_resnet(encoder_s1, "s1")
        elif 'ViT' in str(type(encoder_s1)):
            print('Attaching CromaViT S1 layers')
            _attach_layers_croma_vit(encoder_s1, "s1")
        else:
            raise NotImplementedError(f"Layer hooks not implemented for S1 encoder type: {type(encoder_s1)}")
    if encoder_s2 is not None:
        if 'ResNet' in str(type(encoder_s2)):
            print('Attaching ResNet S2 layers')
            _attach_layers_resnet(encoder_s2, "s2")
        elif 'ViT' in str(type(encoder_s2)):
            print('Attaching CromaViT S2 layers')
            _attach_layers_croma_vit(encoder_s2, "s2")
        else:
            raise NotImplementedError(f"Layer hooks not implemented for S2 encoder type: {type(encoder_s2)}")
            

    # if encoder_s2 is not None:
    #     _attach_layers(encoder_s2, "s2")


    # raise NotImplementedError("Debugging.")

    print("Registered layer hooks for embedding extraction.")
    print("Layers for S1 encoder:", list(layer_caches["s1"].keys()))
    print("Layers for S2 encoder:", list(layer_caches["s2"].keys()))
    print("Total hooks registered:", len(handles))
    print("Hook handles:", handles)

    return layer_caches, handles, controller


def _encoder_accepts_lorentz(method) -> bool:
    inst = getattr(method, "__self__", None)  
    return getattr(inst, "is_lorentz", False)
    


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

    has_posthead = hasattr(model, "compute_posthead") and inspect.ismethod(
        getattr(model, "compute_posthead")
    )
    has_projected = hasattr(model, "compute_projected") and inspect.ismethod(
        getattr(model, "compute_projected")
    )

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
        # Keep hooks active across the entire extraction loop so CKA receives
        # activations from every sample before we clean them up.
        try:
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
                # print(f"Processing sample UID={uid}: S1 shape={s1_tensor.shape}, S2 shape={s2_tensor.shape}")

                ctx = (
                        torch.cuda.amp.autocast(enabled=False)
                        if s2_tensor.is_cuda
                        else contextlib.nullcontext()
                        )

                s1_raw = None
                s2_raw = None
                s1_norm = None
                s2_norm = None
                s1_backbone = None
                s2_backbone = None

                with ctx:
                    # print(model)
                    assert has_posthead == has_projected, "Model must have both posthead and projected methods or re-implement."

                    if has_posthead:
                        # print('Extracting raw embeddings')
                        if model.encoder_s1 is not None:
                            s1_raw = model.compute_posthead(s1_tensor, modality='s1')
                        s2_raw = model.compute_posthead(s2_tensor, modality='s2')

                        assert s2_raw is not None, "S2 raw embedding extraction returned None"


                        with capture_controller.suspend():
                            # now extract backbone embeddings
                            # print('Extracting backbone embeddings')
                            if model.encoder_s1 is not None:
                                s1_backbone = model.compute_backbone(
                                    s1_tensor.float(), modality='s1'
                                )
                            s2_backbone = model.compute_backbone(
                            s2_tensor.float(), modality='s2')


                            # print('Extracting norm embeddings')
                            if model.encoder_s1 is not None:
                                s1_norm = model.compute_projected(s1_tensor, modality='s1')

                            s2_norm = model.compute_projected(s2_tensor, modality='s2')


                    else:
                        # print('Extracting backbone embeddings')
                        if model.encoder_s1 is not None:
                            s1_backbone = model.compute_backbone(
                                s1_tensor.float(), modality='s1'
                            )
                            assert s1_backbone is not None, "S1 backbone extraction returned None"
                        s2_backbone = model.compute_backbone(
                            s2_tensor.float(), modality='s2')



                if s1_raw is not None:
                    if s1_raw.squeeze(0).dim() != 2:
                        raise ValueError("Extracted S1 embeddings must be 2D after squeezing")
                    s1_vectors.append(s1_raw.squeeze(0).cpu().to(torch.float32))
                if s1_norm is not None:
                    if s1_norm.squeeze(0).dim() != 2:
                        raise ValueError("Extracted normalized S1 embeddings must be 2D after squeezing")
                    s1_norm_vectors.append(s1_norm.squeeze(0).cpu().to(torch.float32))

                if s2_raw is not None:
                    if s2_raw.squeeze(0).dim() != 2:
                        raise ValueError("Extracted S2 embeddings must be 2D after squeezing")
                    s2_vectors.append(s2_raw.squeeze(0).cpu().to(torch.float32))
                if s2_norm is not None:
                    if s2_norm.squeeze(0).dim() != 2:
                        raise ValueError("Extracted normalized S2 embeddings must be 2D after squeezing")
                    s2_norm_vectors.append(s2_norm.squeeze(0).cpu().to(torch.float32))

                if model.encoder_s1 is not None and s1_backbone is not None:
                    s1_backbone_vectors.append(s1_backbone.squeeze(0).cpu().to(torch.float32))
                if s2_backbone is not None:
                    s2_backbone_vectors.append(s2_backbone.squeeze(0).cpu().to(torch.float32))
                sample_ids.append(str(uid))
        finally:
            for handle in handles:
                handle.remove()

    

    def _stack(list_tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not list_tensors:
            return None
        # if there are 2 dims, stack on first dim
        if list_tensors[0].dim() == 2:
            # if the first dim is 64
            assert list_tensors[0].shape[0] == 64, "Expected first dimension to be 64"
            return torch.cat(list_tensors, dim=0)
        return torch.stack(list_tensors, dim=0)
    
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
    s1_sv = compute_singular_values(s1.raw) if s1.raw is not None else np.empty(0, dtype=np.float32)
    s2_sv = compute_singular_values(s2.raw) if s2.raw is not None else np.empty(0, dtype=np.float32)
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
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="magma", origin='lower')
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


def plot_epoch_diagnostics_s2only(epoch_diag: EpochDiagnostics, output_dir: Path, label: str, plot_even_odd=True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    print("Total S2 layers:", len(epoch_diag.s2_layers))
    # print("Odd idx:", s2_odd_idx)
    # print("Even idx:", s2_even_idx)
    # print("Odd CKA submatrix shape:",
    #     epoch_diag.s2_within_cka[np.ix_(s2_odd_idx, s2_odd_idx)].shape)
    # print("Even CKA submatrix shape:",
    #     epoch_diag.s2_within_cka[np.ix_(s2_even_idx, s2_even_idx)].shape)
    

    # Left: S2 within-encoder CKA
    _plot_cka(
        axes[0],
        epoch_diag.s2_within_cka,
        epoch_diag.s2_layers,
        epoch_diag.s2_layers,
        title="S2 within-encoder",
        xlabel="Layer",
        ylabel="Layer",
    )

    if plot_even_odd:
        odd_idx, even_idx = _split_odd_even_indices(epoch_diag.s2_layers)
        # print("S2 odd layers:", len(odd_idx))
        # print("S2 even layers:", len(even_idx))
        # print(epoch_diag.s2_layers.shape)

        # check if 
        print("Full CKA shape:", epoch_diag.s2_within_cka.shape)
        print("Odd submatrix shape:",
            epoch_diag.s2_within_cka[np.ix_(odd_idx, odd_idx)].shape)
        print("Even submatrix shape:",
            epoch_diag.s2_within_cka[np.ix_(even_idx, even_idx)].shape)
        
        # check if the odd and even submatrices are identical
        if epoch_diag.s2_within_cka is not None:
            odd_submatrix = epoch_diag.s2_within_cka[np.ix_(odd_idx, odd_idx)]
            even_submatrix = epoch_diag.s2_within_cka[np.ix_(even_idx, even_idx)]
            print("Odd submatrix:\n", odd_submatrix)
            print("Even submatrix:\n", even_submatrix)
            # asseert they are not identical
            if np.array_equal(odd_submatrix, even_submatrix):
                print("Warning: Odd and even submatrices are identical!")


        _plot_cka_subset(
            axes[1],
            epoch_diag.s2_within_cka,
            epoch_diag.s2_layers,
            odd_idx,
            title="S2 within-encoder (odd layers)",
        )

        _plot_cka_subset(
            axes[2],
            epoch_diag.s2_within_cka,
            epoch_diag.s2_layers,
            even_idx,
            title="S2 within-encoder (even layers)",
        )
    else:
        axes[1].axis("off")
        axes[2].axis("off")

    for ax in axes[:3]:      # last one is the text box
        ax.set_aspect('equal', adjustable='box')


    # Right: summary text
    axes[3].axis("off")
    axes[3].text(0.5, 0.5,
        f"{label}:\nEpoch {epoch_diag.epoch}\n"
        f"Samples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}",
        ha="center",
        va="center",
        transform=axes[3].transAxes,  # ← BUG: Using axes[1] instead of axes[3]
        fontsize=14,
    )

    fig.suptitle(f"S2 Embedding diagnostics — {epoch_diag.label}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = output_dir / "embedding_diagnostics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def _split_vit_hook_indices(layer_names: list[str]) -> dict[str, list[int]]:
    """
    Group CROMA ViT layer names into 3 hook types:
      - 'ln'        : *input_norm* of attn or ffn
      - 'core'      : self-attn or FFN output (before residual add)
      - 'residual'  : post-residual sublayer output

    Assumes names like:
      layer0.attn.ln, layer0.ffn.ln
      layer0.attn.core, layer0.ffn.core
      layer0.attn.residual, layer0.ffn.residual
    """
    groups = {"ln": [], "core": [], "residual": []}

    for i, name in enumerate(layer_names):
        if name.endswith(".ln") or ".ln." in name:
            groups["ln"].append(i)
        elif name.endswith(".core") or ".core." in name:
            groups["core"].append(i)
        elif name.endswith(".residual") or ".residual." in name:
            groups["residual"].append(i)

    return groups



def plot_epoch_diagnostics_transformer(
    epoch_diag: EpochDiagnostics, output_dir: Path, label: str
) -> Path:
    """
    Plot CKA diagnostics for CROMA-style Transformer encoders.

    Layout (2x4):
      Row 0 (S1): [all] [LN hooks] [core hooks] [residual hooks]
      Row 1 (S2): [all] [LN hooks] [core hooks] [residual hooks]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    print(
        f"Plotting Transformer CKA matrices:\n"
        f"  S1 layers={epoch_diag.s1_layers}\n"
        f"  S2 layers={epoch_diag.s2_layers}"
    )

    # === S1 ENCODER ROW ===

    # [0,0]: S1 full (all hooks)
    _plot_cka(
        axes[0, 0],
        epoch_diag.s1_within_cka,
        epoch_diag.s1_layers,
        epoch_diag.s1_layers,
        title="S1 within-encoder (all hooks)",
        xlabel="Layer",
        ylabel="Layer",
    )

    if epoch_diag.s1_within_cka is not None and epoch_diag.s1_layers:
        s1_groups = _split_vit_hook_indices(epoch_diag.s1_layers)

        # [0,1]: LN hooks
        _plot_cka_subset(
            axes[0, 1],
            epoch_diag.s1_within_cka,
            epoch_diag.s1_layers,
            s1_groups["ln"],
            title="S1 within-encoder (LN hooks)",
        )

        # [0,2]: core hooks (attn/FFN)
        _plot_cka_subset(
            axes[0, 2],
            epoch_diag.s1_within_cka,
            epoch_diag.s1_layers,
            s1_groups["core"],
            title="S1 within-encoder (core hooks)",
        )

        # [0,3]: residual hooks
        _plot_cka_subset(
            axes[0, 3],
            epoch_diag.s1_within_cka,
            epoch_diag.s1_layers,
            s1_groups["residual"],
            title="S1 within-encoder (residual hooks)",
        )
    else:
        for j in range(1, 4):
            axes[0, j].axis("off")

    # === S2 ENCODER ROW ===

    # [1,0]: S2 full (all hooks)
    _plot_cka(
        axes[1, 0],
        epoch_diag.s2_within_cka,
        epoch_diag.s2_layers,
        epoch_diag.s2_layers,
        title="S2 within-encoder (all hooks)",
        xlabel="Layer",
        ylabel="Layer",
    )

    if epoch_diag.s2_within_cka is not None and epoch_diag.s2_layers:
        s2_groups = _split_vit_hook_indices(epoch_diag.s2_layers)

        # [1,1]: LN hooks
        _plot_cka_subset(
            axes[1, 1],
            epoch_diag.s2_within_cka,
            epoch_diag.s2_layers,
            s2_groups["ln"],
            title="S2 within-encoder (LN hooks)",
        )

        # [1,2]: core hooks
        _plot_cka_subset(
            axes[1, 2],
            epoch_diag.s2_within_cka,
            epoch_diag.s2_layers,
            s2_groups["core"],
            title="S2 within-encoder (core hooks)",
        )

        # [1,3]: residual hooks
        _plot_cka_subset(
            axes[1, 3],
            epoch_diag.s2_within_cka,
            epoch_diag.s2_layers,
            s2_groups["residual"],
            title="S2 within-encoder (residual hooks)",
        )
    else:
        for j in range(1, 4):
            axes[1, j].axis("off")

    fig.suptitle(
        f"Transformer embedding diagnostics — {epoch_diag.label}\n"
        f"Epoch {epoch_diag.epoch} | Samples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = output_dir / "embedding_diagnostics_transformer.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path

def _split_odd_even_indices(layer_names: list[str]) -> tuple[list[int], list[int]]:
    """
    Return indices for 'odd' and 'even' layers in a ResNet-like model.

    We follow Kornblith et al.:
      - "odd"  = bn3 outputs inside bottlenecks
      - "even" = post-residual block outputs (our `.block_out` hooks)
    """
    odd_idx: list[int] = []
    even_idx: list[int] = []

    for i, name in enumerate(layer_names):
        if name.endswith(".bn3"):
            odd_idx.append(i)
        elif name.endswith(".block_out"):
            even_idx.append(i)

    if not odd_idx or not even_idx:
        raise ValueError(
            f"Could not find both .bn3 and .block_out layers in names:\n{layer_names}"
        )

    return odd_idx, even_idx


    # for i, name in enumerate(layer_names):
    #     # bn3 inside a bottleneck
    #     if "bn3" in name:
    #         odd_idx.append(i)
    #     # post-residual relu – often something like 'layer3.5.relu'
    #     elif name.endswith("relu") and "layer" in name:
    #         even_idx.append(i)

    # # Fallback: use index parity if name-based detection fails
    # if not odd_idx or not even_idx:
    #     # raise NotImplementedError("Fallback to index parity not implemented.")
    #     odd_idx = [i for i in range(len(layer_names)) if i % 2 == 1]
    #     even_idx = [i for i in range(len(layer_names)) if i % 2 == 0]

    # return odd_idx, even_idx

def _plot_cka_subset(ax, full_matrix: np.ndarray, layer_names: list[str],
                    indices: list[int], title: str) -> None:
    if full_matrix is None or full_matrix.size == 0 or not indices:
        ax.text(0.5, 0.5, "CKA unavailable", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    sub_mat = full_matrix[np.ix_(indices, indices)]
    # re-use your existing helper
    sub_names = [layer_names[i] for i in indices]
    _plot_cka(ax, sub_mat, sub_names, sub_names,
              title=title, xlabel="Layer #", ylabel="Layer #")

def plot_epoch_diagnostics(epoch_diag: EpochDiagnostics, output_dir: Path, label: str, plot_even_odd) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2x4 layout: Top row = S1, Bottom row = S2, Cross-encoder in top-right
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    print(
        f'Plotting CKA matrices:\n'
        f'  S1 layers={epoch_diag.s1_layers}\n'
        f'  S2 layers={epoch_diag.s2_layers}'
    )

    # === TOP ROW (S1 ENCODER) ===
    
    # [0,0]: S1 full
    _plot_cka(
        axes[0, 0],
        epoch_diag.s1_within_cka,
        epoch_diag.s1_layers,
        epoch_diag.s1_layers,
        title="S1 within-encoder (all layers)",
        xlabel="Layer",
        ylabel="Layer",
    )

    # [0,1] & [0,2]: S1 odd/even (if enabled)
    if plot_even_odd:
        s1_odd_idx, s1_even_idx = _split_odd_even_indices(epoch_diag.s1_layers)

        print("Odd:", [epoch_diag.s1_layers[i] for i in s1_odd_idx])
        print("Even:", [epoch_diag.s1_layers[i] for i in s1_even_idx])


        _plot_cka_subset(
            axes[0, 1],
            epoch_diag.s1_within_cka,
            epoch_diag.s1_layers,
            s1_odd_idx,
            title="S1 within-encoder (odd layers)",
        )

        _plot_cka_subset(
            axes[0, 2],
            epoch_diag.s1_within_cka,
            epoch_diag.s1_layers,
            s1_even_idx,
            title="S1 within-encoder (even layers)",
        )
    else:
        axes[0, 1].axis("off")
        axes[0, 2].axis("off")

    # [0,3]: Cross-encoder
    _plot_cka(
        axes[0, 3],
        epoch_diag.cross_cka,
        epoch_diag.cross_s2_layers,
        epoch_diag.cross_s1_layers,
        title="Cross encoder",
        xlabel="S2 layer",
        ylabel="S1 layer",
    )

    # === BOTTOM ROW (S2 ENCODER) ===
    
    # [1,0]: S2 full
    _plot_cka(
        axes[1, 0],
        epoch_diag.s2_within_cka,
        epoch_diag.s2_layers,
        epoch_diag.s2_layers,
        title="S2 within-encoder (all layers)",
        xlabel="Layer",
        ylabel="Layer",
    )

    # [1,1] & [1,2]: S2 odd/even (if enabled)
    if plot_even_odd:
        s2_odd_idx, s2_even_idx = _split_odd_even_indices(epoch_diag.s2_layers)

        _plot_cka_subset(
            axes[1, 1],
            epoch_diag.s2_within_cka,
            epoch_diag.s2_layers,
            s2_odd_idx,
            title="S2 within-encoder (odd layers)",
        )

        _plot_cka_subset(
            axes[1, 2],
            epoch_diag.s2_within_cka,
            epoch_diag.s2_layers,
            s2_even_idx,
            title="S2 within-encoder (even layers)",
        )
    else:
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

    # [1,3]: Summary text
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.5, 0.5,
        f"{label}:\nEpoch {epoch_diag.epoch}\n"
        f"Samples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}",
        ha="center",
        va="center",
        transform=axes[1, 3].transAxes,
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8)
    )

    fig.suptitle(
        f"Embedding diagnostics — {epoch_diag.label}\n"
        f"Epoch {epoch_diag.epoch} | Samples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}",
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = output_dir / "embedding_diagnostics.png"
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


# def _build_subset(dataset: torch.utils.data.Dataset, subset_size: int, seed: int) -> torch.utils.data.Dataset:
#     total = len(dataset)
#     if subset_size <= 0 or subset_size >= total:
#         indices = list(range(total))
#     else:
#         rng = np.random.default_rng(seed)
#         indices = sorted(rng.choice(total, size=subset_size, replace=False).tolist())
#     return torch.utils.data.Subset(dataset, indices)


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
