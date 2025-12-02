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
    # add
    s1_odd_layers: List[str]
    s1_odd_within_cka: Optional[np.ndarray]
    s1_even_layers: List[str]
    s1_even_within_cka: Optional[np.ndarray]
    s1_residual_layers: List[str]
    s1_residual_within_cka: Optional[np.ndarray]
    s2_odd_layers: List[str]
    s2_odd_within_cka: Optional[np.ndarray]
    s2_even_layers: List[str]
    s2_even_within_cka: Optional[np.ndarray]
    s2_residual_layers: List[str]
    s2_residual_within_cka: Optional[np.ndarray]
    s1_group_layers: Dict[str, List[str]]
    s1_group_within_cka: Dict[str, Optional[np.ndarray]]
    s2_group_layers: Dict[str, List[str]]
    s2_group_within_cka: Dict[str, Optional[np.ndarray]]


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
    # tensor = tensor.to('cuda')
    return tensor.to(dtype=torch.float32)


_SUFFIX_ORDER = {
    "conv1": 0,
    "bn1":   1,
    "conv2": 2,
    "bn2":   3,
    "conv3": 4,
    "bn3":   5,
    "block_out": 6,
}

def _layer_sort_key(name: str) -> Tuple[int, int, int, str]:
    """
    Sort key for layer names that handles multiple naming conventions:
    - ResNet: "layer1.0.conv1" -> (layer_idx=1, block_idx=0, suffix_rank, suffix)
    - CROMA: "layer0.attn.ln" -> (layer_idx=0, sublayer_rank, operation_rank, operation)
    - DOFA/ScaleMAE: "blocks.0.attn.layernorm" -> (layer_idx=0, sublayer_rank, operation_rank, operation)
    """
    parts = name.split(".")
    
    # Handle 4-part names: "blocks.0.attn.layernorm"
    if len(parts) == 4 and parts[0] == "blocks":
        _, layer_str, sublayer, operation = parts
        
        if layer_str.isdigit():
            layer_idx = int(layer_str)
            
            # DOFA/ScaleMAE sublayer and operation mapping
            sublayer_rank = {"attn": 0, "mlp": 1}.get(sublayer, 999)
            operation_rank = {
                "layernorm": 0, "ln": 0,           # Layer norm
                "op": 1, "core": 1,                # Core operation  
                "scale": 2,                        # Scale (DOFA-specific)
                "residual": 3                      # Residual
            }.get(operation, 999)
            
            return (layer_idx, sublayer_rank, operation_rank, f"{sublayer}.{operation}")
        else:
            return (999, 999, 999, name)
    
    # Handle 3-part names: "layer0.attn.ln" or "layer1.0.conv1"
    elif len(parts) == 3:
        layer_str, middle_str, suffix = parts
        
        # Extract layer index
        if layer_str.startswith("layer"):
            layer_idx = int(layer_str.replace("layer", ""))
        else:
            return (999, 999, 999, name)
        
        # Check if middle part is numeric (ResNet) or text (CROMA)
        if middle_str.isdigit():
            # ResNet style: "layer1.0.conv1"
            block_idx = int(middle_str)
            suffix_rank = _SUFFIX_ORDER.get(suffix, 998)
            return (layer_idx, block_idx, suffix_rank, suffix)
        
        else:
            # CROMA style: "layer0.attn.ln"
            sublayer = middle_str
            operation = suffix
            
            sublayer_rank = {"attn": 0, "ffn": 1, "mlp": 1}.get(sublayer, 999)
            operation_rank = {
                "ln": 0, "layernorm": 0,           # Layer norm
                "core": 1, "op": 1,                # Core operation
                "scale": 2,                        # Scale
                "residual": 3                      # Residual
            }.get(operation, 999)
            
            return (layer_idx, sublayer_rank, operation_rank, f"{sublayer}.{operation}")
    
    # Fallback for other patterns
    return (999, 999, 999, name)


def _order_layers(layer_dict: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(layer_dict.keys(), key=_layer_sort_key)


def compute_cross_encoder_cka(
    s1_layers: Dict[str, torch.Tensor],
    s2_layers: Dict[str, torch.Tensor],
    cuda_cka,
) -> Tuple[List[str], List[str], Optional[np.ndarray]]:
    device = "cuda"
    max_samples = 200

    # Order layer names
    names_s1 = _order_layers(s1_layers)
    names_s2 = _order_layers(s2_layers)

    if not names_s1 or not names_s2:
        return [], [], None

    # ---- Decide shared subsampling indices from first layer of each encoder ----
    first_s1 = names_s1[0]
    t1 = _prepare_cka_tensor(s1_layers[first_s1])
    if t1 is None:
        raise ValueError(f"CKA preparation returned None for S1 layer {first_s1}")

    first_s2 = names_s2[0]
    t2 = _prepare_cka_tensor(s2_layers[first_s2])
    if t2 is None:
        raise ValueError(f"CKA preparation returned None for S2 layer {first_s2}")

    N1 = t1.shape[0]
    N2 = t2.shape[0]
    if N1 != N2:
        raise ValueError(
            f"S1 and S2 must have same #samples for CKA: "
            f"S1[{first_s1}] has {N1}, S2[{first_s2}] has {N2}"
        )

    N = N1
    if N > max_samples:
        idx = torch.randperm(N)[:max_samples]
        t1 = t1[idx]
        t2 = t2[idx]
    else:
        idx = None

    # ---- Prepare S1 on CPU with shared idx ----
    prepared_s1: Dict[str, torch.Tensor] = {first_s1: t1}
    for name in names_s1[1:]:
        t = _prepare_cka_tensor(s1_layers[name])
        if t is None:
            raise ValueError(f"CKA preparation returned None for S1 layer {name}")
        if t.shape[0] != N:
            raise ValueError(
                f"All S1 layers must have same #samples for CKA: "
                f"{first_s1} has {N}, but {name} has {t.shape[0]}"
            )
        if idx is not None:
            t = t[idx]
        prepared_s1[name] = t

    # ---- Prepare S2 on CPU with same idx ----
    prepared_s2: Dict[str, torch.Tensor] = {first_s2: t2}
    for name in names_s2[1:]:
        t = _prepare_cka_tensor(s2_layers[name])
        if t is None:
            raise ValueError(f"CKA preparation returned None for S2 layer {name}")
        if t.shape[0] != N:
            raise ValueError(
                f"All S2 layers must have same #samples for CKA: "
                f"{first_s2} has {N}, but {name} has {t.shape[0]}"
            )
        if idx is not None:
            t = t[idx]
        prepared_s2[name] = t

    names_1 = [name for name in names_s1 if name in prepared_s1]
    names_2 = [name for name in names_s2 if name in prepared_s2]

    if not names_1 or not names_2:
        return [], [], None

    matrix = np.full((len(names_1), len(names_2)), np.nan, dtype=np.float32)

    # ---- Pairwise CKA: only current pair on CUDA ----
    for i, name_i in enumerate(names_1):
        x_cpu = prepared_s1[name_i]  # [N', D]
        x = x_cpu.to(device, non_blocking=True)

        for j, name_j in enumerate(names_2):
            y_cpu = prepared_s2[name_j]
            assert x_cpu.dtype == y_cpu.dtype, (
                f"Mismatched dtypes in CKA computation: "
                f"{x_cpu.dtype} vs {y_cpu.dtype}"
            )

            y = y_cpu.to(device, non_blocking=True)

            value = cuda_cka.linear_CKA(x, y)
            if value is None:
                raise ValueError("CKA computation returned None")

            val = float(value.clamp(0.0, 1.0).item())
            matrix[i, j] = val

            del y, value

        del x

    return names_1, names_2, matrix




def compute_within_encoder_cka(
    layer_features: Dict[str, torch.Tensor],
    cuda_cka,
) -> Tuple[List[str], Optional[np.ndarray]]:
    # print("Computing within encsoder CKA")

    device = "cuda"
    max_samples = 200

    # Order layer names
    # print(layer_features.keys())
    ordered = _order_layers(layer_features)

    # Prepare features and choose ONE subsampling index shared by all layers
    prepared: Dict[str, torch.Tensor] = {}

    if not ordered:
        return [], None

    # Use the first layer to decide subsampling indices
    first_name = ordered[0]
    t0 = _prepare_cka_tensor(layer_features[first_name])
    if t0 is None:
        raise ValueError(f"CKA preparation returned None for layer {first_name}")

    N = t0.shape[0]
    if N > max_samples:
        idx = torch.randperm(N)[:max_samples]
        t0 = t0[idx]
    else:
        idx = None  # no subsampling needed

    prepared[first_name] = t0

    # Prepare all other layers with the SAME idx
    for name in ordered[1:]:
        t = _prepare_cka_tensor(layer_features[name])
        if t is None:
            raise ValueError(f"CKA preparation returned None for layer {name}")

        if t.shape[0] != N:
            raise ValueError(
                f"All layers must have same #samples for CKA: "
                f"{first_name} has {N}, but {name} has {t.shape[0]}"
            )

        if idx is not None:
            t = t[idx]

        prepared[name] = t

    names = [name for name in ordered if name in prepared]
    size = len(names)
    if size == 0:
        return [], None

    matrix = np.full((size, size), np.nan, dtype=np.float32)

    # Diagonal is 1.0 by definition
    for i in range(size):
        matrix[i, i] = 1.0

    # Pairwise CKA – only current row/column on CUDA
    for i, name_i in enumerate(names):
        x_cpu = prepared[name_i]  # [N', D] on CPU
        x = x_cpu.to(device, non_blocking=True)

        for j in range(i + 1, size):
            y_cpu = prepared[names[j]]
            assert x_cpu.dtype == y_cpu.dtype, (
                f"Mismatched dtypes in CKA computation: "
                f"{x_cpu.dtype} vs {y_cpu.dtype}"
            )

            y = y_cpu.to(device, non_blocking=True)

            value = cuda_cka.linear_CKA(x, y)
            if value is None:
                raise ValueError("CKA computation returned None")

            val = float(value.clamp(0.0, 1.0).item())
            matrix[i, j] = matrix[j, i] = val

            del y, value

        del x
        torch.cuda.empty_cache()

    return names, matrix



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
            tensor = tensor.detach().to("cpu")
            cache.append(tensor)

        return hook

    def _make_block_hook(layer_key: str, layer_name: str):
        base = _make_layer_hook(layer_key, layer_name)

        def block_hook(module, inputs, output):
            # Capture block output
            return base(module, inputs, output)

        return block_hook
    
    def _attach_layers_rcf(encoder: nn.Module, key: str):
        """
        Attach hooks for RandomConvFeatures:

        conv[0] : first conv  (in_chans -> 256)   -> "<key>: conv.0"
        conv[2] : second conv (256 -> out_dim)    -> "<key>: conv.2"
        encoder : final pooled embedding          -> "<key>: block_out"
        """
        if not hasattr(encoder, "conv") or not isinstance(encoder.conv, nn.Sequential):
            raise ValueError(f"Expected encoder with .conv Sequential, got: {type(encoder)}")

        # First conv: conv[0]
        if len(encoder.conv) > 0 and isinstance(encoder.conv[0], nn.Conv2d):
            conv1_name = "conv.0"
            handles.append(
                encoder.conv[0].register_forward_hook(
                    _make_layer_hook(key, conv1_name)
                )
            )
        # Second conv: conv[2]
        if len(encoder.conv) > 2 and isinstance(encoder.conv[2], nn.Conv2d):
            conv2_name = "conv.2"
            handles.append(
                encoder.conv[2].register_forward_hook(
                    _make_layer_hook(key, conv2_name)
                )
            )
        # Final pooled representation (what forward() returns)
        block_out_name = "block_out"
        handles.append(
            encoder.register_forward_hook(
                _make_block_hook(key, block_out_name)
            )
        )



    def _attach_layers_resnet(encoder: nn.Module, key: str):
        for name, module in encoder.named_modules():
            if module.__class__.__name__ == "Bottleneck" and hasattr(module, "bn2"):
                bn2_name = f"{name}.bn2"
                handles.append(
                    module.bn2.register_forward_hook(
                        _make_layer_hook(key, bn2_name)
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

    def _attach_layers_dofa(encoder: nn.Module, key: str):
        """
        Attach 4 hook points for each sublayer (attention + MLP) in each DOFA/timm ViT Block.

        For a Block named "blocks.i":

        Attention sublayer:
            blocks.i.attn.layernorm : output of norm1(x)
            blocks.i.attn.scale     : output of ls1(attn_out)
            blocks.i.attn.op        : output of attn(norm1(x))
            blocks.i.attn.residual  : x_in + drop_path1(ls1(attn(norm1(x))))

        MLP sublayer:
            blocks.i.mlp.layernorm  : output of norm2(x)
            blocks.i.mlp.scale      : output of ls2(mlp_out)
            blocks.i.mlp.op         : output of mlp(norm2(x))
            blocks.i.mlp.residual   : x_attn + drop_path2(ls2(mlp(norm2(x))))

        Assumes timm-style Block forward:
            x = x + drop_path1(ls1(attn(norm1(x))))
            x = x + drop_path2(ls2(mlp(norm2(x))))
        """
        # Per-block buffers keyed by module id
        block_buf = {}

        for name, module in encoder.named_modules():
            # timm/DOFA ViT blocks are class "Block"
            if module.__class__.__name__ == "Block":
                block_id = id(module)
                block_name = name  # e.g. "blocks.0"

                block_buf[block_id] = {
                    "name": block_name,
                    "x_in": None,    # block input
                    "dp1_out": None, # output of drop_path1
                    "dp2_out": None, # output of drop_path2
                }
                buf = block_buf[block_id]

                # ---------- Block-level hooks to reconstruct residuals ----------

                def _block_pre_hook(m, inputs, buf=buf):
                    # Save input to the whole block
                    x_in = inputs[0]
                    buf["x_in"] = x_in

                def _block_post_hook(m, inputs, output, buf=buf):
                    """
                    At this point, the block forward is done and we have:
                    x_in      : input to the block
                    dp1_out   : drop_path1(ls1(attn_out))
                    dp2_out   : drop_path2(ls2(mlp_out))

                    We reconstruct:
                    x_attn = x_in + dp1_out
                    x_mlp  = x_attn + dp2_out
                    and feed them through _make_layer_hook as "residual" activations.
                    """
                    x_in = buf["x_in"]
                    dp1_out = buf["dp1_out"]
                    dp2_out = buf["dp2_out"]
                    block_name = buf["name"]

                    if x_in is None or dp1_out is None or dp2_out is None:
                        # Incomplete info; don't crash
                        return

                    # Attention residual (after first sublayer)
                    x_attn = x_in + dp1_out
                    # MLP residual (after second sublayer)
                    x_mlp = x_attn + dp2_out

                    # Manually invoke layer hooks for these two residual points
                    attn_res_name = f"{block_name}.attn.residual"
                    mlp_res_name = f"{block_name}.mlp.residual"

                    attn_res_hook = _make_layer_hook(key, attn_res_name)
                    mlp_res_hook = _make_layer_hook(key, mlp_res_name)

                    # Signature: hook(module, inputs, output)
                    attn_res_hook(m, (None,), x_attn)
                    mlp_res_hook(m, (None,), x_mlp)

                    # Reset buffer
                    buf["x_in"] = None
                    buf["dp1_out"] = None
                    buf["dp2_out"] = None

                handles.append(module.register_forward_pre_hook(_block_pre_hook))
                handles.append(module.register_forward_hook(_block_post_hook))

                # ---------- Attention sublayer hooks ----------

                if hasattr(module, "norm1"):
                    ln_name = f"{block_name}.attn.layernorm"
                    handles.append(
                        module.norm1.register_forward_hook(
                            _make_layer_hook(key, ln_name)
                        )
                    )

                if hasattr(module, "attn"):
                    op_name = f"{block_name}.attn.op"
                    handles.append(
                        module.attn.register_forward_hook(
                            _make_layer_hook(key, op_name)
                        )
                    )

                if hasattr(module, "ls1"):
                    scale_name = f"{block_name}.attn.scale"
                    handles.append(
                        module.ls1.register_forward_hook(
                            _make_layer_hook(key, scale_name)
                        )
                    )

                if hasattr(module, "drop_path1"):
                    def _dp1_hook(m, inputs, output, buf=buf):
                        buf["dp1_out"] = output
                    handles.append(
                        module.drop_path1.register_forward_hook(_dp1_hook)
                    )

                # ---------- MLP sublayer hooks ----------

                if hasattr(module, "norm2"):
                    ln2_name = f"{block_name}.mlp.layernorm"
                    handles.append(
                        module.norm2.register_forward_hook(
                            _make_layer_hook(key, ln2_name)
                        )
                    )

                if hasattr(module, "mlp"):
                    mlp_op_name = f"{block_name}.mlp.op"
                    handles.append(
                        module.mlp.register_forward_hook(
                            _make_layer_hook(key, mlp_op_name)
                        )
                    )

                if hasattr(module, "ls2"):
                    mlp_scale_name = f"{block_name}.mlp.scale"
                    handles.append(
                        module.ls2.register_forward_hook(
                            _make_layer_hook(key, mlp_scale_name)
                        )
                    )

                if hasattr(module, "drop_path2"):
                    def _dp2_hook(m, inputs, output, buf=buf):
                        buf["dp2_out"] = output
                    handles.append(
                        module.drop_path2.register_forward_hook(_dp2_hook)
                    )

    def _attach_layers_scalemae(encoder: nn.Module, key: str):
        """
        Attach 4 hook points for each sublayer (attention + MLP) in each ScaleMAE ViT Block.

        For a Block named "blocks.i":

        Attention sublayer:
            blocks.i.attn.layernorm : output of norm1(x)
            blocks.i.attn.scale     : output of ls1(attn_out)
            blocks.i.attn.op        : output of attn(norm1(x))
            blocks.i.attn.residual  : x_in + drop_path1(ls1(attn(norm1(x))))

        MLP sublayer:
            blocks.i.mlp.layernorm  : output of norm2(x_attn)
            blocks.i.mlp.scale      : output of ls2(mlp_out)
            blocks.i.mlp.op         : output of mlp(norm2(x_attn))
            blocks.i.mlp.residual   : x_attn + drop_path2(ls2(mlp(norm2(x_attn))))

        Assumes timm-style Block forward:
            x = x + drop_path1(ls1(attn(norm1(x))))
            x = x + drop_path2(ls2(mlp(norm2(x))))
        """
        # Per-block buffers keyed by module id
        block_buf = {}

        for name, module in encoder.named_modules():
            # ScaleMAE uses timm-like ViT Block modules
            if module.__class__.__name__ == "Block":
                block_id = id(module)
                block_name = name  # e.g. "blocks.0", "blocks.1", ...

                block_buf[block_id] = {
                    "name": block_name,
                    "x_in": None,    # input to the whole block
                    "dp1_out": None, # output of drop_path1(ls1(attn_out))
                    "dp2_out": None, # output of drop_path2(ls2(mlp_out))
                }
                buf = block_buf[block_id]

                # ---------- Block-level hooks to reconstruct residuals ----------

                def _block_pre_hook(m, inputs, buf=buf):
                    # Save input to the whole block
                    x_in = inputs[0]
                    buf["x_in"] = x_in

                def _block_post_hook(m, inputs, output, buf=buf):
                    """
                    At this point, the block forward is done and we have:
                    x_in      : input to the block
                    dp1_out   : drop_path1(ls1(attn_out))
                    dp2_out   : drop_path2(ls2(mlp_out))

                    We reconstruct:
                    x_attn = x_in + dp1_out
                    x_mlp  = x_attn + dp2_out
                    and feed them through _make_layer_hook as "residual" activations.
                    """
                    x_in = buf["x_in"]
                    dp1_out = buf["dp1_out"]
                    dp2_out = buf["dp2_out"]
                    block_name = buf["name"]

                    if x_in is None or dp1_out is None or dp2_out is None:
                        # Incomplete info; don't crash
                        return

                    # Attention residual (after first sublayer)
                    x_attn = x_in + dp1_out
                    # MLP residual (after second sublayer)
                    x_mlp = x_attn + dp2_out

                    # Use your existing hook factory to store these
                    attn_res_name = f"{block_name}.attn.residual"
                    mlp_res_name = f"{block_name}.mlp.residual"

                    attn_res_hook = _make_layer_hook(key, attn_res_name)
                    mlp_res_hook = _make_layer_hook(key, mlp_res_name)

                    # Signature: hook(module, inputs, output)
                    attn_res_hook(m, (None,), x_attn)
                    mlp_res_hook(m, (None,), x_mlp)

                    # Reset buffer
                    buf["x_in"] = None
                    buf["dp1_out"] = None
                    buf["dp2_out"] = None

                handles.append(module.register_forward_pre_hook(_block_pre_hook))
                handles.append(module.register_forward_hook(_block_post_hook))

                # ---------- Attention sublayer hooks ----------

                if hasattr(module, "norm1"):
                    ln_name = f"{block_name}.attn.layernorm"
                    handles.append(
                        module.norm1.register_forward_hook(
                            _make_layer_hook(key, ln_name)
                        )
                    )

                if hasattr(module, "attn"):
                    op_name = f"{block_name}.attn.op"
                    handles.append(
                        module.attn.register_forward_hook(
                            _make_layer_hook(key, op_name)
                        )
                    )

                if hasattr(module, "ls1"):
                    scale_name = f"{block_name}.attn.scale"
                    handles.append(
                        module.ls1.register_forward_hook(
                            _make_layer_hook(key, scale_name)
                        )
                    )

                if hasattr(module, "drop_path1"):
                    def _dp1_hook(m, inputs, output, buf=buf):
                        buf["dp1_out"] = output
                    handles.append(
                        module.drop_path1.register_forward_hook(_dp1_hook)
                    )

                # ---------- MLP sublayer hooks ----------

                if hasattr(module, "norm2"):
                    ln2_name = f"{block_name}.mlp.layernorm"
                    handles.append(
                        module.norm2.register_forward_hook(
                            _make_layer_hook(key, ln2_name)
                        )
                    )

                if hasattr(module, "mlp"):
                    mlp_op_name = f"{block_name}.mlp.op"
                    handles.append(
                        module.mlp.register_forward_hook(
                            _make_layer_hook(key, mlp_op_name)
                        )
                    )

                if hasattr(module, "ls2"):
                    mlp_scale_name = f"{block_name}.mlp.scale"
                    handles.append(
                        module.ls2.register_forward_hook(
                            _make_layer_hook(key, mlp_scale_name)
                        )
                    )

                if hasattr(module, "drop_path2"):
                    def _dp2_hook(m, inputs, output, buf=buf):
                        buf["dp2_out"] = output
                    handles.append(
                        module.drop_path2.register_forward_hook(_dp2_hook)
                    )



    real_model = model
    encoder_s1 = getattr(real_model, "encoder_s1", None)
    encoder_s2 = getattr(real_model, "encoder_s2")


    # print('ENCODER S1:', encoder_s1)
    print('ENCODER S2:', encoder_s2)

    print('ENCODER S1 TYPE:', type(encoder_s1))
    print('ENCODER S2 TYPE:', type(encoder_s2))
    # print model architecture


    # if the encoders are resnets
    if encoder_s1 is not None:
        enc_type1 = str(type(encoder_s1))
        if 'ResNet' in enc_type1:
            print('Attaching ResNet S1 layers')
            _attach_layers_resnet(encoder_s1, "s1")
        elif 'croma' in enc_type1.lower():
            print('Attaching CromaViT S1 layers')
            _attach_layers_croma_vit(encoder_s1, "s1")
        else:
            raise NotImplementedError(f"Layer hooks not implemented for S1 encoder type: {type(encoder_s1)}")
    if encoder_s2 is not None:
        enc_type = str(type(encoder_s2))
        print(f"S2 encoder type: {enc_type}")

        if "ResNet" in enc_type:
            print("Attaching ResNet S2 layers")
            _attach_layers_resnet(encoder_s2, "s2")

        elif "DOFA" in enc_type or "ScaleMAE" in enc_type or 'VisionTransformer' in enc_type:
            print("Attaching DOFA ViT S2 layers")
            _attach_layers_dofa(encoder_s2, "s2")

        # elif "ScaleMAE" in enc_type:
        #     print("Attaching ScaleMAE ViT S2 layers")
        #     _attach_layers_scalemae(encoder_s2, "s2")

        elif "croma" in enc_type.lower():
            # This is your CROMA ViT (and any other plain ViT you want to treat the same way)
            print("Attaching CromaViT S2 layers")
            _attach_layers_croma_vit(encoder_s2, "s2")
        elif "RandomConvFeatures" in enc_type:
            print('Attaching RandomConvFeatures S2 layers')
            _attach_layers_rcf(encoder_s2, "s2")

        else:
            print(enc_type)
            raise NotImplementedError(
                f"Layer hooks not implemented for S2 encoder type: {type(encoder_s2)}"
            )

 

    print("Registered layer hooks for embedding extraction.")
    # print("Layers for S1 encoder:", list(layer_caches["s1"].keys()))
    # print("Layers for S2 encoder:", list(layer_caches["s2"].keys()))
    print("Total hooks registered:", len(handles))
    # print("Hook handles:", handles)

    return layer_caches, handles, controller


def _encoder_accepts_lorentz(method) -> bool:
    inst = getattr(method, "__self__", None)  
    return getattr(inst, "is_lorentz", False)
    
def plot_epoch_diagnostics_croma(
    epoch_diag: EpochDiagnostics, output_dir: Path, label: str
) -> Path:
    """
    Plot CKA diagnostics for CROMA-style Transformer encoders.

    Layout (2x4):
      Row 0 (S1): [all] [LN hooks] [core hooks] [residual hooks]
      Row 1 (S2): [all] [LN hooks] [core hooks] [cross-encoder]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    print(
        f"Plotting CROMA CKA matrices:\n"
        f"  S1 layers={epoch_diag.s1_layers}\n"
        f"  S2 layers={epoch_diag.s2_layers}"
    )

    # === TOP ROW (S1 ENCODER) ===
    
    # [0,0]: S1 full (all hooks)
    _plot_cka(
        axes[0, 0],
        epoch_diag.s1_within_cka,
        epoch_diag.s1_layers,
        epoch_diag.s1_layers,
        title="S1 within-encoder (all hooks)",
    )

    # [0,1]: S1 LN hooks
    s1_ln_cka = epoch_diag.s1_group_within_cka.get("ln")
    s1_ln_layers = epoch_diag.s1_group_layers.get("ln", [])
    _plot_cka(
        axes[0, 1],
        s1_ln_cka,
        s1_ln_layers,
        s1_ln_layers,
        title="S1 within-encoder (LN hooks)",
    )

    # [0,2]: S1 core hooks  
    s1_core_cka = epoch_diag.s1_group_within_cka.get("core")
    s1_core_layers = epoch_diag.s1_group_layers.get("core", [])
    _plot_cka(
        axes[0, 2],
        s1_core_cka,
        s1_core_layers,
        s1_core_layers,
        title="S1 within-encoder (core hooks)",
    )

    # [0,3]: S1 residual hooks
    s1_residual_cka = epoch_diag.s1_group_within_cka.get("residual")
    s1_residual_layers = epoch_diag.s1_group_layers.get("residual", [])
    _plot_cka(
        axes[0, 3],
        s1_residual_cka,
        s1_residual_layers,
        s1_residual_layers,
        title="S1 within-encoder (residual hooks)",
    )

    # === BOTTOM ROW (S2 ENCODER) ===
    
    # [1,0]: S2 full (all hooks)
    _plot_cka(
        axes[1, 0],
        epoch_diag.s2_within_cka,
        epoch_diag.s2_layers,
        epoch_diag.s2_layers,
        title="S2 within-encoder (all hooks)",
    )

    # [1,1]: S2 LN hooks
    s2_ln_cka = epoch_diag.s2_group_within_cka.get("ln")
    s2_ln_layers = epoch_diag.s2_group_layers.get("ln", [])
    _plot_cka(
        axes[1, 1],
        s2_ln_cka,
        s2_ln_layers,
        s2_ln_layers,
        title="S2 within-encoder (LN hooks)",
    )

    # [1,2]: S2 core hooks
    s2_core_cka = epoch_diag.s2_group_within_cka.get("core")
    s2_core_layers = epoch_diag.s2_group_layers.get("core", [])
    _plot_cka(
        axes[1, 2],
        s2_core_cka,
        s2_core_layers,
        s2_core_layers,
        title="S2 within-encoder (core hooks)",
    )

    # [1,3]: Cross-encoder CKA
    _plot_cka(
        axes[1, 3],
        epoch_diag.cross_cka,
        epoch_diag.cross_s2_layers,
        epoch_diag.cross_s1_layers,
        title="Cross encoder",
    )

    fig.suptitle(
        f"CROMA embedding diagnostics — {epoch_diag.label}\n"
        f"Epoch {epoch_diag.epoch} | Samples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = output_dir / "embedding_diagnostics_croma.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_epoch_diagnostics_scalemae(
    epoch_diag: EpochDiagnostics, output_dir: Path, label: str
) -> Path:
    """Plot CKA diagnostics for ScaleMAE/DOFA-style transformer encoders."""

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    print(
        f"Plotting ScaleMAE/DOFA Transformer CKA matrices:\n",
    )

    _plot_cka(
        axes[0],
        epoch_diag.s2_within_cka,
        epoch_diag.s2_layers,
        epoch_diag.s2_layers,
        title="S2 within-encoder (all hooks)",
        # xlabel="Layer",
        # ylabel="Layer",
    )

    if epoch_diag.s2_within_cka is not None and epoch_diag.s2_layers:
        s2_groups = _split_scalemae_hook_indices(epoch_diag.s2_layers)

        def _plot_subset(ax, indices: list[int], title: str) -> None:
            if not indices:
                ax.text(
                    0.5,
                    0.5,
                    "CKA unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                return

            sub_mat = epoch_diag.s2_within_cka[np.ix_(indices, indices)]
            sub_names = [epoch_diag.s2_layers[i] for i in indices]
            _plot_cka(
                ax,
                sub_mat,
                sub_names,
                sub_names,
                title=title,
            )

        _plot_subset(
            axes[1], s2_groups["layernorm"], "S2 within-encoder (layernorm)"
        )
        _plot_subset(
            axes[2], s2_groups["scale"], "S2 within-encoder (scale)"
        )
        _plot_subset(
            axes[3], s2_groups["op"], "S2 within-encoder (op)"
        )
        _plot_subset(
            axes[4], s2_groups["residual"], "S2 within-encoder (residual)"
        )
    else:
        for ax in axes[1:]:
            ax.axis("off")

    fig.suptitle(
        f"ScaleMAE embedding diagnostics — {epoch_diag.label}\n"
        f"Epoch {epoch_diag.epoch} | Samples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = output_dir / "embedding_diagnostics_scalemae.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path

def extract_embeddings_for_dataset(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    input_dtype: torch.dtype,
    device: torch.device,
    max_batches_cka: int,
    autocast,
    register_layer_hooks: bool = True,
) -> Tuple[ModalityEmbeddings, ModalityEmbeddings, List[str]]:
    base_dataset, indices = _unwrap_subset(dataset)
    if register_layer_hooks:
        layer_cache, handles, capture_controller = _register_layer_hooks(model)
    else:
        layer_cache = {"s1": {}, "s2": {}}
        handles = []

        class _NoopController:
            enabled = False
            @contextlib.contextmanager
            def suspend(self):
                yield

        capture_controller = _NoopController()

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
            # Add batch dimension for single patch samples.
            return tensor.unsqueeze(0)
        if tensor.ndim == 4:
            # Preserve batch if it already exists; only squeeze explicit singleton.
            return tensor if tensor.shape[0] != 1 else tensor.squeeze(0)
        raise ValueError("Unsupported sample shape for embedding extraction")

    print("Extracting embeddings for num files (64 images per file)", len(indices))

    # -------- NEW: counter for how many samples contributed to CKA --------
    cka_samples = 0
    cka_limit = max_batches_cka  # e.g., 40

    with torch.no_grad():
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

                s1_tensor = _to_tensor(s1_img).to(device=device, dtype=input_dtype)
                s2_tensor = _to_tensor(s2_img).to(device=device, dtype=input_dtype)

                ctx = (
                    torch.autocast(device_type="cuda")
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
                    assert has_posthead == has_projected, (
                        "Model must have both posthead and projected methods or re-implement."
                    )

                    if has_posthead:
                        # --------------------------------------------------
                        # Decide whether this sample should be captured for CKA
                        # --------------------------------------------------
                        capture_for_cka = cka_samples < cka_limit

                        if capture_for_cka:
                            # Hooks ACTIVE: this sample contributes to CKA
                            if model.encoder_s1 is not None:
                                s1_raw = model.compute_posthead(s1_tensor, modality="s1")
                            s2_raw = model.compute_posthead(s2_tensor, modality="s2")

                            assert s2_raw is not None, "S2 raw embedding extraction returned None"

                            cka_samples += 1  # this sample counted toward the 40
                        else:
                            # Hooks SUSPENDED: no more CKA activations after limit
                            with capture_controller.suspend():
                                if model.encoder_s1 is not None:
                                    s1_raw = model.compute_posthead(s1_tensor, modality="s1")
                                s2_raw = model.compute_posthead(s2_tensor, modality="s2")
                                assert s2_raw is not None, "S2 raw embedding extraction returned None"

                        # Backbone / projected embeddings never contribute to CKA
                        # so they stay under suspend as you already had:
                        with capture_controller.suspend():
                            if model.encoder_s1 is not None:
                                s1_backbone = model.compute_backbone(
                                    s1_tensor.float(), modality="s1"
                                )
                            s2_backbone = model.compute_backbone(
                                s2_tensor.float(), modality="s2"
                            )

                            if model.encoder_s1 is not None:
                                s1_norm = model.compute_projected(s1_tensor, modality="s1")
                            s2_norm = model.compute_projected(s2_tensor, modality="s2")

                    else:
                        # No posthead: just backbone (already under no hooks for CKA)
                        if model.encoder_s1 is not None:
                            s1_backbone = model.compute_backbone(
                                s1_tensor.float(), modality="s1"
                            )
                            assert s1_backbone is not None, "S1 backbone extraction returned None"
                        s2_backbone = model.compute_backbone(
                            s2_tensor.float(), modality="s2"
                        )

                # --- rest of your code unchanged: push vectors to lists, etc. ---
                def _flatten_embeddings(t: torch.Tensor) -> torch.Tensor:
                    return t.flatten(start_dim=1) if t.ndim > 2 else t

                if s1_raw is not None:
                    s1_vectors.append(_flatten_embeddings(s1_raw).cpu().to(torch.float32))
                if s1_norm is not None:
                    s1_norm_vectors.append(_flatten_embeddings(s1_norm).cpu().to(torch.float32))

                if s2_raw is not None:
                    s2_vectors.append(_flatten_embeddings(s2_raw).cpu().to(torch.float32))
                if s2_norm is not None:
                    s2_norm_vectors.append(_flatten_embeddings(s2_norm).cpu().to(torch.float32))

                if model.encoder_s1 is not None and s1_backbone is not None:
                    s1_backbone_vectors.append(_flatten_embeddings(s1_backbone).cpu().to(torch.float32))
                if s2_backbone is not None:
                    s2_backbone_vectors.append(_flatten_embeddings(s2_backbone).cpu().to(torch.float32))

                sample_ids.append(str(uid))

        finally:
            for handle in handles:
                handle.remove()


    

    def _stack(list_tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not list_tensors:
            return None
        if list_tensors[0].dim() == 2:
            return torch.cat(list_tensors, dim=0)
        return torch.stack(list_tensors, dim=0)
    
    print('Gathering layer activations')
    
    # s1_layers = {name: torch.cat(tensors, dim=0) for name, tensors in layer_cache["s1"].items() if tensors}
    # s2_layers = {name: torch.cat(tensors, dim=0) for name, tensors in layer_cache["s2"].items() if tensors}

    # Process layers incrementally to avoid memory explosion
    s1_layers = {}
    s2_layers = {}
    
    # Process S1 layers
    for name, tensors in layer_cache["s1"].items():
        if tensors:
            # print(f"Processing S1 layer: {name} with {len(tensors)} tensors")
            try:
                s1_layers[name] = torch.cat(tensors, dim=0)
            except RuntimeError as e:
                print(f"Failed to concatenate S1 layer {name}: {e}")
                # Clear the tensors to free memory
                del tensors
                layer_cache["s1"][name] = []
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
            # Clear the cache immediately after processing
            layer_cache["s1"][name] = []

    # Process S2 layers
    for name, tensors in layer_cache["s2"].items():
        if tensors:
            # print(f"Processing S2 layer: {name} with {len(tensors)} tensors")
            try:
                s2_layers[name] = torch.cat(tensors, dim=0)
            except RuntimeError as e:
                print(f"Failed to concatenate S2 layer {name}: {e}")
                # Clear the tensors to free memory
                del tensors
                layer_cache["s2"][name] = []
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                continue
            # Clear the cache immediately after processing
            layer_cache["s2"][name] = []

    # Clear the entire cache
    layer_cache.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # for name, tensors in layer_cache["s2"].items():
    #     print(f"S2 Layer captured: {name} with {len(tensors)} tensors")

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
    import CKA
    device = s1.raw.device if s1.raw is not None else s2.raw.device
    cuda_cka = CKA.CudaCKA(device=device)

    s1_sv = compute_singular_values(s1.raw) if s1.raw is not None else np.empty(0, dtype=np.float32)
    s2_sv = compute_singular_values(s2.raw) if s2.raw is not None else np.empty(0, dtype=np.float32)
    s1_layers, s1_cka = compute_within_encoder_cka(s1.layer_activations, cuda_cka)
    s2_layers, s2_cka = compute_within_encoder_cka(s2.layer_activations, cuda_cka)
    cross_s1, cross_s2, cross_cka = compute_cross_encoder_cka(s1.layer_activations, s2.layer_activations, cuda_cka)

    odd_dict = {k: v for k, v in s2.layer_activations.items() if k.endswith(".bn2")}
    odd_names, odd_cka = compute_within_encoder_cka(odd_dict, cuda_cka)

    even_dict = {k: v for k, v in s2.layer_activations.items() if k.endswith(".block_out")}
    even_names, even_cka = compute_within_encoder_cka(even_dict, cuda_cka)

    residual_dict = {k: v for k, v in s2.layer_activations.items() if k.endswith(".residual")}
    residual_names, residual_cka = compute_within_encoder_cka(residual_dict, cuda_cka)



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
        s1_residual_layers=[],
        s1_residual_within_cka=None,
        s2_odd_layers=odd_names,
        s2_odd_within_cka=odd_cka,
        s2_even_layers=even_names,
        s2_even_within_cka=even_cka,
        s2_residual_layers=residual_names,
        s2_residual_within_cka=residual_cka,
        s1_group_layers={},
        s1_group_within_cka={},
        s2_group_layers={},
        s2_group_within_cka={},
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


def _plot_cka(
    ax,
    matrix: Optional[np.ndarray],
    x_labels: List[str],
    y_labels: List[str],
    *,
    title: str,
) -> None:
    """
    Plot a CKA matrix with sensible defaults.

    - Handles missing/NaN matrices.
    - Uses origin='upper' (no vertical flip).
    - Uses aspect='equal' so cells are square.
    - Downsamples ticks to at most ~10 per axis.
    - Only enforces x==y labels when the matrix is square and
      both label lists have the same length (i.e., within-encoder CKA).
    """
    if matrix is None or matrix.size == 0 or np.all(np.isnan(matrix)):
        ax.text(
            0.5,
            0.5,
            "CKA unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Main image: no vertical flip, square cells
    im = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        cmap="magma",
        origin="lower",
    )
    ax.set_title(title)

    # Only enforce x_labels == y_labels when it's a square within-encoder matrix
    if (
        matrix.ndim == 2
        and matrix.shape[0] == matrix.shape[1]
        and len(x_labels) == len(y_labels)
    ):
        if any(a != b for a, b in zip(x_labels, y_labels)):
            raise AssertionError(
                f"Mismatch between x_labels and y_labels:\n"
                f"x_labels={x_labels}\n\ny_labels={y_labels}"
            )

    ax.set_xlabel("Layer #")
    ax.set_ylabel("Layer #")

    # Sensible tick downsampling (works for both within- and cross-encoder)
    def _downsample_indices(n: int, max_ticks: int = 10) -> List[int]:
        if n <= 0:
            return []
        if n <= max_ticks:
            return list(range(n))
        step = int(np.ceil(n / max_ticks))
        return list(range(0, n, step))

    n_x = len(x_labels)
    n_y = len(y_labels)

    xticks = _downsample_indices(n_x)
    yticks = _downsample_indices(n_y)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, ha="right")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    # Colorbar for this axis
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Linear CKA")




def _split_croma_vit_hook_indices(layer_names: list[str]) -> dict[str, list[int]]:
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


def _split_scalemae_hook_indices(layer_names: list[str]) -> dict[str, list[int]]:
    """
    Group ScaleMAE/DOFA hook names emitted by ``_attach_layers_scalemae`` and
    ``_attach_layers_dofa`` into layernorm/scale/op/residual categories.

    Uses naming patterns like:
      blocks.0.attn.layernorm, blocks.0.attn.scale, blocks.0.attn.op, blocks.0.attn.residual
      blocks.0.mlp.layernorm,  blocks.0.mlp.scale,  blocks.0.mlp.op,  blocks.0.mlp.residual
    """
    groups = {"layernorm": [], "scale": [], "op": [], "residual": []}

    for i, name in enumerate(layer_names):
        if ".layernorm" in name:
            groups["layernorm"].append(i)
        elif name.endswith(".residual"):
            groups["residual"].append(i)
        elif ".scale" in name:
            groups["scale"].append(i)
        elif ".op" in name:
            groups["op"].append(i)

    return groups


def _split_transformer_hook_indices(layer_names: list[str]) -> dict[str, list[int]]:
    """
    Dispatch to a hook grouping strategy based on layer name patterns.
    """
    if any(
        pattern in name
        for name in layer_names
        for pattern in (".layernorm", ".scale", ".op")
    ):
        return _split_scalemae_hook_indices(layer_names)

    if any(".ln" in name for name in layer_names):
        return _split_croma_vit_hook_indices(layer_names)

    return {"ln": [], "core": [], "residual": []}


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

    print('Submatrix shape:')
    print(sub_mat.shape)
    _plot_cka(ax, sub_mat, sub_names, sub_names,
              title=title)




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


def plot_epoch_diagnostics_s2only(epoch_diag: EpochDiagnostics, output_dir: Path, label: str, plot_even_odd=True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    print("Total S2 layers:", len(epoch_diag.s2_layers))

    # --- [0]: S2 full CKA ---
    _plot_cka(
        axes[0],
        epoch_diag.s2_within_cka,
        epoch_diag.s2_layers,
        epoch_diag.s2_layers,
        title="S2 within-encoder (all layers)",
    )

    # --- [1] & [2]: odd/even ---
    # print number of odd/even layers
    print("Plotting S2 odd/even layer CKA matrices")
    print("Total S2 odd layers:", len(epoch_diag.s2_odd_layers))
    print("Total S2 even layers:", len(epoch_diag.s2_even_layers))
    if plot_even_odd:
        _plot_cka(
            axes[1],
            epoch_diag.s2_odd_within_cka,
            epoch_diag.s2_odd_layers,
            epoch_diag.s2_odd_layers,
            title="S2 within-encoder (odd layers)",
        )

        _plot_cka(
            axes[2],
            epoch_diag.s2_even_within_cka,
            epoch_diag.s2_even_layers,
            epoch_diag.s2_even_layers,
            title="S2 within-encoder (even layers)",
        )
    else:
        axes[1].axis("off")
        axes[2].axis("off")

    for ax in axes[:3]:
        ax.set_aspect('equal', adjustable='box')

    # --- [3]: Summary ---
    axes[3].axis("off")
    axes[3].text(
        0.5, 0.5,
        f"{label}:\nEpoch {epoch_diag.epoch}\n"
        f"Samples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}",
        ha="center",
        va="center",
        fontsize=14,
        transform=axes[3].transAxes,
    )

    fig.suptitle(f"S2 Embedding diagnostics — {epoch_diag.label}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = output_dir / "embedding_diagnostics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path



def plot_epoch_diagnostics(epoch_diag: EpochDiagnostics, output_dir: Path, label: str, plot_even_odd) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # print(
    #     f'Plotting CKA matrices:\n'
    #     # f'  S1 layers={epoch_diag.s1_layers}\n'
    #     # f'  S2 layers={epoch_diag.s2_layers}'
    # )


    # === TOP ROW (S1) ===

    # [0,0] full
    _plot_cka(
        axes[0, 0],
        epoch_diag.s1_within_cka,
        epoch_diag.s1_layers,
        epoch_diag.s1_layers,
        title="S1 within-encoder (all layers)"
    )

    # [0,1] odd
    _plot_cka(
        axes[0, 1],
        epoch_diag.s1_odd_within_cka,
        epoch_diag.s1_odd_layers,
        epoch_diag.s1_odd_layers,
        title="S1 within-encoder (odd layers)",
    )

    # [0,2] even
    _plot_cka(
        axes[0, 2],
        epoch_diag.s1_even_within_cka,
        epoch_diag.s1_even_layers,
        epoch_diag.s1_even_layers,
        title="S1 within-encoder (even layers)",
    )

    # [0,3] cross
    _plot_cka(
        axes[0, 3],
        epoch_diag.cross_cka,
        epoch_diag.cross_s2_layers,
        epoch_diag.cross_s1_layers,
        title="Cross encoder",
    )

    # === BOTTOM ROW (S2) ===

    # [1,0] full
    _plot_cka(
        axes[1, 0],
        epoch_diag.s2_within_cka,
        epoch_diag.s2_layers,
        epoch_diag.s2_layers,
        title="S2 within-encoder (all layers)",
    )

    # [1,1] odd
    _plot_cka(
        axes[1, 1],
        epoch_diag.s2_odd_within_cka,
        epoch_diag.s2_odd_layers,
        epoch_diag.s2_odd_layers,
        title="S2 within-encoder (odd layers)",
    )

    # [1,2] even
    _plot_cka(
        axes[1, 2],
        epoch_diag.s2_even_within_cka,
        epoch_diag.s2_even_layers,
        epoch_diag.s2_even_layers,
        title="S2 within-encoder (even layers)",
    )

    # [1,3] summary
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.5, 0.5,
        f"{label}:\nEpoch {epoch_diag.epoch}\n"
        f"Samples: {len(epoch_diag.ids)}*64={len(epoch_diag.ids)*64}",
        ha="center",
        va="center",
        fontsize=14,
        transform=axes[1, 3].transAxes,
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
    print("Saved embedding diagnostics plot to:", output_path)
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
