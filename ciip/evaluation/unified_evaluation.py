"""Concise end-to-end evaluation entrypoint.

This module provides a single callable ``run_full_evaluation`` that executes the
three evaluation pipelines requested by the research team plus the optional
Lorentz-specific visualisations.  The implementation intentionally reuses the
utilities that already exist across the codebase (``linearprobe_comparison``,
``export_neuco_embeddings`` and ``visualizations/ssl4eo``) instead of invoking
their CLI entrypoints so that everything can be orchestrated from Python.

The function expects a ``ModelEvalConfig`` describing the dataset locations,
output directory and model source.  By default it consumes CIIP checkpoints,
but users can also request TorchGeo ResNet50 encoders initialised with the
Sentinel-2 DINO or MoCo weights.
"""

from __future__ import annotations

import json
import logging
import contextlib
import re
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.manifold import TSNE
# from ciip.evaluation.ssl4eo_retrieval import compute_cross_modal_retrieval
import subprocess

from torchgeo.datasets import EuroSAT, BigEarthNet
from torchvision import transforms

from ciip.eval_utils import CustomTransform
from ciip.evaluation.model_utils import EvaluationAdapter, build_evaluation_adapter
from ciip.model_ciip import LorentzCIIP
from ciip.open_clip_train import data
import os
import torchvision.transforms as T
from torchvision.models import resnet152, ResNet152_Weights


def _looks_like_vit_checkpoint(checkpoint_path: Optional[Path]) -> bool:
    if checkpoint_path is None:
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else {}
    for key in state_dict.keys():
        if (
            "transformer" in key
            or "positional_embedding" in key
            or "class_embedding" in key
            or "ln_post" in key
        ):
            return True
    return False

## EUROSAT STANDARDIZATION VALUES
EUROSATMEAN = {
    'B01': 1354.40546513,
    'B02': 1118.24399958,
    'B03': 1042.92983953,
    'B04': 947.62620298,
    'B05': 1199.47283961,
    'B06': 1999.79090914,
    'B07': 2369.22292565,
    'B08': 2296.82608323,
    'B09': 732.08340178,
    'B10': 12.11327804,
    'B11': 1819.01027855,
    'B12': 1118.92391149,
    'B8A': 2594.14080798,
}

EUROSATSTD = {
    'B01': 245.71762908,
    'B02': 333.00778264,
    'B03': 395.09249139,
    'B04': 593.75055589,
    'B05': 566.4170017,
    'B06': 861.18399006,
    'B07': 1086.63139075,
    'B08': 1117.98170791,
    'B09': 404.91978886,
    'B10': 4.77584468,
    'B11': 1002.58768311,
    'B12': 761.30323499,
    'B8A': 1231.58581042,
}

EUROSATBANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
EUROSAT_S1_MEAN = {
    "VV": -12.577,
    "VH": -20.265,
}
EUROSAT_S1_STD = {
    "VV": 5.179,
    "VH": 5.872,
}
EUROSAT_S1_BANDS = ("VV", "VH")
EUROSAT_CLASS_NAMES = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake",
}
##

# BigEarthNet uses Sentinel-2 imagery without B10 (12 bands).
BIGEARTHNET_BANDS = tuple(b for b in EUROSATBANDS if b != "B10")
BIGEARTHNET_MEAN = [
    383.02593994140625,   # B01
    487.1451721191406,    # B02
    707.4555053710938,    # B03
    726.1536254882812,    # B04
    1098.5545654296875,   # B05
    1900.9755859375,      # B06
    2184.544189453125,    # B07
    2329.376953125,       # B08
    2387.28515625,        # B8A
    2358.551513671875,    # B09
    1878.415283203125,    # B11
    1237.1298828125,      # B12
]
BIGEARTHNET_STD = [
    455.3548889160156,    # B01
    511.12603759765625,   # B02
    543.4214477539062,    # B03
    678.1015625,          # B04
    686.7001953125,       # B05
    992.5777587890625,    # B06
    1162.4130859375,      # B07
    1267.3634033203125,   # B08
    1245.698974609375,    # B8A
    1190.7650146484375,   # B09
    1112.59716796875,     # B11
    879.724609375,        # B12
]
BIGEARTHNET_STATS = {
    band: {"mean": mean, "std": std}
    for band, mean, std in zip(BIGEARTHNET_BANDS, BIGEARTHNET_MEAN, BIGEARTHNET_STD)
}


def _first_conv_in_channels(module: Optional[nn.Module]) -> Optional[int]:
    if module is None:
        return None
    conv1 = getattr(module, "conv1", None)
    if isinstance(conv1, nn.Conv2d):
        return conv1.in_channels
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            return submodule.in_channels
    return None


def _infer_model_in_channels(
    adapter: EvaluationAdapter, fallback: int, *, modality: str = "s2"
) -> int:
    """Best-effort detection of the channel count for the requested modality."""

    normalized = modality.lower()
    encoders = []
    if normalized == "s1":
        encoders.append(getattr(adapter, "encoder_s1", None))
        encoders.append(getattr(adapter, "encoder_s2", None))
    else:
        encoders.append(getattr(adapter, "encoder_s2", None))
        encoders.append(getattr(adapter, "encoder_s1", None))
    encoders.append(getattr(adapter, "base_model", None))

    for encoder in encoders:
        in_channels = _first_conv_in_channels(encoder)
        if in_channels is not None:
            return in_channels
    return fallback


def _resolve_eurosat_bands(num_channels: int) -> Tuple[str, ...]:
    """Return the EuroSAT band tuple matching the model channel budget."""

    if num_channels == 10:
        # Drop B01 and B10 (and B09) to align to MS-CLIP 10-channel inputs.
        return ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    if num_channels >= len(EUROSATBANDS):
        return EUROSATBANDS
    if num_channels == 12:
        return tuple(band for band in EUROSATBANDS if band != "B10")
    if num_channels == 3:
        return ("B04", "B03", "B02")
    if num_channels < 1:
        raise ValueError("Model must expose at least one Sentinel-2 channel")
    logging.warning(
        "Model exposes %d Sentinel-2 channels; defaulting to the first %d EuroSAT bands.",
        num_channels,
        num_channels,
    )
    return tuple(EUROSATBANDS[:num_channels])


def _resolve_bigearthnet_bands(num_channels: int) -> Tuple[str, ...]:
    """Return the BigEarthNet band tuple matching the model channel budget."""

    if num_channels == 10:
        # Drop B01 and B09/B10 equivalents to fit 10-channel MS-CLIP inputs.
        return ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    if num_channels >= len(BIGEARTHNET_BANDS):
        return BIGEARTHNET_BANDS
    if num_channels == 3:
        return ("B04", "B03", "B02")
    if num_channels < 1:
        raise ValueError("Model must expose at least one Sentinel-2 channel")
    logging.warning(
        "Model exposes %d Sentinel-2 channels; defaulting to the first %d BigEarthNet bands.",
        num_channels,
        num_channels,
    )
    return tuple(BIGEARTHNET_BANDS[:num_channels])


def _align_s2_channels(image: torch.Tensor, expected_channels: Optional[int]) -> torch.Tensor:
    """Drop the cirrus band when the model only supports 12 Sentinel-2 bands."""

    if expected_channels is None or image.ndim < 3:
        return image

    # channel_dim = 1 if image.ndim >= 4 else 0
    if image.ndim >= 5:
        channel_dim = 2  # (B, T, C, H, W)
    elif image.ndim >= 4:
        channel_dim = 1  # (B, C, H, W)
    else:
        channel_dim = 0
    current_channels = image.size(channel_dim)

    if current_channels == expected_channels:
        return image

    if expected_channels == 10 and current_channels >= 10:
        indices = torch.arange(current_channels, device=image.device)
        if current_channels == 13:
            # EuroSAT ordering: drop B01 (0), B09 (9), B10 (10)
            keep = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        elif current_channels == 12:
            # BigEarthNet ordering (no B10): drop B01 (0) and B09 (9)
            keep = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        else:
            # Fallback: drop first two channels
            keep = list(range(2, min(current_channels, expected_channels + 2)))
        keep_indices = indices[keep]
        return torch.index_select(image, channel_dim, keep_indices)

    if current_channels == 13 and expected_channels == 12:
        # Drop B10 (0-indexed channel 10) to match 12-band encoders.
        indices = torch.arange(current_channels, device=image.device)
        keep_indices = torch.cat([indices[:10], indices[11:]])
        return torch.index_select(image, channel_dim, keep_indices)

    if current_channels == 12 and expected_channels == 13:
        # Insert zeroed B10 channel to align to 13-band encoders.
        if channel_dim == 0:
            zeros = torch.zeros(
                (1, *image.shape[1:]),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat([image[:10], zeros, image[10:]], dim=0)
        if channel_dim == 1:
            zeros = torch.zeros(
                (image.size(0), 1, *image.shape[2:]),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat([image[:, :10], zeros, image[:, 10:]], dim=1)
        if channel_dim == 2:
            zeros = torch.zeros(
                (image.size(0), image.size(1), 1, *image.shape[3:]),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat(
                [image[:, :, :10], zeros, image[:, :, 10:]],
                dim=2,
            )

    logging.warning(
        "Unable to automatically align Sentinel-2 channels (expected %s, observed %s).",
        expected_channels,
        current_channels,
    )
    return image


@contextlib.contextmanager
def _use_adapter_modality(adapter: EvaluationAdapter, modality: str):
    """Temporarily switch the adapter to the requested modality."""

    setter = getattr(adapter, "set_active_modality", None)
    getter = getattr(adapter, "get_active_modality", None)
    if setter is None or getter is None:
        yield
        return

    normalized = (modality or "").lower()
    previous = getter()
    if not normalized or normalized == previous:
        yield
        return

    setter(normalized)
    try:
        yield
    finally:
        setter(previous)

from ciip.evaluation.export_neuco_embeddings import (  # type: ignore
    E2SChallengeDataset,
    InputResizer,
    SSL4EONormalize,
    NeuCoNormalize,
    Divideby10000Normalize,
    CromaNormalize,
    TemporalMean,
    collate_fn,
    create_submission_from_dict,
)
from visualizations.ssl4eo.embedding_collapse_diagnostics import (  # type: ignore
    DEFAULT_S2_BANDS,
    EpochDiagnostics,
    ModalityEmbeddings,
    compute_projection,
    compute_singular_values,
    compute_within_encoder_cka,
    ensure_hydra_original_cwd,
    extract_embeddings_for_dataset,
    plot_epoch_diagnostics,
    plot_epoch_diagnostics_s2only,
    plot_epoch_diagnostics_scalemae,
    plot_epoch_diagnostics_croma,
    plot_projection,
    preprocess_projection_data,
    ModalityEmbeddings,
    compute_within_encoder_cka,
    compute_cross_encoder_cka
)
from CKA import CudaCKA
from visualizations.ssl4eo.hyperbolic_visualization import (  # type: ignore
    compute_hyperbolic_context,
    plot_angle_aperture,
    plot_angular_pca,
    plot_cone_polar,
    plot_radial_histogram,
    _extract_curvature_from_state,
    compute_lorentz_pos_neg,
    plot_pos_neg_hist
)
from visualizations.ssl4eo.hyperbolic_retrieval import compute_cross_modal_retrieval
from ciip.evaluation.id_metrics import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SSL4EO_MODEL_TRANSFORMS,
    S2ScaleTransform,
    compute_global_id_metrics,
    prepare_embeddings_for_id,
    save_id_metrics,
    select_ssl4eo_transform,
)
from ciip.open_clip_train.data import SSL4EODataset

NORMALIZATION_METHOD_DIVIDE = "divideby10000"
NORMALIZATION_METHOD_BANDWISE = "bandwisenorm"
NORMALIZATION_METHOD_SSL4EO = "ssl4eonorm"
NORMALIZATION_METHODS = (
    NORMALIZATION_METHOD_DIVIDE,
    NORMALIZATION_METHOD_BANDWISE,
    NORMALIZATION_METHOD_SSL4EO,
)
DEFAULT_NORMALIZATION_METHOD = NORMALIZATION_METHOD_DIVIDE


@dataclass
class ModelEvalConfig:
    eurosat_root: Path
    neuco_root: Path
    output_dir: Path
    checkpoint: Optional[Path] = None
    model_type: str = "ciip_checkpoint"
    model_weights: Optional[str] = None
    ciip_framework: Optional[str] = None
    model_in_channels: int = 13
    evaluation_modality: str = "s2"
    croma_weights: Optional[Path] = None
    croma_image_resolution: int = 120
    enable_ssl4eo: bool = True
    neuco_modalities: Sequence[str] = ("s2l1c",)
    neuco_resize: Optional[Tuple[int, int]] = None
    neuco_seasons: int = 1
    tsne_samples: int = 1500
    pca_samples: int = 5000
    random_seed: int = 0
    model_path: Optional[str] = None
    ssl4eo_root: Optional[Path] = None
    ssl4eo_subset_size: int = 2048
    ssl4eo_subset_seed: int = 0
    ssl4eo_s2_tier: str = "s2c"
    ssl4eo_s2_bands: Sequence[str] = DEFAULT_S2_BANDS
    ssl4eo_image_dimension: int = 224
    eurosat_image_size: int = 224
    bigearthnet_root: Optional[Path] = None
    bigearthnet_image_size: int = 224
    normalization_method: str = DEFAULT_NORMALIZATION_METHOD
    matryoshka_dims: Optional[Sequence[int]] = None
    matryoshka_feature: str = "backbone"
    stats_max_batches: int = 0


@dataclass
class EmbeddingBundle:
    backbone: Optional[np.ndarray]
    projected: Optional[np.ndarray]
    labels: Optional[np.ndarray] = None
    multi_labels: Optional[np.ndarray] = None
    ids: Optional[List[str]] = None


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


def _extract_image_tensor(batch) -> Optional[torch.Tensor]:
    if isinstance(batch, dict):
        return batch.get("image") if "image" in batch else batch.get("data")
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return None


def _to_nchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 5:
        channel_candidates = {1, 2, 3, 4, 10, 12, 13}
        if tensor.shape[1] in {10, 12, 13}:
            b, c, t, h, w = tensor.shape
            return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if tensor.shape[2] in {10, 12, 13}:
            b, t, c, h, w = tensor.shape
            return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if tensor.shape[1] in channel_candidates and tensor.shape[2] not in channel_candidates:
            b, c, t, h, w = tensor.shape
            return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if tensor.shape[2] in channel_candidates:
            b, t, c, h, w = tensor.shape
            return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    raise ValueError(f"Unsupported tensor shape {tuple(tensor.shape)} for band stats")


def _print_band_stats(
    label: str,
    loader: DataLoader,
    *,
    bands: Optional[Sequence[str]] = None,
    max_batches: int = 0,
) -> None:
    total_sum: Optional[torch.Tensor] = None
    total_sumsq: Optional[torch.Tensor] = None
    total_pixels = 0
    batches = 0

    for idx, batch in enumerate(loader):
        if max_batches and idx >= max_batches:
            break
        images = _extract_image_tensor(batch)
        if images is None:
            continue
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        images = _to_nchw(images.detach()).to(dtype=torch.float64, device="cpu")
        total_pixels += images.shape[0] * images.shape[2] * images.shape[3]
        batch_sum = images.sum(dim=(0, 2, 3))
        batch_sumsq = (images ** 2).sum(dim=(0, 2, 3))
        total_sum = batch_sum if total_sum is None else total_sum + batch_sum
        total_sumsq = batch_sumsq if total_sumsq is None else total_sumsq + batch_sumsq
        batches += 1

    if total_sum is None or total_sumsq is None or total_pixels == 0:
        print(f"{label}: unable to compute band stats (no samples).")
        return

    mean = total_sum / total_pixels
    var = (total_sumsq / total_pixels) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0))
    band_note = f" bands={list(bands)}" if bands is not None else ""
    batch_note = f" batches={batches}" + (f"/{max_batches}" if max_batches else "")
    print(
        f"{label} bandwise stats after transforms:{band_note}{batch_note} "
        f"mean={mean.tolist()} std={std.tolist()}"
    )


### FOR EUROSAT and NEUCOBENCH
def _extract_embeddings(
    adapter: EvaluationAdapter,
    dataloader: DataLoader,
    *,
    device: torch.device,
    require_ids: bool = False,
    expected_in_channels: Optional[int] = None,
    modality: str = "s2",
    pca_output_dir: Optional[Path] = None,
) -> EmbeddingBundle:
    backbone_vectors: List[np.ndarray] = []
    projected_vectors: List[np.ndarray] = []
    labels: List[int] = []
    multi_labels: List[List[int]] = []
    ids: List[str] = []

    pca_tensor: torch.Tensor    
    for i, batch in enumerate(dataloader):
        if isinstance(batch, dict):
            image = batch.get("image") if "image" in batch else batch.get("data")
            batch_labels = batch.get("label")
            batch_multi_labels = batch.get("multi_label")
            batch_ids = batch.get("file_name")
        else:
            image, batch_labels = batch
            batch_multi_labels = None
            batch_ids = None

        # print bandwise mean std of images
        # if i % 20 == 0:
        #     print(f'Batch {i} image bandwise mean: {image.mean(dim=[0,2,3]) if isinstance(image, torch.Tensor) else "N/A"}')

        # if any pixels is outside of 0-1, print the band stats
        # if isinstance(image, torch.Tensor):
        #     if (image < 0).any() or (image > 1).any():
        #         print(f'Batch {i} image bandwise stats:')
        #         for b in range(image.shape[1]):
        #             band = image[:, b, :, :]
        #             print(f'  Band {b}: min {band.min().item()}, max {band.max().item()}, mean {band.mean().item()}, std {band.std().item()}')

        # keep a running mean of each band across batches, print out at the end of training

        # move to cuda
        image = image.to(device)
        adapter.to(device)

        # print(f'Image type: {type(image)}')
        # print(f'Image shape: {image.shape if isinstance(image, torch.Tensor) else "N/A"}')

        if isinstance(image, dict) and not getattr(adapter, "supports_multimodal_dict", False):
            image = image[next(iter(image))]

        reshape_back = False
        seasons = 1
        if isinstance(image, torch.Tensor) and image.ndim == 5:
            # Flatten temporal and batch dimensions before feeding the model.
            batch_size, seasons, channels, height, width = image.shape
            image = image.reshape(batch_size * seasons, channels, height, width)
            reshape_back = True
        if modality.lower() == "s2" and isinstance(image, torch.Tensor):
            image = _align_s2_channels(image, expected_in_channels)

        prepared_inputs = adapter.prepare_inputs(image, device=device, modality=modality)

        with torch.no_grad():
            outputs = adapter.compute_embeddings(
                prepared_inputs, modality=modality
            )

        if isinstance(outputs, dict):
            backbone = outputs.get("backbone")
            projected = outputs.get("projected", None)
            # print(f'Backbone shape: {backbone.shape if backbone is not None else "N/A"}')
        elif isinstance(outputs, tuple) and len(outputs) == 3:
            backbone, _, projected = outputs
        else:
            raise RuntimeError("Unexpected outputs from compute_embeddings")

        if reshape_back:
            # Restore (batch, seasons, ...) layout for downstream consumers.
            if backbone is not None:
                backbone = backbone.reshape(batch_size, seasons, *backbone.shape[1:])
                # if i == 0:
                #     pca_tensor = backbone
                # else:
                #     pca_tensor = torch.cat((pca_tensor, backbone), dim=0)
                # # plot pca color coded by season dimension (4 seasons)
                # if i == 20:
                #     try:
                #         if backbone is not None and isinstance(backbone, torch.Tensor):
                #             # backbone is (batch, seasons, dim) after reshape_back above
                #             # use pca tensor
                #             ## TODO
                #             batch_n, n_seasons = backbone.shape[0], backbone.shape[1]
                #             flat = backbone.reshape(batch_n * n_seasons, -1).cpu().numpy()
                #             # need at least 2 samples and 2 dims for PCA
                #             if flat.shape[0] >= 2 and flat.shape[1] >= 2:
                #                 pca = PCA(n_components=2, random_state=0)
                #                 coords = pca.fit_transform(flat)
                #                 season_labels = np.tile(np.arange(n_seasons), batch_n)
                #                 cmap = plt.get_cmap("tab10")
                #                 fig, ax = plt.subplots(figsize=(6, 5))
                #                 for s in range(n_seasons):
                #                     mask = season_labels == s
                #                     ax.scatter(coords[mask, 0], coords[mask, 1], s=16, color=cmap(s), label=f"season_{s}", alpha=1)
                #                 ax.set_title("PCA of Backbone Embeddings (colored by season)")
                #                 ax.set_xlabel("PC 1")
                #                 ax.set_ylabel("PC 2")
                #                 ax.legend(title="Season", fontsize="small", markerscale=1.5)
                #                 ax.grid(alpha=0.2)
                #                 fig.tight_layout()
                #                 # save a small diagnostic file to the requested directory (defaults to temp)
                #                 save_dir = Path(pca_output_dir) if pca_output_dir is not None else Path(tempfile.gettempdir())
                #                 save_dir.mkdir(parents=True, exist_ok=True)
                #                 out_path = save_dir / f"_batch_pca_seasons_{int(time.time()*1000)}.png"
                #                 fig.savefig(out_path, dpi=150)
                #                 plt.close(fig)
                #                 # logging.info("Saved per-batch season PCA to %s", out_path)
                #     except Exception as exc:  # keep extraction robust
                #         logging.warning("Skipping per-batch season PCA: %s", exc)
            if projected is not None:
                projected = projected.reshape(batch_size, seasons, *projected.shape[1:])
            

        if backbone is None:
            raise RuntimeError("Backbone embeddings cannot be None")
        # if backbone has 3 dims average over seasons
        if backbone.ndim == 3:
            backbone = backbone.mean(dim=1)
        
        # print(f'Backbone shape: {backbone.shape if backbone is not None else "N/A"}')
        backbone_np = _to_numpy(backbone)
        backbone_vectors.append(backbone_np)
        # batch_size = backbone_np.shape[0]

        if projected is not None:
            projected_vectors.append(_to_numpy(projected))

        if batch_labels is not None:
            labels.extend(batch_labels.cpu().tolist())
        if batch_multi_labels is not None:
            multi_labels.extend(batch_multi_labels.cpu().tolist())
        if require_ids and batch_ids is not None:
            ids.extend([str(item) for item in batch_ids])

    backbone_array: Optional[np.ndarray]
    if backbone_vectors:
        backbone_array = np.concatenate(backbone_vectors, axis=0)
    else:
        backbone_array = None

    projected_array = (
        np.concatenate(projected_vectors, axis=0) if projected_vectors else None
    )

    bundle = EmbeddingBundle(
        backbone=backbone_array,
        projected=projected_array,
        labels=np.asarray(labels, dtype=np.int64) if labels else None,
        multi_labels=np.asarray(multi_labels, dtype=np.int64) if multi_labels else None,
        ids=ids if ids else None,
    )
    return bundle



def _build_eurosat_loaders(
    config: ModelEvalConfig, *, bands: Sequence[str], modality: str
) -> Dict[str, DataLoader]:
    normalized = modality.lower()
    if normalized == "s1":
        raise ValueError("EuroSAT dataset does not support Sentinel-1 modality")

    # Compute per-band stats in the same order as the resolved band tuple
    mean = [EUROSATMEAN[b] for b in bands]
    std = [EUROSATSTD[b] for b in bands]

    target_size = int(config.eurosat_image_size)
    eval_resize = max(target_size, int(round(target_size * 256 / 224)))

    use_imagenet_rgb = (config.model_weights or "").lower() == "remoteclip"
    normalization_method = config.normalization_method.lower()
    if use_imagenet_rgb:
        norm_layer = transforms.Compose(
            [Divideby10000Normalize(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        )
    elif normalization_method == NORMALIZATION_METHOD_BANDWISE:
        norm_layer = transforms.Normalize(mean=mean, std=std)
    elif normalization_method == NORMALIZATION_METHOD_SSL4EO:
        norm_layer = SSL4EONormalize()
    else:
        norm_layer = Divideby10000Normalize()
    print(f"EuroSAT using normalization method: {norm_layer}")
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(target_size),
                transforms.RandomHorizontalFlip(),
                norm_layer,
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize(eval_resize),
                transforms.CenterCrop(target_size),
                norm_layer,
            ]
        ),
    }

    train_transform = CustomTransform(data_transforms["train"])
    eval_transform = CustomTransform(data_transforms["eval"])

    datasets = {
        split: EuroSAT(
            root=str(config.eurosat_root),
            split=split,
            bands=bands,
            transforms=train_transform if split == "train" else eval_transform,
            download=True,
        )
        for split in ("train", "val", "test")
    }

    loaders = {
        split: DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        for split, dataset in datasets.items()
    }
    # print each split
    for split, dataset in datasets.items():
        # print(f"EuroSAT {split} dataset size: {len(dataset)} samples")
        assert len(dataset) > 0, f"EuroSAT {split} dataset is empty!"
    return loaders


class _SingleLabelBigEarthNet(torch.utils.data.Dataset):
    """Wrap BigEarthNet multi-label targets into a single class via argmax."""

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            image = sample.get("image") if "image" in sample else sample.get("data")
            label = sample.get("label") if "label" in sample else sample.get("labels")
        else:
            image, label = sample

        multi_label = None
        if isinstance(label, torch.Tensor):
            if label.ndim > 0 and label.numel() > 1:
                label_idx = int(torch.argmax(label).item())
                multi_label = label
            else:
                label_idx = int(label.item())
        elif isinstance(label, (list, tuple)):
            label_idx = int(label[0])
        else:
            label_idx = int(label)

        return {"image": image, "label": label_idx, "multi_label": multi_label}


def _filter_missing_bigearthnet_samples(dataset: BigEarthNet) -> None:
    """Drop BigEarthNet entries missing required band folders/files."""
    data_root = Path(dataset.root) / dataset.metadata["s2"]["directory"]
    if not data_root.exists() or not any(data_root.iterdir()):
        print(
            f"BigEarthNet: expected imagery under {data_root} but found none; "
            "marking dataset as empty."
        )
        dataset.folders = []
        return

    required_keys = []
    uses_s2 = dataset.bands in ("s2", "all") or (
        isinstance(dataset.bands, (list, tuple))
        and all(isinstance(b, str) and b.startswith("B") for b in dataset.bands)
    )
    if uses_s2:
        required_keys.append("s2")
    if dataset.bands in ("s1", "all"):
        required_keys.append("s1")

    def _has_band_folder(folder_path: str, band_key: str) -> bool:
        path = Path(folder_path)
        if not path.is_dir():
            return False

        # Fast path: check the expected band file; fall back to any .tif presence.
        suffix = "B01.tif" if band_key == "s2" else "VV.tif"
        expected_file = path / f"{path.name}_{suffix}"
        if expected_file.exists():
            return True
        return any(path.glob("*.tif"))

    valid_folders = []
    missing = 0
    for folder in dataset.folders:
        if all(_has_band_folder(folder[key], key) for key in required_keys):
            valid_folders.append(folder)
        else:
            missing += 1

    if missing:
        print(
            f"BigEarthNet: filtered out {missing} samples missing required "
            f"{str(dataset.bands).upper()} imagery (kept {len(valid_folders)})."
        )
    dataset.folders = valid_folders


def _bigearthnet_stats_for_bands(bands: Sequence[str]) -> Tuple[List[float], List[float]]:
    mean: List[float] = []
    std: List[float] = []
    for band in bands:
        stats = BIGEARTHNET_STATS.get(band)
        if stats is None:
            raise KeyError(f"Unknown BigEarthNet band '{band}' in requested band set {bands}")
        mean.append(stats["mean"])
        std.append(stats["std"])
    return mean, std


def _resolve_bigearthnet_root(root: Path) -> Path:
    """Handle nested BigEarthNet extracts (root/BigEarthNet-v1.0/BigEarthNet-v1.0)."""
    base = Path(root)
    csv_present = (base / "bigearthnet-train.csv").exists()
    nested = base / "BigEarthNet-v1.0"
    double_nested = nested / "BigEarthNet-v1.0"

    if csv_present:
        return base
    if (nested / "bigearthnet-train.csv").exists():
        print(f"BigEarthNet: detected nested layout with CSVs, using root {nested}")
        return nested
    if (double_nested / "bigearthnet-train.csv").exists():
        print(f"BigEarthNet: detected double-nested layout with CSVs, using root {double_nested}")
        return double_nested
    # Fall back to base so TorchGeo can still attempt to verify.
    return base


def _build_bigearthnet_loaders(
    config: ModelEvalConfig, *, expected_channels: Optional[int] = None
) -> Tuple[Dict[str, DataLoader], Sequence[str]]:
    if config.bigearthnet_root is None:
        raise ValueError("bigearthnet_root must be provided for BigEarthNet evaluation.")

    channel_budget = expected_channels or config.model_in_channels
    bands = _resolve_bigearthnet_bands(channel_budget)
    band_mean, band_std = _bigearthnet_stats_for_bands(bands)
    band_indices = [BIGEARTHNET_BANDS.index(b) for b in bands]
    select_bands = len(band_indices) != len(BIGEARTHNET_BANDS)

    target_size = int(config.bigearthnet_image_size)
    eval_resize = max(target_size, int(round(target_size * 256 / 224)))

    class _SelectBandsTransform:
        def __init__(self, indices: Sequence[int]):
            self.indices = list(indices)

        def __call__(self, img: torch.Tensor) -> torch.Tensor:
            if not isinstance(img, torch.Tensor):
                return img
            if img.ndim >= 4:
                channel_dim = 1  # (B, C, H, W) or (T, C, H, W)
            elif img.ndim == 3:
                channel_dim = 0  # (C, H, W)
            else:
                return img
            index_tensor = torch.as_tensor(self.indices, device=img.device)
            return torch.index_select(img, channel_dim, index_tensor)

    selector = _SelectBandsTransform(band_indices) if select_bands else None

    train_steps: List[object] = []
    eval_steps: List[object] = []
    if selector is not None:
        train_steps.append(selector)
        eval_steps.append(selector)
    use_imagenet_rgb = (config.model_weights or "").lower() == "remoteclip"
    normalization_method = config.normalization_method.lower()
    if use_imagenet_rgb:
        band_norm_layer = transforms.Compose(
            [Divideby10000Normalize(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        )
    elif normalization_method == NORMALIZATION_METHOD_BANDWISE:
        band_norm_layer = transforms.Normalize(mean=band_mean, std=band_std)
    elif normalization_method == NORMALIZATION_METHOD_SSL4EO:
        band_norm_layer = SSL4EONormalize()
    else:
        band_norm_layer = Divideby10000Normalize()
    print(f"band norm method: {band_norm_layer}")
    train_steps.extend(
        [
            transforms.RandomResizedCrop(target_size),
            transforms.RandomHorizontalFlip(),
            band_norm_layer,
        ]
    )
    eval_steps.extend(
        [
            transforms.CenterCrop(target_size),
            band_norm_layer,
        ]
    )

    data_transforms = {
        "train": transforms.Compose(train_steps),
        "eval": transforms.Compose(eval_steps),
    }

    train_transform = CustomTransform(data_transforms["train"])
    eval_transform = CustomTransform(data_transforms["eval"])

    print("Building BigEarthNet datasets...")

    dataset_root = _resolve_bigearthnet_root(Path(config.bigearthnet_root))

    datasets: Dict[str, _SingleLabelBigEarthNet] = {}
    for split in ("train", "val", "test"):
        base_dataset = BigEarthNet(
            root=str(dataset_root),
            split=split,
            bands="s2",
            transforms=train_transform if split == "train" else eval_transform,
            download=False,
            num_classes=19
        )
        # _filter_missing_bigearthnet_samples(base_dataset)
        if len(base_dataset) == 0:
            data_dir = Path(dataset_root) / base_dataset.metadata["s2"]["directory"]
            raise FileNotFoundError(
                f"BigEarthNet split '{split}' has no available samples under {dataset_root}. "
                f"Ensure the dataset is fully extracted (expected imagery in {data_dir}) "
                f"or update --bigearthnet-root."
            )
        datasets[split] = _SingleLabelBigEarthNet(base_dataset)

    loaders = {
        split: DataLoader(dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
        for split, dataset in datasets.items()
    }
    for split, dataset in datasets.items():
        print(f"BigEarthNet {split} dataset size: {len(dataset)} samples")
        assert len(dataset) > 0, f"BigEarthNet {split} dataset is empty!"

    return loaders, bands


def _build_neuco_loader(
    config: ModelEvalConfig,
    *,
    modalities: Sequence[str],
) -> DataLoader:
    transform_steps: List[object] = []
    # if modality.lower() != "s1":
    # if model is dino use 

    # if model is dino use SSL4EONormalize
    if config.model_type == 'croma':
        print(f"Resizing neuco to 120 for CROMA model")
        transform_steps.append(transforms.Resize((config.croma_image_resolution, config.croma_image_resolution)))
        transform_steps.append(CromaNormalize(use_8_bit=False))

    else:
        transform_steps.append(InputResizer(224))
        use_imagenet_rgb = (config.model_weights or "").lower() == "remoteclip"
        normalization_method = config.normalization_method.lower()
        if use_imagenet_rgb:
            print("Using ImageNet normalization for RemoteCLIP NeuCo inputs")
            transform_steps.append(Divideby10000Normalize())
            transform_steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        elif normalization_method == NORMALIZATION_METHOD_BANDWISE:
            print("Using NeuCo bandwise normalization for NeuCo inputs")
            transform_steps.append(NeuCoNormalize())
        elif normalization_method == NORMALIZATION_METHOD_SSL4EO:
            print("Using SSL4EO normalization for NeuCo inputs")
            transform_steps.append(SSL4EONormalize())
        else:
            print("Using divide-by-10000 normalization for NeuCo inputs")
            transform_steps.append(Divideby10000Normalize())

    # transform_steps.append(TemporalMean())  # skip temporal averaging; keep per-season inputs
    # if config.neuco_resize is not None:
    #     transform_steps.append(InputResizer(config.neuco_resize))
    transform = transforms.Compose(transform_steps)


    # print(f"Building NeuCo dataset with modalities: {modalities}")
    rgb = True if config.model_in_channels == 3 else False
    dataset = E2SChallengeDataset(
        data_path=str(config.neuco_root),
        modalities=list(modalities),
        seasons=config.neuco_seasons,
        concat=True,
        output_file_name=True,
        transform=transform,
        rgb=rgb
    )

    # check size of datset
    # print(f"NeuCo dataset size: {len(dataset)} samples")
    assert len(dataset) > 0, "NeuCo dataset is empty!"
    # quit()

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        # collate_fn=collate_fn,
    )
    # print shape of first batch
    for batch in loader:
        if isinstance(batch, dict):
            image = batch.get("image") if "image" in batch else batch.get("data")
        else:
            image, _ = batch
        # print(f"NeuCo batch image shape: {image.shape if isinstance(image, torch.Tensor) else 'N/A'}")
    return loader

def _build_ssl4eo_dataset(config: ModelEvalConfig) -> torch.utils.data.Dataset:
    if config.ssl4eo_root is None:
        raise RuntimeError("SSL4EO dataset root must be provided for diagnostics")

    ensure_hydra_original_cwd()

    s2_transform = select_ssl4eo_transform(config.model_weights)
    if s2_transform is None:
        print(f"Using default SSL4EO transform for tier {config.ssl4eo_s2_tier}")
        norm_layer = (
            SSL4EONormalize()
            if config.normalization_method.lower() == NORMALIZATION_METHOD_BANDWISE
            else Divideby10000Normalize()
        )
        s2_transform = transforms.Compose([
                transforms.CenterCrop(224),
                norm_layer,
            ])

    dataset = SSL4EODataset(
        root=str(config.ssl4eo_root.expanduser()),
        s2_tier=str(config.ssl4eo_s2_tier),
        seasons=[0,1,2,3],
        num_timestamps=4,
        # s2_bands=list(config.ssl4eo_s2_bands),
        transforms={'s1': data.get_transform('s1', is_train=False), 's2': s2_transform},
        is_train=False,

        # s2_tier=config.neuco_modalities[0],  # use first modality as tier
    )

    total = len(dataset)
    subset_size = config.ssl4eo_subset_size
    if subset_size > 0 and subset_size < total:
        indices = sorted(np.random.choice(total, size=subset_size, replace=False).tolist())
        dataset = Subset(dataset, indices)
    return dataset


def _extract_ssl4eo_embeddings(
    config: ModelEvalConfig,
    model: nn.Module,
    *,
    device: torch.device,
    max_batches_cka: int
) -> Tuple[ModalityEmbeddings, ModalityEmbeddings, List[str]]:
    dataset = _build_ssl4eo_dataset(config)

    # Ensure S2 inputs match model channel expectations (e.g., drop B1/B9/B10 for 10-ch MS-CLIP).
    if config.model_in_channels == 10 and isinstance(getattr(dataset, "transforms", None), dict):
        orig_s2 = dataset.transforms.get("s2")

        def _align_transform(tensor: torch.Tensor) -> torch.Tensor:
            # Apply original transform first (normalization expects original channel count),
            # then drop bands to the 10-channel MS-CLIP layout.
            transformed = orig_s2(tensor) if callable(orig_s2) else tensor
            return _align_s2_channels(transformed, 10)

        dataset.transforms["s2"] = _align_transform

    autocast = (
        (lambda: torch.cuda.amp.autocast())
        if device.type == "cuda"
        else contextlib.nullcontext
    )

    input_dtype = getattr(model, "dtype_s2", torch.float32)
    if device.type != "cuda" and input_dtype in {torch.float16, torch.bfloat16}:
        input_dtype = torch.float32

    s1_embeddings, s2_embeddings, sample_ids = extract_embeddings_for_dataset(
        model,
        dataset,
        input_dtype=input_dtype,
        device=device,
        autocast=autocast,
        max_batches_cka=max_batches_cka
    )

    assert s2_embeddings is not None, "S2 embeddings should not be None"
    return s1_embeddings, s2_embeddings, sample_ids


def _run_linear_probe(
    config: ModelEvalConfig,
    embeddings: Dict[str, EmbeddingBundle],
    *,
    output_dir: Path,
    label: str,
    norm_method: str,
    slice_dim: Optional[int] = None,
    feature_override: Optional[str] = None,
) -> None:
    percents = (0.01, 0.1, 1.0)
    rng = np.random.default_rng(config.random_seed)

    def _maybe_slice(features: np.ndarray) -> np.ndarray:
        if slice_dim is None:
            return features
        if features.shape[1] < slice_dim:
            raise ValueError(
                f"Requested matryoshka dim {slice_dim} exceeds feature dim {features.shape[1]}."
            )
        return features[:, :slice_dim]

    def _evaluate_logreg(feature_key: str, batch_norm: bool) -> Dict[float, Dict[str, float]]:
        results: Dict[float, Dict[str, float]] = {}
        # print(embeddings)
        train = embeddings["train"]
        val = embeddings["val"]
        test = embeddings["test"]

        # raise NotImplementedError("Debugging embeddings extraction")

        train_feats = _maybe_slice(getattr(train, feature_key))
        val_feats = _maybe_slice(getattr(val, feature_key))
        test_feats = _maybe_slice(getattr(test, feature_key))

        print(
            f"Linear probe ({label}) using '{feature_key}' embeddings: dim={train_feats.shape[1]} "
            f"(batch_norm={batch_norm}, slice_dim={slice_dim})"
        )

        if batch_norm:
            mean = train_feats.mean(axis=0, keepdims=True)
            std = train_feats.std(axis=0, keepdims=True) + 1e-10
            train_feats = (train_feats - mean) / std
            val_feats = (val_feats - mean) / std
            test_feats = (test_feats - mean) / std

        has_multi = (
            train.multi_labels is not None
            and val.multi_labels is not None
            and test.multi_labels is not None
        )

        for pct in percents:
            n_samples = max(1, int(pct * len(train_feats)))
            indices = rng.choice(len(train_feats), size=n_samples, replace=False)

            clf = LogisticRegression(max_iter=4000, multi_class="multinomial", solver="lbfgs")
            clf.fit(train_feats[indices], train.labels[indices])

            val_pred = clf.predict(val_feats)
            test_pred = clf.predict(test_feats)

            val_acc = accuracy_score(val.labels, val_pred)
            val_f1 = f1_score(val.labels, val_pred, average="weighted")
            test_acc = accuracy_score(test.labels, test_pred)
            test_f1 = f1_score(test.labels, test_pred, average="weighted")

            results[pct] = {
                "val_accuracy": float(val_acc),
                "val_f1": float(val_f1),
                "test_accuracy": float(test_acc),
                "test_f1": float(test_f1),
            }

            if has_multi:
                ovr = OneVsRestClassifier(
                    LogisticRegression(max_iter=4000, solver="lbfgs")
                )
                ovr.fit(train_feats[indices], train.multi_labels[indices])
                val_scores = ovr.predict_proba(val_feats)
                test_scores = ovr.predict_proba(test_feats)
                results[pct]["val_map_micro"] = float(
                    average_precision_score(val.multi_labels, val_scores, average="micro")
                )
                results[pct]["test_map_micro"] = float(
                    average_precision_score(test.multi_labels, test_scores, average="micro")
                )

        return results

    def _evaluate_knn(feature_key: str, batch_norm: bool, n_neighbors: int = 20) -> Dict[float, Dict[str, float]]:
        """Use train features as keys and classify val/test with kNN (k=20) across the same percents."""
        results: Dict[float, Dict[str, float]] = {}
        train = embeddings["train"]
        val = embeddings["val"]
        test = embeddings["test"]

        train_feats = _maybe_slice(getattr(train, feature_key))
        val_feats = _maybe_slice(getattr(val, feature_key))
        test_feats = _maybe_slice(getattr(test, feature_key))

        if batch_norm:
            mean = train_feats.mean(axis=0, keepdims=True)
            std = train_feats.std(axis=0, keepdims=True) + 1e-10
            train_feats = (train_feats - mean) / std
            val_feats = (val_feats - mean) / std
            test_feats = (test_feats - mean) / std

        for pct in percents:
            n_samples = max(1, int(pct * len(train_feats)))
            indices = rng.choice(len(train_feats), size=n_samples, replace=False)
            k = min(n_neighbors, n_samples)

            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(train_feats[indices], train.labels[indices])

            val_pred = clf.predict(val_feats)
            test_pred = clf.predict(test_feats)

            val_acc = accuracy_score(val.labels, val_pred)
            val_f1 = f1_score(val.labels, val_pred, average="weighted")
            test_acc = accuracy_score(test.labels, test_pred)
            test_f1 = f1_score(test.labels, test_pred, average="weighted")

            results[pct] = {
                "val_accuracy": float(val_acc),
                "val_f1": float(val_f1),
                "test_accuracy": float(test_acc),
                "test_f1": float(test_f1),
            }

        return results

    def _available_feature(name: str) -> bool:
        feature = getattr(embeddings["train"], name)
        return feature is not None and feature.size > 0

    probe_specs = []
    marker_map = {"backbone": "o", "projected": "s"}

    if feature_override is not None:
        if not _available_feature(feature_override):
            logging.warning("Requested feature '%s' not available for linear probe; skipping.", feature_override)
            return
        probe_specs.extend(
            [
                (feature_override, f"{feature_override}_batchnorm", True),
            ]
        )
    elif _available_feature("backbone"):
        probe_specs.extend(
            [
                ("backbone", "backbone_batchnorm", True),
            ]
        )

    #     if _available_feature(name):
    #         probe_specs.append((name, name, False))
    #         probe_specs.append((name, f"{name}_batchnorm", True))

    if not probe_specs:
        logging.warning("No embeddings available for linear probe; skipping")
        return


    method_tag = norm_method.lower()
    plots_dir = output_dir / f"linear_probe_{method_tag}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for feature_key, suffix, use_batch_norm in probe_specs:
        lr_metrics = _evaluate_logreg(feature_key, batch_norm=use_batch_norm)
        knn_metrics = _evaluate_knn(feature_key, batch_norm=use_batch_norm)
        with (plots_dir / f"{label}_{suffix}_metrics.json").open("w") as handle:
            json.dump(lr_metrics, handle, indent=2)
        with (plots_dir / f"{label}_{suffix}_knn_metrics.json").open("w") as handle:
            json.dump(knn_metrics, handle, indent=2)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    

    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        acc = [metrics[str(x)]["test_accuracy"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        ax1.plot(xs, acc, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})")

    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        f1_vals = [metrics[str(x)]["test_f1"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        ax2.plot(xs, f1_vals, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})")

    ax1.set_xlabel("Training fraction")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title(f"Test Accuracy ({label})")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Training fraction")
    ax2.set_ylabel("Test F1")
    ax2.set_title(f"Test F1 ({label})")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / f"{label}_combined_curves.png", dpi=200)
    plt.close(fig)

    # KNN plots
    fig_knn, (knn_ax1, knn_ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_knn_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        acc = [metrics[str(x)]["test_accuracy"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        knn_ax1.plot(xs, acc, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})")

    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_knn_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        f1_vals = [metrics[str(x)]["test_f1"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        knn_ax2.plot(xs, f1_vals, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})")

    knn_ax1.set_xlabel("Training fraction")
    knn_ax1.set_ylabel("Test Accuracy")
    knn_ax1.set_title(f"KNN Test Accuracy ({label})")
    knn_ax1.grid(alpha=0.3)
    knn_ax1.legend()

    knn_ax2.set_xlabel("Training fraction")
    knn_ax2.set_ylabel("Test F1")
    knn_ax2.set_title(f"KNN Test F1 ({label})")
    knn_ax2.grid(alpha=0.3)
    knn_ax2.legend()

    fig_knn.tight_layout()
    fig_knn.savefig(plots_dir / f"{label}_knn_combined_curves.png", dpi=200)
    plt.close(fig_knn)

    logging.info("Linear probe results saved to %s", plots_dir)


def _plot_eurosat_tsne(
    bundle: EmbeddingBundle,
    *,
    output_dir: Path,
    label: str,
    max_samples: int,
    seed: int,
    model_title: Optional[str],
) -> None:
    if bundle.labels is None:
        logging.warning("No labels provided for EuroSAT embeddings; skipping t-SNE plot")
        return

    feature_types = [
        ("backbone", bundle.backbone, "Backbone"), 
        ("projected", bundle.projected, "Projected")
    ]

    # Filter to only non-None features with data
    available_features = [
        (name, features, display_name) 
        for name, features, display_name in feature_types 
        if features is not None and features.size > 0
    ]

    if not available_features:
        logging.warning("No embeddings available for EuroSAT t-SNE plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    for feature_name, features, display_name in available_features:
        labels = bundle.labels
        total_samples = len(features)
        rng = np.random.default_rng(seed)
        
        # Sample if needed
        if total_samples > max_samples:
            indices = rng.choice(total_samples, size=max_samples, replace=False)
            features_sampled = features[indices]
            labels_sampled = labels[indices]
        else:
            features_sampled = features
            labels_sampled = labels

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
        embeddings_2d = tsne.fit_transform(features_sampled)

        # Plot
        unique_labels = sorted(np.unique(labels_sampled))
        cmap = plt.get_cmap("tab20", len(unique_labels))

        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, class_label in enumerate(unique_labels):
            mask = labels_sampled == class_label
            label_text = EUROSAT_CLASS_NAMES.get(int(class_label), str(class_label))
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                s=8,
                color=cmap(idx),
                label=label_text,
                alpha=0.7,
            )

        title = model_title if model_title is not None else "EuroSAT t-SNE"
        ax.set_title(f"{title} - {display_name}")
        ax.set_xlabel("t-SNE component 1")
        ax.set_ylabel("t-SNE component 2")
        ax.legend(title="Class", fontsize="small", markerscale=2)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_tsne_{feature_name}.png", dpi=200)
        plt.close(fig)
        
        # logging.info(f"Saved t-SNE plot for {display_name} features: {output_dir / f'{label}_tsne_{feature_name}.png'}")


def _plot_eurosat_pca(
    bundle: EmbeddingBundle,
    *,
    output_dir: Path,
    label: str,
    max_samples: int,
    seed: int,
    model_title: Optional[str],
    modality_label: str,
) -> None:
    feature_types = [
        ("backbone", bundle.backbone, "Backbone"),
        ("projected", bundle.projected, "Projected"),
    ]
    available_features = [
        (name, features, display_name)
        for name, features, display_name in feature_types
        if features is not None and features.size > 0
    ]
    if not available_features:
        logging.warning("No embeddings available for EuroSAT PCA plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    modality_label = modality_label.upper()
    palette = {"S2": "#1f77b4", "S1": "#ff7f0e"}

    for feature_name, features, display_name in available_features:
        total_samples = len(features)
        if total_samples > max_samples:
            indices = rng.choice(total_samples, size=max_samples, replace=False)
            features_sampled = features[indices]
        else:
            features_sampled = features

        pca = PCA(n_components=2, random_state=seed)
        embeddings_2d = pca.fit_transform(features_sampled)

        fig, ax = plt.subplots(figsize=(8, 6))
        label_array = np.repeat(modality_label, len(embeddings_2d))
        unique_labels = sorted(np.unique(label_array))

        for idx, modality in enumerate(unique_labels):
            mask = label_array == modality
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                s=12,
                color=palette.get(modality, plt.get_cmap("tab10")(idx)),
                label=modality,
                alpha=0.8,
            )

        ax.set_title(f"PCA ({feature_name}, raw)")
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        if len(unique_labels) > 1:
            ax.legend(title="Modality", fontsize="small", markerscale=2)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_pca_{feature_name}.png", dpi=200)
        plt.close(fig)

        # logging.info(
        #     "Saved PCA plot for %s features: %s",
        #     display_name,
        #     output_dir / f"{label}_pca_{feature_name}.png",
        # )


def _export_neuco(
    bundle: EmbeddingBundle,
    output_dir: Path,
    *,
    label: str,
) -> None:
    if bundle.ids is None:
        raise RuntimeError("NeuCo export requires file identifiers")

    def _write(features: np.ndarray, suffix: str) -> None:
        rows = {idx: vec for idx, vec in zip(bundle.ids or [], features)}
        df = create_submission_from_dict(rows)
        df.to_csv(output_dir / f"neuco_{label}_{suffix}.csv", index=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle.backbone is not None:
        _write(bundle.backbone, "backbone")
    # if bundle.projected is not None:
    #     _write(bundle.projected, "projected")


def _run_embedding_diagnostics(
    config: ModelEvalConfig,
    s1_bundle: Optional[ModalityEmbeddings],
    s2_bundle: ModalityEmbeddings,
    *,
    sample_ids: Optional[Sequence[str]] = None,
    output_dir: Path,
) -> None:
    print(f"Running embedding diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    import CKA
    device = s2_bundle.backbone.device if s2_bundle.backbone is not None else torch.device("cpu")
    # print(device)
    cuda_cka = CKA.CudaCKA(device='cuda')

    def _detach_or_none(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return tensor.detach() if tensor is not None else None

    def _prepare_modality(bundle: Optional[ModalityEmbeddings]) -> Optional[Dict[str, object]]:
        if bundle is None:
            return None
        return {
            "backbone": _detach_or_none(getattr(bundle, "backbone", None)),
            "projected": _detach_or_none(getattr(bundle, "projected", None)),
            "layers": bundle.layer_activations or {},
        }

    modality_tensors: Dict[str, Optional[Dict[str, object]]] = {
        "s1": _prepare_modality(s1_bundle),
        "s2": _prepare_modality(s2_bundle),
    }
    if modality_tensors["s2"] is None:
        raise ValueError("S2 embeddings are required for diagnostics")

    def _maybe_singular(modality: str, feature: str) -> Optional[np.ndarray]:
        tensors = modality_tensors.get(modality)
        if tensors is None:
            return None
        tensor = tensors.get(feature)
        if tensor is None:
            return None
        return compute_singular_values(tensor)  # type: ignore[arg-type]

    # Within-encoder CKA containers
    s1_layers: List[str] = []
    s1_within: Optional[np.ndarray] = None
    s1_odd_layers: List[str] = []
    s1_odd_within: Optional[np.ndarray] = None
    s1_even_layers: List[str] = []
    s1_even_within: Optional[np.ndarray] = None
    s1_residual_layers: List[str] = []
    s1_residual_within: Optional[np.ndarray] = None

    s2_layers: List[str] = []
    s2_within: Optional[np.ndarray] = None
    s2_odd_layers: List[str] = []
    s2_odd_within: Optional[np.ndarray] = None
    s2_even_layers: List[str] = []
    s2_even_within: Optional[np.ndarray] = None
    s2_residual_layers: List[str] = []
    s2_residual_within: Optional[np.ndarray] = None
    s1_group_layers: Dict[str, List[str]] = {}
    s1_group_within: Dict[str, Optional[np.ndarray]] = {}
    s2_group_layers: Dict[str, List[str]] = {}
    s2_group_within: Dict[str, Optional[np.ndarray]] = {}

    # Cross-encoder CKA containers
    cross_s1_layers: List[str] = []
    cross_s2_layers: List[str] = []
    cross_matrix: Optional[np.ndarray] = None

    projected_cka: Optional[float] = None

    def _collect_transformer_groups(layer_dict: Dict[str, torch.Tensor]):
        group_layers: Dict[str, List[str]] = {}
        group_within: Dict[str, Optional[np.ndarray]] = {}

        names = list(layer_dict.keys())
        is_scalemae_style = any(
            pattern in name
            for name in names
            for pattern in (".layernorm", ".scale", ".op")
        )

        if is_scalemae_style:
            specs = {
                "layernorm": [k for k in names if ".layernorm" in k],
                "scale": [k for k in names if ".scale" in k],
                "op": [k for k in names if ".op" in k],
                "residual": [k for k in names if ".residual" in k],
            }
        else:
            specs = {
                "ln": [k for k in names if (k.endswith(".ln") or ".ln." in k)],
                "core": [k for k in names if (k.endswith(".core") or ".core." in k)],
                "residual": [k for k in names if ".residual" in k],
            }

        for group, keys in specs.items():
            subset = {k: layer_dict[k] for k in keys}
            if subset:
                layers, cka_matrix = compute_within_encoder_cka(subset, cuda_cka)
            else:
                layers, cka_matrix = [], None
            group_layers[group] = layers
            group_within[group] = cka_matrix

        return group_layers, group_within

    # --------------------
    # S1 within-encoder CKA
    # --------------------
    if modality_tensors["s1"] and modality_tensors["s1"]["layers"]:
        s1_all_dict = modality_tensors["s1"]["layers"]  # type: ignore[arg-type]
        # print
        s1_layers, s1_within = compute_within_encoder_cka(s1_all_dict, cuda_cka) # 32x32

        layer_names = list(s1_all_dict.keys())
        has_resnet_hooks = any(name.endswith(".bn2") for name in layer_names)
        has_transformer_hooks = any(
            pattern in name
            for name in layer_names
            for pattern in (".layernorm", ".ln", ".op", ".core", ".scale", ".residual")
        )

        if has_resnet_hooks:
            s1_odd_dict = {k: v for k, v in s1_all_dict.items() if k.endswith(".bn2")}
            if s1_odd_dict:
                s1_odd_layers, s1_odd_within = compute_within_encoder_cka(s1_odd_dict, cuda_cka)

            s1_even_dict = {k: v for k, v in s1_all_dict.items() if k.endswith(".block_out")}
            if s1_even_dict:
                s1_even_layers, s1_even_within = compute_within_encoder_cka(s1_even_dict, cuda_cka)

            s1_group_layers = {
                "odd": s1_odd_layers,
                "even": s1_even_layers,
            }
            s1_group_within = {
                "odd": s1_odd_within,
                "even": s1_even_within,
            }

        if has_transformer_hooks:
            s1_group_layers, s1_group_within = _collect_transformer_groups(s1_all_dict)

            # Do not reuse odd/even slots for transformer groups
            s1_odd_layers, s1_odd_within = [], None
            s1_even_layers, s1_even_within = [], None
            s1_residual_layers, s1_residual_within = [], None

    elif s1_bundle is not None:
        logging.warning("S1 layer activations unavailable; skipping S1 CKA diagnostics")

    # --------------------
    # S2 within-encoder CKA
    # --------------------
    if modality_tensors["s2"] and modality_tensors["s2"]["layers"]:
        s2_all_dict = modality_tensors["s2"]["layers"]  # type: ignore[arg-type]

        # print('')
        s2_layers, s2_within = compute_within_encoder_cka(s2_all_dict, cuda_cka)

        layer_names = list(s2_all_dict.keys())
        has_resnet_hooks = any(name.endswith(".bn2") for name in layer_names)
        has_transformer_hooks = any(
            pattern in name
            for name in layer_names
            for pattern in (".layernorm", ".ln", ".op", ".core", ".scale", ".residual")
        )

        if has_resnet_hooks:
            s2_odd_dict = {k: v for k, v in s2_all_dict.items() if k.endswith(".bn2")}
            if s2_odd_dict:
                s2_odd_layers, s2_odd_within = compute_within_encoder_cka(s2_odd_dict, cuda_cka)

            s2_even_dict = {k: v for k, v in s2_all_dict.items() if k.endswith(".block_out")}
            if s2_even_dict:
                s2_even_layers, s2_even_within = compute_within_encoder_cka(s2_even_dict, cuda_cka)

        if has_transformer_hooks:
            s2_group_layers, s2_group_within = _collect_transformer_groups(s2_all_dict)

            s2_odd_layers, s2_odd_within = [], None
            s2_even_layers, s2_even_within = [], None
            s2_residual_layers, s2_residual_within = [], None

    else:
        logging.warning("S2 layer activations unavailable; skipping S2 CKA diagnostics")

    # --------------------
    # Cross-encoder CKA (unchanged)
    # --------------------
    if (
        modality_tensors["s1"]
        and modality_tensors["s1"]["layers"]
        and modality_tensors["s2"]
        and modality_tensors["s2"]["layers"]
    ):
        cross_s1_layers, cross_s2_layers, cross_matrix = compute_cross_encoder_cka(
            modality_tensors["s1"]["layers"],  # type: ignore[arg-type]
            modality_tensors["s2"]["layers"],  # type: ignore[arg-type]
            cuda_cka,
        )


    s1_projected_singular = _maybe_singular("s1", "projected")
    s1_backbone_singular = _maybe_singular("s1", "backbone")
    s2_projected_singular = _maybe_singular("s2", "projected")
    s2_backbone_singular = _maybe_singular("s2", "backbone")

   

    # ----- Package CKA results (full, odd, even, cross, projected) -----
    cka_payload: Dict[str, object] = {
        "s1_full": None,
        "s1_odd": None,
        "s1_even": None,
        "s2_full": None,
        "s2_odd": None,
        "s2_even": None,
        "cross": None,
        "projected_similarity": projected_cka,
    }

    # S1 full
    if s1_layers and s1_within is not None:
        cka_payload["s1_full"] = {
            "layers": s1_layers,
            "matrix": s1_within.tolist(),
        }

    # S1 odd (bn3)
    if s1_odd_layers and s1_odd_within is not None:
        cka_payload["s1_odd"] = {
            "layers": s1_odd_layers,
            "matrix": s1_odd_within.tolist(),
        }

    # S1 even (block_out)
    if s1_even_layers and s1_even_within is not None:
        cka_payload["s1_even"] = {
            "layers": s1_even_layers,
            "matrix": s1_even_within.tolist(),
        }

    if s1_group_layers:
        cka_payload["s1_groups"] = {
            group: {
                "layers": layers,
                "matrix": (
                    s1_group_within[group].tolist()
                    if s1_group_within.get(group) is not None
                    else None
                ),
            }
            for group, layers in s1_group_layers.items()
        }

    # S2 full
    if s2_layers and s2_within is not None:
        cka_payload["s2_full"] = {
            "layers": s2_layers,
            "matrix": s2_within.tolist(),
        }

    # S2 odd (bn3)
    if s2_odd_layers and s2_odd_within is not None:
        cka_payload["s2_odd"] = {
            "layers": s2_odd_layers,
            "matrix": s2_odd_within.tolist(),
        }

    # S2 even (block_out)
    if s2_even_layers and s2_even_within is not None:
        cka_payload["s2_even"] = {
            "layers": s2_even_layers,
            "matrix": s2_even_within.tolist(),
        }

    if s2_group_layers:
        cka_payload["s2_groups"] = {
            group: {
                "layers": layers,
                "matrix": (
                    s2_group_within[group].tolist()
                    if s2_group_within.get(group) is not None
                    else None
                ),
            }
            for group, layers in s2_group_layers.items()
        }

    # Cross-encoder CKA
    if cross_matrix is not None:
        cka_payload["cross"] = {
            "s1_layers": cross_s1_layers,
            "s2_layers": cross_s2_layers,
            "matrix": cross_matrix.tolist(),
        }

    # (optional) write to JSON, if you want it on disk:
    # with (output_dir / "cka.json").open("w", encoding="utf-8") as handle:
    #     json.dump(cka_payload, handle, indent=2)


    spectra: List[Tuple[str, str, np.ndarray]] = []
    if s1_projected_singular is not None:
        spectra.append(("s1", "projected", s1_projected_singular))
    if s1_backbone_singular is not None:
        spectra.append(("s1", "backbone", s1_backbone_singular))
    if s2_projected_singular is not None:
        spectra.append(("s2", "projected", s2_projected_singular))
    if s2_backbone_singular is not None:
        spectra.append(("s2", "backbone", s2_backbone_singular))

    # # check type of payload
    # for entry in cka_payload.items():
    #     key, value = entry
    #     if value is not None and not isinstance(value, dict):
    #         print(value)
    #         # raise ValueError(f"CKA payload '{key}' is not a dict as expected")

    #     #     logging.info(f"CKA payload '{key}' has keys: {list(value.keys())}")
    #     # if type is tensor, conver to np
    #     if isinstance(value, torch.Tensor):
    #         cka_payload[key] = value.cpu().numpy()
    #         print(f"Converted CKA payload '{key}' from tensor to numpy array")
    # with (output_dir / "cka.json").open("w", encoding="utf-8") as handle:
    #     json.dump(cka_payload, handle, indent=2)

    label_source = (
        (config.checkpoint.stem if config.checkpoint is not None else None)
        or config.model_weights
        or config.model_type
    )

    epoch_source = (
        (config.checkpoint.stem if config.checkpoint is not None else "")
        or (config.model_weights or "")
    )
    epoch_match = re.search(r"epoch[_=-]?(\d+)", epoch_source)
    epoch_value = int(epoch_match.group(1)) if epoch_match else 0

    if config.model_type == "torchgeo_resnet50":
        epoch_value = None
        label_source = config.model_weights

    s2_raw = getattr(s2_bundle, "raw", None)
    sample_count = int(s2_raw.shape[0]) if s2_raw is not None and s2_raw.ndim > 0 else 0
    diagnostic_ids = (
        [str(item) for item in sample_ids]
        if sample_ids is not None and len(sample_ids) > 0
        else [str(index) for index in range(sample_count)]
    )

    def _as_array(values: Optional[np.ndarray]) -> np.ndarray:
        return values if values is not None else np.empty(0, dtype=np.float32)

    epoch_diagnostics = EpochDiagnostics(
        label=label_source,
        epoch=epoch_value,
        ids=diagnostic_ids,
        s1=s1_bundle,
        s2=s2_bundle,
        s1_singular_values=_as_array(s1_projected_singular),
        s2_singular_values=_as_array(s2_projected_singular),
        s1_layers=s1_layers,
        s2_layers=s2_layers,
        s1_within_cka=s1_within,
        s2_within_cka=s2_within,
        cross_cka=cross_matrix,
        cross_s1_layers=cross_s1_layers,
        cross_s2_layers=cross_s2_layers,
        ## add
        s1_odd_layers=s1_odd_layers,
        s1_odd_within_cka=s1_odd_within,
        s1_even_layers=s1_even_layers,
        s1_even_within_cka=s1_even_within,
        s1_residual_layers=s1_residual_layers,
        s1_residual_within_cka=s1_residual_within,
        s2_odd_layers=s2_odd_layers,
        s2_odd_within_cka=s2_odd_within,
        s2_even_layers=s2_even_layers,
        s2_even_within_cka=s2_even_within,
        s2_residual_layers=s2_residual_layers,
        s2_residual_within_cka=s2_residual_within,
        s1_group_layers=s1_group_layers,
        s1_group_within_cka=s1_group_within,
        s2_group_layers=s2_group_layers,
        s2_group_within_cka=s2_group_within,
    )

    can_plot_full = (
        s1_bundle is not None
        and modality_tensors["s1"] is not None
    )
    can_plot_s2_only = (
        s1_bundle is None
    )

    normalized_weights = (config.model_weights or "").lower()
    is_transformer = config.model_type in ("croma", "croma_vit", "croma_s1s2") or any(
        token in normalized_weights for token in ("dofa", "scalemae", "vitsmall", "vit", 'visiontransformer')
    )
    is_scalemae = any(token in normalized_weights for token in ("scalemae", "dofa", 'visiontransformer', 'vit'))
    print(f"Model type: {config.model_type}, Weights: {normalized_weights}, is_transformer: {is_transformer}, is_scalemae: {is_scalemae}")
    is_resnet_based = (
        config.model_type == "torchgeo_resnet50"
        or normalized_weights.startswith("resnet")
        or normalized_weights == "rcf_13ch"
    )
    is_ciip_or_moco = config.model_type == "ciip_checkpoint" or (
        config.model_type == "torchgeo_resnet50" and normalized_weights == "moco"
    )

    # if resnet architecture, plot even-layer cka and odd-layer cka
    plot_even_odd = is_resnet_based or config.model_type == "ciip_checkpoint"

    if is_transformer:
        if is_scalemae:
            plot_epoch_diagnostics_scalemae(
                epoch_diagnostics, output_dir, label="transformer_hooks"
            )
        else:
            plot_epoch_diagnostics_croma(
                epoch_diagnostics, output_dir, label="transformer_hooks"
            )
    elif is_ciip_or_moco:
        if can_plot_full:
            plot_epoch_diagnostics(
                epoch_diagnostics,
                output_dir,
                label="embeddings_raw",
                plot_even_odd=True,
            )
        else:
            logging.info(
                "Skipping full epoch diagnostics plot; required modalities missing for %s",
                config.model_type,
            )
    elif is_resnet_based and (can_plot_s2_only or can_plot_full):
        plot_epoch_diagnostics_s2only(
            epoch_diagnostics,
            output_dir,
            label="s2_backbone_raw",
            plot_even_odd=plot_even_odd,
        )
    else:
        logging.info("Skipping epoch diagnostics plots due to missing embedding features")

    if not spectra:
        logging.warning("No features available for singular value diagnostics")
    else:
        n = len(spectra)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols)) if ncols > 0 else 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax, (modality, feature_name, spectrum) in zip(axes, spectra):
            ax.plot(np.arange(len(spectrum)), spectrum, marker=".")
            ax.set_xlabel("Component")
            ax.set_ylabel("Singular value")
            ax.set_title(f"{modality.upper()} {feature_name}")
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 100)

        for ax in axes[len(spectra):]:
            ax.axis("off")

        fig.suptitle("Singular Values per Modality / Feature", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(output_dir / "singular_values_all.png", dpi=200)
        plt.close(fig)

    rng = np.random.default_rng(config.random_seed)

    def _stack_features(feature_name: str) -> Tuple[np.ndarray, np.ndarray]:
        tensors: List[np.ndarray] = []
        label_arrays: List[np.ndarray] = []
        if modality_tensors["s1"] and modality_tensors["s1"].get(feature_name) is not None:
            s1_np = modality_tensors["s1"][feature_name].cpu().numpy()  # type: ignore[index]
            tensors.append(s1_np)
            label_arrays.append(np.repeat("S1", len(s1_np)))
        if modality_tensors["s2"] and modality_tensors["s2"].get(feature_name) is not None:
            s2_np = modality_tensors["s2"][feature_name].cpu().numpy()  # type: ignore[index]
            tensors.append(s2_np)
            label_arrays.append(np.repeat("S2", len(s2_np)))
        if not tensors:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype="<U2")
        features = np.concatenate(tensors, axis=0)
        labels = np.concatenate(label_arrays, axis=0)
        return features, labels

    def _sample(features: np.ndarray, labels: np.ndarray, maximum: int) -> Tuple[np.ndarray, np.ndarray]:
        if maximum <= 0 or maximum >= len(features):
            return features, labels
        indices = rng.choice(len(features), size=maximum, replace=False)
        return features[indices], labels[indices]

    

    for feature_name in ("projected", "backbone"):
        # ciip is lorentz
        curv = None
        if 'lorentz' in config.model_type and feature_name == "projected":
            use = 'poincare'
            curv = config.curvature

        else:
            use = 'zscore'


        combined, modality_labels = _stack_features(feature_name)
        if combined.size == 0:
            continue
        for mode in ("raw", use):
            subset, subset_labels = _sample(combined, modality_labels, config.tsne_samples)
            processed = preprocess_projection_data(subset, mode=mode, random_state=config.random_seed, curvature=curv)
            coords = compute_projection(processed, method="tsne", random_state=config.random_seed)
            if coords is not None:
                suffix = mode
                plot_projection(
                    coords,
                    subset_labels,
                    output_dir / f"tsne_{feature_name}_{suffix}.png",
                    title=f"t-SNE ({feature_name}, {suffix})",
                )

            subset, subset_labels = _sample(combined, modality_labels, config.pca_samples)
            processed = preprocess_projection_data(subset, mode=mode, random_state=config.random_seed)
            if processed.shape[0] >= 2:
                suffix = "zscore" if mode == "zscore" else "raw"
                pca_coords = PCA(n_components=2, random_state=config.random_seed).fit_transform(processed)
                plot_projection(
                    pca_coords,
                    subset_labels,
                    output_dir / f"pca_{feature_name}_{suffix}.png",
                    title=f"PCA ({feature_name}, {suffix})",
                )
    print(f"Embedding diagnostics saved to {output_dir}")

def _run_hyperbolic_visualisations(
    s1_proj_features: torch.Tensor,
    s2_proj_features: torch.Tensor,
    *,
    output_dir: Path,
    model: LorentzCIIP,
    aperture_logk: Optional[float],
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = compute_hyperbolic_context(
        model,
        s1_proj_features,
        s2_proj_features,
        aperture_logk=aperture_logk,
    )
    positive_angles = context["positive_angles"].cpu().numpy()
    aperture_s1 = context["aperture_s1"].cpu().numpy()
    aperture_s2 = context["aperture_s2"].cpu().numpy()
    s1_dirs = context["s1_dirs"].cpu().numpy()
    s2_dirs = context["s2_dirs"].cpu().numpy()
    s1_distances = context["s1_distances"].cpu().numpy()
    s2_distances = context["s2_distances"].cpu().numpy()

    angles_mat = context["angles"]          # (N, N), on device
    N = angles_mat.size(0)
    # positives: diagonal α(x_i, y_i)
    pos_alpha = context["positive_angles"].cpu().numpy()
    # negatives: all off-diagonals α(x_i, y_j), i != j
    neg_alpha = angles_mat[~torch.eye(N, dtype=torch.bool, device=angles_mat.device)].cpu().numpy()

    # Convert to "similarity" the way the loss does: κ = -α
    pos_angle_sim = -pos_alpha
    neg_angle_sim = -neg_alpha

    plot_pos_neg_hist(
        pos_angle_sim,
        neg_angle_sim,
        use_distance=False,
        out_path=output_dir / "angle_sim_hist.png",
    )

    plot_angle_aperture(positive_angles, aperture_s1, aperture_s2, output_dir / "angle_aperture.png")
    plot_radial_histogram(s1_distances, s2_distances, output_dir / "radial_histogram.png")
    plot_angular_pca(s1_dirs, s2_dirs, s1_distances, s2_distances, output_dir / "angular_pca.png")
    plot_cone_polar(positive_angles, aperture_s1, aperture_s2, output_dir / "cone_polar.png", sample_size=256, seed=seed)

    

def run_full_evaluation(config: ModelEvalConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    if config.model_type == "croma":
        config.eurosat_image_size = config.croma_image_resolution

    adapter = build_evaluation_adapter(
        model_type=config.model_type,
        checkpoint=config.checkpoint,
        model_weights=config.model_weights,
        in_chans=config.model_in_channels,
        croma_weights=config.croma_weights,
        croma_image_resolution=config.croma_image_resolution,
        ciip_framework=config.ciip_framework,
        enable_s1=config.evaluation_modality.lower() == "s1",
    )
    print(f'Model loaded')
    # print(adapter)
    device = torch.device(torch.device("cuda") if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise ValueError
    print(device)
    adapter = adapter.to(device)
    adapter.eval()
    base_model = getattr(adapter, "base_model", adapter)
    is_lorentz = getattr(adapter, "is_lorentz", False)
    # save curvature to config if lorentz
    if is_lorentz and hasattr(base_model, "curvature"):
        config.curvature = float(base_model.curvature)
        print(f"Lorentz curvature: {config.curvature}")


    output_root = Path(config.output_dir)
    os.makedirs(output_root, exist_ok=True)
    
    target_modality = config.evaluation_modality.lower()
    if target_modality not in {"s1", "s2"}:
        raise ValueError("evaluation_modality must be either 's1' or 's2'")

    model_channels = _infer_model_in_channels(
        adapter, config.model_in_channels, modality=target_modality
    )
    thirteen_band_models = {
        "dofa_base_s2_13ch",
        "vitsmall16_s2_all_moco",
        "resnet18_s2_all_moco",
        "moco",
        "dino",
    }
    if (config.model_weights and config.model_weights in thirteen_band_models) or (
        config.model_type and config.model_type in thirteen_band_models
    ):
        model_channels = max(model_channels, 13)

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    normalization_method = config.normalization_method.lower()
    if normalization_method not in NORMALIZATION_METHODS:
        raise ValueError(
            "normalization_method must be one of "
            f"{', '.join(NORMALIZATION_METHODS)}; got {config.normalization_method!r}"
        )


    if args.disable_eurosat == False:
        if target_modality == "s1":
            eurosat_bands = EUROSAT_S1_BANDS
        else:
            eurosat_bands = _resolve_eurosat_bands(model_channels)
        logging.info(
            "EuroSAT linear probe will use %d Sentinel-%s bands (%s)",
            len(eurosat_bands),
            "1" if target_modality == "s1" else "2",
            ", ".join(eurosat_bands),
        )
        

        # check if dir exists
        eurosat_output_dir = output_dir / f"linear_probe_{normalization_method}"
        if True:
            eurosat_loaders = _build_eurosat_loaders(
                config, bands=eurosat_bands, modality=target_modality
            )
            _print_band_stats(
                "EuroSAT eval",
                eurosat_loaders.get("val", next(iter(eurosat_loaders.values()))),
                bands=eurosat_bands,
                max_batches=config.stats_max_batches,
            )
            eurosat_embeddings: Dict[str, EmbeddingBundle] = {}
            with _use_adapter_modality(adapter, target_modality):
                for split, loader in eurosat_loaders.items():
                    eurosat_embeddings[split] = _extract_embeddings(
                        adapter,
                        loader,
                        device=device,
                        expected_in_channels=model_channels,
                        modality=target_modality,
                    )
            _run_linear_probe(
                config,
                eurosat_embeddings,
                output_dir=output_dir,
                label="eurosat",
                norm_method=normalization_method,
            )
            if config.matryoshka_dims:
                for dim in config.matryoshka_dims:
                    mat_output_dir = output_dir / f"matryoshka_dim_{dim}"
                    _run_linear_probe(
                        config,
                        eurosat_embeddings,
                        output_dir=mat_output_dir,
                        label="eurosat",
                        norm_method=normalization_method,
                        slice_dim=dim,
                        feature_override="backbone",
                    )
            _plot_eurosat_tsne(
                eurosat_embeddings["test"],
                output_dir=eurosat_output_dir,
                label="eurosat",
                max_samples=config.tsne_samples,
                seed=config.random_seed,
                model_title=config.model_path,
            )
            # _plot_eurosat_pca(
            #     eurosat_embeddings["test"],
            #     output_dir=eurosat_output_dir,
            #     label="eurosat",
            #     max_samples=config.pca_samples,
            #     seed=config.random_seed,
            #     model_title=config.model_path,
            #     modality_label=target_modality,
            # )
            # print("EuroSAT PCA plot saved.")

        # clean up dataset, not needed anymore
        del eurosat_loaders
        del eurosat_embeddings


    if not args.disable_bigearthnet:
        print("Running BigEarthNet linear probe...")
        bigearthnet_loaders, bigearthnet_bands = _build_bigearthnet_loaders(
            config, expected_channels=model_channels
        )
        _print_band_stats(
            "BigEarthNet eval",
            bigearthnet_loaders.get("val", next(iter(bigearthnet_loaders.values()))),
            bands=bigearthnet_bands,
            max_batches=config.stats_max_batches,
        )
        print("BigEarthNet loaders built.")
        bigearthnet_embeddings: Dict[str, EmbeddingBundle] = {}
        with _use_adapter_modality(adapter, "s2"):
            for split, loader in bigearthnet_loaders.items():
                bigearthnet_embeddings[split] = _extract_embeddings(
                    adapter,
                    loader,
                    device=device,
                    expected_in_channels=model_channels,
                    modality="s2",
                )
        print("BigEarthNet embeddings extracted.")
        if config.matryoshka_dims:
            for dim in config.matryoshka_dims:
                mat_output_dir = output_dir / f"matryoshka_dim_{dim}"
                _run_linear_probe(
                    config,
                    bigearthnet_embeddings,
                    output_dir=mat_output_dir,
                    label="bigearthnet",
                    norm_method=normalization_method,
                    slice_dim=dim,
                    feature_override="backbone",
                )
        else:
            _run_linear_probe(
                config,
                bigearthnet_embeddings,
                output_dir=output_dir,
                label="bigearthnet",
                norm_method=normalization_method,
            )


    if args.disable_neuco == False:
        neuco_output_dir = output_dir / f"neuco_{normalization_method}"
        base_modalities: List[str] = list(config.neuco_modalities)
        s2_candidates = [m for m in base_modalities if m in ("s2l2a", "s2l1c")]
        has_dual_neuco = (
            len(base_modalities) == 2
            and "s1" in base_modalities
            and len(s2_candidates) > 0
            and config.model_type in {"torchgeo_resnet50", "ciip_checkpoint", "croma"}
        )

        def _run_single_neuco(modality: str, active_modality: str) -> None:
            suffix = "_s1" if active_modality.lower() == "s1" else ""
            csv_out_backbone = neuco_output_dir / "neuco_export" / f"neuco_{modality}{suffix}_backbone.csv"
            # csv_out_projected = neuco_output_dir / "neuco_export" / f"neuco_{modality}_projected.csv"

            if False: #csv_out_backbone.exists():
                pass
                # print(f"NeuCo embeddings already exist for modality {modality}, skipping extraction.")
                # neuco_bundle = None
            else:
                print(f"Extracting NeuCo embeddings for modality {modality} into {csv_out_backbone.parent}")
                neuco_output_dir.mkdir(parents=True, exist_ok=True)
                neuco_loader = _build_neuco_loader(config, modalities=[modality])
                _print_band_stats(
                    f"NeuCo eval ({modality})",
                    neuco_loader,
                    max_batches=config.stats_max_batches,
                )

                expected_channels = (
                    _infer_model_in_channels(adapter, config.model_in_channels, modality=active_modality)
                )
                with _use_adapter_modality(adapter, active_modality):
                    neuco_bundle = _extract_embeddings(
                        adapter,
                        neuco_loader,
                        device=device,
                        require_ids=True,
                        expected_in_channels=expected_channels,
                        modality=active_modality,
                        pca_output_dir=neuco_output_dir,
                    )

                _export_neuco(neuco_bundle, neuco_output_dir / "neuco_export", label=modality)
                print("Saved NeuCo embeddings to ", neuco_output_dir / "neuco_export")

            print(f'model type: {config.model_type}')
            print(f'model weights: {config.model_weights}')
            if config.model_type == "croma" or ('dofa' in config.model_weights.lower() if config.model_weights else False):
                embedding_dim_backbone = "768"
            elif config.model_type == "torchgeo_resnet50":
                embedding_dim_backbone = "2048"
            elif 'resnet18' in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "512"
            elif 'llama' in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "512"
            elif 'remoteclip' in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "768"
            elif 'resnet50' in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "2048"
            elif 'vitsmall' in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "384"
            elif 'scalemae' in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "1024"
            else:
                backbone_tensor = getattr(neuco_bundle, "backbone", None) if neuco_bundle is not None else None

                embedding_dim_backbone = (
                    str(backbone_tensor.shape[1]) if backbone_tensor is not None else "2048"
                )

            print(f"NeuCo embedding dimension for {modality}: {embedding_dim_backbone}")
            backbone_tensor = getattr(neuco_bundle, "backbone", None) if neuco_bundle is not None else None
            if backbone_tensor is not None:
                print(
                    "NeuCo benchmark using backbone embeddings dim: "
                    f"{backbone_tensor.shape[1]} (embedding_dim arg: {embedding_dim_backbone})"
                )
            else:
                print(
                    "NeuCo benchmark embedding_dim arg: "
                    f"{embedding_dim_backbone} (backbone tensor unavailable)"
                )

            if config.matryoshka_dims:
                for dim in config.matryoshka_dims:
                    if neuco_bundle is None or neuco_bundle.backbone is None:
                        raise RuntimeError("NeuCo backbone embeddings are required for Matryoshka evaluation.")
                    if dim > neuco_bundle.backbone.shape[1]:
                        raise ValueError(
                            f"Matryoshka dim {dim} exceeds NeuCo backbone dimension {neuco_bundle.backbone.shape[1]}."
                        )
                    mat_output_dir = neuco_output_dir / f"matryoshka_dim_{dim}"
                    mat_export_dir = mat_output_dir / "neuco_export"
                    mat_export_dir.mkdir(parents=True, exist_ok=True)
                    mat_bundle = EmbeddingBundle(
                        backbone=neuco_bundle.backbone[:, :dim],
                        projected=neuco_bundle.projected,
                        labels=neuco_bundle.labels,
                        multi_labels=neuco_bundle.multi_labels,
                        ids=neuco_bundle.ids,
                    )
                    mat_csv = mat_export_dir / f"neuco_{modality}{suffix}_backbone.csv"
                    _export_neuco(mat_bundle, mat_export_dir, label=modality)
                    cmd = [
                        "python", "/local/ms-data/NeuCo-Bench/benchmark/main.py",
                        "--annotation_path", "/local/ms-data/SSL4EO-S12-downstream/labels",
                        "--output_dir", mat_output_dir,
                        "--config", "/local/ms-data/NeuCo-Bench/benchmark/config.yaml",
                        "--method_name", "backbone",
                        "--phase", "testing",
                        "--submission_file", mat_csv,
                        "--embedding_dim", str(dim)
                    ]
                    env = os.environ.copy()
                    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                    subprocess.run(cmd, check=True, env=env)
            else:
                cmd = [
                    "python", "/local/ms-data/NeuCo-Bench/benchmark/main.py",
                    "--annotation_path", "/local/ms-data/SSL4EO-S12-downstream/labels",
                    "--output_dir", neuco_output_dir,
                    "--config", "/local/ms-data/NeuCo-Bench/benchmark/config.yaml",
                    "--method_name", "backbone",
                    "--phase", "testing",
                    "--submission_file", csv_out_backbone,
                    "--embedding_dim", embedding_dim_backbone
                ]
                env = os.environ.copy()
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                subprocess.run(cmd, check=True, env=env)

        if has_dual_neuco:
            print("Dual NeuCo modalities detected; running S2 and S1 evaluations separately.")
            s2_modality = s2_candidates[0]
            _run_single_neuco(s2_modality, "s2")
            # _run_single_neuco("s1", "s1")
        else:
            neuco_modalities: List[str] = base_modalities
            # print('NeuCo modalities from config: ', neuco_modalities)
            # print('Target modality: ', target_modality)
            if target_modality == "s1":
                neuco_modalities = ["s1"]
            elif not neuco_modalities:
                neuco_modalities = ["s2l1c"]

            if len(neuco_modalities) > 1:
                print('Only useing 1st modality for neuco benchmark')
            modality = neuco_modalities[0]
            _run_single_neuco(modality, target_modality)


    ssl4eo_available = (
        config.enable_ssl4eo
        and config.ssl4eo_root is not None
        and getattr(adapter, "supports_ssl4eo", True)
    )

    assert ssl4eo_available or not config.enable_ssl4eo, "SSL4EO diagnostics requested but not available"

    s1_ssl4eo = s2_ssl4eo = None
    max_batches_cka=20
    if ssl4eo_available:
        s1_ssl4eo, s2_ssl4eo, ssl4eo_ids = _extract_ssl4eo_embeddings(
            config,
            adapter,
            device=device,
            max_batches_cka=max_batches_cka
        )

        # create Path fo diagnostics
        diagnostics_dir = os.path.join(output_dir, "embedding_diagnostics")
        diagnostics_dir = Path(diagnostics_dir)

        _run_embedding_diagnostics(
            config,
            s1_ssl4eo,
            s2_ssl4eo,
            sample_ids=ssl4eo_ids,
            output_dir= diagnostics_dir #output_dir + "embedding_diagnostics",
        )
        logging.info("Completed SSL4EO diagnostics")
        # print output dir
        print("SSL4EO diagnostics saved to ", output_dir)
    else:
        logging.info(
            "Skipping SSL4EO diagnostics (enable_ssl4eo=%s, root=%s, supports_ssl4eo=%s)",
            config.enable_ssl4eo,
            config.ssl4eo_root,
            getattr(adapter, "supports_ssl4eo", True),
        )
    curvature = None
    if (
        is_lorentz
        and s1_ssl4eo is not None
        and s1_ssl4eo.projected is not None
        and s2_ssl4eo is not None
        and s2_ssl4eo.projected is not None
    ):
        if not isinstance(base_model, LorentzCIIP):
            raise TypeError("Expected LorentzCIIP model when is_lorentz is True")
        # use SSL4EO post-head projected features for hyperbolic visualisations
        s1_proj_feats = s1_ssl4eo.projected.to(device=device, dtype=torch.float32)
        s2_proj_feats = s2_ssl4eo.projected.to(device=device, dtype=torch.float32)
        _run_hyperbolic_visualisations(
            s1_proj_feats,
            s2_proj_feats,
            output_dir=output_dir / "hyperbolic",
            model=base_model,
            aperture_logk=None,
            seed=config.random_seed,
        )

        curvature = _extract_curvature_from_state(base_model, device=device, dtype=s1_proj_feats.dtype)


    if config.model_weights == 'dino':
        print("Skipping SSL4EO cross-modal retrieval for TorchGeo dino ResNet50 model")
        return
    

    # use_backbone_retrieval = (
    #     config.model_type == "croma"
    #     or config.model_weights == 'moco'
    # )
    # use_projected_retrieval = (
    #     config.model_type == 'ciip_checkpoints'
    #     and not use_backbone_retrieval
    # )

    # if use_backbone_retrieval:
    #     logging.info(
    #         "Using backbone embeddings for cross-modal retrieval (model_type=%s, model_weights=%s)",
    #         config.model_type,
    #         config.model_weights,
    #     )
    #     retrieval_s1 = s1_ssl4eo.backbone
    #     retrieval_s2 = s2_ssl4eo.backbone

    #     retrieval_metrics = compute_cross_modal_retrieval(retrieval_s1, retrieval_s2, curvature=curvature)
    #     retrieval_metrics['num_samples'] = s1_ssl4eo.backbone.shape[0]
    #     #output_dir / "ssl4eo_retrieval.json"
    #     retrieval_path = Path(output_dir) / "ssl4eo_retrieval_backbone.json"
    #     retrieval_path.write_text(json.dumps(retrieval_metrics, indent=2, sort_keys=True))
    #     logging.info("SSL4EO cross-modal retrieval metrics: %s", retrieval_metrics)

    # elif use_projected_retrieval:
    #     # print the attributes of s1_ssl4eo
    #     # print("SSL4EO S1 EmbeddingBundle attributes:", dir(s1_ssl4eo))
    #     logging.info("Using projected embeddings for cross-modal retrieval")
    #     # print number of samples
    #     print(f'Num samples for cross-modal: {s1_ssl4eo.projected.shape[0]}')
    #     retrieval_s1 = s1_ssl4eo.projected
    #     retrieval_s2 = s2_ssl4eo.projected
    #     retrieval_metrics = compute_cross_modal_retrieval(retrieval_s1, retrieval_s2, curvature=curvature)
    #     retrieval_path = Path(output_dir) / "ssl4eo_retrieval_projected.json"
    #     retrieval_metrics['num_samples'] = s1_ssl4eo.projected.shape[0]
    #     retrieval_path.write_text(json.dumps(retrieval_metrics, indent=2, sort_keys=True))
    #     logging.info("SSL4EO projected cross-modal retrieval metrics: %s", retrieval_metrics)

    # else:
    #     print('Did not do cross-modal retrieval')



__all__ = ["ModelEvalConfig", "run_full_evaluation"]


# python evaluation/unified_evaluation.py --model-in-channels 12 --ciip-epoch 10 --ssl4eo-subset-size 50 --model-path '2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16';
# python evaluation/unified_evaluation.py --model-in-channels 12 --ciip-epoch 30 --ssl4eo-subset-size 50 --model-path '2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16';
# python evaluation/unified_evaluation.py --model-in-channels 12 --ciip-epoch 90 --ssl4eo-subset-size 50 --model-path '2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16';
# python evaluation/unified_evaluation.py --model-in-channels 12 --ciip-epoch 75 --ssl4eo-subset-size 50;


if __name__ == "__main__":
    import argparse
    import tempfile, time, os

    parser = argparse.ArgumentParser(description="Run the unified CIIP evaluation pipeline.")
    parser.add_argument(
        "--model-type",
        default="ciip_checkpoint",
        choices=["ciip_checkpoint", "torchgeo_resnet50", "croma", "backbone_only"],
        help="Model source to evaluate.",
    )
    # parser.add_argument("--checkpoint", type=Path, help="Checkpoint path for CIIP/Lorentz models.")
    parser.add_argument(
        "--model-weights",
        choices=[
            "dino",
            "moco",
            "rcf_13ch",
            "dofa_base_s2_13ch",
            "scalemae_large_rgb",
            "resnet18_s2_all_moco",
            "resnet18_s2_rgb_moco",
            "resnet50_s2_rgb_moco",
            "resnet152_imagenet_rgb",
            "vitsmall16_s2_all_moco",
            "llama3_ms_clip_base",
            "remoteclip",
        ],
        help="Pretrained weight selection for the requested model type.",
    )
    parser.add_argument("--croma-weights", type=Path,  help="Path to the pretrained CROMA weights.") # '/home/juro4948/ciip/comparison/CROMA-main/CROMA_base.pt'
    parser.add_argument("--croma-image-resolution", type=int, default=120, help="Input resolution expected by the CROMA model.")
    parser.add_argument("--model-in-channels", type=int, help="Number of input channels for TorchGeo ResNet models.")
    parser.add_argument("--model-path", type=str, help="Experiment path identifier for the model.")
    parser.add_argument(
        "--ciip-framework",
        choices=["modified_resnet", "transformer", "resnet18", "resnet50"],
        help="Backbone framework for CIIP checkpoints (defaults to auto-detect).",
    )
    
    parser.add_argument("--tsne-samples", type=int, default=1500, help="Samples used for t-SNE visualisations.")
    parser.add_argument("--pca-samples", type=int, default=5000, help="Samples used for PCA visualisations.")
    parser.add_argument("--ssl4eo-subset-size", type=int, default=5, help="Subset size for SSL4EO embedding extraction.")
    parser.add_argument("--ssl4eo-subset-seed", type=int, default=0, help="Subset seed for SSL4EO sampling.")
    parser.add_argument("--neuco-modalities", nargs="*", default=["s2l2a"], help="NeuCo modalities to export.")
    parser.add_argument("--ssl4eo-s2-tier", choices=['s2l1c', 's2l2a'], default='s2l2a', help="Sentinel-2 data tier to use for SSL4EO embeddings.")
    parser.add_argument("--neuco-seasons", type=int, default=4, help="Number of seasons for NeuCo extraction.") # i believe these are averaged
    parser.add_argument(
        "--evaluation-modality",
        choices=["s1", "s2"],
        default="s2",
        help="Sentinel modality to use for EuroSAT and NeuCo evaluations.",
    )
    parser.add_argument(
        "--normalization-method",
        choices=NORMALIZATION_METHODS,
        default=DEFAULT_NORMALIZATION_METHOD,
        help="Normalization applied to Sentinel-2 / optical inputs.",
    )
    # parser.add_argument("--neuco-resize", type=int, nargs=2, help="Resize dimensions for NeuCo images.")

    parser.add_argument("--disable-eurosat", action="store_true", help="Skip EuroSAT linear probe evaluation.")
    parser.add_argument("--disable-bigearthnet", action="store_true", help="Skip BigEarthNet linear probe evaluation.")
    parser.add_argument(
        "--matryoshka-dims",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate matryoshka slices at these embedding dimensions.",
    )
    parser.add_argument(
        "--matryoshka-feature",
        type=str,
        choices=("backbone",),
        default="backbone",
        help="Embedding space to slice for matryoshka evaluation.",
    )
    parser.add_argument("--disable-neuco", action="store_true", help="Skip NeuCo benchmark evaluation.")
    parser.add_argument("--disable-ssl4eo", action="store_true", help="Skip SSL4EO diagnostics even if a root is provided.")
    parser.add_argument("--ciip-epoch", type=int, default=10, help="Epoch number for CIIP checkpoint to evaluate.")
    parser.add_argument("--bigearthnet-root", type=Path, default=Path("/local/ms-data/BigEarthNet/"), help="Root directory for BigEarthNet data.")
    parser.add_argument("--bigearthnet-image-size", type=int, default=224, help="Input resolution for BigEarthNet evaluation.")
    parser.add_argument(
        "--stats-max-batches",
        type=int,
        default=0,
        help="Limit batches when computing bandwise stats (0 = full dataset).",
    )
    args = parser.parse_args()


# trained on old SSL4EO, 13 bands
## vanilla
    # '2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp': epochs 5, 10 up to 85
## hyperbolic
    # '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/': 'curv_init_1', epochs 0-24
    # '2025_11_06-19_39_33-model_resnet50-lr_0.001-b_128-j_6-p_amp': 'curv_init_0.5', epochs 0-14
    # '2025_11_07-09_24_18-model_resnet50-lr_0.001-b_128-j_6-p_amp': 'curv_init_0.1', epochs 0-16
    
# trained on v1.1, 12 bands
    # '2025_11_12-12_28_55-model_resnet50-lr_0.001-b_2-j_6-p_amp': 'curv_init_0.1', epochs ..., increased lr for curvature param

    # if model_path arg is empty, set default
    if args.model_weights == "remoteclip" and args.model_in_channels is None:
        args.model_in_channels = 3

    if args.model_in_channels:# and args.neuco_modalities:
        if args.model_in_channels == 12:
            print('Setting modalities to s2l2a')
            # args.neuco_modalities = ['s2l2a']
            args.ssl4eo_s2_tier = 's2l2a'
        if args.model_in_channels == 13:
            print('Setting modalities to s2l1c')
            # args.neuco_modalities = ['s2l1c']
            args.ssl4eo_s2_tier = 's2l1c'
        if args.model_in_channels == 10 and (args.model_weights == "llama3_ms_clip_base"):
            print('Setting modalities to s2l2a for MS-CLIP (10ch)')
            args.ssl4eo_s2_tier = 's2l2a'
        if args.model_in_channels == 3 or args.model_weights == "remoteclip":
            print('Setting modalities to rgb for RGB model')
            args.ssl4eo_s2_tier = 'rgb'
        

        
            
    if args.model_type == "ciip_checkpoint":
        if not args.model_path:
            args.model_path = '2025_11_21-11_32_16-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16'
        # '2025_11_19-18_54_18-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16/'
        # '2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp'
        # '2025_11_17-12_26_37-model_resnet50-lr_0.001-b_2-j_6-p_amp'
        # '2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp'
        # '2025_11_18-16_14_01-model_resnet50-lr_0.001-b_2-j_6-p_amp'
        # '2025_11_17-12_26_37-model_resnet50-lr_0.001-b_2-j_6-p_amp' # not working??
        # '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/'
        # '2025_11_14-10_56_41-model_resnet50-lr_0.001-b_2-j_6-p_amp'
        args.model_root = '/local/ms-data/SSL4EO/model/'
        # '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/'
        # '/local/ms-data/SSL4EO/model/2025_11_13-08_33_13-model_resnet50-lr_0.001-b_2-j_6-p_amp/checkpoints/epoch_10.pt'
        # 
        
        # '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/'
        # '2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp'
        # '2025_11_07-09_24_18-model_resnet50-lr_0.001-b_128-j_6-p_amp'
        # '2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp'
        checkpoint_root = Path(args.model_root) / args.model_path / "checkpoints"
        # args.output_dir = Path("diagnostics/output")
        
        # args.checkpoint=Path(f"{checkpoint_root}/epoch_10.pt")
        args.checkpoint=Path(f"{checkpoint_root}/epoch_{args.ciip_epoch}.pt")

    else:
        args.checkpoint = None
    args.eurosat_root=Path("/local/ms-data/EuroSAT/")
    args.neuco_root=Path("/local/ms-data/SSL4EO-S12-downstream/data")

    if (
        args.model_type == "scalemae_large_rgb"
        or "rgb" in (args.model_weights or "")
    ):
        args.ssl4eo_s2_tier = 'rgb'


    
    if args.model_type == 'croma':
        output_dir = f"/home/juro4948/ciip/diagnostics/unified_eval/{args.model_type}/"
    elif args.model_type == 'torchgeo_resnet50':
        output_dir = f"/home/juro4948/ciip/diagnostics/unified_eval/{args.model_weights}/"
    elif args.model_type == 'backbone_only':
        output_dir = f"/home/juro4948/ciip/diagnostics/unified_eval/{args.model_weights or 'backbone_only'}/"
    elif args.model_type == 'ciip_checkpoint':
        output_dir = f"/home/juro4948/ciip/diagnostics/unified_eval/{args.model_type}/"
        output_dir = output_dir + args.model_path.strip('/') + f"/epoch_{args.ciip_epoch}/"
        is_vit_run = (
            args.ciip_framework == "transformer"
            or ("vit" in (args.model_path or "").lower())
        )
        if not is_vit_run and args.ciip_framework is None:
            is_vit_run = _looks_like_vit_checkpoint(args.checkpoint)
        if is_vit_run:
            output_dir = output_dir.rstrip("/") + "_meanpool/"
    else:
        raise ValueError("Invalid model type")
    args.output_dir = Path(output_dir)

     #dino_13bands/") #curv_init_1_epoch10/ curv_init_1
    args.ssl4eo_root=Path("/local/ms-data/SSL4EOv1.1/train")


    if args.model_type == "ciip_checkpoint" and args.checkpoint is None:
        parser.error("--checkpoint is required when --model-type=ciip_checkpoint")
    if args.model_type in {"torchgeo_resnet50", "backbone_only"} and args.model_weights is None:
        parser.error("--model-weights must be provided for the selected model type")
    if args.model_type == "croma" and args.croma_weights is None:
        parser.error("--croma-weights must be provided when --model-type=croma")
    

    cfg = ModelEvalConfig(
        eurosat_root=args.eurosat_root,
        neuco_root=args.neuco_root,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        model_weights=args.model_weights,
        ciip_framework=args.ciip_framework,
        model_in_channels=args.model_in_channels,
        croma_weights=args.croma_weights,
        croma_image_resolution=args.croma_image_resolution,
        enable_ssl4eo=not args.disable_ssl4eo,
        matryoshka_dims=tuple(args.matryoshka_dims) if args.matryoshka_dims else None,
        matryoshka_feature=str(args.matryoshka_feature),
        ssl4eo_root=args.ssl4eo_root,
        tsne_samples=args.tsne_samples,
        pca_samples=args.pca_samples,
        ssl4eo_subset_size=args.ssl4eo_subset_size,
        ssl4eo_subset_seed=args.ssl4eo_subset_seed,
        neuco_modalities=tuple(args.neuco_modalities),
        neuco_seasons=args.neuco_seasons,
        evaluation_modality=args.evaluation_modality,
        ssl4eo_s2_tier=args.ssl4eo_s2_tier,
        bigearthnet_root=args.bigearthnet_root,
        bigearthnet_image_size=args.bigearthnet_image_size,
        normalization_method=args.normalization_method,
        stats_max_batches=int(args.stats_max_batches),
    )
    run_full_evaluation(cfg)
