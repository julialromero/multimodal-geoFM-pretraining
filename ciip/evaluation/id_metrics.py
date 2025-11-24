"""Helper utilities for intrinsic dimension diagnostics and SSL4EO transforms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import skdim.id as id
import torch
import torch.nn as nn
import torchvision.transforms as T

# Reusable normalization constants for ImageNet-pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class S2ScaleTransform(nn.Module):
    """Scale Sentinel-2 pixel values by a constant factor."""

    def __init__(self, scale: float = 10000.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale


# Predefined SSL4EO transforms keyed by weight names to ensure consistent preprocessing
SSL4EO_MODEL_TRANSFORMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "dofa_base_s2_13ch": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    "scalemae_large_rgb": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    "resnet18_s2_all_moco": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    "resnet50_s2_all_moco": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    "resnet18_s2_rgb_moco": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    "resnet50_s2_rgb_moco": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    "resnet152_imagenet_rgb": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    ),
    "vitsmall16_s2_all_moco": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
}


def select_ssl4eo_transform(model_weights: Optional[str]) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Return the SSL4EO transform matching the provided weight key, if any."""

    weight_key = (model_weights or "").lower()
    return SSL4EO_MODEL_TRANSFORMS.get(weight_key)


def prepare_embeddings_for_id(tensor: torch.Tensor) -> Optional[np.ndarray]:
    """Convert embeddings to NumPy and drop degenerate entries."""

    if tensor is None:
        return None

    Z = tensor.detach().cpu().to(torch.float64).numpy()
    if Z.size == 0:
        return None

    uniques = np.unique(Z, axis=0)
    if len(uniques) != len(Z):
        logging.info("Removing %d duplicate embeddings before ID computation", len(Z) - len(uniques))
    Z = uniques

    var_per_dim = Z.var(axis=0)
    logging.info(
        "Embedding variance – min: %.6f, max: %.6f; zero-var dims: %d",
        var_per_dim.min(),
        var_per_dim.max(),
        np.sum(var_per_dim < 1e-10),
    )

    norms = np.linalg.norm(Z, axis=1)
    valid = (norms > 1e-6) & np.isfinite(norms)
    Z = Z[valid]
    if Z.size == 0:
        return None

    return Z


def compute_global_id_metrics(Z: np.ndarray) -> Dict[str, float]:
    """Compute FisherS, MLE, MoM, and TLE intrinsic dimensions for embeddings."""

    metrics: Dict[str, float] = {}

    def _compute_tle(Z: np.ndarray) -> float:
        tle = id.TLE()
        tle_pw = tle.fit_transform_pw(Z, n_neighbors=20)
        tle_pw = tle_pw[tle_pw >= 0]
        if tle_pw.size == 0:
            raise ValueError("TLE produced no valid pointwise estimates")
        return float(np.nanmean(tle_pw))

    metrics["fishers"] = float(id.FisherS().fit_transform(Z))
    metrics["mle"] = float(id.MLE(neighborhood_based=True).fit_transform(Z, n_neighbors=20))
    metrics["mom"] = float(id.MOM().fit_transform(Z, n_neighbors=20))
    metrics["tle"] = _compute_tle(Z)

    return metrics


def save_id_metrics(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    if not metrics:
        return

    with output_path.open("w", encoding="utf-8") as handle:
        for key in sorted(metrics):
            handle.write(f"{key}\n")
            for metric, value in metrics[key].items():
                handle.write(f"  {metric}: {value:.4f}\n")
            handle.write("\n")

    logging.info("Intrinsic dimension metrics saved to %s", output_path)

