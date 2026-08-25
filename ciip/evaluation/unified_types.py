"""Configuration and feature contracts for unified evaluation."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import numpy as np
from ciip.evaluation.normalization_contract import DEFAULT_NORMALIZATION_METHOD

DEFAULT_S2_BANDS: Sequence[str] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)


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
    enable_eurosat: bool = True
    enable_bigearthnet: bool = True
    enable_neuco: bool = True
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
