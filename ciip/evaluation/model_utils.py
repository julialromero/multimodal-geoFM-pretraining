"""Utility helpers for loading evaluation models without circular imports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchgeo.models import ResNet50_Weights, resnet50

from ciip.model_ciip import CIIP, LorentzCIIP


class EvaluationAdapter(nn.Module):
    """Common interface used by unified evaluation to extract embeddings."""

    supports_ssl4eo: bool = True
    supports_hyperbolic: bool = False
    is_lorentz: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.base_model: nn.Module = self
        self.dtype_s2: torch.dtype = torch.float32

    # These helpers are intentionally thin wrappers so that the evaluation
    # pipeline can treat all models uniformly.
    def compute_backbone(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def compute_posthead(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def compute_projected(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class CiipEvaluationAdapter(EvaluationAdapter):
    """Adapter around CIIP/LorentzCIIP checkpoints."""

    def __init__(self, model: nn.Module, *, is_lorentz: bool) -> None:
        super().__init__()
        self.base_model = model
        self.encoder_s2 = model.encoder_s2  # type: ignore[attr-defined]
        self.dtype_s2 = getattr(model, "dtype_s2", torch.float32)
        self.is_lorentz = is_lorentz
        self.supports_hyperbolic = bool(is_lorentz)

    def compute_backbone(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder_s2(images.type(self.dtype_s2))
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return features

    def compute_posthead(self, images: torch.Tensor) -> torch.Tensor:
        if self.is_lorentz:
            post = self.base_model.encode_s2(images, lorentz=False, normalize=False)
        else:
            post = self.base_model.encode_s2(images, normalize=False)
        if post.ndim > 2:
            post = post.flatten(start_dim=1)
        return post

    def compute_projected(self, images: torch.Tensor) -> torch.Tensor:
        if self.is_lorentz:
            projected = self.base_model.encode_s2(images, lorentz=True)
        else:
            projected = self.base_model.encode_s2(images, normalize=True)
        if projected.ndim > 2:
            projected = projected.flatten(start_dim=1)
        return projected

    # Expose multimodal helpers for downstream diagnostics.
    def encode_s1(self, *args, **kwargs):  # pragma: no cover - passthrough
        return self.base_model.encode_s1(*args, **kwargs)

    def encode_s2(self, *args, **kwargs):  # pragma: no cover - passthrough
        return self.base_model.encode_s2(*args, **kwargs)


class TorchGeoResNetAdapter(EvaluationAdapter):
    """Adapter for TorchGeo ResNet50 checkpoints (Sentinel-2 SSL weights)."""

    supports_ssl4eo = False

    def __init__(self, *, weights: Optional[ResNet50_Weights], in_chans: int = 13) -> None:
        super().__init__()
        backbone = resnet50(weights=None, in_chans=in_chans)
        if weights is not None:
            state_dict = weights.get_state_dict(progress=True)
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logging.warning(
                    "Loaded ResNet50 weights with missing=%s, unexpected=%s", missing, unexpected
                )

        # TorchGeo exposes the classification head via ``fc``. Preserve it as a
        # projection module if available, otherwise fall back to identity.
        projection = getattr(backbone, "fc", None)
        self.projection_head = projection if isinstance(projection, nn.Module) else None
        if self.projection_head is not None:
            backbone.fc = nn.Identity()

        self.base_model = backbone
        self.encoder_s2 = backbone
        self.dtype_s2 = torch.float32

    def compute_backbone(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder_s2(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return features

    def compute_posthead(self, images: torch.Tensor) -> torch.Tensor:
        features = self.compute_backbone(images)
        if self.projection_head is not None:
            features = self.projection_head(features)
        return features

    def compute_projected(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.compute_posthead(images), dim=1)

    def encode_s2(  # pragma: no cover - compatibility wrapper
        self,
        tensor: torch.Tensor,
        *,
        normalize: bool = False,
        lorentz: bool = False,
        project_hyperbolic: bool = False,
        post_head: bool = True,
    ) -> torch.Tensor:
        if lorentz:
            raise ValueError("Lorentz projection is not supported for ResNet adapters")
        features = self.compute_posthead(tensor) if post_head else self.compute_backbone(tensor)
        if normalize or project_hyperbolic:
            features = F.normalize(features, dim=1)
        return features


def _clean_state_dict(raw_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip DataParallel prefixes from checkpoint state dictionaries."""

    return {key.replace("module.", ""): value for key, value in raw_state.items()}


def _infer_projection_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Infer embedding and pre-projection dimensionality from the checkpoint."""

    if "encoder_s2.proj.weight" in state_dict:
        weight = state_dict["encoder_s2.proj.weight"]
        return int(weight.shape[0]), int(weight.shape[1])
    if "encoder_s2.fc.weight" in state_dict:
        weight = state_dict["encoder_s2.fc.weight"]
        return int(weight.shape[0]), int(weight.shape[0])
    raise RuntimeError("Unable to infer embedding dimensions from checkpoint")


def build_model_from_checkpoint(checkpoint: Path) -> Tuple[nn.Module, bool]:
    """Instantiate a CIIP or LorentzCIIP model from a checkpoint path."""

    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = _clean_state_dict(state_dict)
    embed_dim, pre_dim = _infer_projection_dims(cleaned)
    is_lorentz = any(key.startswith("curv") or "lorentz" in key for key in cleaned)

    kwargs = dict(
        embed_dim=embed_dim,
        pre_projection_dim=pre_dim,
        s1_resolution=224,
        s1_layers=(3, 4, 6, 3),
        s1_width=32,
        s1_patch_size=16,
        s1_bands=2,
        s2_resolution=224,
        s2_layers=(3, 4, 6, 3),
        s2_width=32,
        s2_patch_size=16,
        s2_bands=13,
        framework="resnet50",
    )

    if is_lorentz:
        model: nn.Module = LorentzCIIP(**kwargs)
    else:
        model = CIIP(**kwargs)

    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    if missing or unexpected:
        logging.warning("Checkpoint loaded with missing=%s, unexpected=%s", missing, unexpected)
        raise RuntimeError(
            "Checkpoint incompatible with model "
            f"(missing={missing}, unexpected={unexpected})"
        )

    model.eval()
    return model, is_lorentz


def _resolve_resnet_weights(name: Optional[str]) -> Optional[ResNet50_Weights]:
    if name is None:
        return None
    normalized = name.lower()
    if normalized == "dino":
        return ResNet50_Weights.SENTINEL2_ALL_DINO
    if normalized == "moco":
        return ResNet50_Weights.SENTINEL2_ALL_MOCO
    raise ValueError(f"Unsupported ResNet50 weights '{name}'. Expected 'dino' or 'moco'.")


def build_evaluation_adapter(
    *,
    model_type: str,
    checkpoint: Optional[Path],
    model_weights: Optional[str],
    in_chans: int = 13,
) -> EvaluationAdapter:
    """Create an evaluation adapter based on the requested model type."""

    if model_type == "ciip_checkpoint":
        if checkpoint is None:
            raise ValueError("checkpoint must be provided when using 'ciip_checkpoint'")
        model, is_lorentz = build_model_from_checkpoint(checkpoint)
        return CiipEvaluationAdapter(model, is_lorentz=is_lorentz)

    if model_type == "torchgeo_resnet50":
        weights = _resolve_resnet_weights(model_weights)
        return TorchGeoResNetAdapter(weights=weights, in_chans=in_chans)

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        "Valid options are 'ciip_checkpoint' and 'torchgeo_resnet50'."
    )


__all__ = [
    "EvaluationAdapter",
    "CiipEvaluationAdapter",
    "TorchGeoResNetAdapter",
    "build_model_from_checkpoint",
    "build_evaluation_adapter",
]

