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

    def __init__(
        self,
        model: nn.Module,
        *,
        is_lorentz: bool,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.encoder_s1 = getattr(model, "encoder_s1", None)
        self.encoder_s2 = getattr(model, "encoder_s2", None)
        self.dtype_s1 = getattr(model, "dtype_s1", torch.float32)
        self.dtype_s2 = getattr(model, "dtype_s2", torch.float32)
        self.is_lorentz = is_lorentz
        self.supports_hyperbolic = bool(is_lorentz)

    def _default_stream(self) -> Tuple[nn.Module, torch.dtype, str]:
        if self.encoder_s2 is not None:
            return self.encoder_s2, self.dtype_s2, "s2"
        if self.encoder_s1 is not None:
            return self.encoder_s1, self.dtype_s1, "s1"
        raise RuntimeError("CIIP adapter does not expose any encoders")

    def _encode_projected(self, images: torch.Tensor, *, project: bool) -> torch.Tensor:
        _, dtype, stream = self._default_stream()
        tensor = images.type(dtype)

        if stream == "s1":
            encode_fn = self.base_model.encode_s1
        else:
            encode_fn = self.base_model.encode_s2

        if self.is_lorentz:
            return encode_fn(tensor, lorentz=project, normalize=False)

        if project:
            return encode_fn(tensor, normalize=True)
        return encode_fn(tensor, normalize=False)

    def compute_backbone(self, images: torch.Tensor) -> torch.Tensor:
        print("Using compute_backbone")
        encoder, dtype, _ = self._default_stream()
        #print encoder
        # print(encoder)
        # if fc layer, replace with identity
        # if hasattr(encoder, 'fc'):
        #     encoder.fc = nn.Identity()
        # # same with proj layer
        # if hasattr(encoder, 'proj'):
        #     encoder.proj = nn.Identity()
        #      features = encoder(images.type(dtype))

        replaced_modules = []
        for attr in ("fc", "proj"):
            module = getattr(encoder, attr, None)
            if isinstance(module, nn.Module):
                replaced_modules.append((attr, module))
                setattr(encoder, attr, nn.Identity())

        try:
            features = encoder(images.type(dtype))
        finally:
            for attr, module in replaced_modules:
                setattr(encoder, attr, module)



        
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        # print 
        # print("Features shape:", features.shape)
        # raise NotImplementedError("compute_backbone is not implemented yet")
        return features

    def compute_posthead(self, images: torch.Tensor) -> torch.Tensor:
        post = self._encode_projected(images, project=False)
        if isinstance(post, (tuple, list)):
            post = post[0]
        if post.ndim > 2:
            post = post.flatten(start_dim=1)
        return post

    def compute_projected(self, images: torch.Tensor) -> torch.Tensor:
        projected = self._encode_projected(images, project=True)
        if isinstance(projected, (tuple, list)):
            projected = projected[0]
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

    supports_ssl4eo = True

    def __init__(self,
            *,
            weights: Optional[ResNet50_Weights],
            in_chans: int = 13,
            in_chans_s1: int = 2,
            dtype_s1: torch.dtype = torch.float32,
        ) -> None:
        super().__init__()
        backbone = resnet50(weights=None, in_chans=in_chans)
        s1_weights = weights[0]
        s2_weights = weights[1]
        if s2_weights is not None:
            state_dict = s2_weights.get_state_dict(progress=True)
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


        # set up s2 encoder:
        self.encoder_s1: Optional[nn.Module] = None
        self.projection_head_s1: Optional[nn.Module] = None
        self.dtype_s1: torch.dtype = dtype_s1
        if s1_weights is not None:
            s1_backbone = resnet50(weights=None, in_chans=in_chans_s1)
            s1_state = s1_weights.get_state_dict(progress=True)
            s1_missing, s1_unexpected = s1_backbone.load_state_dict(s1_state, strict=False)
            if s1_missing or s1_unexpected:
                logging.warning(
                    "Loaded ResNet50 S1 weights with missing=%s, unexpected=%s",
                    s1_missing,
                    s1_unexpected,
                )

            s1_projection = getattr(s1_backbone, "fc", None)
            if isinstance(s1_projection, nn.Module):
                self.projection_head_s1 = s1_projection
                s1_backbone.fc = nn.Identity()

            self.encoder_s1 = s1_backbone

    def _default_stream(self) -> Tuple[nn.Module, torch.dtype, str]:
        if self.encoder_s2 is not None:
            return self.encoder_s2, self.dtype_s2, "s2"
        if self.encoder_s1 is not None:
            return self.encoder_s1, self.dtype_s1, "s1"
        raise RuntimeError("TorchGeoResNet adapter does not expose any encoders")
        

    def compute_backbone(self, images: torch.Tensor) -> torch.Tensor:
        encoder, dtype, _ = self._default_stream()
        features = encoder(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return features

    def compute_posthead(self, images: torch.Tensor) -> torch.Tensor:
        return None

    def compute_projected(self, images: torch.Tensor) -> torch.Tensor:
        return None

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
        features = self.encoder_s2(tensor)
        if normalize or project_hyperbolic:
            features = F.normalize(features, dim=1)
        return features
    
    def encode_s1(  # pragma: no cover - compatibility wrapper
        self,
        tensor: torch.Tensor,
        *,
        normalize: bool = False,
        lorentz: bool = False,
        project_hyperbolic: bool = False,
        post_head: bool = True,
    ) -> torch.Tensor:
        if self.encoder_s1 is None:
            return None
            # raise RuntimeError("encoder_s1 is not initialized.")
        if lorentz:
            raise ValueError("Lorentz projection is not supported for ResNet adapters")

        features = self.encoder_s1(tensor)
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

def _infer_s2_bands(state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer the number of S2 bands from the checkpoint's first conv layer."""
    
    # Look for the first convolutional layer in the S2 encoder
    conv_keys = [
        "encoder_s2.conv1.weight",           # Direct conv1
        "encoder_s2.backbone.conv1.weight",  # If backbone is wrapped
        "encoder_s2.model.conv1.weight",     # Alternative wrapping
    ]
    
    for key in conv_keys:
        if key in state_dict:
            weight = state_dict[key]
            # Conv2d weight shape is [out_channels, in_channels, kernel_h, kernel_w]
            return int(weight.shape[1])  # in_channels = number of bands
    
    # Fallback: try to find any conv1 weight
    for key in state_dict.keys():
        if "conv1.weight" in key and "encoder_s2" in key:
            weight = state_dict[key]
            return int(weight.shape[1])
    
    # Default fallback
    logging.warning("Could not infer S2 bands from checkpoint, defaulting to 13")
    return 13

def build_model_from_checkpoint(checkpoint: Path) -> Tuple[nn.Module, bool]:
    """Instantiate a CIIP or LorentzCIIP model from a checkpoint path."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = _clean_state_dict(state_dict)
    embed_dim, pre_dim = _infer_projection_dims(cleaned)
    is_lorentz = any(key.startswith("curv") or "lorentz" in key for key in cleaned)


    # infer nummber of s2 bands in checkpoint
    s2_bands = _infer_s2_bands(cleaned)
    logging.info(f"Inferred S2 bands from checkpoint: {s2_bands}")


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
        s2_bands=s2_bands,
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
        return None, ResNet50_Weights.SENTINEL2_ALL_DINO
    if normalized == "moco":
        return ResNet50_Weights.SENTINEL1_GRD_MOCO, ResNet50_Weights.SENTINEL2_ALL_MOCO
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

