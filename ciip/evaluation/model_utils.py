"""Utility helpers for loading evaluation models without circular imports."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchgeo.models import (
    DOFABase16_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    ScaleMAELarge16_Weights,
    ViTSmall16_Weights,
    dofa_base_patch16_224,
    resnet18,
    resnet50,
    resnet152,
    scalemae_large_patch16,
    vit_small_patch16_224,
)

from ciip.model_ciip import CIIP, LorentzCIIP
from torchvision.models import resnet152, ResNet152_Weights

_CROMA_MODULE: Optional[ModuleType] = None


class RandomConvFeatures(nn.Module):
    """Random Convolutional Filters baseline (13-channel, 512-dim GAP)."""

    def __init__(self, in_chans: int = 13, out_dim: int = 512):
        super().__init__()

        torch.manual_seed(42)

        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.mean(dim=(2, 3))





def _load_croma_module() -> ModuleType:
    """Dynamically import the CROMA helper without requiring package installation."""

    global _CROMA_MODULE
    if _CROMA_MODULE is not None:
        return _CROMA_MODULE

    module_path = Path(__file__).resolve().parents[2] / "comparison" / "CROMA-main" / "use_croma.py"
    if not module_path.exists():
        raise FileNotFoundError(
            "CROMA weights requested but 'use_croma.py' was not found. "
            "Expected it under comparison/CROMA-main/use_croma.py."
        )

    spec = importlib.util.spec_from_file_location("croma_use_croma", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import CROMA utilities from '{module_path}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CROMA_MODULE = module
    return module


class EvaluationAdapter(nn.Module):
    """Common interface used by unified evaluation to extract embeddings."""

    supports_ssl4eo: bool = True
    supports_hyperbolic: bool = False
    is_lorentz: bool = False
    supports_multimodal_dict: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.base_model: nn.Module = self
        self.dtype_s2: torch.dtype = torch.float32
        self.dtype_s1: torch.dtype = torch.float32
        self._active_modality: str = "s2"


    def compute_embeddings(
        self, images: Any, modality: str = "s2"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return backbone, post-head and projected embeddings for ``images``."""

        backbone = self.compute_backbone(images, modality=modality)
        projected = self.compute_projected(images, modality=modality)
        return backbone, None, projected

    def forward(self, images: Any, *, modality: str = "s2", output: str = "projected"):
        if output == "backbone":
            return self.compute_backbone(images, modality=modality)
        if output == "projected":
            return self.compute_projected(images, modality=modality)
        if output == "both":
            backbone = self.compute_backbone(images, modality=modality)
            projected = self.compute_projected(images, modality=modality)
            return backbone, projected
        raise ValueError(f"Unsupported output mode '{output}'.")

    def prepare_inputs(self, batch: Any, *, device: torch.device, modality='s2') -> Any:
        """Prepare a batch prior to embedding extraction.

        The default implementation expects a tensor containing Sentinel-2 data.
        Sub-classes can override this method to support multi-modal inputs.
        """

        if isinstance(batch, dict):
            raise TypeError("Multi-modal dictionaries are not supported by this adapter")

        tensor = batch
        # print(f'Preparing inputs for modality: {modality}, tensor shape: {tensor.shape if isinstance(tensor, torch.Tensor) else "N/A"}')
        if tensor.ndim == 5 and tensor.shape[1] == 1:  # (B, T, C, H, W)
            tensor = tensor.squeeze(1)

        active_modality = modality
        # active_modality = self.get_active_modality()
        dtype_attr = f"dtype_{active_modality}"
        input_dtype = getattr(self, dtype_attr, self.dtype_s2)
        if device.type != "cuda" and input_dtype in {torch.float16, torch.bfloat16}:
            input_dtype = torch.float32
        return tensor.to(device=device, dtype=input_dtype, non_blocking=True)

    def get_active_modality(self) -> str:
        return getattr(self, "_active_modality", "s2")

    def supports_modality(self, modality: str) -> bool:
        normalized = modality.lower()
        if normalized == "s1":
            return getattr(self, "encoder_s1", None) is not None
        return True

    def set_active_modality(self, modality: str) -> None:
        normalized = modality.lower()
        if normalized not in {"s1", "s2"}:
            raise ValueError(f"Unsupported modality '{modality}'. Expected 's1' or 's2'.")
        if normalized == "s1" and not self.supports_modality("s1"):
            raise ValueError("Adapter does not expose an S1 encoder; cannot activate 's1'.")
        self._active_modality = normalized


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
        preferred = self.get_active_modality()
        if preferred == "s1" and self.encoder_s1 is not None:
            return self.encoder_s1, self.dtype_s1, "s1"
        if preferred == "s2" and self.encoder_s2 is not None:
            return self.encoder_s2, self.dtype_s2, "s2"
        if self.encoder_s2 is not None:
            return self.encoder_s2, self.dtype_s2, "s2"
        if self.encoder_s1 is not None:
            return self.encoder_s1, self.dtype_s1, "s1"
        raise RuntimeError("CIIP adapter does not expose any encoders")

    def compute_backbone(self, images: torch.Tensor, modality='s2') -> torch.Tensor:
        # encoder, dtype, _ = self._default_stream()
        if modality == "s1":
            encoder = self.encoder_s1
            dtype = self.dtype_s1
        else:
            encoder = self.encoder_s2
            dtype = self.dtype_s2

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
        return features

    def compute_projected(self, images: torch.Tensor, modality='s2') -> torch.Tensor:
        # _, dtype, stream = self._default_stream()
        tensor = images.type(self.dtype_s2 if modality == "s2" else self.dtype_s1)

        if modality == "s1":
            encode_fn = self.base_model.encode_s1
        else:
            encode_fn = self.base_model.encode_s2

        if self.is_lorentz:
            return encode_fn(tensor, normalize=True, lorentz=True)
        return encode_fn(tensor, normalize=True)

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
            expected_s2_in = state_dict.get("conv1.weight", torch.empty(0)).shape[1] if isinstance(state_dict, dict) else None
            configured_s2_in = backbone.conv1.weight.shape[1]
            if expected_s2_in is None or expected_s2_in == configured_s2_in:
                missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logging.warning(
                        "Loaded ResNet50 weights with missing=%s, unexpected=%s", missing, unexpected
                    )
            else:
                logging.warning(
                    "Skipping S2 weights load: checkpoint expects %d channels but model is configured for %d.",
                    expected_s2_in,
                    configured_s2_in,
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
            print('Setting up S1 encoder')
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

        # If we only configured an S1 encoder (e.g., in_chans != 13), prefer it as the base.
        if self.encoder_s2 is None and self.encoder_s1 is not None:
            self.base_model = self.encoder_s1

            # # replace fc wtih identity
            # self.encoder_s1.fc = nn.Identity()
            # self.encoder_s2.fc = nn.Identity()

    def _default_stream(self) -> Tuple[nn.Module, torch.dtype, str]:
        preferred = self.get_active_modality()
        if preferred == "s1" and self.encoder_s1 is not None:
            return self.encoder_s1, self.dtype_s1, "s1"
        if preferred == "s2" and self.encoder_s2 is not None:
            return self.encoder_s2, self.dtype_s2, "s2"
        if self.encoder_s2 is not None:
            return self.encoder_s2, self.dtype_s2, "s2"
        if self.encoder_s1 is not None:
            return self.encoder_s1, self.dtype_s1, "s1"
        raise RuntimeError("TorchGeoResNet adapter does not expose any encoders")
        

    def compute_backbone(self, images: torch.Tensor, modality: str = "s2") -> torch.Tensor:
        # encoder, dtype, _ = self._default_stream()
        if modality == "s1":
            encoder = self.encoder_s1
            dtype = self.dtype_s1
        else:
            encoder = self.encoder_s2
            dtype = self.dtype_s2
        features = encoder(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return features
    
    def compute_embeddings(self, images, modality = "s2"):
        backbone = self.compute_backbone(images, modality=modality)
        return backbone, None, None

    # def compute_projected(
    #     self, images: torch.Tensor, modality: str = "s2"
    # ) -> Optional[torch.Tensor]:
    #     return None

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


# Sentinel-2 central wavelengths in microns (13 bands, B1..B12 incl. B10)
S2_WAVELENGTHS_UM: List[float] = [
    0.443,  # B1
    0.490,  # B2
    0.560,  # B3
    0.665,  # B4
    0.705,  # B5
    0.740,  # B6
    0.783,  # B7
    0.842,  # B8
    0.865,  # B8A
    0.945,  # B9
    1.375,  # B10 (cirrus)
    1.610,  # B11
    2.190,  # B12
]

# CROMA uses 12 optical bands (B1–B12 without B10)
S2_WAVELENGTHS_UM_OPTICAL_12: List[float] = [
    w for i, w in enumerate(S2_WAVELENGTHS_UM) if i != 10
]

def _extract_backbone_outputs(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Forward images through a backbone and pool features for evaluation."""

    # if backbone is tpe dofa
    if backbone.__class__.__name__ == "DOFA":
        # print(images.shape)
        features = backbone.forward_features(images, wavelengths=S2_WAVELENGTHS_UM)
        # print(features.shape)
        if features.ndim == 3:
            features = features.mean(dim=1)  # (B, D)
        elif features.ndim != 2:
            raise RuntimeError(f"Unexpected DOFA feature shape: {features.shape}")


    elif hasattr(backbone, "forward_features"):
        features = backbone.forward_features(images)
    else:
        features = backbone(images)

    if isinstance(features, dict):
        for key in ("x", "tokens", "sequence", "feat", "last_hidden_state"):
            candidate = features.get(key)
            if isinstance(candidate, torch.Tensor):
                features = candidate
                break
        else:
            raise RuntimeError(
                "Backbone returned a dict without recognizable feature tensors; "
                f"keys={list(features.keys())}"
            )

    if isinstance(features, (tuple, list)):
        features = features[0]

    if not isinstance(features, torch.Tensor):
        raise TypeError(
            "Backbone outputs must be tensors for unified evaluation; "
            f"received {type(features)}"
        )

    if features.ndim == 3:
        features = features.mean(dim=1)
    elif features.ndim == 4:
        features = features.mean(dim=(2, 3))
    elif features.ndim > 2:
        features = features.flatten(start_dim=1)
    return features


class BackboneOnlyAdapter(EvaluationAdapter):
    """Adapter for backbones that only expose Sentinel-2 features."""

    supports_ssl4eo = True

    def __init__(self, model: nn.Module, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.base_model = model
        self.encoder_s2 = model
        self.encoder_s1 = None
        self.dtype_s2 = dtype
        self.dtype_s1 = dtype

    def compute_backbone(self, images: torch.Tensor, modality: str = "s2") -> torch.Tensor:
        # # get in channels
        # in_channels = images.shape[1]
        # model_in_channels = self.encoder_s2.conv1.in_channels
        # if in_channels != model_in_channels:
        #     if model_in_channels == 3:
        #         # select rgb from images (3,2,1)
        #         images = images[:, [3, 2, 1], :, :]

        if modality.lower() != "s2":
            raise ValueError("This adapter only supports Sentinel-2 imagery.")
        return _extract_backbone_outputs(self.encoder_s2, images)

    def compute_embeddings(
        self, images: Any, modality: str = "s2"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # # get in channels
        # in_channels = images.shape[1]
        # model_in_channels = self.encoder_s2.conv1.in_channels
        # if in_channels != model_in_channels:
        #     if model_in_channels == 3:
        #         # select rgb from images (3,2,1)
        #         images = images[:, [3, 2, 1], :, :]

        backbone = self.compute_backbone(images, modality=modality)
        return backbone, None, None


class OpenClipVisionAdapter(EvaluationAdapter):
    """Adapter for OpenCLIP vision encoders (RGB-only)."""

    supports_ssl4eo = True

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.base_model = clip_model
        self.encoder_s2 = getattr(clip_model, "visual", clip_model)
        self.encoder_s1 = None

        try:
            self.dtype_s2 = next(self.encoder_s2.parameters()).dtype
        except StopIteration:
            self.dtype_s2 = torch.float32
        self.dtype_s1 = self.dtype_s2

    def prepare_inputs(self, batch: Any, *, device: torch.device, modality: str = "s2") -> torch.Tensor:
        tensor = super().prepare_inputs(batch, device=device, modality=modality)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def compute_backbone(self, images: torch.Tensor, modality: str = "s2") -> torch.Tensor:
        features = self.encoder_s2(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            features = features[:, 0]
        elif features.ndim > 2:
            features = features.mean(dim=tuple(range(2, features.ndim)))
        return features

    def compute_projected(self, images: torch.Tensor, modality: str = "s2") -> Optional[torch.Tensor]:
        encode_image = getattr(self.base_model, "encode_image", None)
        if encode_image is None:
            return None

        projected = encode_image(images)
        if isinstance(projected, (tuple, list)):
            projected = projected[0]
        if projected.ndim == 3:
            projected = projected[:, 0]
        return projected


class CromaEvaluationAdapter(EvaluationAdapter):
    """Adapter around the published CROMA base checkpoint."""

    supports_ssl4eo = True
    supports_multimodal_dict = True

    def __init__(self, *, weights_path: Path, image_resolution: int = 120) -> None:
        super().__init__()

        if weights_path is None:
            raise ValueError("'weights_path' must be provided to load CROMA.")

        module = _load_croma_module()
        PretrainedCROMA = getattr(module, "PretrainedCROMA")

        model = PretrainedCROMA(
            pretrained_path=str(weights_path),
            size="base",
            modality="both",
            image_resolution=image_resolution,
        )

        self.base_model = model.eval()
        self.encoder_s1 = getattr(model, "s1_encoder", None)
        self.encoder_s2 = getattr(model, "s2_encoder", None)
        self.dtype_s1 = torch.float32
        self.dtype_s2 = torch.float32
        self.image_resolution = image_resolution
        # self._last_outputs: Dict[str, torch.Tensor] = {}
        # self._last_inputs: Optional[Dict[str, Optional[torch.Tensor]]] = None

    def prepare_inputs(self, batch: Any, *, device: torch.device, modality: Optional[str]) -> Dict[str, Optional[torch.Tensor]]:
        # print(type(batch))
        
        if isinstance(batch, dict):
            # print(batch.keys())
            
            sar_tensor = self._extract_modality(batch, {"sar", "s1", "sentinel1"})
            optical_tensor = self._extract_modality(batch, {"optical", "s2", "s2l1c", "s2l2a", "sentinel2"})
        else:
            if modality:
                # print(f'Modality specified: {modality}')
                if modality.lower() == "s1":
                    sar_tensor = batch
                    optical_tensor = None
                else:
                    sar_tensor = None
                    optical_tensor = batch
            else:
                if self.get_active_modality() == "s1":
                    sar_tensor = batch
                    optical_tensor = None
                else:
                    sar_tensor = None
                    optical_tensor = batch
            

        if optical_tensor is not None:
            optical_prepared = (
                self._prepare_optical(optical_tensor, device) if optical_tensor is not None else None
            )
            return optical_prepared
        if sar_tensor is not None:
            sar_prepared = self._prepare_sar(sar_tensor, device) if sar_tensor is not None else None
            return sar_prepared

        raise ValueError("CROMA adapter requires at least one modality in the input batch.")

    def compute_embeddings(
        self, inputs: Dict[str, Optional[torch.Tensor]],
        modality: str = "s2"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # print(f'Modality: {modality}, Input data dim: {inputs.shape if isinstance(inputs, torch.Tensor) else "dict"}')
        # if shape of inputs has 5 dims, squeeze 2nd dim
        if inputs.ndim == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)

        outputs = self._encode_modality(inputs, modality)

        # modality = self.get_active_modality()
        if modality == "s1":
            sar_encodings = outputs.get("sar_encodings")
            sar_gap = outputs.get("sar_gap")
            if sar_encodings is None or sar_gap is None:
                raise RuntimeError("SAR modality is required when evaluating CROMA with 's1'.")
            backbone = sar_encodings.flatten(start_dim=1)
            # projected = F.normalize(sar_gap, p=2, dim=1, eps=1e-12)
            return {'backbone': sar_gap} # sar_gap is the backbone features

        optical_encodings = outputs.get("optical_encodings")
        optical_gap = outputs.get("optical_gap")
        if optical_encodings is None or optical_gap is None:
            raise RuntimeError("Optical modality is required when evaluating CROMA.")

        patch_embeddings = optical_encodings.flatten(start_dim=1)
        # projected = F.normalize(optical_gap, p=2, dim=1, eps=1e-12)
        return {'backbone': optical_gap}

    def compute_backbone(self, images: Any, modality: str = "s2") -> torch.Tensor:
        prepared = self._ensure_prepared_inputs(images, modality)
        outputs = self.compute_embeddings(prepared, modality)
        return outputs['backbone']

    def _ensure_prepared_inputs(
        self, images: Any, modality: str
    ) -> Dict[str, Optional[torch.Tensor]]:
        if isinstance(images, dict):
            candidate = images
        else:
            if not isinstance(images, torch.Tensor):
                raise TypeError(
                    "CROMA adapters expect tensors or modality dictionaries for embedding extraction."
                )
            candidate = {"optical": None, "sar": None}
            # print(modality)
            if modality.lower() == "s1":
                candidate["sar"] = images
            else:
                candidate["optical"] = images

        device = None
        for value in candidate.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        if device is None:
            device = next(self.base_model.parameters()).device

        prepared = self.prepare_inputs(candidate, device=device, modality=modality)
        # if not isinstance(prepared, dict):
        #     raise TypeError("CROMA prepare_inputs must return a dictionary of modalities.")
        return prepared

    # def compute_joint_embeddings(
    #     self, inputs: Optional[Dict[str, Optional[torch.Tensor]]] = None
    # ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    #     cached_inputs = inputs if inputs is not None else self._last_inputs
    #     outputs = self._encode_modalities(cached_inputs, include_joint=True)
    #     joint_encodings = outputs.get("joint_encodings")
    #     joint_gap = outputs.get("joint_gap")
    #     if joint_encodings is None or joint_gap is None:
    #         return None
    #     joint_backbone = joint_encodings.flatten(start_dim=1)
    #     joint_projected = F.normalize(joint_gap, p=2, dim=1, eps=1e-12)
    #     return joint_backbone, joint_gap, joint_projected

    def _encode_modality(
        self,
        inputs: Optional[Dict[str, Optional[torch.Tensor]]],
        modality: str,
        # *,
        # include_joint: bool,
    ) -> Dict[str, torch.Tensor]:
        if inputs is None:
            inputs = getattr(self, "_last_inputs", None)
            if inputs is None:
                raise RuntimeError("No cached inputs available for computing CROMA embeddings.")


        device = inputs.device
        attn_bias = self.base_model.attn_bias.to(device=device, dtype=inputs.dtype)
        outputs: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            if modality == 's2':
                # print('Encoding Optical modality')
                # print(inputs.shape)
                optical_enc = self.base_model.s2_encoder(imgs=inputs, attn_bias=attn_bias)
                optical_gap = self.base_model.GAP_FFN_s2(optical_enc.mean(dim=1))
                outputs["optical_encodings"] = optical_enc
                outputs["optical_gap"] = optical_gap

            elif modality == 's1': 
                # print('Encoding SAR modality')
                # print(inputs.shape)
                sar_enc = self.base_model.s1_encoder(imgs=inputs, attn_bias=attn_bias)
                sar_gap = self.base_model.GAP_FFN_s1(sar_enc.mean(dim=1))
                outputs["sar_encodings"] = sar_enc
                outputs["sar_gap"] = sar_gap

            # if include_joint and sar_enc is not None and optical_enc is not None:
            #     joint_enc = self.base_model.cross_encoder(
            #         x=sar_enc,
            #         context=optical_enc,
            #         relative_position_bias=attn_bias,
            #     )
            #     joint_gap = joint_enc.mean(dim=1)
            #     outputs["joint_encodings"] = joint_enc
            #     outputs["joint_gap"] = joint_gap

        # self._last_outputs = outputs
        # self._last_inputs = {"optical": optical, "sar": sar}
        return outputs

    def _prepare_optical(self, tensor: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        if tensor is None:
            raise ValueError("Optical imagery is required for CROMA evaluation.")

        tensor = tensor.to(device=device, dtype=self.dtype_s2, non_blocking=True)
        if tensor.ndim == 5:
            tensor = tensor.mean(dim=1)
        if tensor.ndim == 4 and tensor.size(1) == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # if tensor.size(1) == 13:
        #     tensor = torch.cat([tensor[:, :10], tensor[:, 11:]], dim=1)
        # if tensor.size(1) != 12:
        #     raise RuntimeError(
        #         f"CROMA expects 12 Sentinel-2 bands; received {tensor.size(1)} bands."
        #     )

        if tensor.size(-1) != self.image_resolution or tensor.size(-2) != self.image_resolution:
            tensor = F.adaptive_avg_pool2d(tensor, output_size=(self.image_resolution, self.image_resolution))
        return tensor

    def _prepare_sar(self, tensor: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
        if tensor is None:
            return None

        tensor = tensor.to(device=device, dtype=self.dtype_s1, non_blocking=True)
        if tensor.ndim == 5:
            tensor = tensor.mean(dim=1)
        if tensor.ndim == 4 and tensor.size(1) == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.size(1) < 2:
            raise RuntimeError("CROMA expects two SAR channels.")
        if tensor.size(1) > 2:
            tensor = tensor[:, :2]

        if tensor.size(-1) != self.image_resolution or tensor.size(-2) != self.image_resolution:
            tensor = F.adaptive_avg_pool2d(tensor, output_size=(self.image_resolution, self.image_resolution))
        return tensor

    @staticmethod
    def _extract_modality(batch: Dict[str, torch.Tensor], aliases: set[str]) -> Optional[torch.Tensor]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and key.lower() in aliases:
                return value
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                return value
        return None
    
def _clean_state_dict(raw_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip DataParallel prefixes from checkpoint state dictionaries."""

    return {key.replace("module.", ""): value for key, value in raw_state.items()}

def _infer_projection_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """Infer embedding and pre-projection dimensionality from the checkpoint."""

    if "encoder_s2.proj.weight" in state_dict:
        weight = state_dict["encoder_s2.proj.weight"]
        print(f'Using proj weight to infer dims: {weight.shape}')
        return int(weight.shape[0]), int(weight.shape[1])
    if "encoder_s2.proj" in state_dict:
        weight = state_dict["encoder_s2.proj"]
        print(f'Using proj tensor to infer dims: {weight.shape}')
        return int(weight.shape[1]), int(weight.shape[0])
    if "encoder_s1.proj" in state_dict:
        weight = state_dict["encoder_s1.proj"]
        print(f'Using S1 proj tensor to infer dims: {weight.shape}')
        return int(weight.shape[1]), int(weight.shape[0])
    if "encoder_s2.fc.weight" in state_dict:
        weight = state_dict["encoder_s2.fc.weight"]
        print(f'Using fc weight to infer dims: {weight.shape}')
        return int(weight.shape[0]), int(weight.shape[1])
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

def _infer_vit_params(state_dict: Dict[str, torch.Tensor], prefix: str) -> Tuple[int, int, int]:
    """Infer ViT width, depth, and patch size from checkpoint keys."""
    conv_key = f"{prefix}.conv1.weight"
    if conv_key not in state_dict:
        raise RuntimeError(f"Unable to infer ViT parameters; missing {conv_key}")
    conv_weight = state_dict[conv_key]
    width = int(conv_weight.shape[0])
    patch_size = int(conv_weight.shape[-1])

    layer_indices = set()
    marker = f"{prefix}.transformer.resblocks."
    for key in state_dict.keys():
        if key.startswith(marker):
            remainder = key[len(marker):]
            idx_str = remainder.split(".", 1)[0]
            if idx_str.isdigit():
                layer_indices.add(int(idx_str))
    layers = max(layer_indices) + 1 if layer_indices else 0
    if layers == 0:
        raise RuntimeError(f"Unable to infer ViT depth from {prefix} resblocks")
    return width, layers, patch_size

def _infer_ciip_framework(state_dict: Dict[str, torch.Tensor]) -> str:
    """Heuristic to distinguish CIIP ResNet vs ViT-style encoders."""
    transformer_markers = (
        "encoder_s1.transformer.",
        "encoder_s2.transformer.",
        "encoder_s1.positional_embedding",
        "encoder_s2.positional_embedding",
        "encoder_s1.proj",
        "encoder_s2.proj",
    )
    for key in state_dict:
        if any(marker in key for marker in transformer_markers):
            return "transformer"
    return "resnet50"


def build_model_from_checkpoint(
    checkpoint: Path,
    *,
    framework: Optional[str] = None,
) -> Tuple[nn.Module, bool]:
    """Instantiate a CIIP or LorentzCIIP model from a checkpoint path."""
    ckpt = torch.load(checkpoint, map_location="cpu",  weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = _clean_state_dict(state_dict)

    # Drop text encoder weights for backwards compatibility. New checkpoints may
    # include a CLIP-style text tower that the eval wrappers do not consume.
    cleaned = {
        k: v for k, v in cleaned.items()
        if not k.startswith("encoder_text.")
    }
    embed_dim, pre_dim = _infer_projection_dims(cleaned)
    print(f'Inferred embed_dim: {embed_dim}, pre_dim: {pre_dim}')
    is_lorentz = any(key.startswith("curv") or "lorentz" in key for key in cleaned)

    print(f'Is Lorentz model: {is_lorentz}')

    # infer nummber of s2 bands in checkpoint
    s2_bands = _infer_s2_bands(cleaned)
    logging.info(f"Inferred S2 bands from checkpoint: {s2_bands}")

    if framework is None:
        framework = _infer_ciip_framework(cleaned)
    logging.info("Using CIIP framework: %s", framework)

    if framework == "transformer":
        s1_width, s1_layers, s1_patch_size = _infer_vit_params(cleaned, "encoder_s1")
        s2_width, s2_layers, s2_patch_size = _infer_vit_params(cleaned, "encoder_s2")
    else:
        s1_width, s1_layers, s1_patch_size = 32, (3, 4, 6, 3), 16
        s2_width, s2_layers, s2_patch_size = 32, (3, 4, 6, 3), 16

    kwargs = dict(
        embed_dim=embed_dim,
        # pre_projection_dim=pre_dim,
        s1_resolution=224,
        s1_layers=s1_layers,
        s1_width=s1_width,
        s1_patch_size=s1_patch_size,
        s1_bands=2,
        s2_resolution=224,
        s2_layers=s2_layers,
        s2_width=s2_width,
        s2_patch_size=s2_patch_size,
        s2_bands=s2_bands,
        framework=framework,
    )

    if is_lorentz:
        model: nn.Module = LorentzCIIP(**kwargs)
    else:
        model = CIIP(**kwargs)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        logging.warning("Checkpoint loaded with missing=%s, unexpected=%s", missing, unexpected)

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


def _build_llama3_ms_clip_adapter() -> EvaluationAdapter:
    try:
        import open_clip  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "open_clip is required to load the Llama3-MS-CLIP-base model for evaluation."
        ) from exc

    weights_path = Path("/home/juro4948/ciip/ciip/Llama3_MS_CLIP_weights.pt")

    model_cfg = open_clip.get_model_config("ViT-B-16")
    desired_in_chans = 10

    if weights_path.exists():
        logging.info("Loading Llama3-MS-CLIP weights from %s", weights_path)
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained=None,
            text_cfg=model_cfg.get("text_cfg"),
        )

        conv_key = "visual.conv1.weight"
        old_conv = model.visual.conv1
        new_conv = nn.Conv2d(
            in_channels=desired_in_chans,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            bias=False,
        )
        model.visual.conv1 = new_conv

        target_state = model.state_dict()

        raw_state = torch.load(weights_path, map_location="cpu")
        if isinstance(raw_state, dict) and "state_dict" in raw_state:
            raw_state = raw_state["state_dict"]

        prefixes = (
            "clip_base_model.model.",
            "image_encoder.model.",
            "text_encoder.model.",
        )

        #print the sate dict keys
        print(f'Raw state dict keys: {list(raw_state.keys())}')
        # print dim of clip_base_model.model.visual.conv1.weight
        print(f'Raw conv1 weight shape: {raw_state["clip_base_model.model.visual.conv1.weight"].shape}')
        
        filtered = {}
        for key, tensor in raw_state.items():
            trimmed = key
            for prefix in prefixes:
                if trimmed.startswith(prefix):
                    trimmed = trimmed[len(prefix) :]
                    break
            if trimmed in target_state and target_state[trimmed].shape == tensor.shape:
                filtered[trimmed] = tensor

        
        # Always try to load conv1 by padding/truncating channels to match target.
        if conv_key in raw_state:
            w_ckpt = raw_state[conv_key]
            if w_ckpt.shape[1] != desired_in_chans:
                if w_ckpt.shape[1] < desired_in_chans:
                    pad = w_ckpt.new_zeros(
                        w_ckpt.shape[0],
                        desired_in_chans - w_ckpt.shape[1],
                        *w_ckpt.shape[2:],
                    )
                    w_ckpt = torch.cat([w_ckpt, pad], dim=1)
                else:
                    w_ckpt = w_ckpt[:, :desired_in_chans, :, :]
            filtered[conv_key] = w_ckpt

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            logging.info("MS-CLIP load: missing keys (ignored): %s", missing)
        if unexpected:
            logging.info("MS-CLIP load: unexpected keys (ignored): %s", unexpected)
    else:
        raise FileNotFoundError(
            f"Local MS-CLIP weights not found at {weights_path}. "
            "Please download the weights and place them at this path."
        )
        # # Fallback to the published vision encoder (ViT-B/16, LAION-2B) that underpins MS-CLIP.
        # logging.warning(
        #     "Local MS-CLIP weights not found at %s; using open_clip pretrained vision backbone.",
        #     weights_path,
        # )
        # model, _, _ = open_clip.create_model_and_transforms(
        #     "ViT-B-16",
        #     pretrained="laion2b_s34b_b88k",
        #     text_cfg=model_cfg.get("text_cfg"),
        # )

    model.eval()
    return OpenClipVisionAdapter(model)


def _build_backbone_adapter(model_weights: Optional[str]) -> EvaluationAdapter:
    if model_weights is None:
        raise ValueError("model_weights must be provided for backbone-only models")

    normalized = model_weights.lower()
    if normalized == "rcf_13ch":
        model = RandomConvFeatures(in_chans=13, out_dim=512)
    elif normalized == "dofa_base_s2_13ch":
        model = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)
    elif normalized == "scalemae_large_rgb":
        model = scalemae_large_patch16(weights=ScaleMAELarge16_Weights.FMOW_RGB)
    elif normalized == "resnet18_s2_all_moco":
        model = resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO)
        model.fc = nn.Identity()
    elif normalized == "resnet18_s2_rgb_moco":
        model = resnet18(weights=ResNet18_Weights.SENTINEL2_RGB_MOCO)
        model.fc = nn.Identity()
    elif normalized == "resnet50_s2_rgb_moco":
        model = resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO)
        model.fc = nn.Identity()
    elif normalized == "resnet152_imagenet_rgb":
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    elif normalized == "vitsmall16_s2_all_moco":
        model = vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_MOCO)
    elif normalized == "llama3_ms_clip_base":
        return _build_llama3_ms_clip_adapter()
    else:
        raise ValueError(f"Unsupported backbone-only weights '{model_weights}'.")

    model.eval()
    return BackboneOnlyAdapter(model)


def build_evaluation_adapter(
    *,
    model_type: str,
    checkpoint: Optional[Path],
    model_weights: Optional[str],
    in_chans: int = 13,
    croma_weights: Optional[Path] = None,
    croma_image_resolution: int = 120,
    ciip_framework: Optional[str] = None,
) -> EvaluationAdapter:
    """Create an evaluation adapter based on the requested model type."""

    if model_type == "ciip_checkpoint":
        if checkpoint is None:
            raise ValueError("checkpoint must be provided when using 'ciip_checkpoint'")
        model, is_lorentz = build_model_from_checkpoint(checkpoint, framework=ciip_framework)
        return CiipEvaluationAdapter(model, is_lorentz=is_lorentz)

    if model_type == "torchgeo_resnet50":
        weights = _resolve_resnet_weights(model_weights)
        return TorchGeoResNetAdapter(weights=weights, in_chans=in_chans)

    if model_type == "backbone_only":
        return _build_backbone_adapter(model_weights)

    if model_type == "croma":
        if croma_weights is None:
            raise ValueError("'croma_weights' must be provided when model_type='croma'.")
        return CromaEvaluationAdapter(
            weights_path=croma_weights,
            image_resolution=int(croma_image_resolution),
        )

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        "Valid options are 'ciip_checkpoint', 'torchgeo_resnet50', 'backbone_only' and 'croma'."
    )


__all__ = [
    "EvaluationAdapter",
    "CiipEvaluationAdapter",
    "TorchGeoResNetAdapter",
    "BackboneOnlyAdapter",
    "OpenClipVisionAdapter",
    "CromaEvaluationAdapter",
    "build_model_from_checkpoint",
    "build_evaluation_adapter",
]
