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

from torchgeo.models import ResNet50_Weights, resnet50

from ciip.model_ciip import CIIP, LorentzCIIP


_CROMA_MODULE: Optional[ModuleType] = None


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

    # # These helpers are intentionally thin wrappers so that the evaluation
    # # pipeline can treat all models uniformly.
    # def compute_backbone(
    #     self, images: torch.Tensor, modality: str = "s2"
    # ) -> torch.Tensor:  # pragma: no cover - interface
    #     raise NotImplementedError

    # def compute_posthead(
    #     self, images: torch.Tensor, modality: str = "s2"
    # ) -> torch.Tensor:  # pragma: no cover - interface
    #     raise NotImplementedError

    # def compute_projected(
    #     self, images: torch.Tensor, modality: str = "s2"
    # ) -> torch.Tensor:  # pragma: no cover - interface
    #     raise NotImplementedError

    def compute_embeddings(self, images: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return backbone, post-head and projected embeddings for ``images``."""

        backbone = self.compute_backbone(images)
        posthead = self.compute_posthead(images)
        projected = self.compute_projected(images)
        return backbone, posthead, projected

    def prepare_inputs(self, batch: Any, *, device: torch.device, modality='s2') -> Any:
        """Prepare a batch prior to embedding extraction.

        The default implementation expects a tensor containing Sentinel-2 data.
        Sub-classes can override this method to support multi-modal inputs.
        """

        if isinstance(batch, dict):
            raise TypeError("Multi-modal dictionaries are not supported by this adapter")

        tensor = batch
        if tensor.ndim == 5:  # (B, T, C, H, W)
            tensor = tensor.mean(dim=1)

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

    def compute_posthead(self, images: torch.Tensor, modality='s2') -> torch.Tensor:
        _, dtype, stream = self._default_stream()
        tensor = images.type(self.dtype_s2 if modality == "s2" else self.dtype_s1)

        if modality == "s1":
            encode_fn = self.base_model.encode_s1
        else:
            encode_fn = self.base_model.encode_s2

        if self.is_lorentz:
            return encode_fn(tensor, normalize=False, lorentz=False)
        return encode_fn(tensor, normalize=False)
    
        # post = self._encode_projected(images, projected=Fakse)
        # if isinstance(post, (tuple, list)):
        #     post = post[0]
        # if post.ndim > 2:
        #     post = post.flatten(start_dim=1)
        # return post

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
        # projected = self._encode_projected(images, project=True)
        # if isinstance(projected, (tuple, list)):
        #     projected = projected[0]
        # if projected.ndim > 2:
        #     projected = projected.flatten(start_dim=1)
        # return projected

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
        encoder, dtype, _ = self._default_stream()
        features = encoder(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return features

    # def compute_posthead(self, images: torch.Tensor, modality: str = "s2") -> torch.Tensor:
    #     return None

    # def compute_projected(self, images: torch.Tensor, modality: str = "s2") -> torch.Tensor:
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
        self._last_outputs: Dict[str, torch.Tensor] = {}
        self._last_inputs: Optional[Dict[str, Optional[torch.Tensor]]] = None

    def prepare_inputs(self, batch: Any, *, device: torch.device, modality: Optional[str]) -> Dict[str, Optional[torch.Tensor]]:
        print(type(batch))
        if isinstance(batch, dict):
            sar_tensor = self._extract_modality(batch, {"sar", "s1", "sentinel1"})
            optical_tensor = self._extract_modality(batch, {"optical", "s2", "s2l1c", "s2l2a", "sentinel2"})
        else:
            if modality:
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
            

        optical_prepared = (
            self._prepare_optical(optical_tensor, device) if optical_tensor is not None else None
        )
        sar_prepared = self._prepare_sar(sar_tensor, device) if sar_tensor is not None else None

        return {"optical": optical_prepared, "sar": sar_prepared}

    def compute_embeddings(
        self, inputs: Dict[str, Optional[torch.Tensor]],
        modality: str = "s2"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        outputs = self._encode_modalities(inputs, include_joint=True)

        # modality = self.get_active_modality()
        if modality == "s1":
            sar_encodings = outputs.get("sar_encodings")
            sar_gap = outputs.get("sar_gap")
            if sar_encodings is None or sar_gap is None:
                raise RuntimeError("SAR modality is required when evaluating CROMA with 's1'.")
            backbone = sar_encodings.flatten(start_dim=1)
            projected = F.normalize(sar_gap, p=2, dim=1, eps=1e-12)
            return backbone, sar_gap, projected

        optical_encodings = outputs.get("optical_encodings")
        optical_gap = outputs.get("optical_gap")
        if optical_encodings is None or optical_gap is None:
            raise RuntimeError("Optical modality is required when evaluating CROMA.")

        backbone = optical_encodings.flatten(start_dim=1)
        projected = F.normalize(optical_gap, p=2, dim=1, eps=1e-12)
        return backbone, optical_gap, projected

    def compute_backbone(self, images: Any, modality: str = "s2") -> torch.Tensor:
        prepared = self._ensure_prepared_inputs(images, modality)
        backbone, _, _ = self.compute_embeddings(prepared, modality)
        return backbone

    # def compute_posthead(self, images: Any, modality: str = "s2") -> torch.Tensor:
    #     prepared = self._ensure_prepared_inputs(images, modality)
    #     _, posthead, _ = self.compute_embeddings(prepared, modality)
    #     if posthead is None:
    #         raise RuntimeError("Post-head embeddings unavailable for CROMA.")
    #     return posthead

    # def compute_projected(self, images: Any, modality: str = "s2") -> torch.Tensor:
    #     prepared = self._ensure_prepared_inputs(images, modality)
    #     _, _, projected = self.compute_embeddings(prepared, modality)
    #     if projected is None:
    #         raise RuntimeError("Projected embeddings unavailable for CROMA.")
    #     return projected

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
            print(modality)
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

        prepared = self.prepare_inputs(candidate, device=device)
        if not isinstance(prepared, dict):
            raise TypeError("CROMA prepare_inputs must return a dictionary of modalities.")
        return prepared

    def compute_joint_embeddings(
        self, inputs: Optional[Dict[str, Optional[torch.Tensor]]] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        cached_inputs = inputs if inputs is not None else self._last_inputs
        outputs = self._encode_modalities(cached_inputs, include_joint=True)
        joint_encodings = outputs.get("joint_encodings")
        joint_gap = outputs.get("joint_gap")
        if joint_encodings is None or joint_gap is None:
            return None
        joint_backbone = joint_encodings.flatten(start_dim=1)
        joint_projected = F.normalize(joint_gap, p=2, dim=1, eps=1e-12)
        return joint_backbone, joint_gap, joint_projected

    def _encode_modalities(
        self,
        inputs: Optional[Dict[str, Optional[torch.Tensor]]],
        *,
        include_joint: bool,
    ) -> Dict[str, torch.Tensor]:
        if inputs is None:
            inputs = getattr(self, "_last_inputs", None)
            if inputs is None:
                raise RuntimeError("No cached inputs available for computing CROMA embeddings.")

        optical = inputs.get("optical")
        sar = inputs.get("sar") if inputs is not None else None

        require_optical = self.get_active_modality() != "s1"
        if optical is None and require_optical:
            raise RuntimeError("Optical imagery must be provided for CROMA evaluation.")

        sample_tensor = optical if optical is not None else sar
        if sample_tensor is None:
            raise RuntimeError("At least one modality must be provided for CROMA evaluation.")

        device = sample_tensor.device
        attn_bias = self.base_model.attn_bias.to(device=device, dtype=sample_tensor.dtype)

        outputs: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            optical_enc = None
            optical_gap = None
            if optical is not None:
                optical_enc = self.base_model.s2_encoder(imgs=optical, attn_bias=attn_bias)
                optical_gap = self.base_model.GAP_FFN_s2(optical_enc.mean(dim=1))
                outputs["optical_encodings"] = optical_enc
                outputs["optical_gap"] = optical_gap

            sar_enc = None
            sar_gap = None
            if sar is not None:
                sar = sar.to(device=device, dtype=self.dtype_s1, non_blocking=True)
                sar_enc = self.base_model.s1_encoder(imgs=sar, attn_bias=attn_bias)
                sar_gap = self.base_model.GAP_FFN_s1(sar_enc.mean(dim=1))
                outputs["sar_encodings"] = sar_enc
                outputs["sar_gap"] = sar_gap

            if include_joint and sar_enc is not None and optical_enc is not None:
                joint_enc = self.base_model.cross_encoder(
                    x=sar_enc,
                    context=optical_enc,
                    relative_position_bias=attn_bias,
                )
                joint_gap = joint_enc.mean(dim=1)
                outputs["joint_encodings"] = joint_enc
                outputs["joint_gap"] = joint_gap

        self._last_outputs = outputs
        self._last_inputs = {"optical": optical, "sar": sar}
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

        if tensor.size(1) == 13:
            tensor = torch.cat([tensor[:, :10], tensor[:, 11:]], dim=1)
        if tensor.size(1) != 12:
            raise RuntimeError(
                f"CROMA expects 12 Sentinel-2 bands; received {tensor.size(1)} bands."
            )

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

def build_model_from_checkpoint(checkpoint: Path) -> Tuple[nn.Module, bool]:
    """Instantiate a CIIP or LorentzCIIP model from a checkpoint path."""
    ckpt = torch.load(checkpoint, map_location="cpu",  weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = _clean_state_dict(state_dict)
    embed_dim, pre_dim = _infer_projection_dims(cleaned)
    print(f'Inferred embed_dim: {embed_dim}, pre_dim: {pre_dim}')
    is_lorentz = any(key.startswith("curv") or "lorentz" in key for key in cleaned)

    print(f'Is Lorentz model: {is_lorentz}')



    # infer nummber of s2 bands in checkpoint
    s2_bands = _infer_s2_bands(cleaned)
    logging.info(f"Inferred S2 bands from checkpoint: {s2_bands}")

    # quit()


    kwargs = dict(
        embed_dim=embed_dim,
        # pre_projection_dim=pre_dim,
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
    print(cleaned.keys())
    #
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
    croma_weights: Optional[Path] = None,
    croma_image_resolution: int = 120,
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

    if model_type == "croma":
        if croma_weights is None:
            raise ValueError("'croma_weights' must be provided when model_type='croma'.")
        return CromaEvaluationAdapter(
            weights_path=croma_weights,
            image_resolution=int(croma_image_resolution),
        )

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        "Valid options are 'ciip_checkpoint', 'torchgeo_resnet50' and 'croma'."
    )


__all__ = [
    "EvaluationAdapter",
    "CiipEvaluationAdapter",
    "TorchGeoResNetAdapter",
    "CromaEvaluationAdapter",
    "build_model_from_checkpoint",
    "build_evaluation_adapter",
]

