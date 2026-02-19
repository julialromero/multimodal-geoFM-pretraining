"""Utility helpers for loading evaluation models without circular imports."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
import yaml

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
_GALILEO_MODULE: Optional[ModuleType] = None
_CIIP_CLASS_CACHE: Dict[str, Tuple[Type[nn.Module], Type[nn.Module]]] = {}
PANGAEA_CONFIG_ROOT = Path("/local/ms-data/pangaea-bench/configs")
GALILEO_SINGLE_FILE = Path("/local/ms-data/galileo/single_file_galileo.py")
GALILEO_DEFAULT_WEIGHTS = Path("/local/ms-data/pangaea-bench/pretrained_models/models/base")
GALILEO_NORMALIZATION_JSON = Path("/local/ms-data/galileo/config/normalization.json")


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


def _normalize_ciip_model_source(source: Optional[str]) -> str:
    if source is None:
        source = os.getenv("CIIP_MODEL_SOURCE", "current")
    normalized = str(source).strip().lower().replace("-", "_")
    aliases = {
        "current": "current",
        "default": "current",
        "legacy": "current",
        "posenc": "posenc",
        "models_posenc": "posenc",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported CIIP model source '{source}'. Expected one of: current, posenc."
        )
    return resolved


def _load_module_from_file(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module '{module_name}' from '{module_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_ciip_classes(ciip_model_source: Optional[str]) -> Tuple[Type[nn.Module], Type[nn.Module]]:
    source = _normalize_ciip_model_source(ciip_model_source)
    cached = _CIIP_CLASS_CACHE.get(source)
    if cached is not None:
        return cached

    if source == "current":
        classes = (CIIP, LorentzCIIP)
        _CIIP_CLASS_CACHE[source] = classes
        return classes

    # Load alternate CIIP modules from ciip/models-posenc using a synthetic package
    # name so relative imports (e.g., ".model") continue to work.
    posenc_dir = Path(__file__).resolve().parents[1] / "models-posenc"
    model_ciip_path = posenc_dir / "model_ciip.py"
    model_path = posenc_dir / "model.py"
    if not model_ciip_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Requested ciip_model_source='posenc' but expected files were not found at "
            f"{model_ciip_path} and {model_path}."
        )

    package_name = "ciip._models_posenc_runtime"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(posenc_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    # Reuse the existing lorentz implementation from the main ciip package.
    import ciip.lorentz as _ciip_lorentz
    import ciip.mae_decoder as _ciip_mae_decoder
    import ciip.masking as _ciip_masking

    sys.modules[f"{package_name}.lorentz"] = _ciip_lorentz
    posenc_mae_decoder_path = posenc_dir / "mae_decoder.py"
    if posenc_mae_decoder_path.exists():
        if f"{package_name}.mae_decoder" not in sys.modules:
            _load_module_from_file(f"{package_name}.mae_decoder", posenc_mae_decoder_path)
    else:
        # models-posenc imports ".mae_decoder"; map that import to the shared ciip module.
        sys.modules[f"{package_name}.mae_decoder"] = _ciip_mae_decoder
    posenc_masking_path = posenc_dir / "masking.py"
    if posenc_masking_path.exists():
        if f"{package_name}.masking" not in sys.modules:
            _load_module_from_file(f"{package_name}.masking", posenc_masking_path)
    else:
        # models-posenc imports ".masking"; map that import to the shared ciip masking module.
        sys.modules[f"{package_name}.masking"] = _ciip_masking
    if f"{package_name}.model" not in sys.modules:
        _load_module_from_file(f"{package_name}.model", model_path)
    if f"{package_name}.model_ciip" not in sys.modules:
        _load_module_from_file(f"{package_name}.model_ciip", model_ciip_path)

    model_ciip_module = sys.modules[f"{package_name}.model_ciip"]
    ciip_cls = getattr(model_ciip_module, "CIIP")
    lorentz_cls = getattr(model_ciip_module, "LorentzCIIP")
    classes = (ciip_cls, lorentz_cls)
    _CIIP_CLASS_CACHE[source] = classes
    return classes





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


def _load_galileo_module() -> ModuleType:
    """Dynamically import the Galileo single-file helper without extra dependencies."""

    global _GALILEO_MODULE
    if _GALILEO_MODULE is not None:
        return _GALILEO_MODULE

    if not GALILEO_SINGLE_FILE.exists():
        raise FileNotFoundError(
            "Galileo weights requested but single_file_galileo.py was not found. "
            f"Expected it at {GALILEO_SINGLE_FILE}."
        )

    spec = importlib.util.spec_from_file_location("galileo_single_file", GALILEO_SINGLE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import Galileo utilities from '{GALILEO_SINGLE_FILE}'.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GALILEO_MODULE = module
    return module


def _load_galileo_normalization_values(path: Path) -> Dict[Any, Any]:
    """Load Galileo normalization JSON using the same key casting as Dataset.load_normalization_values."""

    if not path.exists():
        raise FileNotFoundError(f"Galileo normalization file not found: {path}")
    with path.open("r") as handle:
        norm_dict = json.load(handle)

    output: Dict[Any, Any] = {}
    for key, value in norm_dict.items():
        if isinstance(key, str) and "n" not in key:
            output[int(key)] = value
        else:
            output[key] = value
    return output


def _load_pangaea_encoder_cfg(encoder_name: str) -> Dict[str, Any]:
    cfg_path = PANGAEA_CONFIG_ROOT / "encoder" / f"{encoder_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing encoder config for {encoder_name}: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())


def _build_remoteclip_model() -> nn.Module:
    if str(PANGAEA_CONFIG_ROOT.parent) not in sys.path:
        sys.path.append(str(PANGAEA_CONFIG_ROOT.parent))
    from pangaea.encoders.remoteclip_encoder import RemoteCLIP_Encoder

    cfg = _load_pangaea_encoder_cfg("remoteclip")
    encoder_weights = Path(cfg["encoder_weights"])
    if not encoder_weights.is_absolute():
        encoder_weights = PANGAEA_CONFIG_ROOT.parent / encoder_weights

    model = RemoteCLIP_Encoder(
        encoder_weights=str(encoder_weights),
        input_bands=cfg["input_bands"],
        input_size=cfg["input_size"],
        embed_dim=cfg["embed_dim"],
        patch_size=cfg["patch_size"],
        width=cfg["width"],
        head_width=cfg["head_width"],
        layers=cfg["layers"],
        mlp_ratio=cfg["mlp_ratio"],
        output_layers=cfg["output_layers"],
        output_dim=cfg["output_dim"],
        download_url=cfg.get("download_url", ""),
    )
    model.load_encoder_weights(logging.getLogger("remoteclip"))
    model.eval()
    return model


def _resolve_pangaea_encoder_weights(encoder_name: str) -> Tuple[Dict[str, Any], Path]:
    cfg = _load_pangaea_encoder_cfg(encoder_name)
    encoder_weights = Path(cfg["encoder_weights"])
    if not encoder_weights.is_absolute():
        encoder_weights = PANGAEA_CONFIG_ROOT.parent / encoder_weights
    return cfg, encoder_weights


def _build_ssl4eo_mae_optical_model() -> nn.Module:
    if str(PANGAEA_CONFIG_ROOT.parent) not in sys.path:
        sys.path.append(str(PANGAEA_CONFIG_ROOT.parent))
    from pangaea.encoders.ssl4eo_mae_encoder import SSL4EO_MAE_OPTICAL_Encoder

    cfg, encoder_weights = _resolve_pangaea_encoder_weights("ssl4eo_mae_optical")
    model = SSL4EO_MAE_OPTICAL_Encoder(
        encoder_weights=encoder_weights,
        input_bands=cfg["input_bands"],
        input_size=cfg["input_size"],
        output_layers=cfg["output_layers"],
        output_dim=cfg["output_dim"],
        download_url=cfg.get("download_url", ""),
        embed_dim=cfg["embed_dim"],
        patch_size=cfg["patch_size"],
        in_chans=cfg["in_chans"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
    )
    model.load_encoder_weights(logging.getLogger("ssl4eo_mae_optical"))
    model.eval()
    return model


def _build_terramind_model(encoder_name: str) -> nn.Module:
    if str(PANGAEA_CONFIG_ROOT.parent) not in sys.path:
        sys.path.append(str(PANGAEA_CONFIG_ROOT.parent))
    from pangaea.encoders import terramind_encoder

    cfg, encoder_weights = _resolve_pangaea_encoder_weights(encoder_name)
    optical_bands = cfg.get("input_bands", {}).get("optical")
    if not optical_bands:
        raise ValueError(f"{encoder_name} config must define optical input_bands.")
    target = str(cfg.get("_target_", "")).strip()
    builder_name = target.rsplit(".", 1)[-1] if target else "terramind_v1_base"
    builder = getattr(terramind_encoder, builder_name, None)
    if builder is None:
        raise ValueError(
            f"Unable to resolve TerraMind builder '{builder_name}' from _target_='{target}'."
        )
    # Keep TerraMind aligned to optical-only downstream evaluation.
    terramind_input_bands = {"optical": optical_bands}
    terramind_modalities = ["S2L2A"]
    model = builder(
        encoder_weights=encoder_weights,
        input_size=cfg["input_size"],
        input_bands=terramind_input_bands,
        output_layers=cfg["output_layers"],
        output_dim=cfg["output_dim"],
        download_url=cfg.get("download_url", ""),
        patch_size=cfg["patch_size"],
        merge_method=cfg.get("merge_method", "mean"),
        modalities=terramind_modalities,
    )
    model.eval()
    return model


def _build_terramind_base_model() -> nn.Module:
    return _build_terramind_model("terramind_base")


def _build_terramind_large_model() -> nn.Module:
    return _build_terramind_model("terramind_large")


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
        replaced_proj = None
        for attr in ("fc", "proj"):
            module = getattr(encoder, attr, None)
            if isinstance(module, nn.Module):
                replaced_modules.append((attr, module))
                setattr(encoder, attr, nn.Identity())
            elif attr == "proj" and module is not None:
                # ViT proj is a Parameter; drop it to expose pre-projection features.
                replaced_proj = module
                setattr(encoder, attr, None)
        try:
            inputs = images.type(dtype)
            if (
                hasattr(encoder, "transformer")
                and hasattr(encoder, "class_embedding")
                and hasattr(encoder, "positional_embedding")
                and hasattr(encoder, "ln_pre")
                and hasattr(encoder, "ln_post")
                and hasattr(encoder, "conv1")
            ):
                x = encoder.conv1(inputs)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                x = torch.cat(
                    [
                        encoder.class_embedding.to(x.dtype)
                        + torch.zeros(
                            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                        ),
                        x,
                    ],
                    dim=1,
                )
                x = x + encoder.positional_embedding.to(x.dtype)
                x = encoder.ln_pre(x)
                x = x.permute(1, 0, 2)
                x = encoder.transformer(x)
                x = x.permute(1, 0, 2)
                x = encoder.ln_post(x)
                x = x[:, 1:, :]
                features = x.mean(dim=1)
            else:
                features = encoder(inputs)
        finally:
            for attr, module in replaced_modules:
                setattr(encoder, attr, module)
            if replaced_proj is not None:
                setattr(encoder, "proj", replaced_proj)

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
            enable_s1: bool = True,
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
        if enable_s1 and s1_weights is not None:
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


class RemoteClipEvaluationAdapter(EvaluationAdapter):
    """Adapter for RemoteCLIP encoder (RGB B4/B3/B2, mean-pooled patch tokens)."""

    supports_ssl4eo = True
    supports_multimodal_dict = True

    def __init__(self, remoteclip_model: nn.Module) -> None:
        super().__init__()
        self.base_model = remoteclip_model.eval()
        self.encoder_s2 = remoteclip_model
        self.encoder_s1 = None

        try:
            self.dtype_s2 = next(self.encoder_s2.parameters()).dtype
        except StopIteration:
            self.dtype_s2 = torch.float32
        self.dtype_s1 = self.dtype_s2

    def prepare_inputs(self, batch: Any, *, device: torch.device, modality: str = "s2") -> Dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            tensor = batch.get("image") if "image" in batch else batch.get("data") or batch.get("optical")
        else:
            tensor = batch
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)  # (B, C, 1, H, W)
        return {"optical": tensor}

    def compute_backbone(self, images: Dict[str, torch.Tensor], modality: str = "s2") -> torch.Tensor:
        outputs = self.encoder_s2(images)
        if isinstance(outputs, (tuple, list)) and outputs:
            feats = outputs[-1]
        else:
            feats = outputs
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        elif feats.ndim == 3:
            if feats.shape[1] > 1:
                feats = feats[:, 1:, :].mean(dim=1)
            else:
                feats = feats.mean(dim=1)
        elif feats.ndim != 2:
            raise RuntimeError(f"Unexpected RemoteCLIP feature shape: {feats.shape}")
        return feats

    def compute_projected(self, images: Dict[str, torch.Tensor], modality: str = "s2") -> Optional[torch.Tensor]:
        return None


class PangaeaOpticalEvaluationAdapter(EvaluationAdapter):
    """Adapter for Pangaea optical encoders returning spatial feature maps."""

    supports_ssl4eo = True
    supports_multimodal_dict = True

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.base_model = model.eval()
        self.encoder_s2 = model
        self.encoder_s1 = None
        try:
            self.dtype_s2 = next(self.encoder_s2.parameters()).dtype
        except StopIteration:
            self.dtype_s2 = torch.float32
        self.dtype_s1 = self.dtype_s2

    def prepare_inputs(self, batch: Any, *, device: torch.device, modality: str = "s2") -> Dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            tensor = batch.get("image") if "image" in batch else batch.get("data") or batch.get("optical")
        else:
            tensor = batch
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device=device, dtype=self.dtype_s2)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)  # (B, C, 1, H, W)
        return {"optical": tensor}

    def compute_backbone(self, images: Dict[str, torch.Tensor], modality: str = "s2") -> torch.Tensor:
        outputs = self.encoder_s2(images)
        if isinstance(outputs, (tuple, list)) and outputs:
            feats = outputs[-1]
        else:
            feats = outputs
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        elif feats.ndim == 3:
            feats = feats.mean(dim=1)
        elif feats.ndim != 2:
            raise RuntimeError(f"Unexpected feature shape from Pangaea encoder: {feats.shape}")
        return feats

    def compute_projected(self, images: Dict[str, torch.Tensor], modality: str = "s2") -> Optional[torch.Tensor]:
        return None


class GalileoS2Wrapper(nn.Module):
    """Thin wrapper around Galileo encoder for Sentinel-2-only evaluation."""

    S2_BAND_ORDERING = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    ]
    S2_BAND_ORDERING_12 = [band for band in S2_BAND_ORDERING if band != "B10"]
    S2_BAND_ORDERING_10 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    def __init__(
        self,
        pretrained_path: Path,
        *,
        patch_size: int = 8,
        month: int = 6,
        do_pool: bool = True,
        add_layernorm_on_exit: bool = True,
    ) -> None:
        super().__init__()

        module = _load_galileo_module()
        Encoder = getattr(module, "Encoder")

        self.encoder = Encoder.load_from_folder(pretrained_path, device=torch.device("cpu"))
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self.dim = self.encoder.embedding_size
        self.patch_size = patch_size
        self.do_pool = do_pool
        self.month = month
        self.add_layernorm_on_exit = add_layernorm_on_exit

        self._space_time_bands = module.SPACE_TIME_BANDS
        self._space_time_groups_idx = module.SPACE_TIME_BANDS_GROUPS_IDX
        self._space_bands = module.SPACE_BANDS
        self._time_bands = module.TIME_BANDS
        self._static_bands = module.STATIC_BANDS
        self._space_band_groups_idx = module.SPACE_BAND_GROUPS_IDX
        self._time_band_groups_idx = module.TIME_BAND_GROUPS_IDX
        self._static_band_groups_idx = module.STATIC_BAND_GROUPS_IDX
        self._s2_bands = module.S2_BANDS
        self.expected_s2_channels = len(self._s2_bands)

        self._s2_group_indices = [
            idx for idx, key in enumerate(self._space_time_groups_idx) if "S2" in key
        ]

        normalizing_dict = _load_galileo_normalization_values(GALILEO_NORMALIZATION_JSON)
        stats = normalizing_dict.get(len(self._space_time_bands))
        if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
            raise ValueError(
                "Galileo normalization stats are missing space-time entries "
                f"for {len(self._space_time_bands)} channels."
            )
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["std"], dtype=torch.float32)
        if mean.numel() != len(self._space_time_bands) or std.numel() != len(self._space_time_bands):
            raise ValueError(
                "Galileo normalization stats have unexpected length: "
                f"mean={mean.numel()}, std={std.numel()}, expected={len(self._space_time_bands)}."
            )

        # Replicate Normalizer(std=True) behavior for space-time bands:
        # mean +/- 2*std for all dynamic channels except NDVI.
        shift = mean - (2.0 * std)
        div = 4.0 * std
        ndvi_idx = self._space_time_bands.index("NDVI") if "NDVI" in self._space_time_bands else None
        if ndvi_idx is not None:
            shift[ndvi_idx] = 0.0
            div[ndvi_idx] = 1.0
        div = torch.where(div == 0, torch.ones_like(div), div)
        self.register_buffer("_space_time_shift", shift, persistent=False)
        self.register_buffer("_space_time_div", div, persistent=False)

    def _resolve_s2_channel_layout(self, c_s2: int) -> tuple[list[int], list[int]]:
        if c_s2 == len(self.S2_BAND_ORDERING):
            input_bands = self.S2_BAND_ORDERING
        elif c_s2 == len(self.S2_BAND_ORDERING_12):
            input_bands = self.S2_BAND_ORDERING_12
        elif c_s2 == len(self.S2_BAND_ORDERING_10):
            input_bands = self.S2_BAND_ORDERING_10
        else:
            raise ValueError(
                "Galileo expects one of 10/12/13 Sentinel-2 channels in known order; "
                f"got {c_s2}."
            )

        source_indices: list[int] = []
        target_indices: list[int] = []
        for source_idx, band_name in enumerate(input_bands):
            if band_name in self._s2_bands:
                source_indices.append(source_idx)
                target_indices.append(self._space_time_bands.index(band_name))

        if len(source_indices) != len(self._s2_bands):
            raise ValueError(
                "Unable to map input Sentinel-2 channels to Galileo space-time bands "
                f"(mapped={len(source_indices)}, expected={len(self._s2_bands)})."
            )
        return source_indices, target_indices

    def _build_months(
        self,
        *,
        b: int,
        t: int,
        device: torch.device,
        months: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if months is None:
            month_idx = int(self.month)
            months = torch.full((b, t), fill_value=month_idx, dtype=torch.long, device=device)
        else:
            if months.ndim == 1:
                if b != 1 or months.shape[0] != t:
                    raise ValueError(
                        f"Galileo months must have shape (B, T)=({b}, {t}) or (T,) for B=1; got {tuple(months.shape)}"
                    )
                months = months.unsqueeze(0)
            if months.shape != (b, t):
                raise ValueError(
                    f"Galileo months must have shape (B, T)=({b}, {t}); got {tuple(months.shape)}."
                )
            months = months.to(device=device, dtype=torch.long)

        if torch.any(months < 0) or torch.any(months > 11):
            raise ValueError("Galileo months must be 0-indexed and within [0, 11].")
        return months

    def _normalize_space_time(self, s_t_x: torch.Tensor) -> torch.Tensor:
        shift = self._space_time_shift.to(device=s_t_x.device, dtype=s_t_x.dtype).view(1, 1, 1, 1, -1)
        div = self._space_time_div.to(device=s_t_x.device, dtype=s_t_x.dtype).view(1, 1, 1, 1, -1)
        return (s_t_x - shift) / div

    def _preprocess_s2(self, s2: torch.Tensor, months: Optional[torch.Tensor] = None):
        if s2.ndim == 4:
            b, h, w, c_s2 = s2.shape
            t = 1
        elif s2.ndim == 5:
            b, h, w, t, c_s2 = s2.shape
        else:
            raise ValueError(f"Expected Galileo input with 4 or 5 dims, got {s2.shape}")

        source_indices, target_indices = self._resolve_s2_channel_layout(c_s2)

        s_t_x = torch.zeros(
            (b, h, w, t, len(self._space_time_bands)),
            dtype=s2.dtype,
            device=s2.device,
        )
        if s2.ndim == 4:
            s_t_x[:, :, :, 0, target_indices] = s2[:, :, :, source_indices]
        else:
            s_t_x[:, :, :, :, target_indices] = s2[:, :, :, :, source_indices]
        s_t_x = self._normalize_space_time(s_t_x)

        s_t_m = torch.ones(
            (b, h, w, t, len(self._space_time_groups_idx)),
            dtype=s2.dtype,
            device=s2.device,
        )
        s_t_m[:, :, :, :, self._s2_group_indices] = 0

        months = self._build_months(b=b, t=t, device=s2.device, months=months)

        return (
            s_t_x,
            torch.zeros((b, h, w, len(self._space_bands)), dtype=s2.dtype, device=s2.device),
            torch.zeros((b, t, len(self._time_bands)), dtype=s2.dtype, device=s2.device),
            torch.zeros((b, len(self._static_bands)), dtype=s2.dtype, device=s2.device),
            s_t_m,
            torch.ones((b, h, w, len(self._space_band_groups_idx)), dtype=s2.dtype, device=s2.device),
            torch.ones((b, t, len(self._time_band_groups_idx)), dtype=s2.dtype, device=s2.device),
            torch.ones((b, len(self._static_band_groups_idx)), dtype=s2.dtype, device=s2.device),
            months,
        )

    def forward(self, s2: torch.Tensor, months: Optional[torch.Tensor] = None) -> torch.Tensor:
        (
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
            month,
        ) = self._preprocess_s2(s2, months=months)

        output = self.encoder(
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
            month,
            patch_size=self.patch_size,
            add_layernorm_on_exit=self.add_layernorm_on_exit,
        )
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = output
        if self.do_pool:
            return self.encoder.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        s2_tokens = s_t_x[:, :, :, :, self._s2_group_indices, :].mean(dim=3)
        s2_tokens = s2_tokens.reshape(s2_tokens.shape[0], -1, s2_tokens.shape[-1])
        return s2_tokens.mean(dim=1)


class GalileoEvaluationAdapter(EvaluationAdapter):
    """Adapter around the Galileo base encoder for Sentinel-2."""

    supports_ssl4eo = True
    supports_temporal_months = True

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.base_model = model.eval()
        self.encoder_s2 = model
        self.encoder_s1 = None
        self.dtype_s2 = torch.float32
        self.dtype_s1 = torch.float32
        self.expected_s2_channels = getattr(model, "expected_s2_channels", None)

    def prepare_inputs(self, batch: Any, *, device: torch.device, modality: str = "s2") -> Any:
        if modality.lower() != "s2":
            raise ValueError("Galileo adapter only supports Sentinel-2 inputs.")

        months: Optional[torch.Tensor] = None
        tensor_input = batch
        if isinstance(batch, dict):
            tensor_input = batch.get("pixels", batch.get("image", batch.get("data")))
            months = batch.get("months")
            if tensor_input is None:
                raise ValueError("Galileo adapter expected 'pixels'/'image'/'data' in input dict.")

        tensor = super().prepare_inputs(tensor_input, device=device, modality=modality)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            tensor = tensor.permute(0, 2, 3, 1).unsqueeze(3)
        elif isinstance(tensor, torch.Tensor) and tensor.ndim == 5:
            tensor = tensor.permute(0, 3, 4, 1, 2)
        if months is not None:
            months = months.to(device=device, dtype=torch.long, non_blocking=True)
            return {"s2": tensor, "months": months}
        return tensor

    def compute_backbone(self, images: Any, modality: str = "s2") -> torch.Tensor:
        if modality.lower() != "s2":
            raise ValueError("Galileo adapter only supports Sentinel-2 inputs.")

        months: Optional[torch.Tensor] = None
        tensor = images
        if isinstance(images, dict):
            tensor = images.get("s2")
            months = images.get("months")
            if tensor is None:
                raise ValueError("Galileo adapter expected 's2' tensor in prepared inputs.")

        if tensor.ndim != 5:
            raise ValueError(f"Galileo adapter expects [B, H, W, T, C] inputs, got {tuple(tensor.shape)}.")
        b, _, _, t, _ = tensor.shape
        if months is None:
            month_idx = int(getattr(self.encoder_s2, "month", 6))
            months = torch.full((b, t), fill_value=month_idx, dtype=torch.long, device=tensor.device)
        features = self.encoder_s2(tensor, months=months)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return features

    def compute_embeddings(
        self, images: Any, modality: str = "s2"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        backbone = self.compute_backbone(images, modality=modality)
        return backbone, None, None


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
    ciip_model_source: Optional[str] = "current",
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
    logging.info("Embedding dims: pre-projection=%d, projected=%d", pre_dim, embed_dim)
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

    ciip_cls, lorentz_cls = _load_ciip_classes(ciip_model_source)
    if is_lorentz:
        model = lorentz_cls(**kwargs)
    else:
        model = ciip_cls(**kwargs)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if "encoder_s2.positional_embedding" in missing:
        raise RuntimeError(
            "Failed to load CIIP checkpoint: missing required key "
            "'encoder_s2.positional_embedding'. "
            "This usually indicates a model architecture/source mismatch."
        )
    if missing or unexpected:
        logging.warning("Checkpoint loaded with missing=%s, unexpected=%s", missing, unexpected)

    model.eval()
    return model, is_lorentz


def _resolve_resnet_weights(
    name: Optional[str],
) -> Tuple[Optional[ResNet50_Weights], Optional[ResNet50_Weights]]:
    if name is None:
        return None, None
    normalized = name.lower()
    if normalized == "dino":
        return None, ResNet50_Weights.SENTINEL2_ALL_DINO
    if normalized == "moco":
        # Use only the Sentinel-2 MOCO weights.
        return None, ResNet50_Weights.SENTINEL2_ALL_MOCO
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

    if hasattr(model, "visual") and hasattr(model.visual, "proj"):
        model.visual.proj = None
    if hasattr(model, "visual") and hasattr(model.visual, "pool_type"):
        model.visual.pool_type = "avg"
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
    elif normalized == "vitsmall16_s2_all_dino":
        model = vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO)
    elif normalized == "llama3_ms_clip_base":
        return _build_llama3_ms_clip_adapter()
    elif normalized == "remoteclip":
        return RemoteClipEvaluationAdapter(_build_remoteclip_model())
    elif normalized == "ssl4eo_mae_optical":
        return PangaeaOpticalEvaluationAdapter(_build_ssl4eo_mae_optical_model())
    elif normalized == "terramind_base":
        return PangaeaOpticalEvaluationAdapter(_build_terramind_base_model())
    elif normalized == "terramind_large":
        return PangaeaOpticalEvaluationAdapter(_build_terramind_large_model())
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
    ciip_model_source: str = "current",
    galileo_weights: Optional[Path] = None,
    enable_s1: bool = True,
) -> EvaluationAdapter:
    """Create an evaluation adapter based on the requested model type."""

    if model_type == "ciip_checkpoint":
        if checkpoint is None:
            raise ValueError("checkpoint must be provided when using 'ciip_checkpoint'")
        model, is_lorentz = build_model_from_checkpoint(
            checkpoint,
            framework=ciip_framework,
            ciip_model_source=ciip_model_source,
        )
        return CiipEvaluationAdapter(model, is_lorentz=is_lorentz)

    if model_type == "torchgeo_resnet50":
        weights = _resolve_resnet_weights(model_weights)
        return TorchGeoResNetAdapter(weights=weights, in_chans=in_chans, enable_s1=enable_s1)

    if model_type == "backbone_only":
        return _build_backbone_adapter(model_weights)

    if model_type in {"galileo_s2", "galileo"}:
        weights_path = galileo_weights or GALILEO_DEFAULT_WEIGHTS
        if not weights_path.exists():
            raise FileNotFoundError(f"Galileo weights not found at {weights_path}.")
        model = GalileoS2Wrapper(weights_path)
        model.eval()
        model.requires_grad_(False)
        return GalileoEvaluationAdapter(model)

    if model_type == "croma":
        if croma_weights is None:
            raise ValueError("'croma_weights' must be provided when model_type='croma'.")
        return CromaEvaluationAdapter(
            weights_path=croma_weights,
            image_resolution=int(croma_image_resolution),
        )

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        "Valid options are 'ciip_checkpoint', 'torchgeo_resnet50', 'backbone_only', 'galileo_s2', 'galileo' and 'croma'."
    )


__all__ = [
    "EvaluationAdapter",
    "CiipEvaluationAdapter",
    "TorchGeoResNetAdapter",
    "BackboneOnlyAdapter",
    "OpenClipVisionAdapter",
    "RemoteClipEvaluationAdapter",
    "PangaeaOpticalEvaluationAdapter",
    "GalileoEvaluationAdapter",
    "CromaEvaluationAdapter",
    "build_model_from_checkpoint",
    "build_evaluation_adapter",
]
