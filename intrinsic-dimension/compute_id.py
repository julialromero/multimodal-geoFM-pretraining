from __future__ import annotations

import contextlib
import logging
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from visualizations.ssl4eo.embedding_collapse_diagnostics import compute_singular_values

import skdim.id as id
import torchvision
from torchvision import transforms as T
from s2geo_dataset import S2Geo
S2_100K_ROOT = Path("/local/ms-data/S2_100K")
from ciip.evaluation.model_utils import build_model_from_checkpoint
from ciip.evaluation.normalization_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODALITY_STATS,
    S2ScaleTransform,
    SelectS2Channels10,
)
from ciip.evaluation.output_utils import write_run_manifest

ssl4eo_s2a_norm_transform = T.Normalize(
    mean=MODALITY_STATS["s2l2a"][0], std=MODALITY_STATS["s2l2a"][1]
)
ssl4eo_s2c_norm_transform = T.Normalize(
    mean=MODALITY_STATS["s2l1c"][0], std=MODALITY_STATS["s2l1c"][1]
)
ssl4eo_rgb_norm_transform = T.Normalize(
    mean=MODALITY_STATS["s2l2a_rgb"][0], std=MODALITY_STATS["s2l2a_rgb"][1]
)


# ---------------------------------------------------------------------------
# TorchGeo / Torch models
# ---------------------------------------------------------------------------
from torchvision.models import resnet152, ResNet152_Weights
from torchgeo.models import (
    # ResNets + weights
    resnet18,
    resnet50,
    
    ResNet18_Weights,
    ResNet50_Weights,
    # ViT-Small multispectral
    vit_small_patch16_224,
    ViTSmall16_Weights,
    # CROMA
    croma_base,
    CROMABase_Weights,
    # DOFA
    dofa_base_patch16_224,
    DOFABase16_Weights,
    # ScaleMAE
    scalemae_large_patch16,
    ScaleMAELarge16_Weights,
)

# ---------------------------------------------------------------------------
# Paths & config (you should swap EuroSAT out for S2-100K later)
# ---------------------------------------------------------------------------
# EUROSAT_ROOT = Path("/local/ms-data/EuroSAT/")
# NEUCO_ROOT = Path("/local/ms-data/SSL4EO-S12-downstream/data")
OUTPUT_DIR = Path("/home/juro4948/ciip/diagnostics/global_id_table1")
PANGAEA_CONFIG_ROOT = Path("/local/ms-data/pangaea-bench/configs")
PANGAEA_WEIGHTS_ROOT = Path("/local/ms-data/pangaea-bench/pretrained_models")

# EUROSAT_IMAGE_SIZE = 224  # CROMA expects 120, we’ll handle that per-model
# EUROSAT_MODALITY = "s2"   # used only by your loader helper

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


from ciip.open_clip_train import data
from ciip.open_clip_train.data import SSL4EODataset
import yaml
def build_s2_ssl4eo_loader(
    in_chans: int,
    batch_size: int = 64,
    num_workers: int = 4,
    # rgb_mode: bool = False,
    return_all_timestamps: bool = False,
) -> DataLoader:
    # if s2_transform is None:
    tier = 's2l1c' if in_chans == 13 else 's2l2a'
        # s2_transform = data.get_transform(tier, is_train=False)

    dataset = SSL4EODataset(
        root=Path("/local/ms-data/SSL4EOv1.1/train"),
        s2_tier=tier,
        seasons=[0,1,2,3],
        num_timestamps=4,
        return_all_timestamps=return_all_timestamps,
        # s2_bands=list(config.ssl4eo_s2_bands),
        transforms=None,
        is_train=False,
    )

    # wrap to return dicts with "image" and "label" keys
    class _SSL4EOWrapped(torch.utils.data.Dataset):
        def __init__(self, base,  in_chans, rgb_mode):
            self.base = base
            self.in_chans = in_chans
            self.rgb_mode = rgb_mode
            # self.resize = T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC)

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            item = self.base[idx]
            x = item["s2"] if isinstance(item, dict) and "s2" in item else item[0]
            y = item.get("label", 0) if isinstance(item, dict) else (item[1] if len(item) > 1 else 0)

            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            elif not isinstance(x, torch.Tensor):
                x = T.ToTensor()(x)  

            if x.ndim == 3 and x.shape[0] not in (3, 12, 13):
                x = x.permute(2, 0, 1).contiguous()
            elif x.ndim == 5:
                # (P, T, C, H, W) -> keep shape; channel selection handled below
                pass

            if self.rgb_mode:
                # print("Before RGB selection:", x.shape)
                if x.ndim == 3:
                    # [C, H, W]
                    if x.shape[0] >= 4:
                        x = x[[3, 2, 1], ...]
                elif x.ndim == 4:
                    # [P, C, H, W]
                    if x.shape[1] >= 4:
                        x = x[:, [3, 2, 1], ...]
                elif x.ndim == 5:
                    # [P, T, C, H, W]
                    if x.shape[2] >= 4:
                        x = x[:, :, [3, 2, 1], ...]
                # print("After RGB selection:", x.shape)

            return {"image": x.float(), "label": int(y)}
    
    ds = _SSL4EOWrapped(dataset, in_chans=in_chans, rgb_mode=(in_chans==3))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=ssl4eo_collate,
    )
    return loader
    
    

def build_s2_100k_loader(
    in_chans: int,
    batch_size: int = 64,
    num_workers: int = 4,
    # rgb_mode: bool = False,
) -> DataLoader:
    """
    Build a DataLoader over SatCLIP's S2-100K using its S2GeoDataset class.
    Yields dicts with keys {"image", "label"} to match the rest of the script.
    """

    rgb_mode = (in_chans == 3)

    # 1) Base SatCLIP dataset (change split as you like: "train", "val", "test")
    base = S2Geo(
        root=S2_100K_ROOT,
        # split="train",          # or "val"/"test" if available in your dump
        # download=False,        
        transform=None,  
        mode="both",     
    )

    # 2) Wrap to format samples like your EuroSAT loader did
    class _S2GeoWrapped(torch.utils.data.Dataset):
        def __init__(self, base,  in_chans, rgb_mode):
            self.base = base
            self.in_chans = in_chans
            self.rgb_mode = rgb_mode
            # self.resize = T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC)

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            item = self.base[idx]
            # SatCLIP typically returns dicts; fall back just in case
            x = item["image"] if isinstance(item, dict) and "image" in item else item[0]
            y = item.get("label", 0) if isinstance(item, dict) else (item[1] if len(item) > 1 else 0)

            # Convert to float tensor, handle HWC/CHW
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            elif not isinstance(x, torch.Tensor):
                # If it's a PIL Image
                x = T.ToTensor()(x)  # -> [C,H,W] in [0,1]

            if x.ndim == 3 and x.shape[0] not in (3, 12, 13):
                x = x.permute(2, 0, 1).contiguous()

            # Select channels to match the model spec
            if self.rgb_mode:
                # S2 RGB convention: B4,B3,B2 -> indices 3,2,1 in [B1..B12]; if 13 bands and includes B8A, adjust accordingly
                # Safe fallback: if ≥4 chans, pick [3,2,1]; else assume already RGB
                if x.shape[0] >= 4:
                    x = x[[3, 2, 1], ...]

            return {"image": x.float(), "label": int(y)}

    ds = _S2GeoWrapped(base, in_chans=in_chans, rgb_mode=rgb_mode)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader



# ---------------------------------------------------------------------------
# Simple random-conv RCF baseline
# ---------------------------------------------------------------------------
class RandomConvFeatures(nn.Module):
    """Random Convolutional Filters baseline (13-channel, 512-dim GAP)."""

    def __init__(self, in_chans: int = 13, out_dim: int = 512):
        super().__init__()

        torch.manual_seed(42)
        
        # Very simple stack; weights are left random and frozen.
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        for p in self.parameters():
            p.requires_grad = False
        self.out_dim = out_dim
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply transform
        # x: (B, C, H, W) → features: (B, D)
        h = self.conv(x)
        return h.mean(dim=(2, 3))  # global average pooling

### CROMA:
import importlib.util

_CROMA_MODULE: Optional[ModuleType] = None
from types import ModuleType
def _load_croma_module() -> ModuleType:
    """Dynamically import the CROMA helper without requiring package installation."""

    global _CROMA_MODULE
    if _CROMA_MODULE is not None:
        return _CROMA_MODULE

    # print( Path(__file__).resolve().parents[0])
    print( Path(__file__).resolve().parents[1])
    module_path = Path(__file__).resolve().parents[1] / "comparison" / "CROMA-main" / "use_croma.py"
    
    print(module_path)
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

def _load_pangaea_encoder_cfg(encoder_name: str) -> Dict:
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

def _load_llama3_ms_clip_model() -> nn.Module:
    try:
        import open_clip  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "open_clip is required to load the llama3_ms_clip_base preset."
        ) from exc

    weights_path = Path("/home/juro4948/ciip/ciip/Llama3_MS_CLIP_weights.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"MS-CLIP weights not found at {weights_path}")

    model_cfg = open_clip.get_model_config("ViT-B-16")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained=None,
        text_cfg=model_cfg.get("text_cfg"),
    )

    desired_in_chans = 10
    conv_key = "visual.conv1.weight"
    old_conv = model.visual.conv1
    model.visual.conv1 = nn.Conv2d(
        in_channels=desired_in_chans,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        bias=False,
    )

    raw_state = torch.load(weights_path, map_location="cpu")
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]

    prefixes = (
        "clip_base_model.model.",
        "image_encoder.model.",
        "text_encoder.model.",
    )

    target_state = model.state_dict()
    filtered = {}
    for key, tensor in raw_state.items():
        trimmed = key
        for prefix in prefixes:
            if trimmed.startswith(prefix):
                trimmed = trimmed[len(prefix):]
                break
        if trimmed in target_state and target_state[trimmed].shape == tensor.shape:
            filtered[trimmed] = tensor

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

    model.load_state_dict(filtered, strict=False)
    if hasattr(model, "visual") and hasattr(model.visual, "proj"):
        model.visual.proj = None
    if hasattr(model, "visual") and hasattr(model.visual, "pool_type"):
        model.visual.pool_type = "avg"
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Model spec / registry
# ---------------------------------------------------------------------------
FeatureFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]


@dataclass
class ModelSpec:
    name: str
    in_chans: int
    rgb_mode: bool  # True means expect 3 channels (B4/B3/B2)
    builder: Callable[[], nn.Module]
    feature_fn: FeatureFn
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    return_all_timestamps_ssl4eo: bool = False
    checkpoint_path: Optional[str] = None
    eval_dims: Optional[List[int]] = None


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _callable_descriptor(fn: Callable[..., object]) -> Dict[str, object]:
    name = getattr(fn, "__name__", fn.__class__.__name__)
    payload: Dict[str, object] = {"name": name}
    code = getattr(fn, "__code__", None)
    if code is not None:
        payload["file"] = code.co_filename
    return payload


def _model_spec_payload(spec: ModelSpec, dataset_name: str) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "name": spec.name,
        "in_chans": spec.in_chans,
        "rgb_mode": spec.rgb_mode,
        "builder": _callable_descriptor(spec.builder),
        "feature_fn": _callable_descriptor(spec.feature_fn),
        "transform": repr(spec.transform) if spec.transform is not None else None,
        "checkpoint_path": spec.checkpoint_path,
    }
    if spec.eval_dims:
        payload["eval_dims"] = list(spec.eval_dims)
    if dataset_name == "SSL4EO":
        payload["return_all_timestamps_ssl4eo"] = spec.return_all_timestamps_ssl4eo
        payload["temporal_aggregation"] = (
            "post-enc temporal mean"
            if spec.return_all_timestamps_ssl4eo
            else "single random season"
        )
    return payload


def _model_results_path(output_dir: Path, spec: ModelSpec) -> Path:
    slug = _FILENAME_SAFE_RE.sub("_", spec.name).strip("_")
    if not slug:
        slug = "model"
    return output_dir / f"{slug.lower()}.json"


def _lowercase_strings(obj: object) -> object:
    if isinstance(obj, str):
        return obj.lower()
    if isinstance(obj, list):
        return [_lowercase_strings(item) for item in obj]
    if isinstance(obj, dict):
        lowered: Dict[object, object] = {}
        for key, value in obj.items():
            lowered_key = key.lower() if isinstance(key, str) else key
            lowered[lowered_key] = _lowercase_strings(value)
        return lowered
    return obj


def _is_ciip_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    model_spec = entry.get("model_spec") or {}
    checkpoint = model_spec.get("checkpoint_path") or entry.get("checkpoint_path")
    return bool(checkpoint)


def _is_ciip_payload(payload: object) -> bool:
    if isinstance(payload, list):
        return any(_is_ciip_entry(item) for item in payload)
    if isinstance(payload, dict):
        if _is_ciip_entry(payload):
            return True
        entries = payload.get("entries")
        if isinstance(entries, list):
            return any(_is_ciip_entry(item) for item in entries)
    return False


def _append_json_entry(path: Path, entry: Dict[str, object]) -> None:
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            payload = []
    else:
        payload = []

    should_lowercase = not (_is_ciip_entry(entry) or _is_ciip_payload(payload))
    if should_lowercase:
        entry = _lowercase_strings(entry)

    def _entry_match(existing: Dict[str, object], incoming: Dict[str, object]) -> bool:
        existing_dataset = existing.get("dataset")
        incoming_dataset = incoming.get("dataset")
        if isinstance(existing_dataset, str) and isinstance(incoming_dataset, str):
            if existing_dataset.lower() != incoming_dataset.lower():
                return False
        elif existing_dataset != incoming_dataset:
            return False

        existing_dim = existing.get("embedding_dim")
        incoming_dim = incoming.get("embedding_dim")
        if existing_dim is None or incoming_dim is None:
            return False
        try:
            return int(existing_dim) == int(incoming_dim)
        except (TypeError, ValueError):
            return False

    def _merge_id_results(
        existing_results: Dict[str, object],
        incoming_results: Dict[str, object],
    ) -> None:
        for dim_key, incoming_dim in incoming_results.items():
            if not isinstance(incoming_dim, dict):
                continue
            existing_dim = existing_results.get(dim_key)
            if not isinstance(existing_dim, dict):
                existing_results[dim_key] = incoming_dim
                continue
            for dataset_key, incoming_metrics in incoming_dim.items():
                if not isinstance(incoming_metrics, dict):
                    existing_dim[dataset_key] = incoming_metrics
                    continue
                existing_metrics = existing_dim.get(dataset_key)
                if isinstance(existing_metrics, dict):
                    existing_metrics.update(incoming_metrics)
                else:
                    existing_dim[dataset_key] = incoming_metrics

    def _merge_entry(existing: Dict[str, object], incoming: Dict[str, object]) -> None:
        def _safe_int(value: object) -> Optional[int]:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return int(value)
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    try:
                        return int(float(value))
                    except ValueError:
                        return None
            return None

        incoming_metrics = incoming.get("metrics")
        if isinstance(incoming_metrics, dict):
            existing_metrics = existing.get("metrics")
            if isinstance(existing_metrics, dict):
                existing_metrics.update(incoming_metrics)
            else:
                existing["metrics"] = incoming_metrics

        incoming_id_results = incoming.get("id_results")
        if isinstance(incoming_id_results, dict):
            existing_id_results = existing.get("id_results")
            if isinstance(existing_id_results, dict):
                _merge_id_results(existing_id_results, incoming_id_results)
            else:
                existing["id_results"] = incoming_id_results

        incoming_dims = incoming.get("dims_for_eval")
        if isinstance(incoming_dims, list):
            existing_dims = existing.get("dims_for_eval")
            if isinstance(existing_dims, list):
                existing_dim_values = {value for value in (_safe_int(d) for d in existing_dims) if value is not None}
                incoming_dim_values = {value for value in (_safe_int(d) for d in incoming_dims) if value is not None}
                merged_dims = sorted(existing_dim_values | incoming_dim_values)
                existing["dims_for_eval"] = merged_dims if merged_dims else existing_dims
            else:
                existing["dims_for_eval"] = incoming_dims

        if "model_spec" not in existing and "model_spec" in incoming:
            existing["model_spec"] = incoming["model_spec"]

    def _write_payload(payload_obj: object) -> None:
        if should_lowercase:
            payload_obj = _lowercase_strings(payload_obj)
        path.write_text(json.dumps(payload_obj, indent=2))

    if isinstance(payload, list):
        for existing in payload:
            if isinstance(existing, dict) and _entry_match(existing, entry):
                _merge_entry(existing, entry)
                _write_payload(payload)
                return
        payload.append(entry)
        _write_payload(payload)
        return

    if isinstance(payload, dict):
        if _entry_match(payload, entry):
            _merge_entry(payload, entry)
            _write_payload(payload)
            return
        if "entries" in payload and isinstance(payload["entries"], list):
            for existing in payload["entries"]:
                if isinstance(existing, dict) and _entry_match(existing, entry):
                    _merge_entry(existing, entry)
                    _write_payload(payload)
                    return
            payload["entries"].append(entry)
            _write_payload(payload)
        else:
            combined = [payload, entry]
            _write_payload(combined)
        return

    _write_payload([entry])


def _ciip_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """CIIP: return backbone features; ViT uses mean-pooled patch tokens (no CLS)."""
    core = model.module if hasattr(model, "module") else model
    encoder = getattr(core, "encoder_s2", None)
    if encoder is None:
        raise AttributeError("CIIP model does not expose encoder_s2.")

    replaced_modules: List[Tuple[str, nn.Module]] = []
    replaced_proj = None
    for attr in ("fc", "proj"):
        module = getattr(encoder, attr, None)
        if isinstance(module, nn.Module):
            replaced_modules.append((attr, module))
            setattr(encoder, attr, nn.Identity())
        elif attr == "proj" and module is not None:
            replaced_proj = module
            setattr(encoder, attr, None)

    try:
        dtype = encoder.conv1.weight.dtype if hasattr(encoder, "conv1") else x.dtype
        inputs = x.type(dtype)
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
                    + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
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
            feats = x.mean(dim=1)
        else:
            feats = encoder(inputs)
    finally:
        for attr, module in replaced_modules:
            setattr(encoder, attr, module)
        if replaced_proj is not None:
            setattr(encoder, "proj", replaced_proj)

    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.ndim == 4:
        feats = feats.mean(dim=(2, 3))
    elif feats.ndim == 3:
        feats = feats.mean(dim=1)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected CIIP feature shape: {feats.shape}")
    return feats


def _resnet_gap_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract global-average-pooled penultimate features from a ResNet.

    This is designed to work with TorchGeo ResNet wrappers, including
    Sentinel-2 MoCo variants, and with plain torchvision ResNet models.

    Returns:
        feats: Tensor of shape (B, D), where D is the penultimate feature dim.
    """

    # 1. If the model has an explicit backbone (common in TorchGeo),
    #    use that to avoid the classification head entirely.
    backbone = None
    for attr in ("backbone", "features", "encoder"):
        if hasattr(model, attr):
            backbone = getattr(model, attr)
            print(f"resnet Using backbone attribute '{attr}'")
            break

    if backbone is not None:
        feats = backbone(x)
    elif hasattr(model, "forward_features"):
        # print("resnet Using forward_features")
        # 2. Some timm-style or TorchGeo wrappers expose forward_features
        feats = model.forward_features(x)
    else:
        # Fallback for torchvision-style ResNet
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)        # (B, C, 1, 1)
        feats = torch.flatten(x, 1)  # (B, C)

    # Handle various output layouts:
    if isinstance(feats, dict):
        # Defensive: if backbone/forward_features returns a dict,
        # try common keys. Adjust if your model uses a different one.
        for key in ("out", "feat", "features", "x"):
            if key in feats and isinstance(feats[key], torch.Tensor):
                feats = feats[key]
                break
        else:
            raise RuntimeError(f"Unexpected dict from ResNet backbone: {feats.keys()}")

    # If we still have a spatial map, do global average pooling.
    if feats.ndim == 4:
        # (B, C, H, W) -> (B, C)
        feats = feats.mean(dim=(2, 3))
    elif feats.ndim == 3:
        # (B, N, D) – very unusual for a ResNet, but pool over N if it happens.
        feats = feats.mean(dim=1)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected ResNet feature shape: {feats.shape}")

    return feats  # (B, D)


def _vit_patch_mean_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ViT-Small: mean-pool patch tokens (exclude CLS)."""
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        if feats.ndim == 3:
            # Assume feats includes CLS as first token
            cls, patches = feats[:, :1, :], feats[:, 1:, :]
            return patches.mean(dim=1)  # (B, D)
        elif feats.ndim == 2:
            return feats
        else:
            raise RuntimeError(f"Unexpected ViT feature shape: {feats.shape}")
    else:
        raise RuntimeError("Model has no forward_features method.")

def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model

def _remoteclip_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    core = _unwrap_model(model)
    if x.ndim == 4:
        optical = x.unsqueeze(2)
    elif x.ndim == 5:
        optical = x
    else:
        raise RuntimeError(f"Unexpected RemoteCLIP input shape: {x.shape}")
    outputs = core({"optical": optical})
    if isinstance(outputs, (list, tuple)) and outputs:
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

def _open_clip_vit_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    core = _unwrap_model(model)
    visual = getattr(core, "visual", core)
    proj = getattr(visual, "proj", None)
    pool_type = getattr(visual, "pool_type", None)
    if proj is not None:
        visual.proj = None
    if pool_type is not None:
        visual.pool_type = "avg"
    try:
        if hasattr(visual, "forward_features"):
            feats = visual.forward_features(x)
        else:
            feats = visual(x)
    finally:
        if proj is not None:
            visual.proj = proj
        if pool_type is not None:
            visual.pool_type = pool_type
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.ndim == 3:
        feats = feats[:, 1:, :].mean(dim=1)
    elif feats.ndim > 2:
        feats = feats.mean(dim=tuple(range(2, feats.ndim)))
    if feats.ndim != 2:
        raise RuntimeError(f"Unexpected MS-CLIP feature shape: {feats.shape}")
    return feats

class CromaNormalize(nn.Module):
    def __init__(self, use_8_bit: bool = False):
        super().__init__()
        self.use_8_bit = use_8_bit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # taken from SatMAE and SeCo
        x = x.float()

        imgs = []
        for channel in range(x.shape[1]):
            min_value = x[:, channel, :, :].mean() - 2 * x[:, channel, :, :].std()
            max_value = x[:, channel, :, :].mean() + 2 * x[:, channel, :, :].std()

            if self.use_8_bit:
                img = (x[:, channel, :, :] - min_value) / (max_value - min_value) * 255.0
                img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)
                imgs.append(img)
            else:
                img = (x[:, channel, :, :] - min_value) / (max_value - min_value)
                img = torch.clip(img, 0, 1).unsqueeze(dim=1)
                imgs.append(img)

        return torch.cat(imgs, dim=1)
    
def _croma_optical_gap_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """CROMA: use optical branch GAP embedding."""
    out: Dict[str, torch.Tensor] = model(optical_images=x)
    # if "optical_GAP" not in out:
    #     raise RuntimeError(f"CROMA output has no optical_GAP key: {out.keys()}")
    return out["optical_GAP"]  # (B, D)



def _dofa_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:

    # """DOFA: use forward_features with Sentinel-2 wavelengths and mean-pool tokens."""
    # x: (B, 13, H, W)
    feats = model.forward_features(x, wavelengths=S2_WAVELENGTHS_UM)
    # If DOFA returns patch tokens (B, N, D), mean-pool over N:
    if feats.ndim == 3:
        feats = feats.mean(dim=1)  # (B, D)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected DOFA feature shape: {feats.shape}")
    return feats

def _scalemae_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ScaleMAE: use forward_features and mean-pool tokens to (B, D)."""
    # x: (B, 3, 224, 224), already normalized appropriately for ScaleMAE

    if not hasattr(model, "forward_features"):
        raise RuntimeError("ScaleMAE model has no forward_features method.")


    feats = model.forward_features(x)
 

    if isinstance(feats, dict):
        # Try some likely keys; adjust if your version uses a different key.
        for key in ("x", "tokens", "sequence", "feat"):
            if key in feats and isinstance(feats[key], torch.Tensor):
                feats = feats[key]
                break
        else:
            raise RuntimeError(f"Unexpected dict from ScaleMAE.forward_features: {feats.keys()}")

    if feats.ndim == 3:
        # (B, N, D) → mean over tokens
        feats = feats.mean(dim=1)  # (B, D)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected ScaleMAE feature shape: {feats.shape}")

    return feats  # (B, D)


class SelectRGB(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] >= 4:
                return x[:, [3, 2, 1], ...]
        elif x.ndim == 3:
            if x.shape[0] >= 4:
                return x[[3, 2, 1], ...]
        return x


# Registry of models to evaluate
def build_model_specs() -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # 1. Random Conv Filters (RCF) – 13-band, 512-dim features
    specs.append(
        ModelSpec(
            name="RCF_13ch",
            in_chans=13,
            rgb_mode=False,
            builder=lambda: RandomConvFeatures(in_chans=13, out_dim=512),
            transform=T.Compose([
                    T.CenterCrop((224, 224)),
                    S2ScaleTransform(scale=10000.0),
                ]),
            feature_fn=lambda m, x: m(x),
        )
    )

    # 2. CROMA optical encoder – 12 optical bands, 120x120, [0,1]
    module = _load_croma_module()
    PretrainedCROMA = getattr(module, "PretrainedCROMA")
    specs.append(
        ModelSpec(
            name="CROMA_optical",
            in_chans=12,
            rgb_mode=False,
            builder=lambda: PretrainedCROMA(
                pretrained_path='/home/juro4948/ciip/comparison/CROMA-main/CROMA_base.pt',
                size="base",
                modality="optical",
                image_resolution=120,
            ),
            # croma_base(
            #     weights=CROMABase_Weights.CROMA_VIT, modalities=['optical'],  # TODO: confirm exact enum
            # ),
            transform = nn.Sequential(
                T.Resize((120, 120)),
                CromaNormalize(use_8_bit=False),  # or True if you want the 8-bit variant
            ),
            feature_fn=_croma_optical_gap_feature_fn,
        )
    )

    

    # 3. DOFA base – 13 bands, use pre-trained base weights
    specs.append(
        ModelSpec(
            name="DOFA_base_S2_13ch",
            in_chans=13,
            rgb_mode=False,
            builder=lambda: dofa_base_patch16_224(
                weights=DOFABase16_Weights.DOFA_MAE,
            ),
            transform = T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0),
                # no extra normalization in the ID pipeline
            ]),
            feature_fn=_dofa_feature_fn,
        )
    )

    # # # 4. ScaleMAE Large (RGB)
    specs.append(
        ModelSpec(
            name="ScaleMAE_large_RGB",
            in_chans=3,
            rgb_mode=True,
            builder=lambda: scalemae_large_patch16(
                weights=ScaleMAELarge16_Weights.FMOW_RGB
            ),
            transform = T.Compose([
                T.CenterCrop((224, 224)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]),
            feature_fn=_scalemae_feature_fn,
        )
    )

    # 5–8. ResNet baselines (MoCo S2 ALL / RGB + ImageNet)
    specs.append(
        ModelSpec(
            name="ResNet18_S2_ALL_MOCO",
            in_chans=13,
            rgb_mode=False,
            builder=lambda: resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO),
            transform = T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0),
                # no explicit normalization in the ID experiment
            ]),
            feature_fn=_resnet_gap_feature_fn
        )
    )
    specs.append(
        ModelSpec(
            name="ResNet50_S2_ALL_MOCO",
            in_chans=13,
            rgb_mode=False,
            builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO),
            transform = T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0),
                # no explicit normalization in the ID experiment
            ]),
            feature_fn=_resnet_gap_feature_fn,
        )
    )
    specs.append(
        ModelSpec(
            name="ResNet50_S2_ALL_DINO",
            in_chans=13,
            rgb_mode=False,
            builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_DINO),
            transform = T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0),
                # no explicit normalization in the ID experiment
            ]),
            feature_fn=_resnet_gap_feature_fn,
        )
    )
    specs.append(
        ModelSpec(
            name="ResNet18_S2_RGB_MOCO",
            in_chans=3,
            rgb_mode=True,
            builder=lambda: resnet18(weights=ResNet18_Weights.SENTINEL2_RGB_MOCO),
            transform = T.Compose([
                T.CenterCrop((224, 224)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            feature_fn=_resnet_gap_feature_fn,
        )
    )
    specs.append(
        ModelSpec(
            name="ResNet50_S2_RGB_MOCO",
            in_chans=3,
            rgb_mode=True,
            builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO),
            transform = T.Compose([
                T.CenterCrop((224, 224)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            feature_fn=_resnet_gap_feature_fn,
        )
    )



    specs.append(
        ModelSpec(
            name="text CIIP, Epoch70",
            in_chans=12,
            rgb_mode=False,
            checkpoint_path="/local/ms-data/SSL4EO/model/2025_12_2-text-s2/checkpoints/epoch_70.pt",
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_12_2-text-s2/checkpoints/epoch_70.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                # S2ScaleTransform(scale=10000.0),
                ssl4eo_s2a_norm_transform,
                # no additional normalization in the ID pipeline
            ]) #get_transform('s2a', is_train=False)
        )
    )

    specs.append(
        ModelSpec(
            name="llama3_ms_clip_base",
            in_chans=10,
            rgb_mode=False,
            builder=_load_llama3_ms_clip_model,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                # S2ScaleTransform(scale=10000.0),
                raise ValueError("LLaMA-3 MS-CLIP model expects 10-channel input; ensure your dataset is configured to provide this."),
                SelectS2Channels10(),
            ]),
            feature_fn=_open_clip_vit_feature_fn,
        )
    )

    specs.append(
        ModelSpec(
            name="remoteclip",
            in_chans=3,
            rgb_mode=True,
            builder=_build_remoteclip_model,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                SelectRGB(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            feature_fn=_remoteclip_feature_fn,
        )
    )

        # 9. ResNet152 ImageNet-1K (RGB only)
    specs.append(
        ModelSpec(
            name="ResNet152_ImageNet_RGB",
            in_chans=3,
            rgb_mode=True,
            builder=lambda: resnet152(weights=ResNet152_Weights.IMAGENET1K_V1),
            transform = T.Compose([
                T.CenterCrop((224, 224)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            feature_fn=_resnet_gap_feature_fn,
        )
    )


    # 10. ViT-Small Sentinel-2 13-band MoCo
    specs.append(
        ModelSpec(
            name="ViTSmall16_S2_ALL_MOCO",
            in_chans=13,
            rgb_mode=False,
            builder=lambda: vit_small_patch16_224(
                weights=ViTSmall16_Weights.SENTINEL2_ALL_MOCO
            ),
            transform = T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0),
                # no additional normalization in the ID pipeline
            ]),
            feature_fn=_vit_patch_mean_feature_fn,
        )
    )



    specs.append(
        ModelSpec(
            name="ViT-DAI CIIP, Epoch300",
            in_chans=12,
            rgb_mode=False,
            return_all_timestamps_ssl4eo=True,
            checkpoint_path="/local/ms-data/SSL4EO/model/1_28-ViT-DAI/checkpoints/epoch_300.pt",
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/1_28-ViT-DAI/checkpoints/epoch_300.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0)
                # ssl4eo_s2a_norm_transform,
                # no additional normalization in the ID pipeline
            ]) #get_transform('s2a', is_train=False)
        )
    )

    specs.append(
        ModelSpec(
            name="Matryoshka ViT-DAI CIIP, Epoch300",
            in_chans=12,
            rgb_mode=False,
            return_all_timestamps_ssl4eo=True,
            checkpoint_path="/local/ms-data/SSL4EO/model/2026_01_29_matryoshka_vit/checkpoints/epoch_300.pt",
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2026_01_29_matryoshka_vit/checkpoints/epoch_300.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_feature_fn,
            eval_dims=[8, 16, 64, 128, 256, 512],
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0)
                # ssl4eo_s2a_norm_transform,
                # no additional normalization in the ID pipeline
            ]) #get_transform('s2a', is_train=False)
        )
    )


    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=0.1, Epoch70",
    #         in_chans=12,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_70.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #         transform=T.Compose([
    #             T.CenterCrop((224, 224)),
    #             S2ScaleTransform(scale=10000.0)
    #             # ssl4eo_s2a_norm_transform,
    #             ]
    #         )
    #     )
    # )




    return specs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_s2_like_loader(
    in_chans: int,
    batch_size: int = 64,
    num_workers: int = 12,
    rgb_mode: bool = False,
) -> DataLoader:
  
    loader = build_s2_100k_loader(
        in_chans=in_chans,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return loader 


def ssl4eo_collate(batch):
    """
    batch: list of dicts from _SSL4EOWrapped.__getitem__
      each dict has:
        "image": [P, C, H, W] or [P, T, C, H, W]
        "label": scalar (per 64-pack)

    Returns:
      "image": [B*T*P, C, H, W]  (T=1 in legacy mode)
      "label": [B*T*P]  (broadcasted per sample)
      "time_dim": scalar T
      "patches_per_file": scalar P
    """
    images = torch.stack([b["image"] for b in batch], dim=0)  # [B, P, ...]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    # Normalize to [B, P, T, C, H, W]
    if images.ndim == 5:
        # [B, P, C, H, W]
        images = images.unsqueeze(2)
    elif images.ndim != 6:
        raise RuntimeError(f"Unexpected SSL4EO batch image shape: {images.shape}")

    B, P, T, C, H, W = images.shape  # P should be 64
    # Move time dimension before patches for easier flattening
    images = images.permute(0, 2, 1, 3, 4, 5)  # [B, T, P, C, H, W]

    images = images.reshape(B * T * P, C, H, W)

    labels = labels.repeat_interleave(T * P)  # [B*T*P]

    return {"image": images, "label": labels, "time_dim": T, "patches_per_file": P}

def build_ssl4eo_like_loader(
    in_chans: int,
    batch_size: int = 64,
    num_workers: int = 12,
    rgb_mode: bool = False,
    return_all_timestamps: bool = False,
) -> DataLoader:
  
    loader = build_s2_ssl4eo_loader(
        in_chans=in_chans,
        batch_size=batch_size,
        num_workers=num_workers,
        return_all_timestamps=return_all_timestamps,
    )

    # bands = _resolve_eurosat_bands(in_chans)
    # loaders = _build_eurosat_loaders(config, bands=bands, modality=EUROSAT_MODALITY)
    return loader #s["test"]


# ---------------------------------------------------------------------------
# Embedding extraction & global ID
# ---------------------------------------------------------------------------
def extract_embeddings_model(
    model: nn.Module,
    feature_fn: FeatureFn,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]],
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    in_chans: Optional[int] = None,
) -> np.ndarray:
    """
    Run the model over the loader and collect (N, D) features.
    """
    model.eval()
    feats_list: List[torch.Tensor] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                image, labels = batch['image'], batch['label']
                time_dim = batch.get("time_dim", 1)
                patches_per_file = batch.get("patches_per_file", 1)
            image = image.to(device)
            model = model.to(device)

            if transform:
                # print(image.shape)
                image = transform(image)
                
            else:
                print('No transform applied during feature extraction.')

            # print bandwise averages/stds
            # print(f"Batch {b_idx}: image band means: {image.mean(dim=[0,2,3])}, stds: {image.std(dim=[0,2,3])}")
            # print(image.shape)
            # if band dim differs,
            if in_chans != image.shape[1]:
                # adjust 2nd dim
                B10 = torch.zeros((1, 1, *image.shape[2:]), dtype=image.dtype, device=image.device)
                image = torch.cat([image[:, :10, :, :], B10.repeat(image.shape[0], 1, 1, 1), image[:, 10:, :, :]], dim=1)
            z = feature_fn(model, image)  # (B, D)


            # remove image and labels to free memory
            del image
            del labels

            # if b_idx == 0:
            #     print(f"Feature shape per batch: {z.shape}")
            if z.ndim != 2:
                # print model class
                print(type(model))
                raise RuntimeError(f"Expected (B, D) features, got {z.shape}")
            # If time_dim > 1, reshape and average embeddings across the temporal axis per patch
            if time_dim > 1:
                # B_total = batch_size_files * T * P
                # Recover batch_size_files
                if patches_per_file is None or patches_per_file == 0:
                    raise RuntimeError("Invalid patches_per_file for temporal averaging.")
                batch_files = z.shape[0] // (time_dim * patches_per_file)
                if batch_files == 0:
                    raise RuntimeError("Failed to infer batch size for temporal averaging.")
                z = z.view(batch_files, time_dim, patches_per_file, -1).mean(dim=1)  # (B_files, P, D)
                z = z.reshape(batch_files * patches_per_file, -1)  # (B_files * P, D)
            feats_list.append(z.cpu())

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    if not feats_list:
        raise RuntimeError("No features extracted – check your loader.")

    Z = torch.cat(feats_list, dim=0).numpy()
    return Z  # (N, D)

def compute_effective_rank(Z: np.ndarray) -> float:
    singular_values = compute_singular_values(torch.from_numpy(Z))
    if singular_values.size == 0:
        return float("nan")
    p = singular_values / (singular_values.sum() + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(H))

def compute_global_id(Z: np.ndarray) -> float:
    """
    Compute global intrinsic dimension using FisherS (angle-based).
    """
    model = id.FisherS()
    global_id = model.fit_transform(Z)
    return float(global_id)

def compute_tle(Z: np.ndarray) -> float:
    """
    Compute global intrinsic dimension using TLE.
    """
    tle = id.TLE()
    tle_pw = tle.fit_transform_pw(Z, n_neighbors=20)  # pointwise IDs
    
    # count number that are below 0.00001
    num_below_threshold = np.sum(tle_pw < 0)
    print("Dropping TLE pw below 0.00001:", num_below_threshold)

    # drop below 0
    tle_pw = tle_pw[tle_pw >= 0]

    # global TLE ignoring NaNs
    global_tle = np.nanmean(tle_pw)
    print("Global TLE (nanmean):", global_tle)
    return float(global_tle)


def compute_id_metrics(Z: np.ndarray) -> Dict[str, float]:
    # gid = compute_global_id(Z)
    # mle_lid = id.MLE(neighborhood_based=True).fit_transform(Z, n_neighbors=20)
    # mom_lid = id.MOM().fit_transform(Z, n_neighbors=20)
    # tle_lid = compute_tle(Z)
    erank = compute_effective_rank(Z)
    return {
        "effective_rank": float(erank),
    }
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # device = torch.device("cuda")
    device = 'cuda:1' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        OUTPUT_DIR,
        task_name="intrinsic_dimension",
        config={
            "output_dir": str(OUTPUT_DIR),
        },
    )

    specs = build_model_specs()

    er_results: Dict[str, float] = {}

    for spec in specs:
        print(f"\n=== Model: {spec.name} ===")
        # try:
        dataset_name = "S2"
        loader = build_s2_like_loader(
            in_chans=spec.in_chans,
            batch_size=64,
            num_workers=6,
            rgb_mode=spec.rgb_mode,
        )

        # dataset_name = "SSL4EO"
        # loader = build_ssl4eo_like_loader(
        #     in_chans=spec.in_chans,
        #     batch_size=8,
        #     num_workers=12,
        #     rgb_mode=spec.rgb_mode,
        #     return_all_timestamps=spec.return_all_timestamps_ssl4eo,
        # )

        model = spec.builder().to(device)
        # print(model)
        # if type resnet
        if isinstance(model, torchvision.models.ResNet):
            # remove final fc layer
            model.fc = nn.Identity()
        # if ciip
        if 'ciip' in spec.name.lower():
            print("CIIP backbone extraction handled in feature_fn.")

        model.eval()

        model = model.to(device)

        with torch.no_grad():
            Z = extract_embeddings_model(
                model=model,
                feature_fn=spec.feature_fn,
                transform=spec.transform,
                loader=loader,
                device=device,
                max_batches=None,  # or limit for debugging
                in_chans=spec.in_chans,
            )
     
        print(f"  Extracted {Z.shape[0]} embeddings with dimension {Z.shape[1]}.")

        Z = Z.astype(np.float64)

        # duplicate samples will cause issues
        uniques = np.unique(Z, axis=0)
        print(f'Removing duplicates: {len(Z) - len(uniques)} samples removed.')

        # remove duplicates
        Z = uniques

        # var_per_dim = Z.var(axis=0)
        # print("Min var:", var_per_dim.min(), "Max var:", var_per_dim.max())
        # print("Dims with ~0 variance:", np.sum(var_per_dim < 1e-10))

        # Remove invalid samples
        embeddings = torch.from_numpy(Z)
        norms = torch.norm(embeddings, dim=1)
        mask_valid = (norms > 1e-6) & torch.isfinite(norms)
        embeddings = embeddings[mask_valid]

        # Check for duplicate/constant vectors
        unique_embeddings = torch.unique(embeddings, dim=0)
        if unique_embeddings.size(0) < 0.9 * embeddings.size(0):
            print("⚠️ Warning: >10% duplicate or constant vectors detected!")

        try:
            embedding_dim = int(Z.shape[1])
            dims_for_eval = [embedding_dim]
            if spec.eval_dims:
                dims_for_eval.extend(spec.eval_dims)
            dims_for_eval = sorted({d for d in dims_for_eval if d > 0})

            dataset_key = dataset_name.lower()
            id_results_by_dim: Dict[str, Dict[str, Dict[str, float]]] = {}
            full_metrics: Optional[Dict[str, float]] = None

            for dim in dims_for_eval:
                if dim > embedding_dim:
                    print(f"  [SKIP] dim {dim} exceeds embedding dim {embedding_dim}.")
                    continue
                Z_dim = Z[:, :dim]
                uniques_dim = np.unique(Z_dim, axis=0)
                if len(uniques_dim) != len(Z_dim):
                    print(f"  [dim {dim}] Removing duplicates: {len(Z_dim) - len(uniques_dim)} samples removed.")
                metrics = compute_id_metrics(uniques_dim)
                id_results_by_dim[str(dim)] = {dataset_key: metrics}
                if dim == embedding_dim:
                    full_metrics = metrics
                print(
                    f"  dim {dim} -> "
                    f"ERank {metrics['effective_rank']:.4f}"
                )

            if full_metrics is None:
                raise RuntimeError("Failed to compute metrics for full embedding dim.")

            er_results[spec.name] = full_metrics["effective_rank"]

            entry = {
                "dataset": dataset_name,
                "embedding_dim": embedding_dim,
                "dims_for_eval": sorted({d for d in dims_for_eval if d <= embedding_dim}),
                "id_results": id_results_by_dim,
                "model_spec": _model_spec_payload(spec, dataset_name),
                "metrics": full_metrics,
            }
            _append_json_entry(_model_results_path(OUTPUT_DIR, spec), entry)

        except Exception as e:
            print(f"  [SKIP] Failed to compute ID for '{spec.name}': {e}")

    # Summary
    print("\n=== Summary of Effective Rank ===")
    for name, gid in er_results.items():
        print(f"{name:30s}: {gid:.4f}")

    # save all to same txt file
    with open(OUTPUT_DIR / "global_id_results.txt", "w") as f:
        f.write("=== Effective Rank ===\n")
        for name, gid in er_results.items():
            f.write(f"{name:30s}: {gid:.4f}\n")

    print(f"\nResults saved to {OUTPUT_DIR / 'global_id_results.txt'}")


if __name__ == "__main__":
    main()
