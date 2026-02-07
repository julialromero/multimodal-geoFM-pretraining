#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import re
import csv
import sys
import yaml
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from hydra.utils import instantiate
import torchvision.transforms as T
import skdim.id as id
import json
from ciip.open_clip_train.data import SSL4EODataset
from ciip.evaluation.model_utils import build_model_from_checkpoint
from ciip.evaluation.normalization_utils import IMAGENET_MEAN, IMAGENET_STD, MODALITY_STATS
from visualizations.ssl4eo.embedding_collapse_diagnostics import compute_singular_values
import neuco_downstream_loader as neuco_loader


DEFAULT_MATRYOSHKA_DIMS: Sequence[int] = (8, 16, 64, 128, 256, 512, 1024)

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

DEFAULT_PREPROCESS = "scaled"

EUROSAT_MEAN = {
    "B01": 1354.40546513,
    "B02": 1118.24399958,
    "B03": 1042.92983953,
    "B04": 947.62620298,
    "B05": 1199.47283961,
    "B06": 1999.79090914,
    "B07": 2369.22292565,
    "B08": 2296.82608323,
    "B09": 732.08340178,
    "B10": 12.11327804,
    "B11": 1819.01027855,
    "B12": 1118.92391149,
    "B8A": 2594.14080798,
}

EUROSAT_STD = {
    "B01": 245.71762908,
    "B02": 333.00778264,
    "B03": 395.09249139,
    "B04": 593.75055589,
    "B05": 566.4170017,
    "B06": 861.18399006,
    "B07": 1086.63139075,
    "B08": 1117.98170791,
    "B09": 404.91978886,
    "B10": 4.77584468,
    "B11": 1002.58768311,
    "B12": 761.30323499,
    "B8A": 1231.58581042,
}

PANGAEA_CONFIG_ROOT = Path("/local/ms-data/pangaea-bench/configs")
PANGAEA_DATA_ROOT = Path("/local/ms-data/pangaea-bench/data")
PANGAEA_RESULTS_CSV = Path("/local/ms-data/pangaea-bench/batch_runs/metrics_summary/summary_metrics.csv")

PANGAEA_DATA_STATS = {
    "ai4smallfarms": {
        "data_mean": {"optical": [750.2136, 1032.7277, 1165.1279, 2416.0448]},
        "data_std": {"optical": [283.3842, 332.728, 518.9025, 702.3791]},
    },
    "hlsburnscars": {
        "data_mean": {"optical": [0.033349706741586264, 0.05701185520536176, 0.05889748132001316, 0.2323245113436119, 0.1972854853760658, 0.11944914225186566]},
        "data_std": {"optical": [0.02269135568823774, 0.026807560223070237, 0.04004109844362779, 0.07791732423672691, 0.08708738838140137, 0.07241979477437814]},
    },
    "mados": {
        "data_mean": {"optical": [0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902, 0.03680401, 0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944]},
        "data_std": {"optical": [0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412, 0.04992323, 0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115]},
    },
    "sen1floods11": {
        "data_mean": {
            "optical": [1626.91600224, 1396.03470631, 1364.06118417, 1218.22847919, 1466.07290663, 2386.90297537, 2845.61256277, 2622.95796892, 3077.48221481, 486.87436782, 63.77861008, 2030.64763024, 1179.16607221],
            "sar": [-10.184408, -16.895273],
        },
        "data_std": {
            "optical": [700.17133846, 739.09452682, 735.2482388, 864.936695, 776.8803358, 921.36834309, 1084.37346097, 1022.63418007, 1196.44255318, 336.61105431, 143.99923282, 980.87061347, 764.60836557],
            "sar": [4.255339, 5.290568],
        },
    },
    "spacenet7": {
        "data_mean": {"optical": [121.826, 106.52838, 78.372116]},
        "data_std": {"optical": [56.717068, 44.517075, 40.451515]},
    },
}

PANGAEA_MODEL_MAP = {
    ("1_28-ViT-DAI", 300): "ciip_s2_vit_epoch300_scaled_12k_dai",
}

PANGAEA_MODEL_ENCODER = {
    "ciip_s2_vit_epoch300_scaled_12k_dai": "ciip_s2_vit",
}


def _ensure_optional_pangaea_deps() -> None:
    try:
        import google.cloud.storage  # noqa: F401
    except Exception:
        import types

        google_mod = sys.modules.setdefault("google", types.ModuleType("google"))
        _ = google_mod
        cloud_mod = sys.modules.setdefault("google.cloud", types.ModuleType("google.cloud"))
        _ = cloud_mod
        storage_mod = types.ModuleType("google.cloud.storage")

        class _DummyClient:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "google.cloud.storage is required for dataset downloading, "
                    "but is not installed. Set auto_download=False or install "
                    "google-cloud-storage."
                )

            @classmethod
            def create_anonymous_client(cls, *args, **kwargs):
                raise RuntimeError(
                    "google.cloud.storage is required for dataset downloading, "
                    "but is not installed. Set auto_download=False or install "
                    "google-cloud-storage."
                )

        storage_mod.Client = _DummyClient
        sys.modules["google.cloud.storage"] = storage_mod
    try:
        import pyDataverse.api  # noqa: F401
    except Exception:
        import types

        py_mod = sys.modules.setdefault("pyDataverse", types.ModuleType("pyDataverse"))
        api_mod = types.ModuleType("pyDataverse.api")

        class _DummyApi:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "pyDataverse is required for dataset downloading, but is not installed. "
                    "Set auto_download=False or install pyDataverse."
                )

        api_mod.DataAccessApi = _DummyApi
        api_mod.NativeApi = _DummyApi
        py_mod.api = api_mod
        sys.modules["pyDataverse.api"] = api_mod


class SSL4EOWrapped(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

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

        return {"image": x.float(), "label": int(y)}


@dataclass
class ModelSpec:
    name: str
    in_chans: int
    builder: Callable[[], nn.Module]
    feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor]
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    s2_tier: Optional[str] = None


# class S2ScaleTransform(nn.Module):
#     def __init__(self, scale: float = 10000.0):
#         super().__init__()
#         self.scale = scale

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x / self.scale


class S2ScaleClampTransform(nn.Module):
    def __init__(self, scale: float = 10000.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / self.scale
        return torch.clamp(x, 0.0, 1.0)


class SelectRGB(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            if x.shape[1] >= 4:
                return x[:, [3, 2, 1], ...]
        elif x.ndim == 3:
            if x.shape[0] >= 4:
                return x[[3, 2, 1], ...]
        return x


class SelectS2Channels10(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim >= 4:
            channel_dim = 1
        elif x.ndim == 3:
            channel_dim = 0
        else:
            return x

        current = x.shape[channel_dim]
        if current == 10:
            return x

        if current == 13:
            keep = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        elif current == 12:
            keep = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        else:
            keep = list(range(min(current, 10)))

        idx = torch.tensor(keep, device=x.device)
        return torch.index_select(x, channel_dim, idx)


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
        for p in self.parameters():
            p.requires_grad = False
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return h.mean(dim=(2, 3))


def build_preprocess_transform(
    method: str,
    tier: str,
    *,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    method = method.lower()
    tier = tier.lower()
    if method == "scaled":
        return T.Compose([
            T.CenterCrop((224, 224)),
            S2ScaleClampTransform(scale=10000.0),
        ])
    if method == "bandwise":
        if mean is None or std is None:
            if tier == "s2l2a":
                mean, std = MODALITY_STATS["s2l2a"]
            else:
                mean, std = MODALITY_STATS["s2l1c"]
        return T.Compose([
            T.CenterCrop((224, 224)),
            T.Normalize(mean=mean, std=std),
        ])
    raise ValueError(f"Unknown preprocess method: {method}")


def _build_s2_transform_for_preset() -> Callable[[torch.Tensor], torch.Tensor]:
    return T.Compose([
        T.CenterCrop((224, 224)),
        S2ScaleClampTransform(scale=10000.0),
    ])


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


def build_non_ciip_spec(preset: str, preprocess: str) -> ModelSpec:
    preset_key = preset.lower()
    if preset_key == "resnet18_s2_all_moco":
        from torchgeo.models import resnet18, ResNet18_Weights
        return ModelSpec(
            name="resnet18_s2_all_moco",
            in_chans=13,
            builder=lambda: resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO),
            feature_fn=_resnet_gap_feature_fn,
            transform=_build_s2_transform_for_preset(),
        )
    if preset_key == "resnet50_s2_all_moco":
        from torchgeo.models import resnet50, ResNet50_Weights
        return ModelSpec(
            name="resnet50_s2_all_moco",
            in_chans=13,
            builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO),
            feature_fn=_resnet_gap_feature_fn,
            transform=_build_s2_transform_for_preset(),
        )
    if preset_key == "dino":
        from torchgeo.models import resnet50, ResNet50_Weights
        return ModelSpec(
            name="dino",
            in_chans=13,
            builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_DINO),
            feature_fn=_resnet_gap_feature_fn,
            transform=_build_s2_transform_for_preset(),
        )
    if preset_key == "vitsmall16_s2_all_moco":
        from torchgeo.models import vit_small_patch16_224, ViTSmall16_Weights
        return ModelSpec(
            name="vitsmall16_s2_all_moco",
            in_chans=13,
            builder=lambda: vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_MOCO),
            feature_fn=_vit_patch_mean_feature_fn,
            transform=_build_s2_transform_for_preset(),
        )
    if preset_key == "dofa_base_s2_13ch":
        from torchgeo.models import dofa_base_patch16_224, DOFABase16_Weights
        return ModelSpec(
            name="dofa_base_s2_13ch",
            in_chans=13,
            builder=lambda: dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE),
            feature_fn=_dofa_feature_fn,
            transform=_build_s2_transform_for_preset(),
        )
    if preset_key == "scalemae_large_rgb":
        from torchgeo.models import scalemae_large_patch16, ScaleMAELarge16_Weights
        return ModelSpec(
            name="scalemae_large_rgb",
            in_chans=3,
            builder=lambda: scalemae_large_patch16(weights=ScaleMAELarge16_Weights.FMOW_RGB),
            feature_fn=_scalemae_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            s2_tier="rgb",
        )
    if preset_key == "resnet152_imagenet_rgb":
        from torchvision.models import resnet152, ResNet152_Weights
        return ModelSpec(
            name="resnet152_imagenet_rgb",
            in_chans=3,
            builder=lambda: resnet152(weights=ResNet152_Weights.IMAGENET1K_V1),
            feature_fn=_resnet_gap_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            s2_tier="rgb",
        )
    if preset_key == "remoteclip":
        return ModelSpec(
            name="remoteclip",
            in_chans=3,
            builder=_build_remoteclip_model,
            feature_fn=_remoteclip_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                SelectRGB(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                # S2ScaleClampTransform(scale=10000.0),
            ]),
            s2_tier="s2l2a",
        )
    if preset_key == "resnet18_s2_rgb_moco":
        from torchgeo.models import resnet18, ResNet18_Weights
        return ModelSpec(
            name="resnet18_s2_rgb_moco",
            in_chans=3,
            builder=lambda: resnet18(weights=ResNet18_Weights.SENTINEL2_RGB_MOCO),
            feature_fn=_resnet_gap_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            s2_tier="rgb",
        )
    if preset_key == "resnet50_s2_rgb_moco":
        from torchgeo.models import resnet50, ResNet50_Weights
        return ModelSpec(
            name="resnet50_s2_rgb_moco",
            in_chans=3,
            builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO),
            feature_fn=_resnet_gap_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            s2_tier="rgb",
        )
    if preset_key == "croma":
        module = _load_croma_module()
        PretrainedCROMA = getattr(module, "PretrainedCROMA")
        return ModelSpec(
            name="croma",
            in_chans=12,
            builder=lambda: PretrainedCROMA(
                pretrained_path="/home/juro4948/ciip/comparison/CROMA-main/CROMA_base.pt",
                size="base",
                modality="optical",
                image_resolution=120,
            ),
            feature_fn=_croma_optical_gap_feature_fn,
            transform=nn.Sequential(
                T.Resize((120, 120)),
                CromaNormalize(use_8_bit=False),
            ),
            s2_tier="s2l2a",
        )
    if preset_key == "llama3_ms_clip_base":
        return ModelSpec(
            name="llama3_ms_clip_base",
            in_chans=10,
            builder=_load_llama3_ms_clip_model,
            feature_fn=_open_clip_vit_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleClampTransform(scale=10000.0),
                SelectS2Channels10(),
            ]),
            s2_tier="s2l2a",
        )
    if preset_key == "rcf_13ch":
        return ModelSpec(
            name="rcf_13ch",
            in_chans=13,
            builder=lambda: RandomConvFeatures(in_chans=13, out_dim=512),
            feature_fn=lambda m, x: m(x),
            transform=_build_s2_transform_for_preset(),
        )
    raise ValueError(f"Unsupported model preset: {preset}")


def eurosat_bandwise_stats(num_channels: int) -> Tuple[List[float], List[float]]:
    bands = neuco_loader.resolve_eurosat_bands(num_channels)
    mean = [EUROSAT_MEAN[b] for b in bands]
    std = [EUROSAT_STD[b] for b in bands]
    return mean, std


@contextlib.contextmanager
def suppress_output(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _checkpoint_run_name_and_epoch(checkpoint: Path) -> Tuple[str, int]:
    run_name = checkpoint.parent.parent.name
    match = re.search(r"epoch_(\d+)\.pt$", checkpoint.name)
    if not match:
        raise ValueError(f"Could not parse epoch from checkpoint name: {checkpoint}")
    return run_name, int(match.group(1))


def _preprocess_tag(method: str) -> str:
    return "divideby10000" if method == "scaled" else "bandwisenorm"


def _load_json(path: Path) -> Dict:
    import json

    return json.loads(path.read_text())


def _select_split(metrics: Dict, split: str) -> Dict:
    if split not in metrics:
        raise KeyError(f"Split {split} not found in metrics keys: {list(metrics.keys())}")
    return metrics[split]


def _select_latest_dir(parent: Path, pattern: str) -> Optional[Path]:
    candidates = sorted(parent.glob(pattern))
    return candidates[-1] if candidates else None


def get_unified_eval_results(
    *,
    checkpoint: Path,
    preprocess: str,
    matryoshka_dim: Optional[int] = None,
    vit_results: str = "auto",
    root: Path = Path("/home/juro4948/ciip/diagnostics/unified_eval/ciip_checkpoint"),
    model_name_override: Optional[str] = None,
) -> Dict[str, Dict]:
    preprocess_tag = _preprocess_tag(preprocess)
    if model_name_override:
        base_run = root / model_name_override
        is_vit_run = False
        meanpool_run = base_run
    else:
        run_name, epoch = _checkpoint_run_name_and_epoch(checkpoint)
        base_run = root / run_name
        is_vit_run = "vit" in run_name.lower()
        meanpool_run = root / f"{run_name}_meanpool"

    def _select_epoch_dir() -> Path:
        if model_name_override:
            return base_run
        if not is_vit_run:
            return base_run / f"epoch_{epoch}"
        if vit_results == "meanpool":
            candidates = [
                base_run / f"epoch_{epoch}_meanpool",
                meanpool_run / f"epoch_{epoch}_meanpool",
                meanpool_run / f"epoch_{epoch}",
            ]
        elif vit_results == "cls":
            candidates = [
                base_run / f"epoch_{epoch}_CLS",
                base_run / f"epoch_{epoch}",
            ]
        else:
            candidates = [
                base_run / f"epoch_{epoch}_meanpool",
                meanpool_run / f"epoch_{epoch}_meanpool",
                meanpool_run / f"epoch_{epoch}",
                base_run / f"epoch_{epoch}_CLS",
                base_run / f"epoch_{epoch}",
            ]
        for cand in candidates:
            if cand.exists():
                return cand
        # Fall back to the first candidate for consistent logging even if it doesn't exist.
        return candidates[0]

    base = _select_epoch_dir()
    cls_suffix = ""
    if matryoshka_dim is not None:
        eurosat_base = base / f"matryoshka_dim_{matryoshka_dim}"
    else:
        eurosat_base = base

    results: Dict[str, Dict] = {}

    # EuroSAT LP + kNN
    eurosat_dir = eurosat_base / f"linear_probe_{preprocess_tag}{cls_suffix}"
    if not eurosat_dir.exists():
        legacy = eurosat_base / f"linear_probe_{preprocess_tag}_CLS"
        if legacy.exists():
            eurosat_dir = legacy
    print(f"[unified_eval] EuroSAT dir: {eurosat_dir}")
    lp_path = eurosat_dir / "eurosat_backbone_batchnorm_metrics.json"
    knn_path = eurosat_dir / "eurosat_backbone_batchnorm_knn_metrics.json"
    if lp_path.exists():
        results["eurosat_linear_probe_0.1"] = _select_split(_load_json(lp_path), "0.1")
    if knn_path.exists():
        results["eurosat_knn_0.1"] = _select_split(_load_json(knn_path), "0.1")

    # NeuCo-Bench summary
    if matryoshka_dim is not None and not model_name_override:
        neuco_dir = base / f"neuco_{preprocess_tag}{cls_suffix}" / f"matryoshka_dim_{matryoshka_dim}" / "testing"
    else:
        neuco_dir = base / f"neuco_{preprocess_tag}{cls_suffix}" / "testing"
    if not neuco_dir.exists():
        legacy = (
            base
            / f"neuco_{preprocess_tag}_CLS"
            / (f"matryoshka_dim_{matryoshka_dim}" if matryoshka_dim is not None else "")
            / "testing"
        )
        if legacy.exists():
            neuco_dir = legacy
    print(f"[unified_eval] NeuCo dir: {neuco_dir}")
    latest = _select_latest_dir(neuco_dir, "backbone_*")
    if latest is not None:
        summary_path = latest / "results_summary.json"
        if summary_path.exists():
            results["neuco_bench"] = _load_json(summary_path)

    return results


def ssl4eo_collate(batch):
    """
    batch: list of dicts from SSL4EODataset
      each dict has:
        "image": [P, C, H, W] or [P, T, C, H, W]
        "label": scalar

    Returns:
      "image": [B*T*P, C, H, W]
      "label": [B*T*P]
      "time_dim": scalar T
      "patches_per_file": scalar P
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    if images.ndim == 5:
        images = images.unsqueeze(2)
    elif images.ndim != 6:
        raise RuntimeError(f"Unexpected SSL4EO batch image shape: {images.shape}")

    B, P, T, C, H, W = images.shape
    images = images.permute(0, 2, 1, 3, 4, 5)
    images = images.reshape(B * T * P, C, H, W)

    labels = labels.repeat_interleave(T * P)

    return {"image": images, "label": labels, "time_dim": T, "patches_per_file": P}


def build_ssl4eo_val_loader(
    root: Path,
    in_chans: int,
    batch_size: int,
    num_workers: int,
    return_all_timestamps: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    s2_tier: Optional[str] = None,
) -> DataLoader:
    tier = s2_tier or ("s2l2a" if in_chans == 12 else "s2l1c")

    dataset = SSL4EODataset(
        root=root,
        s2_tier=tier,
        seasons=[0, 1, 2, 3],
        num_timestamps=4,
        return_all_timestamps=return_all_timestamps,
        transforms=None,
        is_train=False,
    )

    ds = SSL4EOWrapped(dataset)

    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=ssl4eo_collate,
        sampler=sampler,
    )
    return loader


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def ciip_s2_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    core = _unwrap_model(model)
    feats = core.encoder_s2(x)
    # print if there is a proj lyaer
    if hasattr(core, "proj_s2") and core.proj_s2 is not None:
        raise RuntimeError("CIIP model has a proj_s2 layer, which is not supported in feature extraction.")
    # print if there is an fc layer
    if hasattr(core, "fc") and core.fc is not None:
        raise RuntimeError("CIIP model has a fc layer, which is not supported in feature extraction.")
    if feats.ndim == 3:
        feats = feats.mean(dim=1)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected CIIP feature shape: {feats.shape}")
    return feats


def _resnet_gap_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    core = _unwrap_model(model)
    backbone = None
    for attr in ("backbone", "features", "encoder"):
        if hasattr(core, attr):
            backbone = getattr(core, attr)
            break
    if backbone is not None:
        feats = backbone(x)
    elif hasattr(core, "forward_features"):
        feats = core.forward_features(x)
    else:
        x = core.conv1(x)
        x = core.bn1(x)
        x = core.relu(x)
        x = core.maxpool(x)
        x = core.layer1(x)
        x = core.layer2(x)
        x = core.layer3(x)
        x = core.layer4(x)
        x = core.avgpool(x)
        feats = torch.flatten(x, 1)
    if isinstance(feats, dict):
        for key in ("out", "feat", "features", "x"):
            if key in feats and isinstance(feats[key], torch.Tensor):
                feats = feats[key]
                break
        else:
            raise RuntimeError(f"Unexpected dict from ResNet backbone: {feats.keys()}")
    if feats.ndim == 4:
        feats = feats.mean(dim=(2, 3))
    elif feats.ndim == 3:
        feats = feats.mean(dim=1)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected ResNet feature shape: {feats.shape}")
    return feats


def _vit_patch_mean_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    core = _unwrap_model(model)
    if not hasattr(core, "forward_features"):
        raise RuntimeError("Model has no forward_features method.")
    feats = core.forward_features(x)
    if feats.ndim == 3:
        feats = feats[:, 1:, :].mean(dim=1)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected ViT feature shape: {feats.shape}")
    return feats


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


def _dofa_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    core = _unwrap_model(model)
    feats = core.forward_features(x, wavelengths=S2_WAVELENGTHS_UM)
    if feats.ndim == 3:
        feats = feats.mean(dim=1)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected DOFA feature shape: {feats.shape}")
    return feats


def _scalemae_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    core = _unwrap_model(model)
    if not hasattr(core, "forward_features"):
        raise RuntimeError("ScaleMAE model has no forward_features method.")
    feats = core.forward_features(x)
    if isinstance(feats, dict):
        for key in ("x", "tokens", "sequence", "feat"):
            if key in feats and isinstance(feats[key], torch.Tensor):
                feats = feats[key]
                break
        else:
            raise RuntimeError(f"Unexpected dict from ScaleMAE.forward_features: {feats.keys()}")
    if feats.ndim == 3:
        feats = feats.mean(dim=1)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected ScaleMAE feature shape: {feats.shape}")
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


def _load_croma_module():
    import importlib.util
    from types import ModuleType

    module_path = Path(__file__).resolve().parents[1] / "comparison" / "CROMA-main" / "use_croma.py"
    if not module_path.exists():
        raise FileNotFoundError(
            "CROMA weights requested but 'use_croma.py' was not found. "
            "Expected it under comparison/CROMA-main/use_croma.py."
        )
    spec = importlib.util.spec_from_file_location("croma_use_croma", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import CROMA utilities from '{module_path}'.")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _croma_optical_gap_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    core = _unwrap_model(model)
    out: Dict[str, torch.Tensor] = core(optical_images=x)
    return out["optical_GAP"]


def extract_embeddings(
    model: nn.Module,
    feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]],
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
    in_chans: Optional[int],
) -> np.ndarray:
    model.eval()
    model = model.to(device)
    feats_list: List[torch.Tensor] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            image = batch["image"]
            time_dim = batch.get("time_dim", 1)
            patches_per_file = batch.get("patches_per_file", 1)

            image = image.to(device)
            if transform:
                image = transform(image)

            if in_chans is not None and in_chans != image.shape[1]:
                b10 = torch.zeros((1, 1, *image.shape[2:]), dtype=image.dtype, device=image.device)
                image = torch.cat([image[:, :10, :, :], b10.repeat(image.shape[0], 1, 1, 1), image[:, 10:, :, :]], dim=1)

            z = feature_fn(model, image)
            if z.ndim != 2:
                raise RuntimeError(f"Expected (B, D) features, got {z.shape}")

            if time_dim > 1:
                batch_files = z.shape[0] // (time_dim * patches_per_file)
                if batch_files == 0:
                    raise RuntimeError("Failed to infer batch size for temporal averaging.")
                z = z.view(batch_files, time_dim, patches_per_file, -1).mean(dim=1)
                z = z.reshape(batch_files * patches_per_file, -1)

            feats_list.append(z.cpu())

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    if not feats_list:
        raise RuntimeError("No features extracted - check your loader.")

    Z = torch.cat(feats_list, dim=0).numpy()
    return Z


def extract_embeddings_simple(
    model: nn.Module,
    feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]],
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
    in_chans: Optional[int],
) -> np.ndarray:
    model.eval()
    model = model.to(device)
    feats_list: List[torch.Tensor] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if isinstance(batch, dict):
                image = batch["image"]
                time_dim = batch.get("time_dim", None)
            else:
                image = batch[0]
                time_dim = None
            image = image.to(device)
            if time_dim is not None and image.ndim == 5:
                bsz, tdim, cdim, h, w = image.shape
                image = image.view(bsz * tdim, cdim, h, w)
            if transform:
                image = transform(image)
            if in_chans is not None and in_chans != image.shape[1]:
                b10 = torch.zeros((1, 1, *image.shape[2:]), dtype=image.dtype, device=image.device)
                image = torch.cat([image[:, :10, :, :], b10.repeat(image.shape[0], 1, 1, 1), image[:, 10:, :, :]], dim=1)

            z = feature_fn(model, image)
            if z.ndim != 2:
                raise RuntimeError(f"Expected (B, D) features, got {z.shape}")
            if time_dim is not None:
                z = z.view(bsz, tdim, -1).mean(dim=1)
            feats_list.append(z.cpu())

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    if not feats_list:
        raise RuntimeError("No features extracted - check your loader.")

    Z = torch.cat(feats_list, dim=0).numpy()
    return Z


def compute_global_ids(Z: np.ndarray) -> Dict[str, float]:
    Z = Z.astype(np.float64)

    uniques = np.unique(Z, axis=0)
    if len(uniques) != len(Z):
        print(f"Removing duplicates: {len(Z) - len(uniques)} samples removed.")
        Z = uniques

    results: Dict[str, float] = {}
    results["fishers"] = float(id.FisherS().fit_transform(Z))
    results["mle"] = float(id.MLE(neighborhood_based=True).fit_transform(Z, n_neighbors=20))
    results["mom"] = float(id.MOM().fit_transform(Z, n_neighbors=20))

    tle = id.TLE()
    tle_pw = tle.fit_transform_pw(Z, n_neighbors=20)
    tle_pw = tle_pw[tle_pw >= 0]
    results["tle"] = float(np.nanmean(tle_pw))

    return results


def compute_effective_rank(Z: np.ndarray) -> float:
    singular_values = compute_singular_values(torch.from_numpy(Z))
    if singular_values.size == 0:
        return float("nan")
    p = singular_values / (singular_values.sum() + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(H))


def compute_matryoshka_ids(Z: np.ndarray, dims: Sequence[int]) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    for dim in dims:
        if dim > Z.shape[1]:
            raise ValueError(f"Requested matryoshka dim {dim} exceeds embedding dim {Z.shape[1]}.")
        results[int(dim)] = compute_global_ids(Z[:, :dim])
    return results


def _normalize_neuco_image(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(image):
        image = image.float()
    # Ensure float tensor
    if image.ndim == 3:
        if image.shape[0] not in (3, 12, 13):
            image = image.permute(2, 0, 1).contiguous()
        return image
    if image.ndim == 4:
        if image.shape[1] in (3, 12, 13):
            # (T, C, H, W)
            return image
        if image.shape[0] in (3, 12, 13) and image.shape[-1] == 4:
            # (C, H, W, T) -> (T, C, H, W)
            return image.permute(3, 0, 1, 2).contiguous()
        if image.shape[-1] in (3, 12, 13) and image.shape[0] == 4:
            # (T, H, W, C) -> (T, C, H, W)
            return image.permute(0, 3, 1, 2).contiguous()
        return image
    # For higher dims, average leading dims until 4D
    while image.ndim > 4:
        image = image.mean(dim=0)
    return _normalize_neuco_image(image)


class NeucoTaskDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: Path, data_root: Path, modality: str, temporal_mode: str, temporal_seed: int):
        df = neuco_loader.load_task_csv(csv_path)
        index = neuco_loader.build_id_index(data_root, modality)
        records = []
        for _, row in df.iterrows():
            image_id = str(row["id"])
            path = index.get(image_id)
            if path is None:
                continue
            records.append((image_id, float(row["label"]), path))
        self.records = records
        self.temporal_mode = temporal_mode
        self.temporal_seed = temporal_seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        image_id, label, path = self.records[idx]
        image = neuco_loader.load_image(path)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = _normalize_neuco_image(image)
        time_dim = None
        if image.ndim == 4 and image.shape[0] in (4,):
            time_dim = int(image.shape[0])
            if self.temporal_mode == "random_one":
                rng = np.random.default_rng(self.temporal_seed + idx)
                t = int(rng.integers(time_dim))
                image = image[t]
                time_dim = None
        if image.ndim == 4 and time_dim is not None and self.temporal_mode == "embed-mean":
            return {"image": image.float(), "label": label, "id": image_id, "time_dim": time_dim}
        if image.ndim == 4:
            image = image.mean(dim=0)
        return {"image": image.float(), "label": label, "id": image_id}


def compute_dataset_id_er(
    name: str,
    loader: DataLoader,
    model: nn.Module,
    feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]],
    device: torch.device,
    in_chans: Optional[int],
    max_batches: Optional[int],
) -> Dict[str, float]:
    print(f"=== Downstream dataset: {name} ===")
    Z = extract_embeddings_simple(
        model=model,
        feature_fn=feature_fn,
        transform=transform,
        loader=loader,
        device=device,
        max_batches=max_batches,
        in_chans=in_chans,
    )
    erank = compute_effective_rank(Z)
    ids = compute_global_ids(Z)
    print(f"{name} effective_rank: {erank}")
    print(f"{name} ids: {ids}")
    return {"effective_rank": erank, **ids}


def compute_dataset_id_er_multi(
    name: str,
    loader: DataLoader,
    model: nn.Module,
    feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]],
    device: torch.device,
    in_chans: Optional[int],
    dims: Sequence[int],
    max_batches: Optional[int],
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
    print(f"=== Downstream dataset: {name} ===")
    Z = extract_embeddings_simple(
        model=model,
        feature_fn=feature_fn,
        transform=transform,
        loader=loader,
        device=device,
        max_batches=max_batches,
        in_chans=in_chans,
    )
    id_by_dim: Dict[int, Dict[str, float]] = {}
    er_by_dim: Dict[int, float] = {}
    for dim in dims:
        if dim > Z.shape[1]:
            print(f"Skipping dim {dim} for {name}: exceeds embedding dim {Z.shape[1]}")
            continue
        Zs = Z[:, :dim]
        erank = compute_effective_rank(Zs)
        ids = compute_global_ids(Zs)
        id_by_dim[dim] = ids
        er_by_dim[dim] = erank
        print(f"{name} dim={dim} effective_rank: {erank}")
        print(f"{name} dim={dim} ids: {ids}")
    return id_by_dim, er_by_dim


def _plot_metric_by_task(
    metrics: Dict[int, Dict[str, Dict[str, float]]],
    tasks: List[str],
    metric_key: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(18, 5))
    for dim, task_map in sorted(metrics.items()):
        y = [task_map.get(task, {}).get(metric_key, float("nan")) for task in tasks]
        plt.plot(tasks, y, marker="o", linewidth=2, label=f"dim_{dim}")
    plt.title(title)
    plt.xlabel("Dataset task")
    plt.ylabel(metric_key)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_id_metrics_grid(
    metrics: Dict[int, Dict[str, Dict[str, float]]],
    tasks: List[str],
    out_path: Path,
) -> None:
    metric_keys = ["fishers", "mle", "mom", "tle"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharex=True)
    for ax, key in zip(axes, metric_keys):
        for dim, task_map in sorted(metrics.items()):
            y = [task_map.get(task, {}).get(key, float("nan")) for task in tasks]
            ax.plot(tasks, y, marker="o", linewidth=2, label=f"dim_{dim}")
        ax.set_title(key)
        ax.set_xlabel("Dataset task")
        ax.set_ylabel("Intrinsic dimension")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=90)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    axes[0].legend(ncol=2)
    fig.suptitle("Intrinsic dimension metrics by dataset task")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_effective_rank_on_ax(
    ax: plt.Axes,
    eranks: Dict[int, Dict[str, float]],
    tasks: List[str],
    title: str,
) -> None:
    for dim, task_map in sorted(eranks.items()):
        y = [task_map.get(task, float("nan")) for task in tasks]
        ax.plot(tasks, y, marker="o", linewidth=2, label=f"dim_{dim}")
    ax.set_title(title)
    ax.set_xlabel("Dataset task")
    ax.set_ylabel("Effective rank")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def _plot_effective_rank(
    eranks: Dict[int, Dict[str, float]],
    tasks: List[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 5))
    _plot_effective_rank_on_ax(ax, eranks, tasks, "Effective rank by dataset task")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_id_and_er_metrics_grid(
    metrics: Dict[int, Dict[str, Dict[str, float]]],
    eranks: Dict[int, Dict[str, float]],
    tasks: List[str],
    out_path: Path,
) -> None:
    metric_keys = ["fishers", "mle", "mom", "tle"]
    fig, axes = plt.subplots(2, 3, figsize=(24, 10), sharex=True)
    axes = axes.reshape(2, 3)
    metric_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]
    ax_er = axes[1, 1]
    axes[1, 2].axis("off")

    for ax, key in zip(metric_axes, metric_keys):
        for dim, task_map in sorted(metrics.items()):
            y = [task_map.get(task, {}).get(key, float("nan")) for task in tasks]
            ax.plot(tasks, y, marker="o", linewidth=2, label=f"dim_{dim}")
        ax.set_title(key)
        ax.set_xlabel("Dataset task")
        ax.set_ylabel("Intrinsic dimension")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=90)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    _plot_effective_rank_on_ax(ax_er, eranks, tasks, "Effective rank by dataset task")

    metric_axes[0].legend(ncol=2)
    ax_er.legend(ncol=2)
    fig.suptitle("Intrinsic dimension metrics and effective rank by dataset task")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_performance_by_task(
    performance: Dict[int, Dict[str, float]],
    tasks: List[str],
    out_path: Path,
) -> None:
    plt.figure(figsize=(18, 5))
    for dim, task_map in sorted(performance.items()):
        y = [task_map.get(task, float("nan")) for task in tasks]
        plt.plot(tasks, y, marker="o", linewidth=2, label=f"dim_{dim}")
    plt.title("Downstream performance by task (EuroSAT/NeuCo)")
    plt.xlabel("Dataset task")
    plt.ylabel("Performance")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_performance_vs_id(
    performance_by_preprocess: Dict[str, Dict[int, Dict[str, float]]],
    eranks: Dict[int, Dict[str, float]],
    id_metrics: Dict[int, Dict[str, Dict[str, float]]],
    out_path: Path,
    *,
    eurosat_task_key: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    preprocess_order = [("bandwise", "Bandwise"), ("scaled", "Scaled")]
    metric_rows = [
        ("Effective rank", "Intrinsic dimension (effective rank)"),
        ("FisherS", "Intrinsic dimension (FisherS)"),
    ]

    def clean_task_name(task: str) -> str:
        return re.sub(r"\[.*\]$", "", task).strip()

    for row_idx, (metric_label, xlabel) in enumerate(metric_rows):
        for col_idx, (preprocess_key, title) in enumerate(preprocess_order):
            ax = axes[row_idx][col_idx]
            perf_by_dim = performance_by_preprocess.get(preprocess_key, {})
            if not perf_by_dim:
                ax.set_title(f"{title} ({metric_label}) (no data)")
                ax.set_xlabel(xlabel)
                ax.grid(True, alpha=0.3)
                continue

            for dim, task_map in sorted(perf_by_dim.items()):
                for task, score in task_map.items():
                    if np.isnan(score):
                        continue
                    base_task = clean_task_name(task)
                    if base_task.startswith("eurosat_"):
                        x_key = eurosat_task_key
                    else:
                        x_key = base_task
                    if metric_label == "Effective rank":
                        x = eranks.get(dim, {}).get(x_key, float("nan"))
                    else:
                        x = id_metrics.get(dim, {}).get(x_key, {}).get("fishers", float("nan"))
                    if np.isnan(x):
                        continue
                    is_eurosat = base_task.startswith("eurosat_")
                    marker = "s" if is_eurosat else "o"
                    ax.scatter(x, score, marker=marker, alpha=0.8)
                    ax.text(x, score, task, fontsize=8, alpha=0.8)

            ax.set_title(f"{title} ({metric_label})")
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)

    axes[0][0].set_ylabel("Performance")
    axes[1][0].set_ylabel("Performance")
    fig.suptitle("Downstream performance vs intrinsic dimension (by preprocess)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _serialize_id_cache(
    id_results: Dict[int, Dict[str, Dict[str, float]]],
    er_results: Dict[int, Dict[str, float]],
    *,
    dims_for_eval: List[int],
    meta: Dict[str, object],
    performance_results: Optional[Dict[int, Dict[str, float]]] = None,
    performance_by_preprocess: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "meta": meta,
        "dims_for_eval": dims_for_eval,
        "id_results": {str(k): v for k, v in id_results.items()},
        "er_results": {str(k): v for k, v in er_results.items()},
    }
    if performance_results is not None:
        payload["performance_results"] = {str(k): v for k, v in performance_results.items()}
    if performance_by_preprocess is not None:
        payload["performance_by_preprocess"] = {
            preprocess: {str(k): v for k, v in per_dim.items()}
            for preprocess, per_dim in performance_by_preprocess.items()
        }
    return payload


def _load_id_cache(cache_path: Path) -> Optional[Dict[str, object]]:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return None


def _cache_meta_matches(
    payload: Dict[str, object],
    cache_meta: Dict[str, object],
    dims_for_eval: List[int],
) -> bool:
    cached_meta = payload.get("meta", {})
    cached_dims = payload.get("dims_for_eval", [])
    return cached_meta == cache_meta and list(cached_dims) == list(dims_for_eval)


def _find_matching_cache(
    cache_dir: Path,
    cache_tag: str,
    preprocess: str,
    cache_meta: Dict[str, object],
    dims_for_eval: List[int],
) -> Tuple[Optional[Path], Optional[Dict[str, object]]]:
    pattern = f"id_metrics_cache_{cache_tag}_{preprocess}*.json"
    matches: List[Tuple[Path, Dict[str, object]]] = []
    for path in sorted(cache_dir.glob(pattern)):
        payload = _load_id_cache(path)
        if isinstance(payload, dict) and _cache_meta_matches(payload, cache_meta, dims_for_eval):
            matches.append((path, payload))
    if not matches:
        return None, None
    if len(matches) > 1:
        print(f"Multiple matching caches found for {cache_tag}/{preprocess}; using {matches[0][0]}")
    return matches[0]


def _unique_cache_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _alias_task_metrics(
    id_results: Dict[int, Dict[str, Dict[str, float]]],
    er_results: Dict[int, Dict[str, float]],
    source_key: str,
    alias_keys: Sequence[str],
) -> None:
    for dim, task_map in id_results.items():
        if source_key not in task_map:
            continue
        for alias in alias_keys:
            task_map.setdefault(alias, task_map[source_key])
    for dim, task_map in er_results.items():
        if source_key not in task_map:
            continue
        for alias in alias_keys:
            task_map.setdefault(alias, task_map[source_key])

def _pangaea_model_for_checkpoint(checkpoint_path: Optional[Path]) -> Optional[str]:
    if checkpoint_path is None:
        return None
    epoch_match = re.search(r"epoch_(\d+)", checkpoint_path.name)
    if not epoch_match:
        return None
    epoch = int(epoch_match.group(1))
    run_name = checkpoint_path.parent.parent.name
    return PANGAEA_MODEL_MAP.get((run_name, epoch))


def _load_pangaea_scores(csv_path: Path, model_name: str) -> Dict[str, Dict[str, float]]:
    if not csv_path.exists():
        return {}
    results: Dict[str, Dict[str, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("model") != model_name:
                continue
            dataset = row.get("dataset")
            if not dataset:
                continue
            downstream = (row.get("downstream_preprocessing") or "").strip().lower()
            if "scaled" in downstream:
                preprocess_key = "scaled"
            elif "mean" in downstream:
                preprocess_key = "bandwise"
            else:
                continue
            try:
                score = float(row.get("mIoU_mean", "nan")) / 100.0
            except ValueError:
                continue
            task_key = f"pangaea_{dataset.strip()}"
            results.setdefault(preprocess_key, {})[task_key] = score
    return results


def _pangaea_preprocess_from_row(row: Dict[str, str]) -> Optional[str]:
    downstream = (row.get("downstream_preprocessing") or "").strip().lower()
    if "scaled" in downstream:
        return "scaled"
    if "mean" in downstream:
        return "bandwise"
    return None


def _pangaea_datasets_for_model(
    csv_path: Path,
    model_name: str,
    preprocess_key: str,
) -> List[Tuple[str, str]]:
    if not csv_path.exists():
        return []
    datasets: Dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("model") != model_name:
                continue
            row_preprocess = _pangaea_preprocess_from_row(row)
            if row_preprocess != preprocess_key:
                continue
            dataset = (row.get("dataset") or "").strip()
            preprocess_name = (row.get("preprocessing") or "").strip()
            if not dataset or not preprocess_name:
                continue
            datasets.setdefault(dataset, preprocess_name)
    return sorted(datasets.items())


def _load_pangaea_dataset_cfg(dataset_name: str) -> Dict:
    cfg_path = PANGAEA_CONFIG_ROOT / "dataset" / f"{dataset_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing dataset config for {dataset_name}: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    if "data_mean" not in cfg or "data_std" not in cfg:
        raise ValueError(f"Dataset config missing data_mean/std for {dataset_name}: {cfg_path}")
    root_path = cfg.get("root_path")
    if root_path:
        root = Path(root_path)
        if not root.is_absolute():
            parts = [part for part in root.parts if part not in (".", "")]
            if parts and parts[0] == "data":
                parts = parts[1:]
            root = PANGAEA_DATA_ROOT.joinpath(*parts) if parts else (PANGAEA_DATA_ROOT / dataset_name)
    else:
        root = PANGAEA_DATA_ROOT / dataset_name
    cfg["root_path"] = str(root)
    cfg["auto_download"] = False
    if not root.exists():
        raise FileNotFoundError(
            f"Pangaea dataset not found at {root}. "
            "Auto-download is disabled; please confirm the dataset path."
        )
    stats = PANGAEA_DATA_STATS.get(dataset_name)
    if stats is None:
        raise ValueError(f"Missing bandwise stats for Pangaea dataset '{dataset_name}'.")
    cfg["data_mean"] = stats["data_mean"]
    cfg["data_std"] = stats["data_std"]
    return cfg


def _load_pangaea_encoder_cfg(encoder_name: str) -> Dict:
    cfg_path = PANGAEA_CONFIG_ROOT / "encoder" / f"{encoder_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing encoder config for {encoder_name}: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())


def _build_pangaea_preprocessor(
    preprocess_name: str,
    dataset_cfg: Dict,
    encoder_cfg: Dict,
    *,
    split: str,
) -> object:
    cfg_path = PANGAEA_CONFIG_ROOT / "preprocessing" / f"{preprocess_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing preprocessing config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    pre_cfg = cfg.get(split) or cfg.get("test") or cfg.get("val") or cfg.get("train")
    if pre_cfg is None:
        raise ValueError(f"No preprocessing stage found in {cfg_path}")
    return instantiate(pre_cfg, dataset_cfg=dataset_cfg, encoder_cfg=encoder_cfg, _recursive_=False)


def _build_pangaea_dataset(
    dataset_name: str,
    preprocess_name: str,
    encoder_cfg: Dict,
    *,
    split: str = "train",
):
    if str(PANGAEA_CONFIG_ROOT.parent) not in sys.path:
        sys.path.append(str(PANGAEA_CONFIG_ROOT.parent))
    try:
        import importlib

        _ensure_optional_pangaea_deps()
        importlib.import_module(f"pangaea.datasets.{dataset_name}")
    except Exception:
        pass
    from pangaea.datasets.base import GeoFMDataset

    dataset_cfg = _load_pangaea_dataset_cfg(dataset_name)
    preprocessor = _build_pangaea_preprocessor(
        preprocess_name,
        dataset_cfg,
        encoder_cfg,
        split=split,
    )
    raw_dataset = instantiate(dataset_cfg, split=split)
    if hasattr(raw_dataset, "auto_download"):
        raw_dataset.auto_download = False
    return GeoFMDataset(raw_dataset, preprocessor)


def _build_pangaea_loader(
    dataset,
    encoder_cfg: Dict,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if str(PANGAEA_CONFIG_ROOT.parent) not in sys.path:
        sys.path.append(str(PANGAEA_CONFIG_ROOT.parent))
    from pangaea.utils.collate_fn import get_collate_fn

    modalities = list((encoder_cfg.get("input_bands") or {}).keys())
    if not modalities:
        modalities = ["optical"]
    collate_fn = get_collate_fn(modalities)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )


def ciip_s2_meanpool_backbone_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    core = model.module if hasattr(model, "module") else model
    encoder = core.encoder_s2
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
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.ndim == 4:
        feats = feats.mean(dim=(-2, -1))
    elif feats.ndim == 3:
        feats = feats.mean(dim=1)
    return feats


def extract_pangaea_embeddings(
    model: nn.Module,
    feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
    modality: Optional[str] = None,
) -> np.ndarray:
    model.eval()
    model = model.to(device)
    feats_list: List[torch.Tensor] = []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            image_dict = batch.get("image", {})
            if modality is None or modality not in image_dict:
                modality_key = next(iter(image_dict.keys()))
            else:
                modality_key = modality
            image = image_dict[modality_key].to(device)
            if image.ndim == 5:
                image = image.mean(dim=2)
            z = feature_fn(model, image)
            if z.ndim != 2:
                raise RuntimeError(f"Expected (B, D) features, got {z.shape}")
            feats_list.append(z.cpu())
            if max_batches is not None and (b_idx + 1) >= max_batches:
                break
    if not feats_list:
        raise RuntimeError("No Pangaea features extracted - check your loader.")
    return torch.cat(feats_list, dim=0).numpy()


def compute_pangaea_id_er_multi(
    name: str,
    loader: DataLoader,
    model: nn.Module,
    feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    device: torch.device,
    dims: Sequence[int],
    max_batches: Optional[int],
    modality: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
    print(f"=== Pangaea dataset: {name} ===")
    Z = extract_pangaea_embeddings(
        model=model,
        feature_fn=feature_fn,
        loader=loader,
        device=device,
        max_batches=max_batches,
        modality=modality,
    )
    id_by_dim: Dict[int, Dict[str, float]] = {}
    er_by_dim: Dict[int, float] = {}
    for dim in dims:
        if dim > Z.shape[1]:
            print(f"Skipping dim {dim} for {name}: exceeds embedding dim {Z.shape[1]}")
            continue
        Zs = Z[:, :dim]
        erank = compute_effective_rank(Zs)
        ids = compute_global_ids(Zs)
        id_by_dim[dim] = ids
        er_by_dim[dim] = erank
        print(f"{name} dim={dim} effective_rank: {erank}")
        print(f"{name} dim={dim} ids: {ids}")
    return id_by_dim, er_by_dim


def setup_ddp(rank: int, world_size: int) -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_gather_embeddings(z: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1 or not dist.is_initialized():
        return z
    local_n = torch.tensor([z.shape[0]], device=z.device)
    sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)
    if z.shape[0] < max_n:
        pad = torch.zeros((max_n - z.shape[0], z.shape[1]), device=z.device, dtype=z.dtype)
        z = torch.cat([z, pad], dim=0)
    gathered = [torch.zeros((max_n, z.shape[1]), device=z.device, dtype=z.dtype) for _ in range(world_size)]
    dist.all_gather(gathered, z)
    out = []
    for g, n in zip(gathered, sizes):
        out.append(g[:n])
    return torch.cat(out, dim=0)


def ddp_worker(rank: int, world_size: int, args) -> None:
    use_ddp = world_size > 1
    if use_ddp:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    is_croma = bool(args.model_preset and args.model_preset.lower() == "croma")

    if args.model_preset:
        spec = build_non_ciip_spec(args.model_preset, args.preprocess)
        tier = spec.s2_tier or ("s2l2a" if spec.in_chans == 12 else "s2l1c")
    else:
        tier = "s2l2a" if args.model_in_chans == 12 else "s2l1c"
        spec = ModelSpec(
            name="CIIP Matryoshka (epoch 100)",
            in_chans=args.model_in_chans,
            builder=lambda: build_model_from_checkpoint(args.checkpoint)[0],
            feature_fn=ciip_s2_feature_fn,
            transform=build_preprocess_transform(args.preprocess, tier),
        )

    with suppress_output():
        loader = build_ssl4eo_val_loader(
            root=args.val_root,
            in_chans=spec.in_chans,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            return_all_timestamps=not args.no_all_timestamps,
            distributed=use_ddp,
            rank=rank,
            world_size=world_size,
            s2_tier=spec.s2_tier,
        )

    with suppress_output():
        model = spec.builder().to(device)
    for p in model.parameters():
        p.requires_grad = False
    if use_ddp and any(p.requires_grad for p in model.parameters()):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        if rank == 0:
            print("Skipping DDP wrapping: model has no trainable parameters.")
    if not args.model_preset:
        core = _unwrap_model(model)
        if hasattr(core, "encoder_s2") and hasattr(core.encoder_s2, "fc"):
            core.encoder_s2.fc = nn.Identity()
        if hasattr(core, "encoder_s1") and hasattr(core.encoder_s1, "fc"):
            core.encoder_s1.fc = nn.Identity()
        if hasattr(core, "encoder_s2") and hasattr(core.encoder_s2, "proj"):
            proj = getattr(core.encoder_s2, "proj", None)
            if isinstance(proj, nn.Module):
                core.encoder_s2.proj = nn.Identity()
            else:
                core.encoder_s2.proj = None
        if hasattr(core, "encoder_s1") and hasattr(core.encoder_s1, "proj"):
            proj = getattr(core.encoder_s1, "proj", None)
            if isinstance(proj, nn.Module):
                core.encoder_s1.proj = nn.Identity()
            else:
                core.encoder_s1.proj = None

    Z_local = extract_embeddings(
        model=model,
        feature_fn=spec.feature_fn,
        transform=spec.transform,
        loader=loader,
        device=device,
        max_batches=args.max_batches,
        in_chans=spec.in_chans,
    )

    Z_local = torch.from_numpy(Z_local).to(device)
    if use_ddp:
        Z_full = all_gather_embeddings(Z_local, world_size)
    else:
        Z_full = Z_local
    if rank == 0:
        Z_full = Z_full.cpu().numpy()
        full_dim = Z_full.shape[1]
        dims_for_eval: List[int] = [full_dim]
        if args.matryoshka_dims:
            for dim in args.matryoshka_dims:
                if dim != full_dim:
                    dims_for_eval.append(dim)
        dims_for_eval = sorted(set(dims_for_eval))
        time_factor = 1 if args.no_all_timestamps else 4
        effective_images = args.batch_size * 64 * time_factor * world_size
        cache_preprocess = "cromanormalize" if is_croma else args.preprocess
        bs_line = (
            f"batch_size={args.batch_size}, patches_per_file=64, "
            f"time_factor={time_factor}, world_size={world_size}, "
            f"effective_images_per_step={effective_images}, preprocess={cache_preprocess}"
        )
        print(f"Extracted {Z_full.shape[0]} embeddings with dim {Z_full.shape[1]}")
        print(f"full: {bs_line}")
        # Plotting/output dir is also used for caching.
        plots_dir = Path(args.plots_dir) if args.plots_dir else (Path(args.output).parent if args.output else Path.cwd())
        plots_dir.mkdir(parents=True, exist_ok=True)
        cache_tag = args.model_preset or args.checkpoint.stem
        cache_dir = Path("/home/juro4948/ciip/diagnostics/global_id_table1/id_downstream")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"id_metrics_cache_{cache_tag}_{cache_preprocess}.json"
        cache_meta = {
            "checkpoint": str(args.checkpoint),
            "model_preset": args.model_preset,
            "preprocess": cache_preprocess,
            "model_in_chans": spec.in_chans,
            "no_all_timestamps": args.no_all_timestamps,
            "no_downstream": args.no_downstream,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "world_size": world_size,
            "val_root": str(args.val_root),
            "pangaea_model": None if args.model_preset else _pangaea_model_for_checkpoint(args.checkpoint),
            "pangaea_csv_mtime": (
                None
                if args.model_preset
                else (PANGAEA_RESULTS_CSV.stat().st_mtime if PANGAEA_RESULTS_CSV.exists() else None)
            ),
        }
        cache_payload = _load_id_cache(cache_path)
        use_cache = False
        if isinstance(cache_payload, dict) and _cache_meta_matches(cache_payload, cache_meta, dims_for_eval):
            use_cache = True
        else:
            matched_path, matched_payload = _find_matching_cache(
                cache_dir, cache_tag, cache_preprocess, cache_meta, dims_for_eval
            )
            if matched_payload is not None:
                cache_path = matched_path if matched_path is not None else cache_path
                cache_payload = matched_payload
                use_cache = True
        if use_cache:
            id_results = {
                int(k): v for k, v in cache_payload.get("id_results", {}).items()
            }
            er_results = {
                int(k): v for k, v in cache_payload.get("er_results", {}).items()
            }
            print(f"Loaded cached ID metrics from {cache_path}")
        else:
            id_results = {}
            er_results = {}
        performance_results: Dict[int, Dict[str, float]] = {}
        performance_by_preprocess: Dict[str, Dict[int, Dict[str, float]]] = {}
        if use_cache:
            cached_perf = cache_payload.get("performance_results", {})
            performance_results = {int(k): v for k, v in cached_perf.items()}
            cached_perf_by_preprocess = cache_payload.get("performance_by_preprocess", {})
            performance_by_preprocess = {
                preprocess: {int(k): v for k, v in per_dim.items()}
                for preprocess, per_dim in cached_perf_by_preprocess.items()
            }
        pangaea_task_names: List[str] = []
        dims_for_unified = args.matryoshka_dims if args.matryoshka_dims else [full_dim]

        if not use_cache:
            for dim in dims_for_eval:
                Zs = Z_full[:, :dim]
                effective_rank_full = compute_effective_rank(Zs)
                results = compute_global_ids(Zs)
                id_results[dim] = {"ssl4eo_val": results}
                er_results[dim] = {"ssl4eo_val": effective_rank_full}
                print(f"dim_{dim} effective_rank: {effective_rank_full}")
                print(results)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for dim in dims_for_eval:
                    f.write(f"dim_{dim}: {bs_line}\n")
                    f.write(f"  effective_rank: {er_results[dim]['ssl4eo_val']}\n")
                    for k, v in id_results[dim]["ssl4eo_val"].items():
                        f.write(f"  {k}: {v}\n")
            print(f"Wrote results to {out_path}")
        if args.report_unified_results:
            print("Unified eval lookup:")
            preprocess_options = ["scaled", "bandwise"]
            vit_variants = [args.vit_results]
            if args.vit_results == "auto":
                vit_variants = ["meanpool", "cls"]
            model_name_override = args.model_preset
            results_root = (
                Path("/home/juro4948/ciip/diagnostics/unified_eval")
                if model_name_override
                else Path("/home/juro4948/ciip/diagnostics/unified_eval/ciip_checkpoint")
            )
            if model_name_override:
                vit_variants = ["none"]
            for preprocess in preprocess_options:
                for dim in dims_for_unified:
                    for vit_variant in vit_variants:
                        unified = get_unified_eval_results(
                            checkpoint=args.checkpoint,
                            preprocess=preprocess,
                            vit_results=vit_variant,
                            matryoshka_dim=(dim if args.matryoshka_dims else None),
                            root=results_root,
                            model_name_override=model_name_override,
                        )
                        print(f"  preprocess={preprocess}, dim={dim}, vit={vit_variant}:")
                        if not unified:
                            print("    No matching unified_eval artifacts found.")
                            continue
                        for key, payload in unified.items():
                            print(f"    {key}: {payload}")
                        if "neuco_bench" in unified:
                            task_results = unified["neuco_bench"].get("task_results", {})
                            for task, entry in task_results.items():
                                if task in neuco_loader.EXCLUDED_TASKS:
                                    continue
                                score = float(entry.get("raw_score", float("nan")))
                                task_key = f"neuco_{task}"
                                if args.vit_results == "auto" and not model_name_override:
                                    task_key = f"{task_key}[{vit_variant}]"
                                performance_by_preprocess.setdefault(preprocess, {}).setdefault(dim, {})[
                                    task_key
                                ] = score
                                if preprocess == args.preprocess and (args.vit_results != "auto" or model_name_override):
                                    performance_results.setdefault(dim, {})[task_key] = score
                        if "eurosat_linear_probe_0.1" in unified:
                            score = float(
                                unified["eurosat_linear_probe_0.1"].get("test_accuracy", float("nan"))
                            )
                            task_key = "eurosat_lp_0.1"
                            if args.vit_results == "auto" and not model_name_override:
                                task_key = f"{task_key}[{vit_variant}]"
                            performance_by_preprocess.setdefault(preprocess, {}).setdefault(dim, {})[
                                task_key
                            ] = score
                            if preprocess == args.preprocess and (args.vit_results != "auto" or model_name_override):
                                performance_results.setdefault(dim, {})[task_key] = score
                        if "eurosat_knn_0.1" in unified:
                            score = float(
                                unified["eurosat_knn_0.1"].get("test_accuracy", float("nan"))
                            )
                            task_key = "eurosat_knn_0.1"
                            if args.vit_results == "auto" and not model_name_override:
                                task_key = f"{task_key}[{vit_variant}]"
                            performance_by_preprocess.setdefault(preprocess, {}).setdefault(dim, {})[
                                task_key
                            ] = score
                            if preprocess == args.preprocess and (args.vit_results != "auto" or model_name_override):
                                performance_results.setdefault(dim, {})[task_key] = score

        if not args.model_preset:
            pangaea_model = _pangaea_model_for_checkpoint(args.checkpoint)
            if pangaea_model:
                pangaea_scores = _load_pangaea_scores(PANGAEA_RESULTS_CSV, pangaea_model)
                for preprocess, task_scores in pangaea_scores.items():
                    for dim in [full_dim]:
                        for task_key, score in task_scores.items():
                            performance_by_preprocess.setdefault(preprocess, {}).setdefault(dim, {})[
                                task_key
                            ] = score
                            if preprocess == args.preprocess and args.vit_results != "auto":
                                performance_results.setdefault(dim, {})[task_key] = score
                if args.preprocess in pangaea_scores:
                    pangaea_task_names = sorted(pangaea_scores[args.preprocess].keys())
        if not args.no_downstream and not use_cache:
            downstream_model = model.module if hasattr(model, "module") else model
            downstream_device = torch.device("cuda:0")
            if is_croma:
                eurosat_transform = spec.transform
            elif args.preprocess == "bandwise":
                e_mean, e_std = eurosat_bandwise_stats(spec.in_chans)
                eurosat_transform = build_preprocess_transform(
                    args.preprocess, tier, mean=e_mean, std=e_std
                )
            else:
                eurosat_transform = build_preprocess_transform(args.preprocess, tier)
            if spec.in_chans == 3:
                if spec.transform is not None:
                    neuco_transform = T.Compose([SelectRGB(), spec.transform])
                else:
                    neuco_transform = SelectRGB()
            else:
                neuco_transform = spec.transform
            # NeuCo-Bench tasks
            tasks = neuco_loader.discover_tasks(args.neuco_labels_root)
            for task in tasks:
                dataset = NeucoTaskDataset(
                    task.path,
                    args.neuco_data_root,
                    args.neuco_modality,
                    args.neuco_temporal_mode,
                    args.neuco_temporal_seed,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=args.downstream_batch_size,
                    shuffle=False,
                    num_workers=args.downstream_num_workers,
                    pin_memory=True,
                    drop_last=False,
                )
                id_by_dim, er_by_dim = compute_dataset_id_er_multi(
                    name=f"neuco_{task.name}",
                    loader=loader,
                    model=downstream_model,
                    feature_fn=spec.feature_fn,
                    transform=neuco_transform,
                    device=downstream_device,
                    in_chans=spec.in_chans,
                    dims=dims_for_eval,
                    max_batches=args.max_batches,
                )
                for dim, vals in id_by_dim.items():
                    id_results.setdefault(dim, {})[f"neuco_{task.name}"] = vals
                for dim, val in er_by_dim.items():
                    er_results.setdefault(dim, {})[f"neuco_{task.name}"] = val
            # EuroSAT train split by default
            eurosat_dataset = neuco_loader.load_eurosat_dataset(
                root=args.eurosat_root,
                split=args.eurosat_split,
                num_channels=spec.in_chans,
            )
            eurosat_loader = DataLoader(
                eurosat_dataset,
                batch_size=args.downstream_batch_size,
                shuffle=False,
                num_workers=args.downstream_num_workers,
                pin_memory=True,
                drop_last=False,
            )
            id_by_dim, er_by_dim = compute_dataset_id_er_multi(
                name=f"eurosat_{args.eurosat_split}",
                loader=eurosat_loader,
                model=downstream_model,
                feature_fn=spec.feature_fn,
                transform=eurosat_transform,
                device=downstream_device,
                in_chans=spec.in_chans,
                dims=dims_for_eval,
                max_batches=args.max_batches,
            )
            for dim, vals in id_by_dim.items():
                id_results.setdefault(dim, {})[f"eurosat_{args.eurosat_split}"] = vals
            for dim, val in er_by_dim.items():
                er_results.setdefault(dim, {})[f"eurosat_{args.eurosat_split}"] = val

            if not args.model_preset:
                pangaea_model = _pangaea_model_for_checkpoint(args.checkpoint)
                if pangaea_model:
                    encoder_name = PANGAEA_MODEL_ENCODER.get(pangaea_model)
                    if not encoder_name:
                        print(f"Missing Pangaea encoder mapping for model {pangaea_model}")
                    else:
                        encoder_cfg = _load_pangaea_encoder_cfg(encoder_name)
                        modality = next(iter((encoder_cfg.get("input_bands") or {"optical": []}).keys()))
                        pangaea_items = _pangaea_datasets_for_model(
                            PANGAEA_RESULTS_CSV, pangaea_model, args.preprocess
                        )
                        for dataset_name, preprocess_name in pangaea_items:
                            dataset = _build_pangaea_dataset(
                                dataset_name,
                                preprocess_name,
                                encoder_cfg,
                                split="train",
                            )
                            loader = _build_pangaea_loader(
                                dataset,
                                encoder_cfg,
                                batch_size=args.downstream_batch_size,
                                num_workers=args.downstream_num_workers,
                            )
                            id_by_dim, er_by_dim = compute_pangaea_id_er_multi(
                                name=f"pangaea_{dataset_name}",
                                loader=loader,
                                model=downstream_model,
                                feature_fn=ciip_s2_meanpool_backbone_fn,
                                device=downstream_device,
                                dims=dims_for_eval,
                                max_batches=args.max_batches,
                                modality=modality,
                            )
                            for dim, vals in id_by_dim.items():
                                id_results.setdefault(dim, {})[f"pangaea_{dataset_name}"] = vals
                            for dim, val in er_by_dim.items():
                                er_results.setdefault(dim, {})[f"pangaea_{dataset_name}"] = val

        if not use_cache:
            cache_payload = _serialize_id_cache(
                id_results,
                er_results,
                dims_for_eval=dims_for_eval,
                meta=cache_meta,
                performance_results=performance_results,
                performance_by_preprocess=performance_by_preprocess,
            )
            write_path = cache_path
            if write_path.exists():
                unique_path = _unique_cache_path(write_path)
                print(f"Warning: cache file exists at {write_path}; writing to {unique_path}")
                write_path = unique_path
            write_path.write_text(json.dumps(cache_payload, indent=2))
            print(f"Saved ID metrics cache to {write_path}")

        # Plotting
        eurosat_base_key = f"eurosat_{args.eurosat_split}"
        eurosat_alias_keys = ["eurosat_knn_0.1", "eurosat_lp_0.1"]
        _alias_task_metrics(id_results, er_results, eurosat_base_key, eurosat_alias_keys)
        # task order
        task_names = ["ssl4eo_val"]
        task_names += [f"neuco_{t.name}" for t in neuco_loader.discover_tasks(args.neuco_labels_root)]
        task_names += [t for t in eurosat_alias_keys if t not in task_names]
        if eurosat_base_key not in task_names:
            task_names.append(eurosat_base_key)
        if pangaea_task_names:
            task_names += [t for t in pangaea_task_names if t not in task_names]
        _plot_id_and_er_metrics_grid(
            metrics=id_results,
            eranks=er_results,
            tasks=task_names,
            out_path=plots_dir / "intrinsic_dimension_metrics_by_task.png",
        )
        print(f"Saved plot: {plots_dir / 'intrinsic_dimension_metrics_by_task.png'}")
        # performance plot (omit ssl4eo_val)
        perf_tasks = [t for t in task_names if t not in ("ssl4eo_val", f"eurosat_{args.eurosat_split}")]
        if pangaea_task_names:
            perf_tasks.extend([t for t in pangaea_task_names if t not in perf_tasks])
        if performance_results:
            _plot_performance_by_task(
                performance=performance_results,
                tasks=perf_tasks,
                out_path=plots_dir / "downstream_performance_by_task.png",
            )
            print(f"Saved plot: {plots_dir / 'downstream_performance_by_task.png'}")
        else:
            print("Skipping performance plot: no unified eval results found.")

        if performance_by_preprocess:
            _plot_performance_vs_id(
                performance_by_preprocess=performance_by_preprocess,
                eranks=er_results,
                id_metrics=id_results,
                out_path=plots_dir / "downstream_performance_vs_intrinsic_dimension.png",
                eurosat_task_key=f"eurosat_{args.eurosat_split}",
            )
            print(f"Saved plot: {plots_dir / 'downstream_performance_vs_intrinsic_dimension.png'}")
        print(f"Outputs directory: {plots_dir}")

    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser(description="Compute intrinsic dimension for CIIP on SSL4EO v1.1 val.")
    parser.add_argument("--val-root", type=Path, default=Path("/local/ms-data/SSL4EOv1.1/val"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/local/ms-data/SSL4EO/model/2025_12_28-20_16_37-model_resnet50-lr_0.001-b_2-j_6-p_amp_bfloat16/checkpoints/epoch_100.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-all-timestamps", action="store_true", help="Disable 4-season averaging.")
    parser.add_argument("--world-size", type=int, default=min(2, torch.cuda.device_count()))
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Force single-GPU execution (disables DDP).",
    )
    parser.add_argument(
        "--vit-results",
        choices=("auto", "cls", "meanpool"),
        default="auto",
        help="Which ViT unified_eval results to read (auto prefers _meanpool if present).",
    )
    parser.add_argument(
        "--model-preset",
        choices=(
            "resnet18_s2_all_moco",
            "resnet50_s2_all_moco",
            "resnet18_s2_rgb_moco",
            "resnet50_s2_rgb_moco",
            "dino",
            "scalemae_large_rgb",
            "dofa_base_s2_13ch",
            "croma",
            "vitsmall16_s2_all_moco",
            "resnet152_imagenet_rgb",
            "llama3_ms_clip_base",
            "remoteclip",
            "rcf_13ch",
        ),
        default=None,
        help="Use a non-CIIP model preset instead of loading a CIIP checkpoint.",
    )
    parser.add_argument(
        "--model-in-chans",
        type=int,
        default=12,
        choices=(12, 13),
        help="Sentinel-2 input channels (12 for S2L2A, 13 for S2L1C).",
    )
    parser.add_argument(
        "--report-unified-results",
        action="store_true",
        default=True,
        help="Print EuroSAT kNN/LP and NeuCo-Bench results from unified_eval outputs.",
    )
    parser.add_argument(
        "--no-downstream",
        action="store_true",
        help="Skip downstream dataset ID/effective-rank computation.",
    )
    parser.add_argument("--downstream-batch-size", type=int, default=32)
    parser.add_argument("--downstream-num-workers", type=int, default=0)
    parser.add_argument("--neuco-labels-root", type=Path, default=Path("/local/ms-data/SSL4EO-S12-downstream/labels"))
    parser.add_argument("--neuco-data-root", type=Path, default=Path("/local/ms-data/SSL4EO-S12-downstream/data"))
    parser.add_argument("--neuco-modality", choices=("s2l2a", "s2l1c", "s1"), default="s2l2a")
    parser.add_argument(
        "--neuco-temporal-mode",
        choices=("embed-mean", "random-one"),
        default="embed-mean",
        help="How to handle NeuCo time dimension: mean embeddings across time or sample one season.",
    )
    parser.add_argument("--neuco-temporal-seed", type=int, default=0)
    parser.add_argument("--eurosat-root", type=Path, default=Path("/local/ms-data/EuroSAT"))
    parser.add_argument("--eurosat-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--plots-dir", type=str, default=None, help="Directory to save plots.")
    parser.add_argument(
        "--matryoshka",
        action="store_true",
        help="Evaluate matryoshka slices at default dimensions.",
    )
    parser.add_argument(
        "--matryoshka-dims",
        type=int,
        nargs="+",
        default=None,
        help="Override matryoshka slice dimensions.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional output text file.")
    args = parser.parse_args()
    args.preprocess = DEFAULT_PREPROCESS
    if args.matryoshka:
        dims = list(DEFAULT_MATRYOSHKA_DIMS)
        if args.matryoshka_dims:
            dims = args.matryoshka_dims
        args.matryoshka_dims = tuple(dims)
    return args


def main() -> None:
    args = parse_args()
    if args.single_gpu:
        args.world_size = 1
    if args.world_size <= 1:
        ddp_worker(0, 1, args)
        return
    mp.set_start_method("spawn", force=True)
    mp.spawn(ddp_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
