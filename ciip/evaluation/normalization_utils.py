"""Shared normalization helpers for evaluation and diagnostics."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T

from ciip.open_clip_train.data import (
    S2L1C_MEAN,
    S2L1C_STD,
    S2L2A_MEAN,
    S2L2A_STD,
    S2RGB_MEAN,
    S2RGB_STD,
)

# Reusable normalization constants for ImageNet-pretrained models
IMAGENET_MEAN = [0.481, 0.457, 0.408]
IMAGENET_STD = [0.269, 0.261, 0.275]

NORMALIZATION_METHOD_DIVIDE = "divideby10000"
NORMALIZATION_METHOD_BANDWISE = "bandwisenorm"
NORMALIZATION_METHOD_SSL4EO = "ssl4eonorm"
NORMALIZATION_METHOD_IMAGENET = "imagenet"
NORMALIZATION_METHOD_GALILEO = "galileo"
NORMALIZATION_METHODS = (
    NORMALIZATION_METHOD_DIVIDE,
    NORMALIZATION_METHOD_BANDWISE,
    NORMALIZATION_METHOD_SSL4EO,
    NORMALIZATION_METHOD_IMAGENET,
    NORMALIZATION_METHOD_GALILEO,
)
DEFAULT_NORMALIZATION_METHOD = NORMALIZATION_METHOD_DIVIDE

_SSL4EO_ALIAS = "ssl4eobandwisenorm"

_S2L2A_KEEP_10 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

IMAGENET_MODEL_WEIGHTS = {
    "scalemae_large_rgb",
    "resnet18_s2_rgb_moco",
    "resnet50_s2_rgb_moco",
    "resnet152_imagenet_rgb",
    "remoteclip",
}

GALILEO_S2_MEAN = [
    1395.3408730676722,
    1395.3408730676722,
    1338.4026921784578,
    1343.09883810357,
    1543.8607982512297,
    2186.2022069512263,
    2525.0932853316694,
    2410.3377187373408,
    2750.2854646886753,
    2750.2854646886753,
    2234.911100061487,
    2234.911100061487,
    1474.5311266077113,
]
GALILEO_S2_STD = [
    917.7041440370853,
    917.7041440370853,
    913.2988423581528,
    1092.678723527555,
    1047.2206083460424,
    1048.0101611156767,
    1143.6903026819996,
    1098.979177731649,
    1204.472755085893,
    1204.472755085893,
    1145.9774063078878,
    1145.9774063078878,
    980.2429840007796,
]

TERRAMIND_S2L2A_MEAN = [
    1390.458,
    1503.317,
    1718.197,
    1853.910,
    2199.100,
    2779.975,
    2987.011,
    3083.234,
    3132.220,
    3162.988,
    2424.884,
    1857.648,
]
TERRAMIND_S2L2A_STD = [
    2106.761,
    2141.107,
    2038.973,
    2134.138,
    2085.321,
    1889.926,
    1820.257,
    1871.918,
    1753.829,
    1797.379,
    1434.261,
    1334.311,
]

MODALITY_STATS: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {
    "s2l1c": (S2L1C_MEAN, S2L1C_STD),
    "s2l2a": (S2L2A_MEAN, S2L2A_STD),
    "s1": ([-12.577, -20.265], [5.179, 5.872]),
    "s2l2a_rgb": (S2RGB_MEAN, S2RGB_STD),
    "s2l2a_10": ([S2L2A_MEAN[i] for i in _S2L2A_KEEP_10], [S2L2A_STD[i] for i in _S2L2A_KEEP_10]),
}

NEUCO_MODALITY_STATS: Dict[str, Tuple[Sequence[float], Sequence[float]]] = {
    "s2l1c": (S2L1C_MEAN, S2L1C_STD),
    "s2l2a": (S2L2A_MEAN, S2L2A_STD),
    "s1": ([-12.577, -20.265], [5.179, 5.872]),
    "s2l2a_rgb": (S2RGB_MEAN, S2RGB_STD),
}


class Divideby10000Normalize:
    """Scales Sentinel-2 pixel values to [0, 1] by dividing by 10000."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img.float() / 10000.0


class SSL4EONormalize:
    """Normalize Sentinel-2/Sentinel-1 tensors using SSL4EO statistics."""

    def _infer_modality(self, num_channels: int) -> str:
        modality_map = {
            13: "s2l1c",
            12: "s2l2a",
            10: "s2l2a_10",
            3: "s2l2a_rgb",
            2: "s1",
        }
        if num_channels not in modality_map:
            raise ValueError(f"Cannot infer modality from number of channels: {num_channels}")
        return modality_map[num_channels]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        channels = img.shape[-3]
        modality = self._infer_modality(channels)
        mean, std = MODALITY_STATS[modality]
        mean_t = torch.as_tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        std_t = torch.as_tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        return (img.float() - mean_t) / std_t


class NeuCoNormalize:
    """Normalize NeuCo tensors using NeuCo challenge statistics."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        channels = img.shape[-3]
        if channels == 13:
            modality = "s2l1c"
        elif channels == 12:
            modality = "s2l2a"
        elif channels == 3:
            modality = "s2l2a_rgb"
        elif channels == 2:
            modality = "s1"
        else:
            raise ValueError(f"Cannot infer modality from number of channels: {channels}")
        mean, std = NEUCO_MODALITY_STATS[modality]
        mean_t = torch.as_tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        std_t = torch.as_tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        return (img.float() - mean_t) / std_t


class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        if torch.is_tensor(x):
            if x.ndim == 4:
                mean_t = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
                std_t = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            elif x.ndim == 3:
                mean_t = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
                std_t = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
            else:
                raise ValueError(f"Expected CxHxW or NxCxHxW input, got {x.shape}")
            min_value = mean_t - 2 * std_t
            max_value = mean_t + 2 * std_t
            denom = max_value - min_value
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            img = (x - min_value) / denom * 255.0
            img = torch.clamp(img, 0, 255).to(torch.uint8)
            return img

        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class CromaNormalize(nn.Module):
    def __init__(
        self,
        size: int | Tuple[int, int] = (120, 120),
        interpol_mode: T.InterpolationMode = T.InterpolationMode.BICUBIC,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        if mean is None or std is None:
            mean, std = MODALITY_STATS["s2l2a"]
        self.normalize = SentinelNormalize(mean, std)
        self.resize = T.Resize(size, interpolation=interpol_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.ndim != 4:
            raise ValueError(f"Expected CxHxW or NxCxHxW input, got {x.shape}")

        # SentinelNormalize -> ToTensor -> Resize (CROMA expects 120x120)
        x = self.normalize(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float().div(255.0)
        x = self.resize(x)

        if squeeze_batch:
            x = x.squeeze(0)
        return x

class SSL4EOTransform(nn.Module):
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        size: int | Tuple[int, int] = (224, 224),
        interpol_mode: T.InterpolationMode = T.InterpolationMode.BICUBIC,
    ):
        super().__init__()
        # 1. Initialize the existing normalization class
        self.normalize = SentinelNormalize(mean, std)
        
        # 2. Setup spatial resize
        if isinstance(size, int):
            size = (size, size)
        self.resize = T.Resize(size, interpolation=interpol_mode)

    def forward(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        # Convert NumPy to Tensor immediately if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Handle batching (T.Resize expects at least 3 dims [C, H, W] or 4 dims [N, C, H, W])
        squeeze_batch = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True

        # Stage 1: Standardize + 2-sigma clipping (Returns uint8 [0, 255])
        # This calls the SentinelNormalize.__call__ logic provided
        x = self.normalize(x)

        # Stage 2: Convert to float and scale to [0, 1]
        # This is the "ToTensor" equivalent scaling
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float().div(255.0)

        # Stage 3: Spatial Resize to 120x120
        x = self.resize(x)

        if squeeze_batch:
            x = x.squeeze(0)
            
        return x
    



class S2ScaleTransform(nn.Module):
    """Scale Sentinel-2 pixel values by a constant factor."""

    def __init__(self, scale: float = 10000.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.scale
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


def resolve_normalization_method(
    model_in_channels: Optional[int],
    normalization_method: Optional[str],
) -> str:
    if model_in_channels == 3:
        return NORMALIZATION_METHOD_IMAGENET #NORMALIZATION_METHOD_BANDWISE
    normalized = (normalization_method or "").strip().lower()
    if normalized == _SSL4EO_ALIAS:
        return NORMALIZATION_METHOD_SSL4EO
    if normalized:
        return normalized
    return DEFAULT_NORMALIZATION_METHOD


def resolve_normalization_method_for_weights(
    model_in_channels: Optional[int],
    normalization_method: Optional[str],
    model_weights: Optional[str],
) -> str:
    method = resolve_normalization_method(model_in_channels, normalization_method)
    if (model_weights or "").lower() in IMAGENET_MODEL_WEIGHTS:
        return NORMALIZATION_METHOD_IMAGENET
    return method


def build_imagenet_normalization() -> T.Compose:
    return T.Compose([S2ScaleTransform(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def build_normalization_transform(
    method: str,
    *,
    bandwise_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    normalized = method.lower()
    if normalized == NORMALIZATION_METHOD_IMAGENET:
        return build_imagenet_normalization()
    if normalized == NORMALIZATION_METHOD_BANDWISE:
        raise ValueError("Bandwise normalization requires mean/std stats, which are not provided by default.")
        if bandwise_stats is None:
            raise ValueError("bandwise normalization requires mean/std stats")
        mean, std = bandwise_stats
        return T.Normalize(mean=mean, std=std)
    if normalized == NORMALIZATION_METHOD_SSL4EO:
        return SSL4EONormalize()
    if normalized == NORMALIZATION_METHOD_GALILEO:
        return T.Normalize(mean=GALILEO_S2_MEAN, std=GALILEO_S2_STD)
    return Divideby10000Normalize()


ssl4eo_transform_13 = SSL4EOTransform(mean=MODALITY_STATS['s2l1c'][0], std=MODALITY_STATS['s2l1c'][1], size=(224, 224))
ssl4eo_transform_12 = SSL4EOTransform(mean=MODALITY_STATS['s2l2a'][0], std=MODALITY_STATS['s2l2a'][1], size=(224, 224))
ssl4eo_transform_3 = SSL4EOTransform(mean=MODALITY_STATS['s2l2a_rgb'][0], std=MODALITY_STATS['s2l2a_rgb'][1], size=(224, 224))
ssl4eo_transform_10 = SSL4EOTransform(mean=MODALITY_STATS['s2l2a_10'][0], std=MODALITY_STATS['s2l2a_10'][1], size=(224, 224))

SSL4EO_MODEL_TRANSFORMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # "dofa_base_s2_13ch": T.Compose(
    #     [
    #         T.CenterCrop((224, 224)),
    #         S2ScaleTransform(scale=10000.0),
    #     ]
    # ),
    "scalemae_large_rgb": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            build_imagenet_normalization(),
        ]
    ),
    "moco": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    # ssl4eo_transform_13,
    "vitsmall16_s2_all_dino": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    # "dino": ssl4eo_transform_13,
    # "resnet18_s2_all_moco": ssl4eo_transform_13,
    # "resnet50_s2_all_moco": ssl4eo_transform_13,
    # "resnet18_s2_rgb_moco": ssl4eo_transform_3,
    # "resnet50_s2_rgb_moco": ssl4eo_transform_3,
    # "resnet152_imagenet_rgb": ssl4eo_transform_3,
    "vitsmall16_s2_all_moco": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    # ssl4eo_transform_13,
    "remoteclip": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.CenterCrop((224, 224)),
            build_imagenet_normalization(),
        ]
    ),
    "galileo_s2": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        ]
    ),
    "galileo": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        ]
    ),
    "ssl4eo_mae_optical": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    "terramind_base": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.Normalize(mean=TERRAMIND_S2L2A_MEAN, std=TERRAMIND_S2L2A_STD),
        ]
    ),
    "terramind_large": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.Normalize(mean=TERRAMIND_S2L2A_MEAN, std=TERRAMIND_S2L2A_STD),
        ]
    ),
    "llama3_ms_clip_base": T.Compose(
        [
            SelectS2Channels10(),
            ssl4eo_transform_10,
        ]
    ),
    "rcf_13ch": ssl4eo_transform_13,
    "croma": T.Compose(
        [
            CromaNormalize(),
        ]
    ),
    "dino": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    "ciip_text_s2": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.Normalize(S2L2A_MEAN, S2L2A_STD),
        ]
    ),
    # ssl4eo_transform_12,
    "ciip_bandwise": ssl4eo_transform_12,
    # ssl4eo_transform_12,
    "ciip_10kscale": T.Compose(
        [
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
}


def select_ssl4eo_transform(model_weights: Optional[str]) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Return the SSL4EO transform matching the provided weight key, if any."""

    weight_key = (model_weights or "").lower()
    if weight_key in SSL4EO_MODEL_TRANSFORMS:
        print(f"Using SSL4EO normalization/transform for model weights '{model_weights}'")
    else:
        print(f"No specific SSL4EO normalization/transform found for model weights '{model_weights}'")
        raise ValueError(f"No SSL4EO normalization/transform found for model weights '{model_weights}'")
    return SSL4EO_MODEL_TRANSFORMS.get(weight_key)
