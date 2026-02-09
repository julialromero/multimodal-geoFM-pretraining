"""Shared normalization helpers for evaluation and diagnostics."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

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
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NORMALIZATION_METHOD_DIVIDE = "divideby10000"
NORMALIZATION_METHOD_BANDWISE = "bandwisenorm"
NORMALIZATION_METHOD_SSL4EO = "ssl4eonorm"
NORMALIZATION_METHOD_IMAGENET = "imagenet"
NORMALIZATION_METHODS = (
    NORMALIZATION_METHOD_DIVIDE,
    NORMALIZATION_METHOD_BANDWISE,
    NORMALIZATION_METHOD_SSL4EO,
    NORMALIZATION_METHOD_IMAGENET,
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


class CromaNormalize(nn.Module):
    def __init__(self, use_8_bit: bool = False):
        super().__init__()
        self.use_8_bit = use_8_bit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        squeeze_batch = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.ndim != 4:
            raise ValueError(f"Expected CxHxW or NxCxHxW input, got {x.shape}")

        imgs = []
        for channel in range(x.shape[1]):
            min_value = x[:, channel, :, :].mean() - 2 * x[:, channel, :, :].std()
            max_value = x[:, channel, :, :].mean() + 2 * x[:, channel, :, :].std()
            denom = max_value - min_value
            if denom == 0:
                denom = torch.tensor(1.0, device=x.device, dtype=x.dtype)

            if self.use_8_bit:
                img = (x[:, channel, :, :] - min_value) / denom * 255.0
                img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)
                imgs.append(img)
            else:
                img = (x[:, channel, :, :] - min_value) / denom
                img = torch.clip(img, 0, 1).unsqueeze(dim=1)
                imgs.append(img)

        output = torch.cat(imgs, dim=1)
        if squeeze_batch:
            output = output.squeeze(0)
        return output


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
    return T.Compose([T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


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
    return Divideby10000Normalize()


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
            build_imagenet_normalization(),
        ]
    ),
    "moco": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
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
            build_imagenet_normalization(),
        ]
    ),
    "resnet50_s2_rgb_moco": T.Compose(
        [
            T.CenterCrop((224, 224)),
            build_imagenet_normalization(),
        ]
    ),
    "resnet152_imagenet_rgb": T.Compose(
        [
            T.CenterCrop((224, 224)),
            build_imagenet_normalization(),
        ]
    ),
    "vitsmall16_s2_all_moco": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    "remoteclip": T.Compose(
        [
            T.CenterCrop((224, 224)),
            build_imagenet_normalization(),
        ]
    ),
    "llama3_ms_clip_base": T.Compose(
        [
            T.CenterCrop((224, 224)),
            T.Normalize(
                mean=[
                    1924.863,
                    2184.553,
                    2340.936,
                    2671.402,
                    3240.082,
                    3468.412,
                    3563.244,
                    3627.704,
                    3416.714,
                    2849.625,
                ],
                std=[
                    1201.092,
                    1219.943,
                    1397.225,
                    1400.035,
                    1373.136,
                    1429.17,
                    1485.025,
                    1447.836,
                    1471.002,
                    1365.307,
                ],
            ),
        ]
    ),
    "rcf_13ch": T.Compose(
        [
            T.CenterCrop((224, 224)),
            S2ScaleTransform(scale=10000.0),
        ]
    ),
    "croma": T.Compose(
        [
            T.Resize((120, 120)),
            CromaNormalize(use_8_bit=False),
        ]
    ),
    "dino": T.Compose(
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
