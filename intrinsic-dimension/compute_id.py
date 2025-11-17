from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import skdim.id as id
from torchvision import transforms as T

# --- Your existing evaluation helpers (used only for the data loader) ---
from ciip.evaluation.unified_evaluation import (
    ModelEvalConfig,
    # _build_eurosat_loaders,
    # _resolve_eurosat_bands,
)
from s2geo_dataset import S2Geo
S2_100K_ROOT = Path("/local/ms-data/S2_100K")
from ciip.evaluation.model_utils import build_model_from_checkpoint

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
EUROSAT_ROOT = Path("/local/ms-data/EuroSAT/")
NEUCO_ROOT = Path("/local/ms-data/SSL4EO-S12-downstream/data")
OUTPUT_DIR = Path("/home/juro4948/ciip/diagnostics/global_id_table1")

EUROSAT_IMAGE_SIZE = 224  # CROMA expects 120, we’ll handle that per-model
EUROSAT_MODALITY = "s2"   # used only by your loader helper

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

def build_s2_100k_loader(
    in_chans: int,
    target_size: int,
    batch_size: int = 64,
    num_workers: int = 4,
    rgb_mode: bool = False,
) -> DataLoader:
    """
    Build a DataLoader over SatCLIP's S2-100K using its S2GeoDataset class.
    Yields dicts with keys {"image", "label"} to match the rest of the script.
    """
    mean = [0.4139, 0.4341, 0.3482, 0.5263],
    std = [0.0010, 0.0010, 0.0013, 0.0013]
    transform = T.Compose([
        T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(mean=mean, std=std),
    ])

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
        def __init__(self, base, target_size, in_chans, rgb_mode):
            self.base = base
            self.in_chans = in_chans
            self.rgb_mode = rgb_mode
            self.resize = T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC)

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
                # likely HWC -> CHW
                x = x.permute(2, 0, 1).contiguous()

            # Select channels to match the model spec
            if self.rgb_mode:
                # S2 RGB convention: B4,B3,B2 -> indices 3,2,1 in [B1..B12]; if 13 bands and includes B8A, adjust accordingly
                # Safe fallback: if ≥4 chans, pick [3,2,1]; else assume already RGB
                if x.shape[0] >= 4:
                    x = x[[3, 2, 1], ...]
            else:
                # Expect 12 or 13 bands; if 12 needed (e.g., without B10), drop band index 10 when present
                if self.in_chans == 12 and x.shape[0] == 13:
                    keep = [i for i in range(13) if i != 10]  # drop cirrus B10
                    x = x[keep, ...]
                # If in_chans==13 and x has 12, you may need to pad or adapt—raise early:
                if self.in_chans != x.shape[0]:
                    raise RuntimeError(f"Channel mismatch: wanted {self.in_chans}, got {x.shape[0]}")

            # Resize (expects CHW)
            x = self.resize(x)

            return {"image": x.float(), "label": int(y)}

    ds = _S2GeoWrapped(base, target_size=target_size, in_chans=in_chans, rgb_mode=rgb_mode)

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
        # x: (B, C, H, W) → features: (B, D)
        h = self.conv(x)
        return h.mean(dim=(2, 3))  # global average pooling


# ---------------------------------------------------------------------------
# Model spec / registry
# ---------------------------------------------------------------------------
FeatureFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]


@dataclass
class ModelSpec:
    name: str
    in_chans: int
    rgb_mode: bool  # True means expect 3 channels (B4/B3/B2)
    target_size: int  # spatial size to which images should be resized/cropped
    builder: Callable[[], nn.Module]
    feature_fn: FeatureFn


def _ciip_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """CIIP: use forward_features with Sentinel-2 wavelengths and mean-pool tokens."""
    # x: (B, 13, H, W)
    # feats = model.forward_features(x,)
    # print(model.encode_s2)
    print(model.encoder_s2)
    raise RuntimeError("Debug CIIP feature fn")
    feats = model.encode_s2(x, normalize=False, post_head=False)
    # If CIIP returns patch tokens (B, N, D), mean-pool over N:
    if feats.ndim == 3:
        feats = feats.mean(dim=1)  # (B, D)
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
        print("resnet Using forward_features")
        # 2. Some timm-style or TorchGeo wrappers expose forward_features
        feats = model.forward_features(x)
    else:
        # # 3. Fallback: mimic torchvision/timm ResNet forward up to layer4
        # #    (penultimate conv features, before avgpool + fc).
        # x = model.conv1(x)
        # x = model.bn1(x)
        # x = model.act1(x) if hasattr(model, "act1") else model.relu(x)
        # x = model.maxpool(x)
        # x = model.layer1(x)
        # x = model.layer2(x)
        # x = model.layer3(x)
        # feats = model.layer4(x)
        raise RuntimeError("Model has no backbone or forward_features method.")

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



def _croma_optical_gap_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """CROMA: use optical branch GAP embedding."""
    # x: (B, 12, H, W) with values in [0, 1] at 120x120
    # print device of model and x
    # print(f"CROMA model device: {next(model.parameters()).device}, x device: {x.device}")
    out: Dict[str, torch.Tensor] = model(x_optical=x)
    if "optical_GAP" not in out:
        raise KeyError("CROMA forward did not return 'optical_GAP'.")
    return out["optical_GAP"]  # (B, D)


def _dofa_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """DOFA: use forward_features with Sentinel-2 wavelengths and mean-pool tokens."""
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
    # TorchGeo ScaleMAE usually wraps a ViT-like backbone.
    # Depending on version, forward_features may return:
    #  - (B, N, D): token sequence
    #  - (B, D): already pooled
    #  - dict with a token field (rare but we handle it)

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



# Registry of models to evaluate
def build_model_specs() -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # # 1. Random Conv Filters (RCF) – 13-band, 512-dim features
    # specs.append(
    #     ModelSpec(
    #         name="RCF_13ch",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: RandomConvFeatures(in_chans=13, out_dim=512),
    #         feature_fn=lambda m, x: m(x),
    #     )
    # )

    # # 2. CROMA optical encoder – 12 optical bands, 120x120, [0,1]
    # specs.append(
    #     ModelSpec(
    #         name="CROMA_optical",
    #         in_chans=12,
    #         rgb_mode=False,
    #         target_size=120,
    #         builder=lambda: croma_base(
    #             weights=CROMABase_Weights.CROMA_VIT, modalities=['optical'], #CROMA_BASE_S1S2  # TODO: confirm exact enum
    #         ),
    #         feature_fn=_croma_optical_gap_feature_fn,
    #     )
    # )

    # # from torchgeo.models import CROMABase_Weights
    # # list(CROMABase_Weights)

    # # 3. DOFA base – 13 bands, use pre-trained base weights
    # specs.append(
    #     ModelSpec(
    #         name="DOFA_base_S2_13ch",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: dofa_base_patch16_224(
    #             weights=DOFABase16_Weights.DOFA_MAE,
    #         ),
    #         feature_fn=_dofa_feature_fn,
    #     )
    # )

    # # 4. ScaleMAE Large (RGB)
    # specs.append(
    #     ModelSpec(
    #         name="ScaleMAE_large_RGB",
    #         in_chans=3,
    #         rgb_mode=True,
    #         target_size=224,
    #         builder=lambda: scalemae_large_patch16(
    #             weights=ScaleMAELarge16_Weights.FMOW_RGB
    #         ),
    #         feature_fn=_scalemae_feature_fn,
    #     )
    # )

    # # # 5–8. ResNet baselines (MoCo S2 ALL / RGB + ImageNet)
    # specs.append(
    #     ModelSpec(
    #         name="ResNet18_S2_ALL_MOCO",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )
    # specs.append(
    #     ModelSpec(
    #         name="ResNet50_S2_ALL_MOCO",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )
    # specs.append(
    #     ModelSpec(
    #         name="ResNet18_S2_RGB_MOCO",
    #         in_chans=3,
    #         rgb_mode=True,
    #         target_size=224,
    #         builder=lambda: resnet18(weights=ResNet18_Weights.SENTINEL2_RGB_MOCO),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )
    specs.append(
        ModelSpec(
            name="ResNet50_S2_RGB_MOCO",
            in_chans=3,
            rgb_mode=True,
            target_size=224,
            builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO),
            feature_fn=_resnet_gap_feature_fn,
        )
    )

    # # 9. ResNet152 ImageNet-1K (RGB only)
    # specs.append(
    #     ModelSpec(
    #         name="ResNet152_ImageNet_RGB",
    #         in_chans=3,
    #         rgb_mode=True,
    #         target_size=224,
    #         builder=lambda: resnet152(weights=ResNet152_Weights.IMAGENET1K_V1),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )

    # # 10. ViT-Small Sentinel-2 13-band MoCo
    # specs.append(
    #     ModelSpec(
    #         name="ViTSmall16_S2_ALL_MOCO",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: vit_small_patch16_224(
    #             weights=ViTSmall16_Weights.SENTINEL2_ALL_MOCO
    #         ),
    #         feature_fn=_vit_patch_mean_feature_fn,
    #     )
    # )

 
    # specs.append(
    #     ModelSpec(
    #         name="VanillaCIIP, Epoch10",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint("/local/ms-data/SSL4EO/model/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints/epoch_10.pt")[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="VanillaCIIP, Epoch40",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint("/local/ms-data/SSL4EO/model/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints/epoch_40.pt")[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=1, Epoch10",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/checkpoints/epoch_10.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=0.1, Epoch10",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_07-09_24_18-model_resnet50-lr_0.001-b_128-j_6-p_amp/checkpoints/epoch_10.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="VanillaCIIP, Epoch25",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint("/local/ms-data/SSL4EO/model/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints/epoch_25.pt")[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=1, Epoch24",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/checkpoints/epoch_24.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=0.1, v1.1/12bands, Epoch10",
    #         in_chans=12,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_13-08_33_13-model_resnet50-lr_0.001-b_2-j_6-p_amp/checkpoints/epoch_10.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=0.1, v1.1/12bands, Epoch40",
    #         in_chans=12,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_14-10_56_41-model_resnet50-lr_0.001-b_2-j_6-p_amp/checkpoints/epoch_40.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Random ResNet (CIIP), v1.1/12bands, Epoch0",
    #         in_chans=12,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_13-08_33_13-model_resnet50-lr_0.001-b_2-j_6-p_amp/checkpoints/epoch_0.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Random ResNet (CIIP), Epoch0",
    #         in_chans=13,
    #         rgb_mode=False,
    #         target_size=224,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_10_09-11_04_30-model_resnet50-lr_0.001-b_128-j_6-p_amp/checkpoints/epoch_0.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    return specs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_s2_like_loader(
    in_chans: int,
    target_size: int,
    batch_size: int = 64,
    num_workers: int = 4,
    rgb_mode: bool = False,
) -> DataLoader:
    """
    Reuses your EuroSAT loader as a stand-in for S2-100K.

    IMPORTANT:
    ----------
    Replace this with your own S2-100K / Sentinel-2 dataloader
    once you have that wired up. The only requirement for the
    rest of the script is that the loader yield batches of
    (images, ...) where images are (B, in_chans, H, W) and
    already resized/cropped to `target_size`.
    """
    # config = ModelEvalConfig(
    #     eurosat_root=EUROSAT_ROOT,
    #     neuco_root=NEUCO_ROOT,
    #     output_dir=OUTPUT_DIR,
    #     checkpoint=None,
    #     model_type="dummy",  # unused
    #     model_weights=None,
    #     model_in_channels=in_chans,
    #     eurosat_image_size=target_size,
    #     enable_ssl4eo=False,
    # )

    loader = build_s2_100k_loader(
        in_chans=in_chans,
        target_size=target_size,
        batch_size=64,
        num_workers=4,
        rgb_mode=False,   # propagate whether the model expects RGB
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
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
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
                # print(batch.k
                # eys())
                image, labels = batch['image'], batch['label']
            # print(x)
            image = image.to(device)
            model = model.to(device)

            # print the devices
            # print(f"Image device: {image.device}, Model device: {next(model.parameters()).device}")


            z = feature_fn(model, image)  # (B, D)

            if b_idx == 0:
                print(f"Feature shape per batch: {z.shape}")
            if z.ndim != 2:
                # print model class
                print(type(model))
                raise RuntimeError(f"Expected (B, D) features, got {z.shape}")
            feats_list.append(z.cpu())

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    if not feats_list:
        raise RuntimeError("No features extracted – check your loader.")

    Z = torch.cat(feats_list, dim=0).numpy()
    return Z  # (N, D)


def compute_global_id(Z: np.ndarray) -> float:
    """
    Compute global intrinsic dimension using FisherS (angle-based).
    """
    model = id.FisherS()
    global_id = model.fit_transform(Z)
    return float(global_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_model_specs()

    results: Dict[str, float] = {}

    for spec in specs:
        print(f"\n=== Model: {spec.name} ===")
        # try:
        loader = build_s2_like_loader(
            in_chans=spec.in_chans,
            target_size=spec.target_size,
            batch_size=64,
            num_workers=4,
            rgb_mode=spec.rgb_mode,
        )
        # except Exception as e:
        #     print(f"  [SKIP] Failed to build loader for in_chans={spec.in_chans}: {e}")
        #     continue

        if spec.name.startswith("CROMA"):        # or spec.name == "CROMA_optical"
            device = torch.device("cpu")
        else:
            device = global_device


        # try:

        model = spec.builder().to(device)
        # except Exception as e:
        #     print(f"  [SKIP] Failed to construct model '{spec.name}': {e}")
        #     continue

        with torch.no_grad():
            Z = extract_embeddings_model(
                model=model,
                feature_fn=spec.feature_fn,
                loader=loader,
                device=device,
                max_batches=None,  # or limit for debugging
            )
            # except Exception as e:
            #     print(f"  [SKIP] Failed to extract embeddings for '{spec.name}': {e}")
            #     continue

        print(f"  Extracted {Z.shape[0]} embeddings with dimension {Z.shape[1]}.")

        try:
            gid = compute_global_id(Z)
            results[spec.name] = gid
            print(f"  Global FisherS ID: {gid:.4f}")
        except Exception as e:
            print(f"  [SKIP] Failed to compute ID for '{spec.name}': {e}")

    # Summary
    print("\n=== Summary of Global IDs (FisherS) ===")
    for name, gid in results.items():
        print(f"{name:30s}: {gid:.4f}")

    # saved path
    print(f"\nResults saved to {OUTPUT_DIR / 'global_id_results.txt'}")


if __name__ == "__main__":
    main()
