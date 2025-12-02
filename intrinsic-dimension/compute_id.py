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
import torchvision
from torchvision import transforms as T
from ciip.open_clip_train.data import get_transform
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
# EUROSAT_ROOT = Path("/local/ms-data/EuroSAT/")
# NEUCO_ROOT = Path("/local/ms-data/SSL4EO-S12-downstream/data")
OUTPUT_DIR = Path("/home/juro4948/ciip/diagnostics/global_id_table1")

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
def build_s2_ssl4eo_loader(
    in_chans: int,
    batch_size: int = 64,
    num_workers: int = 4,
    # rgb_mode: bool = False,
) -> DataLoader:
    # if s2_transform is None:
    tier = 's2l1c' if in_chans == 13 else 's2l2a'
        # s2_transform = data.get_transform(tier, is_train=False)

    dataset = SSL4EODataset(
        root=Path("/local/ms-data/SSL4EOv1.1/train"),
        s2_tier=tier,
        seasons=[0,1,2,3],
        num_timestamps=4,
        # s2_bands=list(config.ssl4eo_s2_bands),
        transforms=None, # {'s1': data.get_transform('s1', is_train=False), 's2': s2_transform},
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



def build_s1_ssl4eo_loader(
    batch_size: int = 64,
    num_workers: int = 4,
    # rgb_mode: bool = False,
) -> DataLoader:

    dataset = SSL4EODataset(
        root=Path("/local/ms-data/SSL4EOv1.1/train"),
        s2_tier='s2l1c',
        seasons=[0,1,2,3],
        num_timestamps=4,
        transforms=None, 
        is_train=False,
    )

    # wrap to return dicts with "image" and "label" keys
    class _SSL4EOWrapped(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            item = self.base[idx]
            x = item["s1"] if isinstance(item, dict) and "s1" in item else item[0]
            y = item.get("label", 0) if isinstance(item, dict) else (item[1] if len(item) > 1 else 0)

            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            elif not isinstance(x, torch.Tensor):
                x = T.ToTensor()(x)  

            if x.ndim == 3 and x.shape[0] not in (3, 12, 13):
                x = x.permute(2, 0, 1).contiguous()

            return {"image": x.float(), "label": int(y)}
    
    ds = _SSL4EOWrapped(dataset)

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


def _ciip_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """CIIP: use forward_features with Sentinel-2 wavelengths and mean-pool tokens."""
    feats = model.encoder_s2(x)
    if feats.shape[1] != 2048:
        raise RuntimeError(f"Unexpected CIIP feature dim: {feats.shape}")
    # feats = model.encoder_s1(x)
    # If CIIP returns patch tokens (B, N, D), mean-pool over N:
    if feats.ndim == 3:
        print("CIIP feature tokens shape:", feats.shape)
        feats = feats.mean(dim=1)  # (B, D)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected CIIP feature shape: {feats.shape}")
    return feats

def _ciip_s1_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    feats = model.encoder_s1(x)
    if feats.shape[1] != 2048:
        raise RuntimeError(f"Unexpected CIIP feature dim: {feats.shape}")
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
        # raise RuntimeError("Model has no backbone or forward_features method.")

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
        # if "SAR_GAP" not in out:
        #     raise RuntimeError(f"CROMA output has no optical_GAP or sar_GAP keys: {out.keys()}")
        # return out['SAR_GAP']
    return out["optical_GAP"]  # (B, D)

def _croma_sar_gap_feature_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """CROMA: use optical branch GAP embedding."""
    out: Dict[str, torch.Tensor] = model(SAR_images=x)
    # if "optical_GAP" not in out:
        # if "SAR_GAP" not in out:
        #     raise RuntimeError(f"CROMA output has no optical_GAP or sar_GAP keys: {out.keys()}")
        # return out['SAR_GAP']
    return out["SAR_GAP"]  # (B, D)


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


# create new transform which divides all pixel values by 10000
class S2ScaleTransform(nn.Module):
    def __init__(self, scale: float = 10000.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale

# Registry of models to evaluate
def build_model_specs() -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # # 1. Random Conv Filters (RCF) – 13-band, 512-dim features
    # specs.append(
    #     ModelSpec(
    #         name="RCF_13ch",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: RandomConvFeatures(in_chans=13, out_dim=512),
    #         transform=T.Compose([
    #                 T.CenterCrop((224, 224)),
    #                 S2ScaleTransform(scale=10000.0),
    #             ]),
    #         feature_fn=lambda m, x: m(x),
    #     )
    # )

    # # 2. CROMA optical encoder – 12 optical bands, 120x120, [0,1]
    # module = _load_croma_module()
    # PretrainedCROMA = getattr(module, "PretrainedCROMA")
    # specs.append(
    #     ModelSpec(
    #         name="CROMA_optical",
    #         in_chans=12,
    #         rgb_mode=False,
    #         builder=lambda: PretrainedCROMA(
    #             pretrained_path='/home/juro4948/ciip/comparison/CROMA-main/CROMA_base.pt',
    #             size="base",
    #             modality="optical",
    #             image_resolution=120,
    #         ),
    #         # croma_base(
    #         #     weights=CROMABase_Weights.CROMA_VIT, modalities=['optical'], #CROMA_BASE_S1S2  # TODO: confirm exact enum
    #         # ),
    #         transform = nn.Sequential(
    #             T.Resize((120, 120)),
    #             CromaNormalize(use_8_bit=False),  # or True if you want the 8-bit variant
    #         ),
    #         feature_fn=_croma_optical_gap_feature_fn,
    #     )
    # )

    

    # # 3. DOFA base – 13 bands, use pre-trained base weights
    # specs.append(
    #     ModelSpec(
    #         name="DOFA_base_S2_13ch",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: dofa_base_patch16_224(
    #             weights=DOFABase16_Weights.DOFA_MAE,
    #         ),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             S2ScaleTransform(scale=10000.0),
    #             # no extra normalization in the ID pipeline
    #         ]),
    #         feature_fn=_dofa_feature_fn,
    #     )
    # )

    # # # 4. ScaleMAE Large (RGB)
    # IMAGENET_MEAN =  [0.485, 0.456, 0.406]
    # IMAGENET_STD  = [0.229, 0.224, 0.225]
    # specs.append(
    #     ModelSpec(
    #         name="ScaleMAE_large_RGB",
    #         in_chans=3,
    #         rgb_mode=True,
    #         builder=lambda: scalemae_large_patch16(
    #             weights=ScaleMAELarge16_Weights.FMOW_RGB
    #         ),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             S2ScaleTransform(scale=10000.0),
    #             T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    #             ]),
    #         feature_fn=_scalemae_feature_fn,
    #     )
    # )

    # # 5–8. ResNet baselines (MoCo S2 ALL / RGB + ImageNet)
    # specs.append(
    #     ModelSpec(
    #         name="ResNet18_S2_ALL_MOCO",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             S2ScaleTransform(scale=10000.0),
    #             # no explicit normalization in the ID experiment
    #         ]),
    #         feature_fn=_resnet_gap_feature_fn
    #     )
    # )
    # specs.append(
    #     ModelSpec(
    #         name="ResNet50_S2_ALL_MOCO",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             S2ScaleTransform(scale=10000.0),
    #             # no explicit normalization in the ID experiment
    #         ]),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )
    # specs.append(
    #     ModelSpec(
    #         name="ResNet18_S2_RGB_MOCO",
    #         in_chans=3,
    #         rgb_mode=True,
    #         builder=lambda: resnet18(weights=ResNet18_Weights.SENTINEL2_RGB_MOCO),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             S2ScaleTransform(scale=10000.0),
    #             T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    #         ]),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )
    # specs.append(
    #     ModelSpec(
    #         name="ResNet50_S2_RGB_MOCO",
    #         in_chans=3,
    #         rgb_mode=True,
    #         builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             S2ScaleTransform(scale=10000.0),
    #             T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    #         ]),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )

    # # 9. ResNet152 ImageNet-1K (RGB only)
    # specs.append(
    #     ModelSpec(
    #         name="ResNet152_ImageNet_RGB",
    #         in_chans=3,
    #         rgb_mode=True,
    #         builder=lambda: resnet152(weights=ResNet152_Weights.IMAGENET1K_V1),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             S2ScaleTransform(scale=10000.0),
    #             T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    #         ]),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )

    # # 10. ViT-Small Sentinel-2 13-band MoCo
    # specs.append(
    #     ModelSpec(
    #         name="ViTSmall16_S2_ALL_MOCO",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: vit_small_patch16_224(
    #             weights=ViTSmall16_Weights.SENTINEL2_ALL_MOCO
    #         ),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             S2ScaleTransform(scale=10000.0),
    #             # no additional normalization in the ID pipeline
    #         ]),
    #         feature_fn=_vit_patch_mean_feature_fn,
    #     )
    # )


    specs.append(
        ModelSpec(
            name="11-22 CIIP, Epoch10",
            in_chans=12,
            rgb_mode=False,
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_10.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0),
                # no additional normalization in the ID pipeline
            ]) #get_transform('s2a', is_train=False)
        )
    )

    specs.append(
        ModelSpec(
            name="11-22 CIIP, Epoch20",
            in_chans=12,
            rgb_mode=False,
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_20.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0),
                # no additional normalization in the ID pipeline
            ]) #get_transform('s2a', is_train=False)
        )
    )

    specs.append(
        ModelSpec(
            name="11-22 CIIP, Epoch30",
            in_chans=12,
            rgb_mode=False,
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_30.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0),
                # no additional normalization in the ID pipeline
            ]) #get_transform('s2a', is_train=False)
        )
    )

    specs.append(
        ModelSpec(
            name="11-22 CIIP, Epoch90",
            in_chans=12,
            rgb_mode=False,
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_90.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_feature_fn,
            transform=T.Compose([
                T.CenterCrop((224, 224)),
                S2ScaleTransform(scale=10000.0),
                # no additional normalization in the ID pipeline
            ]) #get_transform('s2a', is_train=False)
        )
    )

    # specs.append(
    #     ModelSpec(
    #         name="VanillaCIIP, Epoch10",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint("/local/ms-data/SSL4EO/model/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints/epoch_10.pt")[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #         transform = get_transform('s2a', is_train=False)
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="VanillaCIIP, Epoch10",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint("/local/ms-data/SSL4EO/model/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints/epoch_10.pt")[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #         transform = get_transform('s2a', is_train=False)
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="VanillaCIIP, Epoch40",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint("/local/ms-data/SSL4EO/model/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints/epoch_40.pt")[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #         transform = get_transform('s2a', is_train=False)
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="VanillaCIIP, Epoch40",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint("/local/ms-data/SSL4EO/model/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints/epoch_40.pt")[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=1, Epoch10",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/checkpoints/epoch_10.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=0.1, Epoch10",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_07-09_24_18-model_resnet50-lr_0.001-b_128-j_6-p_amp/checkpoints/epoch_10.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="VanillaCIIP, Epoch25",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint("/local/ms-data/SSL4EO/model/2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp/checkpoints/epoch_25.pt")[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=1, Epoch24",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/checkpoints/epoch_24.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=0.1, v1.1/12bands, Epoch10",
    #         in_chans=12,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_13-08_33_13-model_resnet50-lr_0.001-b_2-j_6-p_amp/checkpoints/epoch_10.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Hyperbolic CIIP, Curv=0.1, v1.1/12bands, Epoch40",
    #         in_chans=12,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_14-10_56_41-model_resnet50-lr_0.001-b_2-j_6-p_amp/checkpoints/epoch_40.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Random ResNet (CIIP), v1.1/12bands, Epoch0",
    #         in_chans=12,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_13-08_33_13-model_resnet50-lr_0.001-b_2-j_6-p_amp/checkpoints/epoch_0.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="Random ResNet (CIIP), Epoch0",
    #         in_chans=13,
    #         rgb_mode=False,
    #         builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_10_09-11_04_30-model_resnet50-lr_0.001-b_128-j_6-p_amp/checkpoints/epoch_0.pt')[0], # first returned is model, second is is_loretnz
    #         feature_fn=_ciip_feature_fn,
    #     )
    # )

    return specs

# Registry of models to evaluate
def build_model_specs_s1() -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # # 1. Random Conv Filters (RCF) – 13-band, 512-dim features
    # specs.append(
    #     ModelSpec(
    #         name="RCF_2ch",
    #         in_chans=2,
    #         rgb_mode=False,
    #         builder=lambda: RandomConvFeatures(in_chans=2, out_dim=512),
    #         transform=T.Compose([
    #                 T.CenterCrop((224, 224)),
    #                 # S2ScaleTransform(scale=10000.0),
    #             ]),
    #         feature_fn=lambda m, x: m(x),
    #     )
    # )

    # # 2. CROMA optical encoder – 12 optical bands, 120x120, [0,1]
    # module = _load_croma_module()
    # PretrainedCROMA = getattr(module, "PretrainedCROMA")
    # specs.append(
    #     ModelSpec(
    #         name="CROMA_SAR_no-normalize",
    #         in_chans=2,
    #         rgb_mode=False,
    #         builder=lambda: PretrainedCROMA(
    #             pretrained_path='/home/juro4948/ciip/comparison/CROMA-main/CROMA_base.pt',
    #             size="base",
    #             modality="SAR",
    #             image_resolution=120,
    #         ),
    #         # croma_base(
    #         #     weights=CROMABase_Weights.CROMA_VIT, modalities=['optical'], #CROMA_BASE_S1S2  # TODO: confirm exact enum
    #         # ),
    #         transform = nn.Sequential(
    #             T.Resize((120, 120)),
    #             # CromaNormalize(use_8_bit=False),  # or True if you want the 8-bit variant
    #         ),
    #         feature_fn=_croma_sar_gap_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="ViTSmall16_S1_GRD_MAE",
    #         in_chans=2,
    #         rgb_mode=False,
    #         builder=lambda: vit_small_patch16_224(
    #             weights=ViTSmall16_Weights.SENTINEL1_GRD_MAE
    #         ),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             # S2ScaleTransform(scale=10000.0),
    #             # no additional normalization in the ID pipeline
    #         ]),
    #         feature_fn=_vit_patch_mean_feature_fn,
    #     )
    # )
    # import torchgeo
    # specs.append(
    #     ModelSpec(
    #         name="ViTSmall14_DINOv2",
    #         in_chans=2,
    #         rgb_mode=False,
    #         builder=lambda: torchgeo.models.vit_small_patch14_dinov2(
    #             weights=torchgeo.models.vit.ViTSmall14_DINOv2_Weights.SENTINEL1_GRD_SOFTCON
    #         ),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             # S2ScaleTransform(scale=10000.0),
    #             # no additional normalization in the ID pipeline
    #         ]),
    #         feature_fn=_vit_patch_mean_feature_fn,
    #     )
    # )



    # specs.append(
    #     ModelSpec(
    #         name="ResNet50_S1_GRD_MOCO",
    #         in_chans=2,
    #         rgb_mode=False,
    #         builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL1_GRD_MOCO),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             # S2ScaleTransform(scale=10000.0),
    #             # no explicit normalization in the ID experiment
    #         ]),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )

    # specs.append(
    #     ModelSpec(
    #         name="ResNet50_S1_GRD_DECUR",
    #         in_chans=2,
    #         rgb_mode=False,
    #         builder=lambda: resnet50(weights=ResNet50_Weights.SENTINEL1_GRD_DECUR),
    #         transform = T.Compose([
    #             T.CenterCrop((224, 224)),
    #             # S2ScaleTransform(scale=10000.0),
    #             # no explicit normalization in the ID experiment
    #         ]),
    #         feature_fn=_resnet_gap_feature_fn,
    #     )
    # )

    ciip_s1_transform = get_transform('s1', is_train=False)

    specs.append(
        ModelSpec(
            name="Random ResNet (CIIP), v1.1/12bands, Epoch0",
            in_chans=2,
            rgb_mode=False,
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_0.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_s1_feature_fn,
            transform=ciip_s1_transform
        )
    )

    specs.append(
        ModelSpec(
            name="S1 11-22 CIIP Epoch10",
            in_chans=2,
            rgb_mode=False,
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_10.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_s1_feature_fn,
            transform=ciip_s1_transform

        )
    )

    specs.append(
        ModelSpec(
            name="S1 11-22 CIIP Epoch30",
            in_chans=2,
            rgb_mode=False,
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_30.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_s1_feature_fn,
            transform=ciip_s1_transform
        )
    )

    specs.append(
        ModelSpec(
            name="S1 11-22 CIIP Epoch90",
            in_chans=2,
            rgb_mode=False,
            builder=lambda: build_model_from_checkpoint('/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_90.pt')[0], # first returned is model, second is is_loretnz
            feature_fn=_ciip_s1_feature_fn,
            transform=ciip_s1_transform
        )
    )

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


def build_ssl4eo_s1_like_loader(
    batch_size: int = 64,
    num_workers: int = 12,
) -> DataLoader:
  
    loader = build_s1_ssl4eo_loader(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return loader


def ssl4eo_collate(batch):
    """
    batch: list of dicts from _SSL4EOWrapped.__getitem__
      each dict has:
        "image": [64, C, H, W]
        "label": scalar (per 64-pack)

    Returns:
      "image": [B*64, C, H, W]
      "label": [B*64]  (broadcasted per sample)
    """
    # Stack along new batch dim: [B, 64, C, H, W]
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    B, P, C, H, W = images.shape  # P should be 64

    # Flatten the (B, P) into one sample dim: [B*P, C, H, W]
    images = images.view(B * P, C, H, W)

    # Broadcast labels per patch if you want one label per sample
    labels = labels.repeat_interleave(P)  # [B*P]

    return {"image": images, "label": labels}

def build_ssl4eo_like_loader(
    in_chans: int,
    batch_size: int = 64,
    num_workers: int = 12,
    rgb_mode: bool = False,
) -> DataLoader:
  
    loader = build_s2_ssl4eo_loader(
        in_chans=in_chans,
        batch_size=batch_size,
        num_workers=num_workers,
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
            image = image.to(device)
            model = model.to(device)

            if transform:
                # print(image.shape)
                image = transform(image)
                
            else:
                print('No transform applied during feature extraction.')

            # print bandwise averages/stds
            print(f"Batch {b_idx}: image band means: {image.mean(dim=[0,2,3])}, stds: {image.std(dim=[0,2,3])}")
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
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # device = torch.device("cuda")
    device = 'cuda' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_model_specs() #_s1

    results: Dict[str, float] = {}
    mle_results: Dict[str, float] = {}
    mom_results: Dict[str, float] = {}
    tle_results: Dict[str, float] = {}

    for spec in specs:
        print(f"\n=== Model: {spec.name} ===")
        # try:
        # loader = build_s2_like_loader(
        #     in_chans=spec.in_chans,
        #     batch_size=64,
        #     num_workers=12,
        #     rgb_mode=spec.rgb_mode,
        # )

        loader = build_ssl4eo_like_loader(
            in_chans=spec.in_chans,
            batch_size=16,
            num_workers=12,
            rgb_mode=spec.rgb_mode,
        )

        # loader = build_ssl4eo_s1_like_loader(
        #     batch_size=32,
        #     num_workers=12
        # )

        model = spec.builder().to(device)
        # print(model)
        # if type resnet
        if isinstance(model, torchvision.models.ResNet):
            # remove final fc layer
            model.fc = nn.Identity()
        # if ciip
        if 'ciip' in spec.name.lower():
            model.encoder_s1.fc = nn.Identity()
            model.encoder_s2.fc = nn.Identity()
            # print(model)
            print("Modified CIIP model to remove final layer.")

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
            gid = compute_global_id(Z)
            results[spec.name] = gid
            print(f"  Global FisherS ID: {gid:.4f}")

            mle_lid = id.MLE(neighborhood_based=True).fit_transform(Z, n_neighbors=20)
            mle_results[spec.name] = mle_lid
            print(f"  Global MLE ID: {mle_lid:.4f}")

            # error
            # mom_pw = id.MOM().fit_transform_pw(Z, n_neighbors=20)
            # print("has NaN:", np.isnan(Z).any())
            # print("has inf:", np.isinf(Z).any())
            # print("MoM pw min/max:", np.nanmin(mom_pw), np.nanmax(mom_pw))


            mom_lid = id.MOM().fit_transform(Z, n_neighbors=20)
            mom_results[spec.name] = mom_lid
            print(f"  Global MoM ID: {mom_lid:.4f}")

            # tle_lid = id.TLE().fit_transform(Z, n_neighbors=20)
            # 
            tle_lid = compute_tle(Z)
            tle_results[spec.name] = tle_lid
            print(f"  Global TLE ID: {tle_lid:.4f}")

        except Exception as e:
            print(f"  [SKIP] Failed to compute ID for '{spec.name}': {e}")

    # Summary
    print("\n=== Summary of Global IDs (FisherS) ===")
    for name, gid in results.items():
        print(f"{name:30s}: {gid:.4f}")

    print("\n=== Summary of Global IDs (MLE) ===")
    for name, gid in mle_results.items():
        print(f"{name:30s}: {gid:.4f}")

    print("\n=== Summary of Global IDs (MoM) ===")
    for name, gid in mom_results.items():
        print(f"{name:30s}: {gid:.4f}")

    print("\n=== Summary of Global IDs (TLE) ===")
    for name, gid in tle_results.items():
        print(f"{name:30s}: {gid:.4f}")

    # save all to same txt file
    with open(OUTPUT_DIR / "global_id_results.txt", "w") as f:
        f.write("=== Global IDs (FisherS) ===\n")
        for name, gid in results.items():
            f.write(f"{name:30s}: {gid:.4f}\n")

        f.write("\n=== Global IDs (MLE) ===\n")
        for name, gid in mle_results.items():
            f.write(f"{name:30s}: {gid:.4f}\n")

        f.write("\n=== Global IDs (MoM) ===\n")
        for name, gid in mom_results.items():
            f.write(f"{name:30s}: {gid:.4f}\n")

        f.write("\n=== Global IDs (TLE) ===\n")
        for name, gid in tle_results.items():
            f.write(f"{name:30s}: {gid:.4f}\n")

    print(f"\nResults saved to {OUTPUT_DIR / 'global_id_results.txt'}")


if __name__ == "__main__":
    main()
