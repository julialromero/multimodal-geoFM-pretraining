"""Construction and checkpoint loading for the legacy CIIP evaluation model."""

from __future__ import annotations

from pathlib import Path

import torch

from ciip.model_ciip import CIIP


def create_ciip_model(embed_dim: int = 512, pre_projection_dim: int = 1024) -> CIIP:
    return CIIP(
        framework="resnet50",
        embed_dim=embed_dim,
        pre_projection_dim=pre_projection_dim,
        s1_resolution=224,
        s1_layers=(3, 4, 6, 3),
        s1_width=32,
        s1_patch_size=16,
        s1_bands=2,
        s2_resolution=224,
        s2_layers=(3, 4, 6, 3),
        s2_width=32,
        s2_patch_size=16,
        s2_bands=13,
    )


def load_ciip_model_checkpoint(checkpoint_path: str | Path) -> CIIP:
    model = create_ciip_model()
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["state_dict"].items()
        if "fc" not in key
    }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "encoder_s1.fc.weight",
        "encoder_s1.fc.bias",
        "encoder_s2.fc.weight",
        "encoder_s2.fc.bias",
    }
    if not set(missing_keys) <= allowed_missing or unexpected_keys:
        raise ValueError(
            f"incompatible checkpoint: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    return model


def modify_ciip_for_eurosat(
    model: CIIP, num_classes: int = 10, freeze_encoder: bool = False
) -> torch.nn.Module:
    """Attach a trainable classification head to the Sentinel-2 encoder."""
    encoder = model.encoder_s2
    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze_encoder
    encoder.fc = torch.nn.Linear(512, num_classes)
    return encoder
