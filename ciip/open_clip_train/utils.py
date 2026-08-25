import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn

from ciip.model_ciip import CIIP, LorentzCIIP
from ..loss import CiipLoss, SigLipLoss


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert convolutional and linear weights to low precision."""

    def _convert_weights(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.to(dtype)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


_SENTINEL = object()


def _resolve_section(cfg, key):
    if cfg is None:
        return None
    if isinstance(cfg, DictConfig):
        try:
            return cfg[key]
        except Exception:
            return None
    if isinstance(cfg, dict):
        return cfg.get(key)
    return getattr(cfg, key, None)


def _get_cfg_value(cfg, key, default=_SENTINEL):
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        if key in cfg:
            return cfg[key]
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_value(key, *cfgs, default=None):
    for cfg in cfgs:
        value = _get_cfg_value(cfg, key, _SENTINEL)
        if value is not _SENTINEL:
            return value
    return default


def create_loss(args):
    datamodule_cfg = _resolve_section(args, "datamodule")
    loss_cfg = _resolve_section(args, "loss")
    if loss_cfg is None:
        loss_cfg = args

    if bool(_resolve_value("distill", args, loss_cfg, default=False)):
        raise NotImplementedError("distillation is not a supported CIIP objective")

    siglip_enabled = bool(_resolve_value("siglip", args, loss_cfg, default=False))
    if siglip_enabled:
        return SigLipLoss(
            rank=_resolve_value("rank", args, datamodule_cfg, default=0),
            world_size=_resolve_value("world_size", args, datamodule_cfg, default=1),
        )
    vc_covariance_weights = _resolve_value("vc_covariance_weights", loss_cfg, args, default=None)
    if isinstance(vc_covariance_weights, (DictConfig, ListConfig)):
        vc_covariance_weights = OmegaConf.to_object(vc_covariance_weights)

    matryoshka_dims = _resolve_value("matryoshka_dims", loss_cfg, args, default=None)
    if isinstance(matryoshka_dims, (DictConfig, ListConfig)):
        matryoshka_dims = OmegaConf.to_object(matryoshka_dims)

    matryoshka_relative_weights = _resolve_value("matryoshka_weights", loss_cfg, args, default=None)
    if isinstance(matryoshka_relative_weights, (DictConfig, ListConfig)):
        matryoshka_relative_weights = OmegaConf.to_object(matryoshka_relative_weights)

    return CiipLoss(
        local_loss=_resolve_value("local_loss", loss_cfg, args, default=False),
        gather_with_grad=_resolve_value("gather_with_grad", loss_cfg, args, default=False),
        cache_labels=_resolve_value("cache_labels", loss_cfg, args, default=True),
        rank=_resolve_value("rank", loss_cfg, args, datamodule_cfg, default=0),
        world_size=_resolve_value("world_size", loss_cfg, args, datamodule_cfg, default=1),
        use_horovod=_resolve_value("horovod", loss_cfg, args, datamodule_cfg, default=False),
        contrastive_weight=_resolve_value("contrastive_weight", loss_cfg, args, default=1.0),
        vc_reg_enabled=_resolve_value("vc_reg_enabled", loss_cfg, args, default=False),
        vc_weight=_resolve_value("vc_weight", loss_cfg, args, default=0.0),
        vc_gamma=_resolve_value("vc_gamma", loss_cfg, args, default=1.0),
        vc_covariance_weights=vc_covariance_weights,
        batch_uniformity_enabled=_resolve_value("batch_uniformity_enabled", loss_cfg, args, default=True),
        batch_uniformity_weight=_resolve_value(
            "batch_uniformity_weight", loss_cfg, args, default=0.05
        ),
        hyperbolic=_resolve_value("hyperbolic", loss_cfg, args, default=False),
        hyperbolic_normalize=_resolve_value("hyperbolic_normalize", loss_cfg, args, default=True),
        hyperbolic_curvature_init=_resolve_value("hyperbolic_curvature_init", loss_cfg, args, default=1.0),
        hyperbolic_eps=_resolve_value("hyperbolic_eps", loss_cfg, args, default=1e-5),
        matryoshka_enabled=_resolve_value("matryoshka_enabled", loss_cfg, args, default=False),
        matryoshka_weight=_resolve_value("matryoshka_weight", loss_cfg, args, default=1.0),
        matryoshka_dims=matryoshka_dims,
        matryoshka_relative_weights=matryoshka_relative_weights,
        matryoshka_normalize=_resolve_value("matryoshka_normalize", loss_cfg, args, default=True),
    )


def create_model(args, device, **overrides):
    """Build the maintained two-sensor model on ``device``."""
    init_logit_scale = overrides.pop("init_logit_scale", np.log(1 / 0.07))
    init_logit_bias = overrides.pop("init_logit_bias", None)
    if overrides:
        names = ", ".join(sorted(overrides))
        raise TypeError(f"unsupported model overrides: {names}")

    def _plain(value):
        return OmegaConf.to_object(value) if OmegaConf.is_config(value) else value

    pretrain = _resolve_section(args.model, "pretrain")
    loss_config = _resolve_section(args, "loss")
    common = {
        "embed_dim": args.model.embed_dim,
        "s1_resolution": args.model.s1_resolution,
        "s1_layers": _plain(args.model.s1_layers),
        "s1_width": args.model.width,
        "s1_patch_size": args.model.s1_patch_size,
        "s1_bands": len(args.model.s1_bands),
        "s2_resolution": args.model.s2_resolution,
        "s2_layers": _plain(args.model.s2_layers),
        "s2_width": args.model.width,
        "s2_patch_size": args.model.s2_patch_size,
        "s2_bands": len(args.model.s2_bands),
        "framework": args.model.framework,
        "pretrain": _resolve_value("load", pretrain, default=False),
        "s1_weights": _resolve_value("s1_weights", pretrain, default=None),
        "s2_weights": _resolve_value("s2_weights", pretrain, default=None),
        "patch_masking": getattr(args.model, "patch_masking", False),
        "patch_mask_ratio": getattr(args.model, "patch_mask_ratio", 0.0),
        "recon_cfg": getattr(args, "recon", None),
        "init_logit_scale": init_logit_scale,
        "init_logit_bias": init_logit_bias,
    }

    if bool(_resolve_value("hyperbolic", loss_config, default=False)):
        model = LorentzCIIP(
            **common,
            curv_init=_resolve_value("curvature_init", loss_config, default=0.1),
            learn_curv=_resolve_value("learn_curv", loss_config, default=True),
            entail_weight=_resolve_value("entail_weight", loss_config, default=0.0),
        )
    else:
        model = CIIP(**common)

    return model.to(device=torch.device(device))
