import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ciip.model_ciip import CIIP, LorentzCIIP
from ciip.model import LayerNorm, Transformer
from ciip.open_clip_train.utils import convert_weights_to_lp, get_cast_dtype
from torchgeo.models import ResNet50_Weights, resnet50


def _resolve_logit_settings(model_cfg) -> tuple[float, Optional[float]]:
    init_logit_scale = getattr(model_cfg, "init_logit_scale", None)
    if init_logit_scale is None:
        init_logit_scale = np.log(1 / 0.07)
    init_logit_bias = getattr(model_cfg, "init_logit_bias", None)
    return float(init_logit_scale), init_logit_bias


class TextEncoder(nn.Module):
    def __init__(
        self,
        *,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self._build_attention_mask(context_length),
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.initialize_parameters()

    @staticmethod
    def _build_attention_mask(context_length: int) -> torch.Tensor:
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def initialize_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def forward(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x


class DualEncoderCIIP(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: TextEncoder,
        init_logit_scale: float,
        init_logit_bias: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.encoder_s2 = image_encoder
        self.encoder_text = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = (
            nn.Parameter(torch.ones([]) * init_logit_bias)
            if init_logit_bias is not None
            else None
        )

    @property
    def dtype_s2(self):
        return self.encoder_s2.conv1.weight.dtype

    @property
    def dtype_text(self):
        return self.encoder_text.token_embedding.weight.dtype

    def encode_s2(self, s2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.encoder_s2(s2.type(self.dtype_s2))
        proj_layer = getattr(self.encoder_s2, "proj", None)
        if isinstance(proj_layer, nn.Module):
            features = proj_layer(features)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.encoder_text(tokens, normalize=normalize)

    def forward(self, s2: torch.Tensor, text: torch.Tensor):
        s2_features = self.encode_s2(s2, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        logit_scale = self.logit_scale.exp()
        out = {
            "s1_features": s2_features,
            "s2_features": text_features,
            "logit_scale": logit_scale,
        }
        if self.logit_bias is not None:
            out["logit_bias"] = self.logit_bias
        return out


def _build_sentinel2_resnet50(model_cfg) -> nn.Module:
    if not getattr(model_cfg.pretrain, "load", False):
        logging.info("Using ResNet50 Sentinel-2 encoder without pretrained weights")
        return resnet50(in_chans=len(model_cfg.s2_bands), num_classes=model_cfg.embed_dim)

    if model_cfg.pretrain.s2_weights == "MOCO":
        weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
    elif model_cfg.pretrain.s2_weights == "DINO":
        weights = ResNet50_Weights.SENTINEL2_ALL_DINO
    else:
        raise ValueError(
            "Unsupported S2 weights: %s. Use 'MOCO' or 'DINO'."
            % model_cfg.pretrain.s2_weights
        )

    logging.info("Loaded pretrained weights for S2 encoder: %s", model_cfg.pretrain.s2_weights)
    return resnet50(
        in_chans=len(model_cfg.s2_bands),
        num_classes=model_cfg.embed_dim,
        weights=weights,
    )


def _build_text_encoder(model_cfg) -> TextEncoder:
    text_cfg = getattr(model_cfg, "text", None) or {}
    return TextEncoder(
        context_length=getattr(text_cfg, "context_length", 77),
        vocab_size=getattr(text_cfg, "vocab_size", 49408),
        transformer_width=getattr(text_cfg, "transformer_width", 512),
        transformer_heads=getattr(text_cfg, "transformer_heads", 8),
        transformer_layers=getattr(text_cfg, "transformer_layers", 12),
        embed_dim=model_cfg.embed_dim,
    )


def build_ciip_architecture(model_cfg, loss_cfg=None):
    loss_cfg = loss_cfg or {}
    init_logit_scale, init_logit_bias = _resolve_logit_settings(model_cfg)

    if getattr(model_cfg, "encoder_pair", None) == "s2_text":
        logging.info(
            "Building Sentinel-2 (ResNet50) + text transformer encoders with embed_dim=%s",
            model_cfg.embed_dim,
        )
        image_encoder = _build_sentinel2_resnet50(model_cfg)
        text_encoder = _build_text_encoder(model_cfg)
        return DualEncoderCIIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            init_logit_scale=init_logit_scale,
            init_logit_bias=init_logit_bias,
        )

    hyperbolic = getattr(loss_cfg, "hyperbolic", False)
    model_cls = LorentzCIIP if hyperbolic else CIIP

    logging.info(
        "Building %s backbone (hyperbolic=%s) with embedding dim %s",
        model_cls.__name__,
        hyperbolic,
        model_cfg.embed_dim,
    )

    return model_cls(
        embed_dim=model_cfg.embed_dim,
        s1_resolution=model_cfg.s1_resolution,
        s1_layers=OmegaConf.to_object(model_cfg.s1_layers),
        s1_width=model_cfg.width,
        s1_patch_size=model_cfg.s1_patch_size,
        s1_bands=len(model_cfg.s1_bands),
        s2_resolution=model_cfg.s2_resolution,
        s2_layers=OmegaConf.to_object(model_cfg.s2_layers),
        s2_width=model_cfg.width,
        s2_patch_size=model_cfg.s2_patch_size,
        s2_bands=len(model_cfg.s2_bands),
        framework=model_cfg.framework,
        pretrain=model_cfg.pretrain.load,
        s1_weights=model_cfg.pretrain.s1_weights,
        s2_weights=model_cfg.pretrain.s2_weights,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        curv_init=getattr(loss_cfg, "curvature_init", None),
        learn_curv=getattr(loss_cfg, "learn_curv", None),
        entail_weight=getattr(loss_cfg, "entail_weight", None),
    )


def finalize_model(model: torch.nn.Module, device: torch.device, precision: str) -> torch.nn.Module:
    cast_dtype = get_cast_dtype(precision)
    model.to(device=device)
    if cast_dtype is not None:
        convert_weights_to_lp(model, dtype=cast_dtype)
    return model


def unwrap_dataparallel(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def maybe_data_parallel(model: torch.nn.Module) -> torch.nn.Module:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logging.info("Wrapping model in DataParallel across %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    return model


__all__ = [
    "build_ciip_architecture",
    "DualEncoderCIIP",
    "finalize_model",
    "maybe_data_parallel",
    "TextEncoder",
    "unwrap_dataparallel",
]
