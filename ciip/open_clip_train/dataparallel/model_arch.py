import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - optional dependency
    hf_hub_download = None

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
        # 2-layer projection head: transformer_width -> transformer_width -> embed_dim
        self.text_proj1 = nn.Linear(transformer_width, transformer_width)
        self.text_proj2 = nn.Linear(transformer_width, embed_dim)

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
        nn.init.normal_(self.text_proj1.weight, std=self.transformer.width ** -0.5)
        if self.text_proj1.bias is not None:
            nn.init.zeros_(self.text_proj1.bias)
        nn.init.normal_(self.text_proj2.weight, std=self.transformer.width ** -0.5)
        if self.text_proj2.bias is not None:
            nn.init.zeros_(self.text_proj2.bias)

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
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = self.text_proj1(x)
        x = F.gelu(x)
        x = self.text_proj2(x)
        return F.normalize(x, dim=-1) if normalize else x


def configure_text_trainable_layers(encoder: TextEncoder) -> None:
    # freeze everything by default
    for param in encoder.parameters():
        param.requires_grad = False

    resblocks = getattr(encoder.transformer, "resblocks", [])
    total_blocks = len(resblocks)
    if total_blocks == 0:
        return

    start_idx = total_blocks // 2
    for idx in range(start_idx, total_blocks):
        for param in resblocks[idx].parameters():
            param.requires_grad = True

    for proj in (encoder.text_proj1, encoder.text_proj2):
        for param in proj.parameters():
            param.requires_grad = True


def _load_ms_clip_text_weights(
    encoder: TextEncoder,
    repo_id: str = "ibm-esa-geospatial/Llama3-MS-CLIP-base",
    filename: str = "Llama3_MS_CLIP_weights.pt",
) -> None:
    """Load MS-CLIP text weights into our TextEncoder.

    Only text-side parameters are used; unexpected keys are ignored.
    """
    logging.info("Loading MS-CLIP text weights from %s (%s)", repo_id, filename)
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("state_dict", state)

    target_state = encoder.state_dict()
    prefixes = [
        "",
        "text_encoder.",
        "text_encoder.model.",
        "text.",
        "encoder_text.",
        "model.text.",
        "module.text.",
        "clip.text.",
        "clip_base_model.model.",
        "text_model.",
        "language_model.",
    ]

    def _strip_prefix(key: str) -> str:
        for p in prefixes:
            if p and key.startswith(p):
                return key[len(p) :]
        return key

    filtered = {}
    for key, tensor in state_dict.items():
        norm_key = _strip_prefix(key)
        if norm_key in target_state and target_state[norm_key].shape == tensor.shape:
            filtered[norm_key] = tensor

    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    if missing:
        logging.info("MS-CLIP text load: missing keys (ignored): %s", missing)
    if unexpected:
        logging.info("MS-CLIP text load: unexpected keys (ignored): %s", unexpected)
    logging.info("MS-CLIP text weights loaded with %d/%d params matched.", len(filtered), len(target_state))


def _load_s2_weights_from_ciip_checkpoint(encoder: nn.Module, ckpt_path: str) -> None:
    """Load only encoder_s2 weights from a CIIP checkpoint into the given encoder."""
    logging.info("Loading Sentinel-2 encoder weights from CIIP checkpoint: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    target_state = encoder.state_dict()

    prefixes = [
        "encoder_s2.",
        "module.encoder_s2.",
        "model.encoder_s2.",
        "module.model.encoder_s2.",
        "",
    ]

    def _strip_prefix(key: str) -> str:
        for p in prefixes:
            if p and key.startswith(p):
                return key[len(p) :]
        return key

    filtered = {}
    for key, tensor in state_dict.items():
        norm_key = _strip_prefix(key)
        if norm_key in target_state and target_state[norm_key].shape == tensor.shape:
            filtered[norm_key] = tensor

    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    if missing:
        logging.info("CIIP S2 load: missing keys (ignored): %s", missing)
    if unexpected:
        logging.info("CIIP S2 load: unexpected keys (ignored): %s", unexpected)
    logging.info("CIIP S2 weights loaded with %d/%d params matched.", len(filtered), len(target_state))


def _load_s1s2_from_ciip_checkpoint(s1_encoder: nn.Module, s2_encoder: nn.Module, ckpt_path: str) -> None:
    logging.info("Loading S1 and S2 encoder weights from CIIP checkpoint: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    def _load_single(target: nn.Module, prefixes: list[str]) -> None:
        target_state = target.state_dict()

        def _strip_prefix(key: str) -> str:
            for p in prefixes:
                if p and key.startswith(p):
                    return key[len(p):]
            return key

        filtered = {}
        for key, tensor in state_dict.items():
            norm_key = _strip_prefix(key)
            if norm_key in target_state and target_state[norm_key].shape == tensor.shape:
                filtered[norm_key] = tensor

        missing, unexpected = target.load_state_dict(filtered, strict=False)
        if missing:
            logging.info("CIIP load: missing keys (ignored): %s", missing)
        if unexpected:
            logging.info("CIIP load: unexpected keys (ignored): %s", unexpected)
        logging.info("Loaded %d/%d params for %s", len(filtered), len(target_state), prefixes[0].rstrip("."))

    _load_single(s1_encoder, ["encoder_s1.", "module.encoder_s1.", "model.encoder_s1.", "module.model.encoder_s1.", ""])
    _load_single(s2_encoder, ["encoder_s2.", "module.encoder_s2.", "model.encoder_s2.", "module.model.encoder_s2.", ""])


def _load_text_weights_from_checkpoint(encoder: TextEncoder, ckpt_path: str) -> None:
    logging.info("Loading text encoder weights from checkpoint: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    target_state = encoder.state_dict()

    prefixes = [
        "",
        "encoder_text.",
        "module.encoder_text.",
        "text_encoder.",
        "text_encoder.model.",
        "text.",
        "model.text.",
        "clip_base_model.model.",
        "module.model.encoder_text.",
        "language_model.",
    ]

    def _strip_prefix(key: str) -> str:
        for p in prefixes:
            if p and key.startswith(p):
                return key[len(p):]
        return key

    filtered = {}
    for key, tensor in state_dict.items():
        norm_key = _strip_prefix(key)
        if norm_key in target_state and target_state[norm_key].shape == tensor.shape:
            filtered[norm_key] = tensor

    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    if missing:
        logging.info("Text load: missing keys (ignored): %s", missing)
    if unexpected:
        logging.info("Text load: unexpected keys (ignored): %s", unexpected)
    logging.info("Loaded %d/%d text params", len(filtered), len(target_state))


def _build_sentinel1_resnet50(model_cfg) -> nn.Module:
    if not getattr(model_cfg.pretrain, "load", False):
        logging.info("Using ResNet50 Sentinel-1 encoder without pretrained weights")
        return resnet50(in_chans=len(model_cfg.s1_bands), num_classes=model_cfg.embed_dim)

    if model_cfg.pretrain.s1_weights == "MOCO":
        weights = ResNet50_Weights.SENTINEL1_ALL_MOCO
    elif model_cfg.pretrain.s1_weights == "DINO":
        weights = ResNet50_Weights.SENTINEL1_ALL_DINO
    else:
        raise ValueError(
            "Unsupported S1 weights: %s. Use 'MOCO' or 'DINO'."
            % model_cfg.pretrain.s1_weights
        )

    logging.info("Loaded pretrained weights for S1 encoder: %s", model_cfg.pretrain.s1_weights)
    return resnet50(
        in_chans=len(model_cfg.s1_bands),
        num_classes=model_cfg.embed_dim,
        weights=weights,
    )


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
        configure_text_trainable_layers(self.encoder_text)

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


class TriEncoderCIIP(nn.Module):
    def __init__(
        self,
        s1_encoder: nn.Module,
        s2_encoder: nn.Module,
        text_encoder: TextEncoder,
        init_logit_scale: float,
        init_logit_bias: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.encoder_s1 = s1_encoder
        self.encoder_s2 = s2_encoder
        self.encoder_text = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.logit_bias = (
            nn.Parameter(torch.ones([]) * init_logit_bias)
            if init_logit_bias is not None
            else None
        )
        configure_text_trainable_layers(self.encoder_text)

    @property
    def dtype_s1(self):
        return self.encoder_s1.conv1.weight.dtype

    @property
    def dtype_s2(self):
        return self.encoder_s2.conv1.weight.dtype

    @property
    def dtype_text(self):
        return self.encoder_text.token_embedding.weight.dtype

    def encode_s1(self, s1: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.encoder_s1(s1.type(self.dtype_s1))
        proj_layer = getattr(self.encoder_s1, "proj", None)
        if isinstance(proj_layer, nn.Module):
            features = proj_layer(features)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_s2(self, s2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.encoder_s2(s2.type(self.dtype_s2))
        proj_layer = getattr(self.encoder_s2, "proj", None)
        if isinstance(proj_layer, nn.Module):
            features = proj_layer(features)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.encoder_text(tokens, normalize=normalize)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, text: torch.Tensor):
        s1_features = self.encode_s1(s1, normalize=True)
        s2_features = self.encode_s2(s2, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        logit_scale = self.logit_scale.exp()
        out = {
            "s1_features": s1_features,
            "s2_features": s2_features,
            "text_features": text_features,
            "logit_scale": logit_scale,
        }
        if self.logit_bias is not None:
            out["logit_bias"] = self.logit_bias
        return out


def _build_sentinel2_resnet50(model_cfg) -> nn.Module:
    # Optional: load S2 weights from a CIIP checkpoint when pairing with text.
    s2_ckpt_path = getattr(model_cfg, "s2_ciip_checkpoint", None)

    if not getattr(model_cfg.pretrain, "load", False):
        logging.info("Using ResNet50 Sentinel-2 encoder without pretrained weights")
        encoder = resnet50(in_chans=len(model_cfg.s2_bands), num_classes=model_cfg.embed_dim)
        if s2_ckpt_path:
            _load_s2_weights_from_ciip_checkpoint(encoder, s2_ckpt_path)
        return encoder

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
    encoder = resnet50(
        in_chans=len(model_cfg.s2_bands),
        num_classes=model_cfg.embed_dim,
        weights=weights,
    )
    if s2_ckpt_path:
        _load_s2_weights_from_ciip_checkpoint(encoder, s2_ckpt_path)
    return encoder


def _build_text_encoder(model_cfg) -> TextEncoder:
    text_cfg = getattr(model_cfg, "text", None) or {}
    encoder = TextEncoder(
        context_length=getattr(text_cfg, "context_length", 77),
        vocab_size=getattr(text_cfg, "vocab_size", 49408),
        transformer_width=getattr(text_cfg, "transformer_width", 512),
        transformer_heads=getattr(text_cfg, "transformer_heads", 8),
        transformer_layers=getattr(text_cfg, "transformer_layers", 12),
        embed_dim=model_cfg.embed_dim,
    )
    if getattr(text_cfg, "load_ms_clip", False):
        if hf_hub_download is None:
            raise ImportError("huggingface_hub is required to load MS-CLIP text weights. Please install it.")
        _load_ms_clip_text_weights(encoder)
    return encoder




def _as_layers(value):
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    resolved = OmegaConf.to_object(value)
    if isinstance(resolved, list) and len(resolved) == 1:
        return int(resolved[0])
    return resolved

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

    if getattr(model_cfg, "encoder_pair", None) == "s1s2_text":
        logging.info(
            "Building Sentinel-1 + Sentinel-2 + text encoders with embed_dim=%s",
            model_cfg.embed_dim,
        )
        s1_encoder = _build_sentinel1_resnet50(model_cfg)
        s2_encoder = _build_sentinel2_resnet50(model_cfg)
        text_encoder = _build_text_encoder(model_cfg)
        s1s2_ckpt = getattr(model_cfg, "s1s2_ciip_checkpoint", None)
        if s1s2_ckpt:
            _load_s1s2_from_ciip_checkpoint(s1_encoder, s2_encoder, s1s2_ckpt)
        text_ckpt = getattr(model_cfg, "text_checkpoint", None)
        if text_ckpt:
            _load_text_weights_from_checkpoint(text_encoder, text_ckpt)
        elif getattr(getattr(model_cfg, "text", None), "load_ms_clip", False):
            # already loaded inside _build_text_encoder when flag set
            pass
        return TriEncoderCIIP(
            s1_encoder=s1_encoder,
            s2_encoder=s2_encoder,
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
        s1_layers=_as_layers(model_cfg.s1_layers),
        s1_width=model_cfg.width,
        s1_patch_size=model_cfg.s1_patch_size,
        s1_bands=len(model_cfg.s1_bands),
        s2_resolution=model_cfg.s2_resolution,
        s2_layers=_as_layers(model_cfg.s2_layers),
        s2_width=model_cfg.width,
        s2_patch_size=model_cfg.s2_patch_size,
        s2_bands=len(model_cfg.s2_bands),
        framework=model_cfg.framework,
        pretrain=model_cfg.pretrain.load,
        s1_weights=model_cfg.pretrain.s1_weights,
        s2_weights=model_cfg.pretrain.s2_weights,
        patch_masking=getattr(model_cfg, "patch_masking", False),
        patch_mask_ratio=getattr(model_cfg, "patch_mask_ratio", 0.0),
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        **({
            "curv_init": getattr(loss_cfg, "curvature_init", None),
            "learn_curv": getattr(loss_cfg, "learn_curv", None),
            "entail_weight": getattr(loss_cfg, "entail_weight", None),
        } if hyperbolic else {}),
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
    "TriEncoderCIIP",
    "finalize_model",
    "maybe_data_parallel",
    "TextEncoder",
    "unwrap_dataparallel",
]
