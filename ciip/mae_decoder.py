from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .masking import get_2d_sincos_pos_embed
from .model import LayerNorm, Transformer


class MAEDecoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_dim: int,
        decoder_dim: int,
        num_layers: int,
        num_heads: int,
        num_patches: int,
        patch_dim: int,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        grid_size = int(round(num_patches ** 0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError("num_patches must form a square grid for 2D sin/cos encoding.")
        pos_embed = get_2d_sincos_pos_embed(
            decoder_dim,
            grid_size,
            device=self.mask_token.device,
            dtype=self.mask_token.dtype,
        )
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        self.transformer = Transformer(decoder_dim, num_layers, num_heads)
        self.norm = LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_dim)
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_embed.weight, std=self.decoder_embed.in_features ** -0.5)
        if self.decoder_embed.bias is not None:
            nn.init.zeros_(self.decoder_embed.bias)
        nn.init.normal_(self.pred.weight, std=self.pred.in_features ** -0.5)
        if self.pred.bias is not None:
            nn.init.zeros_(self.pred.bias)

    def forward(self, tokens: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(tokens)
        b, num_keep, dim = x.shape
        mask_tokens = self.mask_token.repeat(b, self.num_patches - num_keep, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, dim))
        x_ = x_ + self.pos_embed
        x_ = x_.permute(1, 0, 2)
        x_ = self.transformer(x_)
        x_ = x_.permute(1, 0, 2)
        x_ = self.norm(x_)
        return self.pred(x_)
