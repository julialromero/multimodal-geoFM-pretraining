from __future__ import annotations

from typing import Optional, Tuple

import torch


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin/cos encoding.")
    compute_dtype = torch.float32
    grid_h = torch.arange(grid_size, device=device, dtype=compute_dtype)
    grid_w = torch.arange(grid_size, device=device, dtype=compute_dtype)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)
    pos_embed = _get_2d_sincos_from_grid(embed_dim, grid)
    return pos_embed.to(dtype=dtype)


def _get_2d_sincos_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def _get_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, device=positions.device, dtype=positions.dtype)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    positions = positions.reshape(-1, 1)
    out = positions * omega.reshape(1, -1)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images to flattened patches (B, N, patch_dim)."""
    if images.dim() != 4:
        raise ValueError(f"Expected images with shape (B, C, H, W), got {images.shape}")
    b, c, h, w = images.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by patch_size.")
    grid_h = h // patch_size
    grid_w = w // patch_size
    patches = images.reshape(b, c, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    return patches.reshape(b, grid_h * grid_w, patch_size * patch_size * c)


def per_patch_normalize(patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True, unbiased=False)
    return (patches - mean) / torch.sqrt(var + eps)


def prepare_reconstruction_targets(
    images: torch.Tensor,
    patch_size: int,
    clip_range: Optional[Tuple[float, float]] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if clip_range is not None:
        images = images.clamp(min=clip_range[0], max=clip_range[1])
    patches = patchify(images, patch_size)
    return per_patch_normalize(patches, eps=eps)


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform per-sample random masking by shuffling patches."""
    b, num_tokens, dim = x.shape
    num_keep = int(round(num_tokens * (1.0 - mask_ratio)))
    num_keep = max(1, min(num_keep, num_tokens))
    noise = torch.rand(b, num_tokens, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :num_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))

    mask = torch.ones([b, num_tokens], device=x.device)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore, ids_keep


def apply_keep_mask(
    x: torch.Tensor,
    keep_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a provided keep mask, returning masked tokens and restoration indices."""
    if keep_mask.dim() != 2:
        raise ValueError("keep_mask must have shape (B, N).")
    keep_mask = keep_mask.to(device=x.device, dtype=torch.bool)
    b, num_tokens, dim = x.shape
    keep_counts = keep_mask.sum(dim=1)
    if not torch.equal(keep_counts, keep_counts[:1].expand_as(keep_counts)):
        raise ValueError("apply_keep_mask requires a fixed keep count per sample.")
    ids = torch.arange(num_tokens, device=x.device).unsqueeze(0).expand(b, -1)
    ids_keep = ids[keep_mask].reshape(b, -1)
    ids_mask = ids[~keep_mask].reshape(b, -1)
    ids_shuffle = torch.cat([ids_keep, ids_mask], dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))
    mask = torch.ones([b, num_tokens], device=x.device)
    mask[:, : ids_keep.shape[1]] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore, ids_keep
