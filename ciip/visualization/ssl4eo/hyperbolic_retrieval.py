"""Cross-modality retrieval metrics for SSL4EO embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch
from ciip import lorentz as L
from ciip.visualization.ssl4eo.embedding_collapse_diagnostics import ModalityEmbeddings


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for computing cross-modality retrieval metrics."""

    ks: Sequence[int] = (1, 5, 10, 100)
    chunk_size: int = 32


def _get_features(embeddings: ModalityEmbeddings, feature_space: str) -> torch.Tensor:
    print(embeddings)
    if feature_space == "projected":
        features = embeddings.projected
    elif feature_space == "backbone":
        features = embeddings.backbone
    else:
        raise ValueError
    # elif feature_space == "raw":
    #     features = embeddings.raw
    # else:
    #     raise ValueError(f"Unsupported feature_space '{feature_space}'. Use 'projected' or 'raw'.")

    if features.ndim != 2:
        raise ValueError("Expected 2-D embeddings with shape (N, D)")

    return features.to(torch.float32)

def _batched_similarity(left: torch.Tensor, right: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Compute pairwise similarities in chunks to control memory usage."""

    if left.shape[1] != right.shape[1]:
        raise ValueError(
            "Embedding dimensionality mismatch between modalities: "
            f"{left.shape[1]} vs {right.shape[1]}"
        )

    similarities = []
    for start in range(0, left.shape[0], chunk_size):
        end = min(start + chunk_size, left.shape[0])
        chunk = left[start:end]
        similarities.append(chunk @ right.T)
    return torch.cat(similarities, dim=0)


def _batched_exterior_angle_similarity(
    left: torch.Tensor,   # hyperbolic points (time+space), shape [L, D]
    right: torch.Tensor,  # hyperbolic points (time+space), shape [R, D]
    chunk_size=256,
    curvature=None
) -> torch.Tensor:
    """
    Return S = -alpha(left_i, right_j) where alpha is the exterior angle in hyperbolic space.
    Larger values mean better matches (smaller angle), so you can keep existing 'topk' code.
    """
    
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"Embedding dimensionality mismatch: {left.shape[1]} vs {right.shape[1]}")

    device, dtype = left.device, left.dtype
    curvature = torch.as_tensor(curvature, device=device, dtype=dtype)
    right = right.to(device=device, dtype=dtype, non_blocking=True)

    sims = []
    for start in range(0, left.shape[0], chunk_size):
        end = min(start + chunk_size, left.shape[0])
        chunk = left[start:end]  # [B, D]
        # L.pairwise_oxy_angle returns the exterior angle matrix [B, R] in radians
        alpha = L.pairwise_oxy_angle(chunk, right, curvature)  # [B, R]
        sims.append(-alpha)  # negate so that higher = better
    return torch.cat(sims, dim=0)


def _recall_at_ks(similarities: torch.Tensor, ks: Iterable[int]) -> Dict[int, float]:
    """Compute recall@K for a similarity matrix assuming diagonal positives."""

    n = similarities.shape[0]
    target = torch.arange(n, device=similarities.device)
    metrics: Dict[int, float] = {}
    for k in ks:
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(k, similarities.shape[1])
        topk = torch.topk(similarities, k=k, dim=1).indices
        hits = (topk == target.unsqueeze(1)).any(dim=1).to(torch.float32)
        metrics[k] = float(hits.mean().item() * 100.0)
    return metrics


@torch.no_grad()
def compute_cross_modal_retrieval(
    
    s1_embeddings: torch.Tensor,
    s2_embeddings: torch.Tensor,
    curvature=None,
    *,
    config: RetrievalConfig | None = None,
) -> Dict[str, float]:
    """Compute S1<->S2 retrieval metrics for SSL4EO embeddings.

    Args:
        s1_embeddings: Embeddings for the Sentinel-1 modality.
        s2_embeddings: Embeddings for the Sentinel-2 modality.
        config: Optional :class:`RetrievalConfig` for metric parameters.

    Returns:
        Mapping from metric name to recall percentage.
    """

    if config is None:
        config = RetrievalConfig()

    # s1 = _get_features(s1_embeddings, feature_space)
    # s2 = _get_features(s2_embeddings, feature_space)

    if s1_embeddings.shape[0] != s2_embeddings.shape[0]:
        raise ValueError(
            "The number of Sentinel-1 and Sentinel-2 embeddings must match for retrieval evaluation."
        )

    if s1_embeddings.numel() == 0 or s2_embeddings.numel() == 0:
        raise ValueError("Cannot compute retrieval metrics on empty embeddings")

    # if loretnz ciip
    if curvature:
        print(curvature) 
        similarities = _batched_exterior_angle_similarity(s1_embeddings, s2_embeddings, config.chunk_size, curvature=curvature)
    else:
        similarities = _batched_similarity(s1_embeddings, s2_embeddings, config.chunk_size)

    s1_to_s2 = _recall_at_ks(similarities, config.ks)
    s2_to_s1 = _recall_at_ks(similarities.T, config.ks)
    print(similarities[0:5,0:5])

    results: Dict[str, float] = {}
    for k, value in s1_to_s2.items():
        results[f"s1_to_s2_r{k}"] = value
    for k, value in s2_to_s1.items():
        results[f"s2_to_s1_r{k}"] = value
    return results


__all__ = ["RetrievalConfig", "compute_cross_modal_retrieval"]