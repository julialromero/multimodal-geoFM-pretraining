"""Helper utilities for intrinsic dimension diagnostics and SSL4EO transforms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import skdim.id as id
import torch


def prepare_embeddings_for_id(tensor: torch.Tensor) -> Optional[np.ndarray]:
    """Convert embeddings to NumPy and drop degenerate entries."""

    if tensor is None:
        return None

    Z = tensor.detach().cpu().to(torch.float64).numpy()
    if Z.size == 0:
        return None

    uniques = np.unique(Z, axis=0)
    if len(uniques) != len(Z):
        logging.info("Removing %d duplicate embeddings before ID computation", len(Z) - len(uniques))
    Z = uniques

    var_per_dim = Z.var(axis=0)
    logging.info(
        "Embedding variance – min: %.6f, max: %.6f; zero-var dims: %d",
        var_per_dim.min(),
        var_per_dim.max(),
        np.sum(var_per_dim < 1e-10),
    )

    norms = np.linalg.norm(Z, axis=1)
    valid = (norms > 1e-6) & np.isfinite(norms)
    Z = Z[valid]
    if Z.size == 0:
        return None

    return Z


def compute_global_id_metrics(Z: np.ndarray) -> Dict[str, float]:
    """Compute FisherS, MLE, MoM, and TLE intrinsic dimensions for embeddings."""

    metrics: Dict[str, float] = {}

    def _compute_tle(Z: np.ndarray) -> float:
        tle = id.TLE()
        tle_pw = tle.fit_transform_pw(Z, n_neighbors=20)
        tle_pw = tle_pw[tle_pw >= 0]
        if tle_pw.size == 0:
            raise ValueError("TLE produced no valid pointwise estimates")
        return float(np.nanmean(tle_pw))

    metrics["fishers"] = float(id.FisherS().fit_transform(Z))
    metrics["mle"] = float(id.MLE(neighborhood_based=True).fit_transform(Z, n_neighbors=20))
    metrics["mom"] = float(id.MOM().fit_transform(Z, n_neighbors=20))
    metrics["tle"] = _compute_tle(Z)

    return metrics


def save_id_metrics(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    if not metrics:
        return

    with output_path.open("w", encoding="utf-8") as handle:
        for key in sorted(metrics):
            handle.write(f"{key}\n")
            for metric, value in metrics[key].items():
                handle.write(f"  {metric}: {value:.4f}\n")
            handle.write("\n")

    logging.info("Intrinsic dimension metrics saved to %s", output_path)
