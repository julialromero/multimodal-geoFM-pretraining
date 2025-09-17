#!/usr/bin/env python3
"""Utilities for visualizing embedding collapse and training instability.

This script loads saved CIIP embeddings across epochs and produces a suite of
plots that are commonly used to diagnose representation collapse:

* Positive vs. negative cosine similarity statistics.
* Cosine similarity histograms per epoch.
* Per-dimension embedding variance heatmaps and summaries.
* Covariance spectrum (singular values) and condition numbers.
* Linear probe accuracy overlays (optional, from CSV exports).
* t-SNE / UMAP projections of the embedding space across epochs.

The implementation follows the data loading patterns used in
``visualizations/ssl4eo/initialization_evaluation.py`` and
``ciip/evaluation/linearprobe_comparison.py`` so it should integrate with the
existing CIIP checkpoints and evaluation outputs.

Example usage::

    python -m visualizations.ssl4eo.embedding_collapse_diagnostics \
        --embedding-root /path/to/extracted_embeddings \
        --output-dir diagnostics/random_init \
        --negative-samples 20000 \
        --linear-probe-csv results.csv \
        --linear-probe-pattern "RandomInit-hal-epoch(\\d+)" \
        --linear-probe-k 1.0 \
        --tsne-samples 800 \
        --umap-samples 800

All plots are saved in the provided output directory; intermediate statistics
are exported as JSON/NumPy files for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    umap = None


_LOGGER = logging.getLogger("embedding_collapse")


@dataclass
class EpochEmbeddings:
    """Container for the embeddings associated with a single epoch."""

    label: str
    epoch_index: int
    path: Path
    s1: Optional[torch.Tensor]
    s2: Optional[torch.Tensor]
    uids: List[str]


@dataclass
class EpochMetrics:
    """Aggregated metrics computed for a single epoch."""

    label: str
    epoch_index: int
    sample_count: int
    cosine_positive: Optional[np.ndarray] = None
    cosine_negative: Optional[np.ndarray] = None
    s1_variance: Optional[np.ndarray] = None
    s2_variance: Optional[np.ndarray] = None
    s1_spectrum: Optional[np.ndarray] = None
    s2_spectrum: Optional[np.ndarray] = None
    s1_condition_number: Optional[float] = None
    s2_condition_number: Optional[float] = None

    def to_summary_dict(self) -> Dict[str, object]:
        """Return a JSON-friendly summary of the metrics."""

        def _safe_stats(values: Optional[np.ndarray]) -> Dict[str, float]:
            if values is None or values.size == 0:
                return {}
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        return {
            "label": self.label,
            "epoch_index": self.epoch_index,
            "sample_count": self.sample_count,
            "cosine_positive": _safe_stats(self.cosine_positive),
            "cosine_negative": _safe_stats(self.cosine_negative),
            "s1_variance": _safe_stats(self.s1_variance),
            "s2_variance": _safe_stats(self.s2_variance),
            "s1_condition_number": self.s1_condition_number,
            "s2_condition_number": self.s2_condition_number,
        }


def _natural_key(value: str) -> List[object]:
    """Return a key for natural sorting of strings containing numbers."""

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\\d+)", value)]


def discover_epoch_dirs(root: Path) -> List[Path]:
    """Return directories that correspond to individual epochs.

    If ``root`` already contains ``.pt`` files, the root itself is treated as a
    single epoch directory. Otherwise we return the immediate subdirectories
    sorted by name (using natural ordering).
    """

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Embedding root '{root}' does not exist")

    if any(root.glob("*.pt")) or any(root.glob("*.pth")):
        return [root]

    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: _natural_key(p.name))
    if not dirs:
        raise FileNotFoundError(
            f"No epoch directories found under '{root}'. Expected subdirectories containing .pt files."
        )
    return dirs


def infer_epoch_index(name: str, fallback: int) -> int:
    """Infer an integer epoch index from a directory or model name."""

    for pattern in (r"epoch[_-]?(\\d+)", r"(\\d+)"):
        match = re.search(pattern, name)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return fallback


def load_epoch_embeddings(
    epoch_dir: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[str]]:
    """Load paired S1/S2 embeddings from a directory of ``.pt`` files."""

    tensor_paths = sorted(
        list(epoch_dir.glob("*.pt")) + list(epoch_dir.glob("*.pth")) + list(epoch_dir.glob("*.pkl"))
    )
    if not tensor_paths:
        _LOGGER.warning("No embedding tensors found in %s", epoch_dir)
        return None, None, []

    if max_files is not None:
        tensor_paths = tensor_paths[:max_files]

    s1_vectors: List[torch.Tensor] = []
    s2_vectors: List[torch.Tensor] = []
    uids: List[str] = []

    for path in tensor_paths:
        try:
            sample = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive branch
            _LOGGER.error("Failed to load %s: %s", path, exc)
            continue

        s1 = sample.get("s1")
        s2 = sample.get("s2")
        if s1 is None or s2 is None:
            # We only keep samples that contain both modalities so that cross
            # cosine similarity is well defined.
            continue

        s1_tensor = torch.as_tensor(s1).reshape(-1)
        s2_tensor = torch.as_tensor(s2).reshape(-1)
        if s1_tensor.ndim != 1 or s2_tensor.ndim != 1:
            _LOGGER.debug("Skipping %s due to unexpected tensor shape", path)
            continue

        s1_vectors.append(s1_tensor)
        s2_vectors.append(s2_tensor)
        uids.append(str(sample.get("uid", path.stem)))

    if not s1_vectors or not s2_vectors:
        _LOGGER.warning("No paired embeddings found in %s", epoch_dir)
        return None, None, []

    s1_stack = torch.stack(s1_vectors, dim=0)
    s2_stack = torch.stack(s2_vectors, dim=0)
    return s1_stack, s2_stack, uids


def collect_embeddings(
    embedding_root: Path,
    *,
    max_files_per_epoch: Optional[int] = None,
) -> List[EpochEmbeddings]:
    """Load embeddings for all epochs contained in ``embedding_root``."""

    epoch_dirs = discover_epoch_dirs(embedding_root)
    epochs: List[EpochEmbeddings] = []
    for idx, epoch_dir in enumerate(epoch_dirs):
        s1, s2, uids = load_epoch_embeddings(epoch_dir, max_files=max_files_per_epoch)
        if s1 is None or s2 is None:
            _LOGGER.warning("Skipping epoch directory %s (no valid embeddings)", epoch_dir)
            continue
        epoch_index = infer_epoch_index(epoch_dir.name, fallback=idx)
        epochs.append(EpochEmbeddings(epoch_dir.name, epoch_index, epoch_dir, s1, s2, uids))

    epochs.sort(key=lambda e: (e.epoch_index, e.label))
    if not epochs:
        raise RuntimeError(f"No embeddings loaded from '{embedding_root}'")
    _LOGGER.info("Loaded embeddings for %d epochs from %s", len(epochs), embedding_root)
    return epochs


def compute_covariance(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Return the covariance matrix for ``tensor`` (N x D)."""

    if tensor.dim() != 2 or tensor.shape[0] < 2:
        return None
    centered = tensor.to(torch.float64) - tensor.mean(dim=0, keepdim=True).to(torch.float64)
    cov = centered.t().matmul(centered) / max(centered.shape[0] - 1, 1)
    return cov


def compute_condition_number(spectrum: np.ndarray, eps: float = 1e-12) -> float:
    """Compute a safe condition number from a descending spectrum."""

    if spectrum.size == 0:
        return float("nan")
    finite = spectrum[spectrum > eps]
    if finite.size == 0:
        return float("inf")
    smallest = float(finite[-1])
    largest = float(spectrum[0])
    if smallest <= eps:
        return float("inf")
    return largest / smallest


def compute_cosine_statistics(
    s1: torch.Tensor,
    s2: torch.Tensor,
    *,
    negative_samples: Optional[int],
    torch_generator: Optional[torch.Generator],
    numpy_generator: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute positive and (sampled) negative cosine similarities."""

    if s1.shape[0] != s2.shape[0]:
        count = min(s1.shape[0], s2.shape[0])
        s1 = s1[:count]
        s2 = s2[:count]

    s1_norm = F.normalize(s1, dim=1)
    s2_norm = F.normalize(s2, dim=1)
    positive = torch.sum(s1_norm * s2_norm, dim=1).cpu().numpy()

    n = s1_norm.shape[0]
    if n < 2:
        return positive, np.empty(0, dtype=np.float32)

    total_pairs = n * (n - 1)
    if negative_samples is None or negative_samples >= total_pairs:
        sim_matrix = s1_norm @ s2_norm.t()
        mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        mask.fill_diagonal_(False)
        negatives = sim_matrix[mask].cpu().numpy()
        if negative_samples is not None and negatives.size > negative_samples:
            indices = numpy_generator.choice(negatives.size, size=negative_samples, replace=False)
            negatives = negatives[indices]
        return positive, negatives

    if torch_generator is None:
        torch_generator = torch.Generator(device=s1.device)
    idx_i = torch.randint(0, n, (negative_samples,), generator=torch_generator)
    idx_j = torch.randint(0, n - 1, (negative_samples,), generator=torch_generator)
    idx_j = idx_j + (idx_j >= idx_i).long()
    negatives = torch.sum(s1_norm[idx_i] * s2_norm[idx_j], dim=1).cpu().numpy()
    return positive, negatives


def compute_epoch_metrics(
    epochs: Sequence[EpochEmbeddings],
    *,
    negative_samples: Optional[int],
    random_seed: int,
) -> List[EpochMetrics]:
    """Compute diagnostic metrics for each epoch."""

    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(random_seed)
    np_gen = np.random.default_rng(random_seed)

    results: List[EpochMetrics] = []
    for epoch in epochs:
        sample_count = epoch.s1.shape[0]
        s1_var = torch.var(epoch.s1, dim=0, unbiased=False).cpu().numpy()
        s2_var = torch.var(epoch.s2, dim=0, unbiased=False).cpu().numpy()

        metrics = EpochMetrics(label=epoch.label, epoch_index=epoch.epoch_index, sample_count=sample_count)
        metrics.s1_variance = s1_var
        metrics.s2_variance = s2_var

        cov_s1 = compute_covariance(epoch.s1)
        cov_s2 = compute_covariance(epoch.s2)
        if cov_s1 is not None:
            eigvals1 = torch.linalg.eigvalsh(cov_s1).flip(0).cpu().numpy()
            metrics.s1_spectrum = eigvals1
            metrics.s1_condition_number = compute_condition_number(eigvals1)
        if cov_s2 is not None:
            eigvals2 = torch.linalg.eigvalsh(cov_s2).flip(0).cpu().numpy()
            metrics.s2_spectrum = eigvals2
            metrics.s2_condition_number = compute_condition_number(eigvals2)

        positive, negative = compute_cosine_statistics(
            epoch.s1,
            epoch.s2,
            negative_samples=negative_samples,
            torch_generator=torch_gen,
            numpy_generator=np_gen,
        )
        metrics.cosine_positive = positive
        metrics.cosine_negative = negative

        results.append(metrics)

    return results


def plot_cosine_summary(metrics: Sequence[EpochMetrics], output_dir: Path) -> None:
    x = [m.epoch_index for m in metrics]
    labels = [m.label for m in metrics]

    pos_mean = [float(np.mean(m.cosine_positive)) if m.cosine_positive is not None else math.nan for m in metrics]
    pos_std = [float(np.std(m.cosine_positive)) if m.cosine_positive is not None else math.nan for m in metrics]
    neg_mean = [float(np.mean(m.cosine_negative)) if m.cosine_negative is not None and m.cosine_negative.size else math.nan for m in metrics]
    neg_std = [float(np.std(m.cosine_negative)) if m.cosine_negative is not None and m.cosine_negative.size else math.nan for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, pos_mean, yerr=pos_std, label="Positive pairs", marker="o", capsize=4)
    ax.errorbar(x, neg_mean, yerr=neg_std, label="Negative pairs", marker="s", capsize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine similarity statistics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cosine_similarity_summary.png", dpi=200)
    plt.close(fig)


def plot_cosine_histograms(metrics: Sequence[EpochMetrics], output_dir: Path, *, bins: int = 50) -> None:
    for metric in metrics:
        if metric.cosine_positive is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        axes[0].hist(metric.cosine_positive, bins=bins, color="#1f77b4", alpha=0.8)
        axes[0].set_title(f"Positive pairs — {metric.label}")
        axes[0].set_xlabel("Cosine similarity")
        axes[0].set_ylabel("Count")

        negatives = metric.cosine_negative if metric.cosine_negative is not None else np.empty(0)
        axes[1].hist(negatives, bins=bins, color="#ff7f0e", alpha=0.8)
        axes[1].set_title(f"Negative pairs — {metric.label}")
        axes[1].set_xlabel("Cosine similarity")
        fig.tight_layout()
        fig.savefig(output_dir / f"cosine_hist_{metric.label}.png", dpi=200)
        plt.close(fig)


def plot_variance_heatmap(
    metrics: Sequence[EpochMetrics],
    output_dir: Path,
    *,
    modality: str,
) -> None:
    variances = [m.s1_variance if modality == "s1" else m.s2_variance for m in metrics]
    if any(v is None for v in variances):
        _LOGGER.warning("Skipping variance heatmap for %s (missing data)", modality)
        return
    variance_matrix = np.stack(variances)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(variance_matrix, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Epoch")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.label for m in metrics])
    ax.set_title(f"{modality.upper()} variance per dimension")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Variance")
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_variance_heatmap.png", dpi=200)
    plt.close(fig)

    # Summary statistics per epoch
    stats = {
        "mean": np.mean(variance_matrix, axis=1),
        "median": np.median(variance_matrix, axis=1),
        "min": np.min(variance_matrix, axis=1),
        "max": np.max(variance_matrix, axis=1),
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [m.epoch_index for m in metrics]
    for name, values in stats.items():
        ax.plot(x, values, marker="o", label=name.capitalize())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Variance")
    ax.set_title(f"{modality.upper()} variance statistics")
    ax.set_xticks(x)
    ax.set_xticklabels([m.label for m in metrics], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_variance_summary.png", dpi=200)
    plt.close(fig)


def plot_spectrum(
    metrics: Sequence[EpochMetrics],
    output_dir: Path,
    *,
    modality: str,
    top_k: int,
) -> None:
    spectra = [m.s1_spectrum if modality == "s1" else m.s2_spectrum for m in metrics]
    valid_spectra = [s for s in spectra if s is not None]
    if not valid_spectra:
        _LOGGER.warning("Skipping spectrum plot for %s (missing data)", modality)
        return

    max_len = min(top_k, min(len(s) for s in valid_spectra))
    if max_len == 0:
        _LOGGER.warning("No eigenvalues available to plot for %s", modality)
        return

    x = [m.epoch_index for m in metrics]
    labels = [m.label for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx in range(max_len):
        values = [s[idx] if s is not None and len(s) > idx else math.nan for s in spectra]
        ax.plot(x, values, marker="o", label=f"λ{idx + 1}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"{modality.upper()} covariance spectrum (top {max_len})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_spectrum.png", dpi=200)
    plt.close(fig)

    condition_numbers = [m.s1_condition_number if modality == "s1" else m.s2_condition_number for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, condition_numbers, marker="o", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Condition number")
    ax.set_title(f"{modality.upper()} covariance condition number")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{modality}_condition_number.png", dpi=200)
    plt.close(fig)


def parse_linear_probe_csv(
    csv_path: Path,
    *,
    model_pattern: Optional[str],
    metric: str,
    k_value: Optional[float],
) -> Dict[int, Tuple[float, Optional[float]]]:
    """Parse a CSV produced by ``output_results_to_csv``.

    Returns a mapping from epoch index to (mean, std) for the requested
    configuration. When multiple rows match, later values override earlier ones.
    """

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        _LOGGER.warning("Linear probe CSV '%s' not found; skipping", csv_path)
        return {}

    pattern = re.compile(model_pattern) if model_pattern else None
    results: Dict[int, Tuple[float, Optional[float]]] = {}

    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        total_cols = len(header)
        split = (total_cols - 1) // 2
        k_headers = header[1 : 1 + split]
        std_headers = header[1 + split : 1 + 2 * split]
        k_values = [float(h.split("=")[1]) for h in k_headers]

        std_map = {float(h.split("=")[1]): idx for idx, h in enumerate(std_headers)}

        for row in reader:
            model_name = row[0]
            if pattern and not pattern.search(model_name):
                continue
            epoch_idx = infer_epoch_index(model_name, fallback=-1)
            if epoch_idx < 0:
                continue

            metrics_mean = [float(v) for v in row[1 : 1 + split]]
            metrics_std = [float(v) for v in row[1 + split : 1 + 2 * split]]

            if k_value is None and metrics_mean:
                results[epoch_idx] = (metrics_mean[0], metrics_std[0] if metrics_std else None)
                continue

            try:
                idx = min(range(len(k_values)), key=lambda i: abs(k_values[i] - k_value))
            except ValueError:  # pragma: no cover - defensive
                continue
            results[epoch_idx] = (metrics_mean[idx], metrics_std[idx] if metrics_std else None)

    return results


def plot_linear_probe_curve(
    linear_probe: Dict[int, Tuple[float, Optional[float]]],
    output_dir: Path,
    *,
    metric: str,
) -> None:
    if not linear_probe:
        _LOGGER.warning("No linear probe results to plot")
        return

    epochs = sorted(linear_probe.keys())
    means = [linear_probe[e][0] for e in epochs]
    stds = [linear_probe[e][1] for e in epochs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(epochs, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Linear probe {metric} vs. epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"linear_probe_{metric}.png", dpi=200)
    plt.close(fig)


def sample_for_projection(
    s1: torch.Tensor,
    s2: torch.Tensor,
    *,
    per_modality: int,
    generator: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a stacked array of embeddings and modality labels for projection."""

    if per_modality <= 0:
        raise ValueError("per_modality must be positive")

    count = min(s1.shape[0], s2.shape[0])
    if count == 0:
        raise ValueError("No paired embeddings available for projection")

    take = min(per_modality, count)
    indices = generator.choice(count, size=take, replace=False)
    stacked = torch.cat([s1[indices], s2[indices]], dim=0)
    labels = np.array(["S1"] * take + ["S2"] * take)
    return stacked.cpu().numpy(), labels


def compute_projection(
    data: np.ndarray,
    *,
    method: str,
    random_state: int,
) -> Optional[np.ndarray]:
    if data.shape[0] < 3:
        return None

    if method == "tsne":
        perplexity = min(30, data.shape[0] - 1)
        if perplexity < 1:
            return None
        reducer = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state)
    elif method == "umap":
        if umap is None:
            _LOGGER.warning("UMAP is not installed; skipping projection")
            return None
        reducer = umap.UMAP(n_components=2, random_state=random_state, init="spectral")
    else:
        raise ValueError(f"Unknown projection method: {method}")

    return reducer.fit_transform(data)


def plot_projection(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], label=label, alpha=0.7, s=18)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def export_metrics(metrics: Sequence[EpochMetrics], output_dir: Path) -> None:
    summary = [m.to_summary_dict() for m in metrics]
    with (output_dir / "metrics_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    s1_var = {m.label: m.s1_variance for m in metrics if m.s1_variance is not None}
    s2_var = {m.label: m.s2_variance for m in metrics if m.s2_variance is not None}
    s1_spec = {m.label: m.s1_spectrum for m in metrics if m.s1_spectrum is not None}
    s2_spec = {m.label: m.s2_spectrum for m in metrics if m.s2_spectrum is not None}

    if s1_var:
        np.savez(output_dir / "s1_variance.npz", **s1_var)
    if s2_var:
        np.savez(output_dir / "s2_variance.npz", **s2_var)
    if s1_spec:
        np.savez(output_dir / "s1_spectrum.npz", **s1_spec)
    if s2_spec:
        np.savez(output_dir / "s2_spectrum.npz", **s2_spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedding collapse diagnostic visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python -m visualizations.ssl4eo.embedding_collapse_diagnostics \\
                --embedding-root /path/to/extracted_embeddings \\
                --output-dir diagnostics/random_init
            """
        ).strip(),
    )
    parser.add_argument("--embedding-root", type=Path, required=True, help="Directory containing per-epoch embeddings")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to store plots and metrics")
    parser.add_argument(
        "--max-files-per-epoch",
        type=int,
        default=None,
        help="Maximum number of embedding files to load per epoch (for faster debugging)",
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=50000,
        help="Number of negative pairs to sample for cosine similarity (None = all)",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--spectrum-top-k", type=int, default=10, help="Number of leading eigenvalues to plot")
    parser.add_argument(
        "--linear-probe-csv",
        type=Path,
        default=None,
        help="Optional CSV from output_results_to_csv containing linear probe metrics",
    )
    parser.add_argument(
        "--linear-probe-pattern",
        type=str,
        default=None,
        help="Regex identifying rows that correspond to the embeddings under study",
    )
    parser.add_argument(
        "--linear-probe-k",
        type=float,
        default=None,
        help="Desired k/percent column from the linear probe CSV (defaults to the first column)",
    )
    parser.add_argument(
        "--linear-probe-metric",
        type=str,
        default="accuracy",
        choices=("accuracy", "f1"),
        help="Metric name used in the CSV file",
    )
    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=0,
        help="Number of samples per modality for t-SNE projections (0 = skip)",
    )
    parser.add_argument(
        "--umap-samples",
        type=int,
        default=0,
        help="Number of samples per modality for UMAP projections (0 = skip)",
    )
    parser.add_argument(
        "--cosine-hist-bins",
        type=int,
        default=50,
        help="Number of bins for cosine similarity histograms",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = collect_embeddings(args.embedding_root, max_files_per_epoch=args.max_files_per_epoch)
    metrics = compute_epoch_metrics(epochs, negative_samples=args.negative_samples, random_seed=args.random_seed)

    export_metrics(metrics, output_dir)
    plot_cosine_summary(metrics, output_dir)
    plot_cosine_histograms(metrics, output_dir, bins=args.cosine_hist_bins)
    plot_variance_heatmap(metrics, output_dir, modality="s1")
    plot_variance_heatmap(metrics, output_dir, modality="s2")
    plot_spectrum(metrics, output_dir, modality="s1", top_k=args.spectrum_top_k)
    plot_spectrum(metrics, output_dir, modality="s2", top_k=args.spectrum_top_k)

    if args.linear_probe_csv:
        linear_probe = parse_linear_probe_csv(
            args.linear_probe_csv,
            model_pattern=args.linear_probe_pattern,
            metric=args.linear_probe_metric,
            k_value=args.linear_probe_k,
        )
        plot_linear_probe_curve(linear_probe, output_dir, metric=args.linear_probe_metric)

    np_gen = np.random.default_rng(args.random_seed)
    if args.tsne_samples > 0:
        for epoch in epochs:
            try:
                data, labels = sample_for_projection(epoch.s1, epoch.s2, per_modality=args.tsne_samples, generator=np_gen)
            except ValueError:
                continue
            coords = compute_projection(data, method="tsne", random_state=args.random_seed)
            if coords is None:
                continue
            plot_projection(coords, labels, output_dir / f"tsne_{epoch.label}.png", title=f"t-SNE — {epoch.label}")

    if args.umap_samples > 0 and umap is not None:
        for epoch in epochs:
            try:
                data, labels = sample_for_projection(epoch.s1, epoch.s2, per_modality=args.umap_samples, generator=np_gen)
            except ValueError:
                continue
            coords = compute_projection(data, method="umap", random_state=args.random_seed)
            if coords is None:
                continue
            plot_projection(coords, labels, output_dir / f"umap_{epoch.label}.png", title=f"UMAP — {epoch.label}")
    elif args.umap_samples > 0:
        _LOGGER.warning("UMAP requested but not installed; skipping")

    _LOGGER.info("Saved plots and metrics to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
