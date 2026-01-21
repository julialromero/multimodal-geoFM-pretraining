#!/usr/bin/env python3
"""Compute hyperbolic radius distributions for SEN12MS S1/S2 pairs."""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import random
import pickle
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from scipy.cluster import hierarchy
from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ciip import lorentz as L
from ciip.evaluation.model_utils import build_evaluation_adapter
from ciip.evaluation.sen12ms_retrieval import Sen12MSRetrievalDataset
from ciip.evaluation.unified_evaluation import _infer_model_in_channels
from ciip.model_ciip import LorentzCIIP
from visualizations.ssl4eo.embedding_collapse_diagnostics import extract_embeddings_for_dataset
from visualizations.ssl4eo.hyperbolic_visualization import compute_hyperbolic_context

_LOG = logging.getLogger("sen12ms_hyperbolic_radii")

IGBP_SIMPLE_MAP = {
    1: "Forest",
    2: "Shrubland",
    3: "Savanna",
    4: "Grassland",
    5: "Wetlands",
    6: "Croplands",
    7: "Urban/Built-up",
    8: "Snow/Ice",
    9: "Barren",
    10: "Water",
}


@dataclass
class Sen12MSHyperbolicRadiusConfig:
    dataset_root: Path = Path("/local/ms-data/SEN12MS")
    output_dir: Path = Path("results/sen12ms_hyperbolic_radii")
    seasons: Sequence[str] = (
        "ROIs2017_winter",
        "ROIs1970_fall",
        "ROIs1868_summer",
        "ROIs1158_spring",
    )
    label_path: Optional[Path] = Path("/local/ms-data/SEN12MS/labels/single_label_IGBPsimple_ClsNum.pkl")
    model_type: str = "ciip_checkpoint"
    checkpoint: Optional[Path] = None
    model_weights: Optional[str] = None
    max_patches_per_season: Optional[int] = None
    image_size: int = 224
    use_ssl4eo_normalization: bool = True
    num_samples: int = 1000
    seed: int = 0
    device: Optional[str] = None


def _sample_subset(dataset: Sen12MSRetrievalDataset, num_samples: int, seed: int) -> Subset:
    rng = random.Random(seed)
    count = min(num_samples, len(dataset))
    indices = rng.sample(range(len(dataset)), k=count)
    return Subset(dataset, indices)


def _plot_radius_histogram(
    s1_radii: torch.Tensor,
    s2_radii: torch.Tensor,
    output: Path,
) -> None:
    s1_np = s1_radii.cpu().numpy()
    s2_np = s2_radii.cpu().numpy()

    bins = max(25, int(math.sqrt(max(len(s1_np), len(s2_np)))))
    plt.figure(figsize=(7, 5))
    plt.hist(s1_np, bins=bins, density=True, alpha=0.55, label="S1", color="tab:blue")
    plt.hist(s2_np, bins=bins, density=True, alpha=0.55, label="S2", color="tab:orange")

    # Optional KDE overlay (only if scipy is available)
    try:
        from scipy.stats import gaussian_kde

        x_min = min(float(s1_np.min()), float(s2_np.min()))
        x_max = max(float(s1_np.max()), float(s2_np.max()))
        xs = torch.linspace(x_min, x_max, steps=400).numpy()
        plt.plot(xs, gaussian_kde(s1_np)(xs), color="tab:blue", lw=2, label="S1 KDE")
        plt.plot(xs, gaussian_kde(s2_np)(xs), color="tab:orange", lw=2, label="S2 KDE")
    except Exception as exc:  # pragma: no cover - optional dependency
        _LOG.info("Skipping KDE overlay (scipy unavailable?): %s", exc)

    plt.xlabel("Hyperbolic radius (geodesic distance from origin)")
    plt.ylabel("Density")
    plt.title("SEN12MS hyperbolic radii by modality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def _split_intra_inter(dist_matrix: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    d = dist_matrix.cpu().numpy()
    lbl = labels.cpu().numpy()
    triu = np.triu_indices_from(d, k=1)
    same = lbl[triu[0]] == lbl[triu[1]]
    intra = d[triu][same]
    inter = d[triu][~same]
    return intra, inter


def _plot_intra_inter_hist(
    intra_s1: np.ndarray,
    inter_s1: np.ndarray,
    intra_s2: np.ndarray,
    inter_s2: np.ndarray,
    intra_all: np.ndarray,
    inter_all: np.ndarray,
    output: Path,
) -> None:
    datasets = [
        ("S1", intra_s1, inter_s1),
        ("S2", intra_s2, inter_s2),
        ("S1+S2", intra_all, inter_all),
    ]
    all_vals = np.concatenate([intra_s1, inter_s1, intra_s2, inter_s2, intra_all, inter_all])
    bins = min(60, max(20, int(np.sqrt(all_vals.size))))
    vmin, vmax = all_vals.min(), all_vals.max()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
    for ax, (title, intra, inter) in zip(axes, datasets):
        ax.hist(intra, bins=bins, range=(vmin, vmax), alpha=0.6, density=True, label="Intra-class", color="tab:blue")
        ax.hist(inter, bins=bins, range=(vmin, vmax), alpha=0.6, density=True, label="Inter-class", color="tab:orange")
        ax.set_title(title)
        ax.set_xlabel("Hyperbolic pairwise distance")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_class_violin(
    s1_radii: torch.Tensor,
    s2_radii: torch.Tensor,
    labels: Optional[torch.Tensor],
    output: Path,
) -> None:
    if labels is None:
        _LOG.info("No labels provided; skipping per-class violin plot.")
        return

    labels_np = labels.cpu().numpy()
    s1_np = s1_radii.cpu().numpy()
    s2_np = s2_radii.cpu().numpy()

    classes = sorted(set(labels_np.tolist()))
    if not classes:
        _LOG.info("Labels tensor empty; skipping per-class violin plot.")
        return

    s1_data = [s1_np[labels_np == cls] for cls in classes]
    s2_data = [s2_np[labels_np == cls] for cls in classes]
    class_names = [IGBP_SIMPLE_MAP.get(int(cls), f"Unknown({int(cls)})") for cls in classes]
    for cls, s1_vals, s2_vals in zip(classes, s1_data, s2_data):
        if len(s1_vals) != len(s2_vals):
            _LOG.warning("Count mismatch for class %s: S1=%d, S2=%d", cls, len(s1_vals), len(s2_vals))
    # get the int(cls) values that are not in the dict and priht
    unknown_classes = [int(cls) for cls in classes if int(cls) not in IGBP_SIMPLE_MAP]
    if unknown_classes:
        _LOG.warning("Unknown class IDs found in labels: %s", unknown_classes)

    # print class counts of the labels
    for cls in classes:
        count = np.sum(labels_np == cls)
        _LOG.info("Class %s (%s): %d samples", cls, IGBP_SIMPLE_MAP.get(int(cls), "Unknown"), count)
        


    positions = list(range(len(classes)))

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=(max(10, len(classes) * 1.6), 6),
        constrained_layout=True,
    )

    parts1 = ax1.violinplot(s1_data, positions=positions, widths=0.8, showmeans=True)
    parts2 = ax2.violinplot(s2_data, positions=positions, widths=0.8, showmeans=True)

    for pc in parts1["bodies"]:
        pc.set_facecolor("tab:blue")
        pc.set_alpha(0.5)
    parts1["cbars"].set_color("tab:blue")
    parts1["cmeans"].set_color("tab:blue")
    ax1.set_ylabel("Hyperbolic radius")
    ax1.set_title("S1")
    ax1.grid(alpha=0.3, axis="y")

    for pc in parts2["bodies"]:
        pc.set_facecolor("tab:orange")
        pc.set_alpha(0.5)
    parts2["cbars"].set_color("tab:orange")
    parts2["cmeans"].set_color("tab:orange")
    ax2.set_title("S2")
    ax2.grid(alpha=0.3, axis="y")

    for ax in (ax1, ax2):
        ax.set_xticks(positions)
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_xlabel("IGBP simple class ID")

    # Align y-axis limits explicitly to ensure matching scales.
    all_vals = np.concatenate([np.concatenate(s1_data), np.concatenate(s2_data)])
    y_min, y_max = all_vals.min(), all_vals.max()
    margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    for ax in (ax1, ax2):
        ax.set_ylim(y_min - margin, y_max + margin)

    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_class_boxplot(
    s1_radii: torch.Tensor,
    s2_radii: torch.Tensor,
    labels: Optional[torch.Tensor],
    output: Path,
) -> None:
    if labels is None:
        _LOG.info("No labels provided; skipping per-class box plot.")
        return

    labels_np = labels.cpu().numpy()
    s1_np = s1_radii.cpu().numpy()
    s2_np = s2_radii.cpu().numpy()

    classes = sorted(set(labels_np.tolist()))
    if not classes:
        _LOG.info("Labels tensor empty; skipping per-class box plot.")
        return

    s1_data = [s1_np[labels_np == cls] for cls in classes]
    s2_data = [s2_np[labels_np == cls] for cls in classes]
    class_names = [IGBP_SIMPLE_MAP.get(int(cls), f"Unknown({int(cls)})") for cls in classes]
    positions = np.arange(len(classes))

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=True,
        figsize=(max(10, len(classes) * 1.6), 6),
        constrained_layout=True,
    )

    ax1.boxplot(s1_data, positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor="tab:blue", alpha=0.5))
    ax2.boxplot(s2_data, positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor="tab:orange", alpha=0.5))

    ax1.set_title("S1")
    ax1.set_ylabel("Hyperbolic radius")
    ax2.set_title("S2")
    for ax in (ax1, ax2):
        ax.set_xticks(positions)
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_xlabel("IGBP simple class ID")
        ax.grid(alpha=0.3, axis="y")

    # Keep y-lims consistent with violin plot by using combined data.
    all_vals = np.concatenate([np.concatenate(s1_data), np.concatenate(s2_data)])
    y_min, y_max = all_vals.min(), all_vals.max()
    margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    for ax in (ax1, ax2):
        ax.set_ylim(y_min - margin, y_max + margin)

    fig.savefig(output, dpi=200)
    plt.close(fig)


def _collect_pos_neg_by_class(
    labels: torch.Tensor,
    cross_dist: torch.Tensor,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    lbl = labels.cpu().numpy()
    dist = cross_dist.cpu().numpy()
    pos_by_cls: dict[int, np.ndarray] = {}
    neg_by_cls: dict[int, np.ndarray] = {}
    classes = sorted(set(lbl.tolist()))
    for cls in classes:
        idxs = np.where(lbl == cls)[0]
        if idxs.size == 0:
            continue
        pos_by_cls[cls] = np.diag(dist)[idxs]
        if idxs.size > 1:
            rolled = np.roll(idxs, -1)
            neg_by_cls[cls] = dist[idxs, rolled]
        else:
            neg_by_cls[cls] = np.array([])
    return pos_by_cls, neg_by_cls


def _plot_pos_neg_box_per_class(
    pos_by_cls: dict[int, np.ndarray],
    neg_by_cls: dict[int, np.ndarray],
    output: Path,
) -> None:
    classes = sorted(pos_by_cls.keys())
    if not classes:
        _LOG.info("No classes for pos/neg box plot; skipping.")
        return
    class_names = [IGBP_SIMPLE_MAP.get(int(c), f"Unknown({int(c)})") for c in classes]

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 1.6), 6), constrained_layout=True)
    positions_pos = np.arange(len(classes)) * 2.0
    positions_neg = positions_pos + 0.8

    ax.boxplot(
        [pos_by_cls[c] for c in classes],
        positions=positions_pos,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="tab:green", alpha=0.5),
    )
    ax.boxplot(
        [neg_by_cls[c] for c in classes],
        positions=positions_neg,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="tab:red", alpha=0.5),
    )

    all_vals = np.concatenate([v for v in pos_by_cls.values()] + [v for v in neg_by_cls.values() if v.size > 0])
    y_min, y_max = all_vals.min(), all_vals.max()
    margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xticks(positions_pos + 0.4)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylabel("Hyperbolic distance")
    ax.set_title("S1–S2 paired distances per class (pos vs. matched neg)")
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, facecolor="tab:green", alpha=0.5),
         plt.Rectangle((0, 0), 1, 1, facecolor="tab:red", alpha=0.5)],
        ["Positive (paired)", "Matched negative"],
        loc="upper right",
    )
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _compute_class_centroids(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    curvature: torch.Tensor,
) -> dict[int, torch.Tensor]:
    centroids = {}
    for cls in sorted(set(labels.tolist())):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        tang = L.log_map0(embeddings[mask], curvature)
        mean_tang = tang.mean(dim=0, keepdim=True)
        centroids[int(cls)] = L.exp_map0(mean_tang, curvature).squeeze(0)
    return centroids


def lorentz_to_poincare(points: torch.Tensor,
                        curvature: torch.Tensor | float) -> torch.Tensor:
    """
    points: (N, D+1) Lorentz hyperboloid coords, first dim is time-like x0.
    returns: (N, D) Poincaré ball coords.
    """
    if isinstance(curvature, torch.Tensor):
        c = float(curvature.detach().cpu().item())
    else:
        c = float(curvature)

    x0 = points[..., 0]          # (N,)
    x_spatial = points[..., 1:]  # (N, D)

    # rescale to curvature -1 (up to scale), if needed
    if c != 1.0:
        scale = math.sqrt(c)
        x0 = x0 * scale
        x_spatial = x_spatial * scale

    denom = (x0 + 1.0).unsqueeze(-1).clamp_min(1e-8)  # (N, 1)
    b = x_spatial / denom                             # (N, D)
    return b


def _plot_poincare_scatter(
    s1_points: torch.Tensor,
    s2_points: torch.Tensor,
    labels: torch.Tensor,
    curvature: torch.Tensor,
    output: Path,
    max_points: int = 5000,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    s1_np = s1_points.detach().cpu().numpy()
    s2_np = s2_points.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    n = labels_np.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        s1_np = s1_np[idx]
        s2_np = s2_np[idx]
        labels_np = labels_np[idx]

    b = np.vstack([s1_np, s2_np])
    pca = PCA(n_components=2)
    b2 = pca.fit_transform(b)

    s1_2d = b2[:s1_np.shape[0], :]
    s2_2d = b2[s1_np.shape[0]:, :]

    classes = np.unique(labels_np)
    cmap = plt.get_cmap("tab10")
    class_to_color = {c: cmap(i % 10) for i, c in enumerate(classes)}

    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_cent = fig.add_subplot(gs[0, 1])
    for c in classes:
        mask = labels_np == c
        color = class_to_color[c]
        name = IGBP_SIMPLE_MAP.get(int(c), f"Class {int(c)}")
        ax_pca.scatter(s1_2d[mask, 0], s1_2d[mask, 1], s=20, marker="o", alpha=1.0, color=color, label=name)
        ax_pca.scatter(s2_2d[mask, 0], s2_2d[mask, 1], s=20, marker="^", alpha=1.0, color=color, label=name)

    class_handles = []
    for c in classes:
        color = class_to_color[c]
        name = IGBP_SIMPLE_MAP.get(int(c), f"Class {int(c)}")
        class_handles.append(plt.Line2D([0], [0], marker="o", color=color, linestyle="", markersize=6, label=name))
    modality_handles = [
        plt.Line2D([0], [0], marker="o", color="k", linestyle="", markersize=8, label="S1 (circle)"),
        plt.Line2D([0], [0], marker="^", color="k", linestyle="", markersize=8, label="S2 (triangle)"),
    ]
    ax_pca.legend(class_handles + modality_handles, [h.get_label() for h in class_handles + modality_handles], fontsize=7, loc="upper left", ncol=2)
    ax_pca.set_title("Poincaré projection (PCA)")
    ax_pca.set_xlabel("dim 1")
    ax_pca.set_ylabel("dim 2")
    ax_pca.set_aspect("equal", adjustable="box")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_pca.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.6)
    ax_pca.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
    ax_pca.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)

    all_x = np.concatenate([s1_2d[:, 0], s2_2d[:, 0]])
    all_y = np.concatenate([s1_2d[:, 1], s2_2d[:, 1]])
    ax_pca.set_xlim(all_x.min()-0.01, all_x.max()+0.01)
    ax_pca.set_ylim(all_y.min()-0.01, all_y.max()+0.01)
    _LOG.info("Poincaré scatter (PCA) x range: [%f, %f]", all_x.min(), all_x.max())
    _LOG.info("Poincaré scatter (PCA) y range: [%f, %f]", all_y.min(), all_y.max())

    # Centroid subplot
    for c in classes:
        mask = labels_np == c
        color = class_to_color[c]
        name = IGBP_SIMPLE_MAP.get(int(c), f"Class {int(c)}")
        c_s1_pca = s1_2d[mask].mean(axis=0)
        c_s2_pca = s2_2d[mask].mean(axis=0)
        ax_cent.scatter(c_s1_pca[0], c_s1_pca[1], s=60, marker="o", color=color, alpha=0.9, label=f"{name} PCA S1")
        ax_cent.scatter(c_s2_pca[0], c_s2_pca[1], s=60, marker="^", color=color, alpha=0.9, label=f"{name} PCA S2")

    class_handles_cent = []
    for c in classes:
        color = class_to_color[c]
        name = IGBP_SIMPLE_MAP.get(int(c), f"Class {int(c)}")
        class_handles_cent.append(plt.Line2D([0], [0], marker="o", color=color, linestyle="", markersize=6, label=name))
    marker_handles_cent = [
        plt.Line2D([0], [0], marker="o", color="k", linestyle="", markersize=8, label="PCA S1 (filled circle)"),
        plt.Line2D([0], [0], marker="^", color="k", linestyle="", markersize=8, label="PCA S2 (filled triangle)"),
    ]
    ax_cent.legend(class_handles_cent + marker_handles_cent, [h.get_label() for h in class_handles_cent + marker_handles_cent], fontsize=7, loc="upper right", ncol=2)
    ax_cent.set_title("Class centroids in PCA")
    ax_cent.set_xlabel("dim 1")
    ax_cent.set_ylabel("dim 2")
    ax_cent.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
    ax_cent.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)

    fig.savefig(output, dpi=200)
    plt.close(fig)


def poincare_2d_from_lorentz(
    points: torch.Tensor,
    curvature: torch.Tensor | float,
    n_components: int = 2,
) -> np.ndarray:
    """
    points: (N, D+1) Lorentz hyperboloid embeddings (time-like coord first).
    curvature: scalar (tensor or float), positive (space curvature -curvature).
    returns: (N, 2) Poincaré disk coords (PCA applied in ball coordinates).
    """

    # Map to full Poincaré ball, then PCA in the ball.
    full_ball = lorentz_to_poincare(points, curvature)      # (N, D)
    pca = PCA(n_components=n_components)
    b2 = pca.fit_transform(full_ball.detach().cpu().numpy())

    return b2


def _plot_poincare_disk(
    b2_s1: np.ndarray,
    b2_s2: np.ndarray,
    labels: Optional[torch.Tensor],
    output: Path,
) -> None:
    if labels is None:
        _LOG.info("No labels for Poincaré disk plot; skipping.")
        return
    labels_np = labels.detach().cpu().numpy()
    classes = np.unique(labels_np)
    cmap = plt.get_cmap("tab10")
    class_to_color = {c: cmap(i % 10) for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(7, 7))
    for c in classes:
        mask = labels_np == c
        color = class_to_color[c]
        name = IGBP_SIMPLE_MAP.get(int(c), f"Class {int(c)}")
        ax.scatter(b2_s1[mask, 0], b2_s1[mask, 1], s=14, alpha=0.7, color=color, marker="o", label=name)
        ax.scatter(b2_s2[mask, 0], b2_s2[mask, 1], s=14, alpha=0.7, color=color, marker="^", label=name)

    class_handles = [
        plt.Line2D([0], [0], marker="o", color=class_to_color[c], linestyle="", markersize=6, label=IGBP_SIMPLE_MAP.get(int(c), f"Class {int(c)}"))
        for c in classes
    ]
    modality_handles = [
        plt.Line2D([0], [0], marker="o", color="k", linestyle="", markersize=8, label="S1 (circle)"),
        plt.Line2D([0], [0], marker="^", color="k", linestyle="", markersize=8, label="S2 (triangle)"),
    ]

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")
    all_x = np.concatenate([b2_s1[:, 0], b2_s2[:, 0]])
    all_y = np.concatenate([b2_s1[:, 1], b2_s2[:, 1]])
    r_max = max(np.sqrt(all_x**2 + all_y**2).max(), 1.0)
    pad = 0.05 * r_max
    ax.set_xlim(-r_max - pad, r_max + pad)
    ax.set_ylim(-r_max - pad, r_max + pad)
    ax.set_xlabel("Poincaré dim 1")
    ax.set_ylabel("Poincaré dim 2")
    ax.set_title("Poincaré disk: S1 vs S2")
    ax.legend(class_handles + modality_handles, [h.get_label() for h in class_handles + modality_handles], fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_hyperbolic_entailment(
    points: torch.Tensor,
    labels: torch.Tensor,
    curvature: torch.Tensor | float,
    apertures: torch.Tensor,
    output: Path,
    max_points: int = 1000,
    num_roots: int = 50,
    seed: int = 0,
    modalities: Optional[np.ndarray] = None,
) -> None:
    """
    Visualize true hyperbolic cone entailment, *plotted on the Poincaré disk*.

    Roots = points with smallest hyperbolic radius (closest to the hyperbolic origin).
    For each root i, we draw arrows i -> j for all j that lie inside its cone
    and are further from the origin in hyperbolic distance.

    Entailment is computed in Lorentz space; only the 2D coordinates are changed
    to a Poincaré disk embedding for visualization.
    """
    rng = np.random.default_rng(seed)

    # ----- Move to CPU / numpy and subsample consistently -----
    pts = points.detach().cpu()
    lbls = labels.detach().cpu().numpy()
    aps = apertures.detach().cpu()
    mods = modalities.astype(str) if modalities is not None else None

    N = lbls.shape[0]
    if N == 0:
        _LOG.warning("No points provided to plot_hyperbolic_entailment; skipping.")
        return

    if N > max_points:
        idx = rng.choice(N, size=max_points, replace=False)
        pts = pts[idx]
        lbls = lbls[idx]
        aps = aps[idx]
        if mods is not None:
            mods = mods[idx]
        N = max_points

    # ----- Hyperbolic geometry: radii and angles (true Lorentz cones) -----
    if isinstance(curvature, torch.Tensor):
        c_val = float(curvature.detach().cpu().item())
    else:
        c_val = float(curvature)

    # Tangent-space vectors at origin; norm = hyperbolic radius
    t_vecs = L.log_map0(pts, c_val)      # (N, D)
    radii = torch.norm(t_vecs, dim=-1)   # (N,)

    # Pairwise exterior angles at the origin between rays to points
    angles = L.pairwise_oxy_angle(pts, pts, c_val).detach().cpu().numpy()  # (N, N)

    radii_np = radii.detach().cpu().numpy()
    aps_np = aps.detach().cpu().numpy()   # cone apertures per point

    # Roots = smallest hyperbolic radii
    num_roots = min(num_roots, N)
    root_indices = np.argsort(radii_np)[:num_roots]

    # ---- True entailment rule in hyperbolic space ----
    # i ⇒ j iff:
    #   (1) r_j > r_i + delta_r   (strictly deeper)
    #   (2) angle(i,j) <= aperture_i + slack
    delta_r = 1e-3
    angle_slack = np.deg2rad(2.0)

    edges: list[tuple[int, int]] = []
    for i in root_indices:
        ri = radii_np[i]
        a_i = aps_np[i]
        for j in range(N):
            if j == i:
                continue
            rj = radii_np[j]
            if rj <= ri + delta_r:
                continue
            ang_ij = angles[i, j]
            if ang_ij <= a_i + angle_slack:
                edges.append((i, j))

    entailed_indices = {j for _, j in edges}
    _LOG.info(
        "Hyperbolic entailment: %d roots, %d entailed nodes, %d edges",
        len(root_indices),
        len(entailed_indices),
        len(edges),
    )

    # ----- 2D visualization: Poincaré disk coordinates -----
    # Entailment *remains* defined in hyperbolic space; only coordinates change.
    b2 = poincare_2d_from_lorentz(pts, c_val)   # (N, 2) numpy array
    coords_2d = b2                               # rename for clarity

    # Colors by class
    classes = np.unique(lbls)
    cmap = plt.get_cmap("tab10")
    class_to_color = {c: cmap(i % 10) for i, c in enumerate(classes)}
    def _marker_for(idx: int) -> str:
        if mods is None:
            return "o"
        return "^" if mods[idx].lower() == "s1" else "o"

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    # All points (light)
    for idx in range(N):
        c = lbls[idx]
        color = class_to_color[c]
        ax.scatter(
            coords_2d[idx, 0],
            coords_2d[idx, 1],
            s=10,
            marker=_marker_for(idx),
            color=color,
            alpha=0.05,
        )

    # Highlight roots (parents)
    for i in root_indices:
        c = lbls[i]
        color = class_to_color[c]
        ax.scatter(
            coords_2d[i, 0],
            coords_2d[i, 1],
            s=80,
            marker=_marker_for(i),
            facecolors=color,
            edgecolors="k",
            linewidths=1.5,
            alpha=0.7,
        )
        ax.scatter(
            coords_2d[i, 0],
            coords_2d[i, 1],
            s=40,
            marker=_marker_for(i),
            facecolors=color,
            edgecolors=color,
            linewidths=1.2,
        )

    # Highlight entailed nodes (children)
    for j in entailed_indices:
        c = lbls[j]
        color = class_to_color[c]
        ax.scatter(
            coords_2d[j, 0],
            coords_2d[j, 1],
            s=40,
            marker=_marker_for(j),
            color=color,
            alpha=0.9,
        )

    # Draw arrows for all entailment edges (straight in Poincaré coords)
    for i, j in edges:
        x0, y0 = coords_2d[i]
        x1, y1 = coords_2d[j]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="k",
                alpha=0.3,
                linewidth=0.5,
            ),
        )

    # Legend: classes + role
    class_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color=color,
            linestyle="",
            markersize=6,
            label=IGBP_SIMPLE_MAP.get(int(c), f"Class {int(c)}"),
        )
        for c, color in class_to_color.items()
    ]
    role_handles = [
        plt.Line2D(
            [0], [0],
            marker="^",
            color="k",
            linestyle="",
            markersize=8,
            markerfacecolor="k",
            label="Root (S1 triangle)",
        ),
        plt.Line2D(
            [0], [0],
            marker="o",
            color="k",
            linestyle="",
            markersize=8,
            label="Entailed (S2 circle)",
        ),
    ]
    ax.legend(
        class_handles + role_handles,
        [h.get_label() for h in class_handles + role_handles],
        fontsize=7,
        loc="upper right",
        ncol=2,
    )

    # Poincaré disk cosmetics
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.6)  # unit circle
    ax.set_aspect("equal", adjustable="box")
    # set to min and max of data
    ax.set_xlim(
        coords_2d[:, 0].min() - 0.05,
        coords_2d[:, 0].max() + 0.05,
    )
    ax.set_ylim(
        coords_2d[:, 1].min() - 0.05,
        coords_2d[:, 1].max() + 0.05,
    )
    ax.set_xlabel("Poincaré dim 1")
    ax.set_ylabel("Poincaré dim 2")
    ax.set_title(f"Hyperbolic cone entailment from {len(root_indices)} roots (Poincaré disk)")

    fig.savefig(output, dpi=200)
    plt.close(fig)



def _plot_class_dendrogram(
    centroids: dict[int, torch.Tensor],
    curvature: torch.Tensor | float,
    output: Path,
    ax=None,
    names: Optional[list[str]] = None,
) -> tuple[Optional[np.ndarray], float]:
    if not centroids:
        _LOG.info("No centroids available; skipping dendrogram.")
        return None, 0.0
    classes = sorted(centroids.keys())
    class_names = names or [IGBP_SIMPLE_MAP.get(int(c), f"Unknown({int(c)})") for c in classes]
    mats = torch.stack([centroids[c] for c in classes])
    curv_val = float(curvature if isinstance(curvature, (int, float)) else curvature.item())
    dists = L.pairwise_dist(mats, mats, curv_val).cpu().numpy()
    condensed = distance.squareform(dists, checks=False)
    linkage = hierarchy.linkage(condensed, method="average")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    hierarchy.dendrogram(
        linkage,
        labels=class_names,
        ax=ax,
        orientation="top",
        color_threshold=0.0,
        link_color_func=lambda _: "tab:gray",
    )
    for lbl in ax.get_xticklabels():
        text = lbl.get_text()
        if text.endswith("_S1"):
            lbl.set_color("tab:blue")
        elif text.endswith("_S2"):
            lbl.set_color("tab:orange")
    ax.set_ylabel("Hyperbolic distance")
    ax.tick_params(axis="x", rotation=45)
    max_height = float(linkage[:, 2].max()) if linkage.size else 0.0
    if output is not None and ax.figure:
        ax.figure.savefig(output, dpi=200)
        plt.close(ax.figure)
    return linkage, max_height


def _extract_tree(linkage: np.ndarray, class_names: list[str]) -> dict:
    if linkage is None:
        return {}
    tree, nodes = hierarchy.to_tree(linkage, rd=True)
    leaves = {n.id: name for n, name in zip(nodes[: len(class_names)], class_names)}

    def _node_name(node_id: int) -> str:
        if node_id in leaves:
            return leaves[node_id]
        return f"node_{node_id}"

    def _walk(node) -> dict:
        name = _node_name(node.id)
        if node.is_leaf():
            return {"name": name, "children": []}
        left = _walk(node.left)
        right = _walk(node.right)
        return {"name": name, "children": [left, right]}

    return _walk(tree)


def _load_labels(label_path: Optional[Path], dataset: Sen12MSRetrievalDataset) -> Optional[torch.Tensor]:
    if label_path is None:
        _LOG.info("No label path provided; skipping label loading.")
        return None
    if not label_path.exists():
        _LOG.warning("Label path %s not found; skipping label loading.", label_path)
        return None

    with label_path.open("rb") as handle:
        labels = pickle.load(handle)

    # If labels are provided as a dict keyed by filenames, map them to dataset order.
    if isinstance(labels, dict):
        ordered: list[int] = []
        missing = 0
        for season, scene_id, patch_id in dataset.samples:
            key = f"{season}_s2_{scene_id}_p{patch_id}.tif"
            value = labels.get(key)
            if value is None:
                missing += 1
                ordered.append(-1)
            else:
                ordered.append(int(value))
        if missing > 0:
            _LOG.warning("Missing %d labels when aligning to dataset order; filled with -1.", missing)
            raise ValueError("Some labels missing")
        return torch.tensor(ordered, dtype=torch.long)

    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    if labels_tensor.numel() != len(dataset):
        _LOG.warning(
            "Loaded %d labels but dataset exposes %d samples. "
            "Ensure ordering matches the canonical SEN12MS file list used by the provided labels.",
            labels_tensor.numel(),
            len(dataset),
        )
    return labels_tensor


def run_radius_profile(config: Sen12MSHyperbolicRadiusConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # if embeddings exist then skip
    existing_path = config.output_dir / "sen12ms_hyperbolic_radii.pt"
    cross_dist = None
    curvature_tensor = None
    s1_proj = None
    s2_proj = None
    if True: #not existing_path.exists():

        adapter = build_evaluation_adapter(
            model_type=config.model_type,
            checkpoint=config.checkpoint,
            model_weights=config.model_weights,
        )
        if not adapter.supports_hyperbolic or not isinstance(adapter.base_model, LorentzCIIP):
            raise RuntimeError("A hyperbolic (LorentzCIIP) checkpoint is required for radius diagnostics.")

        device_str = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        adapter.to(device)
        adapter.eval()

        expected_channels = _infer_model_in_channels(adapter, fallback=13, modality="s2")
        dataset = Sen12MSRetrievalDataset(
            config.dataset_root,
            seasons=config.seasons,
            expected_s2_channels=expected_channels,
            image_size=config.image_size,
            max_patches_per_season=config.max_patches_per_season,
            normalize=config.use_ssl4eo_normalization,
        )
        labels_tensor = _load_labels(config.label_path, dataset)
        subset = _sample_subset(dataset, config.num_samples, config.seed)
        _LOG.info("Sampling %d paired patches from SEN12MS", len(subset))

        input_dtype = getattr(adapter, "dtype_s2", torch.float32)
        if device.type != "cuda" and input_dtype in {torch.float16, torch.bfloat16}:
            input_dtype = torch.float32

        autocast = torch.autocast(device_type="cuda") if device.type == "cuda" else contextlib.nullcontext()
        s1_embeddings, s2_embeddings, sample_ids = extract_embeddings_for_dataset(
            adapter,
            subset,
            input_dtype=input_dtype,
            device=device,
            max_batches_cka=0,
            autocast=autocast,
            register_layer_hooks=False,
        )

        if s1_embeddings is None or s1_embeddings.projected is None:
            raise RuntimeError("S1 projected embeddings were not produced; ensure the model exposes an S1 encoder.")
        if s2_embeddings is None or s2_embeddings.projected is None:
            raise RuntimeError("S2 projected embeddings were not produced.")

        
        s1_proj = s1_embeddings.projected.to(device=device)
        s2_proj = s2_embeddings.projected.to(device=device)

        with torch.no_grad():
            context = compute_hyperbolic_context(
                adapter.base_model,
                s1_proj,
                s2_proj,
                aperture_logk=None,
            )

        s1_radii = context["s1_distances"].cpu()
        s2_radii = context["s2_distances"].cpu()
        cross_dist = context["hyperbolic_distances"].cpu()
        curvature_tensor = context["curvature"].to(device=device)

        subset_labels = None
        if labels_tensor is not None:
            try:
                # Subset.indices holds indices into the wrapped dataset; align directly.
                subset_labels = labels_tensor[subset.indices]
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning("Failed to align labels to sampled subset: %s", exc)

        # s1_radii, s2_radii, subset_labels = _align_lengths(s1_radii, s2_radii, subset_labels)
        # Ensure lengths match; raise if not to avoid silent misalignment.
        if len(s1_radii) != len(s2_radii):
            raise ValueError(f"S1/S2 length mismatch: {len(s1_radii)} vs {len(s2_radii)}")
        if subset_labels is not None and len(subset_labels) != len(s1_radii):
            raise ValueError(
                f"Label length mismatch: {len(subset_labels)} vs radii {len(s1_radii)}"
            )

        torch.save(
            {
                "s1_radii": s1_radii,
                "s2_radii": s2_radii,
                "ids": sample_ids,
                "labels": subset_labels,
                "hyperbolic_distances": cross_dist,
                "curvature": curvature_tensor,
            },
            config.output_dir / "sen12ms_hyperbolic_radii.pt",
        )

    else:
        _LOG.info("Radii file %s already exists; loading.", existing_path)
        data = torch.load(existing_path)
        s1_radii = data["s1_radii"]
        s2_radii = data["s2_radii"]
        sample_ids = data["ids"]
        subset_labels = data.get("labels", None)
        cross_dist = data.get("hyperbolic_distances")
        curvature_tensor = torch.as_tensor(data.get("curvature")) if data.get("curvature") is not None else None
        s1_embeddings = None
        s2_embeddings = None

    _plot_radius_histogram(s1_radii, s2_radii, config.output_dir / "sen12ms_radii_hist.png")
    _plot_class_violin(
        s1_radii,
        s2_radii,
        subset_labels,
        config.output_dir / "sen12ms_radii_violin_per_class.png",
    )
    _plot_class_boxplot(
        s1_radii,
        s2_radii,
        subset_labels,
        config.output_dir / "sen12ms_radii_box_per_class.png",
    )
    if subset_labels is not None and cross_dist is not None and curvature_tensor is not None:
        # Pairwise intra/inter for S1, S2, and combined
        if s1_proj is not None and s2_proj is not None:
            pairwise_s1 = L.pairwise_dist(s1_proj, s1_proj, curvature_tensor).cpu()
            pairwise_s2 = L.pairwise_dist(s2_proj, s2_proj, curvature_tensor).cpu()
            all_proj = torch.cat([s1_proj, s2_proj], dim=0)
            labels_all = torch.cat([subset_labels, subset_labels], dim=0)
            pairwise_all = L.pairwise_dist(all_proj, all_proj, curvature_tensor).cpu()

            intra_s1, inter_s1 = _split_intra_inter(pairwise_s1, subset_labels)
            intra_s2, inter_s2 = _split_intra_inter(pairwise_s2, subset_labels)
            intra_all, inter_all = _split_intra_inter(pairwise_all, labels_all)

            _plot_intra_inter_hist(
                intra_s1,
                inter_s1,
                intra_s2,
                inter_s2,
                intra_all,
                inter_all,
                config.output_dir / "sen12ms_pairwise_intra_inter.png",
            )

            # Class centroids and dendrograms (S1, S2, combined)
            centroids_s1 = _compute_class_centroids(s1_proj, subset_labels, curvature_tensor)
            centroids_s2 = _compute_class_centroids(s2_proj, subset_labels, curvature_tensor)
            common_classes = sorted(set(centroids_s1.keys()) & set(centroids_s2.keys()))
            if not common_classes:
                _LOG.warning("No common classes between S1 and S2 centroids; skipping dendrograms.")
                common_classes = []

            fig = plt.figure(figsize=(18, 10), constrained_layout=True)
            gs = fig.add_gridspec(2, 2)
            ax_s1 = fig.add_subplot(gs[0, 0])
            ax_s2 = fig.add_subplot(gs[0, 1])
            ax_joint = fig.add_subplot(gs[1, :])

            linkage_s1, h1 = _plot_class_dendrogram(centroids_s1, curvature_tensor, None, ax=ax_s1)
            ax_s1.set_title("S1")
            linkage_s2, h2 = _plot_class_dendrogram(centroids_s2, curvature_tensor, None, ax=ax_s2)
            ax_s2.set_title("S2")
            joint_centroids = []
            joint_names: list[str] = []
            for cls in common_classes:
                name = IGBP_SIMPLE_MAP.get(int(cls), f"Unknown({int(cls)})")
                joint_centroids.append(centroids_s1[cls])
                joint_names.append(f"{name}_S1")
                joint_centroids.append(centroids_s2[cls])
                joint_names.append(f"{name}_S2")
            if joint_centroids:
                joint_tensor = torch.stack(joint_centroids)
                joint_dict = {i: joint_tensor[i] for i in range(len(joint_centroids))}
                linkage_joint, h_joint = _plot_class_dendrogram(
                    joint_dict,
                    curvature_tensor,
                    None,
                    ax=ax_joint,
                    names=joint_names,
                )
            else:
                linkage_joint, h_joint = None, 0.0
            ax_joint.set_title("Joint S1/S2")
            y_max = max(h1, h2, h_joint)
            for ax in (ax_s1, ax_s2, ax_joint):
                ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1)
            fig.savefig(config.output_dir / "sen12ms_class_dendrograms.png", dpi=200)
            plt.close(fig)

            # Save tree for combined centroids
            if linkage_joint is not None:
                tree = _extract_tree(linkage_joint, joint_names)
                with (config.output_dir / "sen12ms_class_tree.json").open("w", encoding="utf-8") as handle:
                    json.dump(tree, handle, indent=2)

        pos_by_cls, neg_by_cls = _collect_pos_neg_by_class(subset_labels, cross_dist)
        _plot_pos_neg_box_per_class(
            pos_by_cls,
            neg_by_cls,
            config.output_dir / "sen12ms_pairwise_pos_neg_per_class.png",
        )

    # Poincaré scatter (requires embeddings and labels)
    if (
        subset_labels is not None
        and curvature_tensor is not None
        and s1_embeddings is not None
        and s2_embeddings is not None
        and s1_embeddings.projected is not None
        and s2_embeddings.projected is not None
    ):
        p_s1 = lorentz_to_poincare(s1_embeddings.projected, curvature_tensor)
        p_s2 = lorentz_to_poincare(s2_embeddings.projected, curvature_tensor)
        _plot_poincare_scatter(
            p_s1,
            p_s2,
            subset_labels,
            curvature_tensor,
            config.output_dir / "poincare_s1_s2_scatter.png",
            seed=config.seed,
        )


        # 2D Poincaré projections
        b2_s1 = poincare_2d_from_lorentz(s1_embeddings.projected, curvature_tensor)  # (N, 2)
        b2_s2 = poincare_2d_from_lorentz(s2_embeddings.projected, curvature_tensor)  # (N, 2)
        _plot_poincare_disk(b2_s1, b2_s2, subset_labels, config.output_dir / "poincare_s1_s2_disk.png")




        # context = compute_hyperbolic_context(model, s1_feats, s2_feats, aperture_logk=...)
        plot_hyperbolic_entailment(
            points=s1_proj,                          # Lorentz embeddings (S1)
            labels=subset_labels,                     # same as before
            curvature=context["curvature"],
            apertures=context["aperture_s1"],         # from your context dict
            output=config.output_dir / "entailment_s1.png",
            max_points=500000,
            num_roots=200,
            seed=config.seed,
            modalities=np.array(["s1"] * len(subset_labels)),
        )
        plot_hyperbolic_entailment(
            points=s2_proj,
            labels=subset_labels,
            curvature=context["curvature"],
            apertures=context["aperture_s2"],
            output=config.output_dir / "entailment_s2.png",
            max_points=500000,
            num_roots=200,
            seed=config.seed,
            modalities=np.array(["s2"] * len(subset_labels)),
        )

        points_all = torch.cat([s1_proj, s2_proj])
        labels_all = torch.cat([subset_labels, subset_labels])
        N1 = s1_proj.shape[0]
        N2 = s2_proj.shape[0]

        modality_all = np.array(["s1"] * N1 + ["s2"] * N2)
        apertures_all = torch.cat([context["aperture_s1"], context["aperture_s2"]])
        plot_hyperbolic_entailment(
            points=points_all,
            labels=labels_all,
            curvature=context["curvature"],
            apertures=apertures_all,
            output=config.output_dir / "entailment_s1_s2_combined.png",
            max_points=5000,
            num_roots=200,
            seed=config.seed,
            modalities=modality_all,
        )



    if subset_labels is not None and cross_dist is not None and curvature_tensor is not None and s1_proj is not None and s2_proj is not None:
        pairwise_s1 = L.pairwise_dist(s1_proj, s1_proj, curvature_tensor).cpu()
        pairwise_s2 = L.pairwise_dist(s2_proj, s2_proj, curvature_tensor).cpu()
        all_proj = torch.cat([s1_proj, s2_proj], dim=0)
        labels_all = torch.cat([subset_labels, subset_labels], dim=0)
        pairwise_all = L.pairwise_dist(all_proj, all_proj, curvature_tensor).cpu()

        intra_s1, inter_s1 = _split_intra_inter(pairwise_s1, subset_labels)
        intra_s2, inter_s2 = _split_intra_inter(pairwise_s2, subset_labels)
        intra_all, inter_all = _split_intra_inter(pairwise_all, labels_all)

        print("Plotting pairwise intra/inter histograms.")
        _plot_intra_inter_hist(
            intra_s1,
            inter_s1,
            intra_s2,
            inter_s2,
            intra_all,
            inter_all,
            config.output_dir / "sen12ms_pairwise_intra_inter.png",
        )

        pos_by_cls, neg_by_cls = _collect_pos_neg_by_class(subset_labels, cross_dist)
        _plot_pos_neg_box_per_class(
            pos_by_cls,
            neg_by_cls,
            config.output_dir / "sen12ms_pairwise_pos_neg_per_class.png",
        )
    else:
        _LOG.info("Skipping pairwise intra/inter and pos/neg plots due to missing labels or distances.")
        # print missing
        if subset_labels is None:
            _LOG.info("Labels are missing.")
        if cross_dist is None:
            _LOG.info("Cross distances are missing.")
        if curvature_tensor is None:
            _LOG.info("Curvature tensor is missing.")
        if s1_proj is None:
            _LOG.info("S1 projections are missing.")
        if s2_proj is None:
            _LOG.info("S2 projections are missing.")    

    _LOG.info(
        "Saved radii for %d samples (S1 mean=%.4f, S2 mean=%.4f) to %s",
        len(sample_ids),
        float(s1_radii.mean()),
        float(s2_radii.mean()),
        config.output_dir,
    )


def _parse_args() -> Sen12MSHyperbolicRadiusConfig:
    parser = argparse.ArgumentParser(description="Hyperbolic radius profiles on SEN12MS.")
    parser.add_argument("--dataset-root", type=Path, default=Sen12MSHyperbolicRadiusConfig.dataset_root)
    parser.add_argument("--output-dir", type=Path, default=Sen12MSHyperbolicRadiusConfig.output_dir)
    parser.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        default=list(Sen12MSHyperbolicRadiusConfig.seasons),
        help="Subset of SEN12MS seasons to sample.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to CIIP checkpoint.")
    parser.add_argument(
        "--model-type",
        type=str,
        default=Sen12MSHyperbolicRadiusConfig.model_type,
        help="Model type (use 'ciip_checkpoint' for CIIP/LorentzCIIP checkpoints).",
    )
    parser.add_argument("--model-weights", type=str, default=None, help="Optional weights identifier.")
    parser.add_argument("--max-patches-per-season", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=Sen12MSHyperbolicRadiusConfig.image_size)
    parser.add_argument("--num-samples", type=int, default=Sen12MSHyperbolicRadiusConfig.num_samples)
    parser.add_argument("--seed", type=int, default=Sen12MSHyperbolicRadiusConfig.seed)
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu/cuda).")
    parser.add_argument(
        "--label-path",
        type=Path,
        default=Sen12MSHyperbolicRadiusConfig.label_path,
        help="Path to single-label_IGBPsimple_ClsNum.pkl for patch labels (one label per patch).",
    )
    parser.add_argument(
        "--use-ssl4eo-normalization",
        action=argparse.BooleanOptionalAction,
        default=Sen12MSHyperbolicRadiusConfig.use_ssl4eo_normalization,
        help="Apply SSL4EO normalization to inputs.",
    )
    args = parser.parse_args()

    return Sen12MSHyperbolicRadiusConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seasons=tuple(args.seasons),
        label_path=args.label_path,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        model_weights=args.model_weights,
        max_patches_per_season=args.max_patches_per_season,
        image_size=args.image_size,
        use_ssl4eo_normalization=args.use_ssl4eo_normalization,
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    run_radius_profile(_parse_args())
