#!/usr/bin/env python3
"""
Parse Sen12MS cross-modal retrieval metrics saved per epoch or embedding dimension and plot them.

Defaults to /home/juro4948/ciip/ciip/results/sen12ms_retrieval_run/ciip/11_22_2025
and writes a plot in the same directory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def _load_epoch_metrics(
    base_dir: Path,
) -> List[Tuple[int, Dict[str, float], Optional[int]]]:
    """Return a list of (epoch, metrics, num_samples) sorted by epoch."""

    results: List[Tuple[int, Dict[str, float], Optional[int]]] = []
    pattern = re.compile(r"epoch(\d+)")

    for sub in base_dir.iterdir():
        if not sub.is_dir():
            continue
        match = pattern.fullmatch(sub.name)
        if not match:
            continue

        epoch = int(match.group(1))
        metrics_path = sub / "sen12ms_retrieval.json"
        if not metrics_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        num_samples = metrics.get("num_samples")
        results.append((epoch, metrics, int(num_samples) if num_samples is not None else None))

    results.sort(key=lambda x: x[0])
    return results


def _plot(metrics: List[Tuple[int, Dict[str, float], Optional[int]]], out_path: Path) -> None:
    """Plot s1->s2 and s2->s1 recall curves over epochs."""

    if not metrics:
        raise RuntimeError(f"No retrieval metrics found under {out_path.parent}")

    epochs = [e for e, _, _ in metrics]
    num_samples = metrics[0][2]
    if num_samples is None:
        raise RuntimeError("num_samples is required to plot recall counts.")

    def collect(prefix: str, key: str) -> List[float]:
        lookup = f"{prefix}_{key}"
        return [float(m.get(lookup, 0.0)) * num_samples / 100.0 for _, m, _ in metrics]

    ks = ("r1", "r5", "r10", "r100")
    directions = ("s1_to_s2", "s2_to_s1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, direction in zip(axes, directions):
        for k in ks:
            series = collect(direction, k)
            ax.plot(epochs, series, marker="o", label=k.upper())
        ax.set_title(direction.replace("_", " → ").upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recall@K (count)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Sen12MS Cross-Modal Retrieval over Epochs", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _load_dim_metrics(
    metrics_path: Path,
) -> Tuple[List[int], Dict[str, Dict[str, float]], int, str]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    dims = [int(d) for d in payload.get("dimensions", [])]
    metrics = payload.get("metrics", {})
    num_samples = payload.get("num_samples")
    feature_space = str(payload.get("feature_space", "embeddings"))
    if not dims or not metrics:
        raise RuntimeError(f"No retrieval metrics found in {metrics_path}")
    if num_samples is None:
        raise RuntimeError(f"num_samples is required to plot recall counts in {metrics_path}")
    return dims, metrics, int(num_samples), feature_space


def _plot_dims(
    dims: List[int],
    metrics: Dict[str, Dict[str, float]],
    num_samples: int,
    feature_space: str,
    out_path: Path,
) -> None:
    ks = ("r1", "r5", "r10", "r100")
    directions = ("s1_to_s2", "s2_to_s1")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    for ax, direction in zip(axes, directions):
        for k in ks:
            series = [
                float(metrics[str(d)].get(f"{direction}_{k}", 0.0)) * num_samples / 100.0
                for d in dims
            ]
            ax.plot(dims, series, marker="o", label=k.upper())
        ax.set_title(direction.replace("_", " → ").upper())
        ax.set_xlabel(f"Embedding dimension ({feature_space})")
        ax.set_ylabel("Recall@K (count)")
        ax.set_xticks(dims)
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"Sen12MS Cross-Modal Retrieval over {feature_space} Dimensions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_dims_mean(
    dims: List[int],
    metrics: Dict[str, Dict[str, float]],
    num_samples: int,
    feature_space: str,
    out_path: Path,
) -> None:
    ks = ("r1", "r5", "r10", "r100")
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for k in ks:
        series = []
        for d in dims:
            entry = metrics[str(d)]
            s1 = float(entry.get(f"s1_to_s2_{k}", 0.0)) * num_samples / 100.0
            s2 = float(entry.get(f"s2_to_s1_{k}", 0.0)) * num_samples / 100.0
            series.append((s1 + s2) / 2.0)
        ax.plot(dims, series, marker="o", label=k.upper())

    ax.set_title("Mean Retrieval Recall over Dimensions")
    ax.set_xlabel(f"Embedding dimension ({feature_space})")
    ax.set_ylabel("Recall@K (count, mean of directions)")
    ax.set_xticks(dims)
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_dims_vs_k(
    dims: List[int],
    metrics: Dict[str, Dict[str, float]],
    num_samples: int,
    feature_space: str,
    out_path: Path,
) -> None:
    ks = ("r1", "r5", "r10", "r100")
    directions = ("s1_to_s2", "s2_to_s1")

    k_labels = [k.upper() for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    for ax, direction in zip(axes, directions):
        for dim in dims:
            series = [
                float(metrics[str(dim)].get(f"{direction}_{k}", 0.0)) * num_samples / 100.0
                for k in ks
            ]
            ax.plot(k_labels, series, marker="o", label=f"D={dim}")
        ax.set_title(direction.replace("_", " → ").upper())
        ax.set_xlabel("Recall@K")
        ax.set_ylabel("Recall@K (count)")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8)

    fig.suptitle(f"Sen12MS Retrieval: {feature_space} dims vs K", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Sen12MS retrieval metrics over epochs.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/home/juro4948/ciip/ciip/results/sen12ms_retrieval_run/ciip/11_22_2025"),
        help="Directory containing epoch*/sen12ms_retrieval.json files.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Path to a sen12ms_retrieval_by_dim.json file for dimension plots.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("auto", "epochs", "dims"),
        default="auto",
        help="Plot epoch-based or dimension-based retrieval curves.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the plot (PNG). Defaults to <base-dir>/sen12ms_retrieval_over_epochs.png",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    metrics_file = args.metrics_file or (base_dir / "sen12ms_retrieval_by_dim.json")
    mode = args.mode

    if mode == "auto":
        mode = "dims" if metrics_file.exists() else "epochs"

    if mode == "dims":
        out_path = args.out or (base_dir / "sen12ms_retrieval_over_dims.png")
        dims, metrics, num_samples, feature_space = _load_dim_metrics(metrics_file)
        _plot_dims(dims, metrics, num_samples, feature_space, out_path)
        mean_out_path = out_path.with_name(f"{out_path.stem}_mean{out_path.suffix}")
        _plot_dims_mean(dims, metrics, num_samples, feature_space, mean_out_path)
        dims_vs_k_out_path = out_path.with_name(f"{out_path.stem}_dims_vs_k{out_path.suffix}")
        _plot_dims_vs_k(dims, metrics, num_samples, feature_space, dims_vs_k_out_path)
        print(f"Saved plots to {out_path}, {mean_out_path}, and {dims_vs_k_out_path}")
    else:
        out_path = args.out or (base_dir / "sen12ms_retrieval_over_epochs.png")
        metrics = _load_epoch_metrics(base_dir)
        _plot(metrics, out_path)
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
