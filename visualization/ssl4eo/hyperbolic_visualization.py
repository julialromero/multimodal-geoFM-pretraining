#!/usr/bin/env python3
"""Visualize hyperbolic embeddings learned by a CIIP model.

This script samples locations from the SSL4EO training dataset and reproduces the
hyperbolic computations implemented in :mod:`ciip.loss` to generate several
diagnostic plots:

1. Angle–aperture scatter plot comparing positive pair angles against the
   acceptance cone widths for Sentinel-1 and Sentinel-2 embeddings.
2. Radial histograms (pre-lift norms) per modality.
3. A 2-D PCA map of the unit direction vectors used in the hyperbolic logits,
   coloured by their hyperbolic distance from the origin.
4. A cone-style polar plot for a subset of pairs that visualizes the positive
   angles alongside their modality-specific acceptance apertures.

Example usage
-------------
    python visualization/ssl4eo/hyperbolic_visualization.py \
        --checkpoint /path/to/checkpoint.pt \
        --config ciip/open_clip_train/config_train.ini \
        --output-dir outputs/hyperbolic_viz \
        --num-locations 64
"""

from __future__ import annotations

import argparse
import ast
import configparser
import csv
import math
import os
import random
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Compatibility shims
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency
    import hydra  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    hydra = types.ModuleType("hydra")
    hydra.utils = types.SimpleNamespace(get_original_cwd=os.getcwd)
    sys.modules["hydra"] = hydra

from ciip.loss import CiipLoss
from ciip.model_ciip import CIIP
from ciip.open_clip_train.data import SSL4EODataset


@dataclass
class DatasetSpec:
    root: str
    s2_tier: str
    target_dim: int
    s2_bands: Sequence[str]
    s1_band_count: int


@dataclass
class ModelSpec:
    embed_dim: int
    pre_projection_dim: int
    s1_layers: Sequence[int]
    s2_layers: Sequence[int]
    width: int
    s1_patch: int
    s2_patch: int
    s1_resolution: int
    s2_resolution: int
    framework: str
    s1_band_count: int
    s2_band_count: int
    pretrain: bool = False
    s1_weights: str = "MOCO"
    s2_weights: str = "MOCO"


def _softplus(x: float) -> float:
    """Scalar softplus helper that mirrors :func:`torch.nn.functional.softplus`."""

    return math.log1p(math.exp(float(x)))


def load_hyp_params(ckpt_path_or_dict, eps_default: float = 1e-5) -> Dict[str, Optional[float]]:
    """Load hyperbolic loss parameters from a checkpoint.

    Parameters
    ----------
    ckpt_path_or_dict:
        Either a checkpoint dictionary or a path/``PathLike`` pointing to one.
    eps_default:
        Fallback epsilon if the checkpoint does not contain it.

    Returns
    -------
    Dict[str, Optional[float]]
        Dictionary with ``c`` (curvature), ``k_ap`` (aperture log-scale),
        ``hyp_scale`` (pre-lift scale), and ``eps`` (epsilon used during
        training).
    """

    if isinstance(ckpt_path_or_dict, (str, bytes, os.PathLike)):
        ckpt = torch.load(ckpt_path_or_dict, map_location="cpu")
    else:
        ckpt = ckpt_path_or_dict

    sd = ckpt.get("state_dict", ckpt)
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint has no state_dict")

    def find(*keys: str):
        for key in keys:
            if key in sd:
                return sd[key]
            module_key = f"module.{key}"
            if module_key in sd:
                return sd[module_key]
        return None

    curvature_alpha = find("loss.curvature_alpha")
    eps = find("loss.hyperbolic_eps")
    ap_logk = find("loss.aperture_logk")
    hyp_scale = find("loss.hyp_scale")

    if curvature_alpha is not None:
        curvature = _softplus(curvature_alpha) + float(eps if eps is not None else eps_default)
    else:
        curvature_raw = find("loss.curvature", "loss.c")
        if curvature_raw is not None:
            curvature = float(curvature_raw)
        else:
            curvature = 1.0 + float(eps_default)

    hyp_scale_value: Optional[float]
    if hyp_scale is not None:
        hyp_scale_value = float(F.softplus(hyp_scale))
    else:
        hyp_scale_value = None

    params = {
        "c": float(curvature),
        "k_ap": float(ap_logk) if ap_logk is not None else None,
        "hyp_scale": hyp_scale_value,
        "eps": float(eps if eps is not None else eps_default),
    }
    return params


def _parse_list(value: str) -> List:
    return list(ast.literal_eval(value)) if value else []


def load_config(config_path: Path) -> Tuple[DatasetSpec, ModelSpec]:
    parser = configparser.ConfigParser()
    if not parser.read(config_path):
        raise FileNotFoundError(f"Could not read configuration file: {config_path}")

    model_section = parser["model"]
    dataset_section = parser["dataset"]

    embed_dim = model_section.getint("embed_dim", fallback=512)
    pre_projection_dim = model_section.getint("pre_projection_dim", fallback=embed_dim)
    s1_layers = tuple(_parse_list(model_section.get("s1_layers", fallback="(3, 4, 6, 3)")))
    s2_layers = tuple(_parse_list(model_section.get("s2_layers", fallback="(3, 4, 6, 3)")))
    width = model_section.getint("width", fallback=32)
    s1_patch = model_section.getint("s1_patch_size", fallback=16)
    s2_patch = model_section.getint("s2_patch_size", fallback=16)
    s1_resolution = model_section.getint("s1_resolution", fallback=224)
    s2_resolution = model_section.getint("s2_resolution", fallback=224)
    framework = model_section.get("framework", fallback="resnet50")

    s1_bands = _parse_list(model_section.get("s1_bands", fallback="[1, 2]"))
    s2_bands = _parse_list(model_section.get("s2_bands", fallback="['1', '2', '3', '4', '5', '6', '7', '8', '8A', '9', '10', '11', '12']"))

    dataset_root = dataset_section.get("root")
    if not dataset_root:
        raise ValueError("Dataset root must be specified in the config file.")

    dataset_spec = DatasetSpec(
        root=dataset_root,
        s2_tier=dataset_section.get("s2_tier", fallback="s2c"),
        target_dim=model_section.getint("s1_resolution", fallback=224),
        s2_bands=s2_bands,
        s1_band_count=len(s1_bands),
    )

    model_spec = ModelSpec(
        embed_dim=embed_dim,
        pre_projection_dim=pre_projection_dim,
        s1_layers=tuple(int(x) for x in s1_layers),
        s2_layers=tuple(int(x) for x in s2_layers),
        width=width,
        s1_patch=s1_patch,
        s2_patch=s2_patch,
        s1_resolution=s1_resolution,
        s2_resolution=s2_resolution,
        framework=framework,
        s1_band_count=len(s1_bands),
        s2_band_count=len(s2_bands),
        pretrain=model_section.getboolean("pretrain", fallback=False),
        s1_weights=model_section.get("s1_weights", fallback="MOCO"),
        s2_weights=model_section.get("s2_weights", fallback="MOCO"),
    )

    return dataset_spec, model_spec


def build_model(spec: ModelSpec, device: torch.device) -> CIIP:
    model = CIIP(
        embed_dim=spec.embed_dim,
        pre_projection_dim=spec.pre_projection_dim,
        s1_resolution=spec.s1_resolution,
        s1_layers=spec.s1_layers,
        s1_width=spec.width,
        s1_patch_size=spec.s1_patch,
        s1_bands=spec.s1_band_count,
        s2_resolution=spec.s2_resolution,
        s2_layers=spec.s2_layers,
        s2_width=spec.width,
        s2_patch_size=spec.s2_patch,
        s2_bands=spec.s2_band_count,
        framework=spec.framework,
        pretrain=spec.pretrain,
        s1_weights=spec.s1_weights,
        s2_weights=spec.s2_weights,
    )
    return model.to(device)


def load_model_weights(model: CIIP, checkpoint_path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Warning: Missing model parameters: {missing}")
    if unexpected:
        print(f"Warning: Unexpected parameters in checkpoint: {unexpected}")
    return checkpoint


def build_loss(args: argparse.Namespace, device: torch.device) -> CiipLoss:
    loss = CiipLoss(
        hyperbolic=args.hyperbolic,
        hyperbolic_normalize=not args.no_hyperbolic_normalize,
        hyperbolic_margin_weight=args.hyperbolic_margin_weight,
        hyperbolic_curvature_init=args.curvature_init,
        hyperbolic_eps=args.hyperbolic_eps,
    )
    return loss.to(device)


def override_curvature(loss: CiipLoss, curvature: float) -> None:
    curvature = max(curvature, loss.hyperbolic_eps)
    alpha = torch.log(torch.expm1(torch.tensor(curvature, device=loss.curvature_alpha.device)))
    with torch.no_grad():
        loss.curvature_alpha.copy_(alpha)


def maybe_load_loss_state(loss: CiipLoss, checkpoint: Dict) -> bool:
    if "loss_state_dict" in checkpoint:
        loss.load_state_dict(checkpoint["loss_state_dict"], strict=False)
        return True
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        prefix = "loss."
        filtered = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if filtered:
            current = loss.state_dict()
            current.update(filtered)
            loss.load_state_dict(current, strict=False)
            return True
    return False


def _normalize_root_path(root: str) -> Path:
    path = Path(root).expanduser().resolve()
    if path.name.lower() in {"s1", "s1c", "s2", "s2a", "s2c"}:
        return path.parent
    return path


def _resolve_subdirectory(base: Path, candidates: Sequence[str], kind: str) -> Path:
    seen = set()
    for name in candidates:
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        candidate = base / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {kind} directory under '{base}'. Searched: {', '.join(seen) or '∅'}."
    )


def _candidate_names(preferred: str, extras: Sequence[str]) -> List[str]:
    variants = [preferred, preferred.lower(), preferred.upper()]
    variants.extend(extras)
    ordered: List[str] = []
    for name in variants:
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def prepare_dataset(spec: DatasetSpec) -> SSL4EODataset:
    base_root = _normalize_root_path(spec.root)

    s1_dir = _resolve_subdirectory(
        base_root,
        _candidate_names("s1", ["s1c", "S1", "S1C"]),
        "Sentinel-1",
    )
    s2_dir = _resolve_subdirectory(
        base_root,
        _candidate_names(spec.s2_tier, ["s2c", "s2a", "s2", "S2C", "S2A", "S2"]),
        "Sentinel-2",
    )

    expected_s1 = base_root / "s1"
    expected_s2 = base_root / spec.s2_tier

    def _build_dataset(root_path: Path) -> SSL4EODataset:
        dataset = SSL4EODataset(
            root=str(root_path),
            s2_tier=spec.s2_tier,
            s2_bands=spec.s2_bands,
            transforms=None,
            target_image_dimension=(spec.target_dim, spec.target_dim),
        )
        return dataset

    if expected_s1.is_dir() and expected_s2.is_dir():
        dataset = _build_dataset(base_root)
        dataset.root = str(base_root)
        return dataset

    with tempfile.TemporaryDirectory() as temp_root:
        temp_path = Path(temp_root)
        (temp_path / "s1").symlink_to(s1_dir, target_is_directory=True)
        (temp_path / spec.s2_tier).symlink_to(s2_dir, target_is_directory=True)
        dataset = _build_dataset(temp_path)

    dataset.s1_dir = str(s1_dir)
    dataset.s2_dir = str(s2_dir)
    dataset.root = str(base_root)

    def _patched_get_sample_uid(self: SSL4EODataset, idx: int, *, _s1_dir=str(s1_dir)) -> Tuple[str, str]:
        location_idx, season_idx = self.int_to_filepath(idx)
        location_folder = self.locations[location_idx]
        filepath = os.path.join(_s1_dir, location_folder)
        unique_id = f"{location_folder}_season{season_idx}"
        return unique_id, filepath

    dataset.get_sample_uid = types.MethodType(_patched_get_sample_uid, dataset)
    return dataset


def sample_indices(dataset: SSL4EODataset, num_locations: int, seed: int) -> Tuple[List[int], List[Dict[str, str]]]:
    rng = np.random.default_rng(seed)
    total_locations = dataset.num_locations
    if total_locations <= 0:
        raise ValueError("Dataset contains no locations.")
    seasons_per_location = dataset.length // dataset.num_locations
    num_locations = min(num_locations, total_locations)
    chosen_locations = rng.choice(total_locations, size=num_locations, replace=False)

    indices: List[int] = []
    metadata: List[Dict[str, str]] = []
    for loc_idx in chosen_locations:
        location_name = dataset.locations[loc_idx]
        s1_location_dir = os.path.join(dataset.s1_dir, location_name)
        season_folders = sorted(os.listdir(s1_location_dir))
        for season_idx in range(seasons_per_location):
            indices.append(loc_idx * seasons_per_location + season_idx)
            season_name = season_folders[season_idx] if season_idx < len(season_folders) else f"season_{season_idx}"
            metadata.append(
                {
                    "location": location_name,
                    "season_index": season_idx,
                    "season_name": season_name,
                }
            )
    return indices, metadata


def stack_features(model: CIIP, dataset: SSL4EODataset, indices: Sequence[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    s1_feats: List[torch.Tensor] = []
    s2_feats: List[torch.Tensor] = []
    with torch.no_grad():
        for idx in indices:
            s1, s2 = dataset[idx]
            s1 = s1.unsqueeze(0).to(device)
            s2 = s2.unsqueeze(0).to(device)
            outputs = model.compute_embeddings(s1, s2)
            s1_feats.append(outputs["s1_features"].squeeze(0))
            s2_feats.append(outputs["s2_features"].squeeze(0))
    return torch.stack(s1_feats, dim=0), torch.stack(s2_feats, dim=0)


def _aperture_from_inner_scaled(
    inner_pos: torch.Tensor,
    eps: float,
    aperture_logk: Optional[float],
) -> torch.Tensor:
    inv_cosh = torch.clamp(1.0 / inner_pos, min=-1.0 + eps, max=1.0 - eps)
    if aperture_logk is not None:
        scale = math.exp(float(aperture_logk))
        inv_cosh = torch.clamp(inv_cosh * scale, min=-1.0 + eps, max=1.0 - eps)
    return torch.asin(inv_cosh)


def compute_hyperbolic_context(
    loss: CiipLoss,
    s1_feats: torch.Tensor,
    s2_feats: torch.Tensor,
    *,
    aperture_logk: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    curvature = loss._get_curvature(dtype=s1_feats.dtype, device=s1_feats.device)
    s1_points = loss._lift_to_hyperboloid(s1_feats, curvature)
    s2_points = loss._lift_to_hyperboloid(s2_feats, curvature)

    s1_dirs, s1_distances, s1_inner = loss._hyperbolic_directions(s1_points, curvature)
    s2_dirs, s2_distances, s2_inner = loss._hyperbolic_directions(s2_points, curvature)

    local_angles = loss._angle_matrix_from_dirs(s1_dirs, s2_dirs)
    positive_angles = torch.diagonal(local_angles)

    eps = float(loss.hyperbolic_eps)
    aperture_s1 = _aperture_from_inner_scaled(s1_inner, eps, aperture_logk)
    aperture_s2 = _aperture_from_inner_scaled(s2_inner, eps, aperture_logk)

    return {
        "positive_angles": positive_angles,
        "aperture_s1": aperture_s1,
        "aperture_s2": aperture_s2,
        "s1_dirs": s1_dirs,
        "s2_dirs": s2_dirs,
        "s1_distances": s1_distances,
        "s2_distances": s2_distances,
        "curvature": curvature,
    }


def _pca_reduce(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    return centered @ components.T


def plot_angle_aperture(positive_angles: np.ndarray, aperture_s1: np.ndarray, aperture_s2: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(positive_angles, 2.0 * aperture_s1, alpha=0.6, label="S1 aperture width", color="tab:blue")
    ax.scatter(positive_angles, 2.0 * aperture_s2, alpha=0.6, label="S2 aperture width", color="tab:orange")
    ax.set_xlabel("Positive pair angle (radians)")
    ax.set_ylabel("Aperture width (radians)")
    ax.set_title("Angle vs. acceptance aperture")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_radial_histogram(s1_norms: np.ndarray, s2_norms: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = max(20, int(math.sqrt(len(s1_norms))))
    ax.hist(s1_norms, bins=bins, alpha=0.6, label="S1", density=True, color="tab:blue")
    ax.hist(s2_norms, bins=bins, alpha=0.6, label="S2", density=True, color="tab:orange")
    ax.set_xlabel("Pre-lift embedding norm")
    ax.set_ylabel("Density")
    ax.set_title("Radial distribution per modality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_angular_pca(s1_dirs: np.ndarray, s2_dirs: np.ndarray, s1_distances: np.ndarray, s2_distances: np.ndarray, output: Path) -> None:
    all_dirs = np.concatenate([s1_dirs, s2_dirs], axis=0)
    coords = _pca_reduce(all_dirs, n_components=2)
    s1_coords = coords[: len(s1_dirs)]
    s2_coords = coords[len(s1_dirs) :]

    distances = np.concatenate([s1_distances, s2_distances])
    norm = Normalize(vmin=distances.min(), vmax=distances.max())

    fig, ax = plt.subplots(figsize=(7, 6))
    sc1 = ax.scatter(s1_coords[:, 0], s1_coords[:, 1], c=s1_distances, cmap="viridis", norm=norm, label="S1", alpha=0.7)
    sc2 = ax.scatter(s2_coords[:, 0], s2_coords[:, 1], c=s2_distances, cmap="magma", norm=norm, label="S2", marker="^", alpha=0.7)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("Angular PCA of hyperbolic directions")
    ax.grid(alpha=0.3)
    ax.legend()

    cbar = fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hyperbolic distance")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_cone_polar(positive_angles: np.ndarray, aperture_s1: np.ndarray, aperture_s2: np.ndarray, output: Path, sample_size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    total = len(positive_angles)
    if total == 0:
        return
    sample_size = min(sample_size, total)
    sample_idx = rng.choice(total, size=sample_size, replace=False)

    theta = positive_angles[sample_idx]
    s1_width = 2.0 * aperture_s1[sample_idx]
    s2_width = 2.0 * aperture_s2[sample_idx]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, polar=True)
    base_s1 = 0.6
    base_s2 = 1.2
    height = 0.45

    bars1 = ax.bar(theta, height, width=s1_width, bottom=base_s1, color="tab:blue", alpha=0.4, edgecolor="none", label="S1 cone")
    bars2 = ax.bar(theta, height, width=s2_width, bottom=base_s2, color="tab:orange", alpha=0.4, edgecolor="none", label="S2 cone")

    ax.scatter(theta, np.full_like(theta, base_s1 + height / 2.0), color="tab:blue", s=10, alpha=0.7)
    ax.scatter(theta, np.full_like(theta, base_s2 + height / 2.0), color="tab:orange", s=10, alpha=0.7)

    ax.set_title("Acceptance cones for sampled pairs")
    ax.set_rticks([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_metadata_csv(records: List[Dict[str, float]], output_path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperbolic embedding space visualisation")
    parser.add_argument("--config", type=Path, default=Path("ciip/open_clip_train/config_train.ini"), help="Path to the training configuration INI file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("hyperbolic_viz"), help="Directory for generated plots.")
    parser.add_argument("--num-locations", type=int, default=64, help="Number of locations to sample (all seasons per location are included).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (defaults to cuda if available).")
    parser.add_argument("--loss-checkpoint", type=Path, default=None, help="Optional checkpoint containing loss state (curvature).")
    parser.add_argument("--curvature", type=float, default=None, help="Override curvature value for the hyperbolic loss.")
    parser.add_argument("--hyperbolic", action="store_true", default=True, help="Enable hyperbolic computations.")
    parser.add_argument("--no-hyperbolic", dest="hyperbolic", action="store_false", help="Disable hyperbolic computations.")
    parser.add_argument("--no-hyperbolic-normalize", action="store_true", help="Disable pre-normalisation before lifting to the hyperboloid.")
    parser.add_argument("--hyperbolic-margin-weight", type=float, default=1.0)
    parser.add_argument("--curvature-init", type=float, default=1.0)
    parser.add_argument("--hyperbolic-eps", type=float, default=1e-5)
    parser.add_argument("--cone-samples", type=int, default=100, help="Number of pairs to visualise in the cone polar plot.")

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_spec, model_spec = load_config(args.config)
    dataset = prepare_dataset(dataset_spec)

    indices, metadata = sample_indices(dataset, args.num_locations, args.seed)

    model = build_model(model_spec, device)
    checkpoint = load_model_weights(model, args.checkpoint, device)
    hyp_params = load_hyp_params(checkpoint, eps_default=args.hyperbolic_eps)

    loss = build_loss(args, device)
    if args.loss_checkpoint is not None:
        loss_checkpoint = torch.load(args.loss_checkpoint, map_location=device)
        maybe_load_loss_state(loss, loss_checkpoint)
    else:
        maybe_load_loss_state(loss, checkpoint)

    loss.hyperbolic_eps = float(hyp_params["eps"])
    default_curvature = float(hyp_params["c"])
    if args.curvature is not None:
        override_curvature(loss, args.curvature)
    else:
        override_curvature(loss, default_curvature)

    s1_feats, s2_feats = stack_features(model, dataset, indices, device)
    hyp_scale = hyp_params["hyp_scale"] if hyp_params["hyp_scale"] is not None else 0.7
    aperture_logk = hyp_params["k_ap"] if hyp_params["k_ap"] is not None else math.log(0.5)

    s1_scaled = s1_feats * float(hyp_scale)
    s2_scaled = s2_feats * float(hyp_scale)

    context = compute_hyperbolic_context(
        loss,
        s1_scaled,
        s2_scaled,
        aperture_logk=aperture_logk,
    )

    positive_angles = context["positive_angles"].detach().cpu().numpy()
    aperture_s1 = context["aperture_s1"].detach().cpu().numpy()
    aperture_s2 = context["aperture_s2"].detach().cpu().numpy()
    s1_dirs = context["s1_dirs"].detach().cpu().numpy()
    s2_dirs = context["s2_dirs"].detach().cpu().numpy()
    s1_distances = context["s1_distances"].detach().cpu().numpy()
    s2_distances = context["s2_distances"].detach().cpu().numpy()

    s1_norms = s1_scaled.norm(dim=-1).detach().cpu().numpy()
    s2_norms = s2_scaled.norm(dim=-1).detach().cpu().numpy()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_angle_aperture(positive_angles, aperture_s1, aperture_s2, output_dir / "angle_aperture_scatter.png")
    plot_radial_histogram(s1_norms, s2_norms, output_dir / "radial_histograms.png")
    plot_angular_pca(s1_dirs, s2_dirs, s1_distances, s2_distances, output_dir / "angular_pca.png")
    plot_cone_polar(positive_angles, aperture_s1, aperture_s2, output_dir / "cone_polar.png", args.cone_samples, args.seed)

    records: List[Dict[str, float]] = []
    for i, meta in enumerate(metadata):
        record = {
            "location": meta["location"],
            "season_index": meta["season_index"],
            "season_name": meta["season_name"],
            "positive_angle": float(positive_angles[i]),
            "aperture_s1": float(aperture_s1[i]),
            "aperture_s2": float(aperture_s2[i]),
            "s1_norm": float(s1_norms[i]),
            "s2_norm": float(s2_norms[i]),
            "s1_distance": float(s1_distances[i]),
            "s2_distance": float(s2_distances[i]),
        }
        records.append(record)

    write_metadata_csv(records, output_dir / "hyperbolic_summary.csv")

    curvature = loss._get_curvature(dtype=s1_feats.dtype, device=device).item()
    aperture_scale = math.exp(aperture_logk) if aperture_logk is not None else None
    print(f"Saved plots to {output_dir.resolve()}")
    print(f"Effective curvature used for visualisation: {curvature:.6f}")
    print(
        "Hyperbolic params — eps: {eps:.2e}, hyp_scale: {scale:.6f}, aperture_scale: {ap:.6f} (logK={logk:.3f})".format(
            eps=loss.hyperbolic_eps,
            scale=float(hyp_scale),
            ap=aperture_scale if aperture_scale is not None else float("nan"),
            logk=aperture_logk if aperture_logk is not None else float("nan"),
        )
    )


if __name__ == "__main__":
    main()
