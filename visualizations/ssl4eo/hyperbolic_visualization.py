#!/usr/bin/env python3
"""Generate hyperbolic diagnostic plots for a trained CIIP checkpoint."""

from __future__ import annotations

import argparse
import ast
import configparser
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from ciip import lorentz as L
from ciip.model_ciip import LorentzCIIP
from ciip.open_clip_train.data import SSL4EODataset
from visualizations.ssl4eo.embedding_collapse_diagnostics import (
    ensure_hydra_original_cwd,
    # load_model_from_checkpoint,
    resolve_input_dtype,
)


def _parse_literal(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_config(path: Path) -> DictConfig:
    path = path.expanduser()
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        config = OmegaConf.load(path)
    elif suffix == ".ini":
        parser = configparser.ConfigParser()
        if not parser.read(path):
            raise FileNotFoundError(path)
        data = {section: {k: _parse_literal(v) for k, v in parser.items(section)} for section in parser.sections()}
        config = OmegaConf.create(data)
    else:
        raise ValueError(f"Unsupported config format: {path}")
    OmegaConf.set_struct(config, False)
    return config


def _dataset_dimension(cfg: DictConfig) -> int:
    if "dimension" in cfg.dataset:
        return int(cfg.dataset.dimension)
    if "s1_resolution" in cfg.model:
        return int(cfg.model.s1_resolution)
    if "s2_resolution" in cfg.model:
        return int(cfg.model.s2_resolution)
    return 224


def prepare_dataset(cfg: DictConfig) -> SSL4EODataset:
    ensure_hydra_original_cwd()
    target_dim = _dataset_dimension(cfg)
    return SSL4EODataset(
        root=str(cfg.dataset.root),
        s2_tier=str(getattr(cfg.dataset, "s2_tier", "s2c")),
        s2_bands=list(cfg.model.s2_bands),
        transforms=None,
        target_image_dimension=(target_dim, target_dim),
    )


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} did not contain a state dict")
    return checkpoint


def load_hyperbolic_params(checkpoint: Dict[str, torch.Tensor], eps_default: float) -> Dict[str, Optional[float]]:
    state_dict = checkpoint.get("state_dict", checkpoint)
    loss_state = checkpoint.get("loss_state_dict")

    def _lookup(key: str) -> Optional[torch.Tensor]:
        if loss_state and key in loss_state:
            return loss_state[key]
        if key in state_dict:
            return state_dict[key]
        module_key = f"loss.{key}"
        if module_key in state_dict:
            return state_dict[module_key]
        return None

    curvature_alpha = _lookup("curvature_alpha")
    eps = _lookup("hyperbolic_eps")
    aperture_logk = _lookup("aperture_logk")
    hyp_scale = _lookup("hyp_scale")

    if curvature_alpha is not None:
        curvature = F.softplus(curvature_alpha).item() + float(eps.item() if eps is not None else eps_default)
    else:
        curvature_raw = _lookup("curvature") or _lookup("c")
        curvature = float(curvature_raw.item()) if curvature_raw is not None else 1.0 + eps_default

    hyp_scale_raw = float(hyp_scale.item()) if hyp_scale is not None else None
    hyp_scale_pos = float(F.softplus(hyp_scale).item()) if hyp_scale is not None else None

    return {
        "c": curvature,
        "k_ap": float(aperture_logk.item()) if aperture_logk is not None else None,
        "hyp_scale": hyp_scale_pos,
        "hyp_scale_raw": hyp_scale_raw,
        "eps": float(eps.item() if eps is not None else eps_default),
    }


def sample_indices(dataset: SSL4EODataset, num_locations: int, seed: int) -> Tuple[List[int], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    num_locations = min(num_locations, dataset.num_locations)
    seasons_per_location = dataset.length // dataset.num_locations

    chosen = rng.choice(dataset.num_locations, size=num_locations, replace=False)
    indices: List[int] = []
    metadata: List[Dict[str, object]] = []
    for loc_idx in chosen:
        location = dataset.locations[loc_idx]
        season_dir = Path(dataset.s1_dir) / location
        if season_dir.exists():
            season_folders = sorted((p for p in season_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
        else:
            season_folders = []
        for season_idx in range(seasons_per_location):
            indices.append(loc_idx * seasons_per_location + season_idx)
            metadata.append(
                {
                    "location": location,
                    "season_index": season_idx,
                    "season_name": season_folders[season_idx].name if season_idx < len(season_folders) else f"season_{season_idx}",
                }
            )
    return indices, metadata


def stack_features(model: torch.nn.Module, dataset: SSL4EODataset, indices: Sequence[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    s1_feats: List[torch.Tensor] = []
    s2_feats: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for idx in indices:
            s1, s2 = dataset[idx]
            outputs = model.compute_embeddings(s1.unsqueeze(0).to(device), s2.unsqueeze(0).to(device))
            s1_feats.append(outputs["s1_features"].squeeze(0).cpu())
            s2_feats.append(outputs["s2_features"].squeeze(0).cpu())
    return torch.stack(s1_feats), torch.stack(s2_feats)


def _aperture_from_inner(inner_pos: torch.Tensor, eps: float, aperture_logk: Optional[float]) -> torch.Tensor:
    inv_cosh = torch.clamp(1.0 / inner_pos, min=-1.0 + eps, max=1.0 - eps)
    if aperture_logk is not None:
        inv_cosh = torch.clamp(inv_cosh * math.exp(aperture_logk), min=-1.0 + eps, max=1.0 - eps)
    return torch.asin(inv_cosh)


def _extract_curvature_from_state(
    model: LorentzCIIP, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    state = model.state_dict()
    try:
        curv_param = state["curv"]
    except KeyError as exc:
        raise KeyError("LorentzCIIP state dict does not contain curvature parameter 'curv'") from exc
    curv_tensor = curv_param.to(device=device, dtype=dtype)
    return torch.exp(curv_tensor)




def compute_hyperbolic_context(
    model: LorentzCIIP,
    s1_feats: torch.Tensor,
    s2_feats: torch.Tensor,
    *,
    aperture_logk: Optional[float],
) -> Dict[str, torch.Tensor]:
    device = s1_feats.device
    if not isinstance(model, LorentzCIIP):
        raise TypeError("compute_hyperbolic_context requires a LorentzCIIP model instance")
    curvature = _extract_curvature_from_state(model, device=device, dtype=s1_feats.dtype)
    s1_points = s1_feats.to(device=device, dtype=s1_feats.dtype)
    s2_points = s2_feats.to(device=device, dtype=s2_feats.dtype)

    # `log_map0` returns tangent-space (Euclidean) vectors anchored at the origin of
    # the model. We keep their magnitudes for radial diagnostics and normalise only
    # for angular plots below.
    s1_tangent = L.log_map0(s1_points, curvature)
    s2_tangent = L.log_map0(s2_points, curvature)
    s1_distances = torch.norm(s1_tangent, dim=-1)
    s2_distances = torch.norm(s2_tangent, dim=-1)

    def _unit_direction(tangent: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        safe_norm = torch.clamp(distances, min=1e-9).unsqueeze(-1)
        return tangent / safe_norm

    s1_dirs = _unit_direction(s1_tangent, s1_distances)
    s2_dirs = _unit_direction(s2_tangent, s2_distances)

    s1_inner = torch.sqrt(1 + curvature * torch.sum(s1_points**2, dim=-1))
    s2_inner = torch.sqrt(1 + curvature * torch.sum(s2_points**2, dim=-1))

    angles = L.pairwise_oxy_angle(s1_points, s2_points, curvature)
    positive_angles = torch.diagonal(angles)
    eps = float(getattr(model, "hyperbolic_eps", 1e-5))
    aperture_s1 = _aperture_from_inner(s1_inner, eps, aperture_logk)
    aperture_s2 = _aperture_from_inner(s2_inner, eps, aperture_logk)
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


def _pca_reduce(data: np.ndarray, components: int = 2) -> np.ndarray:
    centered = data - data.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:components].T


def plot_angle_aperture(angles: np.ndarray, aperture_s1: np.ndarray, aperture_s2: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(angles, 2.0 * aperture_s1, label="S1", alpha=0.6, color="tab:blue")
    ax.scatter(angles, 2.0 * aperture_s2, label="S2", alpha=0.6, color="tab:orange")
    ax.set_xlabel("Positive pair angle (rad)")
    ax.set_ylabel("Aperture width (rad)")
    ax.set_title("Angle vs. acceptance aperture")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_radial_histogram(s1_norms: np.ndarray, s2_norms: np.ndarray, output: Path) -> None:
    # print(s1_norms)
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = max(20, int(math.sqrt(len(s1_norms))))
    ax.hist(s1_norms, bins=bins, alpha=0.6, density=True, label="S1", color="tab:blue")
    ax.hist(s2_norms, bins=bins, alpha=0.6, density=True, label="S2", color="tab:orange")
    ax.set_xlabel("Geodesic distance from origin (hyperbolic radius)")
    ax.set_ylabel("Density")
    ax.set_title("Radial distribution per modality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_angular_pca(
    s1_dirs: np.ndarray,
    s2_dirs: np.ndarray,
    s1_distances: np.ndarray,
    s2_distances: np.ndarray,
    output: Path,
) -> None:
    coords = _pca_reduce(np.concatenate([s1_dirs, s2_dirs], axis=0), components=2)
    s1_coords = coords[: len(s1_dirs)]
    s2_coords = coords[len(s1_dirs) :]
    distances = np.concatenate([s1_distances, s2_distances])
    norm = Normalize(vmin=distances.min(), vmax=distances.max())
    fig, ax = plt.subplots(figsize=(7, 6))
    sc1 = ax.scatter(s1_coords[:, 0], s1_coords[:, 1], c=s1_distances, cmap="viridis", norm=norm, alpha=0.7, label="S1")
    ax.scatter(s2_coords[:, 0], s2_coords[:, 1], c=s2_distances, cmap="magma", norm=norm, alpha=0.7, marker="^", label="S2")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("Angular PCA of hyperbolic directions")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04).set_label("Hyperbolic distance")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_cone_polar(
    positive_angles: np.ndarray,
    aperture_s1: np.ndarray,
    aperture_s2: np.ndarray,
    output: Path,
    sample_size: int,
    seed: int,
) -> None:
    if len(positive_angles) == 0:
        return
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, len(positive_angles))
    indices = rng.choice(len(positive_angles), size=sample_size, replace=False)
    theta = positive_angles[indices]
    s1_width = 2.0 * aperture_s1[indices]
    s2_width = 2.0 * aperture_s2[indices]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, polar=True)
    base_s1, base_s2, height = 0.6, 1.2, 0.45
    ax.bar(theta, height, width=s1_width, bottom=base_s1, color="tab:blue", alpha=0.4, edgecolor="none", label="S1 cone")
    ax.bar(theta, height, width=s2_width, bottom=base_s2, color="tab:orange", alpha=0.4, edgecolor="none", label="S2 cone")
    ax.scatter(theta, np.full_like(theta, base_s1 + height / 2), color="tab:blue", s=10, alpha=0.7)
    ax.scatter(theta, np.full_like(theta, base_s2 + height / 2), color="tab:orange", s=10, alpha=0.7)
    ax.set_title("Acceptance cones for sampled pairs")
    ax.set_rticks([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_metadata_csv(records: List[Dict[str, object]], output: Path) -> None:
    if not records:
        return
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperbolic embedding visualisations")
    parser.add_argument("--config", type=Path, required=True, help="Training config used for the checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint to load")
    parser.add_argument("--output-dir", type=Path, default=Path("hyperbolic_viz"), help="Directory for generated plots")
    parser.add_argument("--num-locations", type=int, default=64, help="Number of locations to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Torch device (defaults to cuda if available)")
    parser.add_argument("--loss-checkpoint", type=Path, default=None, help="Optional checkpoint containing loss state")
    parser.add_argument("--curvature", type=float, default=None, help="Override curvature for the loss")
    parser.add_argument("--hyperbolic", action="store_true", default=True)
    parser.add_argument("--no-hyperbolic", dest="hyperbolic", action="store_false")
    parser.add_argument("--no-hyperbolic-normalize", action="store_true")
    parser.add_argument("--curvature-init", type=float, default=1.0)
    parser.add_argument("--hyperbolic-eps", type=float, default=1e-5)
    parser.add_argument("--cone-samples", type=int, default=100, help="Samples for the cone polar plot")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    dataset = prepare_dataset(config)
    indices, metadata = sample_indices(dataset, args.num_locations, args.seed)

    checkpoint = load_checkpoint(args.checkpoint, device)
    precision = str(getattr(config.model, "precision", "fp32"))
    input_dtype = resolve_input_dtype(precision)
    model = load_model_from_checkpoint(
        config,
        args.checkpoint,
        device=device,
        input_dtype=input_dtype,
        w_path=None,
        skip_final_fc=False,
        use_orthogonal_mapping=False,
    )

    hyp_params = load_hyperbolic_params(checkpoint, args.hyperbolic_eps)
    setattr(model, "hyperbolic_eps", float(hyp_params["eps"]))
    if args.curvature is not None and isinstance(model, LorentzCIIP):
        new_curv = max(float(args.curvature), 1e-8)
        with torch.no_grad():
            model.curv.copy_(model.curv.new_tensor(math.log(new_curv)))

    s1_feats, s2_feats = stack_features(model, dataset, indices, device)
    s1_feats_device = s1_feats.to(device)
    s2_feats_device = s2_feats.to(device)
    aperture_logk = hyp_params["k_ap"] if hyp_params["k_ap"] is not None else math.log(0.5)

    with torch.no_grad():
        context = compute_hyperbolic_context(
            model,
            s1_feats_device,
            s2_feats_device,
            aperture_logk=aperture_logk,
        )
    positive_angles = context["positive_angles"].cpu().numpy()
    aperture_s1 = context["aperture_s1"].cpu().numpy()
    aperture_s2 = context["aperture_s2"].cpu().numpy()
    s1_dirs = context["s1_dirs"].cpu().numpy()
    s2_dirs = context["s2_dirs"].cpu().numpy()
    s1_distances = context["s1_distances"].cpu().numpy()
    s2_distances = context["s2_distances"].cpu().numpy()

    curvature_tensor = context["curvature"].to(device)
    with torch.no_grad():
        s1_prelift = L.log_map0(s1_feats_device, curvature_tensor)
        s2_prelift = L.log_map0(s2_feats_device, curvature_tensor)

    s1_norms = s1_prelift.norm(dim=-1).cpu().numpy()
    s2_norms = s2_prelift.norm(dim=-1).cpu().numpy()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_angle_aperture(positive_angles, aperture_s1, aperture_s2, output_dir / "angle_aperture_scatter.png")
    plot_radial_histogram(s1_norms, s2_norms, output_dir / "radial_histograms.png")
    plot_angular_pca(s1_dirs, s2_dirs, s1_distances, s2_distances, output_dir / "angular_pca.png")
    plot_cone_polar(positive_angles, aperture_s1, aperture_s2, output_dir / "cone_polar.png", args.cone_samples, args.seed)

    records: List[Dict[str, object]] = []
    for idx, meta in enumerate(metadata):
        records.append(
            {
                **meta,
                "positive_angle": float(positive_angles[idx]),
                "aperture_s1": float(aperture_s1[idx]),
                "aperture_s2": float(aperture_s2[idx]),
                "s1_norm": float(s1_norms[idx]),
                "s2_norm": float(s2_norms[idx]),
                "s1_distance": float(s1_distances[idx]),
                "s2_distance": float(s2_distances[idx]),
            }
        )
    write_metadata_csv(records, output_dir / "hyperbolic_summary.csv")

    used_curvature = float(curvature_tensor.item())
    aperture_scale = math.exp(aperture_logk) if aperture_logk is not None else float("nan")
    print(f"Saved plots to {output_dir.resolve()}")
    print(f"Effective curvature: {used_curvature:.6f}")
    eps_value = getattr(model, "hyperbolic_eps", 1e-5)
    print(
        "Hyperbolic params — eps: {eps:.2e}, aperture_scale: {ap:.6f} (logK={logk:.3f})".format(
            eps=eps_value,
            ap=aperture_scale,
            logk=aperture_logk if aperture_logk is not None else float("nan"),
        )
    )


if __name__ == "__main__":
    main()
