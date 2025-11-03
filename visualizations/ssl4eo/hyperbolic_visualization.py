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


    python /home/juro4948/ciip/visualizations/ssl4eo/hyperbolic_visualization.py \
        --checkpoint /local/ms-data/SSL4EO/model/2025_11_01-21_12_21-model_resnet50-lr_0.001-b_128-j_6-p_amp/checkpoints/epoch_16.pt \
        --config /home/juro4948/ciip/ciip/open_clip_train/configs/prod_default.yaml \
        --output-dir /home/juro4948/ciip/diagnostics/hyperbolic_viz \
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

try:
    from hydra.core.global_hydra import GlobalHydra
    import hydra
    if not GlobalHydra.instance().is_initialized():
        # Lightweight init; we’re not composing any config here
        hydra.initialize(config_path=None, job_name="hyperbolic_viz", version_base=None)
except Exception:
    pass
try:
    import hydra.utils as _hydra_utils
    _old_get_cwd = _hydra_utils.get_original_cwd
    def _safe_get_original_cwd():
        try:
            return _old_get_cwd()
        except Exception:
            # Not running under Hydra: fall back to current working dir
            return os.getcwd()
    _hydra_utils.get_original_cwd = _safe_get_original_cwd
except Exception:
    pass

import sys
sys.path.append('/home/juro4948/ciip')
sys.path.append('/home/juro4948/ciip/ciip')
from loss import CiipLoss
from model_ciip import CIIP
sys.path.append('/home/juro4948/ciip/ciip/open_clip_train')
from data import SSL4EODataset
from open_clip import get_input_dtype
from visualizations.ssl4eo.embedding_collapse_diagnostics import *

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


from types import SimpleNamespace

def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj

def load_config(config_path):
    """
    Returns (dataset_spec, model_spec) as plain dicts.
    Supports Hydra/OmegaConf YAML (.yaml/.yml) and INI (.ini).
    """
    p = Path(config_path)
    ext = p.suffix.lower()

    # --- YAML / Hydra ---
    if ext in (".yaml", ".yml"):
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(p)

            # prefer common hydra-style keys; fallback to empty dicts
            # tweak these selectors to match your repo’s schema
            def pick(*paths, default=None):
                for path in paths:
                    try:
                        val = OmegaConf.select(cfg, path)
                        if val is not None:
                            return OmegaConf.to_container(val, resolve=True)
                    except Exception:
                        pass
                return default


            dataset_spec = pick("data", "dataset", default={})
            model_spec   = pick("model", "ciip", "train.model", default={})
            dataset_spec = _to_ns(dataset_spec)
            model_spec   = _to_ns(model_spec)

            return dataset_spec, model_spec

        except Exception as e:
            # fallback to plain PyYAML if OmegaConf isn't available
            import yaml
            with open(p, "r") as f:
                cfg = yaml.safe_load(f) or {}
            dataset_spec = cfg.get("data", cfg.get("dataset", {})) or {}
            model_spec   = cfg.get("model", cfg.get("ciip", {})) or {}
            dataset_spec = _to_ns(dataset_spec)
            model_spec   = _to_ns(model_spec)
            return dataset_spec, model_spec

    # --- INI ---
    elif ext in (".ini",):
        import configparser
        parser = configparser.ConfigParser()
        ok = parser.read(config_path)
        if not ok:
            raise FileNotFoundError(config_path)
        # map sections to your expected dicts
        dataset_spec = dict(parser["dataset"]) if "dataset" in parser else {}
        model_spec   = dict(parser["model"])   if "model" in parser else {}
        return dataset_spec, model_spec

    else:
        raise ValueError(f"Unsupported config format: {ext} (use .yaml/.yml or .ini)")


def load_model_from_checkpoint(
    config: DictConfig,
    checkpoint_path: Path,
    *,
    device: torch.device,
    input_dtype: torch.dtype,
    w_path: Optional[Path] = None,
    skip_final_fc: bool = False,
    use_orthogonal_mapping: bool = False,
) -> torch.nn.Module:
    """Instantiate a CIIP model and load weights from ``checkpoint_path``."""

    model = create_model(config, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    def _check_and_replace_fc_layers(model, state_dict):
        """Check fc layer dimensions and replace if incompatible."""
        replacements_made = []
        
        for encoder_name in ['encoder_s1', 'encoder_s2']:
            if not hasattr(model, encoder_name):
                continue
                
            encoder = getattr(model, encoder_name)
            if not hasattr(encoder, 'fc'):
                continue
                
            fc_weight_key = f"{encoder_name}.fc.weight"
            fc_bias_key = f"{encoder_name}.fc.bias"
            
            if fc_weight_key not in state_dict:
                continue
                
            # Get dimensions from checkpoint and model
            checkpoint_weight = state_dict[fc_weight_key]
            model_fc = encoder.fc
            
            checkpoint_shape = checkpoint_weight.shape
            model_shape = model_fc.weight.shape
            
            if checkpoint_shape != model_shape:
                # _LOGGER.warning(
                #     f"FC layer dimension mismatch in {encoder_name}: "
                #     f"checkpoint {checkpoint_shape} vs model {model_shape}. "
                #     f"Replacing with compatible layer."
                # )
                
                # Create new fc layer with checkpoint dimensions
                in_features = checkpoint_shape[1]
                out_features = checkpoint_shape[0]
                has_bias = fc_bias_key in state_dict
                
                new_fc = nn.Linear(in_features, out_features, bias=has_bias)
                setattr(encoder, 'fc', new_fc)
                replacements_made.append(f"{encoder_name}.fc")
        
        return replacements_made

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        # _LOGGER.info(f"Initial load failed: {e}. Attempting fixes...")
        
        # Try removing module prefix
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(cleaned, strict=True)
        except Exception:
            # Remove problematic keys
            if "encoder_s1.W" in cleaned:
                del cleaned["encoder_s1.W"]
            
            # Check and replace incompatible fc layers
            replacements = _check_and_replace_fc_layers(model, cleaned)
            # if replacements:
            #     _LOGGER.info(f"Replaced incompatible layers: {replacements}")
            
            # Check for remaining compatibility issues
            missing, unexpected = compare_state_keys(model, cleaned)
            
            # Filter out the layers we just replaced from missing keys
            missing_filtered = set()
            for key in missing:
                is_replaced_layer = any(key.startswith(repl) for repl in replacements)
                if not is_replaced_layer:
                    missing_filtered.add(key)
            
            if missing_filtered or unexpected:
                # _LOGGER.warning(f"Missing keys: {missing_filtered}")
                # _LOGGER.warning(f"Unexpected keys: {unexpected}")
                
                # Only raise error if there are critical missing keys
                critical_missing = {k for k in missing_filtered 
                                  if not any(pattern in k for pattern in ['fc.weight', 'fc.bias', 'W'])}
                if critical_missing:
                    raise RuntimeError(
                        f"Checkpoint {checkpoint_path} has critical incompatibilities: {critical_missing}"
                    )
            
            # Load with non-strict mode to handle the replaced layers
            model.load_state_dict(cleaned, strict=False)
        else:
            state_dict = cleaned

    use_w_mapping = bool(use_orthogonal_mapping and w_path is not None and w_path.exists())
    if use_w_mapping:
        try:
            orthogonal = torch.load(w_path, map_location=device)
        except Exception as exc:  # pragma: no cover - defensive branch
            # _LOGGER.warning("Failed to load orthogonal mapping from %s: %s", w_path, exc)
            use_w_mapping = False
        else:
            if hasattr(model, "encoder_s1"):
                model.encoder_s1.register_buffer("W", orthogonal)
    if hasattr(model, "encoder_s1") and hasattr(model.encoder_s1, "apply_orthogonal_matrix"):
        model.encoder_s1.apply_orthogonal_matrix = use_w_mapping

    if input_dtype is not None:
        model = model.to(device, dtype=input_dtype, non_blocking=True)
    else:
        model = model.to(device, non_blocking=True)

    if skip_final_fc:
        if hasattr(model, "encoder_s1") and hasattr(model.encoder_s1, "fc"):
            model.encoder_s1.fc = nn.Identity()
        if hasattr(model, "encoder_s2") and hasattr(model.encoder_s2, "fc"):
            model.encoder_s2.fc = nn.Identity()

    model.eval()
    return model
    
def resolve_input_dtype(precision: str) -> torch.dtype:
    """Resolve the tensor dtype used for model inputs."""

    dtype: Optional[torch.dtype] = None
    if get_input_dtype is not None:
        dtype = get_input_dtype(precision)
    if dtype is not None:
        return dtype

    precision = precision.lower()
    if "bf16" in precision:
        return torch.bfloat16
    if "fp16" in precision or "amp" in precision:
        return torch.float16
    return torch.float32

def build_model(spec, checkpoint_path, device):
    input_dtype = resolve_input_dtype(spec.model.precision)
    model = load_model_from_checkpoint(
            spec,
            checkpoint_path,
            device=device,
            input_dtype=input_dtype,
            w_path=None,
            skip_final_fc=False,
            use_orthogonal_mapping=False,
        )
    
    return model.to(device)



# def load_model_weights(model: CIIP, checkpoint_path: Path, device: torch.device) -> Dict:
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     if "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"]
#     else:
#         state_dict = checkpoint

#     cleaned = {}
#     for key, value in state_dict.items():
#         if key.startswith("module."):
#             cleaned[key[len("module."):]] = value
#         else:
#             cleaned[key] = value

#     missing, unexpected = model.load_state_dict(cleaned, strict=False)
#     if missing:
#         print(f"Warning: Missing model parameters: {missing}")
#     if unexpected:
#         print(f"Warning: Unexpected parameters in checkpoint: {unexpected}")
#     return checkpoint


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


def prepare_dataset(spec: DatasetSpec, model_spec: DatasetSpec) -> SSL4EODataset:
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
            s2_bands=model_spec.s2_bands,
            transforms=None,
            target_image_dimension=(224, 224),
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

    # dataset_spec, model_spec = load_config(args.config)
    config = OmegaConf.load(args.config)
    OmegaConf.set_struct(config, False)

    # print(dataset_spec)
    # print(model_spec)
    dataset = prepare_dataset(config.dataset, config.model)

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
