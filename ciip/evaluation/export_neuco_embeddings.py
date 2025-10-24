#!/usr/bin/env python3
"""Export CIIP S2 encoder embeddings for NeuCo-Bench evaluation."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import zarr
    from zarr.storage import ZipStore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The `zarr` package is required to read NeuCo-Bench tiles. "
        "Install it with `pip install zarr` or add it to your environment."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PROJECT_ROOT, PACKAGE_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from ciip.eval_utils import create_ciip_model  # type: ignore


# Sentinel-2 per-band statistics reused from the training pipeline
S2_L1C_MEAN = np.array(
    [
        1605.57504906,
        1390.78157673,
        1314.8729939,
        1363.52445545,
        1549.44374991,
        2091.74883118,
        2371.7172463,
        2299.90463006,
        2560.29504086,
        830.06605044,
        22.10351321,
        2177.07172323,
        1524.06546312,
    ],
    dtype=np.float32,
)
S2_L1C_STD = np.array(
    [
        786.78685367,
        850.34818441,
        875.06484736,
        1138.84957046,
        1122.17775652,
        1161.59187054,
        1274.39184232,
        1248.42891965,
        1345.52684884,
        577.31607053,
        51.15431158,
        1336.09932639,
        1136.53823676,
    ],
    dtype=np.float32,
)

S2_L2A_MEAN = np.array(
    [
        752.40087073,
        884.29673756,
        1144.16202635,
        1297.47289228,
        1624.90992062,
        2194.6423161,
        2422.21248945,
        2517.76053101,
        2581.64687018,
        2645.51888987,
        2368.51236873,
        1805.06846033,
    ],
    dtype=np.float32,
)
S2_L2A_STD = np.array(
    [
        1108.02887453,
        1155.15170768,
        1183.6292542,
        1368.11351514,
        1370.265037,
        1355.55390699,
        1416.51487101,
        1474.78900051,
        1439.3086061,
        1582.28010962,
        1455.52084939,
        1343.48379601,
    ],
    dtype=np.float32,
)


MODALITY_STATS = {
    "s2l1c": (S2_L1C_MEAN, S2_L1C_STD),
    "s2l2a": (S2_L2A_MEAN, S2_L2A_STD),
}


@dataclass
class DatasetConfig:
    data_root: Path
    modality: str
    expected_channels: int
    resize_hw: Optional[Tuple[int, int]]
    time_reduction: str
    clip_range: Optional[Tuple[float, float]]
    scale: Optional[float]
    normalize_mode: str


class NeucoZarrDataset(Dataset[Tuple[str, torch.Tensor]]):
    """Dataset that loads NeuCo-Bench tiles stored as ``*.zarr.zip`` files."""

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        modality_dir = cfg.data_root / cfg.modality
        if not modality_dir.exists():
            raise FileNotFoundError(f"Modality directory not found: {modality_dir}")

        self.paths = sorted(modality_dir.glob("*.zarr.zip"))
        if not self.paths:
            raise FileNotFoundError(f"No zarr.zip files found under {modality_dir}")

        self.ids = [p.stem.replace(".zarr", "") for p in self.paths]
        self._stats = MODALITY_STATS.get(cfg.modality)
        if cfg.normalize_mode == "percentile" and self._stats is None:
            raise ValueError(
                f"No per-band statistics registered for modality '{cfg.modality}'. "
                "Use --normalize-mode=none or provide supported modality."
            )

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        sample_id = self.ids[index]
        path = self.paths[index]
        array = _load_zarr_array(path)
        tensor = _to_chw_tensor(
            array,
            expected_channels=self.cfg.expected_channels,
            time_reduction=self.cfg.time_reduction,
        )
        tensor = _apply_preprocessing(
            tensor,
            normalize_mode=self.cfg.normalize_mode,
            stats=self._stats,
            scale=self.cfg.scale,
            clip_range=self.cfg.clip_range,
        )
        if self.cfg.resize_hw is not None and tuple(tensor.shape[-2:]) != self.cfg.resize_hw:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=self.cfg.resize_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return sample_id, tensor.contiguous()


def _load_zarr_array(path: Path) -> np.ndarray:
    """Load the first dense array stored in a ``.zarr.zip`` archive."""

    with ZipStore(str(path), mode="r") as store:
        root = zarr.open(store=store, mode="r")
        if isinstance(root, zarr.Array):
            return np.asarray(root[...])

        if not isinstance(root, zarr.Group):  # pragma: no cover - defensive
            raise TypeError(f"Unexpected zarr root type for {path}: {type(root)!r}")

        preferred_keys = ("patch", "image", "arr_0", "data")
        for key in preferred_keys:
            if key in root:
                return np.asarray(root[key][...])

        array_keys = list(root.array_keys())
        if not array_keys:
            raise ValueError(f"No arrays found inside {path}")
        if len(array_keys) > 1:
            logging.warning(
                "Multiple arrays found in %s; defaulting to '%s'", path, array_keys[0]
            )
        return np.asarray(root[array_keys[0]][...])


def _to_chw_tensor(
    array: np.ndarray,
    *,
    expected_channels: int,
    time_reduction: str,
) -> torch.Tensor:
    """Convert an arbitrary array layout to a ``(C, H, W)`` float tensor."""

    arr = np.asarray(array)
    if arr.ndim < 2:
        raise ValueError(f"Expected array with at least 2 dimensions, got shape {arr.shape}")

    channel_axis = None
    for axis, size in enumerate(arr.shape):
        if size == expected_channels:
            channel_axis = axis
            break
    if channel_axis is None:
        raise ValueError(
            f"Unable to locate channel axis with size {expected_channels} in shape {arr.shape}"
        )

    arr = np.moveaxis(arr, channel_axis, 0)

    if arr.ndim > 3:
        # sort non-channel axes so that small temporal dims come first
        other_axes = list(range(1, arr.ndim))
        other_axes.sort(key=lambda ax: arr.shape[ax])
        arr = np.transpose(arr, (0, *other_axes))
        reducer = _select_reducer(time_reduction)
        while arr.ndim > 3:
            arr = reducer(arr, axis=1)

    if arr.ndim != 3:
        raise ValueError(f"Expected (C, H, W) array after reduction, got shape {arr.shape}")

    return torch.from_numpy(arr.astype(np.float32, copy=False))


def _select_reducer(strategy: str):
    strategy = strategy.lower()
    if strategy == "mean":
        return np.mean
    if strategy == "median":
        return np.median
    if strategy == "max":
        return np.max
    if strategy == "first":
        return lambda arr, axis: np.take(arr, indices=0, axis=axis)
    raise ValueError(f"Unsupported time reduction strategy: {strategy}")


def _apply_preprocessing(
    tensor: torch.Tensor,
    *,
    normalize_mode: str,
    stats: Optional[Tuple[np.ndarray, np.ndarray]],
    scale: Optional[float],
    clip_range: Optional[Tuple[float, float]],
) -> torch.Tensor:
    normalize_mode = normalize_mode.lower()
    if normalize_mode not in {"none", "percentile", "standardize"}:
        raise ValueError(f"Unknown normalize mode: {normalize_mode}")

    x = tensor.to(torch.float32)
    if scale is not None:
        x = x / scale
    if clip_range is not None:
        x = x.clamp(min=clip_range[0], max=clip_range[1])

    if normalize_mode == "none" or stats is None:
        return x

    mean, std = stats
    mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)

    if normalize_mode == "standardize":
        return (x - mean_t) / std_t

    min_val = mean_t - 2 * std_t
    max_val = mean_t + 2 * std_t
    x = (x - min_val) / (max_val - min_val)
    return x.clamp_(0.0, 1.0)


def _collate(batch: Sequence[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    ids, tensors = zip(*batch)
    stacked = torch.stack(tensors, dim=0)
    return list(ids), stacked


def _infer_model_dims(state_dict: dict) -> Tuple[int, int]:
    embed_dim = None
    pre_proj_dim = None
    for key, value in state_dict.items():
        if key.endswith("encoder_s2.proj.weight"):
            embed_dim, pre_proj_dim = value.shape[0], value.shape[1]
        elif key.endswith("encoder_s2.fc.weight"):
            pre_proj_dim = value.shape[0]
        if embed_dim is not None and pre_proj_dim is not None:
            break
    if pre_proj_dim is None:
        raise ValueError("Could not infer pre-projection dimension from checkpoint")
    if embed_dim is None:
        # fall back to encoder_s2.fc bias for embed dim == pre-projection dim
        embed_dim = pre_proj_dim
    return embed_dim, pre_proj_dim


def _load_model(weights_path: Path, embed_dim: Optional[int], pre_proj_dim: Optional[int]) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    inferred_embed, inferred_preproj = _infer_model_dims(state_dict)
    embed_dim = embed_dim or inferred_embed
    pre_proj_dim = pre_proj_dim or inferred_preproj

    model = create_ciip_model(embed_dim=embed_dim, pre_projection_dim=pre_proj_dim)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "encoder_s1.fc.weight",
        "encoder_s1.fc.bias",
        "encoder_s2.fc.weight",
        "encoder_s2.fc.bias",
    }
    unexpected = set(unexpected)
    remaining_missing = set(missing) - allowed_missing
    if remaining_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint compatibility issue. Missing: {sorted(remaining_missing)}, "
            f"Unexpected: {sorted(unexpected)}"
        )
    logging.info("Loaded checkpoint from %s", weights_path)
    return model


def _prepare_encoder(model: torch.nn.Module) -> torch.nn.Module:
    encoder = model.encoder_s2  # type: ignore[attr-defined]
    proj = getattr(encoder, "proj", None)
    if isinstance(proj, torch.nn.Module) and not isinstance(proj, torch.nn.Identity):
        logging.info("Disabling projection head on encoder_s2 to expose pre-projection features")
        encoder.proj = torch.nn.Identity()
    return encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="NeuCo-Bench root directory")
    parser.add_argument(
        "--modality",
        default="s2l1c",
        choices=sorted(MODALITY_STATS.keys()),
        help="NeuCo-Bench modality sub-folder",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to CIIP checkpoint (.pt)")
    parser.add_argument("--csv-out", type=Path, required=True, help="Output CSV file path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count")
    parser.add_argument("--device", default=None, help="Torch device override (default: auto-detect)")
    parser.add_argument("--embed-dim", type=int, default=None, help="Override embed dimension if checkpoint metadata is absent")
    parser.add_argument(
        "--pre-projection-dim",
        type=int,
        default=None,
        help="Override pre-projection dimension if checkpoint metadata is absent",
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=13,
        help="Expected number of Sentinel-2 bands in each tile",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Resize tiles to the provided spatial size before encoding",
    )
    parser.add_argument(
        "--time-reduction",
        choices=["mean", "median", "max", "first"],
        default="mean",
        help="Reduction to apply when multiple temporal slices are present",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10000.0,
        help="Value to divide raw radiances by before optional clipping",
    )
    parser.add_argument(
        "--clip",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 1.0),
        help="Range to clip values to after scaling (set to omit for no clipping)",
    )
    parser.add_argument(
        "--normalize-mode",
        choices=["none", "percentile", "standardize"],
        default="percentile",
        help="Per-band normalization strategy",
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        default=None,
        help="Validate that extracted features match the requested dimensionality",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use autocast(fp16/bf16) inference if the device supports it",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a tqdm progress bar while encoding",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logging.info("Using device: %s", device)

    model = _load_model(args.weights, args.embed_dim, args.pre_projection_dim)
    model.eval()
    model.to(device)
    encoder = _prepare_encoder(model).to(device)
    encoder.eval()

    clip_range = tuple(args.clip) if args.clip is not None else None
    dataset_cfg = DatasetConfig(
        data_root=args.data_root,
        modality=args.modality,
        expected_channels=args.expected_channels,
        resize_hw=tuple(args.resize) if args.resize is not None else None,
        time_reduction=args.time_reduction,
        clip_range=clip_range,
        scale=args.scale,
        normalize_mode=args.normalize_mode,
    )
    dataset = NeucoZarrDataset(dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    header_written = False

    autocast_context = torch.autocast(device_type=device.type, enabled=args.amp) if args.amp else nullcontext()

    with args.csv_out.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        iterator: Iterable = tqdm(dataloader, desc="Encoding", unit="batch") if args.progress else dataloader
        for batch_ids, batch_tensor in iterator:
            batch_tensor = batch_tensor.to(device)
            with torch.no_grad(), autocast_context:
                feats = encoder(batch_tensor)
            feats = feats.float().cpu().numpy()

            if args.target_dim is not None and feats.shape[1] != args.target_dim:
                raise ValueError(
                    f"Expected embeddings of dim {args.target_dim}, got {feats.shape[1]}"
                )

            if not header_written:
                header = ["id"] + [f"e{i}" for i in range(feats.shape[1])]
                writer.writerow(header)
                header_written = True

            for sid, emb in zip(batch_ids, feats):
                writer.writerow([sid] + emb.tolist())

    logging.info("Saved embeddings to %s", args.csv_out)


class nullcontext:  # pragma: no cover - lightweight shim for Python <3.7 compat
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        return False


if __name__ == "__main__":
    main()
