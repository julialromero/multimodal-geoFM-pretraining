#!/usr/bin/env python3
"""
Export embeddings for NeuCo-Bench from a pretrained CIIP model
(closely following the style of:
  - examples/data/dataset.py
  - examples/data/submission_utils.py
  - examples/S2_dino_embeddings.py
in the NeuCo-Bench repo)

Assumptions:
- Data root: /.../SSL4EO-S12-downstream/
- Modalities under data/: s1/, s2l1c/, s2l2a/
- Tiles are .zarr.zip inside part-* subfolders (or plain .zarr directories)
- Downstream labels use the tile filename stem (id column)

This script builds a Dataset/DataLoader to iterate tiles, runs your CIIP encoder,
and writes a NeuCo-style CSV: id,e0,...,e{D-1}
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from zarr.storage import ZipStore

from ciip.eval_utils import create_ciip_model

# ------------------------
# Config & normalisation
# ------------------------

S2L1C_MEAN = [2607.345, 2393.068, 2320.225, 2373.963, 2562.536, 3110.071, 3392.832, 3321.154, 3583.77, 1838.712, 1021.753, 3205.112, 2545.798]
S2L1C_STD = [786.523, 849.702, 875.318, 1143.578, 1126.248, 1161.98, 1273.505, 1246.79, 1342.755, 576.795, 45.626, 1340.347, 1145.036]

S2L2A_MEAN = [1793.243, 1924.863, 2184.553, 2340.936, 2671.402, 3240.082, 3468.412, 3563.244, 3627.704, 3711.071, 3416.714, 2849.625]
S2L2A_STD = [1160.144, 1201.092, 1219.943, 1397.225, 1400.035, 1373.136, 1429.17, 1485.025, 1447.836, 1652.703, 1471.002, 1365.307]

S1GRD_MEAN = [-12.577, -20.265]
S1GRD_STD = [5.179, 5.872]

MODALITY_STATS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "s2l1c": (S2L1C_MEAN, S2L1C_STD),
    "s2l2a": (S2L2A_MEAN, S2L2A_STD),
    # add s1 if you use it
}

# # ------------------------

def _open_xr_zarr_any(spath: str) -> xr.Dataset:
    """Open a zarr store (zip or dir), robust to consolidated/non-consolidated."""
    if spath.endswith(".zarr.zip"):
        with ZipStore(spath, mode="r") as zs:
            try:
                return xr.open_zarr(zs, consolidated=True)
            except Exception:
                return xr.open_zarr(zs, consolidated=False)
    try:
        return xr.open_zarr(spath, consolidated=True)
    except Exception:
        return xr.open_zarr(spath, consolidated=False)

# def _isel_seasons_on_dataset(ds: xr.Dataset, seasons: Sequence[int], randomize: bool, k: int) -> xr.Dataset:
#     if "time" not in ds.dims:
#         return ds
#     sel = list(seasons)
#     if randomize and len(sel) > 1:
#         import torch as _torch
#         order = _torch.randperm(len(sel)).tolist()[:k]
#         sel = [sel[i] for i in order]
#     else:
#         sel = sel[:k]
#     season_index = xr.DataArray(sel, dims="time")
#     return ds.isel(time=season_index)

# def _pick_spatial_var(ds: xr.Dataset, preferred: Optional[str]) -> xr.DataArray:
#     """Choose a variable with ('y','x') dims. Prefer `preferred` if suitable."""
#     def has_spatial(da: xr.DataArray) -> bool:
#         d = set(da.dims)
#         return ("y" in d) and ("x" in d)

#     if preferred and preferred in ds.data_vars and has_spatial(ds[preferred]):
#         return ds[preferred]

#     candidates = []
#     for name, da in ds.data_vars.items():
#         if has_spatial(da):
#             size = int(np.prod(da.shape))
#             candidates.append((size, name, da))
#     if not candidates:
#         raise ValueError(f"No data variable with ('y','x') found. Vars: {[(k, tuple(ds[k].dims)) for k in ds.data_vars]}")
#     candidates.sort(reverse=True)
#     return candidates[0][2]

# def _da_to_tc_hw(da: xr.DataArray) -> np.ndarray:
#     """
#     Convert any spatial DataArray to [T, C, H, W] float32.
#     Handles extra dims like 'sample' by stacking them with 'time' into a single time-like dim
#     without renaming (avoids conflicts).
#     """
#     dims = list(da.dims)

#     # require spatial dims
#     if "y" not in dims or "x" not in dims:
#         raise ValueError(f"Expected 'y' and 'x' dims; got dims {da.dims}")

#     # decide the time-like dim name we will use for transpose
#     time_like = None

#     if "sample" in dims and "time" in dims:
#         # stack into a temp dim that does NOT clash with any existing name
#         da = da.stack(stacked=("sample", "time"))  # new dim 'stacked'
#         time_like = "stacked"
#     elif "time" in dims:
#         time_like = "time"
#     elif "sample" in dims:
#         # treat 'sample' as time-like
#         time_like = "sample"
#     else:
#         # no time/sample → add singleton time
#         time_like = "time"
#         da = da.expand_dims({time_like: [0]})

#     # ensure band exists
#     if "band" not in da.dims:
#         da = da.expand_dims(band=[0])

#     # canonical order using the chosen time-like dim
#     da = da.transpose(time_like, "band", "y", "x", missing_dims="ignore")

#     arr = da.values  # shape [T,C,H,W]
#     if arr.ndim != 4:
#         raise ValueError(f"Expected 4D after transpose, got {arr.shape} with dims {da.dims}")
#     return arr.astype(np.float32, copy=False)



# def _to_tensor_tc_hw(values: np.ndarray) -> torch.Tensor:
#     """np [T,C,H,W] -> torch float32 [T,C,H,W]"""
#     if values.ndim == 3:
#         values = values[None, ...]
#     if values.ndim != 4:
#         raise ValueError(f"Expected [T,C,H,W], got {values.shape}")
#     return torch.from_numpy(values.astype(np.float32, copy=False))

# def _reduce_time(x: torch.Tensor, how: Optional[str]) -> torch.Tensor:
#     """[T,C,H,W] -> [C,H,W] if how is not None; else return [T,C,H,W]."""
#     if how is None:
#         return x
#     if x.ndim != 4:
#         raise ValueError(f"time reduction expects [T,C,H,W], got {tuple(x.shape)}")
#     if how == "first":
#         return x[0]
#     if how == "mean":
#         return x.mean(dim=0)
#     if how == "median":
#         return x.median(dim=0).values
#     raise ValueError(f"Unknown time_reduction={how}")

# def _apply_preprocessing(
#     tensor: torch.Tensor,
#     *,
#     normalize_mode: str,
#     stats: Optional[Tuple[np.ndarray, np.ndarray]],
#     scale: Optional[float],
#     clip_range: Optional[Tuple[float, float]],
# ) -> torch.Tensor:
#     """Mimic example-style preprocessing: scale (/10000), clip, optional standardize/percentile window."""
#     x = tensor.to(torch.float32)
#     if scale is not None:
#         x = x / float(scale)
#     if clip_range is not None:
#         x = x.clamp(min=float(clip_range[0]), max=float(clip_range[1]))

#     nm = (normalize_mode or "none").lower()
#     if nm == "none" or stats is None:
#         return x

#     mean, std = stats
#     mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
#     std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)

#     if nm == "standardize":
#         return (x - mean_t) / std_t

#     # simple percentile-ish windowing via mean±2*std to [0,1]
#     lo = mean_t - 2 * std_t
#     hi = mean_t + 2 * std_t
#     x = (x - lo) / (hi - lo)
#     return x.clamp_(0.0, 1.0)


# # ------------------------
# # Dataset (NeuCo style)
# # ------------------------

# class NeucoZarrDataset(Dataset[Tuple[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]):
#     """
#     - discovers samples via first modality folder (recursive part-*/ *.zarr.zip)
#     - swaps modality token in path to locate other modalities (same filename stem)
#     - opens zarr with ZipStore if needed
#     - selects seasons on the DATASET (robust if var lacks 'time')
#     - picks a SPATIAL var (has 'y','x'), converts to [T,C,H,W]
#     - optional +1000 shift for S2
#     - per-modality normalization
#     - optional time reduction + resize
#     - returns (id, concatenated tensor) if cfg.concat else (id, dict per modality)
#     """
#     def __init__(self, cfg: DatasetConfig) -> None:
#         self.cfg = cfg

#         first_mod = cfg.modalities[0]
#         first_dir = (cfg.data_root / first_mod)
#         if not first_dir.exists():
#             raise FileNotFoundError(f"Modality directory not found: {first_dir}")

#         samples = sorted(str(p) for p in first_dir.rglob("*.zarr.zip"))
#         if not samples:
#             samples = sorted(str(p) for p in first_dir.rglob("*.zarr"))
#         if not samples:
#             raise FileNotFoundError(f"No zarr stores found under {first_dir} (recursive)")
#         self.samples: List[str] = samples
#         self.ids: List[str] = [Path(p).name.replace(".zarr.zip","").replace(".zarr","") for p in samples]

#         # modality stats map
#         self._stats = {m: MODALITY_STATS.get(m) for m in cfg.modalities}
#         if cfg.normalize_mode == "percentile":
#             missing = [m for m,s in self._stats.items() if s is None]
#             if missing:
#                 logging.warning("No percentile stats for %s — proceeding with 'none' for those.", missing)

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, idx: int) -> Tuple[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
#         base_path = self.samples[idx]
#         sample_id = self.ids[idx]
#         # locate same id across modalities by swapping the modality token in path
#         paths = [base_path] + [base_path.replace(self.cfg.modalities[0] + "/", m + "/") for m in self.cfg.modalities[1:]]

#         seasons = list(self.cfg.possible_seasons)
#         k = self.cfg.seasons

#         per_mod: Dict[str, torch.Tensor] = {}
#         for modality, spath in zip(self.cfg.modalities, paths):
#             ds = _open_xr_zarr_any(spath)
#             ds = _isel_seasons_on_dataset(ds, seasons, self.cfg.randomize_seasons, k)
#             da = _pick_spatial_var(ds, self.cfg.dataset_name)
#             values = _da_to_tc_hw(da)  # np [T,C,H,W]

#             # optional S2 shift (+1000) to align SSL4EO-S12 v1.1
#             if self.cfg.shift_s2_channels and modality in ("s2l1c","s2l2a"):
#                 values = values + 1000.0

#             t = _to_tensor_tc_hw(values)  # [T,C,H,W]

#             # normalization per modality
#             t_for_norm = t if self.cfg.time_reduction is None else _reduce_time(t, None)  # keep time during per-sample ops
#             t = _apply_preprocessing(
#                 t_for_norm,
#                 normalize_mode=self.cfg.normalize_mode,
#                 stats=self._stats.get(modality),
#                 scale=self.cfg.scale,
#                 clip_range=self.cfg.clip_range,
#             )
#             # time reduce if requested
#             t = _reduce_time(t, self.cfg.time_reduction)  # [C,H,W] or [T,C,H,W]

#             # resize (handles both [C,H,W] and [T,C,H,W])
#             if self.cfg.resize_hw is not None:
#                 if t.ndim == 3:
#                     if tuple(t.shape[-2:]) != self.cfg.resize_hw:
#                         t = F.interpolate(t.unsqueeze(0), size=self.cfg.resize_hw, mode="bilinear", align_corners=False).squeeze(0)
#                 elif t.ndim == 4:
#                     if tuple(t.shape[-2:]) != self.cfg.resize_hw:
#                         t = F.interpolate(t, size=self.cfg.resize_hw, mode="bilinear", align_corners=False)
#                 else:
#                     raise ValueError(f"Unexpected tensor shape {tuple(t.shape)}")

#             per_mod[modality] = t.contiguous()

#         if not self.cfg.concat:
#             return sample_id, per_mod

#         # concat over channel axis (-3)
#         first = next(iter(per_mod.values()))
#         if first.ndim == 4:
#             # [T,C,H,W] — ensure same T across modalities
#             T = first.shape[0]
#             tensors = []
#             for m in self.cfg.modalities:
#                 t = per_mod[m]
#                 if t.ndim != 4 or t.shape[0] != T:
#                     raise ValueError("All modalities must share the same time length before concat.")
#                 tensors.append(t)
#             out = torch.cat(tensors, dim=-3)
#         else:
#             # [C,H,W]
#             out = torch.cat([per_mod[m] for m in self.cfg.modalities], dim=-3)

#         return sample_id, out


# # ------------------------
# # Submission CSV writer
# # ------------------------

# class SubmissionWriter:
#     """Stream embeddings to NeuCo-Bench CSV: id,e0,...,e{D-1}"""
#     def __init__(self, csv_path: Path, dim: Optional[int] = None):
#         self.csv_path = csv_path
#         self.dim = dim
#         self._fh = None
#         self._writer = None
#         self._header_written = False

#     def __enter__(self):
#         self.csv_path.parent.mkdir(parents=True, exist_ok=True)
#         self._fh = self.csv_path.open("w", newline="")
#         self._writer = csv.writer(self._fh)
#         return self

#     def write_batch(self, ids: List[str], emb: np.ndarray):
#         if self.dim is not None and emb.shape[1] != self.dim:
#             raise ValueError(f"Expected embedding dim {self.dim}, got {emb.shape[1]}")
#         if not self._header_written:
#             D = emb.shape[1]
#             self._writer.writerow(["id"] + [f"e{i}" for i in range(D)])
#             self._header_written = True
#         for sid, row in zip(ids, emb):
#             self._writer.writerow([sid] + row.tolist())

#     def __exit__(self, exc_type, exc, tb):
#         if self._fh:
#             self._fh.close()


# # ------------------------
# # Model loading (CIIP)
# # ------------------------

def _infer_model_dims(state_dict: dict) -> Tuple[int, int]:
    """Return (embed_dim, pre_proj_dim) by peeking known keys."""
    embed_dim = None
    pre_proj_dim = None
    for k, v in state_dict.items():
        if k.endswith("encoder_s2.proj.weight"):
            embed_dim, pre_proj_dim = v.shape[0], v.shape[1]
        elif k.endswith("encoder_s2.fc.weight"):
            pre_proj_dim = v.shape[0]
        if embed_dim is not None and pre_proj_dim is not None:
            break
    if pre_proj_dim is None:
        raise ValueError("Could not infer pre-projection dimension from checkpoint")
    if embed_dim is None:
        embed_dim = pre_proj_dim
    return embed_dim, pre_proj_dim

def load_ciip_model(weights_path: Path, embed_dim: Optional[int], pre_proj_dim: Optional[int]) -> torch.nn.Module:
    if 'dino' not in str(weights_path).lower() and 'moco' not in str(weights_path).lower():
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        i_embed, i_pre = _infer_model_dims(sd)
        embed_dim = embed_dim or i_embed
        pre_proj_dim = pre_proj_dim or i_pre

    model = create_ciip_model(embed_dim=embed_dim, pre_projection_dim=pre_proj_dim)
    if 'dino' not in str(weights_path).lower() and 'moco' not in str(weights_path).lower():
        missing, unexpected = model.load_state_dict(sd, strict=False)
        allowed_missing = {
            "encoder_s1.fc.weight", "encoder_s1.fc.bias",
            "encoder_s2.fc.weight", "encoder_s2.fc.bias",
        }
        unexpected = set(unexpected)
        remain_missing = set(missing) - allowed_missing
        if remain_missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch. Missing: {sorted(remain_missing)}, Unexpected: {sorted(unexpected)}")
        logging.info("Loaded checkpoint from %s", weights_path)


    return model

def prepare_s2_encoder(model: torch.nn.Module) -> torch.nn.Module:
    enc = model.encoder_s2  # type: ignore[attr-defined]
    fc = getattr(enc, "fc", None)
    if isinstance(fc, torch.nn.Module) and not isinstance(fc, torch.nn.Identity):
        logging.info("Disabling fc head on encoder_s2 (using pre-projection features).")
        enc.fc = torch.nn.Identity()
    return enc




class E2SChallengeDataset(Dataset):

    def __init__(self, 
                 data_path: str = None, 
                 transform = None, 
                 modalities: List[str] = None,
                 dataset_name: str = 'bands', 
                 seasons: int = 4, 
                 randomize_seasons: bool = False,
                 concat: bool = True,
                 output_file_name: bool = False,
                 shift_s2_channels: bool = True,
                ):
        """Dataset class for the embed2scale challenge data

        Parameters
        ----------
        data_path : str, path-like
            Path to challenge data. Assumes that under data_path there are 3 subfolders, named after the modalities.
        transform : torch.Compose
            Transformations to apply to the data
        modalities : list[str]
            List of modalities to include. Should correpond to the subfolders under data_path.
        dataset_name : str
            Name of dataset in zarr archive. Use 'bands' here. Defaults to 'bands'.
        seasons : int
            Number of seasons to load. Must be integer between 1 and 4. Default is 4.
        randomize_seasons : bool
            Toggle randomized order of seasons. If True, the order of the seasons will be randomized. Default is False.
        concat : bool
            Toggle concatenating the modalities along the channel dimension. Default is True.
        output_file_name : bool
            Toggle output of the file name.
        shift_s2_channels : bool
            Toggle shifting the S2 channels by 1000 to align to SSL4EO-S12 v1.1. Default is True, where the challenge data S2 channels are 
            shifted upward 1000 to have the range as SSL4EO-S12 v1.1. The background is that ESA decided 
            from 2022-01-25 to shift the DN values of S2 by 1000 upward. SSL4EO-S12 v1.1 includes this shift, 
            while the challenge data does not.

        Returns
        -------
        torch.Tensor or dict
            If output_file_name=False, outputs a torch.Tensor. 
            If output_file_name=True, outputs a dictionary with fields 'data' and 'file_name'. 'data' is a torch.Tensor if concat=True and a dict with one field per modality, each containing a torch.Tensor if False. 'file_name' is the id of the loaded file.
        """

        self.data_path = data_path
        self.transform = transform
        self.modalities = modalities
        self.dataset_name = dataset_name
        assert isinstance(seasons, int) and (1 <= seasons <= 4), "Number of seasons must be integer between 1 and 4."
        
        self.seasons = seasons
        self.randomize_seasons = randomize_seasons
        if not randomize_seasons:
            self.possible_seasons = list(range(seasons))
        else:
            self.possible_seasons = list(range(4))
        assert len(modalities) > 0, "No modalities provided."
        self.concat = concat
        self.output_file_name = output_file_name
        self.shift_s2_channels = shift_s2_channels

        modalities[0] = modalities[0].upper()
        
        print(os.path.join(data_path, modalities[0], '*', '*.zarr.zip'))
        self.samples = glob.glob(os.path.join(data_path, modalities[0], '*', '*.zarr.zip'))
        if self.samples == []:
            modalities[0] = modalities[0].upper()
            print(os.path.join(data_path, modalities[0], '*.zarr.zip'))

            self.samples = glob.glob(os.path.join(data_path, modalities[0], '*.zarr.zip'))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample_path = self.samples[idx]
        file_name = os.path.splitext(os.path.basename(sample_path))[0].replace('.zarr', '')
        if self.randomize_seasons:
            seasons = [self.possible_seasons[ind] for ind in torch.randperm(len(self.possible_seasons)).tolist()[:self.seasons]]
        else:
            seasons = self.possible_seasons
        sample_paths = [sample_path] + [sample_path.replace(self.modalities[0]+'/', modality+'/') for modality in self.modalities[1:]]
        data = {}
        
        for modality, sample_path in zip(self.modalities, sample_paths):
            season_index = xr.DataArray(seasons, dims='time')
            # data[modality] = xr.open_zarr(sample_path).isel(time=season_index)[self.dataset_name].values
            data[modality] = _open_xr_zarr_any(sample_path).isel(time=season_index)[self.dataset_name].values

            # Add shift to align S2 channels with SSL4EO-S12 v1.1
            if self.shift_s2_channels and (modality in ['s2l1c', 's2l2a']):
                data[modality] += 1000

        n_bands_per_modality = {m: d.shape[-3] for m, d in data.items()}
        start_ind_of_modality = {m: n for m, n in zip(self.modalities, [0] + np.cumsum(list(n_bands_per_modality.values())).tolist())}

        # Concatenate data
        data = np.concatenate(list(data.values()), axis=-3)
        data = data.astype(np.float32) 
        data = torch.from_numpy(data)
        
        # Transform
        if self.transform is not None:
            data = self.transform(data)
            
        if not self.concat:
            data = {m: data[..., start_ind_of_modality[m]: start_ind_of_modality[m] + n_bands_per_modality[m], :, :] for m in self.modalities}

        if self.output_file_name:
            
            return {'data': data, 'file_name': file_name}
        else:
            
            return data

def collate_fn(batch):
    if isinstance(batch, dict) or isinstance(batch, torch.Tensor):
        # Single sample
        return batch
    elif isinstance(batch, list) and isinstance(batch[0], torch.Tensor):
        # Concatenate tensors along sample dim
        return torch.concat(batch, dim=0)
    elif isinstance(batch, list) and isinstance(batch[0], dict):
        file_names = [sample['file_name'] for sample in batch]
        data = [sample['data'] for sample in batch]
        if isinstance(data[0], torch.Tensor):
            data = torch.concat(data, dim=0)
        elif isinstance(data[0], dict):
            data = {
                m: torch.concat([b[m] for b in data], dim=0)
                for m in data[0].keys()
            }
        return {'data': data, 'file_name': file_names}

class InputResizer(nn.Module):
    """
    Resizes spatial dimensions of input tensor via adaptive average pooling.
    """

    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adaptive_pool(x)

class Normalize:
    """
    Normalizes image tensor for DINO: scales to [0,1] range by dividing by 10000.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.float() / 10000.0
        return torch.clamp(img, 0.0, 1.0)

class TemporalMean(nn.Module):
    """
    Averages over the time dimension (first dim).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)


import pandas as pd


def create_submission_from_dict(emb_dict):
    """
    Assume dictionary has format:
    {hash-id0: embedding0, hash-id1: embedding1, ...}
    """
    df_submission = pd.DataFrame.from_dict(emb_dict, orient='index')

    # Reset index with name 'id'
    df_submission.index.name = 'id'
    df_submission.reset_index(drop=False, inplace=True)

    return df_submission


def test_submission(
    path_to_submission: str,
    expected_embedding_ids: set,
    embedding_dim: int = 1024
) -> bool:
    # Load data
    df = pd.read_csv(path_to_submission, header=0)

    # Verify that 'id' is in columns
    if 'id' not in df.columns:
        raise ValueError("Submission file must contain column 'id'.")

    # Temporarily set index to 'id'
    df.set_index('id', inplace=True)

    # Check that all samples are included
    submitted_embeddings = set(df.index)
    missing = expected_embedding_ids.difference(submitted_embeddings)
    if missing:
        n_missing = len(missing)
        raise ValueError(f"Submission is missing {n_missing} embeddings.")

    # Check that embeddings have the correct length
    if df.shape[1] != embedding_dim:
        raise ValueError(
            f"{embedding_dim} embedding dimensions expected, "
            f"but provided embeddings have {df.shape[1]} dimensions."
        )

    # Convert columns to float
    try:
        for col in df.columns:
            df[col] = df[col].astype(float)
    except Exception as e:
        raise ValueError(
            "Failed to convert embedding values to float. "
            "Check for invalid characters (e.g., empty strings, letters). "
            f"Original error: {e}"
        )

    # Check for any NaNs
    if df.isna().any().any():
        raise ValueError("Embeddings contain NaN values.")

    return True


# ------------------------
# CLI / main
# ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export NeuCo-Bench embeddings from CIIP")
    p.add_argument("--data-root", type=Path, default="/local/ms-data/SSL4EO-S12-downstream/data")
    p.add_argument("--modalities", nargs="+", default=["s2l1c"], choices=["s1","s2l1c","s2l2a"])
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--csv-out", type=Path, required=True)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)

    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--pre-projection-dim", type=int, default=None)
    p.add_argument("--target-dim", type=int, default=None, help="Validate emitted feature dim, else infer")

    p.add_argument("--resize", type=int, nargs=2, metavar=("H","W"), default=None)
    p.add_argument("--time-reduction", choices=["first","mean","median"], default="mean")
    p.add_argument("--expected-channels", type=int, default=13)

    p.add_argument("--scale", type=float, default=10000.0)
    p.add_argument("--clip", type=float, nargs=2, metavar=("MIN","MAX"), default=(0.0, 1.0))
    p.add_argument("--normalize-mode", choices=["none","percentile","standardize"], default="percentile")
    p.add_argument("--shift-s2-channels", action="store_true", help="Apply +1000 to S2 (align SSL4EO-S12 v1.1)")
    p.add_argument("--no-shift-s2-channels", action="store_true", help="Disable +1000 shift for S2")

    p.add_argument("--seasons", type=int, default=1)
    p.add_argument("--randomize-seasons", action="store_true")

    p.add_argument("--progress", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


SPACE_CHOICES = ("pre", "post")  # pre = encoder features; post = proj+L2
DEFAULT_SPACE = "pre"

def _squeeze_time(x: torch.Tensor) -> torch.Tensor:
    # Expect (B, T, C, H, W) after TemporalMean() -> (B,1,C,H,W) — squeeze T
    if x.ndim == 5 and x.size(1) == 1:
        return x.squeeze(1)
    return x

def _build_resizer(resize_hw: Optional[Tuple[int,int]]) -> nn.Module:
    if resize_hw is None:
        return nn.Identity()
    return InputResizer(output_size=resize_hw)

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger("export_neuco")

    # --------------------------------------------------
    # Basic setup
    # --------------------------------------------------
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    if args.no_shift_s2_channels:
        shift_flag = False
    else:
        shift_flag = True if args.shift_s2_channels else False

    # Which embedding space to export? (env var toggle without changing CLI)
    export_space = os.environ.get("EXPORT_SPACE", DEFAULT_SPACE)
    if export_space not in SPACE_CHOICES:
        logger.warning("Unknown EXPORT_SPACE=%s, defaulting to '%s'", export_space, DEFAULT_SPACE)
        export_space = DEFAULT_SPACE
    logger.info("Exporting '%s'-projection embeddings", export_space)

    # --------------------------------------------------
    # Dataset / dataloader
    # --------------------------------------------------
    data_root = str(args.data_root)
    modalities = args.modalities  # e.g., ["s2l1c"]
    resize_hw = tuple(args.resize) if args.resize is not None else None

    transform = transforms.Compose([
        Normalize(),     # scale to [0,1]
        TemporalMean(),  # average over seasonal timesteps -> (T=1)
    ])

    dataset = E2SChallengeDataset(
        data_path=data_root,
        modalities=modalities,
        seasons=args.seasons,
        dataset_name='bands',
        transform=transform,
        concat=True,
        output_file_name=True,
        shift_s2_channels=shift_flag,
    )
    logger.info("Dataset samples: %d", len(dataset))
    # Optional peek
    # sample0 = dataset[0]['data']; logger.info("Sample tensor shape: %s", tuple(sample0.shape))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )

    # --------------------------------------------------
    # Model / encoder
    # --------------------------------------------------
    
    model = load_ciip_model(args.weights, args.embed_dim, args.pre_projection_dim).to(device).eval()

    if args.weights == 'dino':
        enc = model.encoder_s2
        w = ResNet50_Weights.SENTINEL2_ALL_DINO
        # load pretrained weights
        enc.load_state_dict(w.get_state_dict(progress=True))

    elif args.weights == 'moco':
        enc = model.encoder_s2
        w = ResNet50_Weights.SENTINEL2_ALL_MOCO
        # load pretrained weights
        enc.load_state_dict(w.get_state_dict(progress=True))


    

    encoder = prepare_s2_encoder(model).to(device).eval()  # encoder_s2 with fc disabled -> pre-proj feats
    # Try to grab projection head if present
    proj = getattr(encoder, "proj", None)  # type: ignore[attr-defined]
    if export_space == "post" and not isinstance(proj, nn.Module):
        logger.warning("No encoder.proj found; falling back to 'pre' features.")
        export_space = "pre"

    resizer = _build_resizer(resize_hw).to(device).eval()

    # --------------------------------------------------
    # Extraction loop
    # --------------------------------------------------
    embeddings: Dict[str, np.ndarray] = {}
    pbar = tqdm(loader, desc="Extracting embeddings", disable=not args.progress)

    for batch in pbar:
        data = batch['data']           # (B, T=1, C, H, W) after TemporalMean
        file_names = batch['file_name']  # List[str] length B

        data = _squeeze_time(data)     # -> (B, C, H, W)
        data = data.to(device, non_blocking=True)
        data = resizer(data)           # resize if requested

        with torch.no_grad():
            # pre-projection features from encoder_s2 (fc disabled by prepare_s2_encoder)
            pre = encoder(data)        # expect (B, pre_dim)
            if isinstance(pre, (tuple, list)):
                pre = pre[0]
            if pre.ndim > 2:
                pre = pre.view(pre.size(0), -1)

            if export_space == "pre":
                feats = pre
            else:
                post = proj(pre)             # (B, embed_dim)
                feats = F.normalize(post, dim=1)

        feats = feats.detach().cpu().to(torch.float32).numpy()  # (B, D)
        for fname, vec in zip(file_names, feats):
            # fname is the exact "id" NeuCo-Bench uses
            embeddings[fname] = vec

    # --------------------------------------------------
    # Save CSV + optional validation
    # --------------------------------------------------
    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df = create_submission_from_dict(embeddings)
    # If user provided a target dim, validate
    if args.target_dim is not None:
        _ok = test_submission(
            path_to_submission=submission_df.to_csv(None, index=False),  # dry path not available; skip here
            expected_embedding_ids=set(embeddings.keys()),
            embedding_dim=args.target_dim
        )
        # (The above helper expects a path; since we have a DF, we’ll skip strict test here.)

    submission_df.to_csv(out_path, index=False)
    logger.info("Wrote %d embeddings → %s (dim=%d)", len(submission_df), str(out_path), submission_df.shape[1]-1)


if __name__ == "__main__":
    main()
