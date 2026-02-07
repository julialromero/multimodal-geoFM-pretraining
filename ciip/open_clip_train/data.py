# import ast
# import json
import glob
import logging
import os
from functools import partial
from dataclasses import dataclass
from multiprocessing import Value
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import pandas as pd
import rasterio
import torch
from open_clip import tokenize as clip_tokenize
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Resize
import xarray as xr
from zarr.storage import ZipStore
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as F
### band statistics: mean & std
# calculated from 50k data
### OLD SSL4EO v1.0 STATS
# S1_MEAN = [-12.54847273, -20.19237134]
# S1_STD = [5.25697717, 5.91150917]

# S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]
# S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

# S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]
# S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]
#######


# V1.1
S2L1C_MEAN = [2607.345, 2393.068, 2320.225, 2373.963, 2562.536, 3110.071, 3392.832, 3321.154, 3583.77, 1838.712, 1021.753, 3205.112, 2545.798]
S2L1C_STD = [786.523, 849.702, 875.318, 1143.578, 1126.248, 1161.98, 1273.505, 1246.79, 1342.755, 576.795, 45.626, 1340.347, 1145.036]

S2L2A_MEAN = [1793.243, 1924.863, 2184.553, 2340.936, 2671.402, 3240.082, 3468.412, 3563.244, 3627.704, 3711.071, 3416.714, 2849.625]
S2L2A_STD = [1160.144, 1201.092, 1219.943, 1397.225, 1400.035, 1373.136, 1429.17, 1485.025, 1447.836, 1652.703, 1471.002, 1365.307]

S1GRD_MEAN = [-12.577, -20.265]
S1GRD_STD = [5.179, 5.872]

S2RGB_MEAN = [2340.936, 2184.553, 1924.863]
S2RGB_STD = [1397.225, 1219.943, 1201.092]

####

def _open_zarr_dataset(source):
    try:
        return xr.open_zarr(source, consolidated=True)
    except Exception:
        return xr.open_zarr(source, consolidated=False)


def _canonicalize_band_label(label: object) -> str:
    text = str(label).strip()
    text = text.upper()
    if text.startswith("B"):
        text = text[1:]
    if text.endswith("A"):
        core = text[:-1].lstrip("0")
        return f"{core or '0'}A"
    core = text.lstrip("0")
    return core or "0"


def _transpose_spatial_first(da: xr.DataArray, band_dim: Optional[str]) -> xr.DataArray:
    dims = list(da.dims)
    ordered: List[str] = []
    if band_dim and band_dim in dims:
        ordered.append(band_dim)
    spatial_candidates = [
        dim
        for dim in ("y", "x", "latitude", "longitude", "lat", "lon", "row", "col")
        if dim in dims and dim not in ordered
    ]
    for dim in dims:
        if dim not in ordered and dim not in spatial_candidates:
            ordered.append(dim)
    ordered.extend([d for d in spatial_candidates if d not in ordered])
    if ordered and ordered != dims:
        da = da.transpose(*ordered)
    return da

DEFAULT_S2_BANDS = ["1", "2", "3", "4", "5", "6", "7", "8", "8A", "9", "11", "12"]
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset


from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

class SSL4EODataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        seasons: list[int] = [0, 1, 2, 3],
        transforms: Optional[Any] = None,
        num_samples_per_file: int = 64,
        num_timestamps: int = 4,
        s2_tier: str = "S2L2A",
        is_train: bool = True,
        temporal_agg: str | None = None,
        return_all_timestamps: bool = False,
    ):
        self.root = Path(root)
        self.is_train = is_train
        self.seasons = seasons
        self.transforms = transforms
        self.temporal_agg = temporal_agg
        # New behavior: optionally return all timestamps so downstream can run the
        # model per-season and average embeddings. Keep backward compatibility by
        # honoring temporal_agg="mean" as a request to return all timestamps
        # without averaging the raw images.
        if self.temporal_agg is not None:
            assert self.temporal_agg in {"mean"}, f"Unsupported temporal_agg: {self.temporal_agg}"
            return_all_timestamps = True if self.temporal_agg == "mean" else return_all_timestamps
        self.return_all_timestamps = return_all_timestamps
        if transforms is not None:
            self.s1_transforms = transforms["s1"]
            self.s2_transforms = transforms["s2"]

        self.num_samples_per_file = num_samples_per_file
        self.num_timestamps = num_timestamps

        assert len(self.seasons) == self.num_timestamps, (
            f"len(seasons)={len(self.seasons)} must equal num_timestamps={self.num_timestamps}"
        )

        self.is_rgb = False
        print(f'S2 tier: {s2_tier}, is_rgb: {self.is_rgb}')
        self.s1_dir = self.root / "S1GRD"
        if s2_tier in ["s2l2a", "S2L2A"]:
            self.s2_dir = self.root / "S2L2A"
        elif s2_tier in ["s2l1c", "S2L1C"]:
            self.s2_dir = self.root / "S2L1C"
        elif s2_tier in ['rgb', 'RGB']:
            self.s2_dir = self.root / "S2L2A"
            self.is_rgb = True
        else:
            raise ValueError(f"Unsupported s2_tier: {s2_tier}")
        
        print(f'S2 tier: {s2_tier}, is_rgb: {self.is_rgb}')
                

        print("S1 dir:", self.s1_dir)
        print("S2 dir:", self.s2_dir)

        assert self.s1_dir.exists(), f"Missing directory: {self.s1_dir}"
        assert self.s2_dir.exists(), f"Missing directory: {self.s2_dir}"

        # Shared filenames between S1 and S2
        self.file_names = sorted(p.name for p in self.s2_dir.glob("*.zarr.zip"))
        self.num_files = len(self.file_names)
        assert self.num_files > 0, f"No zarr files found in {self.s2_dir}"

        # Index by file (location), NOT by sample
        self._len = self.num_files
        print("Number of files (dataset length):", self._len)

    def __len__(self) -> int:
        return self._len

    # -----------------------------------------------------
    def _open_zarr_pair(self, file_idx: int):
        """Open matching S1 and S2 zarr datasets for the given file index."""
        fname = self.file_names[file_idx]
        s1_path = self.s1_dir / fname
        s2_path = self.s2_dir / fname

        assert s1_path.exists(), f"Missing file: {s1_path}"
        assert s2_path.exists(), f"Missing file: {s2_path}"

        ds_s1 = xr.open_zarr(s1_path)
        ds_s2 = xr.open_zarr(s2_path)
        return ds_s1, ds_s2

    # -----------------------------------------------------
    def _apply_transforms(
        self,
        s1: torch.Tensor,  # (P, 2, H, W)
        s2: torch.Tensor,  # (P, 12, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply joint spatial transforms + per-modality transforms.

        s1, s2: (P, C, H, W) where P = num_samples_per_file (64).
        We apply the SAME crop/flip to all 64 samples and both modalities,
        then apply s1_transforms/s2_transforms per sample.
        """
        if self.transforms is None:
            return s1, s2

        P, C1, H, W = s1.shape
        _, C2, H2, W2 = s2.shape
        assert C1 == 2, f"Expected 2 S1 channels, got {C1}"
        assert C2 == 12 or C2 == 13 or C2 == 3, f"Expected 12 S2 channels, got {C2}"
        assert (H, W) == (H2, W2)

        s1_tensor = s1
        s2_tensor = s2

        # print descr stats of s2 before transforms
        # print(f'S2 before transforms: mean={s2_tensor.mean().item():.4f}, std={s2_tensor.std().item():.4f}, min={s2_tensor.min().item():.4f}, max={s2_tensor.max().item():.4f}')
    

        if self.is_train:
            # Use first sample to get crop params, then apply to all P
            i, j, h, w = T.RandomCrop.get_params(s1_tensor[0], output_size=(224, 224))
            s1_tensor = torch.stack(
                [F.crop(s1_tensor[p], i, j, h, w) for p in range(P)],
                dim=0,
            )
            s2_tensor = torch.stack(
                [F.crop(s2_tensor[p], i, j, h, w) for p in range(P)],
                dim=0,
            )

            # Joint horizontal flip
            if random.random() < 0.5:
                s1_tensor = torch.flip(s1_tensor, dims=[3])  # flip W
                s2_tensor = torch.flip(s2_tensor, dims=[3])

            # Joint vertical flip
            if random.random() < 0.5:
                s1_tensor = torch.flip(s1_tensor, dims=[2])  # flip H
                s2_tensor = torch.flip(s2_tensor, dims=[2])

        # Apply modality-specific transforms per patch
        s1_out = torch.stack(
            [self.s1_transforms(s1_tensor[p]) for p in range(P)],
            dim=0,
        )
        # print shapes
        # print(f'S2 shape before s2 transforms: {s2_tensor.shape}')
        # print transforms
        # print(f'S2 transforms: {self.s2_transforms}')
        s2_out = torch.stack(
            [self.s2_transforms(s2_tensor[p]) for p in range(P)],
            dim=0,
        )

        # print out descriptive statistics of s2
        # print(f'S2 after transforms: mean={s2_out.mean().item():.4f}, std={s2_out.std().item():.4f}, min={s2_out.min().item():.4f}, max={s2_out.max().item():.4f}')

        return s1_out, s2_out

    # -----------------------------------------------------
    def __getitem__(self, file_idx: int) -> Dict[str, Any]:
        """Index by file: return all 64 samples for one random season or all seasons."""
        # Random time index (shared across modalities for this file)
        time_idx = np.random.randint(0, self.num_timestamps)
        season_name = self.seasons[time_idx]

        ds_s1, ds_s2 = self._open_zarr_pair(file_idx)

        bands_s1 = ds_s1["bands"]
        bands_s2 = ds_s2["bands"]

        # Expect (sample=64, time=4, band=C, y, x)
        P, T, C_s1, H, W = bands_s1.shape
        P2, T2, C_s2, H2, W2 = bands_s2.shape

        assert P == self.num_samples_per_file, f"Expected {self.num_samples_per_file} samples/file, got {P}"
        assert P2 == P and T2 == T and (H2, W2) == (H, W)
        assert C_s1 == 2, f"Unexpected S1 channel count: {C_s1}"
        assert C_s2 == 12 or C_s2 == 13 or C_s2 == 3, f"Unexpected S2 channel count: {C_s2}"

        if self.return_all_timestamps:
            # Return all seasons so the caller can feed them individually and
            # average embeddings. No averaging happens here.
            arr_s1 = bands_s1.values
            arr_s2 = bands_s2.values
            s1_list = []
            s2_list = []
            for t in range(T):
                s1_np = arr_s1[:, t]  # (P, C1, H, W)
                s2_np = arr_s2[:, t]  # (P, C2, H, W)

                s1_t = torch.from_numpy(s1_np.astype("float32"))
                s2_t = torch.from_numpy(s2_np.astype("float32"))

                if self.is_rgb:
                    if s2_t.ndim == 3 and s2_t.shape[0] >= 4:
                        s2_t = s2_t[[3, 2, 1], ...]
                    elif s2_t.ndim == 4 and s2_t.shape[1] >= 4:
                        s2_t = s2_t[:, [3, 2, 1], ...]

                s1_t, s2_t = self._apply_transforms(s1_t, s2_t)
                s1_list.append(s1_t)
                s2_list.append(s2_t)

            # (P, T, C, H, W)
            s1 = torch.stack(s1_list, dim=1)
            s2 = torch.stack(s2_list, dim=1)
        else:
            s1_np = bands_s1.isel(time=time_idx).values  # (P, C1, H, W)
            s2_np = bands_s2.isel(time=time_idx).values  # (P, C2, H, W)

            s1 = torch.from_numpy(s1_np.astype("float32"))
            s2 = torch.from_numpy(s2_np.astype("float32"))

            if self.is_rgb:
                if s2.ndim == 3 and s2.shape[0] >= 4:
                    s2 = s2[[3, 2, 1], ...]
                elif s2.ndim == 4 and s2.shape[1] >= 4:
                    s2 = s2[:, [3, 2, 1], ...]

            s1, s2 = self._apply_transforms(s1, s2)

        ds_s1.close()
        ds_s2.close()

        return {
            "s1": s1,  # (P, 2, 224, 224) or (P, T, 2, 224, 224)
            "s2": s2,  # (P, 12, 224, 224) or (P, T, 12/3, 224, 224)
            # optional metadata if you want:
            # "file_idx": file_idx,
            # "time_idx": time_idx,
            # "season": season_name,
        }


class SSL4EOTextDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        seasons: list[int] = [0, 1, 2, 3],
        transforms: Optional[Any] = None,
        num_samples_per_file: int = 64,
        num_timestamps: int = 4,
        s2_tier: str = "S2L2A",
        caption_column: str = "caption",
        tokenizer: Optional[Callable[[list[str]], torch.Tensor]] = None,
        is_train: bool = True,
    ):
        self.root = Path(root)
        self.is_train = is_train
        self.seasons = seasons
        self.transforms = transforms
        self.caption_column = caption_column
        self.tokenize = tokenizer or partial(clip_tokenize, context_length=77)
        if transforms is not None:
            self.s2_transforms = transforms.get("s2")
        else:
            self.s2_transforms = None

        self.num_samples_per_file = num_samples_per_file
        self.num_timestamps = num_timestamps

        assert len(self.seasons) == self.num_timestamps, (
            f"len(seasons)={len(self.seasons)} must equal num_timestamps={self.num_timestamps}"
        )

        self.is_rgb = False
        self.captions_dir = self.root / "captions"
        if s2_tier in ["s2l2a", "S2L2A", "s2a"]:
            self.s2_dir = self.root / "S2L2A"
        elif s2_tier in ["s2l1c", "S2L1C", "s2c"]:
            self.s2_dir = self.root / "S2L1C"
        elif s2_tier in ["rgb", "RGB"]:
            self.s2_dir = self.root / "S2L2A"
            self.is_rgb = True
        else:
            raise ValueError(f"Unsupported s2_tier: {s2_tier}")

        assert self.s2_dir.exists(), f"Missing directory: {self.s2_dir}"
        assert self.captions_dir.exists(), f"Missing directory: {self.captions_dir}"

        self.file_basenames: list[str] = []
        for path in sorted(self.s2_dir.glob("*.zarr.zip")):
            base = path.name.split(".zarr.zip")[0]
            if (self.captions_dir / f"{base}.csv").exists():
                self.file_basenames.append(base)
        self.num_files = len(self.file_basenames)
        assert self.num_files > 0, f"No paired zarr/caption files found in {self.s2_dir}"

        self._len = self.num_files

    def __len__(self) -> int:
        return self._len

    def _open_zarr_and_captions(self, base_name: str):
        s2_path = self.s2_dir / f"{base_name}.zarr.zip"
        caption_path = self.captions_dir / f"{base_name}.csv"
        assert s2_path.exists(), f"Missing file: {s2_path}"
        assert caption_path.exists(), f"Missing file: {caption_path}"
        ds_s2 = xr.open_zarr(s2_path)
        captions_df = pd.read_csv(caption_path, index_col=0)
        if self.caption_column not in captions_df.columns and len(captions_df.columns) > 0:
            captions_df = captions_df.rename(columns={captions_df.columns[0]: self.caption_column})
        return ds_s2, captions_df

    def _apply_s2_only_transforms(self, s2: torch.Tensor) -> torch.Tensor:
        if self.transforms is None or self.s2_transforms is None:
            return s2

        P, C, H, W = s2.shape
        assert C in (3, 12, 13), f"Unexpected S2 channel count: {C}"

        s2_tensor = s2
        if self.is_train:
            i, j, h, w = T.RandomCrop.get_params(s2_tensor[0], output_size=(224, 224))
            s2_tensor = torch.stack([F.crop(s2_tensor[p], i, j, h, w) for p in range(P)], dim=0)
            if random.random() < 0.5:
                s2_tensor = torch.flip(s2_tensor, dims=[3])
            if random.random() < 0.5:
                s2_tensor = torch.flip(s2_tensor, dims=[2])

        s2_out = torch.stack([self.s2_transforms(s2_tensor[p]) for p in range(P)], dim=0)
        return s2_out

    def __getitem__(self, file_idx: int) -> Dict[str, Any]:
        time_idx = np.random.randint(0, self.num_timestamps)
        base_name = self.file_basenames[file_idx]

        ds_s2, captions_df = self._open_zarr_and_captions(base_name)

        arr_s2 = ds_s2["bands"].values
        P, T, C_s2, H, W = arr_s2.shape
        assert P == self.num_samples_per_file, f"Expected {self.num_samples_per_file} samples/file, got {P}"
        assert T == self.num_timestamps, f"Unexpected number of timestamps: {T}"

        captions_raw = captions_df[self.caption_column].to_numpy()
        try:
            captions = captions_raw.reshape(self.num_samples_per_file, self.num_timestamps)
        except ValueError:
            captions = captions_raw.reshape(self.num_samples_per_file, -1)
        captions_for_time = captions[:, time_idx].tolist()

        s2_np = arr_s2[:, time_idx]
        s2 = torch.from_numpy(s2_np.astype("float32"))

        if self.is_rgb:
            if s2.ndim == 4 and s2.shape[1] >= 4:
                s2 = s2[:, [3, 2, 1], ...]

        s2 = self._apply_s2_only_transforms(s2)
        text_tokens = self.tokenize(captions_for_time)
        if isinstance(text_tokens, list):
            text_tokens = torch.stack(text_tokens)

        return {
            "s2": s2,
            "text": text_tokens,
        }


class SSL4EOAlignedTrioDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        seasons: list[int] = [0, 1, 2, 3],
        transforms: Optional[Any] = None,
        num_samples_per_file: int = 64,
        num_timestamps: int = 4,
        s2_tier: str = "S2L2A",
        caption_column: str = "caption",
        tokenizer: Optional[Callable[[list[str]], torch.Tensor]] = None,
        is_train: bool = True,
    ):
        self.root = Path(root)
        self.is_train = is_train
        self.seasons = seasons
        self.transforms = transforms
        self.caption_column = caption_column
        self.tokenize = tokenizer or partial(clip_tokenize, context_length=77)
        if transforms is not None:
            self.s1_transforms = transforms.get("s1")
            self.s2_transforms = transforms.get("s2")
        else:
            self.s1_transforms = None
            self.s2_transforms = None

        self.num_samples_per_file = num_samples_per_file
        self.num_timestamps = num_timestamps

        assert len(self.seasons) == self.num_timestamps, (
            f"len(seasons)={len(self.seasons)} must equal num_timestamps={self.num_timestamps}"
        )

        self.is_rgb = False
        self.s1_dir = self.root / "S1GRD"
        self.captions_dir = self.root / "captions"
        if s2_tier in ["s2l2a", "S2L2A", "s2a"]:
            self.s2_dir = self.root / "S2L2A"
        elif s2_tier in ["s2l1c", "S2L1C", "s2c"]:
            self.s2_dir = self.root / "S2L1C"
        elif s2_tier in ["rgb", "RGB"]:
            self.s2_dir = self.root / "S2L2A"
            self.is_rgb = True
        else:
            raise ValueError(f"Unsupported s2_tier: {s2_tier}")

        assert self.s1_dir.exists(), f"Missing directory: {self.s1_dir}"
        assert self.s2_dir.exists(), f"Missing directory: {self.s2_dir}"
        assert self.captions_dir.exists(), f"Missing directory: {self.captions_dir}"

        self.file_basenames: list[str] = []
        for path in sorted(self.s2_dir.glob("*.zarr.zip")):
            base = path.name.split(".zarr.zip")[0]
            if (self.captions_dir / f"{base}.csv").exists() and (self.s1_dir / f"{base}.zarr.zip").exists():
                self.file_basenames.append(base)
        self.num_files = len(self.file_basenames)
        assert self.num_files > 0, f"No paired zarr/caption files found in {self.s2_dir}"

        self._len = self.num_files

    def __len__(self) -> int:
        return self._len

    def _open_triplet(self, base_name: str):
        s1_path = self.s1_dir / f"{base_name}.zarr.zip"
        s2_path = self.s2_dir / f"{base_name}.zarr.zip"
        caption_path = self.captions_dir / f"{base_name}.csv"
        assert s1_path.exists(), f"Missing file: {s1_path}"
        assert s2_path.exists(), f"Missing file: {s2_path}"
        assert caption_path.exists(), f"Missing file: {caption_path}"
        ds_s1 = xr.open_zarr(s1_path)
        ds_s2 = xr.open_zarr(s2_path)
        captions_df = pd.read_csv(caption_path, index_col=0)
        if self.caption_column not in captions_df.columns and len(captions_df.columns) > 0:
            captions_df = captions_df.rename(columns={captions_df.columns[0]: self.caption_column})
        return ds_s1, ds_s2, captions_df

    def _apply_transforms(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transforms is None or self.s1_transforms is None or self.s2_transforms is None:
            return s1, s2

        P, C1, H, W = s1.shape
        _, C2, H2, W2 = s2.shape
        assert C1 == 2, f"Unexpected S1 channel count: {C1}"
        assert C2 in (3, 12, 13), f"Unexpected S2 channel count: {C2}"
        assert (H, W) == (H2, W2)

        s1_tensor = s1
        s2_tensor = s2
        if self.is_train:
            i, j, h, w = T.RandomCrop.get_params(s1_tensor[0], output_size=(224, 224))
            s1_tensor = torch.stack([F.crop(s1_tensor[p], i, j, h, w) for p in range(P)], dim=0)
            s2_tensor = torch.stack([F.crop(s2_tensor[p], i, j, h, w) for p in range(P)], dim=0)
            if random.random() < 0.5:
                s1_tensor = torch.flip(s1_tensor, dims=[3])
                s2_tensor = torch.flip(s2_tensor, dims=[3])
            if random.random() < 0.5:
                s1_tensor = torch.flip(s1_tensor, dims=[2])
                s2_tensor = torch.flip(s2_tensor, dims=[2])

        s1_out = torch.stack([self.s1_transforms(s1_tensor[p]) for p in range(P)], dim=0)
        s2_out = torch.stack([self.s2_transforms(s2_tensor[p]) for p in range(P)], dim=0)
        return s1_out, s2_out

    def __getitem__(self, file_idx: int) -> Dict[str, Any]:
        time_idx = np.random.randint(0, self.num_timestamps)
        base_name = self.file_basenames[file_idx]

        ds_s1, ds_s2, captions_df = self._open_triplet(base_name)

        arr_s1 = ds_s1["bands"].values
        arr_s2 = ds_s2["bands"].values
        P, T, C_s1, H, W = arr_s1.shape
        P2, T2, C_s2, H2, W2 = arr_s2.shape
        assert P == self.num_samples_per_file, f"Expected {self.num_samples_per_file} samples/file, got {P}"
        assert P2 == P and T2 == T and (H2, W2) == (H, W)
        assert C_s1 == 2, f"Unexpected S1 channel count: {C_s1}"
        assert C_s2 in (3, 12, 13), f"Unexpected S2 channel count: {C_s2}"

        captions_raw = captions_df[self.caption_column].to_numpy()
        try:
            captions = captions_raw.reshape(self.num_samples_per_file, self.num_timestamps)
        except ValueError:
            captions = captions_raw.reshape(self.num_samples_per_file, -1)
        captions_for_time = captions[:, time_idx].tolist()

        s1_np = arr_s1[:, time_idx]
        s2_np = arr_s2[:, time_idx]

        s1 = torch.from_numpy(s1_np.astype("float32"))
        s2 = torch.from_numpy(s2_np.astype("float32"))

        if self.is_rgb and s2.ndim == 4 and s2.shape[1] >= 4:
            s2 = s2[:, [3, 2, 1], ...]

        s1, s2 = self._apply_transforms(s1, s2)

        text_tokens = self.tokenize(captions_for_time)
        if isinstance(text_tokens, list):
            text_tokens = torch.stack(text_tokens)

        return {
            "s1": s1,
            "s2": s2,
            "text": text_tokens,
        }


from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    # batch is a list of dicts, e.g., {"s1": (64,2,H,W), "s2": (64,12,H,W)} or {"s2": (64,12,H,W), "text": (64, L)}
    if isinstance(batch, list) and isinstance(batch[0], dict):
        return {
            k: torch.concat([b[k] for b in batch], dim=0)
            for k in batch[0].keys()
        }


# ssl4eo
class SSL4EODatasetOld(Dataset):
    def __init__(self, root, s2_tier, s2_bands, transforms=None, target_image_dimension=(264, 264)):
        self.root = root
        self.num_locations = None
        self.length = None
        self.s1_paths = []
        self.s2_paths = []
        self.s2_bands = s2_bands
        # https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html
        self.resize_transform = Resize(target_image_dimension)
        # self.pair_aug = CIIPPairAug(
        #     out_size=target_image_dimension,
        #     cutout_p=0.25,
        #     normalize_s1=self.normalize_s1,
        #     normalize_s2=self.normalize_s2,
        # )


        if 0 in self.s2_bands:
            raise ValueError('Band index should be between 1 and 12')
        self.transforms = transforms
        self.s2_tier = s2_tier

        if self.s2_tier == "s2a":
            stat_bands = ["1", "2", "3", "4", "5", "6", "7", "8", "8A", "9", "11", "12"]
            self.s2_stats = {b: (m, s) for b, m, s in zip(stat_bands, S2L2A_MEAN, S2L2A_STD)}
        else:
            stat_bands = ["1", "2", "3", "4", "5", "6", "7", "8", "8A", "9", "10", "11", "12"]
            self.s2_stats = {b: (m, s) for b, m, s in zip(stat_bands, S2L1C_MEAN, S2L1C_STD)}

        original_working_directory = hydra.utils.get_original_cwd()
        data_parent_directory = "/".join(original_working_directory.split("/")[:-2])

        self.s1_dir = os.path.join(data_parent_directory, os.path.join(self.root, 's1'))
        self.s2_dir = os.path.join(data_parent_directory, os.path.join(self.root, self.s2_tier))

        s1_samples = os.listdir(self.s1_dir)
        s2_samples = os.listdir(self.s2_dir)

        assert len(s1_samples) == len(s2_samples), 'Number of locations in S1 and S2 should be the same, but got {} and {}.'.format(len(s1_samples), len(s2_samples))

        self.num_locations = len(s1_samples)
        self.length = self.num_locations * 4

        # Limit 1000 samples for test
        # self.max_samples = min(self.length, 2001)

        # print(f'Number of samples in dataset: {self.max_samples}')
        print(f'Number of locations in dataset: {self.num_locations}')

        # location list
        self.locations = s1_samples

        logging.debug('Done loading data.')


    def __len__(self):
        return self.length
        # Adjust for testing
        # return self.max_samples


    def __getitem__(self, idx):
        ### get the sample corresponding to idx
        location_idx, season_idx = self.int_to_filepath(idx)

        location_folder = self.locations[location_idx]

        path_to_s1 = os.path.join(self.s1_dir, location_folder)
        path_to_s2 = os.path.join(self.s2_dir, location_folder)

        s1_season_folders = sorted(os.listdir(path_to_s1))
        s2_season_folders = sorted(os.listdir(path_to_s2))

        path_to_s1_season = os.path.join(path_to_s1, s1_season_folders[season_idx])
        path_to_s2_season = os.path.join(path_to_s2, s2_season_folders[season_idx])

        # ############ Load and stack s1 images ##############
        vh_path = os.path.join(path_to_s1_season, 'VH.tif')
        vv_path = os.path.join(path_to_s1_season, 'VV.tif')


        vh_image, _ = self.read_raster_image(vh_path)
        vv_image, _ = self.read_raster_image(vv_path)

        # # Normalize the VH and VV bands
        # # vh_image = self.normalize_image(vh_image)
        # # vv_image = self.normalize_image(vv_image)

        # vv_image = torch.from_numpy(self.normalize(vv_image, S1_MEAN[0], S1_STD[0])).unsqueeze(0)
        # vh_image = torch.from_numpy(self.normalize(vh_image, S1_MEAN[1], S1_STD[1])).unsqueeze(0)

        # vh_image = self.resize_transform(vh_image).squeeze(0)
        # vv_image = self.resize_transform(vv_image).squeeze(0)

        # s1_composite_image = torch.stack((vh_image, vv_image), dim=0)
        

        # ############### Load s2 images ###################
        # s2_band_images = []
        # for band in self.s2_bands:
        #     band_path = os.path.join(path_to_s2_season, f'B{band}.tif')
        #     band_img, _ = self.read_raster_image(band_path)
        #     if band not in self.s2_stats:
        #         raise ValueError(f"Statistics for band {band} not found")
        #     mean, std = self.s2_stats[band]
        #     band_img = torch.from_numpy(self.normalize(band_img, mean, std)).unsqueeze(0)
        #     band_img = self.resize_transform(band_img).squeeze(0)
        #     s2_band_images.append(band_img)

        # s2_composite_image = torch.stack(s2_band_images, dim=0)

        vh = torch.from_numpy(vh_image).float().unsqueeze(0)
        vv = torch.from_numpy(vv_image).float().unsqueeze(0)

        # Geometry first (resize); no normalization yet
        vh = self.resize_transform(vh).squeeze(0)
        vv = self.resize_transform(vv).squeeze(0)
        s1_composite_image = torch.stack([vh, vv], dim=0)   # [2,H,W]

        s2_bands_t = []
        for band in self.s2_bands:
            arr, _ = self.read_raster_image(os.path.join(path_to_s2_season, f'B{band}.tif'))
            t = torch.from_numpy(arr).float().unsqueeze(0)
            t = self.resize_transform(t).squeeze(0)   # geometry first
            s2_bands_t.append(t)
        s2_composite_image = torch.stack(s2_bands_t, dim=0)

        # print(f'S1 composite image shape: {s1_composite_image.shape}')
        # print(f'S2 composite image shape: {s2_composite_image.shape}')


        ## TODO: double check Image input to transforms ?
        if self.transforms is not None:
            s1_composite_image = self.transforms(s1_composite_image)
            s2_composite_image = self.transforms(s2_composite_image)

        s1_composite_image = self.normalize_s1(s1_composite_image)
        s2_composite_image = self.normalize_s2(s2_composite_image, self.s2_stats, self.s2_bands)

        return (s1_composite_image, s2_composite_image)


  
    def normalize_s1(self, x):  # x: [2,H,W] in your chosen S1 domain
        mean = torch.tensor(S1GRD_MEAN, dtype=x.dtype, device=x.device)[:, None, None]
        std  = torch.tensor(S1GRD_STD,  dtype=x.dtype, device=x.device)[:, None, None]
        return (x - mean) / std

    def normalize_s2(self, x , s2_stats, s2_bands):
        means, stds = [], []
        for b in s2_bands:
            m, s = s2_stats[str(b)]     # <-- string key
            means.append(m); stds.append(s)
        mean = torch.tensor(means, dtype=x.dtype, device=x.device)[:, None, None]
        std  = torch.tensor(stds,  dtype=x.dtype, device=x.device)[:, None, None]
        return (x - mean) / std


    def int_to_filepath(self, i):
        if i < 0 or i >= self.length:
            raise ValueError("Integer i must be in the range [0, n)")
        
        location_idx = i // 4
        season_idx = i % 4
        
        return location_idx, season_idx

    def read_raster_image(self, image_path):
        with rasterio.open(image_path) as src:
            return src.read(1), src.profile

    #### our custom normalization function for image-wise norm
    # def normalize_image(self, image):
    #     """Normalize image data to the range [0, 1]"""
    #     image_min, image_max = np.min(image), np.max(image)
    #     return (image - image_min) / (image_max - image_min)
    

    #### band-wise normalization based on mean and std calculate from 50k data
    ## taken from: https://github.com/zhu-xlab/SSL4EO-S12/blob/2156913c5d8e5a2c572a5b000f0d5eaed6fc3192/src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py#L36
    # normalize: standardize + percentile
    def normalize(self, img, mean, std):
        """Map band values using dataset mean/std so that μ±2σ → [0, 1]."""
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value)
        img = np.clip(img, 0, 1).astype(np.float32)
        return img
    

    def get_sample_uid(self, idx):
        location_idx, season_idx = self.int_to_filepath(idx)
        location_folder = self.locations[location_idx]

        filepath = os.path.join(self.root, 's1', location_folder)

        unique_id = f'{location_folder}_season{season_idx}'
        return (unique_id, filepath)
 


# taken from: https://github.com/zhu-xlab/SSL4EO-S12/blob/2156913c5d8e5a2c572a5b000f0d5eaed6fc3192/src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py#L127
class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def generate_splits(dataset, val_frac, seed=None):
    rng = np.random.default_rng(seed)
    val_indices = rng.choice(range(len(dataset)), int(val_frac * len(dataset)))

    # all other indices are for training
    all_indices = np.arange(len(dataset))
    train_indices = np.setdiff1d(all_indices, val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def plot_pixel_value_distributions(dataset, band_names=['0', '1'], modality='s1', title='Pixel Value Distribution'):
    import matplotlib.pyplot as plt
    print(f'Plotting pixel value distributions for {modality}...')

    # Collect pixel values for each band
    band_values = {band: [] for band in band_names}

    # print length of dataset
    print(f'Number of samples in dataset: {len(dataset)}')

    if modality == 's1':
        for s1, s2 in dataset:
            # s2: (P, C, H, W)
            s1_np = s1.numpy()
            for i, band in enumerate(band_names):
                band_data = s1_np[:, i, :, :].flatten()
                band_values[band].extend(band_data.tolist())
    else:
        for s1, s2 in dataset:
            # s2: (P, C, H, W)
            s2_np = s2.numpy()
            for i, band in enumerate(band_names):
                band_data = s2_np[:, i, :, :].flatten()
                band_values[band].extend(band_data.tolist())

    # Plot histograms
    print(f'Generating histograms for {modality}...')
    num_bands = len(band_names)
    cols = 4
    rows = (num_bands + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, band in enumerate(band_names):
        axes[i].hist(band_values[band], bins=50, color='blue', alpha=0.7)
        axes[i].set_title(f'Band {band} Pixels |  {title}')
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    # plt.title(title) 

    plt.tight_layout()
    # save plot to output dir
    plt.savefig(f'pixel_value_distribution_{modality}.png', bbox_inches='tight')
    print(f'Saved pixel value distribution plot for {modality} to pixel_value_distribution_{modality}.png')
    plt.close()
    

def get_ssl4eo_dataset(args, is_train, transforms, tokenizer=None):
    root = args.dataset.root
    # root = args.dataset.train_data if is_train else args.val_data
    assert root
    if args.dataset.s2_tier == "s2a":
        default_bands = ["1", "2", "3", "4", "5", "6", "7", "8", "8A", "9", "11", "12"]
    else:
        default_bands = ["1", "2", "3", "4", "5", "6", "7", "8", "8A", "9", "10", "11", "12"]

    # dataset = SSL4EODataset(
    #     root, # root file path
    #     s2_tier="S2L2A",
    #     seasons=[0,1,2,3],
    #     transforms=None,  # transforms
    #     is_train=True
    # )
    # # sample 2000 samples to plot pixel distribution
    # data_sample = Subset(dataset, np.random.choice(range(len(dataset)), size=10, replace=False))


    # # plot histogram of pixel values
    # # plot_pixel_value_distributions(data_sample, band_names=['0','1'], modality='s1', title='s1 without transforms')
    # plot_pixel_value_distributions(data_sample, band_names=default_bands, modality='s2', title='s2 without transforms')

    model_cfg = getattr(args, "model", None)
    encoder_pair = getattr(model_cfg, "encoder_pair", "s1s2")
    if encoder_pair == "s2_text":
        dataset = SSL4EOTextDataset(
            root,
            seasons=[0, 1, 2, 3],
            transforms=transforms,
            s2_tier=args.dataset.s2_tier,
            tokenizer=tokenizer,
            is_train=is_train,
        )
    elif encoder_pair == "s1s2_text":
        dataset = SSL4EOAlignedTrioDataset(
            root,
            seasons=[0, 1, 2, 3],
            transforms=transforms,
            s2_tier=args.dataset.s2_tier,
            tokenizer=tokenizer,
            is_train=is_train,
        )
    else:
        dataset = SSL4EODataset(
            root,
            seasons=[0,1,2,3],
            transforms=transforms,  
            s2_tier = args.dataset.s2_tier,
            is_train=is_train
        )

    # # sample 2000 samples to plot pixel distribution
    # data_sample = Subset(dataset, np.random.choice(range(len(dataset)), size=10, replace=False))
    # # plot histogram of pixel values
    # # plot_pixel_value_distributions(data_sample, band_names=['0','1'], modality='s1', title='s1 with transforms')
    # plot_pixel_value_distributions(data_sample, band_names=default_bands, modality='s2', title='s2 with transforms')


    return dataset_to_datainfo(args, dataset, is_train)



class Divideby10000Normalize:
    """
    Normalizes image tensor for DINO: scales to [0,1] range by dividing by 10000.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.float() / 10000.0
        return torch.clamp(img, 0.0, 1.0)
    
def dataset_to_datainfo(args, dataset, is_train):
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset, drop_last=True) if args.datamodule.distributed and is_train else None
    shuffle = is_train and sampler is None
    num_workers = getattr(getattr(args, "model", None), "workers", 0)

    dataloader = DataLoader(
        dataset,
        batch_size=args.datamodule.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        # drop_last=is_train,
        drop_last=True,
        collate_fn=collate_fn,
        
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    logging.info("Dataloader samples: %s, batches: %s", num_samples, dataloader.num_batches)


    return DataInfo(dataloader, sampler)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)




def get_dataset_fn(data_path, dataset_type):
    if dataset_type == 'ssl4eo':
        return get_ssl4eo_dataset
    # elif dataset_type == "webdataset":
    #     return get_wds_dataset
    # elif dataset_type == "csv":
    #     return get_csv_dataset
    # elif dataset_type == "synthetic":
    #     return get_synthetic_dataset
    # elif dataset_type == "auto":
    #     ext = data_path.split('.')[-1]
    #     if ext in ['csv', 'tsv']:
    #         return get_csv_dataset
    #     elif ext in ['tar']:
    #         return get_wds_dataset
    #     else:
    #         raise ValueError(
    #             f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    


import random
class BandwiseJitter(torch.nn.Module):
    def __init__(self, sigma=0.02, kind="multiplicative", p=0.8):
        super().__init__()
        assert kind in ("multiplicative", "additive")
        self.sigma = sigma
        self.kind = kind
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        C, H, W = x.shape
        # jitter per band, handle the batch dimension -> each file contains 64 images 
        eps = torch.randn( C, 1, 1, device=x.device, dtype=x.dtype) * self.sigma
        # eps = torch.randn(C, 1, 1, device=x.device, dtype=x.dtype) * self.sigma
        if self.kind == "multiplicative":
            return x * (1.0 + eps)
        else:
            return x + eps
        
class Clamp_S1:
    """
    Normalizes image tensor for DINO: scales to [0,1] range by dividing by 10000.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = torch.clamp(img, -25.0, 0.0)
        return img + 25.0 / 25.0
    


def get_transform(modality, is_train):
    if modality == "s1":
        if is_train:
            return transforms.Compose([
                # transforms.Normalize(mean=S1GRD_MEAN, std=S1GRD_STD),
                Clamp_S1()
            ])
        else:
            # raise NotImplementedError("S1 validation transforms - check norm method.")
            return transforms.Compose([
                transforms.CenterCrop(224),
                Clamp_S1()
                # transforms.Normalize(mean=S1GRD_MEAN, std=S1GRD_STD),
            ])
    elif modality.lower() == "s2a" or modality.lower() == "s2l2a":
        if is_train:
            return transforms.Compose([
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)],
                    p=0.3,
                ),
                BandwiseJitter(sigma=0.02, kind="multiplicative", p=0.8),
                # transforms.Normalize(mean=S2L2A_MEAN, std=S2L2A_STD),
                Divideby10000Normalize(),
                # clamp 0-1

            ])
        else:
            # raise NotImplementedError("S2 validation transforms - check norm method.")
            return transforms.Compose([
                    transforms.CenterCrop(224),
                    # transforms.Normalize(mean=S2L2A_MEAN, std=S2L2A_STD),
                    Divideby10000Normalize(),
                ])
    elif modality.lower() == "s2c" or modality.lower() == "s2l1c":
        if is_train:
            return transforms.Compose([
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)],
                    p=0.3,
                ),
                BandwiseJitter(sigma=0.02, kind="multiplicative", p=0.8),
                transforms.Normalize(mean=S2L1C_MEAN, std=S2L1C_STD),
                

            ])
        else:
            print("Using S2 13 band L1C normalization.")
            return transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Normalize(mean=S2L1C_MEAN, std=S2L1C_STD),
            ])
    elif modality=='rgb':
        if is_train:
            raise NotImplementedError("RGB transforms not implemented yet.")
        else:
            return transforms.Compose([
                transforms.CenterCrop(224),
                transforms.Normalize(mean=S2RGB_MEAN, std=S2RGB_STD),
            ])
    else:
        raise ValueError(f"Unsupported modality: {modality}")

# def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
def get_data(args, preprocess_fns=None):
    data = {}
    model_cfg = getattr(args, "model", None)
    encoder_pair = getattr(model_cfg, "encoder_pair", "s1s2")
    tokenizer = None
    if encoder_pair in ("s2_text", "s1s2_text"):
        context_length = getattr(getattr(model_cfg, "text", None), "context_length", 77)
        tokenizer = partial(clip_tokenize, context_length=context_length)

    if args.dataset.dataset_type == "ssl4eo":
        ### make splits
        if args.train.use_val:
            raise NotImplementedError("use_val not implemented for ssl4eo yet.")
            # first prepare full dataset
            full_datainfo = get_dataset_fn(args.dataset.train_data, args.dataset.dataset_type)(
                args, is_train=True, transforms=preprocess_fns) # is_train and transforms don't matter here
            
            # then make splits
            full_dataset = full_datainfo.dataloader.dataset
            data['train'], data['val'] = generate_splits(full_dataset, args.train.val_frac, seed=46)

            # then convert to datainfo
            data['train'] = dataset_to_datainfo(args, data['train'], is_train=True)
            data['val'] = dataset_to_datainfo(args, data['val'], is_train=False)


        else:
            transforms_map = {}
            if encoder_pair == "s2_text":
                s2_preprocess = get_transform(args.dataset.s2_tier, is_train=True)
                assert s2_preprocess is not None
                transforms_map = {"s2": s2_preprocess}
            else:
                s1_preprocess = get_transform("s1", is_train=True)
                s2_preprocess = get_transform(args.dataset.s2_tier, is_train=True)
                assert s1_preprocess is not None and s2_preprocess is not None
                transforms_map = {"s1": s1_preprocess, "s2": s2_preprocess}
                
            data['train'] = get_dataset_fn(args.dataset.train_data, args.dataset.dataset_type)(
                args, is_train=True, transforms=transforms_map, tokenizer=tokenizer)

        return data
    
    raise NotImplementedError("Only ssl4eo dataset type is supported for now.")
    # if args.dataset.train_data or args.dataset.dataset_type == "synthetic":
    #     data["train"] = get_dataset_fn(args.dataset.train_data, args.dataset.dataset_type)(
    #         args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    # if args.val_data:
    #     data["val"] = get_dataset_fn(args.val_data, args.dataset.dataset_type)(
    #         args, preprocess_val, is_train=False, tokenizer=tokenizer)

    # if args.imagenet_val is not None:
    #     data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    # if args.imagenet_v2 is not None:
    #     data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")
    
    # return data


def _update_running_stats(
    running_sum: torch.Tensor,
    running_sumsq: torch.Tensor,
    running_count: torch.Tensor,
    batch: torch.Tensor,
) -> None:
    batch = batch.double()
    running_sum += batch.sum(dim=(0, 2, 3))
    running_sumsq += (batch ** 2).sum(dim=(0, 2, 3))
    running_count += batch.shape[0] * batch.shape[2] * batch.shape[3]


def compute_ssl4eo_band_stats(
    root: str | Path,
    s2_tier: str = "S2L2A",
    max_files: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    transforms_map = {
        "s1": get_transform("s1", is_train=True),
        "s2": get_transform(s2_tier, is_train=True),
    }
    dataset = SSL4EODataset(
        root=root,
        seasons=[0, 1, 2, 3],
        transforms=transforms_map,
        s2_tier=s2_tier,
        is_train=True,
    )

    num_files = len(dataset) if max_files is None else min(max_files, len(dataset))
    if num_files < 1:
        raise ValueError("No files selected for statistics computation.")

    first_sample = dataset[0]
    s1_channels = first_sample["s1"].shape[1]
    s2_channels = first_sample["s2"].shape[1]

    s1_sum = torch.zeros(s1_channels, dtype=torch.float64)
    s1_sumsq = torch.zeros(s1_channels, dtype=torch.float64)
    s1_count = torch.zeros(s1_channels, dtype=torch.float64)
    s2_sum = torch.zeros(s2_channels, dtype=torch.float64)
    s2_sumsq = torch.zeros(s2_channels, dtype=torch.float64)
    s2_count = torch.zeros(s2_channels, dtype=torch.float64)

    for idx in range(num_files):
        sample = dataset[idx]
        _update_running_stats(s1_sum, s1_sumsq, s1_count, sample["s1"])
        _update_running_stats(s2_sum, s2_sumsq, s2_count, sample["s2"])

    s1_mean = s1_sum / s1_count
    s2_mean = s2_sum / s2_count
    s1_var = s1_sumsq / s1_count - s1_mean ** 2
    s2_var = s2_sumsq / s2_count - s2_mean ** 2
    s1_std = torch.sqrt(torch.clamp(s1_var, min=0.0))
    s2_std = torch.sqrt(torch.clamp(s2_var, min=0.0))

    return s1_mean, s1_std, s2_mean, s2_std


def _format_band_stats(prefix: str, bands: list[str], mean: torch.Tensor, std: torch.Tensor) -> None:
    print(f"{prefix} bands: {bands}")
    print(f"{prefix} mean: {mean.tolist()}")
    print(f"{prefix} std: {std.tolist()}")


def _parse_band_names(s2_tier: str, s2_channels: int) -> list[str]:
    tier = s2_tier.lower()
    if tier in ("s2l2a", "s2a"):
        return DEFAULT_S2_BANDS[:s2_channels]
    if tier in ("s2l1c", "s2c"):
        return ["1", "2", "3", "4", "5", "6", "7", "8", "8A", "9", "10", "11", "12"][:s2_channels]
    if tier in ("rgb",):
        return ["4", "3", "2"][:s2_channels]
    return [str(i) for i in range(s2_channels)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute SSL4EO bandwise mean/std after transforms.")
    parser.add_argument("--root", type=str, required=True, help="SSL4EO root (contains S1GRD/S2L2A).")
    parser.add_argument("--s2-tier", type=str, default="S2L2A", help="S2 tier: S2L2A, S2L1C, or RGB.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files processed.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for transform randomness.")
    args = parser.parse_args()

    s1_mean, s1_std, s2_mean, s2_std = compute_ssl4eo_band_stats(
        root=args.root,
        s2_tier=args.s2_tier,
        max_files=args.max_files,
        seed=args.seed,
    )
    _format_band_stats("S1", ["VV", "VH"][: len(s1_mean)], s1_mean, s1_std)
    s2_bands = _parse_band_names(args.s2_tier, len(s2_mean))
    _format_band_stats("S2", s2_bands, s2_mean, s2_std)


########################################################################################

# class CsvDataset(Dataset):
#     def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
#         logging.debug(f'Loading csv data from {input_filename}.')
#         df = pd.read_csv(input_filename, sep=sep)

#         self.images = df[img_key].tolist()
#         self.captions = df[caption_key].tolist()
#         self.transforms = transforms
#         logging.debug('Done loading data.')

#         self.tokenize = tokenizer

#     def __len__(self):
#         return len(self.captions)

#     def __getitem__(self, idx):
#         images = self.transforms(Image.open(str(self.images[idx])))
#         texts = self.tokenize([str(self.captions[idx])])[0]
#         return images, texts


# def expand_urls(urls, weights=None):
#     if weights is None:
#         expanded_urls = wds.shardlists.expand_urls(urls)
#         return expanded_urls, None
#     if isinstance(urls, str):
#         urllist = urls.split("::")
#         weights = weights.split('::')
#         assert len(weights) == len(urllist),\
#             f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
#         weights = [float(weight) for weight in weights]
#         all_urls, all_weights = [], []
#         for url, weight in zip(urllist, weights):
#             expanded_url = list(braceexpand.braceexpand(url))
#             expanded_weights = [weight for _ in expanded_url]
#             all_urls.extend(expanded_url)
#             all_weights.extend(expanded_weights)
#         return all_urls, all_weights
#     else:
#         all_urls = list(urls)
#         return all_urls, weights


# def get_dataset_size(shards):
#     shards_list, _ = expand_urls(shards)
#     dir_path = os.path.dirname(shards_list[0])
#     sizes_filename = os.path.join(dir_path, 'sizes.json')
#     len_filename = os.path.join(dir_path, '__len__')
#     if os.path.exists(sizes_filename):
#         sizes = json.load(open(sizes_filename, 'r'))
#         total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
#     elif os.path.exists(len_filename):
#         # FIXME this used to be eval(open(...)) but that seemed rather unsafe
#         total_size = ast.literal_eval(open(len_filename, 'r').read())
#     else:
#         total_size = None  # num samples undefined
#         # some common dataset sizes (at time of authors last download)
#         # CC3M (train): 2905954
#         # CC12M: 10968539
#         # LAION-400M: 407332084
#         # LAION-2B (english): 2170337258
#     num_shards = len(shards_list)
#     return total_size, num_shards


# def get_imagenet(args, preprocess_fns, split):
#     assert split in ["train", "val", "v2"]
#     is_train = split == "train"
#     preprocess_train, preprocess_val = preprocess_fns

#     if split == "v2":
#         from imagenetv2_pytorch import ImageNetV2Dataset
#         dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
#     else:
#         if is_train:
#             data_path = args.imagenet_train
#             preprocess_fn = preprocess_train
#         else:
#             data_path = args.imagenet_val
#             preprocess_fn = preprocess_val
#         assert data_path

#         dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

#     if is_train:
#         idxs = np.zeros(len(dataset.targets))
#         target_array = np.array(dataset.targets)
#         k = 50
#         for c in range(1000):
#             m = target_array == c
#             n = len(idxs[m])
#             arr = np.zeros(n)
#             arr[:k] = 1
#             np.random.shuffle(arr)
#             idxs[m] = arr

#         idxs = idxs.astype('int')
#         sampler = SubsetRandomSampler(np.where(idxs)[0])
#     else:
#         sampler = None

#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.datamodule.batch_size,
#         num_workers=args.model.workers,
#         sampler=sampler,
#     )

#     return DataInfo(dataloader=dataloader, sampler=sampler)


# def count_samples(dataloader):
#     os.environ["WDS_EPOCH"] = "0"
#     n_elements, n_batches = 0, 0
#     for images, texts in dataloader:
#         n_batches += 1
#         n_elements += len(images)
#         assert len(images) == len(texts)
#     return n_elements, n_batches


# def filter_no_caption_or_no_image(sample):
#     has_caption = ('txt' in sample)
#     has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
#     return has_caption and has_image


# def log_and_continue(exn):
#     """Call in an exception handler to ignore any exception, issue a warning, and continue."""
#     logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
#     return True


# def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
#     """Return function over iterator that groups key, value pairs into samples.

#     :param keys: function that splits the key into key and extension (base_plus_ext)
#     :param lcase: convert suffixes to lower case (Default value = True)
#     """
#     current_sample = None
#     for filesample in data:
#         assert isinstance(filesample, dict)
#         fname, value = filesample["fname"], filesample["data"]
#         prefix, suffix = keys(fname)
#         if prefix is None:
#             continue
#         if lcase:
#             suffix = suffix.lower()
#         # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
#         #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
#         #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
#         if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
#             if valid_sample(current_sample):
#                 yield current_sample
#             current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
#         if suffixes is None or suffix in suffixes:
#             current_sample[suffix] = value
#     if valid_sample(current_sample):
#         yield current_sample


# def tarfile_to_samples_nothrow(src, handler=log_and_continue):
#     # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
#     streams = url_opener(src, handler=handler)
#     files = tar_file_expander(streams, handler=handler)
#     samples = group_by_keys_nothrow(files, handler=handler)
#     return samples


# def pytorch_worker_seed(increment=0):
#     """get dataloader worker seed from pytorch"""
#     worker_info = get_worker_info()
#     if worker_info is not None:
#         # favour using the seed already created for pytorch dataloader workers if it exists
#         seed = worker_info.seed
#         if increment:
#             # space out seed increments so they can't overlap across workers in different iterations
#             seed += increment * max(1, worker_info.num_workers)
#         return seed
#     # fallback to wds rank based seed
#     return wds.utils.pytorch_worker_seed()


# _SHARD_SHUFFLE_SIZE = 2000
# _SHARD_SHUFFLE_INITIAL = 500
# _SAMPLE_SHUFFLE_SIZE = 5000
# _SAMPLE_SHUFFLE_INITIAL = 1000


# class detshuffle2(wds.PipelineStage):
#     def __init__(
#             self,
#             bufsize=1000,
#             initial=100,
#             seed=0,
#             epoch=-1,
#     ):
#         self.bufsize = bufsize
#         self.initial = initial
#         self.seed = seed
#         self.epoch = epoch

#     def run(self, src):
#         if isinstance(self.epoch, SharedEpoch):
#             epoch = self.epoch.get_value()
#         else:
#             # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
#             # situation as different workers may wrap at different times (or not at all).
#             self.epoch += 1
#             epoch = self.epoch
#         rng = random.Random()
#         if self.seed < 0:
#             # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
#             seed = pytorch_worker_seed(epoch)
#         else:
#             # This seed to be deterministic AND the same across all nodes/workers in each epoch
#             seed = self.seed + epoch
#         rng.seed(seed)
#         return _shuffle(src, self.bufsize, self.initial, rng)


# class ResampledShards2(IterableDataset):
#     """An iterable dataset yielding a list of urls."""

#     def __init__(
#         self,
#         urls,
#         weights=None,
#         nshards=sys.maxsize,
#         worker_seed=None,
#         deterministic=False,
#         epoch=-1,
#     ):
#         """Sample shards from the shard list with replacement.

#         :param urls: a list of URLs as a Python list or brace notation string
#         """
#         super().__init__()
#         urls, weights = expand_urls(urls, weights)
#         self.urls = urls
#         self.weights = weights
#         if self.weights is not None:
#             assert len(self.urls) == len(self.weights),\
#                 f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
#         assert isinstance(self.urls[0], str)
#         self.nshards = nshards
#         self.rng = random.Random()
#         self.worker_seed = worker_seed
#         self.deterministic = deterministic
#         self.epoch = epoch

#     def __iter__(self):
#         """Return an iterator over the shards."""
#         if isinstance(self.epoch, SharedEpoch):
#             epoch = self.epoch.get_value()
#         else:
#             # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
#             # situation as different workers may wrap at different times (or not at all).
#             self.epoch += 1
#             epoch = self.epoch
#         if self.deterministic:
#             # reset seed w/ epoch if deterministic
#             if self.worker_seed is None:
#                 # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
#                 seed = pytorch_worker_seed(epoch)
#             else:
#                 seed = self.worker_seed() + epoch
#             self.rng.seed(seed)
#         for _ in range(self.nshards):
#             if self.weights is None:
#                 yield dict(url=self.rng.choice(self.urls))
#             else:
#                 yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


# def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
#     input_shards = args.dataset.train_data if is_train else args.val_data
#     assert input_shards is not None
#     resampled = getattr(args, 'dataset_resampled', False) and is_train

#     num_shards = None
#     if is_train:
#         if args.train_num_samples is not None:
#             num_samples = args.train_num_samples
#         else:
#             num_samples, num_shards = get_dataset_size(input_shards)
#             if not num_samples:
#                 raise RuntimeError(
#                     'Currently, the number of dataset samples must be specified for the training dataset. '
#                     'Please specify it via `--train-num-samples` if no dataset length info is present.')
#     else:
#         # Eval will just exhaust the iterator if the size is not specified.
#         num_samples = args.val_num_samples or 0 

#     shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

#     if is_train and args.dataset.train_data_upsampling_factors is not None:
#         assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
#     if resampled:
#         pipeline = [ResampledShards2(
#             input_shards,
#             weights=args.dataset.train_data_upsampling_factors,
#             deterministic=True,
#             epoch=shared_epoch,
#         )]
#     else:
#         pipeline = [wds.SimpleShardList(input_shards)]

#     # at this point we have an iterator over all the shards
#     if is_train:
#         if not resampled:
#             pipeline.extend([
#                 detshuffle2(
#                     bufsize=_SHARD_SHUFFLE_SIZE,
#                     initial=_SHARD_SHUFFLE_INITIAL,
#                     seed=args.seed,
#                     epoch=shared_epoch,
#                 ),
#                 wds.split_by_node,
#                 wds.split_by_worker,
#             ])
#         pipeline.extend([
#             # at this point, we have an iterator over the shards assigned to each worker at each node
#             tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
#             wds.shuffle(
#                 bufsize=_SAMPLE_SHUFFLE_SIZE,
#                 initial=_SAMPLE_SHUFFLE_INITIAL,
#             ),
#         ])
#     else:
#         pipeline.extend([
#             wds.split_by_worker,
#             # at this point, we have an iterator over the shards assigned to each worker
#             wds.tarfile_to_samples(handler=log_and_continue),
#         ])
#     pipeline.extend([
#         wds.select(filter_no_caption_or_no_image),
#         wds.decode("pilrgb", handler=log_and_continue),
#         wds.rename(image="jpg;png;jpeg;webp", text="txt"),
#         wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
#         wds.to_tuple("image", "text"),
#         wds.batched(args.datamodule.batch_size, partial=not is_train)
#     ])

#     dataset = wds.DataPipeline(*pipeline)

#     if is_train:
#         if not resampled:
#             num_shards = num_shards or len(expand_urls(input_shards)[0])
#             assert num_shards >= args.model.workers * args.world_size, 'number of shards must be >= total workers'
#         # roll over and repeat a few samples to get same number of full batches on each node
#         round_fn = math.floor if floor else math.ceil
#         global_batch_size = args.datamodule.batch_size * args.world_size
#         num_batches = round_fn(num_samples / global_batch_size)
#         num_workers = max(1, args.model.workers)
#         num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
#         num_batches = num_worker_batches * num_workers
#         num_samples = num_batches * global_batch_size
#         dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
#     else:
#         # last batches are partial, eval is done on single (master) node
#         num_batches = math.ceil(num_samples / args.datamodule.batch_size)

#     dataloader = wds.WebLoader(
#         dataset,
#         batch_size=None,
#         shuffle=False,
#         num_workers=args.model.workers,
#         persistent_workers=args.model.workers > 0,
#     )

#     # FIXME not clear which approach is better, with_epoch before vs after dataloader?
#     # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
#     # if is_train:
#     #     # roll over and repeat a few samples to get same number of full batches on each node
#     #     global_batch_size = args.datamodule.batch_size * args.world_size
#     #     num_batches = math.ceil(num_samples / global_batch_size)
#     #     num_workers = max(1, args.model.workers)
#     #     num_batches = math.ceil(num_batches / num_workers) * num_workers
#     #     num_samples = num_batches * global_batch_size
#     #     dataloader = dataloader.with_epoch(num_batches)
#     # else:
#     #     # last batches are partial, eval is done on single (master) node
#     #     num_batches = math.ceil(num_samples / args.datamodule.batch_size)

#     # add meta-data to dataloader instance for convenience
#     dataloader.num_batches = num_batches
#     dataloader.num_samples = num_samples

#     return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


# def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
#     input_filename = args.dataset.train_data if is_train else args.val_data
#     assert input_filename
#     dataset = CsvDataset(
#         input_filename,
#         preprocess_fn,
#         img_key=args.csv_img_key,
#         caption_key=args.csv_caption_key,
#         sep=args.csv_separator,
#         tokenizer=tokenizer
#     )
#     num_samples = len(dataset)
#     sampler = DistributedSampler(dataset) if args.model.distributed and is_train else None
#     shuffle = is_train and sampler is None

#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.datamodule.batch_size,
#         shuffle=shuffle,
#         num_workers=args.model.workers,
#         pin_memory=True,
#         sampler=sampler,
#         drop_last=is_train,
#     )
#     dataloader.num_samples = num_samples
#     dataloader.num_batches = len(dataloader)

#     return DataInfo(dataloader, sampler)


# class SyntheticDataset(Dataset):

#     def __init__(
#             self,
#             transform=None,
#             image_size=(224, 224),
#             caption="Dummy caption",
#             dataset_size=100,
#             tokenizer=None,
#     ):
#         self.transform = transform
#         self.image_size = image_size
#         self.caption = caption
#         self.image = Image.new('RGB', image_size)
#         self.dataset_size = dataset_size

#         self.preprocess_txt = lambda text: tokenizer(text)[0]

#     def __len__(self):
#         return self.dataset_size

#     def __getitem__(self, idx):
#         if self.transform is not None:
#             image = self.transform(self.image)
#         return image, self.preprocess_txt(self.caption)


# def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
#     image_size = preprocess_fn.transforms[0].size
#     dataset = SyntheticDataset(
#         transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
#     num_samples = len(dataset)
#     sampler = DistributedSampler(dataset) if args.model.distributed and is_train else None
#     shuffle = is_train and sampler is None

#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.datamodule.batch_size,
#         shuffle=shuffle,
#         num_workers=args.model.workers,
#         pin_memory=True,
#         sampler=sampler,
#         drop_last=is_train,
#     )
#     dataloader.num_samples = num_samples
#     dataloader.num_batches = len(dataloader)

#     return DataInfo(dataloader, sampler)
