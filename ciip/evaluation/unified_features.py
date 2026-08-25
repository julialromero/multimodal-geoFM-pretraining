"""Dataset construction and feature extraction for unified evaluation."""

from __future__ import annotations
import contextlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import EuroSAT, BigEarthNet
from torchvision import transforms
from ciip.evaluation.transforms import ImageSampleTransform
from ciip.evaluation.export_neuco_embeddings import E2SChallengeDataset, InputResizer
from ciip.models.evaluation_adapters import EvaluationAdapter
from ciip.evaluation.normalization import (
    NORMALIZATION_METHOD_BANDWISE,
    NORMALIZATION_METHOD_IMAGENET,
    NORMALIZATION_METHOD_SSL4EO,
    NeuCoNormalize,
    SSL4EONormalize,
    build_normalization_transform,
    resolve_normalization_method_for_weights,
    select_ssl4eo_transform,
)
from ciip.evaluation.unified_types import EmbeddingBundle, ModelEvalConfig
from ciip.open_clip_train import data
from ciip.open_clip_train.data import SSL4EODataset
from ciip.visualization.ssl4eo.embedding_collapse_diagnostics import (
    ModalityEmbeddings,
    ensure_hydra_original_cwd,
    extract_embeddings_for_dataset,
)

EUROSATMEAN = {
    "B01": 1354.40546513,
    "B02": 1118.24399958,
    "B03": 1042.92983953,
    "B04": 947.62620298,
    "B05": 1199.47283961,
    "B06": 1999.79090914,
    "B07": 2369.22292565,
    "B08": 2296.82608323,
    "B09": 732.08340178,
    "B10": 12.11327804,
    "B11": 1819.01027855,
    "B12": 1118.92391149,
    "B8A": 2594.14080798,
}

EUROSATSTD = {
    "B01": 245.71762908,
    "B02": 333.00778264,
    "B03": 395.09249139,
    "B04": 593.75055589,
    "B05": 566.4170017,
    "B06": 861.18399006,
    "B07": 1086.63139075,
    "B08": 1117.98170791,
    "B09": 404.91978886,
    "B10": 4.77584468,
    "B11": 1002.58768311,
    "B12": 761.30323499,
    "B8A": 1231.58581042,
}

EUROSATBANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)

EUROSAT_S1_MEAN = {
    "VV": -12.577,
    "VH": -20.265,
}

EUROSAT_S1_STD = {
    "VV": 5.179,
    "VH": 5.872,
}

EUROSAT_S1_BANDS = ("VV", "VH")

EUROSAT_CLASS_NAMES = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake",
}

BIGEARTHNET_BANDS = tuple(b for b in EUROSATBANDS if b != "B10")

BIGEARTHNET_MEAN = [
    383.02593994140625,  # B01
    487.1451721191406,  # B02
    707.4555053710938,  # B03
    726.1536254882812,  # B04
    1098.5545654296875,  # B05
    1900.9755859375,  # B06
    2184.544189453125,  # B07
    2329.376953125,  # B08
    2387.28515625,  # B8A
    2358.551513671875,  # B09
    1878.415283203125,  # B11
    1237.1298828125,  # B12
]

BIGEARTHNET_STD = [
    455.3548889160156,  # B01
    511.12603759765625,  # B02
    543.4214477539062,  # B03
    678.1015625,  # B04
    686.7001953125,  # B05
    992.5777587890625,  # B06
    1162.4130859375,  # B07
    1267.3634033203125,  # B08
    1245.698974609375,  # B8A
    1190.7650146484375,  # B09
    1112.59716796875,  # B11
    879.724609375,  # B12
]

BIGEARTHNET_STATS = {
    band: {"mean": mean, "std": std}
    for band, mean, std in zip(BIGEARTHNET_BANDS, BIGEARTHNET_MEAN, BIGEARTHNET_STD)
}


def _looks_like_vit_checkpoint(checkpoint_path: Optional[Path]) -> bool:
    if checkpoint_path is None:
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint if isinstance(checkpoint, dict) else {}
    for key in state_dict.keys():
        if (
            "transformer" in key
            or "positional_embedding" in key
            or "class_embedding" in key
            or "ln_post" in key
        ):
            return True
    return False


def _first_conv_in_channels(module: Optional[nn.Module]) -> Optional[int]:
    if module is None:
        return None
    conv1 = getattr(module, "conv1", None)
    if isinstance(conv1, nn.Conv2d):
        return conv1.in_channels
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            return submodule.in_channels
    return None


def _infer_model_in_channels(
    adapter: EvaluationAdapter, fallback: int, *, modality: str = "s2"
) -> int:
    """Best-effort detection of the channel count for the requested modality."""

    normalized = modality.lower()
    encoders = []
    if normalized == "s1":
        encoders.append(getattr(adapter, "encoder_s1", None))
        encoders.append(getattr(adapter, "encoder_s2", None))
    else:
        encoders.append(getattr(adapter, "encoder_s2", None))
        encoders.append(getattr(adapter, "encoder_s1", None))
    encoders.append(getattr(adapter, "base_model", None))

    for encoder in encoders:
        in_channels = _first_conv_in_channels(encoder)
        if in_channels is not None:
            return in_channels
    return fallback


def _resolve_eurosat_bands(num_channels: int) -> Tuple[str, ...]:
    """Return the EuroSAT band tuple matching the model channel budget."""

    if num_channels == 10:
        return ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    if num_channels >= len(EUROSATBANDS):
        return EUROSATBANDS
    if num_channels == 12:
        return tuple(band for band in EUROSATBANDS if band != "B10")
    if num_channels == 3:
        return ("B04", "B03", "B02")
    if num_channels < 1:
        raise ValueError("Model must expose at least one Sentinel-2 channel")
    logging.warning(
        "Model exposes %d Sentinel-2 channels; defaulting to the first %d EuroSAT bands.",
        num_channels,
        num_channels,
    )
    return tuple(EUROSATBANDS[:num_channels])


def _resolve_bigearthnet_bands(num_channels: int) -> Tuple[str, ...]:
    """Return the BigEarthNet band tuple matching the model channel budget."""

    if num_channels == 10:
        return ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    if num_channels >= len(BIGEARTHNET_BANDS):
        return BIGEARTHNET_BANDS
    if num_channels == 3:
        return ("B04", "B03", "B02")
    if num_channels < 1:
        raise ValueError("Model must expose at least one Sentinel-2 channel")
    logging.warning(
        "Model exposes %d Sentinel-2 channels; defaulting to the first %d BigEarthNet bands.",
        num_channels,
        num_channels,
    )
    return tuple(BIGEARTHNET_BANDS[:num_channels])


def _align_s2_channels(image: torch.Tensor, expected_channels: Optional[int]) -> torch.Tensor:
    """Drop the cirrus band when the model only supports 12 Sentinel-2 bands."""

    if expected_channels is None or image.ndim < 3:
        return image

    if image.ndim >= 5:
        channel_dim = 2  # (B, T, C, H, W)
    elif image.ndim >= 4:
        channel_dim = 1  # (B, C, H, W)
    else:
        channel_dim = 0
    current_channels = image.size(channel_dim)

    if current_channels == expected_channels:
        return image

    if expected_channels == 10 and current_channels >= 10:
        indices = torch.arange(current_channels, device=image.device)
        if current_channels == 13:
            keep = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        elif current_channels == 12:
            keep = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        else:
            keep = list(range(2, min(current_channels, expected_channels + 2)))
        keep_indices = indices[keep]
        return torch.index_select(image, channel_dim, keep_indices)

    if current_channels == 13 and expected_channels == 12:
        indices = torch.arange(current_channels, device=image.device)
        keep_indices = torch.cat([indices[:10], indices[11:]])
        return torch.index_select(image, channel_dim, keep_indices)

    if current_channels == 12 and expected_channels == 13:
        if channel_dim == 0:
            zeros = torch.zeros(
                (1, *image.shape[1:]),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat([image[:10], zeros, image[10:]], dim=0)
        if channel_dim == 1:
            zeros = torch.zeros(
                (image.size(0), 1, *image.shape[2:]),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat([image[:, :10], zeros, image[:, 10:]], dim=1)
        if channel_dim == 2:
            zeros = torch.zeros(
                (image.size(0), image.size(1), 1, *image.shape[3:]),
                dtype=image.dtype,
                device=image.device,
            )
            return torch.cat(
                [image[:, :, :10], zeros, image[:, :, 10:]],
                dim=2,
            )

    logging.warning(
        "Unable to automatically align Sentinel-2 channels (expected %s, observed %s).",
        expected_channels,
        current_channels,
    )
    return image


def _use_adapter_modality(adapter: EvaluationAdapter, modality: str):
    """Temporarily switch the adapter to the requested modality."""

    setter = getattr(adapter, "set_active_modality", None)
    getter = getattr(adapter, "get_active_modality", None)
    if setter is None or getter is None:
        yield
        return

    normalized = (modality or "").lower()
    previous = getter()
    if not normalized or normalized == previous:
        yield
        return

    setter(normalized)
    try:
        yield
    finally:
        setter(previous)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


def _extract_image_tensor(batch) -> Optional[torch.Tensor]:
    if isinstance(batch, dict):
        return batch.get("image") if "image" in batch else batch.get("data")
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return None


def _to_nchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 5:
        channel_candidates = {1, 2, 3, 4, 10, 12, 13}
        if tensor.shape[1] in {10, 12, 13}:
            b, c, t, h, w = tensor.shape
            return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if tensor.shape[2] in {10, 12, 13}:
            b, t, c, h, w = tensor.shape
            return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if tensor.shape[1] in channel_candidates and tensor.shape[2] not in channel_candidates:
            b, c, t, h, w = tensor.shape
            return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if tensor.shape[2] in channel_candidates:
            b, t, c, h, w = tensor.shape
            return tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    raise ValueError(f"Unsupported tensor shape {tuple(tensor.shape)} for band stats")


def _print_band_stats(
    label: str,
    loader: DataLoader,
    *,
    bands: Optional[Sequence[str]] = None,
    max_batches: int = 0,
) -> None:
    total_sum: Optional[torch.Tensor] = None
    total_sumsq: Optional[torch.Tensor] = None
    total_pixels = 0
    batches = 0

    for idx, batch in enumerate(loader):
        if max_batches and idx >= max_batches:
            break
        images = _extract_image_tensor(batch)
        if images is None:
            continue
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        images = _to_nchw(images.detach()).to(dtype=torch.float64, device="cpu")
        total_pixels += images.shape[0] * images.shape[2] * images.shape[3]
        batch_sum = images.sum(dim=(0, 2, 3))
        batch_sumsq = (images**2).sum(dim=(0, 2, 3))
        total_sum = batch_sum if total_sum is None else total_sum + batch_sum
        total_sumsq = batch_sumsq if total_sumsq is None else total_sumsq + batch_sumsq
        batches += 1

    if total_sum is None or total_sumsq is None or total_pixels == 0:
        print(f"{label}: unable to compute band stats (no samples).")
        return

    mean = total_sum / total_pixels
    var = (total_sumsq / total_pixels) - mean**2
    std = torch.sqrt(torch.clamp(var, min=0.0))
    band_note = f" bands={list(bands)}" if bands is not None else ""
    batch_note = f" batches={batches}" + (f"/{max_batches}" if max_batches else "")
    print(
        f"{label} bandwise stats after transforms:{band_note}{batch_note} "
        f"mean={mean.tolist()} std={std.tolist()}"
    )


def _extract_embeddings(
    adapter: EvaluationAdapter,
    dataloader: DataLoader,
    *,
    device: torch.device,
    require_ids: bool = False,
    expected_in_channels: Optional[int] = None,
    modality: str = "s2",
    pca_output_dir: Optional[Path] = None,
) -> EmbeddingBundle:
    backbone_vectors: List[np.ndarray] = []
    projected_vectors: List[np.ndarray] = []
    labels: List[int] = []
    multi_labels: List[List[int]] = []
    ids: List[str] = []

    for i, batch in enumerate(dataloader):
        if isinstance(batch, dict):
            image = batch.get("image") if "image" in batch else batch.get("data")
            batch_labels = batch.get("label")
            batch_multi_labels = batch.get("multi_label")
            batch_ids = batch.get("file_name")
        else:
            image, batch_labels = batch
            batch_multi_labels = None
            batch_ids = None




        image = image.to(device)
        adapter.to(device)


        if isinstance(image, dict) and not getattr(adapter, "supports_multimodal_dict", False):
            image = image[next(iter(image))]

        reshape_back = False
        seasons = 1
        if isinstance(image, torch.Tensor) and image.ndim == 5:
            batch_size, seasons, channels, height, width = image.shape
            image = image.reshape(batch_size * seasons, channels, height, width)
            reshape_back = True
        if modality.lower() == "s2" and isinstance(image, torch.Tensor):
            image = _align_s2_channels(image, expected_in_channels)

        prepared_inputs = adapter.prepare_inputs(image, device=device, modality=modality)

        with torch.no_grad():
            outputs = adapter.compute_embeddings(prepared_inputs, modality=modality)

        if isinstance(outputs, dict):
            backbone = outputs.get("backbone")
            projected = outputs.get("projected", None)
        elif isinstance(outputs, tuple) and len(outputs) == 3:
            backbone, _, projected = outputs
        else:
            raise RuntimeError("Unexpected outputs from compute_embeddings")

        if reshape_back:
            if backbone is not None:
                backbone = backbone.reshape(batch_size, seasons, *backbone.shape[1:])
            if projected is not None:
                projected = projected.reshape(batch_size, seasons, *projected.shape[1:])

        if backbone is None:
            raise RuntimeError("Backbone embeddings cannot be None")
        if backbone.ndim == 3:
            backbone = backbone.mean(dim=1)

        backbone_np = _to_numpy(backbone)
        backbone_vectors.append(backbone_np)

        if projected is not None:
            projected_vectors.append(_to_numpy(projected))

        if batch_labels is not None:
            labels.extend(batch_labels.cpu().tolist())
        if batch_multi_labels is not None:
            multi_labels.extend(batch_multi_labels.cpu().tolist())
        if require_ids and batch_ids is not None:
            ids.extend([str(item) for item in batch_ids])

    backbone_array: Optional[np.ndarray]
    if backbone_vectors:
        backbone_array = np.concatenate(backbone_vectors, axis=0)
    else:
        backbone_array = None

    projected_array = np.concatenate(projected_vectors, axis=0) if projected_vectors else None

    bundle = EmbeddingBundle(
        backbone=backbone_array,
        projected=projected_array,
        labels=np.asarray(labels, dtype=np.int64) if labels else None,
        multi_labels=np.asarray(multi_labels, dtype=np.int64) if multi_labels else None,
        ids=ids if ids else None,
    )
    return bundle


def _build_eurosat_loaders(
    config: ModelEvalConfig, *, bands: Sequence[str], modality: str
) -> Dict[str, DataLoader]:
    normalized = modality.lower()
    if normalized == "s1":
        raise ValueError("EuroSAT dataset does not support Sentinel-1 modality")

    mean = [EUROSATMEAN[b] for b in bands]
    std = [EUROSATSTD[b] for b in bands]

    target_size = int(config.eurosat_image_size)
    eval_resize = max(target_size, int(round(target_size * 256 / 224)))

    normalization_method = resolve_normalization_method_for_weights(
        config.model_in_channels, config.normalization_method, config.model_weights
    )
    norm_layer = build_normalization_transform(
        normalization_method,
        bandwise_stats=(mean, std),
    )
    print(f"EuroSAT using normalization method: {norm_layer}")
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(target_size),
                transforms.RandomHorizontalFlip(),
                norm_layer,
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize(eval_resize),
                transforms.CenterCrop(target_size),
                norm_layer,
            ]
        ),
    }

    train_transform = ImageSampleTransform(data_transforms["train"])
    eval_transform = ImageSampleTransform(data_transforms["eval"])

    datasets = {
        split: EuroSAT(
            root=str(config.eurosat_root),
            split=split,
            bands=bands,
            transforms=train_transform if split == "train" else eval_transform,
            download=True,
        )
        for split in ("train", "val", "test")
    }

    loaders = {
        split: DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        for split, dataset in datasets.items()
    }
    for split, dataset in datasets.items():
        assert len(dataset) > 0, f"EuroSAT {split} dataset is empty!"
    return loaders


class _SingleLabelBigEarthNet(torch.utils.data.Dataset):
    """Wrap BigEarthNet multi-label targets into a single class via argmax."""

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            image = sample.get("image") if "image" in sample else sample.get("data")
            label = sample.get("label") if "label" in sample else sample.get("labels")
        else:
            image, label = sample

        multi_label = None
        if isinstance(label, torch.Tensor):
            if label.ndim > 0 and label.numel() > 1:
                label_idx = int(torch.argmax(label).item())
                multi_label = label
            else:
                label_idx = int(label.item())
        elif isinstance(label, (list, tuple)):
            label_idx = int(label[0])
        else:
            label_idx = int(label)

        return {"image": image, "label": label_idx, "multi_label": multi_label}


def _filter_missing_bigearthnet_samples(dataset: BigEarthNet) -> None:
    """Drop BigEarthNet entries missing required band folders/files."""
    data_root = Path(dataset.root) / dataset.metadata["s2"]["directory"]
    if not data_root.exists() or not any(data_root.iterdir()):
        print(
            f"BigEarthNet: expected imagery under {data_root} but found none; "
            "marking dataset as empty."
        )
        dataset.folders = []
        return

    required_keys = []
    uses_s2 = dataset.bands in ("s2", "all") or (
        isinstance(dataset.bands, (list, tuple))
        and all(isinstance(b, str) and b.startswith("B") for b in dataset.bands)
    )
    if uses_s2:
        required_keys.append("s2")
    if dataset.bands in ("s1", "all"):
        required_keys.append("s1")

    def _has_band_folder(folder_path: str, band_key: str) -> bool:
        path = Path(folder_path)
        if not path.is_dir():
            return False

        suffix = "B01.tif" if band_key == "s2" else "VV.tif"
        expected_file = path / f"{path.name}_{suffix}"
        if expected_file.exists():
            return True
        return any(path.glob("*.tif"))

    valid_folders = []
    missing = 0
    for folder in dataset.folders:
        if all(_has_band_folder(folder[key], key) for key in required_keys):
            valid_folders.append(folder)
        else:
            missing += 1

    if missing:
        print(
            f"BigEarthNet: filtered out {missing} samples missing required "
            f"{str(dataset.bands).upper()} imagery (kept {len(valid_folders)})."
        )
    dataset.folders = valid_folders


def _bigearthnet_stats_for_bands(bands: Sequence[str]) -> Tuple[List[float], List[float]]:
    mean: List[float] = []
    std: List[float] = []
    for band in bands:
        stats = BIGEARTHNET_STATS.get(band)
        if stats is None:
            raise KeyError(f"Unknown BigEarthNet band '{band}' in requested band set {bands}")
        mean.append(stats["mean"])
        std.append(stats["std"])
    return mean, std


def _resolve_bigearthnet_root(root: Path) -> Path:
    """Handle nested BigEarthNet extracts (root/BigEarthNet-v1.0/BigEarthNet-v1.0)."""
    base = Path(root)
    csv_present = (base / "bigearthnet-train.csv").exists()
    nested = base / "BigEarthNet-v1.0"
    double_nested = nested / "BigEarthNet-v1.0"

    if csv_present:
        return base
    if (nested / "bigearthnet-train.csv").exists():
        print(f"BigEarthNet: detected nested layout with CSVs, using root {nested}")
        return nested
    if (double_nested / "bigearthnet-train.csv").exists():
        print(f"BigEarthNet: detected double-nested layout with CSVs, using root {double_nested}")
        return double_nested
    return base


def _build_bigearthnet_loaders(
    config: ModelEvalConfig, *, expected_channels: Optional[int] = None
) -> Tuple[Dict[str, DataLoader], Sequence[str]]:
    if config.bigearthnet_root is None:
        raise ValueError("bigearthnet_root must be provided for BigEarthNet evaluation.")

    channel_budget = expected_channels or config.model_in_channels
    bands = _resolve_bigearthnet_bands(channel_budget)
    band_mean, band_std = _bigearthnet_stats_for_bands(bands)
    band_indices = [BIGEARTHNET_BANDS.index(b) for b in bands]
    select_bands = len(band_indices) != len(BIGEARTHNET_BANDS)

    target_size = int(config.bigearthnet_image_size)

    class _SelectBandsTransform:
        def __init__(self, indices: Sequence[int]):
            self.indices = list(indices)

        def __call__(self, img: torch.Tensor) -> torch.Tensor:
            if not isinstance(img, torch.Tensor):
                return img
            if img.ndim >= 4:
                channel_dim = 1  # (B, C, H, W) or (T, C, H, W)
            elif img.ndim == 3:
                channel_dim = 0  # (C, H, W)
            else:
                return img
            index_tensor = torch.as_tensor(self.indices, device=img.device)
            return torch.index_select(img, channel_dim, index_tensor)

    selector = _SelectBandsTransform(band_indices) if select_bands else None

    train_steps: List[object] = []
    eval_steps: List[object] = []
    if selector is not None:
        train_steps.append(selector)
        eval_steps.append(selector)
    normalization_method = resolve_normalization_method_for_weights(
        config.model_in_channels, config.normalization_method, config.model_weights
    )
    if normalization_method == NORMALIZATION_METHOD_BANDWISE:
        band_norm_layer = transforms.Normalize(mean=band_mean, std=band_std)
    elif normalization_method == NORMALIZATION_METHOD_SSL4EO:
        band_norm_layer = SSL4EONormalize()
    else:
        band_norm_layer = build_normalization_transform(normalization_method)
    print(f"band norm method: {band_norm_layer}")
    train_steps.extend(
        [
            transforms.RandomResizedCrop(target_size),
            transforms.RandomHorizontalFlip(),
            band_norm_layer,
        ]
    )
    eval_steps.extend(
        [
            transforms.CenterCrop(target_size),
            band_norm_layer,
        ]
    )

    data_transforms = {
        "train": transforms.Compose(train_steps),
        "eval": transforms.Compose(eval_steps),
    }

    train_transform = ImageSampleTransform(data_transforms["train"])
    eval_transform = ImageSampleTransform(data_transforms["eval"])

    print("Building BigEarthNet datasets...")

    dataset_root = _resolve_bigearthnet_root(Path(config.bigearthnet_root))

    datasets: Dict[str, _SingleLabelBigEarthNet] = {}
    for split in ("train", "val", "test"):
        base_dataset = BigEarthNet(
            root=str(dataset_root),
            split=split,
            bands="s2",
            transforms=train_transform if split == "train" else eval_transform,
            download=False,
            num_classes=19,
        )
        if len(base_dataset) == 0:
            data_dir = Path(dataset_root) / base_dataset.metadata["s2"]["directory"]
            raise FileNotFoundError(
                f"BigEarthNet split '{split}' has no available samples under {dataset_root}. "
                f"Ensure the dataset is fully extracted (expected imagery in {data_dir}) "
                f"or update --bigearthnet-root."
            )
        datasets[split] = _SingleLabelBigEarthNet(base_dataset)

    loaders = {
        split: DataLoader(dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
        for split, dataset in datasets.items()
    }
    for split, dataset in datasets.items():
        print(f"BigEarthNet {split} dataset size: {len(dataset)} samples")
        assert len(dataset) > 0, f"BigEarthNet {split} dataset is empty!"

    return loaders, bands


def _build_neuco_loader(
    config: ModelEvalConfig,
    *,
    modalities: Sequence[str],
) -> DataLoader:
    transform_steps: List[object] = []
    if config.model_type != "ciip_checkpoint":
        model_transform = select_ssl4eo_transform(config.model_weights)
        if model_transform is None:
            model_transform = select_ssl4eo_transform(config.model_type)
        if model_transform is None:
            raise ValueError(
                "No SSL4EO transform found for model_weights="
                f"{config.model_weights!r} or model_type={config.model_type!r}"
            )
        transform_steps.append(model_transform)
    else:
        transform_steps.append(InputResizer(224))
        normalization_method = resolve_normalization_method_for_weights(
            config.model_in_channels, config.normalization_method, config.model_weights
        )

        if normalization_method == NORMALIZATION_METHOD_BANDWISE:
            print("Using NeuCo bandwise normalization for NeuCo inputs")
            transform_steps.append(NeuCoNormalize())
        elif normalization_method == NORMALIZATION_METHOD_SSL4EO:
            print("Using SSL4EO normalization for NeuCo inputs")
            transform_steps.append(SSL4EONormalize())
        else:
            if normalization_method == NORMALIZATION_METHOD_IMAGENET:
                print("Using ImageNet normalization for NeuCo inputs")
            else:
                print("Using divide-by-10000 normalization for NeuCo inputs")
            transform_steps.append(build_normalization_transform(normalization_method))

    transform = transforms.Compose(transform_steps)

    rgb = True if config.model_in_channels == 3 else False
    dataset = E2SChallengeDataset(
        data_path=str(config.neuco_root),
        modalities=list(modalities),
        seasons=config.neuco_seasons,
        concat=True,
        output_file_name=True,
        transform=transform,
        rgb=rgb,
    )

    assert len(dataset) > 0, "NeuCo dataset is empty!"

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
    )
    for batch in loader:
        if isinstance(batch, dict):
            image = batch.get("image") if "image" in batch else batch.get("data")
        else:
            image, _ = batch
    return loader


def _build_ssl4eo_dataset(config: ModelEvalConfig) -> torch.utils.data.Dataset:
    if config.ssl4eo_root is None:
        raise RuntimeError("SSL4EO dataset root must be provided for diagnostics")

    ensure_hydra_original_cwd()

    s2_transform = select_ssl4eo_transform(config.model_weights)
    if s2_transform is None:
        print(f"Using default SSL4EO transform for tier {config.ssl4eo_s2_tier}")
        normalization_method = resolve_normalization_method_for_weights(
            config.model_in_channels, config.normalization_method, config.model_weights
        )
        if normalization_method in {NORMALIZATION_METHOD_BANDWISE, NORMALIZATION_METHOD_SSL4EO}:
            norm_layer = SSL4EONormalize()
        else:
            norm_layer = build_normalization_transform(normalization_method)
        s2_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                norm_layer,
            ]
        )

    dataset = SSL4EODataset(
        root=str(config.ssl4eo_root.expanduser()),
        s2_tier=str(config.ssl4eo_s2_tier),
        seasons=[0, 1, 2, 3],
        num_timestamps=4,
        transforms={"s1": data.get_transform("s1", is_train=False), "s2": s2_transform},
        is_train=False,
    )

    total = len(dataset)
    subset_size = config.ssl4eo_subset_size
    if subset_size > 0 and subset_size < total:
        indices = sorted(np.random.choice(total, size=subset_size, replace=False).tolist())
        dataset = Subset(dataset, indices)
    return dataset


def _extract_ssl4eo_embeddings(
    config: ModelEvalConfig, model: nn.Module, *, device: torch.device, max_batches_cka: int
) -> Tuple[ModalityEmbeddings, ModalityEmbeddings, List[str]]:
    dataset = _build_ssl4eo_dataset(config)

    if config.model_in_channels == 10 and isinstance(getattr(dataset, "transforms", None), dict):
        orig_s2 = dataset.transforms.get("s2")

        def _align_transform(tensor: torch.Tensor) -> torch.Tensor:
            transformed = orig_s2(tensor) if callable(orig_s2) else tensor
            return _align_s2_channels(transformed, 10)

        dataset.transforms["s2"] = _align_transform

    autocast = (
        (lambda: torch.cuda.amp.autocast()) if device.type == "cuda" else contextlib.nullcontext
    )

    input_dtype = getattr(model, "dtype_s2", torch.float32)
    if device.type != "cuda" and input_dtype in {torch.float16, torch.bfloat16}:
        input_dtype = torch.float32

    s1_embeddings, s2_embeddings, sample_ids = extract_embeddings_for_dataset(
        model,
        dataset,
        input_dtype=input_dtype,
        device=device,
        autocast=autocast,
        max_batches_cka=max_batches_cka,
    )

    assert s2_embeddings is not None, "S2 embeddings should not be None"
    return s1_embeddings, s2_embeddings, sample_ids
