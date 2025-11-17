"""Concise end-to-end evaluation entrypoint.

This module provides a single callable ``run_full_evaluation`` that executes the
three evaluation pipelines requested by the research team plus the optional
Lorentz-specific visualisations.  The implementation intentionally reuses the
utilities that already exist across the codebase (``linearprobe_comparison``,
``export_neuco_embeddings`` and ``visualizations/ssl4eo``) instead of invoking
their CLI entrypoints so that everything can be orchestrated from Python.

The function expects a ``ModelEvalConfig`` describing the dataset locations,
output directory and model source.  By default it consumes CIIP checkpoints,
but users can also request TorchGeo ResNet50 encoders initialised with the
Sentinel-2 DINO or MoCo weights.
"""

from __future__ import annotations

import json
import logging
import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.manifold import TSNE
# from ciip.evaluation.ssl4eo_retrieval import compute_cross_modal_retrieval

from torchgeo.datasets import EuroSAT
from torchvision import transforms

from ciip.eval_utils import CustomTransform
from ciip.evaluation.model_utils import EvaluationAdapter, build_evaluation_adapter
from ciip.model_ciip import LorentzCIIP
import os

## EUROSAT STANDARDIZATION VALUES
EUROSATMEAN = {
    'B01': 1354.40546513,
    'B02': 1118.24399958,
    'B03': 1042.92983953,
    'B04': 947.62620298,
    'B05': 1199.47283961,
    'B06': 1999.79090914,
    'B07': 2369.22292565,
    'B08': 2296.82608323,
    'B09': 732.08340178,
    'B10': 12.11327804,
    'B11': 1819.01027855,
    'B12': 1118.92391149,
    'B8A': 2594.14080798,
}

EUROSATSTD = {
    'B01': 245.71762908,
    'B02': 333.00778264,
    'B03': 395.09249139,
    'B04': 593.75055589,
    'B05': 566.4170017,
    'B06': 861.18399006,
    'B07': 1086.63139075,
    'B08': 1117.98170791,
    'B09': 404.91978886,
    'B10': 4.77584468,
    'B11': 1002.58768311,
    'B12': 761.30323499,
    'B8A': 1231.58581042,
}

EUROSATBANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
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
##


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

    if num_channels >= len(EUROSATBANDS):
        return EUROSATBANDS
    if num_channels == 12:
        return tuple(band for band in EUROSATBANDS if band != "B10")
    if num_channels < 1:
        raise ValueError("Model must expose at least one Sentinel-2 channel")
    logging.warning(
        "Model exposes %d Sentinel-2 channels; defaulting to the first %d EuroSAT bands.",
        num_channels,
        num_channels,
    )
    return tuple(EUROSATBANDS[:num_channels])


def _align_s2_channels(image: torch.Tensor, expected_channels: Optional[int]) -> torch.Tensor:
    """Drop the cirrus band when the model only supports 12 Sentinel-2 bands."""

    if expected_channels is None or image.ndim < 3:
        return image

    # channel_dim = 1 if image.ndim >= 4 else 0
    if image.ndim >= 5:
        channel_dim = 2  # (B, T, C, H, W)
    elif image.ndim >= 4:
        channel_dim = 1  # (B, C, H, W)
    else:
        channel_dim = 0
    current_channels = image.size(channel_dim)

    if current_channels == expected_channels:
        return image

    if current_channels == 13 and expected_channels == 12:
        # Drop B10 (0-indexed channel 10) to match 12-band encoders.
        indices = torch.arange(current_channels, device=image.device)
        keep_indices = torch.cat([indices[:10], indices[11:]])
        return torch.index_select(image, channel_dim, keep_indices)

    logging.warning(
        "Unable to automatically align Sentinel-2 channels (expected %s, observed %s).",
        expected_channels,
        current_channels,
    )
    return image


@contextlib.contextmanager
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

from ciip.evaluation.export_neuco_embeddings import (  # type: ignore
    E2SChallengeDataset,
    InputResizer,
    Normalize,
    TemporalMean,
    collate_fn,
    create_submission_from_dict,
)
from visualizations.ssl4eo.embedding_collapse_diagnostics import (  # type: ignore
    DEFAULT_S2_BANDS,
    EpochDiagnostics,
    ModalityEmbeddings,
    compute_cross_encoder_cka,
    compute_linear_cka,
    compute_projection,
    compute_singular_values,
    compute_within_encoder_cka,
    ensure_hydra_original_cwd,
    extract_embeddings_for_dataset,
    plot_epoch_diagnostics,
    plot_epoch_diagnostics_s2only,
    plot_projection,
    preprocess_projection_data,
    ModalityEmbeddings,
    compute_within_encoder_cka,
    compute_cross_encoder_cka
)
from visualizations.ssl4eo.hyperbolic_visualization import (  # type: ignore
    compute_hyperbolic_context,
    plot_angle_aperture,
    plot_angular_pca,
    plot_cone_polar,
    plot_radial_histogram,
    _extract_curvature_from_state,
    compute_lorentz_pos_neg,
    plot_pos_neg_hist
)
from visualizations.ssl4eo.hyperbolic_retrieval import compute_cross_modal_retrieval
from ciip.open_clip_train.data import SSL4EODataset

@dataclass
class ModelEvalConfig:
    eurosat_root: Path
    neuco_root: Path
    output_dir: Path
    checkpoint: Optional[Path] = None
    model_type: str = "ciip_checkpoint"
    model_weights: Optional[str] = None
    model_in_channels: int = 13
    evaluation_modality: str = "s2"
    croma_weights: Optional[Path] = None
    croma_image_resolution: int = 120
    enable_ssl4eo: bool = True
    neuco_modalities: Sequence[str] = ("s2l1c",)
    neuco_resize: Optional[Tuple[int, int]] = None
    neuco_seasons: int = 1
    tsne_samples: int = 1500
    pca_samples: int = 5000
    random_seed: int = 0
    model_path: Optional[str] = None
    ssl4eo_root: Optional[Path] = None
    ssl4eo_subset_size: int = 2048
    ssl4eo_subset_seed: int = 0
    ssl4eo_s2_tier: str = "s2c"
    ssl4eo_s2_bands: Sequence[str] = DEFAULT_S2_BANDS
    ssl4eo_image_dimension: int = 224
    eurosat_image_size: int = 224


@dataclass
class EmbeddingBundle:
    backbone: Optional[np.ndarray]
    posthead: np.ndarray
    projected: np.ndarray
    labels: Optional[np.ndarray] = None
    ids: Optional[List[str]] = None


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


### FOR EUROSAT and NEUCOBENCH
def _extract_embeddings(
    adapter: EvaluationAdapter,
    dataloader: DataLoader,
    *,
    device: torch.device,
    require_ids: bool = False,
    expected_in_channels: Optional[int] = None,
    modality: str = "s2",
) -> EmbeddingBundle:
    backbone_vectors: List[np.ndarray] = []
    posthead_vectors: List[np.ndarray] = []
    projected_vectors: List[np.ndarray] = []
    labels: List[int] = []
    ids: List[str] = []

    
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if isinstance(batch, dict):
            # print(batch)
            # print(batch)
            # print(batch.keys())
            # print(batch['data'])
            # image = batch.get("image") # or 
            # get image if key 'image' exists else 'data'
            image = batch.get("image") if "image" in batch else batch.get("data")
            # image = batch.get("data")
            batch_labels = batch.get("label")
            batch_ids = batch.get("file_name")
        else:
            image, batch_labels = batch
            batch_ids = None

        


        if isinstance(image, dict) and not getattr(adapter, "supports_multimodal_dict", False):
            image = image[next(iter(image))]

        prepared_inputs = adapter.prepare_inputs(image, device=device)

        with torch.no_grad():
            backbone, post, projected = adapter.compute_embeddings(prepared_inputs)

        if backbone is None:
            raise RuntimeError("Backbone embeddings cannot be None")
        if modality.lower() == "s2" and isinstance(image, torch.Tensor):
            image = _align_s2_channels(image, expected_in_channels)

            input_dtype = adapter.dtype_s2
            if device.type != "cuda" and input_dtype in {torch.float16, torch.bfloat16}:
                input_dtype = torch.float32
            image = image.to(device=device, dtype=input_dtype, non_blocking=True)

        backbone_np = _to_numpy(backbone)
        backbone_vectors.append(backbone_np)
        batch_size = backbone_np.shape[0]

        if post is None:
            posthead_vectors.append(np.zeros((batch_size, 0), dtype=np.float32))
        else:
            posthead_vectors.append(_to_numpy(post))

        if projected is None:
            projected_vectors.append(np.zeros((batch_size, 0), dtype=np.float32))
        else:
            projected_vectors.append(_to_numpy(projected))

        if batch_labels is not None:
            labels.extend(batch_labels.cpu().tolist())
        if require_ids and batch_ids is not None:
            ids.extend([str(item) for item in batch_ids])

    backbone_array: Optional[np.ndarray]
    if backbone_vectors:
        backbone_array = np.concatenate(backbone_vectors, axis=0)
    else:
        backbone_array = None

    bundle = EmbeddingBundle(
        backbone=backbone_array,
        posthead=np.concatenate(posthead_vectors, axis=0),
        projected=np.concatenate(projected_vectors, axis=0),
        labels=np.asarray(labels, dtype=np.int64) if labels else None,
        ids=ids if ids else None,
    )
    return bundle




def _build_eurosat_loaders(
    config: ModelEvalConfig, *, bands: Sequence[str], modality: str
) -> Dict[str, DataLoader]:
    normalized = modality.lower()
    if normalized == "s1":
        mean = [EUROSAT_S1_MEAN[b] for b in bands]
        std = [EUROSAT_S1_STD[b] for b in bands]
    else:
        mean = [EUROSATMEAN[b] for b in bands]
        std = [EUROSATSTD[b] for b in bands]

    target_size = int(config.eurosat_image_size)
    eval_resize = max(target_size, int(round(target_size * 256 / 224)))

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(target_size),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize(eval_resize),
                transforms.CenterCrop(target_size),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    }

    train_transform = CustomTransform(data_transforms["train"])
    eval_transform = CustomTransform(data_transforms["eval"])

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
    # print each split
    for split, dataset in datasets.items():
        print(f"EuroSAT {split} dataset size: {len(dataset)} samples")
        assert len(dataset) > 0, f"EuroSAT {split} dataset is empty!"
    return loaders


def _build_neuco_loader(
    config: ModelEvalConfig,
    *,
    modalities: Sequence[str],
    modality: str,
) -> DataLoader:
    transform_steps: List[object] = []
    if modality.lower() != "s1":
        transform_steps.append(Normalize())
    transform_steps.append(TemporalMean())
    if config.neuco_resize is not None:
        transform_steps.append(InputResizer(config.neuco_resize))
    transform = transforms.Compose(transform_steps)
    print(f"Building NeuCo dataset with modalities: {config.neuco_modalities}")
    dataset = E2SChallengeDataset(
        data_path=str(config.neuco_root),
        modalities=list(modalities),
        seasons=config.neuco_seasons,
        concat=True,
        output_file_name=True,
        transform=transform,
    )

    # check size of datset
    print(f"NeuCo dataset size: {len(dataset)} samples")
    assert len(dataset) > 0, "NeuCo dataset is empty!"

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    return loader


def _build_ssl4eo_dataset(config: ModelEvalConfig) -> torch.utils.data.Dataset:
    if config.ssl4eo_root is None:
        raise RuntimeError("SSL4EO dataset root must be provided for diagnostics")

    ensure_hydra_original_cwd()
    dataset = SSL4EODataset(
        root=str(config.ssl4eo_root.expanduser()),
        # s2_tier=str(config.ssl4eo_s2_tier),
        # s2_bands=list(config.ssl4eo_s2_bands),
        transforms=None,
        target_image_dimension=(config.ssl4eo_image_dimension, config.ssl4eo_image_dimension),
        s2_tier=config.neuco_modalities[0],  # use first modality as tier
    )

    total = len(dataset)
    subset_size = config.ssl4eo_subset_size
    if subset_size > 0 and subset_size < total:
        rng = np.random.default_rng(config.ssl4eo_subset_seed)
        indices = sorted(rng.choice(total, size=subset_size, replace=False).tolist())
        dataset = Subset(dataset, indices)
    return dataset


def _extract_ssl4eo_embeddings(
    config: ModelEvalConfig,
    model: nn.Module,
    *,
    device: torch.device,
) -> Tuple[ModalityEmbeddings, ModalityEmbeddings, List[str]]:
    dataset = _build_ssl4eo_dataset(config)

    autocast = (
        (lambda: torch.cuda.amp.autocast())
        if device.type == "cuda"
        else contextlib.nullcontext
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
    )

    assert s2_embeddings is not None, "S2 embeddings should not be None"
    return s1_embeddings, s2_embeddings, sample_ids


def _run_linear_probe(
    config: ModelEvalConfig,
    embeddings: Dict[str, EmbeddingBundle],
    *,
    output_dir: Path,
    label: str,
) -> None:
    percents = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    rng = np.random.default_rng(config.random_seed)

    def _evaluate(feature_key: str, batch_norm) -> Dict[float, Dict[str, float]]:
        results: Dict[float, Dict[str, float]] = {}
        # print(embeddings)
        train = embeddings["train"]
        val = embeddings["val"]
        test = embeddings["test"]

        # raise NotImplementedError("Debugging embeddings extraction")

        train_feats = getattr(train, feature_key)
        val_feats = getattr(val, feature_key)
        test_feats = getattr(test, feature_key)

        if batch_norm:
            mean = train_feats.mean(axis=0, keepdims=True)
            std = train_feats.std(axis=0, keepdims=True) + 1e-10
            train_feats = (train_feats - mean) / std
            val_feats = (val_feats - mean) / std
            test_feats = (test_feats - mean) / std

        for pct in percents:
            n_samples = max(1, int(pct * len(train_feats)))
            indices = rng.choice(len(train_feats), size=n_samples, replace=False)

            clf = LogisticRegression(max_iter=4000, multi_class="multinomial", solver="lbfgs")
            clf.fit(train_feats[indices], train.labels[indices])

            val_pred = clf.predict(val_feats)
            test_pred = clf.predict(test_feats)

            val_acc = accuracy_score(val.labels, val_pred)
            val_f1 = f1_score(val.labels, val_pred, average="weighted")
            test_acc = accuracy_score(test.labels, test_pred)
            test_f1 = f1_score(test.labels, test_pred, average="weighted")

            results[pct] = {
                "val_accuracy": float(val_acc),
                "val_f1": float(val_f1),
                "test_accuracy": float(test_acc),
                "test_f1": float(test_f1),
            }

        return results

    
    
    if embeddings["train"].posthead.sum() == 0:
        probe_specs = (
            ("backbone", "backbone", False),
            ("backbone", "backbone_batchnorm", True),
        )
        marker_map = {"backbone": "o"}

    else:
        probe_specs = (
            ("backbone", "backbone", False),
            ("posthead", "posthead", False),
            ("projected", "projected", False),
            ("backbone", "backbone_batchnorm", True),
            ("posthead", "posthead_batchnorm", True),
            ("projected", "projected_batchnorm", True),
        )
        marker_map = {"backbone": "o", "posthead": "^", "projected": "s"}


    plots_dir = output_dir / "linear_probe"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics = _evaluate(feature_key, batch_norm=use_batch_norm)
        with (plots_dir / f"{label}_{suffix}_metrics.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    

    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        acc = [metrics[str(x)]["test_accuracy"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        ax1.plot(xs, acc, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})")

    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        f1_vals = [metrics[str(x)]["test_f1"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        ax2.plot(xs, f1_vals, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})")

    ax1.set_xlabel("Training fraction")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title(f"Test Accuracy ({label})")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Training fraction")
    ax2.set_ylabel("Test F1")
    ax2.set_title(f"Test F1 ({label})")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / f"{label}_combined_curves.png", dpi=200)
    plt.close(fig)

    logging.info("Linear probe results saved to %s", plots_dir)

    #     xs = sorted(metrics)
    #     acc = [metrics[x]["test_accuracy"] for x in xs]
    #     f1_vals = [metrics[x]["test_f1"] for x in xs]

    #     fig, ax = plt.subplots(figsize=(6, 4))
    #     ax.plot(xs, acc, marker="o", label="test accuracy")
    #     ax.plot(xs, f1_vals, marker="s", label="test F1")
    #     ax.set_xlabel("Training fraction")
    #     ax.set_ylabel("Score")
    #     ax.set_title(f"Linear probe ({label}, {pretty})")
    #     ax.grid(alpha=0.3)
    #     ax.legend()
    #     fig.tight_layout()
    #     fig.savefig(plots_dir / f"{label}_{key}_curves.png", dpi=200)
    #     plt.close(fig)

    # # print the save path
    # logging.info(f"Linear probe results saved to {plots_dir}")

def _plot_eurosat_tsne(
    bundle: EmbeddingBundle,
    *,
    output_dir: Path,
    label: str,
    max_samples: int,
    seed: int,
    model_title: Optional[str],
) -> None:
    if bundle.labels is None:
        logging.warning("No labels provided for EuroSAT embeddings; skipping t-SNE plot")
        return

    features = None
    if bundle.posthead is not None and bundle.posthead.size > 0:
        features = bundle.posthead
    elif bundle.backbone is not None and bundle.backbone.size > 0:
        features = bundle.backbone
    elif bundle.projected is not None and bundle.projected.size > 0:
        features = bundle.projected

    if features is None:
        logging.warning("No embeddings available for EuroSAT t-SNE plot")
        return

    labels = bundle.labels
    total_samples = len(features)
    rng = np.random.default_rng(seed)
    if total_samples > max_samples:
        indices = rng.choice(total_samples, size=max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    embeddings_2d = tsne.fit_transform(features)

    unique_labels = sorted(np.unique(labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, class_label in enumerate(unique_labels):
        mask = labels == class_label
        label_text = EUROSAT_CLASS_NAMES.get(int(class_label), str(class_label))
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            s=8,
            color=cmap(idx),
            label=label_text,
            alpha=0.7,
        )

    title = model_title if model_title is not None else "EuroSAT t-SNE"
    ax.set_title(title)
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.legend(title="Class", fontsize="small", markerscale=2)
    ax.grid(alpha=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_tsne.png", dpi=200)
    plt.close(fig)


def _export_neuco(
    bundle: EmbeddingBundle,
    output_dir: Path,
    *,
    label: str,
) -> None:
    if bundle.ids is None:
        raise RuntimeError("NeuCo export requires file identifiers")

    def _write(features: np.ndarray, suffix: str) -> None:
        rows = {idx: vec for idx, vec in zip(bundle.ids or [], features)}
        df = create_submission_from_dict(rows)
        df.to_csv(output_dir / f"neuco_{label}_{suffix}.csv", index=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle.backbone is not None:
        _write(bundle.backbone, "backbone")
    _write(bundle.posthead, "posthead")
    _write(bundle.projected, "projected")


def _run_embedding_diagnostics(
    config: ModelEvalConfig,
    s1_bundle: ModalityEmbeddings,
    s2_bundle: ModalityEmbeddings,
    *,
    sample_ids: Optional[Sequence[str]] = None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # print(s1_bundle.layer_activations)
    # if empty s1_bundle riase error
    if s1_bundle is not None:
        if not s1_bundle.layer_activations or not s2_bundle.layer_activations:
            raise ValueError("Layer activations are empty, cannot compute CKA diagnostics")
        modality_tensors = {
            "s1": {
                "backbone": s1_bundle.backbone.detach() if s1_bundle.backbone is not None else None,
                "posthead": s1_bundle.raw.detach(),
                "projected": s1_bundle.projected.detach(),
                "layers": s1_bundle.layer_activations,
            },
            "s2": {
                "backbone": s2_bundle.backbone.detach() if s2_bundle.backbone is not None else None,
                "posthead": s2_bundle.raw.detach(),
                "projected": s2_bundle.projected.detach(),
                "layers": s2_bundle.layer_activations,
            },
        }
        assert modality_tensors["s1"]["layers"], "No S1 layer activations for CKA"
        assert modality_tensors["s2"]["layers"], "No S2 layer activations for CKA"
    
        s1_layers, s1_within = compute_within_encoder_cka(modality_tensors["s1"]["layers"])
        cross_s1_layers, cross_s2_layers, cross_matrix = compute_cross_encoder_cka(modality_tensors["s1"]["layers"], modality_tensors["s2"]["layers"])
        projected_cka = compute_linear_cka(modality_tensors["s1"]["projected"], modality_tensors["s2"]["projected"])
        s1_posthead_singular = compute_singular_values(modality_tensors["s1"]["posthead"])
        s1_projected_singular = compute_singular_values(modality_tensors["s1"]["projected"])
        s1_backbone_singular = compute_singular_values(modality_tensors["s1"]["backbone"])
        s2_layers, s2_within = compute_within_encoder_cka(modality_tensors["s2"]["layers"])
        s2_posthead_singular = compute_singular_values(modality_tensors["s2"]["posthead"])
        s2_projected_singular = compute_singular_values(modality_tensors["s2"]["projected"])
        s2_backbone_singular = compute_singular_values(modality_tensors["s2"]["backbone"])
        cka_payload = {
            "s1": {
                "layers": s1_layers,
                "matrix": s1_within.tolist() if s1_within is not None else None,
            },
            "s2": {
                "layers": s2_layers,
                "matrix": s2_within.tolist() if s2_within is not None else None,
            },
            "cross": {
                "s1_layers": cross_s1_layers,
                "s2_layers": cross_s2_layers,
                "matrix": cross_matrix.tolist() if cross_matrix is not None else None,
            },
            "projected_similarity": projected_cka,
        }

        spectra = [
            ("s1", "posthead", s1_posthead_singular),
            ("s1", "projected", s1_projected_singular),
            ("s2", "posthead", s2_posthead_singular),
            ("s2", "projected", s2_projected_singular),
            ("s1", 'backbone', s1_backbone_singular),
            ("s2", 'backbone', s2_backbone_singular),

        ]
    

    else:
        if not s2_bundle.layer_activations:
            raise ValueError("Layer activations are empty, cannot compute CKA diagnostics")

        modality_tensors = {
            "s2": {
                "posthead": s2_bundle.raw.detach(),
                "projected": s2_bundle.projected.detach(),
                "layers": s2_bundle.layer_activations,
            },
            "s1": None
        }
        
        # print(modality_tensors["s2"]["layers"])
        assert modality_tensors["s2"]["layers"], "No S2 layer activations for CKA"
        s2_layers, s2_within = compute_within_encoder_cka(modality_tensors["s2"]["layers"])
        s2_posthead_singular = compute_singular_values(modality_tensors["s2"]["posthead"])
        s2_projected_singular = compute_singular_values(modality_tensors["s2"]["projected"])
        s2_backbone_singular = compute_singular_values(modality_tensors["s2"]["backbone"])
        cka_payload = {
            "s2": {
                "layers": s2_layers,
                "matrix": s2_within.tolist() if s2_within is not None else None,
            },
        }
        spectra = [
            ("s2", "posthead", s2_posthead_singular),
            ("s2", "projected", s2_projected_singular),
            ("s2", 'backbone', s2_backbone_singular),
        ]
        # assert that dims match
        

        s1_posthead_singular = None
        s1_projected_singular = None
        s1_layers = None
        s1_within = None
        cross_s1_layers = None
        cross_matrix = None
        cross_s2_layers = None

    if s2_posthead_singular.shape != s2_projected_singular.shape:
            raise ValueError("S2 posthead and projected singular values have different dimensions")


    with (output_dir / "cka.json").open("w", encoding="utf-8") as handle:
        json.dump(cka_payload, handle, indent=2)

    label_source = (
        (config.checkpoint.stem if config.checkpoint is not None else None)
        or config.model_weights
        or config.model_type
    )
    
    epoch_source = (
        (config.checkpoint.stem if config.checkpoint is not None else "")
        or (config.model_weights or "")
    )
    epoch_match = re.search(r"epoch[_=-]?(\d+)", epoch_source)
    epoch_value = int(epoch_match.group(1)) if epoch_match else 0

    # if torchgeo then set epoch value to None
    if config.model_type == "torchgeo_resnet50":
        epoch_value = None
        label_source = config.model_weights
        

    sample_count = int(s2_bundle.raw.shape[0]) if s2_bundle.raw.ndim > 0 else 0
    diagnostic_ids = (
        [str(item) for item in sample_ids]
        if sample_ids is not None and len(sample_ids) > 0
        else [str(index) for index in range(sample_count)]
    )

    epoch_diagnostics = EpochDiagnostics(
        label=label_source,
        epoch=epoch_value,
        ids=diagnostic_ids,
        s1=s1_bundle,
        s2=s2_bundle,
        s1_singular_values=s1_posthead_singular,
        s2_singular_values=s2_posthead_singular,
        s1_layers=s1_layers,
        s2_layers=s2_layers,
        s1_within_cka=s1_within,
        s2_within_cka=s2_within,
        cross_cka=cross_matrix,
        cross_s1_layers=cross_s1_layers,
        cross_s2_layers=cross_s2_layers,
    )
    if s1_bundle is not None:
        if config.model_type == "torchgeo_resnet50":
            plot_epoch_diagnostics_s2only(epoch_diagnostics, output_dir, label='backbone_raw')
        plot_epoch_diagnostics(epoch_diagnostics, output_dir, label='posthead_raw')


    else:
        if config.model_type == "torchgeo_resnet50":
            plot_epoch_diagnostics_s2only(epoch_diagnostics, output_dir, label='s2_backbone_raw')
        else:
            raise ValueError("Model type not supported for S1 missing case")

    

    n=len(spectra)
    ncols = min(3, n)  # up to 3 columns
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()  # flatten in case of single subplot

    for ax, (modality, feature_name, spectrum) in zip(axes, spectra):
        ax.plot(np.arange(len(spectrum)), spectrum, marker=".")
        ax.set_xlabel("Component")
        ax.set_ylabel("Singular value")
        ax.set_title(f"{modality.upper()} {feature_name}")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 100)

    # Hide any unused subplots
    for ax in axes[len(spectra):]:
        ax.axis("off")

    fig.suptitle("Singular Values per Modality / Feature", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "singular_values_all.png", dpi=200)
    plt.close(fig)


        # if spectrum.size == 0:
        #     continue
        # fig, ax = plt.subplots(figsize=(6, 4))
        # ax.plot(np.arange(len(spectrum)), spectrum, marker=".")
        # ax.set_xlabel("Component")
        # ax.set_ylabel("Singular value")
        # ax.set_title(f"Singular values ({modality.upper()} {feature_name})")
        # ax.grid(alpha=0.3)
        # ax.set_xlim(0,100)
        # fig.tight_layout()
        # fig.savefig(output_dir / f"singular_values_{modality}_{feature_name}.png", dpi=200)
        # plt.close(fig)

    rng = np.random.default_rng(config.random_seed)

    # def _stack_features(feature_name: str) -> Tuple[np.ndarray, np.ndarray]:

    #     s1_np = modality_tensors["s1"][feature_name].cpu().numpy()
    #     s2_np = modality_tensors["s2"][feature_name].cpu().numpy()
    #     labels = np.concatenate((np.repeat("S1", len(s1_np)), np.repeat("S2", len(s2_np))))
    #     features = np.concatenate((s1_np, s2_np), axis=0)
    #     return features, labels


    def _stack_features(feature_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if modality_tensors["s1"] is None:
            s2_np = modality_tensors["s2"][feature_name].cpu().numpy()
            labels = np.repeat("S2", len(s2_np))
            features = s2_np
        else:
            s1_np = modality_tensors["s1"][feature_name].cpu().numpy()
            s2_np = modality_tensors["s2"][feature_name].cpu().numpy()
            #assert shapes match
            assert s1_np.shape[0] == s2_np.shape[0], "S1 and S2 feature counts do not match for stacking shapes are {} and {}".format(s1_np.shape, s2_np.shape)
            labels = np.concatenate((np.repeat("S1", len(s1_np)), np.repeat("S2", len(s2_np))))
            features = np.concatenate((s1_np, s2_np), axis=0)
        return features, labels

    def _sample(features: np.ndarray, labels: np.ndarray, maximum: int) -> Tuple[np.ndarray, np.ndarray]:
        if maximum <= 0 or maximum >= len(features):
            return features, labels
        indices = rng.choice(len(features), size=maximum, replace=False)
        return features[indices], labels[indices]

    for feature_name in ("posthead", "projected", 'backbone'):
        combined, modality_labels = _stack_features(feature_name)
        if combined.size == 0:
            continue
        for mode in ("none", "zscore"):
            subset, subset_labels = _sample(combined, modality_labels, config.tsne_samples)
            processed = preprocess_projection_data(subset, mode=mode, random_state=config.random_seed)
            coords = compute_projection(processed, method="tsne", random_state=config.random_seed)
            if coords is not None:
                suffix = "zscore" if mode == "zscore" else "raw"
                plot_projection(
                    coords,
                    subset_labels,
                    output_dir / f"tsne_{feature_name}_{suffix}.png",
                    title=f"t-SNE ({feature_name}, {suffix})",
                )

            subset, subset_labels = _sample(combined, modality_labels, config.pca_samples)
            processed = preprocess_projection_data(subset, mode=mode, random_state=config.random_seed)
            if processed.shape[0] >= 2:
                suffix = "zscore" if mode == "zscore" else "raw"
                pca_coords = PCA(n_components=2, random_state=config.random_seed).fit_transform(processed)
                plot_projection(
                    pca_coords,
                    subset_labels,
                    output_dir / f"pca_{feature_name}_{suffix}.png",
                    title=f"PCA ({feature_name}, {suffix})",
                )


def _run_hyperbolic_visualisations(
    s1_proj_features: torch.Tensor,
    s2_proj_features: torch.Tensor,
    *,
    output_dir: Path,
    model: LorentzCIIP,
    aperture_logk: Optional[float],
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = compute_hyperbolic_context(
        model,
        s1_proj_features,
        s2_proj_features,
        aperture_logk=aperture_logk,
    )
    positive_angles = context["positive_angles"].cpu().numpy()
    aperture_s1 = context["aperture_s1"].cpu().numpy()
    aperture_s2 = context["aperture_s2"].cpu().numpy()
    s1_dirs = context["s1_dirs"].cpu().numpy()
    s2_dirs = context["s2_dirs"].cpu().numpy()
    s1_distances = context["s1_distances"].cpu().numpy()
    s2_distances = context["s2_distances"].cpu().numpy()

    angles_mat = context["angles"]          # (N, N), on device
    N = angles_mat.size(0)
    # positives: diagonal α(x_i, y_i)
    pos_alpha = context["positive_angles"].cpu().numpy()
    # negatives: all off-diagonals α(x_i, y_j), i != j
    neg_alpha = angles_mat[~torch.eye(N, dtype=torch.bool, device=angles_mat.device)].cpu().numpy()

    # Convert to "similarity" the way the loss does: κ = -α
    pos_angle_sim = -pos_alpha
    neg_angle_sim = -neg_alpha

    plot_pos_neg_hist(
        pos_angle_sim,
        neg_angle_sim,
        use_distance=False,
        out_path=output_dir / "angle_sim_hist.png",
    )

    # # Lorentz “similarity” (hyperbolic analogue of cosine sim)
    # lorentz_mat = context["pairwise_lorentz"]       # (N, N)
    # pos_sim = context["positive_lorentz"].cpu().numpy()       # diag
    # neg_sim = lorentz_mat[~torch.eye(lorentz_mat.size(0), dtype=torch.bool)].cpu().numpy()

    # # # Or distances
    # # dist_mat = ctx["pairwise_dist"]
    # # pos_dist = ctx["positive_dist"]
    # # neg_dist = dist_mat[~torch.eye(dist_mat.size(0), dtype=torch.bool)]

    # plot_pos_neg_hist(pos_sim, neg_sim, use_distance=False, out_path= output_dir /"lorentz_sim_hist.png")

    plot_angle_aperture(positive_angles, aperture_s1, aperture_s2, output_dir / "angle_aperture.png")
    plot_radial_histogram(s1_distances, s2_distances, output_dir / "radial_histogram.png")
    plot_angular_pca(s1_dirs, s2_dirs, s1_distances, s2_distances, output_dir / "angular_pca.png")
    plot_cone_polar(positive_angles, aperture_s1, aperture_s2, output_dir / "cone_polar.png", sample_size=256, seed=seed)


def run_full_evaluation(config: ModelEvalConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    if config.model_type == "croma":
        config.eurosat_image_size = config.croma_image_resolution
        if config.neuco_resize is None:
            config.neuco_resize = (
                config.croma_image_resolution,
                config.croma_image_resolution,
            )

    adapter = build_evaluation_adapter(
        model_type=config.model_type,
        checkpoint=config.checkpoint,
        model_weights=config.model_weights,
        in_chans=config.model_in_channels,
        croma_weights=config.croma_weights,
        croma_image_resolution=config.croma_image_resolution,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = adapter.to(device)
    adapter.eval()
    base_model = getattr(adapter, "base_model", adapter)
    is_lorentz = getattr(adapter, "is_lorentz", False)

    target_modality = config.evaluation_modality.lower()
    if target_modality not in {"s1", "s2"}:
        raise ValueError("evaluation_modality must be either 's1' or 's2'")

    model_channels = _infer_model_in_channels(
        adapter, config.model_in_channels, modality=target_modality
    )
    if target_modality == "s1":
        eurosat_bands = EUROSAT_S1_BANDS
    else:
        eurosat_bands = _resolve_eurosat_bands(model_channels)
    logging.info(
        "EuroSAT linear probe will use %d Sentinel-%s bands (%s)",
        len(eurosat_bands),
        "1" if target_modality == "s1" else "2",
        ", ".join(eurosat_bands),
    )

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # check if dir exists
    eurosat_output_dir = output_dir / "linear_probe"
    if True: #not eurosat_output_dir.exists():
        eurosat_loaders = _build_eurosat_loaders(
            config, bands=eurosat_bands, modality=target_modality
        )
        eurosat_embeddings: Dict[str, EmbeddingBundle] = {}
        with _use_adapter_modality(adapter, target_modality):
            for split, loader in eurosat_loaders.items():
                eurosat_embeddings[split] = _extract_embeddings(
                    adapter,
                    loader,
                    device=device,
                    expected_in_channels=model_channels,
                    modality=target_modality,
                )
        # _run_linear_probe(config, eurosat_embeddings, output_dir=output_dir, label="eurosat")
        _plot_eurosat_tsne(
            eurosat_embeddings["test"],
            output_dir=eurosat_output_dir,
            label="eurosat",
            max_samples=config.tsne_samples,
            seed=config.random_seed,
            model_title=config.model_path,
        )


    # # first check if the output csvs already exist
    # neuco_output_dir = output_dir / "neuco"
    # neuco_modalities: List[str] = list(config.neuco_modalities)
    # if target_modality == "s1":
    #     neuco_modalities = ["s1"]
    # elif not neuco_modalities:
    #     neuco_modalities = ["s2l1c"]

    # if len(neuco_modalities) > 1:
    #     print('Only useing 1st modality for neuco benchmark')
    # modality = neuco_modalities[0]
    # csv_out_backbone = neuco_output_dir / "neuco_export" / f"neuco_{modality}_backbone.csv"
    # csv_out_posthead = neuco_output_dir / "neuco_export" / f"neuco_{modality}_posthead.csv"
    # csv_out_projected = neuco_output_dir / "neuco_export" /f"neuco_{modality}_projected.csv"
    
    # if csv_out_backbone.exists() and csv_out_posthead.exists() and csv_out_projected.exists():
    #     print("NeuCo embeddings already exist, skipping extraction.")
    
 
    # # else:
    # if True:
    #     print("csvs at {}, {}, {} do not exist, extracting NeuCo embeddings.".format(csv_out_backbone, csv_out_posthead, csv_out_projected))
    #     neuco_output_dir.mkdir(parents=True, exist_ok=True)
    #     neuco_loader = _build_neuco_loader(
    #         config, modalities=neuco_modalities, modality=target_modality
    #     )
    #     with _use_adapter_modality(adapter, target_modality):
    #         neuco_bundle = _extract_embeddings(
    #             adapter,
    #             neuco_loader,
    #             device=device,
    #             require_ids=True,
    #             expected_in_channels=model_channels,
    #             modality=target_modality,
    #         )
    
    #     _export_neuco(neuco_bundle, neuco_output_dir / "neuco_export", label=modality)
    #     print('Saved NeuCo embeddings to ', neuco_output_dir / "neuco_export")

    
#     # launch eval
#     import subprocess
#     # get size of embedings
#     try:
#         embedding_dim_backbone = str(neuco_bundle.backbone.shape[1])
#         embedding_dim_posthead = str(neuco_bundle.posthead.shape[1])
#     except:
#         embedding_dim_backbone = "2048"  # default to 512 if extraction failed
#         embedding_dim_posthead = "1024"
#     # if model type ois torchgeo resnet
#     if config.model_type == "torchgeo_resnet50":
#         embedding_dim_backbone = "2048"
#         embedding_dim_posthead = "2048"

#     print(f"NeuCo embedding dimension: {embedding_dim_backbone}")


#     cmd = [
#         "python", "/local/ms-data/NeuCo-Bench/benchmark/main.py",
#         "--annotation_path", "/local/ms-data/SSL4EO-S12-downstream/labels",
#         "--output_dir", neuco_output_dir,
#         "--config", "/local/ms-data/NeuCo-Bench/benchmark/config.yaml",
#         "--method_name", "backbone",
#         "--phase", "testing",
#         "--submission_file", csv_out_backbone,
#         "--embedding_dim", embedding_dim_backbone
#     ]

# # python /local/ms-data/NeuCo-Bench/benchmark/main.py \
# #   --annotation_path /local/ms-data/SSL4EO-S12-downstream/labels \
# #   --output_dir /home/juro4948/ciip/diagnostics/unified_eval/curv_init_1/neuco_export \
# #   --config /local/ms-data/NeuCo-Bench/benchmark/config.yaml \
# #   --method_name backbone \
# #   --phase testing \
# #   --submission_file /home/juro4948/ciip/diagnostics/unified_eval/curv_init_1/neuco_export/neuco_s2l1c_backbone.csv
    
#     cmd1 = [
#         "python", "/local/ms-data/NeuCo-Bench/benchmark/main.py",
#         "--annotation_path", "/local/ms-data/SSL4EO-S12-downstream/labels",
#         "--output_dir", neuco_output_dir,
#         "--config", "/local/ms-data/NeuCo-Bench/benchmark/config.yaml",
#         "--method_name", "posthead",
#         "--phase", "testing",
#         "--submission_file", csv_out_posthead,
#         "--embedding_dim", embedding_dim_posthead
#     ]
#     cmd2 = [
#         "python", "/local/ms-data/NeuCo-Bench/benchmark/main.py",
#         "--annotation_path", "/local/ms-data/SSL4EO-S12-downstream/labels",
#         "--output_dir", neuco_output_dir,
#         "--config", "/local/ms-data/NeuCo-Bench/benchmark/config.yaml",
#         "--method_name", "projected",
#         "--phase", "testing",
#         "--submission_file", csv_out_projected,
#         "--embedding_dim", embedding_dim_posthead
#     ]
#     env = os.environ.copy()
#     env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#     subprocess.run(cmd, check=True, env=env)
#     if config.model_type != "torchgeo_resnet50":
#         subprocess.run(cmd1, check=True, env=env)
#         subprocess.run(cmd2, check=True, env=env)



    ssl4eo_available = (
        config.enable_ssl4eo
        and config.ssl4eo_root is not None
        and getattr(adapter, "supports_ssl4eo", True)
    )

    assert ssl4eo_available or not config.enable_ssl4eo, "SSL4EO diagnostics requested but not available"

    s1_ssl4eo = s2_ssl4eo = None
    if ssl4eo_available:
        s1_ssl4eo, s2_ssl4eo, ssl4eo_ids = _extract_ssl4eo_embeddings(
            config,
            adapter,
            device=device,
        )

        _run_embedding_diagnostics(
            config,
            s1_ssl4eo,
            s2_ssl4eo,
            sample_ids=ssl4eo_ids,
            output_dir=output_dir / "embedding_diagnostics",
        )
        logging.info("Completed SSL4EO diagnostics")
        # print output dir
        print("SSL4EO diagnostics saved to ", output_dir)
    else:
        logging.info(
            "Skipping SSL4EO diagnostics (enable_ssl4eo=%s, root=%s, supports_ssl4eo=%s)",
            config.enable_ssl4eo,
            config.ssl4eo_root,
            getattr(adapter, "supports_ssl4eo", True),
        )
    curvature = None
    if is_lorentz and s1_ssl4eo is not None and s2_ssl4eo is not None:
        if not isinstance(base_model, LorentzCIIP):
            raise TypeError("Expected LorentzCIIP model when is_lorentz is True")
        # use SSL4EO post-head projected features for hyperbolic visualisations
        s1_proj_feats = s1_ssl4eo.projected.to(device=device, dtype=torch.float32)
        s2_proj_feats = s2_ssl4eo.projected.to(device=device, dtype=torch.float32)
        _run_hyperbolic_visualisations(
            s1_proj_feats,
            s2_proj_feats,
            output_dir=output_dir / "hyperbolic",
            model=base_model,
            aperture_logk=None,
            seed=config.random_seed,
        )

        curvature = _extract_curvature_from_state(base_model, device=device, dtype=s1_proj_feats.dtype)
        
    # compute retrieval metrics
    retrieval_metrics = compute_cross_modal_retrieval(s1_ssl4eo, s2_ssl4eo, curvature=curvature)
    retrieval_path = output_dir / "ssl4eo_retrieval.json"
    retrieval_path.write_text(json.dumps(retrieval_metrics, indent=2, sort_keys=True))
    logging.info("SSL4EO cross-modal retrieval metrics: %s", retrieval_metrics)


__all__ = ["ModelEvalConfig", "run_full_evaluation"]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the unified CIIP evaluation pipeline.")
    parser.add_argument(
        "--model-type",
        default="ciip_checkpoint",
        choices=["ciip_checkpoint", "torchgeo_resnet50", "croma"],
        help="Model source to evaluate.",
    )
    # parser.add_argument("--checkpoint", type=Path, help="Checkpoint path for CIIP/Lorentz models.")
    parser.add_argument("--model-weights", choices=["dino", "moco"], help="TorchGeo ResNet50 weight selection.")
    parser.add_argument("--croma-weights", type=Path, help="Path to the pretrained CROMA weights.")
    parser.add_argument("--croma-image-resolution", type=int, default=120, help="Input resolution expected by the CROMA model.")
    parser.add_argument("--model-in-channels", type=int, default=13, help="Number of input channels for TorchGeo ResNet models.")
    parser.add_argument("--model-path", type=str, help="Experiment path identifier for the model.")
    # parser.add_argument("--eurosat-root", type=Path, required=True, help="EuroSAT dataset root directory.")
    # parser.add_argument("--neuco-root", type=Path, required=True, help="NeuCo-Bench dataset root directory.")
    # parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write evaluation artifacts.")
    # parser.add_argument("--ssl4eo-root", type=Path, help="SSL4EO dataset root for diagnostics.")
    parser.add_argument("--disable-ssl4eo", action="store_true", help="Skip SSL4EO diagnostics even if a root is provided.")
    parser.add_argument("--tsne-samples", type=int, default=1500, help="Samples used for t-SNE visualisations.")
    parser.add_argument("--pca-samples", type=int, default=5000, help="Samples used for PCA visualisations.")
    parser.add_argument("--ssl4eo-subset-size", type=int, default=50, help="Subset size for SSL4EO embedding extraction.")
    parser.add_argument("--ssl4eo-subset-seed", type=int, default=0, help="Subset seed for SSL4EO sampling.")
    parser.add_argument("--neuco-modalities", nargs="*", default=["s2l2a"], help="NeuCo modalities to export.")
    parser.add_argument("--neuco-seasons", type=int, default=4, help="Number of seasons for NeuCo extraction.") # i believe these are averaged
    parser.add_argument(
        "--evaluation-modality",
        choices=["s1", "s2"],
        default="s2",
        help="Sentinel modality to use for EuroSAT and NeuCo evaluations.",
    )

    args = parser.parse_args()


# trained on old SSL4EO, 13 bands
## vanilla
    # '2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp': epochs 5, 10 up to 85
## hyperbolic
    # '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/': 'curv_init_1', epochs 0-24
    # '2025_11_06-19_39_33-model_resnet50-lr_0.001-b_128-j_6-p_amp': 'curv_init_0.5', epochs 0-14
    # '2025_11_07-09_24_18-model_resnet50-lr_0.001-b_128-j_6-p_amp': 'curv_init_0.1', epochs 0-16
    
# trained on v1.1, 12 bands
    # '2025_11_12-12_28_55-model_resnet50-lr_0.001-b_2-j_6-p_amp': 'curv_init_0.1', epochs ..., increased lr for curvature param

    # if model_path arg is empty, set default
    if args.model_path is None:
        args.model_path =  '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/'
        # '2025_11_14-10_56_41-model_resnet50-lr_0.001-b_2-j_6-p_amp'
    args.model_root = '/local/ms-data/SSL4EO/model/'
    # '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/'
    # '/local/ms-data/SSL4EO/model/2025_11_13-08_33_13-model_resnet50-lr_0.001-b_2-j_6-p_amp/checkpoints/epoch_10.pt'
    # 
    
    # '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/'
    # '2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp'
    # '2025_11_07-09_24_18-model_resnet50-lr_0.001-b_128-j_6-p_amp'
    # '2025_09_11-14_15_30-model_resnet50-lr_0.0005-b_128-j_6-p_amp'
    checkpoint_root = Path(args.model_root) / args.model_path / "checkpoints"
    # args.output_dir = Path("diagnostics/output")

    args.checkpoint=Path(f"{checkpoint_root}/epoch_10.pt")
    args.eurosat_root=Path("/local/ms-data/EuroSAT/")
    args.neuco_root=Path("/local/ms-data/SSL4EO-S12-downstream/data")
    args.output_dir=Path("/home/juro4948/ciip/diagnostics/unified_eval/curv_init_1_epoch10/") #dino_13bands/") #curv_init_1_epoch10/ curv_init_1
    args.ssl4eo_root=Path("/local/ms-data/SSL4EOv1.1/train")


    if args.model_type == "ciip_checkpoint" and args.checkpoint is None:
        parser.error("--checkpoint is required when --model-type=ciip_checkpoint")
    if args.model_type == "torchgeo_resnet50" and args.model_weights is None:
        parser.error("--model-weights must be provided for torchgeo_resnet50 models")
    if args.model_type == "croma" and args.croma_weights is None:
        parser.error("--croma-weights must be provided when --model-type=croma")
    

    cfg = ModelEvalConfig(
        eurosat_root=args.eurosat_root,
        neuco_root=args.neuco_root,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        model_weights=args.model_weights,
        model_in_channels=args.model_in_channels,
        croma_weights=args.croma_weights,
        croma_image_resolution=args.croma_image_resolution,
        enable_ssl4eo=not args.disable_ssl4eo,
        ssl4eo_root=args.ssl4eo_root,
        tsne_samples=args.tsne_samples,
        pca_samples=args.pca_samples,
        ssl4eo_subset_size=args.ssl4eo_subset_size,
        ssl4eo_subset_seed=args.ssl4eo_subset_seed,
        neuco_modalities=tuple(args.neuco_modalities),
        neuco_seasons=args.neuco_seasons,
        evaluation_modality=args.evaluation_modality,
    )
    run_full_evaluation(cfg)


