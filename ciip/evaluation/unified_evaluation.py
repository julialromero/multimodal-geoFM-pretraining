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

from torchgeo.datasets import EuroSAT
from torchvision import transforms

from ciip.eval_utils import CustomTransform
from ciip.evaluation.model_utils import EvaluationAdapter, build_evaluation_adapter
from ciip.model_ciip import LorentzCIIP


## EUROSAT STANDARDIZATION VALUES
MEAN = {
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

STD = {
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

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
##

from ciip.evaluation.export_neuco_embeddings import (  # type: ignore
    E2SChallengeDataset,
    Normalize,
    TemporalMean,
    collate_fn,
    create_submission_from_dict,
)
from visualizations.ssl4eo.embedding_collapse_diagnostics import (  # type: ignore
    DEFAULT_S2_BANDS,
    compute_linear_cka,
    compute_projection,
    compute_singular_values,
    ensure_hydra_original_cwd,
    extract_embeddings_for_dataset,
    plot_projection,
    preprocess_projection_data,
    ModalityEmbeddings
)
from visualizations.ssl4eo.hyperbolic_visualization import (  # type: ignore
    compute_hyperbolic_context,
    plot_angle_aperture,
    plot_angular_pca,
    plot_cone_polar,
    plot_radial_histogram,
)
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
    enable_ssl4eo: bool = True
    neuco_modalities: Sequence[str] = ("s2l1c",)
    neuco_resize: Optional[Tuple[int, int]] = None
    neuco_seasons: int = 1
    tsne_samples: int = 1500
    pca_samples: int = 5000
    random_seed: int = 0
    ssl4eo_root: Optional[Path] = None
    ssl4eo_subset_size: int = 2048
    ssl4eo_subset_seed: int = 0
    ssl4eo_s2_tier: str = "s2c"
    ssl4eo_s2_bands: Sequence[str] = DEFAULT_S2_BANDS
    ssl4eo_image_dimension: int = 264


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
) -> EmbeddingBundle:
    backbone_vectors: List[np.ndarray] = []
    posthead_vectors: List[np.ndarray] = []
    projected_vectors: List[np.ndarray] = []
    labels: List[int] = []
    ids: List[str] = []

    
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if isinstance(batch, dict):
            image = batch.get("image") # or batch.get("data")
            batch_labels = batch.get("label")
            batch_ids = batch.get("file_name")
        else:
            image, batch_labels = batch
            batch_ids = None


        if isinstance(image, dict):
            image = image[next(iter(image))]

        if image.ndim == 5:  # (B, T, C, H, W)
            image = image.mean(dim=1)

        input_dtype = adapter.dtype_s2
        if device.type != "cuda" and input_dtype in {torch.float16, torch.bfloat16}:
            input_dtype = torch.float32
        image = image.to(device=device, dtype=input_dtype, non_blocking=True)

        with torch.no_grad():
            backbone = adapter.compute_backbone(image)
            post = adapter.compute_posthead(image)
            projected = adapter.compute_projected(image)

        backbone_vectors.append(_to_numpy(backbone))
        posthead_vectors.append(_to_numpy(post))
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


def _build_eurosat_loaders(config: ModelEvalConfig) -> Dict[str, DataLoader]:
    mean = [MEAN[b] for b in BANDS]
    std = [STD[b] for b in BANDS]

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
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
            bands=BANDS,
            transforms=train_transform if split == "train" else eval_transform,
            download=True,
        )
        for split in ("train", "val", "test")
    }

    loaders = {
        split: DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        for split, dataset in datasets.items()
    }
    return loaders


def _build_neuco_loader(config: ModelEvalConfig) -> DataLoader:
    transform = transforms.Compose([Normalize(), TemporalMean()])
    dataset = E2SChallengeDataset(
        data_path=str(config.neuco_root),
        modalities=list(config.neuco_modalities),
        seasons=config.neuco_seasons,
        concat=True,
        output_file_name=True,
        transform=transform,
    )
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
        s2_tier=str(config.ssl4eo_s2_tier),
        s2_bands=list(config.ssl4eo_s2_bands),
        transforms=None,
        target_image_dimension=(config.ssl4eo_image_dimension, config.ssl4eo_image_dimension),
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
) -> EmbeddingBundle:
    dataset = _build_ssl4eo_dataset(config)

    autocast = (
        (lambda: torch.cuda.amp.autocast(device_type="cuda"))
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

    # s1_bundle = EmbeddingBundle(
    #     backbone=s1_embeddings.raw.cpu().numpy(), # raw post-head
    #     projected=s1_embeddings.normalized.cpu().numpy(), # L2/hyperbolic projection post-head
    #     labels=None,
    #     ids=sample_ids if sample_ids else None,
    # )

    # s2_bundle = EmbeddingBundle(
    #     backbone=s2_embeddings.raw.cpu().numpy(), # raw post-head
    #     projected=s2_embeddings.normalized.cpu().numpy(), # L2/hyperbolic projection post-head
    #     labels=None,
    #     ids=sample_ids if sample_ids else None,
    # )
    return s1_embeddings, s2_embeddings


def _run_linear_probe(
    config: ModelEvalConfig,
    embeddings: Dict[str, EmbeddingBundle],
    *,
    output_dir: Path,
    label: str,
) -> None:
    percents = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    rng = np.random.default_rng(config.random_seed)

    def _evaluate(feature_key: str, zscore_norm: bool) -> Dict[float, Dict[str, float]]:
        results: Dict[float, Dict[str, float]] = {}
        train = embeddings["train"]
        val = embeddings["val"]
        test = embeddings["test"]

        train_feats = getattr(train, feature_key)
        val_feats = getattr(val, feature_key)
        test_feats = getattr(test, feature_key)

        if zscore_norm:
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

    plots_dir = output_dir / "linear_probe"
    plots_dir.mkdir(parents=True, exist_ok=True)

    probe_specs = (
        ("backbone", "backbone", False),
        ("posthead", "posthead", False),
        ("projected", "projected", False),
        ("backbone", "backbone_zscore_norm", True),
        ("posthead", "posthead_zscore_norm", True),
        ("projected", "projected_zscore_norm", True),
    )

    for feature_key, suffix, use_zscore_norm in probe_specs:
        metrics = _evaluate(feature_key, zscore_norm=use_zscore_norm)
        with (plots_dir / f"{label}_{suffix}_metrics.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    marker_map = {"backbone": "o", "posthead": "^", "projected": "s"}

    for feature_key, suffix, use_zscore_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        acc = [metrics[str(x)]["test_accuracy"] for x in xs]
        linestyle = "--" if use_zscore_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "z-score" if use_zscore_norm else "raw"
        ax1.plot(xs, acc, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})")

    for feature_key, suffix, use_zscore_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        f1_vals = [metrics[str(x)]["test_f1"] for x in xs]
        linestyle = "--" if use_zscore_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "z-score" if use_zscore_norm else "raw"
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
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    modality_tensors = {
        "s1": {
            "posthead": s1_bundle.raw.detach(),
            "projected": s1_bundle.projected.detach(),
            "layers": s1_bundle.layer_activations,
        },
        "s2": {
            "posthead": s2_bundle.raw.detach(),
            "projected": s2_bundle.projected.detach(),
            "layers": s2_bundle.layer_activations,
        },
    }

    s1_layers, s1_within = compute_within_encoder_cka(modality_tensors["s1"]["layers"])
    s2_layers, s2_within = compute_within_encoder_cka(modality_tensors["s2"]["layers"])
    cross_s1_layers, cross_s2_layers, cross_matrix = compute_cross_encoder_cka(
        modality_tensors["s1"]["layers"], modality_tensors["s2"]["layers"]
    )
    projected_cka = compute_linear_cka(
        modality_tensors["s1"]["projected"], modality_tensors["s2"]["projected"]
    )

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

    with (output_dir / "cka.json").open("w", encoding="utf-8") as handle:
        json.dump(cka_payload, handle, indent=2)

    spectra = [
        ("s1", "posthead", compute_singular_values(modality_tensors["s1"]["posthead"])),
        ("s1", "projected", compute_singular_values(modality_tensors["s1"]["projected"])),
        ("s2", "posthead", compute_singular_values(modality_tensors["s2"]["posthead"])),
        ("s2", "projected", compute_singular_values(modality_tensors["s2"]["projected"])),
    ]

    for modality, feature_name, spectrum in spectra:
        if spectrum.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(len(spectrum)), spectrum, marker="o")
        ax.set_xlabel("Component")
        ax.set_ylabel("Singular value")
        ax.set_title(f"Singular values ({modality.upper()} {feature_name})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"singular_values_{modality}_{feature_name}.png", dpi=200)
        plt.close(fig)

    rng = np.random.default_rng(config.random_seed)

    def _stack_features(feature_name: str) -> Tuple[np.ndarray, np.ndarray]:
        s1_np = modality_tensors["s1"][feature_name].cpu().numpy()
        s2_np = modality_tensors["s2"][feature_name].cpu().numpy()
        labels = np.concatenate((np.repeat("S1", len(s1_np)), np.repeat("S2", len(s2_np))))
        features = np.concatenate((s1_np, s2_np), axis=0)
        return features, labels

    def _sample(features: np.ndarray, labels: np.ndarray, maximum: int) -> Tuple[np.ndarray, np.ndarray]:
        if maximum <= 0 or maximum >= len(features):
            return features, labels
        indices = rng.choice(len(features), size=maximum, replace=False)
        return features[indices], labels[indices]

    for feature_name in ("posthead", "projected"):
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

    plot_angle_aperture(positive_angles, aperture_s1, aperture_s2, output_dir / "angle_aperture.png")
    plot_radial_histogram(np.linalg.norm(s1_proj_features.cpu().numpy(), axis=1), np.linalg.norm(s2_proj_features.cpu().numpy(), axis=1), output_dir / "radial_histogram.png")
    plot_angular_pca(s1_dirs, s2_dirs, s1_distances, s2_distances, output_dir / "angular_pca.png")
    plot_cone_polar(positive_angles, aperture_s1, aperture_s2, output_dir / "cone_polar.png", sample_size=256, seed=seed)


def run_full_evaluation(config: ModelEvalConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    adapter = build_evaluation_adapter(
        model_type=config.model_type,
        checkpoint=config.checkpoint,
        model_weights=config.model_weights,
        in_chans=config.model_in_channels,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = adapter.to(device)
    adapter.eval()
    base_model = getattr(adapter, "base_model", adapter)
    is_lorentz = getattr(adapter, "is_lorentz", False)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eurosat_loaders = _build_eurosat_loaders(config)
    eurosat_embeddings: Dict[str, EmbeddingBundle] = {}
    for split, loader in eurosat_loaders.items():
        # The backbone features are not normalized (pre-prof)
        # the post-head features are raw euclidean and not normalized
        eurosat_embeddings[split] = _extract_embeddings(
            adapter,
            loader,
            device=device,
        )

    _run_linear_probe(config, eurosat_embeddings, output_dir=output_dir, label="eurosat")

    neuco_loader = _build_neuco_loader(config)
    neuco_bundle = _extract_embeddings(
        adapter,
        neuco_loader,
        device=device,
        require_ids=True,
    )
    _export_neuco(neuco_bundle, output_dir / "neuco_export", label="s2l1c")

    ssl4eo_available = (
        config.enable_ssl4eo
        and config.ssl4eo_root is not None
        and getattr(adapter, "supports_ssl4eo", True)
    )

    s1_ssl4eo = s2_ssl4eo = None
    if ssl4eo_available:
        s1_ssl4eo, s2_ssl4eo = _extract_ssl4eo_embeddings(
            config,
            adapter,
            device=device,
        )

        _run_embedding_diagnostics(
            config,
            s1_ssl4eo,
            s2_ssl4eo,
            output_dir=output_dir / "embedding_diagnostics",
        )
    else:
        logging.info(
            "Skipping SSL4EO diagnostics (enable_ssl4eo=%s, root=%s, supports_ssl4eo=%s)",
            config.enable_ssl4eo,
            config.ssl4eo_root,
            getattr(adapter, "supports_ssl4eo", True),
        )

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


__all__ = ["ModelEvalConfig", "run_full_evaluation"]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the unified CIIP evaluation pipeline.")
    parser.add_argument("--model-type", default="ciip_checkpoint", choices=["ciip_checkpoint", "torchgeo_resnet50"], help="Model source to evaluate.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path for CIIP/Lorentz models.")
    parser.add_argument("--model-weights", choices=["dino", "moco"], help="TorchGeo ResNet50 weight selection.")
    parser.add_argument("--model-in-channels", type=int, default=13, help="Number of input channels for TorchGeo ResNet models.")
    parser.add_argument("--eurosat-root", type=Path, required=True, help="EuroSAT dataset root directory.")
    parser.add_argument("--neuco-root", type=Path, required=True, help="NeuCo-Bench dataset root directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write evaluation artifacts.")
    parser.add_argument("--ssl4eo-root", type=Path, help="SSL4EO dataset root for diagnostics.")
    parser.add_argument("--disable-ssl4eo", action="store_true", help="Skip SSL4EO diagnostics even if a root is provided.")
    parser.add_argument("--tsne-samples", type=int, default=1500, help="Samples used for t-SNE visualisations.")
    parser.add_argument("--pca-samples", type=int, default=5000, help="Samples used for PCA visualisations.")
    parser.add_argument("--ssl4eo-subset-size", type=int, default=2048, help="Subset size for SSL4EO embedding extraction.")
    parser.add_argument("--ssl4eo-subset-seed", type=int, default=0, help="Subset seed for SSL4EO sampling.")
    parser.add_argument("--neuco-modalities", nargs="*", default=["s2l1c"], help="NeuCo modalities to export.")
    parser.add_argument("--neuco-seasons", type=int, default=1, help="Number of seasons for NeuCo extraction.")

    args = parser.parse_args()

    if args.model_type == "ciip_checkpoint" and args.checkpoint is None:
        parser.error("--checkpoint is required when --model-type=ciip_checkpoint")
    if args.model_type == "torchgeo_resnet50" and args.model_weights is None:
        parser.error("--model-weights must be provided for torchgeo_resnet50 models")

    cfg = ModelEvalConfig(
        eurosat_root=args.eurosat_root,
        neuco_root=args.neuco_root,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        model_weights=args.model_weights,
        model_in_channels=args.model_in_channels,
        enable_ssl4eo=not args.disable_ssl4eo,
        ssl4eo_root=args.ssl4eo_root,
        tsne_samples=args.tsne_samples,
        pca_samples=args.pca_samples,
        ssl4eo_subset_size=args.ssl4eo_subset_size,
        ssl4eo_subset_seed=args.ssl4eo_subset_seed,
        neuco_modalities=tuple(args.neuco_modalities),
        neuco_seasons=args.neuco_seasons,
    )
    run_full_evaluation(cfg)


