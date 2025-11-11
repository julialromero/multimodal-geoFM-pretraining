"""Concise end-to-end evaluation entrypoint.

This module provides a single callable ``run_full_evaluation`` that executes the
three evaluation pipelines requested by the research team plus the optional
Lorentz-specific visualisations.  The implementation intentionally reuses the
utilities that already exist across the codebase (``linearprobe_comparison``,
``export_neuco_embeddings`` and ``visualizations/ssl4eo``) instead of invoking
their CLI entrypoints so that everything can be orchestrated from Python.

The function expects a ``ModelEvalConfig`` describing the checkpoint, dataset
locations and output directory.  The checkpoint is inspected to decide whether
the CIIP or LorentzCIIP architecture should be instantiated and to infer the
projection dimensionality so no manual knobs are required when switching
between the two model families.
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
import torch.nn.functional as F
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
from ciip.model_ciip import CIIP, LorentzCIIP

from ciip.evaluation.linearprobe_comparison import (  # type: ignore
    BANDS,
    MEAN,
    STD,
)
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
    checkpoint: Path
    eurosat_root: Path
    neuco_root: Path
    output_dir: Path
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
    post_head_raw: np.ndarray
    post_head_proj: np.ndarray
    labels: Optional[np.ndarray] = None
    ids: Optional[List[str]] = None


def _clean_state_dict(raw_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key.replace("module.", ""): value
        for key, value in raw_state.items()
    }


def _infer_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    if "encoder_s2.proj.weight" in state_dict:
        weight = state_dict["encoder_s2.proj.weight"]
        return int(weight.shape[0]), int(weight.shape[1])
    if "encoder_s2.fc.weight" in state_dict:
        weight = state_dict["encoder_s2.fc.weight"]
        return int(weight.shape[0]), int(weight.shape[0])
    raise RuntimeError("Unable to infer embedding dimensions from checkpoint")


def _build_model(checkpoint: Path) -> Tuple[nn.Module, bool]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    cleaned = _clean_state_dict(state_dict)
    embed_dim, pre_dim = _infer_dims(cleaned)
    is_lorentz = any(key.startswith("curv") or "lorentz" in key for key in cleaned)

    kwargs = dict(
        embed_dim=embed_dim,
        pre_projection_dim=pre_dim,
        s1_resolution=224,
        s1_layers=(3, 4, 6, 3),
        s1_width=32,
        s1_patch_size=16,
        s1_bands=2,
        s2_resolution=224,
        s2_layers=(3, 4, 6, 3),
        s2_width=32,
        s2_patch_size=16,
        s2_bands=13,
        framework="resnet50",
    )

    if is_lorentz:
        model: nn.Module = LorentzCIIP(**kwargs)
    else:
        model = CIIP(**kwargs)

    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    # allowed_missing = {
    #     "encoder_s1.fc.weight",
    #     "encoder_s1.fc.bias",
    #     "encoder_s2.fc.weight",
    #     "encoder_s2.fc.bias",
    # }
    # remaining_missing = {key for key in missing if key not in allowed_missing}
    # if remaining_missing or unexpected:
    #     raise RuntimeError(
    #         f"Checkpoint incompatible with model (missing={remaining_missing}, unexpected={unexpected})"
    #     )
    if missing or unexpected:
        logging.warning(
            f"Checkpoint loaded with missing={missing}, unexpected={unexpected}"
        )
        raise RuntimeError(
            f"Checkpoint incompatible with model (missing={missing}, unexpected={unexpected})"
        )
    model.eval()
    return model, is_lorentz


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


def _flatten_features(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 2:
        return tensor.flatten(start_dim=1)
    return tensor

### FOR EUROSAT
def _extract_embeddings_eurosat(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    is_lorentz: bool,
    require_ids: bool = False,
) -> EmbeddingBundle:
    backbone_vectors: List[np.ndarray] = []
    projected_vectors: List[np.ndarray] = []
    labels: List[int] = []
    ids: List[str] = []

    
    model.eval()

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

        image = image.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            pre = model.encoder_s2(image.type(model.dtype_s2))  
            pre = _flatten_features(pre)

            if is_lorentz:
                post = model.encode_s2(image, lorentz=False)
            else:
                post = model.encode_s2(image, normalize=False)
                # post = F.normalize(post, dim=-1)
            post = _flatten_features(post)

        backbone_vectors.append(_to_numpy(pre))
        projected_vectors.append(_to_numpy(post))

        if batch_labels is not None:
            labels.extend(batch_labels.cpu().tolist())
        if require_ids and batch_ids is not None:
            ids.extend([str(item) for item in batch_ids])

    bundle = EmbeddingBundle(
        backbone=np.concatenate(backbone_vectors, axis=0),
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

    bundle = EmbeddingBundle(
        backbone=s2_embeddings.raw.cpu().numpy(),
        projected=s2_embeddings.normalized.cpu().numpy(),
        labels=None,
        ids=sample_ids if sample_ids else None,
    )
    return bundle


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
        train = embeddings["train"]
        val = embeddings["val"]
        test = embeddings["test"]

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

    plots_dir = output_dir / "linear_probe"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for key, pretty in (("backbone", "backbone"), ("projected", "projected"),
                         ('backbone', 'batch-norm'), ('projected', 'batch-norm')):
        metrics = _evaluate(key, batch_norm=(pretty == 'batch-norm'))
        with (plots_dir / f"{label}_{key}_metrics.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)

        # Create single combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot accuracy on left subplot
    for key, pretty in (("backbone", "backbone"), ("projected", "projected"),
                        ('backbone', 'batch-norm'), ('projected', 'batch-norm')):
        metrics_file = plots_dir / f"{label}_{key}_metrics.json"
        with metrics_file.open("r") as handle:
            metrics = json.load(handle)
        
        xs = sorted([float(x) for x in metrics.keys()])
        acc = [metrics[str(x)]["test_accuracy"] for x in xs]
        
        linestyle = '--' if 'batch-norm' in pretty else '-'
        marker = 's' if 'projected' in key else 'o'
        ax1.plot(xs, acc, marker=marker, linestyle=linestyle, label=f"{key} ({pretty})")

    # Plot F1 on right subplot
    for key, pretty in (("backbone", "backbone"), ("projected", "projected"),
                        ('backbone', 'batch-norm'), ('projected', 'batch-norm')):
        metrics_file = plots_dir / f"{label}_{key}_metrics.json"
        with metrics_file.open("r") as handle:
            metrics = json.load(handle)
        
        xs = sorted([float(x) for x in metrics.keys()])
        f1_vals = [metrics[str(x)]["test_f1"] for x in xs]
        
        linestyle = '--' if 'batch-norm' in pretty else '-'
        marker = 's' if 'projected' in key else 'o'
        ax2.plot(xs, f1_vals, marker=marker, linestyle=linestyle, label=f"{key} ({pretty})")

        # Configure subplots
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

        # print the save path
        logging.info(f"Linear probe results saved to {plots_dir}")

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
    _write(bundle.backbone, "backbone")
    _write(bundle.projected, "projected")


def _run_embedding_diagnostics(
    config: ModelEvalConfig,
    bundle: EmbeddingBundle,
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_backbone = torch.from_numpy(bundle.backbone)
    torch_projected = torch.from_numpy(bundle.projected)

    cka_value = compute_linear_cka(torch_backbone, torch_projected)
    with (output_dir / "cka.json").open("w") as handle:
        json.dump({"linear_cka": cka_value}, handle, indent=2)

    singular_backbone = compute_singular_values(torch_backbone)
    singular_projected = compute_singular_values(torch_projected)

    for spectrum, name in ((singular_backbone, "backbone"), (singular_projected, "projected")):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(len(spectrum)), spectrum, marker="o")
        ax.set_xlabel("Component")
        ax.set_ylabel("Singular value")
        ax.set_title(f"Singular values ({name})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"singular_values_{name}.png", dpi=200)
        plt.close(fig)

    rng = np.random.default_rng(config.random_seed)

    def _sample(features: np.ndarray, labels: Optional[np.ndarray], maximum: int) -> Tuple[np.ndarray, np.ndarray]:
        if labels is None:
            labels = np.zeros(len(features), dtype=int)
        maximum = min(maximum, len(features))
        indices = rng.choice(len(features), size=maximum, replace=False)
        return features[indices], labels[indices]

    for features, label, mode in (
        (bundle.backbone, "backbone", "zscore"),
        (bundle.projected, "projected", "l2"),
    ):
        subset, subset_labels = _sample(features, bundle.labels, config.tsne_samples)
        processed = preprocess_projection_data(subset, mode=mode, random_state=config.random_seed)
        coords = compute_projection(processed, method="tsne", random_state=config.random_seed)
        if coords is not None:
            plot_projection(coords, subset_labels, output_dir / f"tsne_{label}.png", title=f"t-SNE ({label})")

        subset, subset_labels = _sample(features, bundle.labels, config.pca_samples)
        processed = preprocess_projection_data(subset, mode=mode, random_state=config.random_seed)
        if processed.shape[0] >= 2:
            pca_coords = PCA(n_components=2, random_state=config.random_seed).fit_transform(processed)
            plot_projection(pca_coords, subset_labels, output_dir / f"pca_{label}.png", title=f"PCA ({label})")


def _run_hyperbolic_visualisations(
    post_features: torch.Tensor,
    *,
    output_dir: Path,
    model: LorentzCIIP,
    aperture_logk: Optional[float],
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = compute_hyperbolic_context(
        model,
        post_features,
        post_features,
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
    plot_radial_histogram(np.linalg.norm(post_features.cpu().numpy(), axis=1), np.linalg.norm(post_features.cpu().numpy(), axis=1), output_dir / "radial_histogram.png")
    plot_angular_pca(s1_dirs, s2_dirs, s1_distances, s2_distances, output_dir / "angular_pca.png")
    plot_cone_polar(positive_angles, aperture_s1, aperture_s2, output_dir / "cone_polar.png", sample_size=256, seed=seed)


def run_full_evaluation(config: ModelEvalConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    model, is_lorentz = _build_model(config.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eurosat_loaders = _build_eurosat_loaders(config)
    eurosat_embeddings: Dict[str, EmbeddingBundle] = {}
    for split, loader in eurosat_loaders.items():
        # The backbone features are not normalized (pre-prof)
        # the post-head features are raw euclidean and not normalized
        eurosat_embeddings[split] = _extract_embeddings_eurosat(
            model,
            loader,
            device=device,
            is_lorentz=is_lorentz,
        )

    _run_linear_probe(config, eurosat_embeddings, output_dir=output_dir, label="eurosat")

    # neuco_loader = _build_neuco_loader(config)
    # neuco_bundle = _extract_embeddings(
    #     model,
    #     neuco_loader,
    #     device=device,
    #     is_lorentz=is_lorentz,
    #     require_ids=True,
    # )
    # _export_neuco(neuco_bundle, output_dir / "neuco_export", label="s2l1c")

    ssl4eo_bundle = _extract_ssl4eo_embeddings(
        config,
        model,
        device=device,
    )

    _run_embedding_diagnostics(
        config,
        ssl4eo_bundle,
        output_dir=output_dir / "embedding_diagnostics",
    )

    if is_lorentz:
        if not isinstance(model, LorentzCIIP):
            raise TypeError("Expected LorentzCIIP model when is_lorentz is True")
        post_feats = torch.from_numpy(eurosat_embeddings["train"].projected).to(device=device, dtype=torch.float32)
        _run_hyperbolic_visualisations(
            post_feats,
            output_dir=output_dir / "hyperbolic",
            model=model,
            aperture_logk=None,
            seed=config.random_seed,
        )


__all__ = ["ModelEvalConfig", "run_full_evaluation"]



if __name__ == "__main__":
    from pathlib import Path
    from ciip.evaluation.unified_evaluation import ModelEvalConfig, run_full_evaluation

    model_root = '/local/ms-data/SSL4EO/model/'
    model_path = '2025_11_05-21_04_44-model_resnet50-lr_0.001-b_128-j_6-p_amp/'
    checkpoint_root = Path(model_root) / model_path / "checkpoints"
    output_dir = Path("diagnostics/output")

    cfg = ModelEvalConfig(
        checkpoint=Path(f"{checkpoint_root}/epoch_10.pt"),
        eurosat_root=Path("/local/ms-data/EuroSAT/"),
        neuco_root=Path("/local/ms-data/SSL4EO-S12-downstream/data"),
        output_dir=Path("/home/juro4948/ciip/diagnostics/unified_eval/curv_init_1"),
        ssl4eo_root=Path("/local/ms-data/SSL4EO/"),
    )
    run_full_evaluation(cfg)


