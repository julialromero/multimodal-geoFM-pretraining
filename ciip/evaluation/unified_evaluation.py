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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
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
    compute_linear_cka,
    compute_projection,
    compute_singular_values,
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


@dataclass
class EmbeddingBundle:
    backbone: np.ndarray
    projected: np.ndarray
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

    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as exc:  # pragma: no cover - informative error path
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(cleaned.keys())
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        fc_keys = [
            "encoder_s1.fc.weight",
            "encoder_s1.fc.bias",
            "encoder_s2.fc.weight",
            "encoder_s2.fc.bias",
        ]
        missing_fc = [key for key in fc_keys if key in missing]
        if missing_fc:
            raise RuntimeError(
                f"Checkpoint missing required FC weights: {missing_fc}"
            ) from exc
        raise RuntimeError(
            f"Checkpoint incompatible with model (missing={missing}, unexpected={unexpected})"
        ) from exc
    model.eval()
    return model, is_lorentz


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


def _flatten_features(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 2:
        return tensor.flatten(start_dim=1)
    return tensor


def _extract_embeddings(
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
            # print(batch)
            image = batch.get("image") # or batch.get("data")
            batch_labels = batch.get("label")
            batch_ids = batch.get("file_name")
        else:
            image, batch_labels = batch
            batch_ids = None

        if image is None:
            continue

        if isinstance(image, dict):
            image = image[next(iter(image))]

        if image.ndim == 5:  # (B, T, C, H, W)
            image = image.mean(dim=1)

        image = image.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            pre = model.encoder_s2(image.type(model.dtype_s2))  # type: ignore[attr-defined]
            pre = _flatten_features(pre)

            if is_lorentz:
                post = model.encode_s2(image, lorentz=True)
            else:
                post = model.encode_s2(image, normalize=False)
                post = F.normalize(post, dim=-1)
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


def _run_linear_probe(
    config: ModelEvalConfig,
    embeddings: Dict[str, EmbeddingBundle],
    *,
    output_dir: Path,
    label: str,
) -> None:
    percents = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    rng = np.random.default_rng(config.random_seed)

    def _evaluate(feature_key: str) -> Dict[float, Dict[str, float]]:
        results: Dict[float, Dict[str, float]] = {}
        train = embeddings["train"]
        val = embeddings["val"]
        test = embeddings["test"]

        if train.labels is None or val.labels is None or test.labels is None:
            raise RuntimeError("Linear probe requires label annotations for all splits")

        train_feats = getattr(train, feature_key)
        val_feats = getattr(val, feature_key)
        test_feats = getattr(test, feature_key)

        for pct in percents:
            n_samples = max(1, int(pct * len(train_feats)))
            indices = rng.choice(len(train_feats), size=n_samples, replace=False)

            clf = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs")
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

    for key, pretty in (("backbone", "backbone"), ("projected", "projected")):
        metrics = _evaluate(key)
        with (plots_dir / f"{label}_{key}_metrics.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)

        xs = sorted(metrics)
        acc = [metrics[x]["test_accuracy"] for x in xs]
        f1_vals = [metrics[x]["test_f1"] for x in xs]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, acc, marker="o", label="test accuracy")
        ax.plot(xs, f1_vals, marker="s", label="test F1")
        ax.set_xlabel("Training fraction")
        ax.set_ylabel("Score")
        ax.set_title(f"Linear probe ({label}, {pretty})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{label}_{key}_curves.png", dpi=200)
        plt.close(fig)

    # print the save path
    logging.info(f"Linear probe results saved to {plots_dir}")


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
    loss,
    aperture_logk: Optional[float],
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = compute_hyperbolic_context(
        loss,
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
        eurosat_embeddings[split] = _extract_embeddings(
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

    _run_embedding_diagnostics(
        config,
        eurosat_embeddings["train"],
        output_dir=output_dir / "embedding_diagnostics",
    )

    if is_lorentz:
        from ciip.loss import CiipLoss  # Imported lazily to avoid extra dependency for CIIP

        loss = CiipLoss(hyperbolic=True)
        loss = loss.to(device)
        post_feats = torch.from_numpy(eurosat_embeddings["train"].projected).to(device=device, dtype=torch.float32)
        _run_hyperbolic_visualisations(
            post_feats,
            output_dir=output_dir / "hyperbolic",
            loss=loss,
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
        output_dir=Path("/home/juro4948/ciip/diagnostics/curv_init_1")
    )
    run_full_evaluation(cfg)


