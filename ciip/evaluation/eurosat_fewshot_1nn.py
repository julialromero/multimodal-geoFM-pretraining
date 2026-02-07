"""EuroSAT 1-shot 5-way evaluation with 1-NN.

This script reuses dataset/model utilities from unified_evaluation to keep
band selection, normalization, and adapter behavior consistent.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT
from torchvision import transforms

from ciip.eval_utils import CustomTransform
from ciip.evaluation.normalization_utils import (
    DEFAULT_NORMALIZATION_METHOD,
    NORMALIZATION_METHODS,
    NORMALIZATION_METHOD_SSL4EO,
    SSL4EONormalize,
    build_normalization_transform,
    resolve_normalization_method,
    select_ssl4eo_transform,
)
from ciip.evaluation.model_utils import build_evaluation_adapter
from ciip.evaluation.output_utils import (
    build_model_tag,
    ensure_dir,
    write_json,
    write_run_manifest,
)
from ciip.evaluation.unified_evaluation import (
    ModelEvalConfig,
    _infer_model_in_channels,
    _resolve_eurosat_bands,
    _use_adapter_modality,
    _extract_embeddings,
)
from ciip.open_clip_train.data import (
    S2L1C_MEAN,
    S2L1C_STD,
    S2L2A_MEAN,
    S2L2A_STD,
    S2RGB_MEAN,
    S2RGB_STD,
)

_SSL4EO_L1C_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
_SSL4EO_L2A_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
_SSL4EO_RGB_BANDS = ("B04", "B03", "B02")
_SSL4EO_BANDWISE_METHOD = "ssl4eobandwisenorm"
_FEWSHOT_NORMALIZATION_METHODS = tuple(NORMALIZATION_METHODS) + (_SSL4EO_BANDWISE_METHOD,)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return features / denom


def _build_class_index(labels: np.ndarray) -> Dict[int, List[int]]:
    class_index: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        class_index.setdefault(int(label), []).append(idx)
    return class_index


def _run_fewshot_1nn(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_way: int,
    k_shot: int,
    queries_per_class: int,
    episodes: int,
    seed: int,
    knn_k: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    class_index = _build_class_index(labels)

    eligible_classes = [
        cls for cls, indices in class_index.items() if len(indices) >= k_shot + queries_per_class
    ]
    if len(eligible_classes) < n_way:
        raise ValueError(
            f"Not enough classes with >= {k_shot + queries_per_class} samples "
            f"for {n_way}-way evaluation (eligible={len(eligible_classes)})."
        )

    max_k = n_way * k_shot
    if knn_k < 1 or knn_k > max_k:
        raise ValueError(f"knn_k must be in [1, {max_k}] for {n_way}-way {k_shot}-shot episodes.")

    episode_accs = []
    for _ in range(episodes):
        chosen_classes = rng.choice(eligible_classes, size=n_way, replace=False)
        support_feats = []
        support_labels = []
        query_feats = []
        query_labels = []

        for class_id, cls in enumerate(chosen_classes):
            indices = rng.choice(
                class_index[int(cls)],
                size=k_shot + queries_per_class,
                replace=False,
            )
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:]

            support_feats.append(features[support_idx])
            support_labels.extend([class_id] * k_shot)
            query_feats.append(features[query_idx])
            query_labels.extend([class_id] * queries_per_class)

        support_feats = np.concatenate(support_feats, axis=0)
        query_feats = np.concatenate(query_feats, axis=0)
        support_labels = np.asarray(support_labels, dtype=np.int64)
        query_labels = np.asarray(query_labels, dtype=np.int64)

        support_feats = _l2_normalize(support_feats)
        query_feats = _l2_normalize(query_feats)

        sims = query_feats @ support_feats.T
        if knn_k == 1:
            nn_indices = np.argmax(sims, axis=1)
            preds = support_labels[nn_indices]
        else:
            topk = np.argpartition(-sims, knn_k - 1, axis=1)[:, :knn_k]
            preds = []
            for row, idxs in enumerate(topk):
                row_labels = support_labels[idxs]
                row_scores = sims[row, idxs]
                cls_scores = np.bincount(row_labels, weights=row_scores, minlength=n_way)
                preds.append(int(np.argmax(cls_scores)))
            preds = np.asarray(preds, dtype=np.int64)
        episode_accs.append(float(np.mean(preds == query_labels)))

    accs = np.asarray(episode_accs, dtype=np.float64)
    mean = float(accs.mean())
    std = float(accs.std(ddof=1)) if accs.size > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(accs.size)) if accs.size > 1 else 0.0
    return {
        "accuracy_mean": mean,
        "accuracy_std": std,
        "accuracy_ci95": ci95,
    }


def _resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    if args.model_type != "ciip_checkpoint":
        return None
    if not args.model_path:
        raise ValueError("--model-path is required for ciip_checkpoint without --checkpoint")
    model_root = Path(args.model_root)
    checkpoint_root = model_root / args.model_path / "checkpoints"
    return checkpoint_root / f"epoch_{args.ciip_epoch}.pt"


def _build_output_dir(args: argparse.Namespace) -> Path:
    return ensure_dir(Path(args.output_dir))


def _ssl4eo_band_stats(bands: Sequence[str]) -> Tuple[List[float], List[float], str]:
    if tuple(bands) == _SSL4EO_RGB_BANDS:
        return list(S2RGB_MEAN), list(S2RGB_STD), "rgb"

    uses_b10 = "B10" in bands
    base_bands = _SSL4EO_L1C_BANDS if uses_b10 else _SSL4EO_L2A_BANDS
    base_mean = S2L1C_MEAN if uses_b10 else S2L2A_MEAN
    base_std = S2L1C_STD if uses_b10 else S2L2A_STD
    tier = "s2l1c" if uses_b10 else "s2l2a"

    mean_map = dict(zip(base_bands, base_mean))
    std_map = dict(zip(base_bands, base_std))
    mean = []
    std = []
    for band in bands:
        if band not in mean_map or band not in std_map:
            raise KeyError(f"No SSL4EO stats available for band '{band}' in tier {tier}.")
        mean.append(mean_map[band])
        std.append(std_map[band])
    return mean, std, tier


def _build_eurosat_loaders(
    config: ModelEvalConfig,
    *,
    bands: Sequence[str],
    model_transform: Optional[torch.nn.Module],
    transform_label: Optional[str],
) -> Tuple[Dict[str, DataLoader], str]:
    target_size = int(config.eurosat_image_size)
    eval_resize = max(target_size, int(round(target_size * 256 / 224)))
    normalization_method = resolve_normalization_method(
        config.model_in_channels, config.normalization_method
    )

    if model_transform is not None:
        norm_label = transform_label or f"ssl4eo_transform:{config.model_weights}"
        norm_layer = model_transform
    # elif normalization_method in {"bandwisenorm", _SSL4EO_BANDWISE_METHOD}:
    #     mean, std, tier = _ssl4eo_band_stats(bands)
    #     norm_layer = transforms.Normalize(mean=mean, std=std)
    #     base_label = "bandwisenorm" if normalization_method == "bandwisenorm" else _SSL4EO_BANDWISE_METHOD
    #     norm_label = f"{base_label}_ssl4eo_{tier}"
    elif normalization_method == NORMALIZATION_METHOD_SSL4EO:
        print("Using SSL4EO global normalization for EuroSAT.")
        # build normalize with stats matching inchannels
        # if len(bands) == 13:
        #         norm_layer = SSL4EONormalize()
        # if set(bands) == set(_SSL4EO_L2A_BANDS):
        #     print("Using SSL4EO L2A stats for normalization.")
        #     norm_layer = SSL4EONormalize(mean=list(S2L2A_MEAN), std=list(S2L2A_STD))
        # elif set(bands) == set(_SSL4EO_L1C_BANDS):
        #     print("Using SSL4EO L1C stats for normalization.")
        #     norm_layer = SSL4EONormalize(mean=list(S2L1C_MEAN), std=list(S2L1C_STD))
        # elif set(bands) == set(_SSL4EO_RGB_BANDS):
        #     print("Using SSL4EO RGB stats for normalization.")
        #     norm_layer = SSL4EONormalize(mean=list(S2RGB_MEAN), std=list(S2RGB_STD))
        # elif len(bands) == 10:
        #     norm_layer = SSL4EONormalize()

        norm_layer = SSL4EONormalize()
        norm_label = normalization_method
    else:
        print(f"Using default Divideby10000 normalization for EuroSAT (normalization_method='{normalization_method}').")
        norm_layer = build_normalization_transform(normalization_method)
        norm_label = normalization_method

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
    for split, dataset in datasets.items():
        assert len(dataset) > 0, f"EuroSAT {split} dataset is empty!"
    return loaders, norm_label


def run_from_args(args: argparse.Namespace) -> Path:
    if args.model_in_channels is None:
        if args.model_weights == "remoteclip":
            args.model_in_channels = 3
        elif args.model_weights == "llama3_ms_clip_base":
            args.model_in_channels = 10
        elif args.model_weights and "rgb" in args.model_weights:
            args.model_in_channels = 3

    if args.model_type == "ciip_checkpoint" and args.checkpoint is None and not args.model_path:
        raise ValueError("--model-path or --checkpoint is required for ciip_checkpoint.")
    if args.model_type in {"torchgeo_resnet50", "backbone_only"} and args.model_weights is None:
        raise ValueError("--model-weights must be provided for the selected model type.")
    if args.model_type == "croma" and args.croma_weights is None:
        raise ValueError("--croma-weights must be provided when --model-type=croma.")

    checkpoint = _resolve_checkpoint(args)

    cfg = ModelEvalConfig(
        eurosat_root=Path(args.eurosat_root),
        neuco_root=Path("/local/ms-data/SSL4EO-S12-downstream/data"),
        output_dir=_build_output_dir(args),
        checkpoint=checkpoint,
        model_type=args.model_type,
        model_weights=args.model_weights,
        ciip_framework=args.ciip_framework,
        model_in_channels=args.model_in_channels if args.model_in_channels is not None else 13,
        croma_weights=args.croma_weights,
        croma_image_resolution=args.croma_image_resolution,
        evaluation_modality="s2",
        eurosat_image_size=args.eurosat_image_size,
        normalization_method=args.normalization_method,
        random_seed=args.seed,
    )

    if cfg.model_type == "croma":
        cfg.eurosat_image_size = cfg.croma_image_resolution

    adapter = build_evaluation_adapter(
        model_type=cfg.model_type,
        checkpoint=cfg.checkpoint,
        model_weights=cfg.model_weights,
        in_chans=cfg.model_in_channels,
        croma_weights=cfg.croma_weights,
        croma_image_resolution=cfg.croma_image_resolution,
        ciip_framework=cfg.ciip_framework,
        enable_s1=not (cfg.model_type == "torchgeo_resnet50" and (cfg.model_weights or "").lower() == "moco"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = adapter.to(device)
    adapter.eval()

    model_channels = _infer_model_in_channels(adapter, cfg.model_in_channels, modality="s2")
    thirteen_band_models = {
        "dofa_base_s2_13ch",
        "vitsmall16_s2_all_moco",
        "resnet18_s2_all_moco",
        "moco",
        "dino",
    }
    if (cfg.model_weights and cfg.model_weights in thirteen_band_models) or (
        cfg.model_type and cfg.model_type in thirteen_band_models
    ):
        model_channels = max(model_channels, 13)

    eurosat_bands = _resolve_eurosat_bands(model_channels)
    print(f"Evaluating model with {model_channels} input channels, using EuroSAT bands: {eurosat_bands}")
    model_transform = None
    transform_label = None
    if args.model_type != "ciip_checkpoint":
        model_transform = select_ssl4eo_transform(args.model_weights)
        if model_transform is not None:
            transform_label = f"ssl4eo_transform:{args.model_weights}"
            print(f"Using non-CIIP transform: {model_transform}")
        else:
            model_transform = select_ssl4eo_transform(args.model_type)
            transform_label = f"ssl4eo_transform:{args.model_type}"
            print(f"Using non-CIIP transform: {model_transform}")
        if model_transform is None:
            raise ValueError(f"No predefined transform found for model_weights='{args.model_weights}'.")
    eurosat_loaders, norm_label = _build_eurosat_loaders(
        cfg,
        bands=eurosat_bands,
        model_transform=model_transform,
        transform_label=transform_label,
    )

    with _use_adapter_modality(adapter, "s2"):
        test_embeddings = _extract_embeddings(
            adapter,
            eurosat_loaders["test"],
            device=device,
            expected_in_channels=model_channels,
            modality="s2",
        )

    if test_embeddings.backbone is not None:
        print(f"Backbone embedding dim: {test_embeddings.backbone.shape[1]}")

    features = getattr(test_embeddings, args.feature)
    if features is None:
        raise ValueError(f"Requested feature '{args.feature}' is not available.")
    labels = test_embeddings.labels
    if labels is None:
        raise ValueError("No labels found for EuroSAT embeddings.")

    print(f"Using feature '{args.feature}' dim: {features.shape[1]}")

    metrics = _run_fewshot_1nn(
        features,
        labels,
        n_way=args.n_way,
        k_shot=args.k_shot,
        queries_per_class=args.queries_per_class,
        episodes=args.episodes,
        seed=args.seed,
        knn_k=args.knn_k,
    )

    results = {
        "dataset": "EuroSAT",
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "queries_per_class": args.queries_per_class,
        "episodes": args.episodes,
        "knn_k": args.knn_k,
        "feature": args.feature,
        "feature_dim": int(features.shape[1]),
        "backbone_dim": int(test_embeddings.backbone.shape[1]) if test_embeddings.backbone is not None else None,
        "normalization_method": norm_label,
        "preprocess_transform": transform_label,
        "model_type": args.model_type,
        "model_weights": args.model_weights,
        "model_path": args.model_path,
        "ciip_epoch": args.ciip_epoch,
        "bands": list(eurosat_bands),
        "device": str(device),
        "seed": args.seed,
    }
    results.update(metrics)

    model_tag = build_model_tag(
        model_type=args.model_type,
        model_weights=args.model_weights,
        model_path=args.model_path,
        ciip_epoch=args.ciip_epoch,
    )
    out_dir = ensure_dir(cfg.output_dir / "eurosat_fewshot" / model_tag)
    write_run_manifest(
        out_dir,
        task_name="eurosat_fewshot_1nn",
        config={
            "model_type": args.model_type,
            "model_weights": args.model_weights,
            "model_path": args.model_path,
            "ciip_epoch": args.ciip_epoch,
            "model_in_channels": args.model_in_channels,
            "feature": args.feature,
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "knn_k": args.knn_k,
            "queries_per_class": args.queries_per_class,
            "episodes": args.episodes,
            "seed": args.seed,
            "normalization_method": args.normalization_method,
            "eurosat_root": args.eurosat_root,
            "eurosat_image_size": args.eurosat_image_size,
        },
    )
    out_path = write_json(out_dir / "results.json", results)

    print(f"{args.knn_k}-NN {args.n_way}-way {args.k_shot}-shot accuracy: {metrics['accuracy_mean']:.4f}")
    print(f"95% CI: ±{metrics['accuracy_ci95']:.4f} (std={metrics['accuracy_std']:.4f})")
    print(f"Saved results to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EuroSAT 1-shot 5-way evaluation with 1-NN using unified_evaluation utilities."
    )
    parser.add_argument(
        "--model-type",
        default="ciip_checkpoint",
        choices=["ciip_checkpoint", "torchgeo_resnet50", "croma", "backbone_only"],
        help="Model source to evaluate.",
    )
    parser.add_argument(
        "--model-weights",
        choices=[
            "dino",
            "moco",
            "rcf_13ch",
            "dofa_base_s2_13ch",
            "scalemae_large_rgb",
            "resnet18_s2_all_moco",
            "resnet18_s2_rgb_moco",
            "resnet50_s2_rgb_moco",
            "resnet152_imagenet_rgb",
            "vitsmall16_s2_all_moco",
            "llama3_ms_clip_base",
            "remoteclip",
        ],
        help="Pretrained weight selection for the requested model type.",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a CIIP checkpoint.")
    parser.add_argument("--model-root", type=str, default="/local/ms-data/SSL4EO/model/", help="Root for CIIP checkpoints.")
    parser.add_argument("--model-path", type=str, help="Experiment path identifier for the model.")
    parser.add_argument("--ciip-epoch", type=int, default=10, help="Epoch number for CIIP checkpoint to evaluate.")
    parser.add_argument(
        "--ciip-framework",
        choices=["modified_resnet", "transformer", "resnet18", "resnet50"],
        help="Backbone framework for CIIP checkpoints (defaults to auto-detect).",
    )
    parser.add_argument("--model-in-channels", type=int, help="Number of input channels for the model.")
    parser.add_argument("--croma-weights", type=Path, help="Path to the pretrained CROMA weights.")
    parser.add_argument("--croma-image-resolution", type=int, default=120, help="Input resolution expected by the CROMA model.")
    parser.add_argument("--eurosat-root", type=Path, default=Path("/local/ms-data/EuroSAT/"), help="Root directory for EuroSAT data.")
    parser.add_argument("--eurosat-image-size", type=int, default=224, help="EuroSAT input resolution.")
    parser.add_argument(
        "--normalization-method",
        choices=_FEWSHOT_NORMALIZATION_METHODS,
        default=DEFAULT_NORMALIZATION_METHOD,
        help="Normalization applied to Sentinel-2 / optical inputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/fewshot_eval"), help="Output directory.")
    parser.add_argument("--feature", choices=["backbone", "projected"], default="backbone", help="Embedding space to use.")
    parser.add_argument("--n-way", type=int, default=5, help="Number of classes per episode.")
    parser.add_argument("--k-shot", type=int, default=1, help="Number of support samples per class.")
    parser.add_argument("--knn-k", type=int, default=1, help="Number of neighbors for k-NN classification.")
    parser.add_argument("--queries-per-class", type=int, default=15, help="Number of query samples per class.")
    parser.add_argument("--episodes", type=int, default=600, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode sampling.")
    args = parser.parse_args()

    run_from_args(args)


if __name__ == "__main__":
    main()
