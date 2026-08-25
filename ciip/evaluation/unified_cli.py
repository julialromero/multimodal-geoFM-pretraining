"""Command-line interface for the unified evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ciip.evaluation.normalization_contract import (
    DEFAULT_NORMALIZATION_METHOD,
    NORMALIZATION_METHODS,
)
from ciip.evaluation.unified_types import ModelEvalConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eurosat-root", type=Path, required=True)
    parser.add_argument("--neuco-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--model-type",
        default="ciip_checkpoint",
        choices=("ciip_checkpoint", "torchgeo_resnet50", "croma", "backbone_only"),
    )
    parser.add_argument("--model-weights")
    parser.add_argument("--model-path")
    parser.add_argument("--ciip-framework")
    parser.add_argument("--model-in-channels", type=int, default=13)
    parser.add_argument("--evaluation-modality", choices=("s1", "s2"), default="s2")
    parser.add_argument("--croma-weights", type=Path)
    parser.add_argument("--croma-image-resolution", type=int, default=120)
    parser.add_argument("--ssl4eo-root", type=Path)
    parser.add_argument("--disable-ssl4eo", action="store_true")
    parser.add_argument("--disable-eurosat", action="store_true")
    parser.add_argument("--disable-bigearthnet", action="store_true")
    parser.add_argument("--disable-neuco", action="store_true")
    parser.add_argument("--ssl4eo-subset-size", type=int, default=2048)
    parser.add_argument("--ssl4eo-subset-seed", type=int, default=0)
    parser.add_argument("--ssl4eo-s2-tier", default="s2c")
    parser.add_argument("--neuco-modalities", nargs="+", default=["s2l1c"])
    parser.add_argument("--neuco-seasons", type=int, default=1)
    parser.add_argument("--tsne-samples", type=int, default=1500)
    parser.add_argument("--pca-samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--eurosat-image-size", type=int, default=224)
    parser.add_argument("--bigearthnet-root", type=Path)
    parser.add_argument("--bigearthnet-image-size", type=int, default=224)
    parser.add_argument(
        "--normalization-method",
        choices=NORMALIZATION_METHODS,
        default=DEFAULT_NORMALIZATION_METHOD,
    )
    parser.add_argument("--matryoshka-dims", type=int, nargs="+")
    parser.add_argument("--matryoshka-feature", choices=("backbone",), default="backbone")
    parser.add_argument("--stats-max-batches", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> ModelEvalConfig:
    if args.model_type == "ciip_checkpoint" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for ciip_checkpoint")
    if args.model_type in {"torchgeo_resnet50", "backbone_only"} and not args.model_weights:
        raise ValueError("--model-weights is required for the selected model type")
    if args.model_type == "croma" and args.croma_weights is None:
        raise ValueError("--croma-weights is required for croma")
    return ModelEvalConfig(
        eurosat_root=args.eurosat_root,
        neuco_root=args.neuco_root,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        model_weights=args.model_weights,
        model_path=args.model_path,
        ciip_framework=args.ciip_framework,
        model_in_channels=args.model_in_channels,
        evaluation_modality=args.evaluation_modality,
        croma_weights=args.croma_weights,
        croma_image_resolution=args.croma_image_resolution,
        enable_ssl4eo=not args.disable_ssl4eo,
        enable_eurosat=not args.disable_eurosat,
        enable_bigearthnet=not args.disable_bigearthnet,
        enable_neuco=not args.disable_neuco,
        ssl4eo_root=args.ssl4eo_root,
        ssl4eo_subset_size=args.ssl4eo_subset_size,
        ssl4eo_subset_seed=args.ssl4eo_subset_seed,
        ssl4eo_s2_tier=args.ssl4eo_s2_tier,
        neuco_modalities=tuple(args.neuco_modalities),
        neuco_seasons=args.neuco_seasons,
        tsne_samples=args.tsne_samples,
        pca_samples=args.pca_samples,
        random_seed=args.random_seed,
        eurosat_image_size=args.eurosat_image_size,
        bigearthnet_root=args.bigearthnet_root,
        bigearthnet_image_size=args.bigearthnet_image_size,
        normalization_method=args.normalization_method,
        matryoshka_dims=tuple(args.matryoshka_dims) if args.matryoshka_dims else None,
        matryoshka_feature=args.matryoshka_feature,
        stats_max_batches=args.stats_max_batches,
    )


def main() -> None:
    from ciip.evaluation.unified_evaluation import run_full_evaluation

    args = build_parser().parse_args()
    run_full_evaluation(config_from_args(args))


if __name__ == "__main__":
    main()
