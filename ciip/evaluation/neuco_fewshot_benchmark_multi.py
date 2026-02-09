"""Run NeuCo limited-label evaluation across multiple models."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from ciip.evaluation.neuco_fewshot_episodic import run_from_args


def _load_model_specs(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        models = data.get("models")
        defaults = data.get("defaults", {})
        if not isinstance(models, list):
            raise ValueError("Expected 'models' to be a list in the JSON file.")
        return models, defaults if isinstance(defaults, dict) else {}
    if isinstance(data, list):
        return data, {}
    raise ValueError("Models JSON must be a list or an object with a 'models' list.")


def _merge_args(base: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        setattr(merged, key, value)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NeuCo limited-label evaluation across multiple models."
    )
    parser.add_argument("--models-json", type=Path, required=True, help="JSON file listing models to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/neuco_fewshot"), help="Output directory.")
    parser.add_argument("--neuco-root", type=Path, default=Path("/local/ms-data/SSL4EO-S12-downstream/data"), help="NeuCo root.")
    parser.add_argument("--neuco-modalities", nargs="*", default=["s2l2a"], help="NeuCo modalities to export.")
    parser.add_argument("--neuco-seasons", type=int, default=4, help="Number of seasons for NeuCo extraction.")
    parser.add_argument(
        "--evaluation-modality",
        choices=["s1", "s2"],
        default="s2",
        help="Sentinel modality to use for NeuCo evaluation.",
    )
    parser.add_argument("--normalization-method", default="divideby10000", help="Normalization method.")
    parser.add_argument("--annotation-path", type=Path, default=Path("/local/ms-data/SSL4EO-S12-downstream/labels"), help="NeuCo labels root.")
    parser.add_argument("--model-root", type=str, default="/local/ms-data/SSL4EO/model/", help="Root for CIIP checkpoints.")
    parser.add_argument("--croma-weights", type=Path, help="Default path to pretrained CROMA weights.")
    parser.add_argument("--croma-image-resolution", type=int, default=120, help="Input resolution expected by CROMA.")
    parser.add_argument("--feature", choices=["backbone", "projected"], default="backbone", help="Embedding space to use.")
    parser.add_argument("--knn-k", type=int, default=3, help="Number of neighbors for k-NN classification/regression.")
    parser.add_argument(
        "--linear-l2-reg",
        type=float,
        default=1e-6,
        help="L2 regularization for the linear regression probe.",
    )
    parser.add_argument("--limited-label-train", type=float, default=0.1, help="Fraction of train labels to keep.")
    parser.add_argument("--limited-label-val", type=float, default=1.0, help="Fraction of eval labels to keep.")
    parser.add_argument(
        "--limited-label-strategy",
        choices=["random", "stratified", "oversampled"],
        default="stratified",
        help="Strategy to subsample labels.",
    )
    parser.add_argument("--stratification-bins", type=int, default=3, help="Bins for stratified sampling on regression labels.")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fallback eval fraction when split flags are missing.")
    parser.add_argument("--task-filter", nargs="*", default=None, help="Optional list of task names to include.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model evaluation.")
    parser.add_argument("--reuse-embeddings", action="store_true", help="Reuse existing NeuCo CSV if present.")
    parser.add_argument("--export-embeddings", action="store_true", help="Export embeddings CSV for reuse.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a summary of per-model result files.",
    )
    args = parser.parse_args()

    models, defaults = _load_model_specs(args.models_json)
    if not models:
        raise ValueError("No models found in the provided JSON file.")

    base = argparse.Namespace(
        model_type=None,
        model_weights=None,
        checkpoint=None,
        model_path=None,
        ciip_epoch=10,
        ciip_framework=None,
        model_in_channels=None,
        model_root=args.model_root,
        croma_weights=args.croma_weights,
        croma_image_resolution=args.croma_image_resolution,
        neuco_root=args.neuco_root,
        neuco_modalities=args.neuco_modalities,
        neuco_seasons=args.neuco_seasons,
        evaluation_modality=args.evaluation_modality,
        normalization_method=args.normalization_method,
        output_dir=args.output_dir,
        annotation_path=args.annotation_path,
        feature=args.feature,
        knn_k=args.knn_k,
        linear_l2_reg=args.linear_l2_reg,
        limited_label_train=args.limited_label_train,
        limited_label_val=args.limited_label_val,
        limited_label_strategy=args.limited_label_strategy,
        stratification_bins=args.stratification_bins,
        val_frac=args.val_frac,
        task_filter=args.task_filter,
        seed=args.seed,
        reuse_embeddings=args.reuse_embeddings,
        export_embeddings=args.export_embeddings,
    )

    outputs: List[Dict[str, str]] = []
    for idx, model_spec in enumerate(models, start=1):
        run_args = _merge_args(base, defaults)
        run_args = _merge_args(run_args, model_spec)
        label = getattr(run_args, "model_path", None) or run_args.model_type
        print(f"Running model {idx}/{len(models)}: {label}")
        out_paths = run_from_args(run_args)
        for out_path in out_paths:
            outputs.append(
                {
                    "model": json.dumps(model_spec),
                    "output": str(out_path),
                }
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.summary_json is not None:
        summary = {"results": outputs}
        args.summary_json.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
