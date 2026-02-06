"""Run EuroSAT 1-shot 5-way 1-NN evaluation across multiple models."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from ciip.evaluation.eurosat_fewshot_1nn import run_from_args


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
        description="Run EuroSAT 1-shot 5-way 1-NN evaluation across multiple models."
    )
    parser.add_argument("--models-json", type=Path, required=True, help="JSON file listing models to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/fewshot_eval"), help="Output directory.")
    parser.add_argument("--eurosat-root", type=Path, default=Path("/local/ms-data/EuroSAT/"), help="EuroSAT root.")
    parser.add_argument("--eurosat-image-size", type=int, default=224, help="EuroSAT input resolution.")
    parser.add_argument("--normalization-method", default="divideby10000", help="Normalization method.")
    parser.add_argument("--feature", choices=["backbone", "projected"], default="backbone", help="Embedding space to use.")
    parser.add_argument("--n-way", type=int, default=5, help="Number of classes per episode.")
    parser.add_argument("--k-shot", type=int, default=1, help="Number of support samples per class.")
    parser.add_argument("--knn-k", type=int, default=1, help="Number of neighbors for k-NN classification.")
    parser.add_argument("--queries-per-class", type=int, default=15, help="Number of query samples per class.")
    parser.add_argument("--episodes", type=int, default=600, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode sampling.")
    parser.add_argument("--model-root", type=str, default="/local/ms-data/SSL4EO/model/", help="Root for CIIP checkpoints.")
    parser.add_argument("--croma-weights", type=Path, help="Default path to pretrained CROMA weights.")
    parser.add_argument("--croma-image-resolution", type=int, default=120, help="Input resolution expected by CROMA.")
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
        eurosat_root=args.eurosat_root,
        eurosat_image_size=args.eurosat_image_size,
        normalization_method=args.normalization_method,
        output_dir=args.output_dir,
        feature=args.feature,
        n_way=args.n_way,
        k_shot=args.k_shot,
        knn_k=args.knn_k,
        queries_per_class=args.queries_per_class,
        episodes=args.episodes,
        seed=args.seed,
    )

    outputs: List[Dict[str, str]] = []
    for idx, model_spec in enumerate(models, start=1):
        run_args = _merge_args(base, defaults)
        run_args = _merge_args(run_args, model_spec)
        print(f"Running model {idx}/{len(models)}: {getattr(run_args, 'model_path', None) or run_args.model_type}")
        out_path = run_from_args(run_args)
        normalization_method = None
        try:
            normalization_method = json.loads(out_path.read_text()).get("normalization_method")
        except (json.JSONDecodeError, OSError):
            normalization_method = getattr(run_args, "normalization_method", None)
        outputs.append(
            {
                "model": json.dumps(model_spec),
                "output": str(out_path),
                "normalization_method": normalization_method,
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
