"""NeuCo-Bench evaluation with optional few-shot label subsampling.

This mirrors unified_evaluation's NeuCo export + benchmark launch, while
reusing the EuroSAT few-shot runner style for model/config handling.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ciip.evaluation.model_utils import build_evaluation_adapter
from ciip.evaluation.unified_evaluation import (
    DEFAULT_NORMALIZATION_METHOD,
    NORMALIZATION_METHODS,
    ModelEvalConfig,
    _build_neuco_loader,
    _export_neuco,
    _extract_embeddings,
    _infer_model_in_channels,
    _use_adapter_modality,
)


def _resolve_checkpoint(args: argparse.Namespace) -> Optional[Path]:
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
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sanitize_tag(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return safe.strip("_")


def _model_tag(args: argparse.Namespace) -> str:
    parts: List[str] = [args.model_type]
    if args.model_weights:
        parts.append(args.model_weights)
    if args.model_path:
        parts.append(args.model_path)
    if args.model_type == "ciip_checkpoint" and args.ciip_epoch is not None:
        parts.append(f"epoch{args.ciip_epoch}")
    tag = "_".join(parts)
    return _sanitize_tag(tag) if tag else "model"


def _infer_embedding_dim_from_csv(csv_path: Path) -> Optional[int]:
    try:
        with csv_path.open() as handle:
            header = handle.readline().strip()
    except OSError:
        return None
    if not header:
        return None
    cols = [col.strip() for col in header.split(",") if col.strip()]
    if cols and cols[0] == "id" and len(cols) > 1:
        return len(cols) - 1
    return None


def _heuristic_embedding_dim(args: argparse.Namespace) -> int:
    weights = (args.model_weights or "").lower()
    if args.model_type == "croma" or "dofa" in weights:
        return 768
    if args.model_type == "torchgeo_resnet50":
        return 2048
    if "resnet18" in weights or "llama" in weights:
        return 512
    if "remoteclip" in weights:
        return 768
    if "resnet50" in weights:
        return 2048
    if "vitsmall" in weights:
        return 384
    if "scalemae" in weights:
        return 1024
    return 2048


def _read_label_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or ["id", "label"]
        rows = [row for row in reader if row.get("label") not in (None, "")]
    return list(fieldnames), rows


def _sample_rows(
    rows: List[Dict[str, str]],
    *,
    k_shot: int,
    task_type: str,
    rng: np.random.Generator,
) -> List[Dict[str, str]]:
    if k_shot <= 0:
        return rows
    task_type = task_type.lower()
    if task_type in {"cls", "classification"}:
        by_label: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            by_label.setdefault(str(row.get("label", "")), []).append(row)
        sampled: List[Dict[str, str]] = []
        for label, group in sorted(by_label.items()):
            if len(group) < k_shot:
                raise ValueError(
                    f"Task label {label} has {len(group)} samples, fewer than k_shot={k_shot}."
                )
            indices = rng.choice(len(group), size=k_shot, replace=False)
            sampled.extend(group[int(idx)] for idx in indices)
        return sampled
    if len(rows) < k_shot:
        raise ValueError(f"Task has {len(rows)} samples, fewer than k_shot={k_shot}.")
    indices = rng.choice(len(rows), size=k_shot, replace=False)
    return [rows[int(idx)] for idx in indices]


def _prepare_fewshot_annotations(
    annotation_root: Path,
    output_root: Path,
    *,
    k_shot: int,
    seed: int,
    task_filter: Optional[Sequence[str]],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    allowed_tasks = set(task_filter) if task_filter else None
    csv_paths = sorted(annotation_root.glob("*.csv"))
    rng = np.random.default_rng(seed)
    for csv_path in csv_paths:
        task_name, task_type = csv_path.stem.split("__", 1)
        if allowed_tasks and task_name not in allowed_tasks:
            continue
        fieldnames, rows = _read_label_rows(csv_path)
        sampled = _sample_rows(rows, k_shot=k_shot, task_type=task_type, rng=rng)
        target = output_root / csv_path.name
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled)
    meta = {
        "source": str(annotation_root),
        "k_shot": k_shot,
        "seed": seed,
        "task_filter": list(task_filter) if task_filter else None,
    }
    (output_root / "fewshot_manifest.json").write_text(json.dumps(meta, indent=2))
    return output_root


def _select_neuco_modality(
    evaluation_modality: str, neuco_modalities: Sequence[str]
) -> Tuple[str, str]:
    evaluation_modality = evaluation_modality.lower()
    modalities = list(neuco_modalities)
    if evaluation_modality == "s1":
        return "s1", "s1"
    s2_candidates = [m for m in modalities if m in ("s2l2a", "s2l1c")]
    if s2_candidates:
        return s2_candidates[0], "s2"
    if modalities:
        return modalities[0], "s2"
    return "s2l1c", "s2"


def _find_latest_benchmark_summary(
    output_dir: Path, *, phase: str, method_name: str
) -> Optional[Path]:
    run_root = output_dir / phase
    if not run_root.exists():
        return None
    candidates = [
        path
        for path in run_root.iterdir()
        if path.is_dir() and path.name.startswith(f"{method_name}_")
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    summary = latest / "results_summary.json"
    if summary.exists():
        return summary
    return latest


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

    normalization_method = args.normalization_method.lower()
    if normalization_method == "ssl4eobandwisenorm":
        normalization_method = "ssl4eonorm"
    if normalization_method not in NORMALIZATION_METHODS:
        raise ValueError(
            f"--normalization-method must be one of {', '.join(NORMALIZATION_METHODS)}; got {args.normalization_method!r}"
        )

    checkpoint = _resolve_checkpoint(args)

    cfg = ModelEvalConfig(
        eurosat_root=Path(args.eurosat_root),
        neuco_root=Path(args.neuco_root),
        output_dir=_build_output_dir(args),
        checkpoint=checkpoint,
        model_type=args.model_type,
        model_weights=args.model_weights,
        ciip_framework=args.ciip_framework,
        model_in_channels=args.model_in_channels if args.model_in_channels is not None else 13,
        croma_weights=args.croma_weights,
        croma_image_resolution=args.croma_image_resolution,
        evaluation_modality=args.evaluation_modality,
        neuco_modalities=tuple(args.neuco_modalities),
        neuco_seasons=args.neuco_seasons,
        normalization_method=normalization_method,
        random_seed=args.seed,
    )

    adapter = build_evaluation_adapter(
        model_type=cfg.model_type,
        checkpoint=cfg.checkpoint,
        model_weights=cfg.model_weights,
        in_chans=cfg.model_in_channels,
        croma_weights=cfg.croma_weights,
        croma_image_resolution=cfg.croma_image_resolution,
        ciip_framework=cfg.ciip_framework,
        enable_s1=cfg.evaluation_modality.lower() == "s1",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = adapter.to(device)
    adapter.eval()

    neuco_modality, active_modality = _select_neuco_modality(
        cfg.evaluation_modality, cfg.neuco_modalities
    )

    model_tag = _model_tag(args)
    neuco_output_dir = cfg.output_dir / f"neuco_{normalization_method}" / model_tag
    neuco_export_dir = neuco_output_dir / "neuco_export"
    neuco_export_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_s1" if active_modality.lower() == "s1" else ""
    csv_out_backbone = neuco_export_dir / f"neuco_{neuco_modality}{suffix}_backbone.csv"

    neuco_bundle = None
    if not (args.reuse_embeddings and csv_out_backbone.exists()):
        neuco_loader = _build_neuco_loader(cfg, modalities=[neuco_modality])
        expected_channels = _infer_model_in_channels(
            adapter, cfg.model_in_channels, modality=active_modality
        )
        with _use_adapter_modality(adapter, active_modality):
            neuco_bundle = _extract_embeddings(
                adapter,
                neuco_loader,
                device=device,
                require_ids=True,
                expected_in_channels=expected_channels,
                modality=active_modality,
            )
        _export_neuco(neuco_bundle, neuco_export_dir, label=neuco_modality)
        print(f"Saved NeuCo embeddings to {neuco_export_dir}")

    if neuco_bundle is not None and neuco_bundle.backbone is not None:
        embedding_dim = int(neuco_bundle.backbone.shape[1])
    else:
        embedding_dim = _infer_embedding_dim_from_csv(csv_out_backbone) or _heuristic_embedding_dim(args)

    annotation_path = Path(args.annotation_path)
    if args.few_shot_k > 0:
        fewshot_dir = neuco_output_dir / "fewshot_labels" / f"k{args.few_shot_k}_seed{args.few_shot_seed}"
        annotation_path = _prepare_fewshot_annotations(
            annotation_path,
            fewshot_dir,
            k_shot=args.few_shot_k,
            seed=args.few_shot_seed,
            task_filter=args.task_filter,
        )

    method_name = args.method_name or model_tag
    if args.few_shot_k > 0:
        method_name = f"{method_name}_k{args.few_shot_k}"

    summary_path = None
    if not args.skip_benchmark:
        benchmark_main = Path(args.benchmark_root) / "main.py"
        cmd = [
            sys.executable,
            str(benchmark_main),
            "--annotation_path",
            str(annotation_path),
            "--output_dir",
            str(neuco_output_dir),
            "--config",
            str(args.benchmark_config),
            "--method_name",
            method_name,
            "--phase",
            args.benchmark_phase,
            "--submission_file",
            str(csv_out_backbone),
            "--embedding_dim",
            str(embedding_dim),
        ]
        if args.exclude_file is not None:
            cmd.extend(["--exclude_file", str(args.exclude_file)])
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        subprocess.run(cmd, check=True, env=env)
        summary_path = _find_latest_benchmark_summary(
            neuco_output_dir, phase=args.benchmark_phase, method_name=method_name
        )

    results = {
        "model_type": args.model_type,
        "model_weights": args.model_weights,
        "model_path": args.model_path,
        "ciip_epoch": args.ciip_epoch,
        "normalization_method": normalization_method,
        "evaluation_modality": active_modality,
        "neuco_modality": neuco_modality,
        "embedding_dim": embedding_dim,
        "submission_csv": str(csv_out_backbone),
        "annotation_path": str(annotation_path),
        "few_shot_k": args.few_shot_k,
        "few_shot_seed": args.few_shot_seed,
        "knn_k": args.knn_k,
        "benchmark_output_dir": str(neuco_output_dir),
        "benchmark_phase": args.benchmark_phase,
        "benchmark_summary": str(summary_path) if summary_path else None,
    }
    out_path = neuco_output_dir / f"neuco_fewshot_run_{model_tag}.json"
    out_path.write_text(json.dumps(results, indent=2))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NeuCo-Bench evaluation with optional few-shot label subsampling."
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
    parser.add_argument("--eurosat-root", type=Path, default=Path("/local/ms-data/EuroSAT/"), help="EuroSAT root (unused).")
    parser.add_argument("--neuco-root", type=Path, default=Path("/local/ms-data/SSL4EO-S12-downstream/data"), help="NeuCo root.")
    parser.add_argument("--neuco-modalities", nargs="*", default=["s2l2a"], help="NeuCo modalities to export.")
    parser.add_argument("--neuco-seasons", type=int, default=4, help="Number of seasons for NeuCo extraction.")
    parser.add_argument(
        "--evaluation-modality",
        choices=["s1", "s2"],
        default="s2",
        help="Sentinel modality to use for NeuCo evaluation.",
    )
    parser.add_argument(
        "--normalization-method",
        choices=tuple(NORMALIZATION_METHODS) + ("ssl4eobandwisenorm",),
        default=DEFAULT_NORMALIZATION_METHOD,
        help="Normalization applied to NeuCo inputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/neuco_fewshot"), help="Output directory.")
    parser.add_argument("--annotation-path", type=Path, default=Path("/local/ms-data/SSL4EO-S12-downstream/labels"), help="NeuCo labels root.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("/local/ms-data/NeuCo-Bench/benchmark"), help="NeuCo benchmark root.")
    parser.add_argument("--benchmark-config", type=Path, default=Path("/local/ms-data/NeuCo-Bench/benchmark/config.yaml"), help="NeuCo benchmark config.")
    parser.add_argument("--benchmark-phase", type=str, default="testing", help="NeuCo benchmark phase.")
    parser.add_argument("--method-name", type=str, default=None, help="Override NeuCo benchmark method name.")
    parser.add_argument("--exclude-file", type=Path, default=None, help="Exclude file for NeuCo benchmark.")
    parser.add_argument("--few-shot-k", type=int, default=0, help="Samples per class/task for few-shot labels.")
    parser.add_argument("--few-shot-seed", type=int, default=0, help="Seed for few-shot label sampling.")
    parser.add_argument("--task-filter", nargs="*", default=None, help="Optional list of task names to include.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model evaluation.")
    parser.add_argument("--reuse-embeddings", action="store_true", help="Reuse existing NeuCo CSV if present.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Only export embeddings; skip benchmark run.")
    parser.add_argument("--knn-k", type=int, default=1, help="Unused (NeuCo benchmark uses linear probes).")
    args = parser.parse_args()

    run_from_args(args)


if __name__ == "__main__":
    main()
