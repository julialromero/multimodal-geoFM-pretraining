#!/usr/bin/env python3
"""Batch runner for Pangaea segmentation evaluations with normalization variants."""

from __future__ import annotations

import argparse
import os
import socket
import csv
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ciip.evaluation.result_records import (
    build_model_tag,
    ensure_dir,
    write_json,
    write_run_manifest,
)



@dataclass(frozen=True)
class ModelSpec:
    encoder: str
    model_type: str
    model_weights: Optional[str] = None
    model_path: Optional[str] = None
    ciip_epoch: Optional[int] = None
    encoder_weights: Optional[str] = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    preprocessing: str
    criterion: str
    task: str = "segmentation"
    decoder: str = "seg_upernet"


NORMALIZATION_VARIANTS = {
    "default": "",
    # "divideby10000": "divideby10000"
}

DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="mados",
        preprocessing="seg_focus_crop",
        criterion="cross_entropy",
    ),
    DatasetSpec(
        name="hlsburnscars",
        preprocessing="seg_default",
        criterion="cross_entropy",
    ),
    DatasetSpec(
        name="spacenet7",
        preprocessing="seg_default",
        criterion="dice",
    ),
    DatasetSpec(
        name="sen1floods11",
        preprocessing="seg_default",
        criterion="cross_entropy",
    ),
    DatasetSpec(
        name="ai4smallfarms",
        preprocessing="seg_default",
        criterion="dice",
    ),
]

MODELS: List[ModelSpec] = [
    # ModelSpec(
    #     encoder="ciip_s2_vit",
    #     model_type="ciip_checkpoint",
    #     model_path="2026_01_29_matryoshka_vit",
    #     ciip_epoch=300,
    #     encoder_weights="/local/ms-data/SSL4EO/model/2026_01_29_matryoshka_vit/checkpoints/epoch_300.pt",
    # ),
    # ModelSpec(
    #     name="ciip_s2_epoch50_bandnorm_8k",
    #     encoder="ciip_s2",
    #     encoder_weights="/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_50.pt",
    # ),

    # ModelSpec(
    #     name="ciip_s2_epoch70_scaled_12k",
    #     encoder="ciip_s2",
    #     encoder_weights="/local/ms-data/SSL4EO/model/2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16/checkpoints/epoch_70.pt",
    # ),

    # ModelSpec(
    #     name="ciip_s2_vit_epoch120_scaled_12k",
    #     encoder="ciip_s2_vit",
    #     encoder_weights="/local/ms-data/SSL4EO/model/1_8_2026/checkpoints/epoch_120.pt",
    # ),
    # ModelSpec(
    #     encoder="ciip_s2_vit",
    #     model_type="ciip_checkpoint",
    #     model_path="1_28-ViT-DAI",
    #     ciip_epoch=300,
    #     encoder_weights="/local/ms-data/SSL4EO/model/1_28-ViT-DAI/checkpoints/epoch_300.pt",
    # ),
    
    # ModelSpec(
    #     encoder="remoteclip",
    #     model_type="backbone_only",
    #     model_weights="remoteclip",
    # ),
    # # # @ codex add llama-ms-clip
    # ModelSpec(
    #     name="llama_ms_clip",
    #     encoder="llama_ms_clip",
    # ),
    ModelSpec(
        name="resnet50_ssl4eo_moco",
        encoder="resnet50_ssl4eo_moco",
    ),
    # add vit small moco
    ModelSpec(
        name="vit_small_ssl4eo_moco",
        encoder="vit_small_ssl4eo_moco",
    ),


    ModelSpec(
        name="scalemae",
        encoder="scalemae",
    ),
    ModelSpec(
        name="croma_optical",
        encoder="croma_optical",
    ),
    # ModelSpec(
    #     name="dofa",
    #     encoder="dofa",
    # ),


    ModelSpec(
        encoder="ssl4eo_moco",
        model_type="torchgeo_resnet50",
        model_weights="moco",
    ),
    # ModelSpec(
    #     name="terramind_large",
    #     encoder="terramind_large",
    # ),
    # ModelSpec(
    #     name='ssl4eo_mae',
    #     encoder="ssl4eo_mae_optical"
    # )
    # ModelSpec(
    #     name="ciip_s2_epoch50_scaled_8k",
    #     encoder="ciip_s2",
    #     encoder_weights="/local/ms-data/SSL4EO/model/2026_01_12-14_13_43-model_resnet50-lr_0.002-b_6-j_6-p_amp_bfloat16/epoch_50.pt",
    # ),
    # ModelSpec(
    #     name="ciip_s2_vit_epoch70_scaled_12k",
    #     encoder="ciip_s2_vit",
    #     encoder_weights="/local/ms-data/SSL4EO/model/1_8_2026/checkpoints/epoch_70.pt",
    # ),

    ModelSpec(
        encoder="ciip_s2_vit",
        model_type="ciip_checkpoint",
        model_path="2026_02_01_vit_ciip_dai_bandwise",
        ciip_epoch=300,
        encoder_weights="/local/ms-data/SSL4EO/model/2026_02_01_vit_ciip_dai_bandwise/checkpoints/epoch_300.pt",
    ),
    
]


def pick_preprocessing(base_name: str, norm_key: str) -> str:
    suffix = NORMALIZATION_VARIANTS[norm_key]
    if not suffix:
        return base_name
    return f"{base_name}_{suffix}"


def find_new_experiment_dir(work_dir: Path, before: set[Path]) -> Path:
    after = {p for p in work_dir.iterdir() if p.is_dir()}
    new_dirs = list(after - before)
    if not new_dirs:
        raise RuntimeError(f"No new experiment directory found under {work_dir}")
    if len(new_dirs) == 1:
        return new_dirs[0]
    return max(new_dirs, key=lambda p: p.stat().st_mtime)


def find_log_file(exp_dir: Path) -> Path:
    for name in ("train.log-0", "test.log-0"):
        cand = exp_dir / name
        if cand.exists():
            return cand
    matches = sorted(exp_dir.glob("*.log-*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No log file found in {exp_dir}")


def parse_segmentation_metrics(log_path: Path, split: str = "test") -> Dict[str, float]:
    metrics = {"mIoU": float("nan"), "mF1": float("nan"), "mAcc": float("nan")}
    section: Optional[str] = None
    mean_line = re.compile(r"\bMean\b.*?([0-9.]+)\s*$")
    acc_line = re.compile(r"Mean Accuracy:\s*([0-9.]+)")

    for line in log_path.read_text().splitlines():
        if f"[{split}] ------- IoU" in line:
            section = "mIoU"
            continue
        if f"[{split}] ------- F1-score" in line:
            section = "mF1"
            continue

        acc_match = acc_line.search(line)
        if acc_match:
            metrics["mAcc"] = float(acc_match.group(1))
            continue

        if section and f"[{split}] Mean" in line:
            match = mean_line.search(line)
            if match:
                metrics[section] = float(match.group(1))
            section = None

    return metrics


def write_experiment_csv(exp_dir: Path, row: Dict[str, object]) -> Path:
    out_path = exp_dir / "metrics.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return out_path


def load_experiment_rows(result_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for csv_path in result_root.rglob("metrics.csv"):
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def plot_metrics(rows: List[Dict[str, object]], output_dir: Path) -> None:
    datasets_order = [d.name for d in DATASETS]
    metrics = [("mIoU", "Mean IoU"), ("mF1", "F1 Score"), ("mAcc", "Accuracy")]
    models = [_model_tag(m) for m in MODELS]
    colors = {name: plt.get_cmap("tab10")(idx % 10) for idx, name in enumerate(models)}

    for norm_key in NORMALIZATION_VARIANTS:
        norm_rows = [r for r in rows if r.get("normalization") == norm_key]
        if not norm_rows:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, (metric_key, metric_label) in zip(axes, metrics):
            for model in models:
                ys = []
                for dataset in datasets_order:
                    value = next(
                        (
                            to_float(r.get(metric_key))
                            for r in norm_rows
                            if r.get("model") == model and r.get("dataset") == dataset
                        ),
                        float("nan"),
                    )
                    ys.append(value)
                ax.plot(datasets_order, ys, marker="o", label=model, color=colors[model])
            ax.set_title(metric_label)
            ax.set_ylabel("Score")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"Segmentation metrics by dataset (norm={norm_key})", y=1.05, fontsize=14)
        fig.tight_layout()

        output_path = output_dir / f"metrics_lineplot_{norm_key}.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _model_tag(model: ModelSpec) -> str:
    return build_model_tag(
        model_type=model.model_type,
        model_weights=model.model_weights,
        model_path=model.model_path,
        ciip_epoch=model.ciip_epoch,
    )


def _sanitize_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip()).strip("_")


def validate_models(models: List[ModelSpec]) -> None:
    for model in models:
        tag = _model_tag(model)
        if not tag:
            raise ValueError(f"Model {model} produced an empty tag.")
        if model.model_type == "ciip_checkpoint":
            if not model.model_path or model.ciip_epoch is None:
                raise ValueError(f"CIIP checkpoints must define model_path and ciip_epoch: {model}")
        if model.model_type == "backbone_only" and not model.model_weights:
            raise ValueError(f"Backbone-only models must define model_weights: {model}")


def load_existing_results(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    existing: set[tuple[str, str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_split = (row.get("label_split") or "").strip().lower()
            if label_split != "10%":
                continue
            dataset = row.get("dataset")
            model = row.get("model")
            normalization = row.get("normalization")
            preprocessing = row.get("preprocessing")
            if not (dataset and model and normalization and preprocessing):
                continue
            existing.add((dataset, model, normalization, preprocessing))
    return existing


def write_results_output(
    output_root: Path,
    *,
    dataset: DatasetSpec,
    model: ModelSpec,
    normalization: str,
    preprocessing: str,
    metrics: Dict[str, float],
    exp_dir: Path,
    log_path: Path,
    label_split: str,
) -> Path:
    model_tag = _model_tag(model)
    output_dir = ensure_dir(
        output_root
        / "pangaea_segmentation"
        / label_split
        / model_tag
        / _sanitize_path_component(dataset.name)
        / _sanitize_path_component(normalization)
        / _sanitize_path_component(preprocessing)
    )
    config = {
        "dataset": dataset.name,
        "encoder": model.encoder,
        "encoder_weights": model.encoder_weights,
        "decoder": dataset.decoder,
        "preprocessing": preprocessing,
        "criterion": dataset.criterion,
        "task": dataset.task,
        "normalization": normalization,
        "model_type": model.model_type,
        "model_weights": model.model_weights,
        "model_path": model.model_path,
        "ciip_epoch": model.ciip_epoch,
        "label_split": label_split,
    }
    write_run_manifest(
        output_dir,
        task_name="pangaea_segmentation",
        config=config,
        extra={"exp_dir": str(exp_dir), "log_path": str(log_path)},
    )
    results = {
        **config,
        "exp_dir": str(exp_dir),
        "log_path": str(log_path),
        "mIoU": metrics["mIoU"],
        "mF1": metrics["mF1"],
        "mAcc": metrics["mAcc"],
    }
    return write_json(output_dir / "results.json", results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pangaea segmentation evaluations.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/local/ms-data/pangaea-bench/batch_runs/fewshot10"),
        help="Directory to store experiment outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write standardized results outputs.",
    )
    parser.add_argument(
        "--torchrun",
        type=str,
        default="torchrun",
        help="torchrun executable to use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default="0,1",
        help="CUDA_VISIBLE_DEVICES value for multi-GPU runs.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=0,
        help="torchrun master port (0 picks a free port automatically).",
    )
    args = parser.parse_args()

    validate_models(MODELS)

    repo_root = Path(__file__).resolve().parents[1]
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_root = (args.output_dir or work_dir / "evaluation_outputs").resolve()

    results_root = work_dir / "metrics" / "fewshot10"
    results_root.mkdir(parents=True, exist_ok=True)
    combined_metrics_path = Path("/local/ms-data/pangaea-bench/batch_runs/metrics_summary/10pct/combined_metrics.csv")
    existing_results = load_existing_results(combined_metrics_path)

    rows: List[Dict[str, object]] = []

    def find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    base_master_port = args.master_port or find_free_port()

    for dataset in DATASETS:
        for model in MODELS:
            for norm_key in NORMALIZATION_VARIANTS:
                preprocessing = pick_preprocessing(dataset.preprocessing, norm_key)
                model_tag = _model_tag(model)
                result_key = (dataset.name, model_tag, norm_key, preprocessing)
                if result_key in existing_results:
                    print(f"Skipping existing result: dataset={dataset.name}, model={model_tag}, "
                          f"normalization={norm_key}, preprocessing={preprocessing}")
                    continue
                master_port = base_master_port + (hash((dataset.name, model_tag, norm_key)) % 1000)
                cmd = [
                    args.torchrun,
                    "--nnodes=1",
                    "--nproc_per_node=1",
                    f"--master_port={master_port}",
                    "pangaea/run.py",
                    "--config-name=train_fewshot10",
                    f"dataset={dataset.name}",
                    f"encoder={model.encoder}",
                    f"decoder={dataset.decoder}",
                    f"preprocessing={preprocessing}",
                    f"criterion={dataset.criterion}",
                    f"task={dataset.task}",
                    f"work_dir={work_dir}",
                    "task.trainer.eval_interval=5",
                    "task.trainer.early_stopping_patience_evals=1000",
                ]
                if dataset.name == "ai4smallfarms":
                    cmd.extend(
                        [
                            "data_replicate=2",
                            "task.trainer.best_metric_key=IoU",
                            "batch_size=2",
                        ]
                    )
              
                if model.encoder_weights:
                    cmd.append(f"encoder.encoder_weights={model.encoder_weights}")

                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
                env.setdefault("MASTER_ADDR", "127.0.0.1")
                env["MASTER_PORT"] = str(master_port)
                print("Running:", " ".join(cmd))
                if args.dry_run:
                    continue

                before_dirs = {p for p in work_dir.iterdir() if p.is_dir()}
                subprocess.run(cmd, check=True, cwd=repo_root, env=env)
                exp_dir = find_new_experiment_dir(work_dir, before_dirs)

                log_path = find_log_file(exp_dir)
                metrics = parse_segmentation_metrics(log_path, split="test")
                label_split = "10%"

                row = {
                    "experiment": exp_dir.name,
                    "dataset": dataset.name,
                    "model": model_tag,
                    "normalization": norm_key,
                    "preprocessing": preprocessing,
                    "mIoU": metrics["mIoU"],
                    "mF1": metrics["mF1"],
                    "mAcc": metrics["mAcc"],
                    "log_path": str(log_path),
                    "exp_dir": str(exp_dir),
                }
                metrics_csv = write_experiment_csv(exp_dir, row)
                write_results_output(
                    output_root,
                    dataset=dataset,
                    model=model,
                    normalization=norm_key,
                    preprocessing=preprocessing,
                    metrics=metrics,
                    exp_dir=exp_dir,
                    log_path=log_path,
                    label_split=label_split,
                )
                rows.append(row)
                existing_results.add(result_key)

                summary_path = results_root / f"{exp_dir.name}_metrics.csv"
                summary_path.write_text(metrics_csv.read_text())

                time.sleep(1)

    if not rows and not args.dry_run:
        rows = load_experiment_rows(work_dir)

    if rows and not args.dry_run:
        summary_csv = results_root / "summary_metrics.csv"
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        plot_metrics(rows, results_root)
        print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
