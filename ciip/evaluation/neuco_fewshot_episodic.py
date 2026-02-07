"""NeuCo limited-label evaluation with stratified sampling and k-NN probes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ciip.evaluation.model_utils import build_evaluation_adapter
from ciip.evaluation.normalization_utils import (
    DEFAULT_NORMALIZATION_METHOD,
    NORMALIZATION_METHODS,
    resolve_normalization_method,
)
from ciip.evaluation.output_utils import (
    build_model_tag,
    ensure_dir,
    write_json,
    write_run_manifest,
)
from ciip.evaluation.unified_evaluation import (
    ModelEvalConfig,
    _build_neuco_loader,
    _export_neuco,
    _extract_embeddings,
    _infer_model_in_channels,
    _use_adapter_modality,
)


_DEV_COLUMN = "cvpr_earthvision_phase_dev"
_EVAL_COLUMN = "cvpr_earthvision_phase_eval"


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


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _iter_label_tasks(annotation_path: Path, task_filter: Optional[Sequence[str]]) -> Iterable[Tuple[str, str, Path]]:
    allowed = set(task_filter) if task_filter else None
    for path in sorted(annotation_path.glob("*.csv")):
        if "__" not in path.stem:
            continue
        task_name, task_type = path.stem.split("__", 1)
        if allowed and task_name not in allowed:
            continue
        yield task_name, task_type, path


def _read_label_rows(label_path: Path) -> List[Dict[str, str]]:
    with label_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = []
        for row in reader:
            label = row.get("label")
            if label is None or label == "":
                continue
            rows.append(row)
    return rows


def _split_rows(rows: List[Dict[str, str]], seed: int, val_frac: float) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    has_dev = rows and _DEV_COLUMN in rows[0]
    has_eval = rows and _EVAL_COLUMN in rows[0]
    if has_dev or has_eval:
        train_rows = [row for row in rows if _truthy(row.get(_DEV_COLUMN))]
        eval_rows = [row for row in rows if _truthy(row.get(_EVAL_COLUMN))]
        if train_rows and eval_rows:
            return train_rows, eval_rows
    if val_frac <= 0:
        return rows, []
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(rows))
    split = int(max(1, len(rows) * (1 - val_frac)))
    train_rows = [rows[idx] for idx in indices[:split]]
    eval_rows = [rows[idx] for idx in indices[split:]]
    return train_rows, eval_rows


def _sample_indices_random(size: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=np.int64)
    if fraction >= 1:
        return np.arange(size)
    if fraction <= 0:
        return np.array([], dtype=np.int64)
    count = max(1, int(size * fraction))
    return rng.choice(size, size=count, replace=False)


def _sample_indices_stratified_class(labels: List[str], fraction: float, rng: np.random.Generator) -> np.ndarray:
    indices_by_class: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(label, []).append(idx)
    selected: List[int] = []
    for indices in indices_by_class.values():
        if fraction <= 0:
            continue
        count = max(1, int(len(indices) * fraction))
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def _sample_indices_stratified_reg(values: np.ndarray, fraction: float, num_bins: int, rng: np.random.Generator) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.int64)
    if values.min() == values.max():
        return _sample_indices_random(values.size, fraction, rng)
    bin_edges = np.linspace(values.min(), values.max(), num_bins + 1)
    binned = np.digitize(values, bin_edges) - 1
    indices_per_bin: Dict[int, List[int]] = {i: [] for i in range(num_bins)}
    for idx, bin_idx in enumerate(binned):
        if bin_idx in indices_per_bin:
            indices_per_bin[bin_idx].append(idx)
    selected: List[int] = []
    for indices in indices_per_bin.values():
        if not indices:
            continue
        if fraction <= 0:
            continue
        count = max(1, int(len(indices) * fraction))
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def _apply_sampling(
    rows: List[Dict[str, str]],
    task_type: str,
    *,
    fraction: float,
    strategy: str,
    num_bins: int,
    rng: np.random.Generator,
) -> List[Dict[str, str]]:
    if fraction >= 1:
        return rows
    if not rows:
        return rows
    labels = [row.get("label", "") for row in rows]
    if strategy == "random":
        indices = _sample_indices_random(len(rows), fraction, rng)
    elif task_type == "cls":
        if strategy != "stratified":
            raise ValueError(f"Unsupported strategy for classification: {strategy}")
        indices = _sample_indices_stratified_class(labels, fraction, rng)
    elif task_type == "regr":
        values = np.asarray([float(label) for label in labels], dtype=np.float64)
        if strategy == "stratified":
            indices = _sample_indices_stratified_reg(values, fraction, num_bins, rng)
        elif strategy == "oversampled":
            binned = np.digitize(values, np.linspace(values.min(), values.max(), num_bins + 1)) - 1
            sorted_indices = np.argsort(binned)
            count = max(1, int(len(rows) * fraction))
            indices = sorted_indices[:count]
        else:
            indices = _sample_indices_random(len(rows), fraction, rng)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return [rows[idx] for idx in indices.tolist()]


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return features / denom


def _knn_classify(
    train_feats: np.ndarray,
    train_labels: List[str],
    test_feats: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if train_feats.shape[0] == 0 or test_feats.shape[0] == 0:
        return np.array([], dtype=np.int64), {}
    labels = sorted(set(train_labels))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    train_encoded = np.asarray([label_to_idx[label] for label in train_labels], dtype=np.int64)
    k = min(k, train_feats.shape[0])
    train_norm = _l2_normalize(train_feats)
    test_norm = _l2_normalize(test_feats)
    sims = test_norm @ train_norm.T
    if k == 1:
        nn_indices = np.argmax(sims, axis=1)
        preds = train_encoded[nn_indices]
        return preds, label_to_idx
    topk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    preds: List[int] = []
    for row_idx, idxs in enumerate(topk):
        row_labels = train_encoded[idxs]
        row_scores = sims[row_idx, idxs]
        weights = np.maximum(row_scores, 0.0)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        cls_scores = np.bincount(row_labels, weights=weights, minlength=len(labels))
        preds.append(int(np.argmax(cls_scores)))
    return np.asarray(preds, dtype=np.int64), label_to_idx


def _knn_regress(
    train_feats: np.ndarray,
    train_values: np.ndarray,
    test_feats: np.ndarray,
    k: int,
) -> np.ndarray:
    if train_feats.shape[0] == 0 or test_feats.shape[0] == 0:
        return np.array([], dtype=np.float64)
    k = min(k, train_feats.shape[0])
    train_norm = _l2_normalize(train_feats)
    test_norm = _l2_normalize(test_feats)
    sims = test_norm @ train_norm.T
    if k == 1:
        nn_indices = np.argmax(sims, axis=1)
        return train_values[nn_indices]
    topk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    preds = np.zeros(test_feats.shape[0], dtype=np.float64)
    for row_idx, idxs in enumerate(topk):
        row_scores = sims[row_idx, idxs]
        distances = 1.0 - row_scores
        weights = 1.0 / (distances + 1e-6)
        pred = np.sum(weights * train_values[idxs]) / np.sum(weights)
        preds[row_idx] = pred
    return preds


def _filter_rows(rows: List[Dict[str, str]], id_to_index: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
    indices: List[int] = []
    labels: List[str] = []
    for row in rows:
        sample_id = (row.get("id") or "").strip()
        if sample_id in id_to_index:
            indices.append(id_to_index[sample_id])
            labels.append(str(row.get("label")))
    return np.asarray(indices, dtype=np.int64), labels


def run_from_args(args: argparse.Namespace) -> List[Path]:
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

    normalization_method = resolve_normalization_method(
        args.model_in_channels, args.normalization_method
    )
    if normalization_method not in NORMALIZATION_METHODS:
        raise ValueError(
            f"--normalization-method must be one of {', '.join(NORMALIZATION_METHODS)}; got {args.normalization_method!r}"
        )

    checkpoint = _resolve_checkpoint(args)

    cfg = ModelEvalConfig(
        eurosat_root=Path("/local/ms-data/EuroSAT/"),
        neuco_root=Path(args.neuco_root),
        output_dir=Path(args.output_dir),
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

    model_tag = build_model_tag(
        model_type=args.model_type,
        model_weights=args.model_weights,
        model_path=args.model_path,
        ciip_epoch=args.ciip_epoch,
    )
    output_dir = ensure_dir(cfg.output_dir / "neuco_fewshot" / model_tag / normalization_method)
    export_dir = ensure_dir(output_dir / "neuco_export")
    write_run_manifest(
        output_dir,
        task_name="neuco_fewshot_episodic",
        config={
            "model_type": args.model_type,
            "model_weights": args.model_weights,
            "model_path": args.model_path,
            "ciip_epoch": args.ciip_epoch,
            "model_in_channels": args.model_in_channels,
            "normalization_method": args.normalization_method,
            "neuco_root": args.neuco_root,
            "annotation_path": args.annotation_path,
            "feature": args.feature,
            "seed": args.seed,
        },
    )

    suffix = "_s1" if active_modality.lower() == "s1" else ""
    csv_out_backbone = export_dir / f"neuco_{neuco_modality}{suffix}_backbone.csv"

    embeddings: Optional[np.ndarray] = None
    projected: Optional[np.ndarray] = None
    ids: List[str] = []

    if args.reuse_embeddings and csv_out_backbone.exists():
        ids, embeddings = _load_neuco_csv_embeddings(csv_out_backbone)
    else:
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
        if neuco_bundle.backbone is None:
            raise RuntimeError("Backbone embeddings missing from NeuCo extraction")
        embeddings = neuco_bundle.backbone
        projected = neuco_bundle.projected
        ids = neuco_bundle.ids or []
        if args.export_embeddings:
            _export_neuco(neuco_bundle, export_dir, label=neuco_modality)
            print(f"Saved NeuCo embeddings to {export_dir}")

    if embeddings is None:
        raise RuntimeError("No NeuCo embeddings available for evaluation")

    features = embeddings if args.feature == "backbone" else projected
    if features is None:
        raise ValueError(f"Requested feature '{args.feature}' is not available.")

    id_to_index = {sample_id: idx for idx, sample_id in enumerate(ids)}

    output_paths: List[Path] = []
    rng = np.random.default_rng(args.seed)
    for task_idx, (task_name, task_type, label_path) in enumerate(
        _iter_label_tasks(Path(args.annotation_path), args.task_filter)
    ):
        rows = _read_label_rows(label_path)
        if not rows:
            print(f"[neuco] No labels in {label_path.name}, skipping.")
            continue
        train_rows, eval_rows = _split_rows(rows, args.seed + task_idx, args.val_frac)
        train_rows = _apply_sampling(
            train_rows,
            task_type,
            fraction=args.limited_label_train,
            strategy=args.limited_label_strategy,
            num_bins=args.stratification_bins,
            rng=rng,
        )
        eval_rows = _apply_sampling(
            eval_rows,
            task_type,
            fraction=args.limited_label_val,
            strategy=args.limited_label_strategy,
            num_bins=args.stratification_bins,
            rng=rng,
        )

        train_indices, train_labels = _filter_rows(train_rows, id_to_index)
        eval_indices, eval_labels = _filter_rows(eval_rows, id_to_index)
        if train_indices.size == 0 or eval_indices.size == 0:
            print(f"[neuco] Task {task_name}: insufficient data after filtering, skipping.")
            continue

        train_feats = features[train_indices]
        eval_feats = features[eval_indices]

        results = {
            "dataset": "NeuCo",
            "task": task_name,
            "task_type": task_type,
            "label_path": str(label_path),
            "limited_label_train": args.limited_label_train,
            "limited_label_val": args.limited_label_val,
            "limited_label_strategy": args.limited_label_strategy,
            "stratification_bins": args.stratification_bins,
            "train_total": len(train_rows),
            "eval_total": len(eval_rows),
            "train_used": int(train_indices.size),
            "eval_used": int(eval_indices.size),
            "knn_k": args.knn_k,
            "feature": args.feature,
            "feature_dim": int(features.shape[1]),
            "normalization_method": normalization_method,
            "evaluation_modality": active_modality,
            "neuco_modality": neuco_modality,
            "model_type": args.model_type,
            "model_weights": args.model_weights,
            "model_path": args.model_path,
            "ciip_epoch": args.ciip_epoch,
            "device": str(device),
            "seed": args.seed,
        }

        if task_type == "cls":
            preds, label_to_idx = _knn_classify(train_feats, train_labels, eval_feats, args.knn_k)
            eval_encoded = []
            valid_mask = []
            for idx, label in enumerate(eval_labels):
                if label in label_to_idx:
                    eval_encoded.append(label_to_idx[label])
                    valid_mask.append(idx)
            if valid_mask:
                preds = preds[valid_mask]
                eval_encoded = np.asarray(eval_encoded, dtype=np.int64)
                accuracy = float(np.mean(preds == eval_encoded)) if preds.size else 0.0
            else:
                accuracy = 0.0
            results.update(
                {
                    "accuracy": accuracy,
                    "eval_dropped": int(len(eval_labels) - len(valid_mask)),
                }
            )
        elif task_type == "regr":
            train_values = np.asarray([float(label) for label in train_labels], dtype=np.float64)
            eval_values = np.asarray([float(label) for label in eval_labels], dtype=np.float64)
            preds = _knn_regress(train_feats, train_values, eval_feats, args.knn_k)
            if preds.size:
                residuals = preds - eval_values
                mae = float(np.mean(np.abs(residuals)))
                rmse = float(np.sqrt(np.mean(residuals ** 2)))
                denom = float(np.sum((eval_values - eval_values.mean()) ** 2))
                r2 = float(1.0 - np.sum(residuals ** 2) / denom) if denom > 0 else 0.0
            else:
                mae = rmse = r2 = 0.0
            results.update({"mae": mae, "rmse": rmse, "r2": r2})
        else:
            print(f"[neuco] Unsupported task type {task_type} in {label_path.name}, skipping.")
            continue

        out_path = write_json(
            output_dir / f"task_{task_name}.json",
            results,
        )
        output_paths.append(out_path)
        print(f"[neuco] Saved results for {task_name} to {out_path}")

    return output_paths


def _load_neuco_csv_embeddings(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or header[0] != "id":
            raise ValueError(f"Unexpected embedding CSV header in {csv_path}")
        ids: List[str] = []
        vectors: List[List[float]] = []
        for row in reader:
            if not row:
                continue
            ids.append(row[0])
            vectors.append([float(x) for x in row[1:]])
    if not vectors:
        raise ValueError(f"No embeddings found in {csv_path}")
    return ids, np.asarray(vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NeuCo limited-label evaluation with stratified sampling and k-NN probes."
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
    parser.add_argument("--feature", choices=["backbone", "projected"], default="backbone", help="Embedding space to use.")
    parser.add_argument("--knn-k", type=int, default=3, help="Number of neighbors for k-NN classification/regression.")
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling.")
    parser.add_argument("--task-filter", nargs="*", default=None, help="Optional list of task names to include.")
    parser.add_argument("--reuse-embeddings", action="store_true", help="Reuse existing NeuCo CSV if present.")
    parser.add_argument("--export-embeddings", action="store_true", help="Export embeddings CSV for reuse.")
    args = parser.parse_args()

    run_from_args(args)


if __name__ == "__main__":
    main()
