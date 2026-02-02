"""Layerwise linear probe/kNN evaluation on NeuCo data.

This script mirrors the NeuCo export path used in ``unified_evaluation.py``,
but taps intermediate ResNet blocks to study where supervision is most
effective. It:
  * loads the S2 encoder from a CIIP-style checkpoint,
  * registers forward hooks on the stem + residual blocks,
  * pools 4D activations (GAP, GAP+GMP, or mean+var),
  * runs Logistic Regression and kNN probes for any provided label splits,
  * optionally exports per-layer embeddings to disk for external benchmarking.

Expected label splits: CSV files with ``id`` and ``label`` columns. By default
the script looks for ``train.csv``, ``val.csv`` and ``test.csv`` under
``--annotation-root``; you can override with explicit ``--train-csv`` etc.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from ciip.evaluation.export_neuco_embeddings import (
    Divideby10000Normalize,
    E2SChallengeDataset,
    InputResizer,
)
from ciip.evaluation.model_utils import (
    S2_WAVELENGTHS_UM,
    build_model_from_checkpoint,
    build_evaluation_adapter,
    EvaluationAdapter,
)


logger = logging.getLogger("neuco_layerwise")


class FeatureProbe:
    """Forward hook that pools activations and caches them on CPU."""

    def __init__(self, name: str, pooling: str = "gap", use_cls: bool = False) -> None:
        self.name = name
        self.pooling = pooling
        self.use_cls = use_cls
        self._features: List[torch.Tensor] = []

    def __call__(self, _module: nn.Module, _inputs, output: torch.Tensor) -> None:
        feat = output.detach()
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim == 3 and self.use_cls:
            feat = feat[:, 0]  # CLS token
        elif feat.ndim == 3:
            print('WARNING JULIA -- SUSPSICIOUS DIM REDUCTION OF non-transformer layer outputs')
            feat = feat.mean(dim=1)  # mean over tokens if no CLS requested
        elif feat.ndim == 4:
            gap = feat.mean(dim=(2, 3))
            if self.pooling == "gap_max":
                gmp = feat.amax(dim=(2, 3))
                feat = torch.cat([gap, gmp], dim=1)
            elif self.pooling == "mean_var":
                var = feat.var(dim=(2, 3), unbiased=False)
                feat = torch.cat([gap, var], dim=1)
            else:
                feat = gap
        elif feat.ndim > 2:
            feat = feat.flatten(start_dim=1)
        self._features.append(feat.cpu())

    def stacked(self) -> torch.Tensor:
        if not self._features:
            return torch.empty(0)
        return torch.cat(self._features, dim=0)

    def pop_last(self) -> torch.Tensor:
        if not self._features:
            raise RuntimeError(f"No cached activations for probe '{self.name}'.")
        return self._features.pop()

    def reset(self) -> None:
        self._features.clear()


def _maybe_identity(module: Optional[nn.Module]) -> Optional[nn.Module]:
    if isinstance(module, nn.Module):
        return module
    return None


def _register_resnet_probes(
    encoder: nn.Module,
    *,
    pooling: str = "gap",
) -> Tuple[Dict[str, FeatureProbe], List[torch.utils.hooks.RemovableHandle]]:
    """Attach probes to common ResNet blocks."""

    probes: Dict[str, FeatureProbe] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _add(name: str, module: Optional[nn.Module]) -> None:
        module = _maybe_identity(module)
        if module is None:
            return
        probe = FeatureProbe(name, pooling=pooling)
        probes[name] = probe
        handles.append(module.register_forward_hook(probe))

    # Stem: after relu (pre-pool) if available; else after pool.
    relu = getattr(encoder, "relu", None) or getattr(encoder, "relu1", None)
    if relu is not None:
        _add("stem", relu)
    elif hasattr(encoder, "maxpool"):
        _add("stem", encoder.maxpool)
    elif hasattr(encoder, "avgpool"):
        _add("stem", encoder.avgpool)

    # Residual blocks
    _add("layer1", getattr(encoder, "layer1", None))
    _add("layer2", getattr(encoder, "layer2", None))
    _add("layer3", getattr(encoder, "layer3", None))
    _add("layer4", getattr(encoder, "layer4", None))

    # Final pooled features (attnpool or avgpool)
    _add("attnpool", getattr(encoder, "attnpool", None))
    _add("backbone", encoder)  # catch the encoder output itself
    return probes, handles


def _collect_vit_blocks(encoder: nn.Module) -> List[nn.Module]:
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
        return list(encoder.encoder.layers)
    if hasattr(encoder, "blocks"):
        return list(encoder.blocks)
    return []


def _register_vit_probes(
    encoder: nn.Module,
    *,
    pooling: str = "gap",
) -> Tuple[Dict[str, FeatureProbe], List[torch.utils.hooks.RemovableHandle]]:
    probes: Dict[str, FeatureProbe] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    blocks = _collect_vit_blocks(encoder)
    if blocks:
        n = len(blocks)
        idxs = sorted(set([0, max(0, n // 3), max(0, 2 * n // 3), n - 1]))
        for idx in idxs:
            probe = FeatureProbe(f"vit_block{idx}", pooling=pooling, use_cls=True)
            probes[probe.name] = probe
            handles.append(blocks[idx].register_forward_hook(probe))

    # Final norm if available
    norm_mod = None
    for attr in ("ln", "norm"):
        norm_mod = getattr(encoder, attr, None)
        if norm_mod is not None:
            break
    if norm_mod is not None:
        probe = FeatureProbe("vit_norm", pooling=pooling, use_cls=True)
        probes[probe.name] = probe
        handles.append(norm_mod.register_forward_hook(probe))

    # Backbone output (encoder forward) – use CLS pooling
    probe = FeatureProbe("backbone", pooling=pooling, use_cls=True)
    probes["backbone"] = probe
    handles.append(encoder.register_forward_hook(probe))
    return probes, handles


def _build_loader(
    data_root: Path,
    *,
    modalities: Sequence[str],
    batch_size: int,
    num_workers: int,
    seasons: int,
    image_size: int,
    model_in_channels: int,
) -> DataLoader:
    transform = transforms.Compose(
        [
            InputResizer(image_size),
            Divideby10000Normalize(),
        ]
    )
    use_rgb = model_in_channels == 3
    dataset = E2SChallengeDataset(
        data_path=str(data_root),
        modalities=list(modalities),
        seasons=seasons,
        concat=True,
        output_file_name=True,
        transform=transform,
        rgb=use_rgb,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )




def _ordered_layers(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    canonical = ["stem", "layer1", "layer2", "layer3", "layer4", "attnpool", "backbone"]
    available = [k for k in canonical if k in metrics]
    # append any extras deterministically
    extras = sorted([k for k in metrics.keys() if k not in canonical])
    return available + extras




def _export_layer_csv(features: np.ndarray, ids: List[str], out_path: Path) -> None:
    """Write embeddings to NeuCo submission CSV format: id,e0,..."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dim = features.shape[1]
    header = ["id"] + [f"e{i}" for i in range(dim)]
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for sample_id, row in zip(ids, features):
            values = ",".join(str(float(x)) for x in row)
            f.write(f"{sample_id},{values}\n")
    logger.info("Wrote embeddings CSV to %s", out_path)


def _run_neuco_benchmark(
    submission: Path,
    *,
    annotation_root: Path,
    output_dir: Path,
    benchmark_root: Path,
    config_path: Path,
    method_name: str,
    phase: str,
    embedding_dim: int,
) -> None:
    main_py = benchmark_root / "main.py"
    cmd = [
        "python",
        str(main_py),
        "--annotation_path",
        str(annotation_root),
        "--output_dir",
        str(output_dir),
        "--config",
        str(config_path),
        "--method_name",
        method_name,
        "--phase",
        phase,
        "--submission_file",
        str(submission),
        "--embedding_dim",
        str(embedding_dim),
    ]
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    logger.info("Running NeuCo benchmark: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _latest_benchmark_result(output_dir: Path, phase: str, method_prefix: str) -> Optional[Path]:
    """Return the newest results_summary.json under output_dir/phase matching prefix."""
    phase_dir = output_dir / phase
    if not phase_dir.exists():
        return None
    candidates = []
    for exp_dir in phase_dir.iterdir():
        if not exp_dir.name.startswith(method_prefix):
            continue
        summary = exp_dir / "results_summary.json"
        if summary.exists():
            candidates.append((summary.stat().st_mtime, summary))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _plot_neuco_task_layers(
    task_to_layer: Dict[str, Dict[str, float]],
    layers: Sequence[str],
    out_path: Path,
) -> None:
    if not task_to_layer:
        return
    tasks = sorted(task_to_layer.keys())
    x = np.arange(len(layers))
    fig, axes = plt.subplots(len(tasks), 1, figsize=(8, 2.5 * len(tasks)), sharex=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        scores = [task_to_layer[task].get(layer, np.nan) for layer in layers]
        ax.plot(x, scores, marker="o", color="#1f77b4")
        ax.set_title(task)
        ax.set_ylabel("Score")
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(layers, rotation=25, ha="right")
    axes[-1].set_xlabel("Layer depth")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def extract_layerwise_features(
    adapter: EvaluationAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    pooling: str,
    model_in_channels=12
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    encoder = getattr(adapter, "encoder_s2", None)
    adapter = adapter.to(device).eval() if adapter is not None else None
    if encoder:
        encoder = encoder.to(device).eval()
    else:
        encoder = adapter
    if _collect_vit_blocks(encoder):
        probes, handles = _register_vit_probes(encoder, pooling=pooling)
    else:
        probes, handles = _register_resnet_probes(encoder, pooling=pooling)
    aggregated: Dict[str, List[torch.Tensor]] = {name: [] for name in probes}
    ids: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            if not isinstance(batch, dict):
                raise TypeError("Expected dict batch with 'data' and 'file_name'.")
            images = batch["data"].to(device, non_blocking=True)
            ids.extend([str(x) for x in batch["file_name"]])

            seasons = 1
            if images.ndim == 5:
                batch_size, seasons, c, h, w = images.shape
                images = images.reshape(batch_size * seasons, c, h, w)

            from ciip.evaluation.unified_evaluation import _align_s2_channels
            images = _align_s2_channels(images, model_in_channels)
            # if model is dofa

            # print(encoder.__class__.__name__.lower())
            if "dofa" in encoder.__class__.__name__.lower():
                _ = encoder(images, wavelengths=S2_WAVELENGTHS_UM)
            # elif "vit" in encoder.__class__.__name__.lower(): # USE THIS FOR CROMA ONLY
            #     _ = encoder(images, attn_bias=None)
            elif "croma" in adapter.__class__.__name__.lower():
                _ = adapter._encode_modality(images, modality="s2")
            else:
                _ = encoder(images)

            for name, probe in probes.items():
                chunk = probe.pop_last()  # shape (batch*seasons, dim)
                if seasons > 1:
                    # reshape back and mean over seasons
                    chunk = chunk.reshape(batch_size, seasons, -1).mean(dim=1)
                aggregated[name].append(chunk.cpu())

    features = {name: torch.cat(tensors, dim=0).numpy() if tensors else np.empty((0,)) for name, tensors in aggregated.items()}
    for handle in handles:
        handle.remove()
    return features, ids


def save_embeddings(
    features: Dict[str, np.ndarray],
    ids: List[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    id_array = np.asarray(ids)
    for name, arr in features.items():
        out_path = out_dir / f"{name}.npz"
        np.savez(out_path, ids=id_array, features=arr)
        logger.info("Saved %s features to %s (shape=%s)", name, out_path, arr.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layerwise NeuCo probes (linear/kNN).")
    parser.add_argument("--model-type", type=str, default="ciip_checkpoint", help="Model type (ciip_checkpoint, backbone_only, torchgeo_resnet50, croma, llama3_ms_clip_base).")
    parser.add_argument("--checkpoint", type=Path, help="Path to CIIP checkpoint (for model-type=ciip_checkpoint).")
    parser.add_argument("--model-weights", type=str, help="Predefined weights key for backbone-only/torchgeo/croma/openclip.")
    parser.add_argument("--model-in-channels", type=int, default=13, help="Input channels for model instantiation.")
    parser.add_argument("--croma-weights", type=Path, help="Path to CROMA weights when model-type=croma.")
    parser.add_argument("--croma-image-resolution", type=int, default=120, help="Resolution for CROMA.")
    parser.add_argument("--neuco-root", type=Path, required=True, help="Root of NeuCo data (modalities subfolders).")
    parser.add_argument("--annotation-root", type=Path, help="Directory with train/val/test CSV label files.")
    parser.add_argument("--train-csv", type=Path, help="Optional explicit train CSV.")
    parser.add_argument("--val-csv", type=Path, help="Optional explicit val CSV.")
    parser.add_argument("--test-csv", type=Path, help="Optional explicit test CSV.")
    parser.add_argument("--label-column", default="label", help="Label column name inside CSV files.")
    parser.add_argument("--modalities", nargs="+", default=["s2l2a"], help="NeuCo modalities (default: s2l2a).")
    parser.add_argument("--seasons", type=int, default=1, help="Number of seasons to average/load.")
    parser.add_argument("--image-size", type=int, default=224, help="Resize shorter side to this value.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pooling", choices=["gap", "gap_max", "mean_var"], default="gap", help="Pooling for 4D activations.")
    parser.add_argument("--train-fracs", type=float, nargs="+", default=[1.0], help="Train fractions for probes.")
    parser.add_argument("--knn-k", type=int, default=20, help="Base k for kNN probes.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store outputs.")
    parser.add_argument("--plot-dir", type=Path, help="Optional directory for layerwise plots (defaults to output-dir).")
    parser.add_argument("--device", default=None, help="Torch device string (default: cuda if available).")
    parser.add_argument("--task-file", default="random_cls__cls.csv", help="NeuCo label file to use when explicit splits are absent.")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction when deriving splits from task file.")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed for any random splitting.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("/local/ms-data/NeuCo-Bench/benchmark"), help="Path to NeuCo benchmark repo root (directory containing main.py).")
    parser.add_argument("--benchmark-config", type=Path, default=Path("/local/ms-data/NeuCo-Bench/benchmark/config.yaml"), help="NeuCo benchmark config file.")
    parser.add_argument("--benchmark-phase", type=str, default="testing", help="NeuCo benchmark phase.")
    parser.add_argument("--method-prefix", type=str, default="layer", help="Prefix for benchmark method names per layer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    encoder: nn.Module
    adapter: Optional[EvaluationAdapter] = None

    if args.model_type == "ciip_checkpoint":
        if args.checkpoint is None:
            raise ValueError("checkpoint must be provided for ciip_checkpoint")
        model, _ = build_model_from_checkpoint(args.checkpoint)
        encoder = getattr(model, "encoder_s2", None)
        if encoder is None:
            raise AttributeError("Checkpoint does not expose encoder_s2.")
    else:
        adapter = build_evaluation_adapter(
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            model_weights=args.model_weights,
            in_chans=args.model_in_channels,
            croma_weights=args.croma_weights,
            croma_image_resolution=args.croma_image_resolution,
        )
        encoder = getattr(adapter, "encoder_s2", None)
        if encoder is None:
            encoder = getattr(adapter, "base_model", None)
        if encoder is None:
            raise AttributeError("Unable to find an S2 encoder or base_model on adapter.")

    if hasattr(encoder, "fc") and isinstance(encoder.fc, nn.Module):
        try:
            if not isinstance(encoder.fc, nn.Identity):
                logger.info("Replacing encoder_s2.fc with Identity for backbone features.")
                encoder.fc = nn.Identity()
        except Exception:
            pass

    loader = _build_loader(
        args.neuco_root,
        modalities=args.modalities,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seasons=args.seasons,
        image_size=args.image_size,
        model_in_channels=args.model_in_channels,
    )

    output_dir = args.output_dir
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    features: Dict[str, np.ndarray] = {}
    ids: List[str] = []

    # Load cached embeddings if present
    cached_npz = sorted(embeddings_dir.glob("*.npz"))
    if cached_npz:
        logger.info("Loading cached embeddings from %s", embeddings_dir)
        ids_loaded = False
        for npz_file in cached_npz:
            data = np.load(npz_file, allow_pickle=True)
            if "features" in data:
                features[npz_file.stem] = data["features"]
                if not ids_loaded and "ids" in data:
                    ids = data["ids"].tolist()
                    ids_loaded = True
            elif data.files:
                features[npz_file.stem] = data[data.files[0]]
                if not ids_loaded and len(data.files) > 1:
                    ids = data[data.files[1]].tolist()
                    ids_loaded = True
        if not ids and features:
            logger.warning("Cached embeddings found but no ids present; re-extracting.")
            features.clear()
            cached_npz = []
    else:
        if not embeddings_dir.exists():
            embeddings_dir.mkdir(parents=True, exist_ok=True)
        # if model is not croma
        if "croma" not in args.model_type:
            features, ids = extract_layerwise_features(
                encoder,
                loader,
                device=device,
                pooling=args.pooling,
                model_in_channels=args.model_in_channels
            )
        else:
            features, ids = extract_layerwise_features(
                adapter,
                loader,
                device=device,
                pooling=args.pooling,
                model_in_channels=args.model_in_channels
            )
        save_embeddings(features, ids, embeddings_dir)

    # Export per-layer CSVs and run official NeuCo benchmark (aligns with unified_evaluation).
    task_scores: Dict[str, Dict[str, float]] = {}
    if args.annotation_root is not None:
        csv_dir = output_dir / "neuco_csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in features.items():
            if arr.size == 0 or arr.ndim < 2:
                continue
            csv_path = csv_dir / f"{name}.csv"
            _export_layer_csv(arr, ids, csv_path)
            try:
                bm_out_dir = output_dir / "neuco_benchmark" / name
                _run_neuco_benchmark(
                    csv_path,
                    annotation_root=args.annotation_root,
                    output_dir=bm_out_dir,
                    benchmark_root=args.benchmark_root,
                    config_path=args.benchmark_config,
                    method_name=f"{args.method_prefix}_{name}",
                    phase=args.benchmark_phase,
                    embedding_dim=arr.shape[1],
                )
                summary = _latest_benchmark_result(bm_out_dir, args.benchmark_phase, f"{args.method_prefix}_{name}")
                if summary is not None:
                    target_summary = output_dir / f"neuco_benchmark_summary_{name}.json"
                    target_summary.write_text(Path(summary).read_text())
                    logger.info("Saved NeuCo benchmark summary for %s to %s", name, target_summary)
                    try:
                        data = json.loads(Path(summary).read_text())
                        task_results = data.get("task_results", {})
                        for task, payload in task_results.items():
                            raw = payload.get("raw_score") if isinstance(payload, dict) else payload
                            if raw is None:
                                continue
                            task_scores.setdefault(task, {})[name] = float(raw)
                    except Exception as exc:
                        logger.warning("Failed to parse NeuCo summary for %s: %s", name, exc)
                else:
                    logger.warning("No results_summary.json found for %s under %s", name, bm_out_dir)
            except Exception as exc:
                logger.warning("NeuCo benchmark failed for %s: %s", name, exc)
        if task_scores:
            layers_order = _ordered_layers(features)
            _plot_neuco_task_layers(task_scores, layers_order, output_dir / "neuco_task_layers.png")
            print('Saved NeuCo task layer plots to', output_dir / "neuco_task_layers.png")

    if args.annotation_root is None and not any([args.train_csv, args.val_csv, args.test_csv]):
        logger.info("No annotation path provided; skipping linear/kNN evaluation.")
        return

    # plot the _run_neuco_benchmark results


if __name__ == "__main__":
    main()
