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
import copy
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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

# Compatibility shim for environments with NumPy>=2 and legacy deps
# (e.g., wandb/timm transitively imported via TorchGeo).
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # type: ignore[attr-defined]

from ciip.evaluation.export_neuco_embeddings import (
    E2SChallengeDataset,
    InputResizer,
)
from ciip.evaluation.normalization_utils import (
    build_normalization_transform,
    resolve_normalization_method_for_weights,
    select_ssl4eo_transform,
)
from ciip.evaluation.model_utils import (
    S2_WAVELENGTHS_UM,
    build_evaluation_adapter,
    EvaluationAdapter,
)
from ciip.evaluation.output_utils import build_model_tag


logger = logging.getLogger("neuco_layerwise")
_AUTO_NORMALIZATION_METHOD = "auto"
_VIT_HOOK_BLOCK_INDICES = (0, 1, 2, 3, 5, 7, 9, 11)


def _sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


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


def _coerce_paths(args: argparse.Namespace) -> argparse.Namespace:
    path_fields = (
        "checkpoint",
        "croma_weights",
        "neuco_root",
        "annotation_root",
        "train_csv",
        "val_csv",
        "test_csv",
        "benchmark_root",
        "benchmark_config",
        "output_dir",
        "plot_dir",
        "model_root",
        "summary_json",
    )
    for field in path_fields:
        value = getattr(args, field, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))
    modalities = getattr(args, "modalities", None)
    if isinstance(modalities, str):
        args.modalities = [modalities]
    train_fracs = getattr(args, "train_fracs", None)
    if isinstance(train_fracs, (float, int)):
        args.train_fracs = [float(train_fracs)]
    return args


def _resolve_checkpoint(args: argparse.Namespace) -> Optional[Path]:
    if getattr(args, "checkpoint", None) is not None:
        return Path(args.checkpoint)
    if args.model_type != "ciip_checkpoint":
        return None
    model_path = getattr(args, "model_path", None)
    if not model_path:
        raise ValueError("ciip_checkpoint requires --checkpoint or --model-path.")
    checkpoint_root = Path(args.model_root) / str(model_path) / "checkpoints"
    return checkpoint_root / f"epoch_{int(args.ciip_epoch)}.pt"


def _resolve_neuco_transform(
    *,
    image_size: int,
    model_in_channels: int,
    model_weights: Optional[str],
    model_type: str,
    normalization_method: Optional[str],
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    explicit = (normalization_method or "").strip().lower()
    if explicit in {"", _AUTO_NORMALIZATION_METHOD}:
        explicit = ""
        for key in (model_weights, model_type):
            if not key:
                continue
            try:
                transform = select_ssl4eo_transform(key)
                return transform, f"ssl4eo_transform:{str(key).lower()}"
            except Exception:
                continue

    method_arg = explicit if explicit else None
    method = resolve_normalization_method_for_weights(
        model_in_channels,
        method_arg,
        model_weights,
    )
    norm_layer = build_normalization_transform(method)
    transform = transforms.Compose(
        [
            InputResizer(image_size),
            norm_layer,
        ]
    )
    return transform, method


def _prepare_filtered_annotation_root(
    annotation_root: Path,
    *,
    output_dir: Path,
    excluded_tasks: Sequence[str],
) -> Path:
    excluded = {task.strip() for task in excluded_tasks if task and task.strip()}
    if not excluded:
        return annotation_root

    filtered_root = output_dir / "annotation_filtered"
    filtered_root.mkdir(parents=True, exist_ok=True)
    for existing in filtered_root.glob("*.csv"):
        existing.unlink()

    kept = 0
    skipped: List[str] = []
    for csv_path in sorted(annotation_root.glob("*.csv")):
        task_name = csv_path.stem.split("__", 1)[0]
        if task_name in excluded:
            skipped.append(csv_path.name)
            continue
        shutil.copy2(csv_path, filtered_root / csv_path.name)
        kept += 1

    if kept == 0:
        raise ValueError(
            f"All annotation CSVs were excluded by task filter {sorted(excluded)} under {annotation_root}."
        )
    logger.info(
        "Filtered NeuCo annotations: kept=%d skipped=%d excluded_tasks=%s",
        kept,
        len(skipped),
        sorted(excluded),
    )
    return filtered_root


class FeatureProbe:
    """Forward hook that pools activations and caches them on CPU."""

    def __init__(
        self,
        name: str,
        pooling: str = "gap",
        use_cls: bool = False,
        token_pooling: str = "mean",
        add_input_residual: bool = False,
        drop_cls_token: bool = False,
    ) -> None:
        self.name = name
        self.pooling = pooling
        self.use_cls = use_cls
        self.token_pooling = token_pooling
        self.add_input_residual = add_input_residual
        self.drop_cls_token = drop_cls_token
        self._features: List[torch.Tensor] = []

    def __call__(self, _module: nn.Module, _inputs, output: torch.Tensor) -> None:
        feat = output.detach()
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if self.add_input_residual and isinstance(_inputs, (tuple, list)) and _inputs:
            residual = _inputs[0]
            if isinstance(residual, torch.Tensor):
                residual = residual.detach()
                if residual.shape == feat.shape:
                    feat = feat + residual
        if feat.ndim == 3:
            # Standardize token layout to (batch, tokens, dim).
            # CIIP/OpenCLIP transformer blocks emit (tokens, batch, dim).
            if feat.shape[0] > feat.shape[1]:
                feat = feat.permute(1, 0, 2)
            if self.use_cls:
                feat = feat[:, 0]  # CLS token
            elif self.drop_cls_token and feat.shape[1] > 1:
                feat = feat[:, 1:, :]
                if self.token_pooling == "mean_var":
                    mean = feat.mean(dim=1)
                    var = feat.var(dim=1, unbiased=False)
                    feat = torch.cat([mean, var], dim=1)
                else:
                    feat = feat.mean(dim=1)
            elif self.token_pooling == "mean_var":
                mean = feat.mean(dim=1)
                var = feat.var(dim=1, unbiased=False)
                feat = torch.cat([mean, var], dim=1)
            else:
                feat = feat.mean(dim=1)
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


def _collect_vit_blocks(encoder: nn.Module) -> List[Tuple[nn.Module, bool]]:
    # torchvision-style ViT
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
        return [(module, False) for module in encoder.encoder.layers]
    # timm-style ViT
    if hasattr(encoder, "blocks"):
        return [(module, False) for module in encoder.blocks]
    # CIIP/OpenCLIP-style ViT (custom Transformer with resblocks)
    transformer = getattr(encoder, "transformer", None)
    if transformer is not None and hasattr(transformer, "resblocks"):
        return [(module, False) for module in transformer.resblocks]
    # CROMA-style ViT (BaseTransformer with per-layer [Attention, FFN] pairs).
    if transformer is not None and hasattr(transformer, "layers"):
        blocks: List[Tuple[nn.Module, bool]] = []
        for layer in transformer.layers:
            if isinstance(layer, (list, tuple, nn.ModuleList)) and len(layer) >= 2:
                ffn = layer[1]
                if isinstance(ffn, nn.Module):
                    # BaseTransformer forward does: x = ffn(x) + x.
                    # Hook FFN and add input residual in probe to match block output.
                    blocks.append((ffn, True))
                    continue
            if isinstance(layer, nn.Module):
                blocks.append((layer, False))
        return blocks
    return []


def _encoder_has_cls_token(encoder: nn.Module) -> bool:
    for attr in ("class_embedding", "class_token", "cls_token"):
        token = getattr(encoder, attr, None)
        if token is not None:
            return True
    return False


def _resolve_vit_backbone_hook_module(encoder: nn.Module) -> nn.Module:
    """Prefer a pre-pooling token module so ViT token pooling flags are honored."""
    candidates = [
        getattr(encoder, "ln_post", None),  # CIIP/OpenCLIP-style ViT
        getattr(encoder, "norm", None),  # timm-style ViT
        getattr(getattr(encoder, "encoder", None), "ln", None),  # torchvision ViT encoder
        getattr(getattr(encoder, "encoder", None), "ln_post", None),
    ]
    for module in candidates:
        if isinstance(module, nn.Module):
            return module
    return encoder


def _register_vit_probes(
    encoder: nn.Module,
    *,
    pooling: str = "gap",
    token_pooling: str = "cls",
) -> Tuple[Dict[str, FeatureProbe], List[torch.utils.hooks.RemovableHandle]]:
    probes: Dict[str, FeatureProbe] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []
    use_cls = token_pooling == "cls"
    drop_cls_token = token_pooling in {"mean", "mean_var"} and _encoder_has_cls_token(encoder)

    blocks = _collect_vit_blocks(encoder)
    if blocks:
        n = len(blocks)
        idxs = [idx for idx in _VIT_HOOK_BLOCK_INDICES if idx < n]
        for idx in idxs:
            block_module, add_input_residual = blocks[idx]
            probe = FeatureProbe(
                f"vit_block{idx}",
                pooling=pooling,
                use_cls=use_cls,
                token_pooling=token_pooling,
                add_input_residual=add_input_residual,
                drop_cls_token=drop_cls_token,
            )
            probes[probe.name] = probe
            handles.append(block_module.register_forward_hook(probe))

    # Backbone representation: prefer pre-pooling token tensor so token pooling
    # (cls/mean/mean_var) is applied consistently.
    backbone_module = _resolve_vit_backbone_hook_module(encoder)
    probe = FeatureProbe(
        "backbone",
        pooling=pooling,
        use_cls=use_cls,
        token_pooling=token_pooling,
        drop_cls_token=drop_cls_token,
    )
    probes["backbone"] = probe
    handles.append(backbone_module.register_forward_hook(probe))
    return probes, handles


def _build_loader(
    data_root: Path,
    *,
    modalities: Sequence[str],
    batch_size: int,
    num_workers: int,
    seasons: int,
    model_in_channels: int,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> DataLoader:
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
    layer_names = list(metrics.keys())
    vit_blocks = []
    for name in layer_names:
        match = re.fullmatch(r"vit_block(\d+)", name)
        if match:
            vit_blocks.append((int(match.group(1)), name))

    # For ViT encoders, keep true depth order: block0..N, norm, backbone.
    if vit_blocks:
        ordered = [name for _, name in sorted(vit_blocks, key=lambda item: item[0])]
        if "backbone" in metrics:
            ordered.append("backbone")
        extras = sorted([name for name in layer_names if name not in ordered and name != "vit_norm"])
        return ordered + extras

    canonical = ["stem", "layer1", "layer2", "layer3", "layer4", "attnpool", "backbone"]
    available = [name for name in canonical if name in metrics]
    extras = sorted([name for name in layer_names if name not in canonical])
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
    vit_token_pooling: str = "cls",
    model_in_channels=12
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    encoder = getattr(adapter, "encoder_s2", None)
    adapter = adapter.to(device).eval() if adapter is not None else None
    if encoder:
        encoder = encoder.to(device).eval()
    else:
        encoder = adapter
    if _collect_vit_blocks(encoder):
        probes, handles = _register_vit_probes(encoder, pooling=pooling, token_pooling=vit_token_pooling)
    else:
        probes, handles = _register_resnet_probes(encoder, pooling=pooling)
    logger.info("Registered probes: %s", ", ".join(probes.keys()))
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
    parser.add_argument("--models-json", type=Path, help="Optional JSON file listing model specs to evaluate.")
    parser.add_argument("--summary-json", type=Path, help="Optional summary output path for multi-model runs.")
    parser.add_argument("--model-type", type=str, default="ciip_checkpoint", help="Model type (ciip_checkpoint, backbone_only, torchgeo_resnet50, croma, llama3_ms_clip_base).")
    parser.add_argument("--checkpoint", type=Path, help="Path to CIIP checkpoint (for model-type=ciip_checkpoint).")
    parser.add_argument("--model-path", type=str, help="CIIP model folder under --model-root when checkpoint is omitted.")
    parser.add_argument("--model-root", type=Path, default=Path("/local/ms-data/SSL4EO/model/"), help="Root for CIIP checkpoints resolved from model-path.")
    parser.add_argument("--ciip-epoch", type=int, default=10, help="Checkpoint epoch when resolving from --model-path.")
    parser.add_argument("--ciip-framework", type=str, default=None, help="Optional CIIP framework override.")
    parser.add_argument("--ciip-model-source", choices=["current", "posenc"], default="current", help="CIIP implementation source.")
    parser.add_argument("--model-weights", type=str, help="Predefined weights key for backbone-only/torchgeo/croma/openclip.")
    parser.add_argument("--model-in-channels", type=int, default=13, help="Input channels for model instantiation.")
    parser.add_argument("--croma-weights", type=Path, help="Path to CROMA weights when model-type=croma.")
    parser.add_argument("--croma-image-resolution", type=int, default=120, help="Resolution for CROMA.")
    parser.add_argument("--normalization-method", type=str, default=_AUTO_NORMALIZATION_METHOD, help="Normalization method or 'auto' to resolve from normalization_utils.")
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
    parser.add_argument("--pooling", choices=["gap", "gap_max", "mean_var"], default="mean_var", help="Pooling for 4D activations.")
    parser.add_argument(
        "--vit-token-pooling",
        choices=["cls", "mean", "mean_var"],
        default="cls",
        help="Token pooling for ViT 3D token outputs.",
    )
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
    parser.add_argument(
        "--exclude-neuco-tasks",
        nargs="*",
        default=["random_cls", "nodata", "random_reg"],
        help="NeuCo task-name prefixes to exclude from benchmark (based on '<task>__*.csv').",
    )
    return parser.parse_args()


def _run_single_model(args: argparse.Namespace, device: torch.device) -> Path:
    args = _coerce_paths(args)
    checkpoint = _resolve_checkpoint(args)
    ciip_model_source = str(args.ciip_model_source).strip().lower().replace("-", "_")
    if ciip_model_source not in {"current", "posenc"}:
        raise ValueError(
            f"Unsupported ciip_model_source={args.ciip_model_source!r}. Expected 'current' or 'posenc'."
        )
    if args.model_type == "ciip_checkpoint":
        logger.info(
            "Loading CIIP checkpoint adapter with source=%s framework=%s checkpoint=%s",
            ciip_model_source,
            args.ciip_framework,
            checkpoint,
        )
    adapter = build_evaluation_adapter(
        model_type=args.model_type,
        checkpoint=checkpoint,
        model_weights=args.model_weights,
        in_chans=args.model_in_channels,
        croma_weights=args.croma_weights,
        croma_image_resolution=args.croma_image_resolution,
        ciip_framework=args.ciip_framework,
        ciip_model_source=ciip_model_source,
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

    transform, norm_label = _resolve_neuco_transform(
        image_size=args.image_size,
        model_in_channels=args.model_in_channels,
        model_weights=args.model_weights,
        model_type=args.model_type,
        normalization_method=args.normalization_method,
    )
    logger.info("Using NeuCo transform: %s", norm_label)

    output_dir = Path(args.output_dir)
    if getattr(args, "_from_models_json", False):
        model_tag = build_model_tag(
            model_type=args.model_type,
            model_weights=args.model_weights,
            model_path=args.model_path,
            ciip_epoch=args.ciip_epoch,
        )
        output_dir = output_dir / model_tag / _sanitize_label(norm_label)

    loader = _build_loader(
        args.neuco_root,
        modalities=args.modalities,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seasons=args.seasons,
        model_in_channels=args.model_in_channels,
        transform=transform,
    )

    plot_dir = args.plot_dir if args.plot_dir is not None else output_dir
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
            if npz_file.stem == "vit_norm":
                continue
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
        if "croma" not in args.model_type.lower():
            features, ids = extract_layerwise_features(
                encoder,
                loader,
                device=device,
                pooling=args.pooling,
                vit_token_pooling=args.vit_token_pooling,
                model_in_channels=args.model_in_channels,
            )
        else:
            features, ids = extract_layerwise_features(
                adapter,
                loader,
                device=device,
                pooling=args.pooling,
                vit_token_pooling=args.vit_token_pooling,
                model_in_channels=args.model_in_channels,
            )
        save_embeddings(features, ids, embeddings_dir)

    # Export per-layer CSVs and run official NeuCo benchmark (aligns with unified_evaluation).
    task_scores: Dict[str, Dict[str, float]] = {}
    if args.annotation_root is not None:
        benchmark_annotation_root = _prepare_filtered_annotation_root(
            args.annotation_root,
            output_dir=output_dir,
            excluded_tasks=args.exclude_neuco_tasks,
        )
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
                    annotation_root=benchmark_annotation_root,
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
            _plot_neuco_task_layers(task_scores, layers_order, plot_dir / "neuco_task_layers.png")
            print("Saved NeuCo task layer plots to", plot_dir / "neuco_task_layers.png")

    if args.annotation_root is None and not any([args.train_csv, args.val_csv, args.test_csv]):
        logger.info("No annotation path provided; skipping linear/kNN evaluation.")
        return output_dir

    return output_dir


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    if args.models_json is None:
        _run_single_model(args, device)
        return

    models, defaults = _load_model_specs(args.models_json)
    if not models:
        raise ValueError("No models found in the provided JSON file.")

    outputs: List[Dict[str, str]] = []
    for idx, model_spec in enumerate(models, start=1):
        run_args = _merge_args(args, defaults)
        run_args = _merge_args(run_args, model_spec)
        run_args._from_models_json = True
        run_args = _coerce_paths(run_args)
        label = (
            getattr(run_args, "model_path", None)
            or getattr(run_args, "model_weights", None)
            or run_args.model_type
        )
        logger.info("Running model %d/%d: %s", idx, len(models), label)
        run_output = _run_single_model(run_args, device)
        outputs.append({"model": json.dumps(model_spec), "output_dir": str(run_output)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps({"results": outputs}, indent=2))
        logger.info("Wrote summary JSON to %s", args.summary_json)


if __name__ == "__main__":
    main()
