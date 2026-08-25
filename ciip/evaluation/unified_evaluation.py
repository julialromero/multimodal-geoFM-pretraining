"""Concise end-to-end evaluation entrypoint.

This module provides a single callable ``run_full_evaluation`` that executes the
three evaluation pipelines requested by the research team plus the optional
Lorentz-specific visualisations.  The implementation intentionally reuses the
utilities that already exist across the codebase (``linearprobe_comparison``,
``export_neuco_embeddings`` and ``ciip/visualization/ssl4eo``) instead of invoking
their CLI entrypoints so that everything can be orchestrated from Python.

The function expects a ``ModelEvalConfig`` describing the dataset locations,
output directory and model source.  By default it consumes CIIP checkpoints,
but users can also request TorchGeo ResNet50 encoders initialised with the
Sentinel-2 DINO or MoCo weights.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

# from ciip.evaluation.ssl4eo_retrieval import compute_cross_modal_retrieval
import subprocess


from ciip.evaluation.unified_types import EmbeddingBundle, ModelEvalConfig
from ciip.evaluation.unified_features import (
    EUROSAT_S1_BANDS,
    _build_bigearthnet_loaders,
    _build_eurosat_loaders,
    _build_neuco_loader,
    _extract_embeddings,
    _extract_ssl4eo_embeddings,
    _infer_model_in_channels,
    _print_band_stats,
    _resolve_eurosat_bands,
    _use_adapter_modality,
)
from ciip.evaluation.unified_presentation import (
    _export_neuco,
    _plot_eurosat_tsne,
    _run_embedding_diagnostics,
    _run_hyperbolic_visualisations,
)
from ciip.models.evaluation_adapters import build_evaluation_adapter
from ciip.model_ciip import LorentzCIIP
import os


## EUROSAT STANDARDIZATION VALUES


##

# BigEarthNet uses Sentinel-2 imagery without B10 (12 bands).


from ciip.visualization.ssl4eo.hyperbolic_visualization import (  # type: ignore
    _extract_curvature_from_state,
)
from ciip.evaluation.normalization import (
    NORMALIZATION_METHODS,
    resolve_normalization_method_for_weights,
)
from ciip.evaluation.result_records import (
    ensure_dir,
    write_run_manifest,
)


@dataclass
@dataclass

### FOR EUROSAT and NEUCOBENCH


def _run_linear_probe(
    config: ModelEvalConfig,
    embeddings: Dict[str, EmbeddingBundle],
    *,
    output_dir: Path,
    label: str,
    norm_method: str,
    slice_dim: Optional[int] = None,
    feature_override: Optional[str] = None,
) -> None:
    percents = (0.01, 0.1, 1.0)
    rng = np.random.default_rng(config.random_seed)

    def _maybe_slice(features: np.ndarray) -> np.ndarray:
        if slice_dim is None:
            return features
        if features.shape[1] < slice_dim:
            raise ValueError(
                f"Requested matryoshka dim {slice_dim} exceeds feature dim {features.shape[1]}."
            )
        return features[:, :slice_dim]

    def _evaluate_logreg(feature_key: str, batch_norm: bool) -> Dict[float, Dict[str, float]]:
        results: Dict[float, Dict[str, float]] = {}
        # print(embeddings)
        train = embeddings["train"]
        val = embeddings["val"]
        test = embeddings["test"]

        # raise NotImplementedError("Debugging embeddings extraction")

        train_feats = _maybe_slice(getattr(train, feature_key))
        val_feats = _maybe_slice(getattr(val, feature_key))
        test_feats = _maybe_slice(getattr(test, feature_key))

        print(
            f"Linear probe ({label}) using '{feature_key}' embeddings: dim={train_feats.shape[1]} "
            f"(batch_norm={batch_norm}, slice_dim={slice_dim})"
        )

        if batch_norm:
            mean = train_feats.mean(axis=0, keepdims=True)
            std = train_feats.std(axis=0, keepdims=True) + 1e-10
            train_feats = (train_feats - mean) / std
            val_feats = (val_feats - mean) / std
            test_feats = (test_feats - mean) / std

        has_multi = (
            train.multi_labels is not None
            and val.multi_labels is not None
            and test.multi_labels is not None
        )

        for pct in percents:
            n_samples = max(1, int(pct * len(train_feats)))
            indices = rng.choice(len(train_feats), size=n_samples, replace=False)

            clf = LogisticRegression(max_iter=4000, multi_class="multinomial", solver="lbfgs")
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

            if has_multi:
                ovr = OneVsRestClassifier(LogisticRegression(max_iter=4000, solver="lbfgs"))
                ovr.fit(train_feats[indices], train.multi_labels[indices])
                val_scores = ovr.predict_proba(val_feats)
                test_scores = ovr.predict_proba(test_feats)
                results[pct]["val_map_micro"] = float(
                    average_precision_score(val.multi_labels, val_scores, average="micro")
                )
                results[pct]["test_map_micro"] = float(
                    average_precision_score(test.multi_labels, test_scores, average="micro")
                )

        return results

    def _evaluate_knn(
        feature_key: str, batch_norm: bool, n_neighbors: int = 20
    ) -> Dict[float, Dict[str, float]]:
        """Use train features as keys and classify val/test with kNN (k=20) across the same percents."""
        results: Dict[float, Dict[str, float]] = {}
        train = embeddings["train"]
        val = embeddings["val"]
        test = embeddings["test"]

        train_feats = _maybe_slice(getattr(train, feature_key))
        val_feats = _maybe_slice(getattr(val, feature_key))
        test_feats = _maybe_slice(getattr(test, feature_key))

        if batch_norm:
            mean = train_feats.mean(axis=0, keepdims=True)
            std = train_feats.std(axis=0, keepdims=True) + 1e-10
            train_feats = (train_feats - mean) / std
            val_feats = (val_feats - mean) / std
            test_feats = (test_feats - mean) / std

        for pct in percents:
            n_samples = max(1, int(pct * len(train_feats)))
            indices = rng.choice(len(train_feats), size=n_samples, replace=False)
            k = min(n_neighbors, n_samples)

            clf = KNeighborsClassifier(n_neighbors=k)
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

    def _available_feature(name: str) -> bool:
        feature = getattr(embeddings["train"], name)
        return feature is not None and feature.size > 0

    probe_specs = []
    marker_map = {"backbone": "o", "projected": "s"}

    if feature_override is not None:
        if not _available_feature(feature_override):
            logging.warning(
                "Requested feature '%s' not available for linear probe; skipping.", feature_override
            )
            return
        probe_specs.extend(
            [
                (feature_override, f"{feature_override}_batchnorm", True),
            ]
        )
    elif _available_feature("backbone"):
        probe_specs.extend(
            [
                ("backbone", "backbone_batchnorm", True),
            ]
        )

    #     if _available_feature(name):
    #         probe_specs.append((name, name, False))
    #         probe_specs.append((name, f"{name}_batchnorm", True))

    if not probe_specs:
        logging.warning("No embeddings available for linear probe; skipping")
        return

    method_tag = norm_method.lower()
    plots_dir = output_dir / f"linear_probe_{method_tag}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for feature_key, suffix, use_batch_norm in probe_specs:
        lr_metrics = _evaluate_logreg(feature_key, batch_norm=use_batch_norm)
        knn_metrics = _evaluate_knn(feature_key, batch_norm=use_batch_norm)
        with (plots_dir / f"{label}_{suffix}_metrics.json").open("w") as handle:
            json.dump(lr_metrics, handle, indent=2)
        with (plots_dir / f"{label}_{suffix}_knn_metrics.json").open("w") as handle:
            json.dump(knn_metrics, handle, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        acc = [metrics[str(x)]["test_accuracy"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        ax1.plot(
            xs, acc, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})"
        )

    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        f1_vals = [metrics[str(x)]["test_f1"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        ax2.plot(
            xs, f1_vals, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})"
        )

    ax1.set_xlabel("Training fraction")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title(f"Test Accuracy ({label})")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Training fraction")
    ax2.set_ylabel("Test F1")
    ax2.set_title(f"Test F1 ({label})")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / f"{label}_combined_curves.png", dpi=200)
    plt.close(fig)

    # KNN plots
    fig_knn, (knn_ax1, knn_ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_knn_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        acc = [metrics[str(x)]["test_accuracy"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        knn_ax1.plot(
            xs, acc, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})"
        )

    for feature_key, suffix, use_batch_norm in probe_specs:
        metrics_file = plots_dir / f"{label}_{suffix}_knn_metrics.json"
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        if not metrics:
            continue

        xs = sorted(float(x) for x in metrics.keys())
        f1_vals = [metrics[str(x)]["test_f1"] for x in xs]
        linestyle = "--" if use_batch_norm else "-"
        marker = marker_map.get(feature_key, "o")
        label_suffix = "batch-norm" if use_batch_norm else "raw"
        knn_ax2.plot(
            xs, f1_vals, marker=marker, linestyle=linestyle, label=f"{feature_key} ({label_suffix})"
        )

    knn_ax1.set_xlabel("Training fraction")
    knn_ax1.set_ylabel("Test Accuracy")
    knn_ax1.set_title(f"KNN Test Accuracy ({label})")
    knn_ax1.grid(alpha=0.3)
    knn_ax1.legend()

    knn_ax2.set_xlabel("Training fraction")
    knn_ax2.set_ylabel("Test F1")
    knn_ax2.set_title(f"KNN Test F1 ({label})")
    knn_ax2.grid(alpha=0.3)
    knn_ax2.legend()

    fig_knn.tight_layout()
    fig_knn.savefig(plots_dir / f"{label}_knn_combined_curves.png", dpi=200)
    plt.close(fig_knn)

    logging.info("Linear probe results saved to %s", plots_dir)

    # logging.info(f"Saved t-SNE plot for {display_name} features: {output_dir / f'{label}_tsne_{feature_name}.png'}")

    # logging.info(
    #     "Saved PCA plot for %s features: %s",
    #     display_name,
    #     output_dir / f"{label}_pca_{feature_name}.png",
    # )

    # if bundle.projected is not None:
    #     _write(bundle.projected, "projected")


def run_full_evaluation(config: ModelEvalConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    if config.model_type == "croma":
        config.eurosat_image_size = config.croma_image_resolution

    adapter = build_evaluation_adapter(
        model_type=config.model_type,
        checkpoint=config.checkpoint,
        model_weights=config.model_weights,
        in_chans=config.model_in_channels,
        croma_weights=config.croma_weights,
        croma_image_resolution=config.croma_image_resolution,
        ciip_framework=config.ciip_framework,
        enable_s1=config.evaluation_modality.lower() == "s1",
    )
    print(f"Model loaded")
    # print(adapter)
    device = torch.device(torch.device("cuda") if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise ValueError
    print(device)
    adapter = adapter.to(device)
    adapter.eval()
    base_model = getattr(adapter, "base_model", adapter)
    is_lorentz = getattr(adapter, "is_lorentz", False)
    # save curvature to config if lorentz
    if is_lorentz and hasattr(base_model, "curvature"):
        config.curvature = float(base_model.curvature)
        print(f"Lorentz curvature: {config.curvature}")

    output_root = ensure_dir(Path(config.output_dir))

    target_modality = config.evaluation_modality.lower()
    if target_modality not in {"s1", "s2"}:
        raise ValueError("evaluation_modality must be either 's1' or 's2'")

    model_channels = _infer_model_in_channels(
        adapter, config.model_in_channels, modality=target_modality
    )
    thirteen_band_models = {
        "dofa_base_s2_13ch",
        "vitsmall16_s2_all_moco",
        "resnet18_s2_all_moco",
        "moco",
        "dino",
    }
    if (config.model_weights and config.model_weights in thirteen_band_models) or (
        config.model_type and config.model_type in thirteen_band_models
    ):
        model_channels = max(model_channels, 13)

    output_dir = ensure_dir(Path(config.output_dir))
    write_run_manifest(
        output_dir,
        task_name="unified_evaluation",
        config={
            "model_type": config.model_type,
            "model_weights": config.model_weights,
            "model_path": config.model_path,
            "ciip_framework": config.ciip_framework,
            "model_in_channels": config.model_in_channels,
            "normalization_method": config.normalization_method,
            "eurosat_root": config.eurosat_root,
            "neuco_root": config.neuco_root,
            "ssl4eo_root": config.ssl4eo_root,
            "enable_ssl4eo": config.enable_ssl4eo,
            "evaluation_modality": config.evaluation_modality,
        },
    )

    normalization_method = resolve_normalization_method_for_weights(
        config.model_in_channels, config.normalization_method, config.model_weights
    )
    print(f"Normalization method: {normalization_method}")
    if normalization_method not in NORMALIZATION_METHODS:
        raise ValueError(
            "normalization_method must be one of "
            f"{', '.join(NORMALIZATION_METHODS)}; got {config.normalization_method!r}"
        )

    if config.enable_eurosat:
        if target_modality == "s1":
            eurosat_bands = EUROSAT_S1_BANDS
        else:
            eurosat_bands = _resolve_eurosat_bands(model_channels)
        logging.info(
            "EuroSAT linear probe will use %d Sentinel-%s bands (%s)",
            len(eurosat_bands),
            "1" if target_modality == "s1" else "2",
            ", ".join(eurosat_bands),
        )

        # check if dir exists
        eurosat_output_dir = output_dir / f"linear_probe_{normalization_method}"
        if True:
            eurosat_loaders = _build_eurosat_loaders(
                config, bands=eurosat_bands, modality=target_modality
            )
            _print_band_stats(
                "EuroSAT eval",
                eurosat_loaders.get("val", next(iter(eurosat_loaders.values()))),
                bands=eurosat_bands,
                max_batches=config.stats_max_batches,
            )
            eurosat_embeddings: Dict[str, EmbeddingBundle] = {}
            with _use_adapter_modality(adapter, target_modality):
                for split, loader in eurosat_loaders.items():
                    eurosat_embeddings[split] = _extract_embeddings(
                        adapter,
                        loader,
                        device=device,
                        expected_in_channels=model_channels,
                        modality=target_modality,
                    )
            _run_linear_probe(
                config,
                eurosat_embeddings,
                output_dir=output_dir,
                label="eurosat",
                norm_method=normalization_method,
            )
            if config.matryoshka_dims:
                for dim in config.matryoshka_dims:
                    mat_output_dir = output_dir / f"matryoshka_dim_{dim}"
                    _run_linear_probe(
                        config,
                        eurosat_embeddings,
                        output_dir=mat_output_dir,
                        label="eurosat",
                        norm_method=normalization_method,
                        slice_dim=dim,
                        feature_override="backbone",
                    )
            _plot_eurosat_tsne(
                eurosat_embeddings["test"],
                output_dir=eurosat_output_dir,
                label="eurosat",
                max_samples=config.tsne_samples,
                seed=config.random_seed,
                model_title=config.model_path,
            )
            # _plot_eurosat_pca(
            #     eurosat_embeddings["test"],
            #     output_dir=eurosat_output_dir,
            #     label="eurosat",
            #     max_samples=config.pca_samples,
            #     seed=config.random_seed,
            #     model_title=config.model_path,
            #     modality_label=target_modality,
            # )
            # print("EuroSAT PCA plot saved.")

        # clean up dataset, not needed anymore
        del eurosat_loaders
        del eurosat_embeddings

    if config.enable_bigearthnet:
        print("Running BigEarthNet linear probe...")
        bigearthnet_loaders, bigearthnet_bands = _build_bigearthnet_loaders(
            config, expected_channels=model_channels
        )
        _print_band_stats(
            "BigEarthNet eval",
            bigearthnet_loaders.get("val", next(iter(bigearthnet_loaders.values()))),
            bands=bigearthnet_bands,
            max_batches=config.stats_max_batches,
        )
        print("BigEarthNet loaders built.")
        bigearthnet_embeddings: Dict[str, EmbeddingBundle] = {}
        with _use_adapter_modality(adapter, "s2"):
            for split, loader in bigearthnet_loaders.items():
                bigearthnet_embeddings[split] = _extract_embeddings(
                    adapter,
                    loader,
                    device=device,
                    expected_in_channels=model_channels,
                    modality="s2",
                )
        print("BigEarthNet embeddings extracted.")
        if config.matryoshka_dims:
            for dim in config.matryoshka_dims:
                mat_output_dir = output_dir / f"matryoshka_dim_{dim}"
                _run_linear_probe(
                    config,
                    bigearthnet_embeddings,
                    output_dir=mat_output_dir,
                    label="bigearthnet",
                    norm_method=normalization_method,
                    slice_dim=dim,
                    feature_override="backbone",
                )
        else:
            _run_linear_probe(
                config,
                bigearthnet_embeddings,
                output_dir=output_dir,
                label="bigearthnet",
                norm_method=normalization_method,
            )

    if config.enable_neuco:
        neuco_output_dir = output_dir / f"neuco_{normalization_method}"
        base_modalities: List[str] = list(config.neuco_modalities)
        s2_candidates = [m for m in base_modalities if m in ("s2l2a", "s2l1c")]
        has_dual_neuco = (
            len(base_modalities) == 2
            and "s1" in base_modalities
            and len(s2_candidates) > 0
            and config.model_type in {"torchgeo_resnet50", "ciip_checkpoint", "croma"}
        )

        def _run_single_neuco(modality: str, active_modality: str) -> None:
            suffix = "_s1" if active_modality.lower() == "s1" else ""
            csv_out_backbone = (
                neuco_output_dir / "neuco_export" / f"neuco_{modality}{suffix}_backbone.csv"
            )
            # csv_out_projected = neuco_output_dir / "neuco_export" / f"neuco_{modality}_projected.csv"

            if False:  # csv_out_backbone.exists():
                pass
                # print(f"NeuCo embeddings already exist for modality {modality}, skipping extraction.")
                # neuco_bundle = None
            else:
                print(
                    f"Extracting NeuCo embeddings for modality {modality} into {csv_out_backbone.parent}"
                )
                neuco_output_dir.mkdir(parents=True, exist_ok=True)
                neuco_loader = _build_neuco_loader(config, modalities=[modality])
                _print_band_stats(
                    f"NeuCo eval ({modality})",
                    neuco_loader,
                    max_batches=config.stats_max_batches,
                )

                expected_channels = _infer_model_in_channels(
                    adapter, config.model_in_channels, modality=active_modality
                )
                with _use_adapter_modality(adapter, active_modality):
                    neuco_bundle = _extract_embeddings(
                        adapter,
                        neuco_loader,
                        device=device,
                        require_ids=True,
                        expected_in_channels=expected_channels,
                        modality=active_modality,
                        pca_output_dir=neuco_output_dir,
                    )

                _export_neuco(neuco_bundle, neuco_output_dir / "neuco_export", label=modality)
                print("Saved NeuCo embeddings to ", neuco_output_dir / "neuco_export")

            print(f"model type: {config.model_type}")
            print(f"model weights: {config.model_weights}")
            if config.model_type == "croma" or (
                "dofa" in config.model_weights.lower() if config.model_weights else False
            ):
                embedding_dim_backbone = "768"
            elif config.model_type == "torchgeo_resnet50":
                embedding_dim_backbone = "2048"
            elif "resnet18" in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "512"
            elif "llama" in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "768"
            elif "remoteclip" in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "768"
            elif "resnet50" in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "2048"
            elif "vitsmall" in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "384"
            elif "scalemae" in config.model_weights.lower() if config.model_weights else False:
                embedding_dim_backbone = "1024"
            else:
                backbone_tensor = (
                    getattr(neuco_bundle, "backbone", None) if neuco_bundle is not None else None
                )

                embedding_dim_backbone = (
                    str(backbone_tensor.shape[1]) if backbone_tensor is not None else "2048"
                )

            print(f"NeuCo embedding dimension for {modality}: {embedding_dim_backbone}")
            backbone_tensor = (
                getattr(neuco_bundle, "backbone", None) if neuco_bundle is not None else None
            )
            if backbone_tensor is not None:
                print(
                    "NeuCo benchmark using backbone embeddings dim: "
                    f"{backbone_tensor.shape[1]} (embedding_dim arg: {embedding_dim_backbone})"
                )
            else:
                print(
                    "NeuCo benchmark embedding_dim arg: "
                    f"{embedding_dim_backbone} (backbone tensor unavailable)"
                )

            if config.matryoshka_dims:
                for dim in config.matryoshka_dims:
                    if neuco_bundle is None or neuco_bundle.backbone is None:
                        raise RuntimeError(
                            "NeuCo backbone embeddings are required for Matryoshka evaluation."
                        )
                    if dim > neuco_bundle.backbone.shape[1]:
                        raise ValueError(
                            f"Matryoshka dim {dim} exceeds NeuCo backbone dimension {neuco_bundle.backbone.shape[1]}."
                        )
                    mat_output_dir = neuco_output_dir / f"matryoshka_dim_{dim}"
                    mat_export_dir = mat_output_dir / "neuco_export"
                    mat_export_dir.mkdir(parents=True, exist_ok=True)
                    mat_bundle = EmbeddingBundle(
                        backbone=neuco_bundle.backbone[:, :dim],
                        projected=neuco_bundle.projected,
                        labels=neuco_bundle.labels,
                        multi_labels=neuco_bundle.multi_labels,
                        ids=neuco_bundle.ids,
                    )
                    mat_csv = mat_export_dir / f"neuco_{modality}{suffix}_backbone.csv"
                    _export_neuco(mat_bundle, mat_export_dir, label=modality)
                    cmd = [
                        "python",
                        "/local/ms-data/NeuCo-Bench/benchmark/main.py",
                        "--annotation_path",
                        "/local/ms-data/SSL4EO-S12-downstream/labels",
                        "--output_dir",
                        mat_output_dir,
                        "--config",
                        "/local/ms-data/NeuCo-Bench/benchmark/config.yaml",
                        "--method_name",
                        "backbone",
                        "--phase",
                        "testing",
                        "--submission_file",
                        mat_csv,
                        "--embedding_dim",
                        str(dim),
                    ]
                    env = os.environ.copy()
                    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                    subprocess.run(cmd, check=True, env=env)
            else:
                cmd = [
                    "python",
                    "/local/ms-data/NeuCo-Bench/benchmark/main.py",
                    "--annotation_path",
                    "/local/ms-data/SSL4EO-S12-downstream/labels",
                    "--output_dir",
                    neuco_output_dir,
                    "--config",
                    "/local/ms-data/NeuCo-Bench/benchmark/config.yaml",
                    "--method_name",
                    "backbone",
                    "--phase",
                    "testing",
                    "--submission_file",
                    csv_out_backbone,
                    "--embedding_dim",
                    embedding_dim_backbone,
                ]
                env = os.environ.copy()
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                subprocess.run(cmd, check=True, env=env)

        if has_dual_neuco:
            print("Dual NeuCo modalities detected; running S2 and S1 evaluations separately.")
            s2_modality = s2_candidates[0]
            _run_single_neuco(s2_modality, "s2")
            # _run_single_neuco("s1", "s1")
        else:
            neuco_modalities: List[str] = base_modalities
            # print('NeuCo modalities from config: ', neuco_modalities)
            # print('Target modality: ', target_modality)
            if target_modality == "s1":
                neuco_modalities = ["s1"]
            elif not neuco_modalities:
                neuco_modalities = ["s2l1c"]

            if len(neuco_modalities) > 1:
                print("Only useing 1st modality for neuco benchmark")
            modality = neuco_modalities[0]
            _run_single_neuco(modality, target_modality)

    ssl4eo_available = (
        config.enable_ssl4eo
        and config.ssl4eo_root is not None
        and getattr(adapter, "supports_ssl4eo", True)
    )

    assert ssl4eo_available or not config.enable_ssl4eo, (
        "SSL4EO diagnostics requested but not available"
    )

    s1_ssl4eo = s2_ssl4eo = None
    max_batches_cka = 20
    if ssl4eo_available:
        s1_ssl4eo, s2_ssl4eo, ssl4eo_ids = _extract_ssl4eo_embeddings(
            config, adapter, device=device, max_batches_cka=max_batches_cka
        )

        # create Path fo diagnostics
        diagnostics_dir = os.path.join(output_dir, "embedding_diagnostics")
        diagnostics_dir = Path(diagnostics_dir)

        _run_embedding_diagnostics(
            config,
            s1_ssl4eo,
            s2_ssl4eo,
            sample_ids=ssl4eo_ids,
            output_dir=diagnostics_dir,  # output_dir + "embedding_diagnostics",
        )
        logging.info("Completed SSL4EO diagnostics")
        # print output dir
        print("SSL4EO diagnostics saved to ", output_dir)
    else:
        logging.info(
            "Skipping SSL4EO diagnostics (enable_ssl4eo=%s, root=%s, supports_ssl4eo=%s)",
            config.enable_ssl4eo,
            config.ssl4eo_root,
            getattr(adapter, "supports_ssl4eo", True),
        )
    curvature = None
    if (
        is_lorentz
        and s1_ssl4eo is not None
        and s1_ssl4eo.projected is not None
        and s2_ssl4eo is not None
        and s2_ssl4eo.projected is not None
    ):
        if not isinstance(base_model, LorentzCIIP):
            raise TypeError("Expected LorentzCIIP model when is_lorentz is True")
        # use SSL4EO post-head projected features for hyperbolic visualisations
        s1_proj_feats = s1_ssl4eo.projected.to(device=device, dtype=torch.float32)
        s2_proj_feats = s2_ssl4eo.projected.to(device=device, dtype=torch.float32)
        _run_hyperbolic_visualisations(
            s1_proj_feats,
            s2_proj_feats,
            output_dir=output_dir / "hyperbolic",
            model=base_model,
            aperture_logk=None,
            seed=config.random_seed,
        )

        curvature = _extract_curvature_from_state(
            base_model, device=device, dtype=s1_proj_feats.dtype
        )

    if config.model_weights == "dino":
        print("Skipping SSL4EO cross-modal retrieval for TorchGeo dino ResNet50 model")
        return

    # use_backbone_retrieval = (
    #     config.model_type == "croma"
    #     or config.model_weights == 'moco'
    # )
    # use_projected_retrieval = (
    #     config.model_type == 'ciip_checkpoints'
    #     and not use_backbone_retrieval
    # )

    # if use_backbone_retrieval:
    #     logging.info(
    #         "Using backbone embeddings for cross-modal retrieval (model_type=%s, model_weights=%s)",
    #         config.model_type,
    #         config.model_weights,
    #     )
    #     retrieval_s1 = s1_ssl4eo.backbone
    #     retrieval_s2 = s2_ssl4eo.backbone

    #     retrieval_metrics = compute_cross_modal_retrieval(retrieval_s1, retrieval_s2, curvature=curvature)
    #     retrieval_metrics['num_samples'] = s1_ssl4eo.backbone.shape[0]
    #     #output_dir / "ssl4eo_retrieval.json"
    #     retrieval_path = Path(output_dir) / "ssl4eo_retrieval_backbone.json"
    #     retrieval_path.write_text(json.dumps(retrieval_metrics, indent=2, sort_keys=True))
    #     logging.info("SSL4EO cross-modal retrieval metrics: %s", retrieval_metrics)

    # elif use_projected_retrieval:
    #     # print the attributes of s1_ssl4eo
    #     # print("SSL4EO S1 EmbeddingBundle attributes:", dir(s1_ssl4eo))
    #     logging.info("Using projected embeddings for cross-modal retrieval")
    #     # print number of samples
    #     print(f'Num samples for cross-modal: {s1_ssl4eo.projected.shape[0]}')
    #     retrieval_s1 = s1_ssl4eo.projected
    #     retrieval_s2 = s2_ssl4eo.projected
    #     retrieval_metrics = compute_cross_modal_retrieval(retrieval_s1, retrieval_s2, curvature=curvature)
    #     retrieval_path = Path(output_dir) / "ssl4eo_retrieval_projected.json"
    #     retrieval_metrics['num_samples'] = s1_ssl4eo.projected.shape[0]
    #     retrieval_path.write_text(json.dumps(retrieval_metrics, indent=2, sort_keys=True))
    #     logging.info("SSL4EO projected cross-modal retrieval metrics: %s", retrieval_metrics)

    # else:
    #     print('Did not do cross-modal retrieval')


__all__ = ["ModelEvalConfig", "run_full_evaluation"]


def main() -> None:
    """Run the command-line interface without mixing argument policy into the pipeline."""
    from ciip.evaluation.unified_cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
