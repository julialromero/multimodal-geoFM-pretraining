"""Presentation and diagnostic outputs for unified evaluation."""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ciip.model_ciip import LorentzCIIP
from ciip.evaluation.export_neuco_embeddings import create_submission_from_dict
from ciip.evaluation.unified_types import EmbeddingBundle, ModelEvalConfig
from ciip.visualization.ssl4eo.embedding_collapse_diagnostics import (
    EpochDiagnostics,
    ModalityEmbeddings,
    compute_projection,
    compute_singular_values,
    compute_within_encoder_cka,
    compute_cross_encoder_cka,
    plot_epoch_diagnostics,
    plot_epoch_diagnostics_s2only,
    plot_epoch_diagnostics_scalemae,
    plot_epoch_diagnostics_croma,
    plot_projection,
    preprocess_projection_data,
)
from ciip.visualization.ssl4eo.hyperbolic_visualization import (
    compute_hyperbolic_context,
    plot_angle_aperture,
    plot_angular_pca,
    plot_cone_polar,
    plot_radial_histogram,
    plot_pos_neg_hist,
)

EUROSAT_CLASS_NAMES = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake",
}


def _plot_eurosat_tsne(
    bundle: EmbeddingBundle,
    *,
    output_dir: Path,
    label: str,
    max_samples: int,
    seed: int,
    model_title: Optional[str],
) -> None:
    if bundle.labels is None:
        logging.warning("No labels provided for EuroSAT embeddings; skipping t-SNE plot")
        return

    feature_types = [
        ("backbone", bundle.backbone, "Backbone"),
        ("projected", bundle.projected, "Projected"),
    ]

    # Filter to only non-None features with data
    available_features = [
        (name, features, display_name)
        for name, features, display_name in feature_types
        if features is not None and features.size > 0
    ]

    if not available_features:
        logging.warning("No embeddings available for EuroSAT t-SNE plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for feature_name, features, display_name in available_features:
        labels = bundle.labels
        total_samples = len(features)
        rng = np.random.default_rng(seed)

        # Sample if needed
        if total_samples > max_samples:
            indices = rng.choice(total_samples, size=max_samples, replace=False)
            features_sampled = features[indices]
            labels_sampled = labels[indices]
        else:
            features_sampled = features
            labels_sampled = labels

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
        embeddings_2d = tsne.fit_transform(features_sampled)

        # Plot
        unique_labels = sorted(np.unique(labels_sampled))
        cmap = plt.get_cmap("tab20", len(unique_labels))

        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, class_label in enumerate(unique_labels):
            mask = labels_sampled == class_label
            label_text = EUROSAT_CLASS_NAMES.get(int(class_label), str(class_label))
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                s=8,
                color=cmap(idx),
                label=label_text,
                alpha=0.7,
            )

        title = model_title if model_title is not None else "EuroSAT t-SNE"
        ax.set_title(f"{title} - {display_name}")
        ax.set_xlabel("t-SNE component 1")
        ax.set_ylabel("t-SNE component 2")
        ax.legend(title="Class", fontsize="small", markerscale=2)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_tsne_{feature_name}.png", dpi=200)
        plt.close(fig)


def _plot_eurosat_pca(
    bundle: EmbeddingBundle,
    *,
    output_dir: Path,
    label: str,
    max_samples: int,
    seed: int,
    model_title: Optional[str],
    modality_label: str,
) -> None:
    feature_types = [
        ("backbone", bundle.backbone, "Backbone"),
        ("projected", bundle.projected, "Projected"),
    ]
    available_features = [
        (name, features, display_name)
        for name, features, display_name in feature_types
        if features is not None and features.size > 0
    ]
    if not available_features:
        logging.warning("No embeddings available for EuroSAT PCA plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    modality_label = modality_label.upper()
    palette = {"S2": "#1f77b4", "S1": "#ff7f0e"}

    for feature_name, features, display_name in available_features:
        total_samples = len(features)
        if total_samples > max_samples:
            indices = rng.choice(total_samples, size=max_samples, replace=False)
            features_sampled = features[indices]
        else:
            features_sampled = features

        pca = PCA(n_components=2, random_state=seed)
        embeddings_2d = pca.fit_transform(features_sampled)

        fig, ax = plt.subplots(figsize=(8, 6))
        label_array = np.repeat(modality_label, len(embeddings_2d))
        unique_labels = sorted(np.unique(label_array))

        for idx, modality in enumerate(unique_labels):
            mask = label_array == modality
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                s=12,
                color=palette.get(modality, plt.get_cmap("tab10")(idx)),
                label=modality,
                alpha=0.8,
            )

        ax.set_title(f"PCA ({feature_name}, raw)")
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        if len(unique_labels) > 1:
            ax.legend(title="Modality", fontsize="small", markerscale=2)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_pca_{feature_name}.png", dpi=200)
        plt.close(fig)


def _export_neuco(
    bundle: EmbeddingBundle,
    output_dir: Path,
    *,
    label: str,
) -> None:
    if bundle.ids is None:
        raise RuntimeError("NeuCo export requires file identifiers")

    def _write(features: np.ndarray, suffix: str) -> None:
        rows = {idx: vec for idx, vec in zip(bundle.ids or [], features)}
        df = create_submission_from_dict(rows)
        df.to_csv(output_dir / f"neuco_{label}_{suffix}.csv", index=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle.backbone is not None:
        _write(bundle.backbone, "backbone")


def _run_embedding_diagnostics(
    config: ModelEvalConfig,
    s1_bundle: Optional[ModalityEmbeddings],
    s2_bundle: ModalityEmbeddings,
    *,
    sample_ids: Optional[Sequence[str]] = None,
    output_dir: Path,
) -> None:
    print("Running embedding diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    import CKA

    # print(device)
    cuda_cka = CKA.CudaCKA(device="cuda")

    def _detach_or_none(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return tensor.detach() if tensor is not None else None

    def _prepare_modality(bundle: Optional[ModalityEmbeddings]) -> Optional[Dict[str, object]]:
        if bundle is None:
            return None
        return {
            "backbone": _detach_or_none(getattr(bundle, "backbone", None)),
            "projected": _detach_or_none(getattr(bundle, "projected", None)),
            "layers": bundle.layer_activations or {},
        }

    modality_tensors: Dict[str, Optional[Dict[str, object]]] = {
        "s1": _prepare_modality(s1_bundle),
        "s2": _prepare_modality(s2_bundle),
    }
    if modality_tensors["s2"] is None:
        raise ValueError("S2 embeddings are required for diagnostics")

    def _maybe_singular(modality: str, feature: str) -> Optional[np.ndarray]:
        tensors = modality_tensors.get(modality)
        if tensors is None:
            return None
        tensor = tensors.get(feature)
        if tensor is None:
            return None
        return compute_singular_values(tensor)  # type: ignore[arg-type]

    # Within-encoder CKA containers
    s1_layers: List[str] = []
    s1_within: Optional[np.ndarray] = None
    s1_odd_layers: List[str] = []
    s1_odd_within: Optional[np.ndarray] = None
    s1_even_layers: List[str] = []
    s1_even_within: Optional[np.ndarray] = None
    s1_residual_layers: List[str] = []
    s1_residual_within: Optional[np.ndarray] = None

    s2_layers: List[str] = []
    s2_within: Optional[np.ndarray] = None
    s2_odd_layers: List[str] = []
    s2_odd_within: Optional[np.ndarray] = None
    s2_even_layers: List[str] = []
    s2_even_within: Optional[np.ndarray] = None
    s2_residual_layers: List[str] = []
    s2_residual_within: Optional[np.ndarray] = None
    s1_group_layers: Dict[str, List[str]] = {}
    s1_group_within: Dict[str, Optional[np.ndarray]] = {}
    s2_group_layers: Dict[str, List[str]] = {}
    s2_group_within: Dict[str, Optional[np.ndarray]] = {}

    # Cross-encoder CKA containers
    cross_s1_layers: List[str] = []
    cross_s2_layers: List[str] = []
    cross_matrix: Optional[np.ndarray] = None

    projected_cka: Optional[float] = None

    def _collect_transformer_groups(layer_dict: Dict[str, torch.Tensor]):
        group_layers: Dict[str, List[str]] = {}
        group_within: Dict[str, Optional[np.ndarray]] = {}

        names = list(layer_dict.keys())
        is_scalemae_style = any(
            pattern in name for name in names for pattern in (".layernorm", ".scale", ".op")
        )

        if is_scalemae_style:
            specs = {
                "layernorm": [k for k in names if ".layernorm" in k],
                "scale": [k for k in names if ".scale" in k],
                "op": [k for k in names if ".op" in k],
                "residual": [k for k in names if ".residual" in k],
            }
        else:
            specs = {
                "ln": [k for k in names if (k.endswith(".ln") or ".ln." in k)],
                "core": [k for k in names if (k.endswith(".core") or ".core." in k)],
                "residual": [k for k in names if ".residual" in k],
            }

        for group, keys in specs.items():
            subset = {k: layer_dict[k] for k in keys}
            if subset:
                layers, cka_matrix = compute_within_encoder_cka(subset, cuda_cka)
            else:
                layers, cka_matrix = [], None
            group_layers[group] = layers
            group_within[group] = cka_matrix

        return group_layers, group_within

    # --------------------
    # S1 within-encoder CKA
    # --------------------
    if modality_tensors["s1"] and modality_tensors["s1"]["layers"]:
        s1_all_dict = modality_tensors["s1"]["layers"]  # type: ignore[arg-type]
        # print
        s1_layers, s1_within = compute_within_encoder_cka(s1_all_dict, cuda_cka)  # 32x32

        layer_names = list(s1_all_dict.keys())
        has_resnet_hooks = any(name.endswith(".bn2") for name in layer_names)
        has_transformer_hooks = any(
            pattern in name
            for name in layer_names
            for pattern in (".layernorm", ".ln", ".op", ".core", ".scale", ".residual")
        )

        if has_resnet_hooks:
            s1_odd_dict = {k: v for k, v in s1_all_dict.items() if k.endswith(".bn2")}
            if s1_odd_dict:
                s1_odd_layers, s1_odd_within = compute_within_encoder_cka(s1_odd_dict, cuda_cka)

            s1_even_dict = {k: v for k, v in s1_all_dict.items() if k.endswith(".block_out")}
            if s1_even_dict:
                s1_even_layers, s1_even_within = compute_within_encoder_cka(s1_even_dict, cuda_cka)

            s1_group_layers = {
                "odd": s1_odd_layers,
                "even": s1_even_layers,
            }
            s1_group_within = {
                "odd": s1_odd_within,
                "even": s1_even_within,
            }

        if has_transformer_hooks:
            s1_group_layers, s1_group_within = _collect_transformer_groups(s1_all_dict)

            # Do not reuse odd/even slots for transformer groups
            s1_odd_layers, s1_odd_within = [], None
            s1_even_layers, s1_even_within = [], None
            s1_residual_layers, s1_residual_within = [], None

    elif s1_bundle is not None:
        logging.warning("S1 layer activations unavailable; skipping S1 CKA diagnostics")

    # --------------------
    # S2 within-encoder CKA
    # --------------------
    if modality_tensors["s2"] and modality_tensors["s2"]["layers"]:
        s2_all_dict = modality_tensors["s2"]["layers"]  # type: ignore[arg-type]

        # print('')
        s2_layers, s2_within = compute_within_encoder_cka(s2_all_dict, cuda_cka)

        layer_names = list(s2_all_dict.keys())
        has_resnet_hooks = any(name.endswith(".bn2") for name in layer_names)
        has_transformer_hooks = any(
            pattern in name
            for name in layer_names
            for pattern in (".layernorm", ".ln", ".op", ".core", ".scale", ".residual")
        )

        if has_resnet_hooks:
            s2_odd_dict = {k: v for k, v in s2_all_dict.items() if k.endswith(".bn2")}
            if s2_odd_dict:
                s2_odd_layers, s2_odd_within = compute_within_encoder_cka(s2_odd_dict, cuda_cka)

            s2_even_dict = {k: v for k, v in s2_all_dict.items() if k.endswith(".block_out")}
            if s2_even_dict:
                s2_even_layers, s2_even_within = compute_within_encoder_cka(s2_even_dict, cuda_cka)

        if has_transformer_hooks:
            s2_group_layers, s2_group_within = _collect_transformer_groups(s2_all_dict)

            s2_odd_layers, s2_odd_within = [], None
            s2_even_layers, s2_even_within = [], None
            s2_residual_layers, s2_residual_within = [], None

    else:
        logging.warning("S2 layer activations unavailable; skipping S2 CKA diagnostics")

    # --------------------
    # Cross-encoder CKA (unchanged)
    # --------------------
    if (
        modality_tensors["s1"]
        and modality_tensors["s1"]["layers"]
        and modality_tensors["s2"]
        and modality_tensors["s2"]["layers"]
    ):
        cross_s1_layers, cross_s2_layers, cross_matrix = compute_cross_encoder_cka(
            modality_tensors["s1"]["layers"],  # type: ignore[arg-type]
            modality_tensors["s2"]["layers"],  # type: ignore[arg-type]
            cuda_cka,
        )

    s1_projected_singular = _maybe_singular("s1", "projected")
    s1_backbone_singular = _maybe_singular("s1", "backbone")
    s2_projected_singular = _maybe_singular("s2", "projected")
    s2_backbone_singular = _maybe_singular("s2", "backbone")

    # ----- Package CKA results (full, odd, even, cross, projected) -----
    cka_payload: Dict[str, object] = {
        "s1_full": None,
        "s1_odd": None,
        "s1_even": None,
        "s2_full": None,
        "s2_odd": None,
        "s2_even": None,
        "cross": None,
        "projected_similarity": projected_cka,
    }

    # S1 full
    if s1_layers and s1_within is not None:
        cka_payload["s1_full"] = {
            "layers": s1_layers,
            "matrix": s1_within.tolist(),
        }

    # S1 odd (bn3)
    if s1_odd_layers and s1_odd_within is not None:
        cka_payload["s1_odd"] = {
            "layers": s1_odd_layers,
            "matrix": s1_odd_within.tolist(),
        }

    # S1 even (block_out)
    if s1_even_layers and s1_even_within is not None:
        cka_payload["s1_even"] = {
            "layers": s1_even_layers,
            "matrix": s1_even_within.tolist(),
        }

    if s1_group_layers:
        cka_payload["s1_groups"] = {
            group: {
                "layers": layers,
                "matrix": (
                    s1_group_within[group].tolist()
                    if s1_group_within.get(group) is not None
                    else None
                ),
            }
            for group, layers in s1_group_layers.items()
        }

    # S2 full
    if s2_layers and s2_within is not None:
        cka_payload["s2_full"] = {
            "layers": s2_layers,
            "matrix": s2_within.tolist(),
        }

    # S2 odd (bn3)
    if s2_odd_layers and s2_odd_within is not None:
        cka_payload["s2_odd"] = {
            "layers": s2_odd_layers,
            "matrix": s2_odd_within.tolist(),
        }

    # S2 even (block_out)
    if s2_even_layers and s2_even_within is not None:
        cka_payload["s2_even"] = {
            "layers": s2_even_layers,
            "matrix": s2_even_within.tolist(),
        }

    if s2_group_layers:
        cka_payload["s2_groups"] = {
            group: {
                "layers": layers,
                "matrix": (
                    s2_group_within[group].tolist()
                    if s2_group_within.get(group) is not None
                    else None
                ),
            }
            for group, layers in s2_group_layers.items()
        }

    # Cross-encoder CKA
    if cross_matrix is not None:
        cka_payload["cross"] = {
            "s1_layers": cross_s1_layers,
            "s2_layers": cross_s2_layers,
            "matrix": cross_matrix.tolist(),
        }

    # (optional) write to JSON, if you want it on disk:
    # with (output_dir / "cka.json").open("w", encoding="utf-8") as handle:
    #     json.dump(cka_payload, handle, indent=2)

    spectra: List[Tuple[str, str, np.ndarray]] = []
    if s1_projected_singular is not None:
        spectra.append(("s1", "projected", s1_projected_singular))
    if s1_backbone_singular is not None:
        spectra.append(("s1", "backbone", s1_backbone_singular))
    if s2_projected_singular is not None:
        spectra.append(("s2", "projected", s2_projected_singular))
    if s2_backbone_singular is not None:
        spectra.append(("s2", "backbone", s2_backbone_singular))

    # # check type of payload
    # for entry in cka_payload.items():
    #     key, value = entry
    #     if value is not None and not isinstance(value, dict):
    #         print(value)
    #         # raise ValueError(f"CKA payload '{key}' is not a dict as expected")

    #     #     logging.info(f"CKA payload '{key}' has keys: {list(value.keys())}")
    #     # if type is tensor, conver to np
    #     if isinstance(value, torch.Tensor):
    #         cka_payload[key] = value.cpu().numpy()
    #         print(f"Converted CKA payload '{key}' from tensor to numpy array")
    # with (output_dir / "cka.json").open("w", encoding="utf-8") as handle:
    #     json.dump(cka_payload, handle, indent=2)

    label_source = (
        (config.checkpoint.stem if config.checkpoint is not None else None)
        or config.model_weights
        or config.model_type
    )

    epoch_source = (config.checkpoint.stem if config.checkpoint is not None else "") or (
        config.model_weights or ""
    )
    epoch_match = re.search(r"epoch[_=-]?(\d+)", epoch_source)
    epoch_value = int(epoch_match.group(1)) if epoch_match else 0

    if config.model_type == "torchgeo_resnet50":
        epoch_value = None
        label_source = config.model_weights

    s2_raw = getattr(s2_bundle, "raw", None)
    sample_count = int(s2_raw.shape[0]) if s2_raw is not None and s2_raw.ndim > 0 else 0
    diagnostic_ids = (
        [str(item) for item in sample_ids]
        if sample_ids is not None and len(sample_ids) > 0
        else [str(index) for index in range(sample_count)]
    )

    def _as_array(values: Optional[np.ndarray]) -> np.ndarray:
        return values if values is not None else np.empty(0, dtype=np.float32)

    epoch_diagnostics = EpochDiagnostics(
        label=label_source,
        epoch=epoch_value,
        ids=diagnostic_ids,
        s1=s1_bundle,
        s2=s2_bundle,
        s1_singular_values=_as_array(s1_projected_singular),
        s2_singular_values=_as_array(s2_projected_singular),
        s1_layers=s1_layers,
        s2_layers=s2_layers,
        s1_within_cka=s1_within,
        s2_within_cka=s2_within,
        cross_cka=cross_matrix,
        cross_s1_layers=cross_s1_layers,
        cross_s2_layers=cross_s2_layers,
        ## add
        s1_odd_layers=s1_odd_layers,
        s1_odd_within_cka=s1_odd_within,
        s1_even_layers=s1_even_layers,
        s1_even_within_cka=s1_even_within,
        s1_residual_layers=s1_residual_layers,
        s1_residual_within_cka=s1_residual_within,
        s2_odd_layers=s2_odd_layers,
        s2_odd_within_cka=s2_odd_within,
        s2_even_layers=s2_even_layers,
        s2_even_within_cka=s2_even_within,
        s2_residual_layers=s2_residual_layers,
        s2_residual_within_cka=s2_residual_within,
        s1_group_layers=s1_group_layers,
        s1_group_within_cka=s1_group_within,
        s2_group_layers=s2_group_layers,
        s2_group_within_cka=s2_group_within,
    )

    can_plot_full = s1_bundle is not None and modality_tensors["s1"] is not None
    can_plot_s2_only = s1_bundle is None

    normalized_weights = (config.model_weights or "").lower()
    is_transformer = config.model_type in ("croma", "croma_vit", "croma_s1s2") or any(
        token in normalized_weights
        for token in ("dofa", "scalemae", "vitsmall", "vit", "visiontransformer")
    )
    is_scalemae = any(
        token in normalized_weights for token in ("scalemae", "dofa", "visiontransformer", "vit")
    )
    print(
        f"Model type: {config.model_type}, Weights: {normalized_weights}, is_transformer: {is_transformer}, is_scalemae: {is_scalemae}"
    )
    is_resnet_based = (
        config.model_type == "torchgeo_resnet50"
        or normalized_weights.startswith("resnet")
        or normalized_weights == "rcf_13ch"
    )
    is_ciip_or_moco = config.model_type == "ciip_checkpoint" or (
        config.model_type == "torchgeo_resnet50" and normalized_weights == "moco"
    )

    # if resnet architecture, plot even-layer cka and odd-layer cka
    plot_even_odd = is_resnet_based or config.model_type == "ciip_checkpoint"

    if is_transformer:
        if is_scalemae:
            plot_epoch_diagnostics_scalemae(
                epoch_diagnostics, output_dir, label="transformer_hooks"
            )
        else:
            plot_epoch_diagnostics_croma(epoch_diagnostics, output_dir, label="transformer_hooks")
    elif is_ciip_or_moco:
        if can_plot_full:
            plot_epoch_diagnostics(
                epoch_diagnostics,
                output_dir,
                label="embeddings_raw",
                plot_even_odd=True,
            )
        else:
            logging.info(
                "Skipping full epoch diagnostics plot; required modalities missing for %s",
                config.model_type,
            )
    elif is_resnet_based and (can_plot_s2_only or can_plot_full):
        plot_epoch_diagnostics_s2only(
            epoch_diagnostics,
            output_dir,
            label="s2_backbone_raw",
            plot_even_odd=plot_even_odd,
        )
    else:
        logging.info("Skipping epoch diagnostics plots due to missing embedding features")

    if not spectra:
        logging.warning("No features available for singular value diagnostics")
    else:
        n = len(spectra)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols)) if ncols > 0 else 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax, (modality, feature_name, spectrum) in zip(axes, spectra):
            ax.plot(np.arange(len(spectrum)), spectrum, marker=".")
            ax.set_xlabel("Component")
            ax.set_ylabel("Singular value")
            ax.set_title(f"{modality.upper()} {feature_name}")
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 100)

        for ax in axes[len(spectra) :]:
            ax.axis("off")

        fig.suptitle("Singular Values per Modality / Feature", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(output_dir / "singular_values_all.png", dpi=200)
        plt.close(fig)

    rng = np.random.default_rng(config.random_seed)

    def _stack_features(feature_name: str) -> Tuple[np.ndarray, np.ndarray]:
        tensors: List[np.ndarray] = []
        label_arrays: List[np.ndarray] = []
        if modality_tensors["s1"] and modality_tensors["s1"].get(feature_name) is not None:
            s1_np = modality_tensors["s1"][feature_name].cpu().numpy()  # type: ignore[index]
            tensors.append(s1_np)
            label_arrays.append(np.repeat("S1", len(s1_np)))
        if modality_tensors["s2"] and modality_tensors["s2"].get(feature_name) is not None:
            s2_np = modality_tensors["s2"][feature_name].cpu().numpy()  # type: ignore[index]
            tensors.append(s2_np)
            label_arrays.append(np.repeat("S2", len(s2_np)))
        if not tensors:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype="<U2")
        features = np.concatenate(tensors, axis=0)
        labels = np.concatenate(label_arrays, axis=0)
        return features, labels

    def _sample(
        features: np.ndarray, labels: np.ndarray, maximum: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if maximum <= 0 or maximum >= len(features):
            return features, labels
        indices = rng.choice(len(features), size=maximum, replace=False)
        return features[indices], labels[indices]

    for feature_name in ("projected", "backbone"):
        # ciip is lorentz
        curv = None
        if "lorentz" in config.model_type and feature_name == "projected":
            use = "poincare"
            curv = config.curvature

        else:
            use = "zscore"

        combined, modality_labels = _stack_features(feature_name)
        if combined.size == 0:
            continue
        for mode in ("raw", use):
            subset, subset_labels = _sample(combined, modality_labels, config.tsne_samples)
            processed = preprocess_projection_data(
                subset, mode=mode, random_state=config.random_seed, curvature=curv
            )
            coords = compute_projection(processed, method="tsne", random_state=config.random_seed)
            if coords is not None:
                suffix = mode
                plot_projection(
                    coords,
                    subset_labels,
                    output_dir / f"tsne_{feature_name}_{suffix}.png",
                    title=f"t-SNE ({feature_name}, {suffix})",
                )

            subset, subset_labels = _sample(combined, modality_labels, config.pca_samples)
            processed = preprocess_projection_data(
                subset, mode=mode, random_state=config.random_seed
            )
            if processed.shape[0] >= 2:
                suffix = "zscore" if mode == "zscore" else "raw"
                pca_coords = PCA(n_components=2, random_state=config.random_seed).fit_transform(
                    processed
                )
                plot_projection(
                    pca_coords,
                    subset_labels,
                    output_dir / f"pca_{feature_name}_{suffix}.png",
                    title=f"PCA ({feature_name}, {suffix})",
                )
    print(f"Embedding diagnostics saved to {output_dir}")


def _run_hyperbolic_visualisations(
    s1_proj_features: torch.Tensor,
    s2_proj_features: torch.Tensor,
    *,
    output_dir: Path,
    model: LorentzCIIP,
    aperture_logk: Optional[float],
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = compute_hyperbolic_context(
        model,
        s1_proj_features,
        s2_proj_features,
        aperture_logk=aperture_logk,
    )
    positive_angles = context["positive_angles"].cpu().numpy()
    aperture_s1 = context["aperture_s1"].cpu().numpy()
    aperture_s2 = context["aperture_s2"].cpu().numpy()
    s1_dirs = context["s1_dirs"].cpu().numpy()
    s2_dirs = context["s2_dirs"].cpu().numpy()
    s1_distances = context["s1_distances"].cpu().numpy()
    s2_distances = context["s2_distances"].cpu().numpy()

    angles_mat = context["angles"]  # (N, N), on device
    N = angles_mat.size(0)
    # positives: diagonal α(x_i, y_i)
    pos_alpha = context["positive_angles"].cpu().numpy()
    # negatives: all off-diagonals α(x_i, y_j), i != j
    neg_alpha = angles_mat[~torch.eye(N, dtype=torch.bool, device=angles_mat.device)].cpu().numpy()

    # Convert to "similarity" the way the loss does: κ = -α
    pos_angle_sim = -pos_alpha
    neg_angle_sim = -neg_alpha

    plot_pos_neg_hist(
        pos_angle_sim,
        neg_angle_sim,
        use_distance=False,
        out_path=output_dir / "angle_sim_hist.png",
    )

    plot_angle_aperture(
        positive_angles, aperture_s1, aperture_s2, output_dir / "angle_aperture.png"
    )
    plot_radial_histogram(s1_distances, s2_distances, output_dir / "radial_histogram.png")
    plot_angular_pca(s1_dirs, s2_dirs, s1_distances, s2_distances, output_dir / "angular_pca.png")
    plot_cone_polar(
        positive_angles,
        aperture_s1,
        aperture_s2,
        output_dir / "cone_polar.png",
        sample_size=256,
        seed=seed,
    )
