# CIIP

CIIP is a multimodal geospatial foundation-model research codebase. It learns a
shared embedding space from aligned Sentinel-1 SAR and Sentinel-2 optical
observations, with Euclidean and Lorentz/hyperbolic model variants. The
repository includes pretraining, checkpoint and embedding utilities, downstream
evaluation, and representation diagnostics.

## What is included

- **Pretraining:** ResNet and vision-transformer encoders, contrastive learning,
  optional patch masking/reconstruction, batch-uniformity and variance/covariance
  regularization, plus single-GPU, data-parallel, and distributed runners.
- **Data:** loaders for aligned SSL4EO-S12 Sentinel-1/Sentinel-2 Zarr data,
  including seasonal sampling, band selection, normalization, and train/validation
  splitting. An S2-text encoder pairing is also available in the data-parallel
  model builder.
- **Evaluation:** linear probes and few-shot/k-NN evaluation on EuroSAT,
  BigEarthNet, and NeuCo tasks; cross-modal retrieval; intrinsic-dimension
  metrics; and comparison with SSL4EO, CROMA, and other supported checkpoints.
- **Visualization:** downstream result summaries, embedding-collapse and SVD
  diagnostics, retrieval plots, and hyperbolic radius, cone, aperture, and
  angular projections.

## Setup

The supplied environment targets Python 3.12, PyTorch 2.4, and CUDA 12.1:

```bash
conda env create -f environment.yml
conda activate ciip
```

CPU-only or newer CUDA installations can use the same package list with a
matching PyTorch build. Run the commands below from the repository root so the
local `ciip` package is importable. Large datasets, model checkpoints, and
experiment outputs are intentionally not stored in Git.

## Pretraining data

Arrange SSL4EO-S12 modalities below one root. Current loaders support `S1GRD`
and `S2L1C`/`S2L2A` Zarr sources; the exact tier and bands are selected in the
Hydra config.

```text
/path/to/ssl4eo/
├── S1GRD/
├── S2L1C/
└── S2L2A/
```

Set `dataset.root`, `dataset.s2_tier`, model bands, batch size, and training
options in `ciip/open_clip_train/configs/local_default.yaml` for small runs.
Select that config explicitly when launching from the repository root:

```bash
python -m ciip.open_clip_train.run_train_val --config-name local_default \
  dataset.root=/path/to/ssl4eo \
  dataset.s2_tier=s2l2a
```

Hydra overrides can configure a run without editing YAML, for example
`train.epochs=100 datamodule.batch_size=128 loss.batch_uniformity_weight=0.05`.
Omit `--config-name local_default` to use the runner's `prod_default.yaml`.
Use `run_train_val_distributed` or the runner under
`ciip/open_clip_train/dataparallel/` for the corresponding multi-GPU workflow.

## Downstream evaluation

The evaluation dispatcher provides a common interface for EuroSAT few-shot,
NeuCo, and unified evaluations. Inspect task-specific options first, then run a
JSON-configured experiment:

```bash
python -m ciip.evaluation.run_downstream --help
python -m ciip.evaluation.run_downstream \
  --config /path/to/evaluation.json \
  --dry-run
python -m ciip.evaluation.run_downstream \
  --config /path/to/evaluation.json
```

`evaluation.json` contains a `task`, shared `defaults`/`script_args`, and
optionally a `models` list. Supported task names are shown by `--help`.
Task scripts can also be called directly for full CLI documentation. TorchGeo
evaluation datasets may download automatically, but SSL4EO and NeuCo paths must
point to local data.

## Visualizations

Aggregate completed downstream runs into publication-ready PNG and CSV files:

```bash
python -m ciip.evaluation.plot_downstream \
  --unified-eval-root diagnostics/unified_eval \
  --output-dir diagnostics/downstream_plots
```

For a trained Lorentz model, generate angle/aperture, radial, angular-PCA, and
cone plots with:

```bash
python -m visualizations.ssl4eo.hyperbolic_visualization \
  --config /path/to/training-config.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir diagnostics/hyperbolic
```

Additional scripts live in `visualizations/ssl4eo/`, `diagnostics/`, and
`intrinsic-dimension/`.

## Example results

Training runs write checkpoints and logs beneath the Hydra output directory.
Evaluation runs produce machine-readable metrics, while plotting scripts turn
those records into figures and summary tables. A typical experiment report
should include:

| Stage | Suggested result |
| --- | --- |
| Pretraining | train/validation contrastive loss and retrieval accuracy by epoch |
| Downstream | EuroSAT/BigEarthNet probe accuracy or F1, and NeuCo task metrics |
| Retrieval | Sentinel-1-to-Sentinel-2 and Sentinel-2-to-Sentinel-1 recall@K |
| Geometry | intrinsic dimension, singular-value spectra, radii, and cone/aperture plots |

Results depend on the checkpoint, split, bands, and random seed, so report the
resolved Hydra config and checkpoint alongside every table or figure. This
repository does not claim a canonical score in the absence of those artifacts.

## Repository map

```text
ciip/                    models, losses, data, training, and evaluation
visualizations/ssl4eo/   embedding and hyperbolic visualizations
diagnostics/             result aggregation and diagnostic scripts
intrinsic-dimension/     standalone intrinsic-dimension analyses
comparison/              supported external-model comparison code
docs/                    cleanup audits and repository-maintenance notes
```

See `ciip/open_clip_train/configs/` for the complete training configuration and
`docs/README.md` for repository audit documentation.
