# Visualizations and evaluations

This guide inventories the maintained evaluation and diagnostic surfaces, the
data they consume, and the artifacts they produce. Most commands require
external datasets and checkpoints that are intentionally absent from Git.
Run modules from the repository root and use `--help` for the complete,
current argument contract.

## Evaluation datasets and tasks

| Dataset / source | Evaluation | Typical output | Entry point |
| --- | --- | --- | --- |
| EuroSAT | k-NN/few-shot classification and trainable linear probes at multiple label fractions | accuracy/F1 JSON or CSV, per-epoch curves and comparison plots | `ciip.evaluation.eurosat_fewshot_1nn`, `ciip.evaluation.linearprobe_comparison`, or `ciip.evaluation.unified_evaluation` |
| NeuCo-Bench / SSL4EO-S12 downstream | Embedding export followed by benchmark tasks; fixed-fraction few-shot, episodic few-shot, and layerwise probes | NeuCo-format embedding CSV, per-task metrics/manifests, summaries | `export_neuco_embeddings`, `neuco_fewshot_benchmark`, `neuco_fewshot_episodic`, `neuco_layerwise_probes` |
| BigEarthNet | Linear-probe evaluation through the unified pipeline | classification metrics and run manifest | `ciip.evaluation.unified_evaluation` |
| PANGAEA | Batch downstream benchmark, including limited-label/few-shot runs | benchmark result index, mIoU tables/boxplots | `ciip.evaluation.pangaea-results.run_pangaea_batch_eval*` and `diagnostics.pangaea_limited_label_results` |
| SSL4EO validation | Paired S1/S2 representation diagnostics, retrieval, and intrinsic dimension | JSON/text metrics, caches, spectra and geometry plots | `compute_intrinsic_dimension_ssl4eo_val`, collapse and hyperbolic modules |
| SEN12MS | S1↔S2 cross-modal retrieval and hyperbolic radius analysis | recall/rank metrics, epoch curves, radius distributions and clustering plots | `plot_sen12ms_retrieval`, `sen12ms_hyperbolic_radii` |
| S2-100K | Global intrinsic-dimension analysis for optical embeddings | ID/effective-rank metrics and manifest | `intrinsic-dimension/compute_id.py` |

Evaluation adapters also support selected TorchGeo pretrained baselines, CROMA,
random convolutional features, and CIIP checkpoints. Comparisons are meaningful
only with the same split, modality/bands, preprocessing, feature space, and
label budget.

## Cross-modal retrieval

Retrieval tests whether the paired observation is recoverable across modalities
without fitting a classifier. `visualizations.ssl4eo.hyperbolic_retrieval`
computes chunked S1→S2 and S2→S1 metrics in projected or backbone space,
including recall@K and rank statistics. The SEN12MS plotter aggregates saved
metrics over epochs or Matryoshka dimensions. Report both directions, sample
count, K values, feature space, distance/geometry, and whether candidates are
unique locations; otherwise an apparently strong recall value is ambiguous.

## Intrinsic dimension and effective rank

The ID utilities remove duplicate/degenerate embeddings and compute FisherS,
MLE, method-of-moments (MoM), and tight-local-estimator (TLE) intrinsic
dimensions. The SSL4EO validation script can compute these globally, by
modality, for downstream/EuroSAT embeddings, and at Matryoshka prefixes. It also
derives SVD-based effective rank. `diagnostics/global_id_table1/` merges these
outputs with downstream performance and generates model/task scatterplots,
including Matryoshka-specific grids.

ID is sensitive to sample count, preprocessing, duplicate removal,
neighborhood settings, and feature normalization. Store the cache metadata and
do not compare estimates produced under different protocols as if they were
the same measurement.

## Embedding-collapse diagnostics

`visualizations.ssl4eo.embedding_collapse_diagnostics` extracts backbone and
projected features for both modalities and provides:

- singular-value spectra, cumulative explained variance, stable/effective rank,
  and numerical rank;
- per-dimension variance and near-zero-variance counts;
- feature norms, finite/duplicate checks, and modality summary statistics;
- paired cosine/L2 similarity and alignment versus mismatched pairs; and
- retrieval metrics via the retrieval helper.

These diagnostics distinguish total collapse, dimensional collapse, redundant
coordinates, norm pathologies, and poor cross-modal alignment. Run them before
interpreting a downstream score, and compare backbone with projected space to
locate where degeneration occurs. The notebooks under
`visualizations/ssl4eo/notebooks/` are exploratory front ends, not the canonical
reproducible entry points.

## Hyperbolic geometry diagnostics

For Lorentz checkpoints, `visualizations.ssl4eo.hyperbolic_visualization`
generates angle/aperture plots, radial distributions, angular PCA projections,
and cone visualizations. `sen12ms_hyperbolic_radii` adds paired modality radius
distributions and hierarchical clustering views. These reveal whether radius
encodes hierarchy, whether points crowd near a boundary, and whether paired
modalities occupy compatible regions. Always load or explicitly report the
checkpoint curvature; using an unrelated curvature invalidates distances and
radii.

## Downstream summary plots

`python -m ciip.evaluation.plot_downstream` discovers unified evaluation
outputs and writes publication-oriented PNG and CSV summaries. The specialized
scripts in `diagnostics/unified_eval/` provide:

- EuroSAT k-NN versus linear-probe comparisons across label fractions,
  preprocessing variants, epochs, and Matryoshka dimensions;
- NeuCo task tables, normalized/weighted aggregate scores, epoch selection,
  and Matryoshka dimension curves; and
- optional animations/contact sheets where the required image dependency is
  installed.

`diagnostics/pangaea_limited_label_results.py` produces raw and task-normalized
limited-label CSVs and a cross-model boxplot. The global-ID merger produces ID
versus performance, S2-ID versus downstream-ID, per-task, normalized, and
Matryoshka variants.

## Initialization and representation-space exploration

`visualizations.ssl4eo.initialization_evaluation` contains exploratory analyses
for pairwise and centroid L2/cosine distances, linear separability, relative
Mahalanobis-style geometry, PCA, t-SNE, and UMAP. It is useful for checkpoint
inspection but currently uses script-level output names and research
dependencies; record configuration manually and prefer maintained CLI tools for
benchmark claims.

## Unified and batch execution

The recommended dispatcher is:

```bash
python -m ciip.evaluation.run_downstream --help
python -m ciip.evaluation.run_downstream --config evaluation.json --dry-run
python -m ciip.evaluation.run_downstream --config evaluation.json
```

It dispatches single- or multi-model EuroSAT few-shot, NeuCo few-shot/episodic,
and unified tasks. Example JSON contracts live in
`ciip/evaluation/examples/`. Direct module invocation remains useful when a
specialized script exposes more controls. Every maintained evaluation writes a
run manifest or machine-readable result where supported; preserve it beside
the plot.

## Additional evaluations worth reporting

The repository already contains more than retrieval, ID, and collapse checks:
few-shot/k-NN, full and limited-label linear probes, layerwise probing,
multi-task NeuCo/PANGAEA transfer, effective rank/SVD, hyperbolic geometry, and
baseline comparisons. For a balanced model report, include at least one metric
from each of **alignment** (bidirectional retrieval), **geometry** (ID/rank and
collapse), **sample-efficient transfer** (few-shot/k-NN), and **trained
transfer** (linear probe or benchmark).

Useful future additions not currently exposed as maintained end-to-end tools
would be zero-shot text classification for the text variants, geographic
holdout/region-shift and temporal/season-shift evaluation, robustness to
missing bands or corruptions, calibrated retrieval uncertainty, and compute /
latency / memory measurements across Matryoshka prefixes. Label these as future
work rather than implying that the present repository generates them.

## Reporting checklist

For every generated table or figure retain: dataset version and split; label
fraction/episode construction; modality and bands; spatial/temporal sampling;
normalization and resize; checkpoint and epoch; backbone versus projection;
Matryoshka dimension; Euclidean versus Lorentz geometry and curvature; seed and
sample count; evaluator arguments; raw machine-readable output; and software
revision. Plots without this context should be treated as exploratory only.
