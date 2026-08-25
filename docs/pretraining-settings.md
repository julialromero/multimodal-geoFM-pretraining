# Pretraining settings

This guide inventories the pretraining variants that are implemented in this
repository. It distinguishes the representation being learned, the objective,
and optional regularizers so experiment names do not conflate independent
choices. The authoritative defaults remain the Hydra YAML files; always retain
the resolved config with a checkpoint.

## Baseline: SAR–optical CIIP

The baseline is a CLIP-style dual encoder over co-located Sentinel-1 SAR (S1)
and Sentinel-2 optical (S2) observations. Each encoder produces an embedding,
and a symmetric contrastive objective makes the matching S1/S2 pair more
similar than other pairs in the batch. Select it with
`model.encoder_pair: default` or `s1s2` (the data-parallel builder accepts both
forms used by the supplied configs) and leave the optional loss flags disabled.
`loss.contrastive_weight` scales the baseline objective. Large effective
batches, produced directly or through `train.accum_freq`, provide more
in-batch negatives.

The model architecture is an independent axis: `model.framework` supports the
repository's ResNet, modified-ResNet, and vision-transformer paths, with band
counts, patch sizes, resolutions, widths, layers, and embedding dimension set
under `model`. `model.pretrain.load` and its S1/S2 weight names optionally
initialize supported ResNet-50 encoders from MoCo/DINO weights; this is
initialization, not a different CIIP loss.

## Modality settings

| Setting | `model.encoder_pair` | What is aligned | Important notes |
| --- | --- | --- | --- |
| SAR–optical | `default` or `s1s2` | S1 and S2 | The standard two-sensor CIIP experiment. |
| Optical–text | `s2_text` | S2 and tokenized text | Implemented by the data-parallel builder. The S2 encoder can be initialized from `model.s2_ciip_checkpoint`; text can use MS-CLIP weights through `model.text.load_ms_clip`. Internally the generic two-tower loss keys still call the towers S1/S2. |
| SAR–optical–text | `s1s2_text` | S1–S2 plus S1–text and S2–text | The data-parallel trainer computes the three pairwise alignments. Optional `model.s1s2_ciip_checkpoint` and `model.text_checkpoint` initialize the vision and text sides. |

Text variants currently belong to the data-parallel path, not every training
runner. The dataset must emit the expected text field/tokens. Treat an
`encoder_pair` spelling, runner, and dataset as one tested contract rather than
assuming all combinations are interchangeable.

## Matryoshka representation learning

Matryoshka training makes prefixes of a single embedding useful at several
capacities. With `loss.matryoshka_enabled: true`, the Euclidean contrastive
loss is evaluated on every prefix in `loss.matryoshka_dims`; per-prefix weights
come from `loss.matryoshka_weights`, the combined objective is scaled by
`loss.matryoshka_weight`, and `loss.matryoshka_normalize` controls L2
normalization after slicing. Dimensions must not exceed `model.embed_dim`.

This is a replacement form of the contrastive term, not an extra encoder.
Downstream evaluation should therefore report the selected prefix dimension.
The current loss explicitly rejects Matryoshka together with the hyperbolic
mode, so these are separate experiment families.

## Lorentz / hyperbolic CIIP

Set `loss.hyperbolic: true` to build `LorentzCIIP` and use Lorentzian geometry
for the shared representation and similarity calculation. Relevant controls
are:

- `loss.curvature_init`: initial positive curvature magnitude.
- `loss.learn_curv`: whether curvature is learned; `train.curvature_lr` gives
  it a dedicated learning rate in the data-parallel optimizer.
- `loss.entail_weight`: weight for the Lorentz entailment-cone term exposed by
  the model. Zero disables it.

Checkpoint the loss state as well as the model state when curvature is learned.
Hyperbolic radii, angles, aperture, and cone plots are geometry-specific sanity
checks described in the visualization guide.

## Optional regularization and objective variants

All weights below are additive to the selected contrastive objective unless
noted otherwise.

### Batch uniformity

`loss.batch_uniformity_enabled` applies a pairwise-distance uniformity penalty
to each modality and `loss.batch_uniformity_weight` scales it. Its purpose is
to spread samples over the representation space and resist concentration. It
is batch-dependent, so batch/effective-batch size is part of the experiment.

### Variance/covariance (VC)

`loss.vc_reg_enabled` enables two anti-collapse terms, scaled together by
`loss.vc_weight`: a variance floor (`loss.vc_gamma`) discourages inactive
coordinates, while the covariance penalty discourages redundant off-diagonal
dimensions. `loss.vc_covariance_weights: [s1, s2]` weights the covariance
penalty per tower. Use the key `vc_reg_enabled`; both supplied Hydra
configurations and the CLI map to this spelling.

### SigLIP objective

`datamodule.siglip: true` (or the equivalent resolved `siglip` value) selects
the pairwise sigmoid loss instead of `CiipLoss`. This is an objective
alternative, not a regularizer, and its distributed neighbor exchange has
different behavior from the baseline global softmax. Distributed SigLIP uses a
single ring-exchange path rather than separate directional implementations. Do
not expect the CIIP regularizer flags above to carry over: the factory returns
`SigLipLoss` directly.

### Patch masking and reconstruction

`model.patch_masking: true` masks a fraction of input patches according to
`model.patch_mask_ratio` and activates modality-specific MAE-style decoders.
The model reconstructs targets for the masked patches and exposes their mean as
`recon_loss`. `recon.lambda` scales that loss; the training code warns and the
term remains inactive when reconstruction config is present but patch masking
is off. `recon.warmup_steps` / `recon.warmup_epochs` describe the intended
reconstruction schedule, but verify the selected runner's effective lambda in
its logs before relying on warm-up behavior.

## Other training axes (not separate objectives)

- **Seasonal SSL4EO sampling and preprocessing:** S1/S2 tier, band selection,
  normalization/transforms, train/validation split, and seasonal choice change
  the data distribution and must be reported.
- **Frozen/partially initialized text:** the text builder can load MS-CLIP or a
  local checkpoint and configures selected text layers as trainable. Record the
  source checkpoint and trainable policy.
- **Distributed strategy:** single GPU, `DataParallel`, DDP, Horovod flags,
  local versus global loss, `gather_with_grad`, and gradient accumulation alter
  the effective negatives and optimization even when the named objective is
  unchanged.
- **Orthogonal mapping:** this experimental, incomplete training branch was
  removed. It is not a supported pretraining setting.
- **Centroid regularization:** inactive constructor placeholders were removed.
  It is not a supported pretraining setting.
- **Distillation/CoCa:** no maintained end-to-end pretraining setting is
  implemented. Maintained Hydra runners reject distillation requests before
  model setup; inactive loss placeholders have been removed.

## Recommended experiment record

For reproducibility, report the modality setting; encoder architecture and
initialization; S1/S2 tier and bands; embedding (and Matryoshka prefix)
dimension; geometry and curvature; every nonzero loss weight; masking ratio and
reconstruction weight; global/effective batch size; distributed gather mode;
precision; optimizer/schedule; seed; checkpoint epoch; and the resolved Hydra
configuration.
