# Phase 4 candidate assessment: CROMA pretraining implementation

## Candidate

- **Path:** `comparison/CROMA-main/pretrain_croma.py`
- **Reason proposed:** the static Python graph has no inbound import edge and the
  module has no executable main guard.
- **Decision:** **RETAIN**

## Dependency and purpose evidence

- The vendored CROMA README explicitly states that `pretrain_croma.py` is needed
  to pretrain CROMA models from scratch. This is a documented supported workflow,
  even though the repository's evaluation path uses `use_croma.py` instead.
- The file contains the CROMA training architecture and loss building blocks,
  including feed-forward, self-attention, cross-attention, transformer, masking,
  reconstruction, and contrastive-loss utilities. It is source code rather than
  a generated artifact.
- Current evaluation modules dynamically load `comparison/CROMA-main/use_croma.py`
  for pretrained-model evaluation. That does not replace the from-scratch
  pretraining implementation or justify deleting it.
- Repository history shows the file arrived with the CROMA comparison and was
  subsequently updated. No evidence identifies it as an accidental duplicate.
- Exact current-tree searches found documentation of the file but no Python
  importer. This is expected for a standalone reference implementation whose
  classes may be imported by user-written pretraining drivers.

## Compatibility evidence

- **Model surface:** the file defines neural-network modules and loss functions;
  deleting it would remove a documented model-pretraining API.
- **Checkpoints:** although the static runtime audit finds no checkpoint calls in
  this file, its architectures can define parameter namespaces for independently
  trained CROMA checkpoints.
- **Hydra/launchers:** no audited Hydra binding or tracked launcher invokes it.
  Absence of a default launcher does not invalidate the README workflow.
- **External launchers:** the operator confirmed no untracked HPC launcher exists,
  and Slurm code remains on `slurm-hpc`.

## Result

The module is retained because documentation establishes an intentional,
supported purpose and removal would reduce CROMA functionality. No owner input
is required. Reconsideration would require explicitly dropping from-scratch
CROMA pretraining support and updating the vendored README in the same review.
