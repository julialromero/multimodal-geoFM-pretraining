# Phase 4 candidate assessment: paired transforms module

## Candidate

- **Path:** `ciip/open_clip_train/transforms.py`
- **Reason proposed:** the current-branch static inventory reports no inbound
  Python import or literal reference to this module.
- **Decision:** **RETAIN**

## Git-visible dependency evidence

- **Current imports:** no active tracked import was found. The distributed runner
  contains a commented historical import, which is not a runtime dependency.
- **Public surface:** the module defines `PairGeom` and `PairAugmented`, reusable
  paired-modality transform classes. Removing the module would remove both import
  paths even though current training uses torchvision transforms directly.
- **Fetched remote refs:** at least one fetched development branch,
  `origin/codex/add-croma-evaluation-to-unified_evaluation`, contains the active
  import `from ciip.open_clip_train import transforms`. Several other fetched
  refs retain the commented distributed-runner import.
- **History:** the file has dedicated package-import maintenance commits, showing
  that it has been treated as a supported module rather than an accidental
  generated artifact.
- **Entry points/checkpoints:** it has no main guard, argument parser, dynamic
  import, or checkpoint load/save signal.
- **Notebooks/launchers:** no tracked notebook or launcher directly references
  the module. Remote launcher evidence does not establish that their invoked
  Python runners cannot reach it through branch-specific code.

## Compatibility evidence

- **Hydra:** no configuration key or search path directly names the module.
- **Checkpoint contract:** the module has no parameters or checkpoint namespace.
- **Runtime behavior:** deletion could break the active fetched-branch import and
  downstream users of `PairGeom` or `PairAugmented`; static current-branch
  absence cannot rule those users out.
- **External launchers:** the operator confirmed that no untracked external HPC
  launchers exist. Tracked Slurm launchers remain isolated on `slurm-hpc`.

## Decision and rollback

The candidate is retained because fetched remote history provides a concrete
consumer and the public transform classes are not covered by a compatibility
shim elsewhere. This is stronger evidence than the current branch's missing
inbound edge. No implementation change occurred, so no rollback is needed.

Reconsideration requires either migrating all known consumers to a supported
replacement with an import shim or proving that the relevant remote work is
obsolete through explicit owner review. Static analysis alone is insufficient.
