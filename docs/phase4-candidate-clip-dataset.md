# Phase 4 candidate assessment: legacy CLIP dataset module

## Candidate

- **Path:** `clip/dataset.py`
- **Reason proposed:** the file duplicates generic CLIP/WebDataset loading
  utilities while active CIIP runners import `ciip.open_clip_train.data`.
- **Decision:** **REMOVED — explicitly approved by the repository owner**

## Git-visible dependency evidence

- **Current imports:** the Phase 0 inventory reports no inbound Python edge or
  literal reference to `clip/dataset.py`.
- **Fetched refs:** an exact search of every fetched remote ref found no
  `from clip.dataset`, `import clip.dataset`, `from clip import dataset`, or
  `clip/dataset.py` reference.
- **Package exports:** `clip/__init__.py` exports only `.clip`; it does not import
  or re-export the dataset module.
- **Active data path:** current training, embedding, visualization, and test code
  imports `get_data` from `ciip.open_clip_train.data`, not `clip.dataset`.
- **Public surface:** the candidate nevertheless defines public CSV, ImageNet,
  WebDataset, and synthetic-dataset utilities. An external Python consumer could
  import those APIs even though no Git-visible caller does.
- **History:** the module entered in commit `3b965be` as a standalone file and
  has since changed alongside embedding/regularization work. History does not
  prove it is safe to remove.
- **Runtime/checkpoints:** it has no main guard, argument parser, dynamic import,
  or checkpoint load/save signal.

## Compatibility evidence

- **Hydra:** no audited configuration or Hydra search path names the module.
- **Model/checkpoint behavior:** the file defines data-loading utilities only and
  has no model parameters or checkpoint namespace. Its deletion should not alter
  the Phase 2 model contracts.
- **Dependencies:** removing it would eliminate imports of WebDataset, pandas,
  braceexpand, Horovod, PIL, NumPy, Torch, and torchvision from this file, but it
  does not prove those packages are unused elsewhere.
- **External launchers:** the operator confirmed no untracked external HPC
  launchers exist. Fetched Slurm launchers remain isolated on `slurm-hpc`; exact
  remote-ref searches found no candidate reference.

## Owner decision and implementation

The repository owner explicitly approved removal of `clip/dataset.py` after the
path was clarified against the separate, retained `ciip/dataset.py`. The file is
removed in a dedicated implementation commit. `clip/__init__.py` is unchanged
because it did not import or export this module.

The audit artifacts are regenerated after staging the deletion, and the complete
dependency-independent suite is required to pass. The rollback is to revert the
removal commit, restoring the exact legacy module if an unknown downstream
consumer is discovered.
