# Phase 4 candidate assessment: CIIP S12 dataset module

## Candidate

- **Path:** `ciip/dataset.py`
- **Owner input:** removal was approved by name.
- **Decision:** **RETAIN — dependency audit overrides removal approval**

This path is distinct from the previously assessed `clip/dataset.py`.

## Dependency evidence

- `ciip/__init__.py` executes `from .dataset import *`, so deleting the file
  without changing the package initializer would break every `import ciip`.
- The module exposes `S12Dataset`, `Subset`, and `generate_splits` as public APIs;
  the wildcard package export also makes them available from `ciip`.
- Exact searches across fetched refs found `tests/test_dataset.py` importing
  `S12Dataset` from `ciip.dataset` on 28 remote branches.
- The consumers include `origin/slurm-hpc`, whose tracked HPC/Slurm implementation
  must remain isolated and functional according to the recorded branch policy.
- No main guard, argument parser, dynamic import, or checkpoint operation was
  found, but those negative signals do not outweigh concrete imports.

## Compatibility impact

Removing the module would break the package initializer and branch-specific
dataset tests. Replacing it would require a supported `S12Dataset` compatibility
path, updates to every known consumer, and validation on `slurm-hpc`; none of
those prerequisites is part of the current cleanup change.

The module has no model checkpoint namespace, but import and data-pipeline
compatibility are sufficient reasons to retain it.

## Result

The removal is not implemented. The cleanup policy requires dependency evidence
to override an approval made before those dependencies were presented. The
candidate is registered as `RETAIN`; reconsideration requires a cross-branch
migration plan and replacement compatibility tests.
