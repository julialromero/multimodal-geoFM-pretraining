# Phase 3 runtime and configuration assessment

This assessment closes the two proposal-only Phase 3 batches. It does not move
or delete files. The Phase 0–2 reports are the evidence base; static evidence is
not treated as proof that an entry point or configuration is unused.

## Runtime package boundary

The training runners and their supporting modules already share the
`ciip/open_clip_train/` boundary. The three Hydra runners resolve configuration
paths relative to their current source locations:

- `ciip/open_clip_train/run_train_val.py` uses `config_path="configs"`;
- `ciip/open_clip_train/run_train_val_distributed.py` uses
  `config_path="configs"`; and
- `ciip/open_clip_train/dataparallel/run_train_val_dataparallel.py` uses
  `config_path="../configs"`.

Moving a runner would therefore change Hydra search behavior unless a wrapper or
explicit compatibility path were retained. The operator confirmed that no
untracked external HPC launcher targets these paths.

**Decision:** retain the current training package and runner paths. A directory
move offers no behavior-preserving organization benefit large enough to offset
the caller and Hydra compatibility risk.

The `ciip/evaluation/` boundary is also retained. Evaluation code and notebooks
may be invoked directly, and the audit cannot establish the absence of external
callers. No import shim or path migration is necessary when the existing layout
is left intact.

## Configuration boundary

The active configuration tree is already grouped under
`ciip/open_clip_train/configs/`, with default configurations at its root and
override fragments under `datamodule/` and `train/`. The Phase 1 configuration
contract resolves all three declared runners to `prod_default.yaml`; the
external audit deliberately does not classify unbound fragments as unused
because Hydra overrides can select them dynamically.

**Decision:** retain all current configuration paths and names. In particular,
do not move `prod_default.yaml`, `local_default.yaml`, or the `datamodule/` and
`train/` fragments. This preserves Hydra search paths, override names, and any
external launch commands.

## Compatibility impact

- **Public paths:** unchanged.
- **Imports and entry points:** unchanged.
- **Hydra paths, keys, and defaults:** unchanged.
- **Checkpoint namespaces, shapes, and dtypes:** unchanged.
- **Notebooks and launchers:** unchanged.
- **Rollback:** documentation-only commit can be reverted; there is no runtime
  migration to undo.

## Result

Phase 3 selected path-preserving documentation and discoverability improvements
and rejected runtime/configuration moves whose compatibility cost exceeded their
organizational value. No existing file was relocated or deleted. Phase 4 may
assess removals individually now that the operator has resolved the
external-launcher limitation; each candidate must still pass the documented
dependency gate.

HPC and Slurm launchers remain tracked on the dedicated `slurm-hpc` branch. They
will not be copied into `distributed`, whose training path remains the current
PyTorch Distributed implementation. Remote-branch launcher evidence remains
part of every relevant candidate assessment.
