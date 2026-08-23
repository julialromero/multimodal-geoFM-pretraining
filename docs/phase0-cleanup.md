# Phase 0 cleanup audit

See the [full cleanup progress dashboard](cleanup-progress.md) for current status
and remaining gates across Phases 0–4.

Phase 0 establishes an evidence base for cleanup; it does **not** remove or
relocate model, training, evaluation, data, or configuration files. A candidate
must remain in place until its imports, command-line entry points, configuration
references, checkpoint compatibility, and remote-branch history have been
reviewed.

## Reproducing the inventory

Authenticate and synchronize Git metadata before generating the report:

```bash
gh auth status
git fetch --all --tags --prune
python tools/phase0_inventory.py
```

The generated [`phase0-inventory.json`](phase0-inventory.json) records every
fetched remote branch and tag with its object ID, date, and subject, except that
the current branch's upstream uses stable `CURRENT_CHECKOUT` revision fields. A
push advances that upstream to the commit containing the report, so embedding
its pre-push SHA would make the checked-in artifact stale immediately. It also
records tracked-file counts and the static import edges for every tracked Python
file. The report is deterministic for a given checkout and fetched-ref state.
The generated report excludes its own path from file counts and byte totals so
regeneration cannot change the values it is measuring.
Its `source_fingerprint` hashes the path and contents of every audited file
except the generated report. Unlike embedding `HEAD`, this fingerprint has no
commit/report cycle and can be reproduced exactly from the commit containing
the report.

Static imports are only the first dependency signal. Dynamic imports, Hydra
targets, shell/Slurm entry points, notebooks, checkpoint key layouts, and paths
constructed at runtime are not provably covered by an AST scan.

## Removal gate

Before removing or moving any candidate, record and review all of the following:

1. Static inbound and outbound Python imports.
2. References in YAML, JSON, shell scripts, Slurm scripts, notebooks, and docs.
3. Dynamic imports and module-based entry points (`python -m ...`).
4. Model construction, state-dict keys, resume paths, and checkpoint consumers.
5. Training and evaluation smoke tests for each affected model family.
6. Whether a remote branch contains a newer or divergent implementation.

An empty static-import list is not proof that a file is unused. No deletion is
authorized by the inventory alone.

## Cleanup phases

The cleanup proceeds in reviewable increments, with model behavior protected by
tests and checkpoint checks throughout:

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 0 | Inventory refs, files, and static Python dependency edges. | Inventory regenerates cleanly and limitations are explicit. |
| 1 | Audit runtime entry points, configs, scripts, notebooks, and documentation references. | Every proposed candidate has an owner or a recorded dependency assessment. |
| 2 | Establish model-construction, forward-pass, and checkpoint compatibility tests. | Affected model families have passing behavior baselines. |
| 3 | Make non-destructive organization and documentation changes. | Imports, entry points, and compatibility tests remain green. |
| 4 | Remove only candidates approved by the audit. | Each deletion links to its completed audit and passes the Phase 2 baselines. |

Phase 0 now includes relative imports and reverse (`imported_by`) edges. The
first Phase 1 pass also records literal module/path references from tracked
Markdown, JSON, YAML, INI, shell, Slurm, and text files. Phase 1 is intentionally
not declared complete: static scans cannot discover runtime-computed references,
and those must be audited before cleanup candidates are proposed.

The generated [`phase1-config-contracts.json`](phase1-config-contracts.json)
resolves each training runner's `@hydra.main` binding, verifies that its selected
configuration exists, and compares direct `args.*` accesses with statically
declared YAML keys. Reported missing keys require review because Hydra overrides
or guarded access may supply them at runtime; they do not authorize code removal.

The generated [`phase1-external-audit.json`](phase1-external-audit.json) covers
tracked notebooks and launchers, launcher paths present on fetched remote
branches, and Hydra fragments not directly bound by the default runners. Phase 1
is complete for content visible in Git. Operator confirmation is still required
for untracked launchers stored only on external HPC systems.

The inventory additionally flags Python main guards, argument-parser creation,
dynamic import calls, and checkpoint load/save calls. These are audit signals,
not declarations that a module is safe or unsafe to change. In particular,
checkpoint calls identify where Phase 2 compatibility baselines are needed.
The generated [`phase1-runtime-audit.md`](phase1-runtime-audit.md) turns these
signals into a concise, source-fingerprinted worklist for that manual review.

## Phase 2 progress

The generated [`phase2-model-contracts.json`](phase2-model-contracts.json)
captures public function, model class, constructor, and forward-method signatures
for the core CIIP, CLIP, masking, loss, Lorentz, MAE, and data-parallel model
modules. It also records literal checkpoint container keys and namespace prefixes
used by loader functions so compatibility-sensitive mappings are reviewable.
The structural snapshot remains the review baseline. Phase 2 is complete: the
repository operator reported all seven construction, forward-pass, and
checkpoint round-trip tests passing with the declared ML dependencies and
representative CIIP configuration on HPC. These tests must be rerun after any
model-facing Phase 3 change.
Dependency-backed baseline tests now live in `tools/test_phase2_runtime.py` and
cover a small CLIP forward pass, state-dict and `build_model` compatibility,
masking invariants, a Lorentz exp/log round trip, and a representative transformer
CIIP forward/checkpoint round trip using the production two-band Sentinel-1 and
twelve-band Sentinel-2 modality contract at smoke-test size. They skip explicitly
when PyTorch—or TorchGeo for the CIIP checks—is unavailable.
The baselines preserve the existing mixed-precision behavior of CLIP
`build_model` and the four-value masking return contract, including `ids_keep`.
