# Repository documentation

This directory contains generated audit evidence and maintained cleanup guides.
Generated files should be updated with their documented generator rather than
edited by hand.

## Cleanup guides

- [`cleanup-progress.md`](cleanup-progress.md) — current status and remaining
  gates across all cleanup phases.
- [`phase0-cleanup.md`](phase0-cleanup.md) — safety policy, audit workflow, and
  regeneration commands.
- [`phase3-organization-plan.md`](phase3-organization-plan.md) — non-destructive
  organization invariants and review batches.
- [`phase3-runtime-config-assessment.md`](phase3-runtime-config-assessment.md) —
  path-preserving decisions for runtime and configuration boundaries.
- [`phase4-removal-gate.md`](phase4-removal-gate.md) — per-candidate evidence
  template, approval rules, and the current removal register.
- [`phase4-candidate-transforms.md`](phase4-candidate-transforms.md) — completed
  assessment retaining the paired-transform module based on remote consumers.
- [`phase4-candidate-clip-dataset.md`](phase4-candidate-clip-dataset.md) — legacy
  CLIP dataset assessment awaiting an explicit owner retain/remove decision.

## Generated evidence

| Artifact | Purpose | Generator |
| --- | --- | --- |
| [`phase0-inventory.json`](phase0-inventory.json) | Git refs, tracked files, imports, reverse dependencies, references, and runtime signals. | `python tools/phase0_inventory.py` |
| [`phase1-runtime-audit.md`](phase1-runtime-audit.md) | Human-readable runtime and checkpoint worklist. | `python tools/phase0_inventory.py` |
| [`phase1-config-contracts.json`](phase1-config-contracts.json) | Hydra entrypoint and statically accessed configuration contracts. | `python tools/phase1_config_contracts.py` |
| [`phase1-external-audit.json`](phase1-external-audit.json) | Notebook, launcher, remote-ref launcher, and Hydra-fragment evidence. | `python tools/phase1_external_audit.py` |
| [`phase2-model-contracts.json`](phase2-model-contracts.json) | Model signatures and checkpoint namespace/container contracts. | `python tools/phase2_model_contracts.py` |

The generated evidence is conservative: a missing static reference does not
prove that a path is unused and never authorizes moving or deleting it.
