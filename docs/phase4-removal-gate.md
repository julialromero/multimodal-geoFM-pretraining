# Phase 4 removal gate

Phase 4 is candidate-driven: no file or symbol is removed merely because a
static audit reports no inbound reference. Each candidate must have its own
completed evidence record and review approval before an implementation commit.

## Global prerequisite

This prerequisite is satisfied: the repository operator confirmed that no
untracked external HPC launchers exist. HPC and Slurm scripts remain tracked on
the dedicated `slurm-hpc` branch and must remain in the fetched-remote evidence
for relevant candidates. They are intentionally not copied into `distributed`,
which continues to use its current PyTorch Distributed training implementation.

## Candidate record

Copy this section for each proposed removal and replace every placeholder.

```text
Candidate path/symbol:
Reason proposed:
Proposed by:

Git-visible dependency evidence:
- inbound Python imports:
- reverse import edges:
- literal documentation/configuration references:
- entry points and argument parsers:
- notebooks and tracked launchers:
- fetched-remote history/launchers:
- dynamic-import review:

Compatibility evidence:
- Hydra path/key/override impact:
- checkpoint namespace/shape/dtype impact:
- applicable structural contracts:
- applicable runtime baselines:
- additional targeted tests:

External evidence:
- HPC launcher confirmation or inventory:
- known downstream consumers:

Decision: BLOCKED | RETAIN | APPROVED
Reviewer:
Approval reference:
Rollback plan:
```

`APPROVED` is valid only when every evidence field is complete, the external
launcher prerequisite is satisfied, and applicable tests pass. Uncertainty
requires `BLOCKED` or `RETAIN`, never approval.

## Current candidate register

| Candidate | Status | Reason |
| --- | --- | --- |
| [`ciip/open_clip_train/transforms.py`](phase4-candidate-transforms.md) | **RETAIN** | A fetched development branch actively imports it, and deleting it would remove the public `PairGeom`/`PairAugmented` paths. |
| [`clip/dataset.py`](phase4-candidate-clip-dataset.md) | **BLOCKED** | No Git-visible caller was found, but deletion of the public `clip.dataset` API requires explicit repository-owner approval. |

## Implementation rules

- Put unrelated removals in separate commits or pull requests.
- Preserve compatibility shims when a public import or executable path must
  remain available.
- Regenerate all audit artifacts after the implementation is staged.
- Run all dependency-independent tests and every applicable Phase 2 runtime
  baseline in the declared ML environment.
- Record the exact commands and results in the removal pull request.
- Revert the candidate removal if callers, checkpoint incompatibilities, or
  model behavior regressions are discovered.
