# Cleanup progress

This dashboard reports progress across the full cleanup plan. A later phase does
not begin merely because its tooling exists; each phase must satisfy its exit
gate, and no file is removed before dependency and compatibility review.

| Phase | Status | Completed | Remaining gate |
| --- | --- | --- | --- |
| 0 — inventory | **Complete** | Remote refs/tags, tracked files, static imports, reverse edges, literal references, and reproducible fingerprints are inventoried. | Regenerate after repository or fetched-ref changes. |
| 1 — runtime/config audit | **Complete for tracked and fetched Git content** | Python entry points, configs, notebooks, tracked launchers, remote-branch launchers, dynamic imports, and checkpoint calls are inventoried. | Repository operators must confirm whether untracked launchers exist on external HPC systems. |
| 2 — compatibility baselines | **In progress** | Public API/checkpoint contracts are captured, and all five executable CLIP, masking, and Lorentz baselines pass in the declared HPC PyTorch environment. | Add representative CIIP configurations and checkpoint compatibility coverage. |
| 3 — non-destructive organization | **Blocked by Phase 2 gate** | No model-affecting reorganization has begun. | Phase 2 runtime and checkpoint baselines must pass first. |
| 4 — audited removal | **Blocked by Phases 1–3** | No cleanup candidate has been removed. | Every candidate needs a completed dependency assessment and passing compatibility baselines. |

The first HPC execution of the five Phase 2 runtime baselines passed the CLIP
forward, checkpoint round-trip, and Lorentz tests. It also exposed two incorrect
test assumptions—not model regressions: `build_model` intentionally converts
applicable weights to float16, and `random_masking` returns `ids_keep` as its
fourth result. After those expectations were corrected, the repository operator
reported a clean HPC rerun with all five tests passing. Phase 2 therefore retains
only the representative CIIP configuration and checkpoint coverage gate.

## Current safety posture

- No existing model, training, evaluation, data, or configuration file has been
  deleted or relocated.
- Static results are evidence for review, not proof that a file is unused.
- Runtime tests skip explicitly when PyTorch is unavailable rather than passing
  silently; they execute automatically in the declared ML environment.
