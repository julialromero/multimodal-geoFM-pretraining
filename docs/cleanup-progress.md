# Cleanup progress

This dashboard reports progress across the full cleanup plan. A later phase does
not begin merely because its tooling exists; each phase must satisfy its exit
gate, and no file is removed before dependency and compatibility review.

| Phase | Status | Completed | Remaining gate |
| --- | --- | --- | --- |
| 0 — inventory | **Complete** | Remote refs/tags, tracked files, static imports, reverse edges, literal references, and reproducible fingerprints are inventoried. | Regenerate after repository or fetched-ref changes. |
| 1 — runtime/config audit | **Complete for tracked and fetched Git content** | Python entry points, configs, notebooks, tracked launchers, remote-branch launchers, dynamic imports, and checkpoint calls are inventoried. | Repository operators must confirm whether untracked launchers exist on external HPC systems. |
| 2 — compatibility baselines | **Complete** | Public API/checkpoint contracts are captured. All seven executable CLIP, masking, Lorentz, and representative CIIP forward/checkpoint baselines pass in the declared HPC PyTorch/TorchGeo environment. | Rerun after model-facing changes. |
| 3 — non-destructive organization | **In progress** | Organization boundaries, invariants, and the first review batches are documented. No existing file has moved. | Audit each proposed move against callers, imports, configs, checkpoints, and the Phase 2 baselines before applying it. |
| 4 — audited removal | **Blocked by Phases 1–3** | No cleanup candidate has been removed. | Every candidate needs a completed dependency assessment and passing compatibility baselines. |

The first HPC execution of the five Phase 2 runtime baselines passed the CLIP
forward, checkpoint round-trip, and Lorentz tests. It also exposed two incorrect
test assumptions—not model regressions: `build_model` intentionally converts
applicable weights to float16, and `random_masking` returns `ids_keep` as its
fourth result. After those expectations were corrected, the repository operator
reported a clean HPC rerun with all five tests passing. The repository operator
subsequently reported that the representative CIIP transformer forward and
strict checkpoint round-trip baselines also pass. These checks preserve the
production framework and two-band Sentinel-1/twelve-band Sentinel-2 modality
contract at smoke-test size, completing the Phase 2 gate.

Phase 3 proceeds according to [`phase3-organization-plan.md`](phase3-organization-plan.md).

## Current safety posture

- No existing model, training, evaluation, data, or configuration file has been
  deleted or relocated.
- Static results are evidence for review, not proof that a file is unused.
- Runtime tests skip explicitly when PyTorch is unavailable rather than passing
  silently; they execute automatically in the declared ML environment.
