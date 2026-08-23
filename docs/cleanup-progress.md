# Cleanup progress

This dashboard reports progress across the full cleanup plan. A later phase does
not begin merely because its tooling exists; each phase must satisfy its exit
gate, and no file is removed before dependency and compatibility review.

| Phase | Status | Completed | Remaining gate |
| --- | --- | --- | --- |
| 0 — inventory | **Complete** | Remote refs/tags, tracked files, static imports, reverse edges, literal references, and reproducible fingerprints are inventoried. | Regenerate after repository or fetched-ref changes. |
| 1 — runtime/config audit | **Complete** | Python entry points, configs, notebooks, tracked launchers, remote-branch launchers, dynamic imports, and checkpoint calls are inventoried. The operator confirmed there are no untracked external HPC launchers. | Regenerate after caller or configuration changes. |
| 2 — compatibility baselines | **Complete** | Public API/checkpoint contracts are captured. All seven executable CLIP, masking, Lorentz, and representative CIIP forward/checkpoint baselines pass in the declared HPC PyTorch/TorchGeo environment. | Rerun after model-facing changes. |
| 3 — non-destructive organization | **Complete** | Documentation and tooling navigation landed; runtime and configuration boundaries were assessed and retained to preserve callers and Hydra behavior. No existing file moved. | Rerun the audits after future organization changes. |
| 4 — audited removal | **In progress** | `ciip/open_clip_train/transforms.py` and `ciip/dataset.py` are retained due concrete consumers; `clip/dataset.py` has no Git-visible caller but is blocked pending an explicit owner decision about its public API. Nothing has been removed. | Owner decides whether to retain or remove **`clip/dataset.py`** (not `ciip/dataset.py`); every approved removal then requires regenerated audits and applicable tests. |

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
