# Phase 3 non-destructive organization plan

Phase 3 improves repository navigation without deleting implementations or
changing runtime behavior. Phase 2 is complete: the repository operator reported
all seven dependency-backed model baselines passing in the declared HPC
PyTorch/TorchGeo environment.

## Invariants

Every Phase 3 change must preserve:

- import paths used by tracked code and fetched remote-branch launchers;
- Hydra configuration paths, keys, defaults, and override behavior;
- command-line entry points and their arguments;
- checkpoint container keys, parameter namespaces, shapes, and dtypes;
- notebook and launcher paths unless their callers are updated in the same
  reviewable change; and
- all structural and dependency-backed Phase 2 contracts.

Moves must use `git mv` and include compatibility shims when an import or entry
point is externally visible. Static absence of a caller is not permission to
move a file. No Phase 3 change may delete a file.

## Review batches

Organization will be split into independently reviewable batches:

1. **Documentation navigation (complete).** [`README.md`](README.md) indexes the
   maintained guides and generated evidence without moving existing files.
2. **Audit-tool discoverability (complete).** [`../tools/README.md`](../tools/README.md)
   documents the generators and tests while retaining every current command and
   module path.
3. **Runtime package proposals.** Assess possible package boundaries for
   training, evaluation, and model code. Apply no move until imports, configs,
   notebooks, launchers, remote history, and checkpoints have been reviewed.
4. **Configuration proposals.** Assess configuration grouping while retaining
   Hydra search paths and override names through compatibility paths.

The first two path-preserving batches are complete. The latter two remain
proposals until their dependency assessments are recorded.

## Per-change evidence

Each organization pull request must record:

- paths affected and whether any public path changes;
- inbound imports and literal/configuration references;
- relevant entry points, notebooks, and launchers;
- checkpoint impact (or why none exists);
- exact static audit and Phase 2 test commands run; and
- a rollback strategy.

External-only HPC launchers remain outside Git's visibility. Operator
confirmation or an explicit path inventory is still required before reorganizing
any runtime or configuration path those launchers might invoke.

## Exit gate

Phase 3 completes only after the approved path-preserving organization batches
land, all generated audits reproduce, and all seven Phase 2 runtime baselines
continue to pass. Phase 4 removals remain prohibited until then.
