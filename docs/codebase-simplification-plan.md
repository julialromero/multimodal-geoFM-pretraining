# Codebase simplification plan

## Goal

Make the maintained training and evaluation workflows easier to understand and
change with fewer files, fewer code paths, and less duplicated orchestration.
The objective is not to reorganize every research artifact into a framework.
Prefer deleting unused behavior, combining equivalent behavior, and keeping
entry points thin over adding new abstraction layers.

## Working rules

1. **Protect behavior before moving it.** Add a small characterization test for
   an active workflow before refactoring it. Tests should cover contracts, not
   implementation details.
2. **Delete before abstracting.** Confirm which variants are still used, remove
   inactive variants, and only share code that has at least two current callers.
3. **Keep changes reviewable.** Each pull request should address one boundary,
   preserve documented commands, and include a measurable deletion or
   simplification.
4. **Do not preserve compatibility forever.** When a path must move, provide a
   warning wrapper for one release or a stated migration window, then delete it.
5. **Keep optional dependencies optional.** Importing core model code must not
   require evaluation datasets, plotting libraries, or experiment trackers.
6. **Measure concision.** Track maintained Python file count, lines in the ten
   largest modules, number of training entry points, and duplicated config keys.
   A phase is not successful if it only redistributes the same complexity.

## Target shape

```text
ciip/
├── models/          encoder, projection, masking, and geometry code
├── training/        shared factories, loop primitives, and thin runners
├── data/            maintained dataset adapters and transforms
├── evaluation/      reusable extraction, metrics, tasks, and thin CLIs
└── visualization/   plotting from machine-readable evaluation outputs
tests/                fast contract tests; optional integration tests marked
scripts/              only thin, user-invoked research or batch commands
docs/                 maintained workflow and design documentation
```

This is a direction, not a requirement to move everything immediately.
Third-party comparison code may remain isolated rather than being made part of
the `ciip` package.

## Phase 1: establish a reliable, minimal project surface

**Purpose:** make installation and automated checks trustworthy before runtime
refactors.

**Progress:** packaging now uses `pyproject.toml`, the package namespace is
lightweight, tests are tracked under `tests/`, the runtime compatibility suite
has moved there, and minimal CI covers installation, focused lint, and fast
tests. A repository-wide mechanical lint pass remains a separate follow-up.

- Replace `setup.py` with a `pyproject.toml` whose project name and package
  metadata describe CIIP. Declare a small core dependency set plus `train`,
  `eval`, `viz`, and `dev` optional groups. Keep `environment.yml` as the
  reproducible GPU environment rather than copying its full contents.
- Stop ignoring `.github/workflows/` and test directories. Move the runtime
  compatibility check from `tools/` to `tests/` and configure test markers for
  optional PyTorch, TorchGeo, GPU, data, and checkpoint requirements.
- Add one lightweight CI job for formatting, linting, import checks, and tests
  that need no external data. Do not add a large tool matrix.
- Add Ruff with a deliberately small initial rule set. Apply mechanical
  formatting separately from behavioral changes.

**Exit criteria**

- `pip install -e '.[dev]'` succeeds in a clean environment.
- Importing `ciip` does not load optional dataset or visualization packages.
- One documented command runs all fast checks.
- The repository no longer ignores tests or workflow definitions.

## Phase 2: remove dead options and make configuration explicit

**Purpose:** reduce the number of apparent features before consolidating code.

**Progress:** the variance/covariance regularizer now has one canonical option
name across YAML, CLI parsing, and loss construction. The incomplete orthogonal
mapping toggle, its broken training branch, and its unused model prototype have
been removed. All three maintained runners now validate their supported encoder
pairings, architectures, dimensions, batch/accumulation values, masking ratio,
validation split, and incompatible geometry settings before setup. Placeholder
removal remains ongoing; inactive distillation/CoCa loss scaffolding has been
deleted and distillation requests now fail during validation.

- Build a short table from the maintained YAML files and runtime access sites:
  config key, default, consumer, supported runner, and status.
- Delete keys and branches that are commented out, never consumed, or described
  as inactive, including legacy aliases once supplied configs no longer use
  them. Do not build another permanent audit generator.
- Validate the remaining Hydra configuration at runner startup. Put validation
  near the owning model, loss, data, or runtime component rather than in one
  repository-wide schema.
- Reduce configs to a documented base plus small overrides. Keep secrets,
  machine paths, and output paths out of tracked defaults.

**Exit criteria**

- Every tracked config key has a runtime consumer or is Hydra-owned.
- Unsupported combinations fail before model or dataset construction with a
  useful message.
- No active behavior relies on two spellings of the same option.
- The number of tracked config keys and conditional feature branches decreases.

## Phase 3: converge training onto one shared core

**Purpose:** remove duplication among single-device, data-parallel, and
distributed training without inventing a training framework.

**Progress:** the single-device and distributed runners now use the same
two-sensor model factory, and the single-device runner uses the shared scheduler
implementation. The model factory has been reduced to common constructor data
plus the Euclidean/Lorentz-specific arguments. All runners now share AdamW
parameter grouping, including no-decay, loss, curvature, and optional text
learning-rate groups. Checkpoint creation, wrapper normalization, state restore,
atomic local saves, and deletion are also shared across runners. Data-parallel
and distributed loops now share one modality-to-model batch contract for S1/S2,
S2/text, and S1/S2/text inputs. Data-parallel text construction internals and
both epoch loops also share primary/reconstruction objective composition and
warm-up behavior. Their direct-batch path now uses one forward, scalar-output
normalization, optional distillation, objective, and backward primitive.
Gradient accumulation now shares feature caching, scalar extraction, and
tracked-microbatch replacement as well. Optimizer stepping, gradient clipping,
AMP scaler updates, and Horovod synchronization use one shared primitive. The
data-parallel module now re-exports the same epoch loop, whose accumulated-batch
contract supports two- and three-tower inputs and conditionally uses DDP
`no_sync`. Only model construction for the text-specific towers remains
strategy-specific.

Implement this as several small pull requests in dependency order:

1. Extract one model factory and one loss factory with tests for the supported
   Euclidean, Lorentz, Matryoshka, and text pairings.
2. Extract checkpoint save/load and optimizer/scheduler construction.
3. Make batch parsing and loss-output naming consistent across runners.
4. Move the common epoch loop into a small set of functions. Pass distributed
   operations in explicitly rather than branching throughout model code.
5. Reduce each runner to configuration, device/distribution setup, and calls
   into the shared core. Remove a runner if a maintained PyTorch strategy can
   provide the same behavior.

Avoid a class hierarchy for runners and avoid a generic plugin system. Plain
functions and small typed result objects are sufficient until a real extension
need exists.

**Exit criteria**

- Model, loss, checkpoint, and optimizer construction each have one maintained
  implementation.
- Training runners contain no copied epoch loop.
- A tiny synthetic batch passes through every supported strategy in tests.
- Documented launch commands and checkpoint keys remain stable or have an
  explicit migration note.

## Phase 4: separate evaluation computation from presentation

**Purpose:** break up the densest modules and make metrics reusable without
duplicating loaders and plotting logic.

- Start with `ciip/evaluation/unified_evaluation.py`, then address the SSL4EO
  intrinsic-dimension and global-ID merger scripts. For each, identify pure
  units: configuration, dataset construction, feature extraction, metric
  calculation, result schema, persistence, and CLI.
- Extract only pure or independently tested units. Leave the existing module as
  a thin CLI that composes them; do not create one file per function.
- Define one versioned, machine-readable evaluation result record containing
  checkpoint, dataset/split, modality/bands, feature space, seed, arguments,
  and metrics.
- Make plotting commands consume result records rather than checkpoints or
  datasets wherever possible. Consolidate repeated CSV/JSON discovery and
  aggregation helpers.
- Move exploratory, one-off analyses to a clearly labeled `scripts/` area or
  delete them when they duplicate a maintained command.

**Exit criteria**

- CLI modules primarily parse arguments and call library functions.
- No maintained evaluation module exceeds roughly 1,000 lines without a
  documented reason.
- Feature extraction and result loading each have one implementation per data
  contract.
- Plots can be regenerated from retained machine-readable outputs.

## Phase 5: normalize names and package boundaries

**Purpose:** make navigation and imports predictable after behavior has
converged.

- Rename first-party hyphenated paths to importable `snake_case` packages,
  including intrinsic-dimension and PANGAEA result utilities.
- Keep vendored or externally derived comparison implementations under
  `comparison/`, with their origin, revision, license, local patches, and
  supported wrapper recorded. Do not restyle vendored code.
- Move maintained visualization library code under the `ciip` namespace. Keep
  only thin executable wrappers outside it if documented commands require them.
- Standardize module names around responsibilities; avoid names such as
  `utils.py`, `main.py`, and `model.py` when a more specific name is available.

**Exit criteria**

- All maintained first-party modules are importable through valid package names.
- Every broad `utils` module has been split, renamed to its actual purpose, or
  justified as a small cohesive helper module.
- Compatibility wrappers have an owner and removal date.

## Phase 6: finish repository hygiene and documentation

**Purpose:** make the simplified structure durable.

- Replace the current ignore rules with grouped rules for Python caches, local
  environments, datasets, Hydra outputs, checkpoints, evaluation outputs, and
  generated figures. Use explicit exceptions for curated examples.
- Keep only current user guides and short design notes. Put commands in one
  canonical location and link to them rather than copying setup instructions.
- Add a small contributor section covering installation, fast checks, optional
  integration checks, and where generated artifacts belong.
- Remove temporary wrappers, deprecation aliases, migration notes, and empty
  directories at the end of their stated window.

**Exit criteria**

- A fresh training or evaluation run does not dirty `git status`.
- The README, environment, package metadata, and CI use the same supported
  Python and dependency story.
- Repository-map documentation matches the actual top-level structure.

## Pull-request sequence

| PR | Scope | Expected simplification |
| --- | --- | --- |
| 1 | Packaging and dependency groups | One install contract; remove broken legacy packaging |
| 2 | Test layout, ignore rules, and minimal CI | One fast-check command; tracked CI and tests |
| 3 | Mechanical format/lint baseline | Consistent source without behavior changes |
| 4 | Config inventory followed by dead-key deletion | Fewer settings and branches |
| 5 | Shared model/loss factories | Remove duplicated construction logic |
| 6 | Shared checkpoints, optimizer, and batch contract | Remove runner-specific variants |
| 7 | Shared epoch loop and runner reduction | Thin execution-strategy entry points |
| 8 | Unified evaluation decomposition | Smaller CLI and reusable evaluation units |
| 9 | ID/diagnostic result-schema convergence | Less extraction and aggregation duplication |
| 10 | Importable path and namespace migration | Predictable first-party packages |
| 11 | Remove migration wrappers and finish docs | No indefinite compatibility clutter |

PRs should be split further if they mix mechanical movement with behavioral
changes. Conversely, do not create a PR solely to rename a private helper when
the rename belongs naturally with a deletion or extraction.

## Review checklist for every phase

- Which maintained command or contract is protected by a test?
- What code, option, branch, or duplicate path was deleted?
- Did this introduce a new abstraction? If so, which current callers require it?
- Did Python file count, large-module line count, or duplicated logic improve?
- Are optional dependencies still isolated?
- Is any compatibility layer time-bounded?
- Does a fresh run keep generated data out of version control?

The plan is complete when the common workflows are shorter and have fewer
choices—not merely when all files have been moved into new directories.
