# Phase 4 candidate assessment: package marker modules

## Candidates

- `ciip/open_clip_train/dataparallel/__init__.py`
- `visualizations/ssl4eo/__init__.py`
- **Decision:** **RETAIN both**

These modules contain only package docstrings and therefore appear as leaf files
in a source-level import graph. Their package role, rather than executable code,
is the relevant dependency.

## Dependency evidence

- `setup.py` uses `setuptools.find_packages()`. Both directories are discovered
  as regular packages because their `__init__.py` markers exist; deleting the
  markers can remove them from package discovery in this build configuration.
- Current code imports
  `ciip.open_clip_train.dataparallel.factory`, `.model_arch`, and `.train` through
  the marked data-parallel package.
- Current evaluation, visualization, and intrinsic-dimension code imports several
  `visualizations.ssl4eo.*` modules through the marked visualization package.
- The markers provide package documentation and make the intended namespace
  boundary explicit to Python tools that do not treat implicit namespace packages
  equivalently.
- Removing either marker offers negligible maintenance benefit while introducing
  installation, discovery, and tooling compatibility risk.

## Compatibility evidence

- The files define no model parameters, Hydra bindings, entry points, dynamic
  imports, or checkpoint operations.
- Their deletion could still alter installed package contents and import
  behavior, which is sufficient to reject removal.
- No compatibility shim is needed when the markers remain in place.

## Result

Both package markers are retained. Empty or docstring-only `__init__.py` files
must not be classified as unused solely because no module imports the marker
itself. No implementation changed, and no owner input is required.
