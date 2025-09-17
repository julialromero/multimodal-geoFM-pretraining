"""Tests for ``visualizations.ssl4eo.embedding_collapse_diagnostics``."""

import logging
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _ensure_torch_stub() -> None:
    """Install a minimal torch stub so the module can be imported during tests."""

    if "torch" in sys.modules:  # Real torch (or another stub) is already available.
        return

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.float64 = "float64"

    class _Generator:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
            pass

        def manual_seed(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
            return None

    torch_stub.Generator = _Generator

    # Basic callables that return ``None`` so that any accidental invocations during
    # import-time (there are none today) have predictable behaviour.
    torch_stub.load = lambda *args, **kwargs: None  # pragma: no cover - simple stub
    torch_stub.as_tensor = lambda *args, **kwargs: types.SimpleNamespace(
        reshape=lambda *a, **k: types.SimpleNamespace()
    )  # pragma: no cover - simple stub
    torch_stub.stack = lambda *args, **kwargs: None  # pragma: no cover - simple stub
    torch_stub.ones_like = lambda *args, **kwargs: None  # pragma: no cover - simple stub
    torch_stub.randint = lambda *args, **kwargs: None  # pragma: no cover - simple stub
    torch_stub.sum = lambda *args, **kwargs: types.SimpleNamespace(
        cpu=lambda: types.SimpleNamespace(numpy=lambda: None)
    )  # pragma: no cover - simple stub
    torch_stub.var = lambda *args, **kwargs: types.SimpleNamespace(
        cpu=lambda: types.SimpleNamespace(numpy=lambda: None)
    )  # pragma: no cover - simple stub
    torch_stub.cat = lambda *args, **kwargs: None  # pragma: no cover - simple stub
    torch_stub.linalg = types.SimpleNamespace(
        eigvalsh=lambda *args, **kwargs: types.SimpleNamespace(
            flip=lambda *a, **k: types.SimpleNamespace(
                cpu=lambda: types.SimpleNamespace(numpy=lambda: None)
            )
        )
    )  # pragma: no cover - simple stub

    nn_module = types.ModuleType("torch.nn")
    functional_module = types.ModuleType("torch.nn.functional")
    nn_module.functional = functional_module
    torch_stub.nn = nn_module

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_module
    sys.modules["torch.nn.functional"] = functional_module


def _ensure_matplotlib_stub() -> None:
    """Install a minimal matplotlib stub if the real dependency is unavailable."""

    if "matplotlib" in sys.modules:
        return

    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")

    class _DummyFig:
        def savefig(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
            return None

        def tight_layout(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
            return None

    class _DummyAx:
        def __getattr__(self, name):  # pragma: no cover - simple stub
            return lambda *args, **kwargs: None

    def _subplots(*args, **kwargs):  # pragma: no cover - simple stub
        return _DummyFig(), _DummyAx()

    pyplot_module.subplots = _subplots
    pyplot_module.close = lambda *args, **kwargs: None  # pragma: no cover - simple stub

    matplotlib_stub.pyplot = pyplot_module

    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_module


def _ensure_numpy_stub() -> None:
    """Install a minimal numpy stub when the dependency is unavailable."""

    if "numpy" in sys.modules:
        return

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = object
    numpy_stub.float32 = "float32"
    numpy_stub.float64 = "float64"
    numpy_stub.mean = lambda *args, **kwargs: 0.0  # pragma: no cover - simple stub
    numpy_stub.std = lambda *args, **kwargs: 0.0  # pragma: no cover - simple stub
    numpy_stub.min = lambda *args, **kwargs: 0.0  # pragma: no cover - simple stub
    numpy_stub.max = lambda *args, **kwargs: 0.0  # pragma: no cover - simple stub
    numpy_stub.median = lambda *args, **kwargs: 0.0  # pragma: no cover - simple stub
    numpy_stub.empty = lambda *args, **kwargs: []  # pragma: no cover - simple stub
    numpy_stub.stack = lambda *args, **kwargs: []  # pragma: no cover - simple stub
    numpy_stub.array = lambda *args, **kwargs: []  # pragma: no cover - simple stub
    numpy_stub.unique = lambda *args, **kwargs: []  # pragma: no cover - simple stub
    numpy_stub.concatenate = lambda *args, **kwargs: []  # pragma: no cover - simple stub
    numpy_stub.savez = lambda *args, **kwargs: None  # pragma: no cover - simple stub

    class _DummyNpGenerator:
        def choice(self, *args, **kwargs):  # pragma: no cover - simple stub
            return []

    numpy_stub.random = types.SimpleNamespace(default_rng=lambda *a, **k: _DummyNpGenerator())

    sys.modules["numpy"] = numpy_stub


def _ensure_sklearn_stub() -> None:
    """Install a minimal scikit-learn stub for optional dependencies."""

    if "sklearn" in sys.modules:
        return

    sklearn_stub = types.ModuleType("sklearn")
    manifold_module = types.ModuleType("sklearn.manifold")

    class _DummyTSNE:
        def fit_transform(self, *args, **kwargs):  # pragma: no cover - simple stub
            return []

    manifold_module.TSNE = _DummyTSNE
    sklearn_stub.manifold = manifold_module

    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.manifold"] = manifold_module


_ensure_torch_stub()
_ensure_matplotlib_stub()
_ensure_numpy_stub()
_ensure_sklearn_stub()

from visualizations.ssl4eo.embedding_collapse_diagnostics import parse_linear_probe_csv


def test_parse_linear_probe_csv_missing_file(tmp_path, caplog):
    """Missing CSVs should be skipped with a warning instead of crashing."""

    missing_csv = tmp_path / "does_not_exist.csv"

    with caplog.at_level(logging.WARNING, logger="embedding_collapse"):
        result = parse_linear_probe_csv(
            missing_csv,
            model_pattern=None,
            metric="accuracy",
            k_value=None,
        )

    assert result == {}
    assert str(missing_csv) in caplog.text
