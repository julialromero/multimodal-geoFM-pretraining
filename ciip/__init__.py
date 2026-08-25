"""CIIP package.

The package namespace is intentionally lightweight. Optional training and data
dependencies are imported only when their public objects are first requested.
"""

from importlib import import_module
from typing import Any


__all__ = [
    "S12Dataset",
    "Subset",
    "available_models",
    "generate_splits",
    "load",
]

_LAZY_EXPORTS = {
    "available_models": (".ciip", "available_models"),
    "load": (".ciip", "load"),
    "S12Dataset": (".dataset", "S12Dataset"),
    "Subset": (".dataset", "Subset"),
    "generate_splits": (".dataset", "generate_splits"),
}


def __getattr__(name: str) -> Any:
    """Load legacy package-level exports on demand."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
