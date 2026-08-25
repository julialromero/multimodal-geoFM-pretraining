"""Validation for the maintained Hydra training configuration."""

from collections.abc import Mapping
from typing import Any


def _get(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for key in path.split("."):
        if isinstance(value, Mapping):
            value = value.get(key, default)
        else:
            value = getattr(value, key, default)
        if value is default:
            break
    return value


def _positive(config: Any, path: str) -> None:
    value = _get(config, path)
    if value is None or value <= 0:
        raise ValueError(f"{path} must be greater than zero; got {value!r}")


def validate_training_config(config: Any, *, runner: str) -> None:
    """Fail early for unsupported or internally inconsistent training settings."""
    supported_runners = {"single", "distributed", "dataparallel"}
    if runner not in supported_runners:
        raise ValueError(f"unknown runner {runner!r}; expected one of {sorted(supported_runners)}")

    encoder_pair = _get(config, "model.encoder_pair", "s1s2")
    allowed_pairs = {"default", "s1s2"}
    if runner == "dataparallel":
        allowed_pairs |= {"s2_text", "s1s2_text"}
    if encoder_pair not in allowed_pairs:
        raise ValueError(
            f"model.encoder_pair={encoder_pair!r} is not supported by the {runner} runner; "
            f"choose one of {sorted(allowed_pairs)}"
        )

    framework = _get(config, "model.framework")
    allowed_frameworks = {"resnet18", "resnet50", "modified_resnet", "transformer"}
    if framework not in allowed_frameworks:
        raise ValueError(
            f"model.framework={framework!r} is unsupported; "
            f"choose one of {sorted(allowed_frameworks)}"
        )

    distill_requested = bool(_get(config, "distill", False)) or bool(
        _get(config, "distill_model") or _get(config, "distill_pretrained")
    )
    if distill_requested:
        raise ValueError("distillation is not supported by the maintained Hydra runners")

    for path in (
        "model.embed_dim",
        "model.s1_resolution",
        "model.s2_resolution",
        "model.s1_patch_size",
        "model.s2_patch_size",
        "datamodule.batch_size",
        "train.accum_freq",
        "train.epochs",
    ):
        _positive(config, path)

    patch_mask_ratio = _get(config, "model.patch_mask_ratio", 0.0)
    if not 0 <= patch_mask_ratio < 1:
        raise ValueError(
            f"model.patch_mask_ratio must be in the interval [0, 1); got {patch_mask_ratio!r}"
        )

    val_frac = _get(config, "train.val_frac", 0.0)
    if not 0 <= val_frac < 1:
        raise ValueError(f"train.val_frac must be in the interval [0, 1); got {val_frac!r}")

    matryoshka_enabled = bool(_get(config, "loss.matryoshka_enabled", False))
    if matryoshka_enabled and bool(_get(config, "loss.hyperbolic", False)):
        raise ValueError("loss.matryoshka_enabled and loss.hyperbolic cannot both be true")
    if matryoshka_enabled:
        dims = _get(config, "loss.matryoshka_dims", ())
        weights = _get(config, "loss.matryoshka_weights", ())
        if isinstance(dims, (str, bytes)):
            raise ValueError("loss.matryoshka_dims must contain at least one dimension")
        try:
            dims = list(dims)
        except TypeError as exc:
            raise ValueError(
                "loss.matryoshka_dims must contain at least one dimension"
            ) from exc
        if not dims:
            raise ValueError("loss.matryoshka_dims must contain at least one dimension")
        if any(not isinstance(dim, int) or dim <= 0 for dim in dims):
            raise ValueError("loss.matryoshka_dims must contain only positive integers")
        embed_dim = _get(config, "model.embed_dim")
        if any(dim > embed_dim for dim in dims):
            raise ValueError("loss.matryoshka_dims cannot exceed model.embed_dim")
        if weights and len(weights) != len(dims):
            raise ValueError("loss.matryoshka_weights must match loss.matryoshka_dims")
