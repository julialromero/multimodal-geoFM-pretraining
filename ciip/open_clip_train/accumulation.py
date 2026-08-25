"""Feature-cache helpers for gradient accumulation."""

import torch

from ciip.open_clip_train.objectives import scalar_output


_UNCACHED_KEYS = ("logit_scale", "logit_bias")


def cache_model_features(cache: dict[str, list[torch.Tensor]], model_output: dict) -> None:
    """Append non-scalar model features used as cross-microbatch negatives."""
    for key, value in model_output.items():
        if key in _UNCACHED_KEYS or value.ndim == 0:
            continue
        cache.setdefault(key, []).append(value)


def build_accumulated_loss_inputs(
    cached_features: dict[str, list[torch.Tensor]],
    model_output: dict,
    microbatch: int,
) -> dict:
    """Combine one tracked forward pass with detached cached negatives."""
    output = dict(model_output)
    loss_inputs = {}
    for key in _UNCACHED_KEYS:
        value = output.pop(key, None)
        if value is not None:
            loss_inputs[key] = scalar_output(value)

    for key, cached in cached_features.items():
        values = list(cached)
        values[microbatch] = output[key]
        loss_inputs[key] = torch.cat(values)

    for key, value in output.items():
        loss_inputs.setdefault(key, value)
    return loss_inputs
