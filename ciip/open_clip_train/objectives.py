"""Shared composition of training objectives."""

from typing import Optional

import torch


def scalar_output(value: torch.Tensor) -> torch.Tensor:
    """Reduce replicated scalar model outputs to one value."""
    return value.mean() if value.ndim > 0 else value


def reconstruction_weight(config, *, epoch: int, step: int) -> float:
    """Resolve the configured reconstruction weight after warm-up."""
    if config is None:
        return 0.0
    weight = float(getattr(config, "lambda", 0.0))
    warmup_steps = getattr(config, "warmup_steps", None)
    warmup_epochs = getattr(config, "warmup_epochs", 0)
    if warmup_steps is not None and step < warmup_steps:
        return 0.0
    if warmup_steps is None and epoch < warmup_epochs:
        return 0.0
    return weight


def compose_training_loss(
    loss,
    loss_inputs: dict,
    *,
    reconstruction: Optional[torch.Tensor],
    reconstruction_config,
    epoch: int,
    step: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Evaluate the primary objective and add optional reconstruction."""
    losses = loss(**loss_inputs, output_dict=True)
    total = sum(losses.values())
    if reconstruction is not None:
        scaled = reconstruction * reconstruction_weight(
            reconstruction_config, epoch=epoch, step=step
        )
        total = total + scaled
        losses["recon_loss"] = scaled
        losses["recon_loss_raw"] = reconstruction
    losses["loss"] = total
    return losses, total


def backward_loss(total: torch.Tensor, scaler=None) -> None:
    """Backpropagate a loss with an optional AMP gradient scaler."""
    if scaler is None:
        total.backward()
    else:
        scaler.scale(total).backward()


def run_training_step(
    model,
    model_inputs: tuple[torch.Tensor, ...],
    loss,
    *,
    autocast,
    reconstruction_config,
    epoch: int,
    step: int,
    scaler=None,
    distillation_model=None,
) -> tuple[dict, dict[str, torch.Tensor], torch.Tensor]:
    """Run forward, objective composition, and backward for one batch."""
    with autocast():
        model_output = model(*model_inputs)
        for name in ("logit_scale", "logit_bias"):
            value = model_output.get(name)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                model_output[name] = scalar_output(value)
        if distillation_model is not None:
            with torch.no_grad():
                distilled = distillation_model(*model_inputs[:2])
            model_output.update({f"dist_{name}": value for name, value in distilled.items()})
        losses, total = compose_training_loss(
            loss,
            model_output,
            reconstruction=model_output.get("recon_loss"),
            reconstruction_config=reconstruction_config,
            epoch=epoch,
            step=step,
        )
    backward_loss(total, scaler)
    return model_output, losses, total
