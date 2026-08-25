"""Shared optimizer construction for maintained training runners."""

from typing import Optional

import torch


def _is_text_parameter(name: str) -> bool:
    name = name.removeprefix("module.")
    return (
        name.startswith("encoder_text")
        or "encoder_text." in name
        or name.startswith("text_encoder")
        or "text_encoder." in name
    )


def create_optimizer(
    model: torch.nn.Module,
    loss: torch.nn.Module,
    train_config,
    loss_config=None,
    *,
    text_lr_scale: Optional[float] = None,
) -> torch.optim.AdamW:
    """Create AdamW groups for decay, text, curvature, and loss parameters."""
    groups = {
        "vision_decay": [],
        "vision_no_decay": [],
        "text_decay": [],
        "text_no_decay": [],
    }
    curvature = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("curv"):
            curvature.append(parameter)
            continue
        prefix = "text" if _is_text_parameter(name) else "vision"
        no_decay = parameter.ndim < 2 or any(
            key in name for key in ("bn", "ln", "bias", "logit_scale")
        )
        groups[f"{prefix}_{'no_decay' if no_decay else 'decay'}"].append(parameter)

    parameter_groups = []

    def add(parameters, *, weight_decay: float, lr_scale: float = 1.0) -> None:
        if parameters:
            parameter_groups.append(
                {
                    "params": parameters,
                    "weight_decay": weight_decay,
                    "lr": train_config.lr * lr_scale,
                    "lr_scale": lr_scale,
                }
            )

    add(groups["vision_no_decay"], weight_decay=0.0)
    add(groups["vision_decay"], weight_decay=train_config.wd)
    effective_text_scale = 1.0 if text_lr_scale is None else text_lr_scale
    add(groups["text_no_decay"], weight_decay=0.0, lr_scale=effective_text_scale)
    add(groups["text_decay"], weight_decay=train_config.wd, lr_scale=effective_text_scale)

    loss_parameters = [parameter for parameter in loss.parameters() if parameter.requires_grad]
    add(loss_parameters, weight_decay=0.0)

    if curvature:
        curvature_init = getattr(loss_config, "curvature_init", 1.0) if loss_config else 1.0
        curvature_init = max(float(curvature_init), 1e-6)
        curvature_lr = getattr(train_config, "curvature_lr", None)
        if curvature_lr is None:
            curvature_lr = train_config.lr / curvature_init
        curvature_lr = max(float(curvature_lr), 0.0)
        curvature_scale = curvature_lr / train_config.lr if train_config.lr > 0 else 1.0
        add(curvature, weight_decay=0.0, lr_scale=curvature_scale)

    return torch.optim.AdamW(
        parameter_groups,
        lr=train_config.lr,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.eps,
    )


def step_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scaler=None,
    grad_clip_norm: Optional[float] = None,
    horovod: bool = False,
) -> None:
    """Apply optional clipping, take an optimizer step, and update AMP state."""
    if scaler is None:
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
        optimizer.step()
        return

    if horovod:
        optimizer.synchronize()
    scaler.unscale_(optimizer)
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
    if horovod:
        with optimizer.skip_synchronize():
            scaler.step(optimizer)
    else:
        scaler.step(optimizer)
    scaler.update()
