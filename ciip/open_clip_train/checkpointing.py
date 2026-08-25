"""Shared checkpoint state and local-file operations."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model for DataParallel/DDP wrappers."""
    return model.module if hasattr(model, "module") else model


def build_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.nn.Module,
    *,
    epoch: int,
    name: str,
    scaler=None,
    step: Optional[int] = None,
) -> dict[str, Any]:
    """Build the canonical training checkpoint mapping."""
    checkpoint = {
        "epoch": epoch,
        "name": name,
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_state_dict": loss.state_dict(),
    }
    if step is not None:
        checkpoint["step"] = step
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def restore_training_checkpoint(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss: Optional[torch.nn.Module] = None,
    scaler=None,
    strict: bool = False,
) -> int:
    """Restore available training state and return the next epoch."""
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    unwrap_model(model).load_state_dict(state_dict, strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as error:
            logging.warning("Optimizer state is incompatible and was not restored: %s", error)
    if loss is not None and "loss_state_dict" in checkpoint:
        loss.load_state_dict(checkpoint["loss_state_dict"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0))


def save_checkpoint(checkpoint: dict[str, Any], path: os.PathLike[str] | str) -> None:
    """Save a checkpoint, replacing the destination atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def remove_checkpoint(path: os.PathLike[str] | str) -> None:
    """Remove a checkpoint if it exists."""
    Path(path).unlink(missing_ok=True)
