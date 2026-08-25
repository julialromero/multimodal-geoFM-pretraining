import logging
import os
from typing import Tuple

import torch
from omegaconf import DictConfig

from ciip.open_clip_train.dataparallel.model_arch import (
    build_ciip_architecture,
    finalize_model,
    maybe_data_parallel,
    unwrap_dataparallel,
)
from ciip.open_clip_train.checkpointing import restore_training_checkpoint
from ciip.open_clip_train.artifact_io import pt_load
from ciip.open_clip_train.factories import create_loss


def _load_checkpoint_if_available(model: torch.nn.Module, resume_path: str, device: torch.device) -> None:
    if not resume_path:
        return
    if not os.path.exists(resume_path):
        logging.warning("Checkpoint %s does not exist; starting from scratch.", resume_path)
        return

    checkpoint = pt_load(resume_path, map_location=device)
    restore_training_checkpoint(checkpoint, model, strict=False)
    logging.info("Loaded checkpoint weights from %s", resume_path)


def create_model(args: DictConfig, device: torch.device) -> torch.nn.Module:
    model = build_ciip_architecture(
        args.model,
        getattr(args, "loss", None),
        recon_cfg=getattr(args, "recon", None),
    )
    model = finalize_model(model, device, getattr(args.model, "precision", "fp32"))
    _load_checkpoint_if_available(model, getattr(args.io, "resume", ""), device)
    return maybe_data_parallel(model)


def create_model_and_loss(args: DictConfig, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    loss = create_loss(args)
    model = create_model(args, device)
    return model, loss


__all__ = ["create_model", "create_model_and_loss", "unwrap_dataparallel"]
