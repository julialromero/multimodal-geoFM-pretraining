import logging
import os
from typing import Optional

import hydra
import numpy as np
import torch
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torchvision.transforms import v2

from ciip.open_clip_train.checkpointing import (
    build_training_checkpoint,
    remove_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
)
from ciip.open_clip_train.data import get_data
from ciip.open_clip_train.config_validation import validate_training_config
from ciip.open_clip_train.dataparallel.factory import (
    create_model_and_loss,
    unwrap_dataparallel,
)
from ciip.open_clip_train.optimizer import create_optimizer
from ciip.open_clip_train.scheduler import (
    const_lr,
    const_lr_cooldown,
    cosine_lr,
    resolve_warmup_steps,
)
from ciip.open_clip_train.dataparallel.train import train_one_epoch


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
CONF = "prod_default"


def _configure_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "out.log")
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@hydra.main(config_path="../configs", config_name=CONF)
def main(args: DictConfig, start_epoch: int = 0):
    validate_training_config(args, runner="dataparallel")
    device = _select_device()
    args.datamodule.device = str(device)
    args.datamodule.rank = 0
    args.datamodule.world_size = 1
    args.datamodule.distributed = False
    args.datamodule.horovod = False

    run_dir = os.path.join(args.io.logs, args.train.name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    args.io.checkpoint_path = checkpoint_dir

    _configure_logging(run_dir)
    logging.info("Starting data-parallel training run at %s", run_dir)

    _seed_all(args.train.seed)

    # set up comet logging if enabled
    experiment: Optional[Experiment] = None
    if args.io.comet_ml:
        experiment = Experiment(
            api_key=args.comet.api_key,
            project_name=args.comet.project_name,
            workspace=args.comet.workspace,
        )

    transforms = None
    if args.dataset.use_transforms:
        transforms = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=(args.model.s1_resolution, args.model.s2_resolution),
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.GaussianBlur(3),
            ]
        )

    data = get_data(args, transforms)
    steps_per_epoch = data["train"].dataloader.num_batches // args.train.accum_freq
    total_steps = steps_per_epoch * args.train.epochs

    model, loss = create_model_and_loss(args, device)

    recon_cfg = getattr(args, "recon", None)
    if recon_cfg is not None:
        if not getattr(args.model, "patch_masking", False):
            logging.warning(
                "Reconstruction config provided but patch_masking is disabled; recon loss will be inactive."
            )
        logging.info("Reconstruction config: %s", recon_cfg)

    optimizer = create_optimizer(
        model,
        loss,
        args.train,
        getattr(args, "loss", None),
        text_lr_scale=0.1,
    )

    warmup_steps = resolve_warmup_steps(
        args.train.warmup,
        getattr(args.train, "warmup_epochs", None),
        steps_per_epoch,
    )
    scheduler = cosine_lr(optimizer, args.train.lr, warmup_steps, total_steps)
    if args.model.skip_scheduler:
        scheduler = const_lr(optimizer, args.train.lr, warmup_steps, total_steps)
    elif getattr(args.model, "cooldown_steps", 0):
        scheduler = const_lr_cooldown(
            optimizer,
            args.train.lr,
            warmup_steps,
            total_steps,
            getattr(args.model, "cooldown_steps"),
            getattr(args.model, "cooldown_power", 1.0),
            getattr(args.model, "cooldown_end_lr", 0.0),
        )

    scaler = None
    if isinstance(args.model.precision, str) and "amp" in args.model.precision:
        scaler = GradScaler()

    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_path = getattr(args.io, "resume", "")
    if resume_path:
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            start_epoch = restore_training_checkpoint(
                checkpoint, model, optimizer=optimizer, loss=loss, scaler=scaler
            )
            logging.info("Resumed from checkpoint %s at epoch %s", resume_path, start_epoch)
        else:
            logging.warning("Resume checkpoint %s not found; starting from scratch.", resume_path)

    for epoch in range(start_epoch, args.train.epochs):
        logging.info("Start epoch %s", epoch)
        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, None, args, tb_writer=None)
        completed_epoch = epoch + 1

        checkpoint = build_training_checkpoint(
            model,
            optimizer,
            loss,
            epoch=completed_epoch,
            name=args.train.name,
            scaler=scaler,
        )

        should_save = args.io.save_logs and (
            completed_epoch == args.train.epochs
            or (args.io.save_frequency > 0 and completed_epoch % args.io.save_frequency == 0)
            or completed_epoch == 1
        )
        if should_save:
            save_path = os.path.join(checkpoint_dir, f"epoch_{completed_epoch}.pt")
            save_checkpoint(checkpoint, save_path)
            logging.info("Saved checkpoint to %s", save_path)
            if experiment is not None:
                experiment.log_parameters({"epoch": completed_epoch, "name": args.train.name})
                log_model(experiment, model=unwrap_dataparallel(model), model_name="CIIP")

        if args.train.delete_previous_checkpoint:
            remove_checkpoint(
                os.path.join(checkpoint_dir, f"epoch_{completed_epoch - 1}.pt")
            )

        if args.train.save_most_recent:
            save_checkpoint(
                checkpoint, os.path.join(checkpoint_dir, LATEST_CHECKPOINT_NAME)
            )


if __name__ == "__main__":
    main()
