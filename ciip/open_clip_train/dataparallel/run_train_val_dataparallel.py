import logging
import os
from typing import Optional

import hydra
import numpy as np
import torch
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from omegaconf import DictConfig
from torch import optim
from torch.cuda.amp import GradScaler
from torchvision.transforms import v2

from ciip.open_clip_train.data import get_data
from ciip.open_clip_train.dataparallel.factory import (
    create_model_and_loss,
    unwrap_dataparallel,
)
from ciip.open_clip_train.scheduler import const_lr, const_lr_cooldown, cosine_lr
from ciip.open_clip_train.train import train_one_epoch


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
CONF = "prod_default"


def _configure_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "out.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@hydra.main(config_path="../configs", config_name=CONF)
def main(args: DictConfig, start_epoch: int = 0):
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
    total_steps = (data["train"].dataloader.num_batches // args.train.accum_freq) * args.train.epochs

    model, loss = create_model_and_loss(args, device)

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = []
    rest_params = []
    curvature_params = []

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if name.endswith("curv"):
            curvature_params.append(param)
            continue
        if exclude(name, param):
            gain_or_bias_params.append(param)
        else:
            rest_params.append(param)

    curvature_param_groups = []
    if curvature_params:
        loss_cfg = getattr(args, "loss", None)
        curvature_init = getattr(loss_cfg, "curvature_init", None) if loss_cfg is not None else None
        if curvature_init is None and loss_cfg is not None:
            curvature_init = getattr(loss_cfg, "hyperbolic_curvature_init", None)
        if curvature_init is None:
            curvature_init = 1.0
        curvature_init = max(float(curvature_init), 1e-6)
        curvature_lr = getattr(args.train, "curvature_lr", None)
        if curvature_lr is None and loss_cfg is not None:
            curvature_lr = getattr(loss_cfg, "curvature_lr", None)
        if curvature_lr is None:
            curvature_lr = args.train.lr * (1.0 / curvature_init)
        curvature_lr = max(float(curvature_lr), 0.0)
        curvature_param_groups.append(
            {"params": curvature_params, "weight_decay": 0.0, "lr": curvature_lr}
        )

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.train.wd},
            *curvature_param_groups,
        ],
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        eps=args.train.eps,
    )

    warmup = max(args.train.warmup, 0)
    scheduler = cosine_lr(optimizer, args.train.lr, warmup, total_steps)
    if args.model.skip_scheduler:
        scheduler = const_lr(optimizer, args.train.lr, warmup, total_steps)
    elif getattr(args.model, "cooldown_steps", 0):
        scheduler = const_lr_cooldown(
            optimizer,
            args.train.lr,
            warmup,
            total_steps,
            getattr(args.model, "cooldown_steps"),
            getattr(args.model, "cooldown_power", 1.0),
            getattr(args.model, "cooldown_end_lr", 0.0),
        )

    scaler = None
    if isinstance(args.model.precision, str) and "amp" in args.model.precision:
        scaler = GradScaler()

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.train.epochs):
        logging.info("Start epoch %s", epoch)
        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, None, args, tb_writer=None)
        completed_epoch = epoch + 1

        checkpoint = {
            "epoch": completed_epoch,
            "name": args.train.name,
            "state_dict": unwrap_dataparallel(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_state_dict": loss.state_dict(),
        }
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()

        should_save = args.io.save_logs and (
            completed_epoch == args.train.epochs
            or (args.io.save_frequency > 0 and completed_epoch % args.io.save_frequency == 0)
            or completed_epoch == 1
        )
        if should_save:
            save_path = os.path.join(checkpoint_dir, f"epoch_{completed_epoch}.pt")
            torch.save(checkpoint, save_path)
            logging.info("Saved checkpoint to %s", save_path)
            if experiment is not None:
                experiment.log_parameters({"epoch": completed_epoch, "name": args.train.name})
                log_model(experiment, model=unwrap_dataparallel(model), model_name="CIIP")

        if args.train.delete_previous_checkpoint:
            prev_path = os.path.join(checkpoint_dir, f"epoch_{completed_epoch - 1}.pt")
            if os.path.exists(prev_path):
                os.remove(prev_path)

        if args.train.save_most_recent:
            tmp_save_path = os.path.join(checkpoint_dir, "tmp.pt")
            latest_save_path = os.path.join(checkpoint_dir, LATEST_CHECKPOINT_NAME)
            torch.save(checkpoint, tmp_save_path)
            os.replace(tmp_save_path, latest_save_path)


if __name__ == "__main__":
    main()
