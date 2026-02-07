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

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = {"vision": [], "text": []}
    rest_params = {"vision": [], "text": []}
    curvature_params = []

    def _strip_module_prefix(name: str) -> str:
        return name[len("module."):] if name.startswith("module.") else name

    def _is_text_param(name: str) -> bool:
        core = _strip_module_prefix(name)
        return core.startswith("encoder_text") or "encoder_text." in core or core.startswith("text_encoder") or "text_encoder." in core

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if name.endswith("curv"):
            curvature_params.append(param)
            continue
        target = "text" if _is_text_param(name) else "vision"
        if exclude(name, param):
            gain_or_bias_params[target].append(param)
        else:
            rest_params[target].append(param)

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
            {
                "params": curvature_params,
                "weight_decay": 0.0,
                "lr": curvature_lr,
                "lr_scale": curvature_lr / args.train.lr if args.train.lr > 0 else 1.0,
            }
        )

    param_groups = []
    text_lr_scale = 0.1  # text encoder uses 10x smaller LR than S1/S2

    def _add_group(params, weight_decay: float, lr_scale: float) -> None:
        if not params:
            return
        param_groups.append(
            {
                "params": params,
                "weight_decay": weight_decay,
                "lr": args.train.lr * lr_scale,
                "lr_scale": lr_scale,
            }
        )

    _add_group(gain_or_bias_params["vision"], 0.0, 1.0)
    _add_group(rest_params["vision"], args.train.wd, 1.0)
    _add_group(gain_or_bias_params["text"], 0.0, text_lr_scale)
    _add_group(rest_params["text"], args.train.wd, text_lr_scale)

    optimizer = optim.AdamW(
        [*param_groups, *curvature_param_groups],
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        eps=args.train.eps,
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

    # Resume training if a checkpoint is provided.
    resume_path = getattr(args.io, "resume", "")
    if resume_path:
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            unwrap_dataparallel(model).load_state_dict(state_dict, strict=False)
            if "optimizer" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except ValueError as err:
                    logging.warning(
                        "Could not load optimizer state due to mismatch (%s). "
                        "Optimizer will be reinitialized.", err)
            if "scaler" in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            if "loss_state_dict" in checkpoint and hasattr(loss, "load_state_dict"):
                loss.load_state_dict(checkpoint["loss_state_dict"])
            start_epoch = int(checkpoint.get("epoch", start_epoch))
            logging.info("Resumed from checkpoint %s at epoch %s", resume_path, start_epoch)
        else:
            logging.warning("Resume checkpoint %s not found; starting from scratch.", resume_path)

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
