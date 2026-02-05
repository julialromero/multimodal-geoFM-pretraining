import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model


import numpy as np
import torch
from torch import optim

from ciip.model_ciip import CIIP
from ciip.open_clip_train.data import get_data
from ciip.open_clip_train.train import evaluate, train_one_epoch
from ciip.open_clip_train.utils import create_loss

from torchvision.transforms import v2
# from torchvision.transforms import *

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
# Change this to local_default for local testing
# CONF = "local_default"
CONF = "prod_default"

logger = logging.getLogger(__name__)

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        for param_group in optimizer.param_groups:
          param_group["lr"] = lr
        return lr
    return _lr_adjuster

@hydra.main(config_path="configs", config_name=CONF)
def main(args: DictConfig, start_epoch=0):

  print("LOCAL_RANK", os.environ.get("LOCAL_RANK"))
  # Get the Local Rank
  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  args.train.device = "cuda:%d" % local_rank

  loss = create_loss(args)

  pre_projection_dim = getattr(args.model, "pre_projection_dim", args.model.embed_dim)

  model = CIIP(embed_dim=args.model.embed_dim,
    pre_projection_dim=pre_projection_dim,
    s1_resolution=args.model.s1_resolution,
    s1_layers=OmegaConf.to_object(args.model.s1_layers),
    s1_width=args.model.width,
    s1_patch_size=args.model.s1_patch_size, # used by transformer
    s1_bands=len(args.model.s1_bands),
    s2_resolution=args.model.s2_resolution,
    s2_layers=OmegaConf.to_object(args.model.s2_layers), #Resnet-34
    s2_width=args.model.width,
    s2_patch_size=args.model.s2_patch_size, # used by transformer
    s2_bands=len(args.model.s2_bands),
    framework=args.model.framework,
    pretrain=args.model.pretrain.load,
    s1_weights=args.model.pretrain.s1_weights,
    s2_weights=args.model.pretrain.s2_weights,
    patch_masking=getattr(args.model, "patch_masking", False),
    patch_mask_ratio=getattr(args.model, "patch_mask_ratio", 0.0))

  model = model.to(args.train.device)

  #setup comet_ml logging
  if(args.io.comet_ml):
      experiment = Experiment(
          api_key=args.comet.api_key,
          project_name=args.comet.project_name,
          workspace=args.comet.workspace
      )


  exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
  include = lambda n, p: not exclude(n, p)
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

  # define transforms
  #TODO: color jitter only works on 3-channel images right now; a custom function would likely be quite useful here

  if args.dataset.use_transforms:
      transforms = v2.Compose([
          v2.RandomResizedCrop(size=(args.model.s1_resolution, args.model.s2_resolution), antialias=True),
          v2.RandomHorizontalFlip(p=0.5),
          v2.RandomVerticalFlip(p=0.5),
          v2.GaussianBlur(3),
      ])
  else:
      transforms = None

  data = get_data(args, transforms)
  # Testing with a small subset
  total_steps = (data["train"].dataloader.num_batches // args.train.accum_freq) * args.train.epochs

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
          # Use a larger learning rate for curvature based on its initialization.
          curvature_lr = args.train.lr * (1.0 / curvature_init)
      curvature_lr = max(float(curvature_lr), 0.0)
      curvature_param_groups.append({
          "params": curvature_params,
          "weight_decay": 0.,
          "lr": curvature_lr,
      })

  optimizer= optim.AdamW(
    [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.train.wd},
        *curvature_param_groups,
    ],
    lr=args.train.lr,
    betas=(args.train.beta1, args.train.beta2),
    eps=args.train.eps,
  )

  scheduler = cosine_lr(optimizer, args.train.lr, args.train.warmup, total_steps)

  dist_model = None
  tb_writer = None
  # TODO(behzad): alternatively we might need to use what is in the original code:
  # scaler = GradScaler() if args.precision == "amp" else None
  scaler = None

  # check if checkpoint outdir exists
  os.makedirs(args.io.checkpoint_path, exist_ok=True)

  for epoch in range(start_epoch, args.train.epochs):

    logger.info(f'Start epoch {epoch}')

    train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer)
    completed_epoch = epoch + 1

    # print(data['train'].dataloader)
    # print(data['val'].dataloader)
    # print("IF statement", 'val' in data)
    # if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
    #     evaluate(model, data, completed_epoch, args, tb_writer)

    original_model = model

    # Saving checkpoints.
    if args.io.save_logs:
        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.train.name,
            "state_dict": original_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.train.epochs \
                or (args.io.save_frequency > 0 and (completed_epoch % args.io.save_frequency) == 0):
            # save checkpoints within outputs file
            torch.save(
                checkpoint_dict,
                os.path.join(args.io.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )

            # log out via comet
            # TBD if we want the whole checkpoint dict or just some specific hyper-params . . .
            if(args.io.comet_ml):
                # Extract only the scalar items
                experiment.log_parameters({
                    "epoch": checkpoint_dict["epoch"],
                    "name": checkpoint_dict["name"],
                    # Add any other scalar or string fields here...
                })
                log_model(experiment, model=original_model, model_name="CIIP!")

        if args.train.delete_previous_checkpoint:
            previous_checkpoint = os.path.join(args.io.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)

        if args.train.save_most_recent:
            # try not to corrupt the latest checkpoint if save fails
            tmp_save_path = os.path.join(args.io.checkpoint_path, "tmp.pt")
            latest_save_path = os.path.join(args.io.checkpoint_path, LATEST_CHECKPOINT_NAME)
            torch.save(checkpoint_dict, tmp_save_path)
            os.replace(tmp_save_path, latest_save_path)



if __name__ == "__main__":
    main()
