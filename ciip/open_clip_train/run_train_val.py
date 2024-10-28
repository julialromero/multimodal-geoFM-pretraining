import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import time

import numpy as np
import torch
from torch import optim

from train import train_one_epoch, evaluate
from data import get_data
from logger import setup_logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)
from model_ciip import CIIP
from loss import CiipLoss

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

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

@hydra.main(config_path="configs", config_name="local_default")
def main(args: DictConfig, start_epoch=0):
  
  log_base_path = os.path.join(args.io.log_path, args.train.name)
  args.io.log_path = None
  os.makedirs(log_base_path, exist_ok=True)
  log_filename = f'out-{args.train.name}{time.time()}_.log'
  args.io.log_path = os.path.join(log_base_path, log_filename)
  print(f'Logging to {args.io.log_path}')
    # if os.path.exists(args.log_path) and not resume_latest:
    #     print(
    #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
    #     )
    #     return -1 

  # args.io.log_level = logging.DEBUG if args.train.debug else logging.INFO
  # setup_logging(args.io.log_path, args.io.log_level)
  # if args.log_path does not exist create it
#   os.makedirs(args.log_path, exist_ok=True)
  # loss = create_loss(args)
  loss = CiipLoss(local_loss=False,
    gather_with_grad=False,
    cache_labels=False,
    rank=0,
    world_size=1,
    use_horovod=False)

  model = CIIP(embed_dim=args.model.embed_dim,
    s1_resolution=args.model.s1_resolution,
    s1_layers=OmegaConf.to_object(args.model.s1_layers),
    s1_width=args.model.width,
    s1_patch_size=args.model.s1_patch_size, # used by transformer
    s1_bands=len(args.model.s1_bands),
    s2_resolution=args.model.s2_resolution,
    s2_layers=OmegaConf.to_object(args.model.s2_layers), #Resnet-34
    s2_width=args.model.width,
    s2_patch_size=args.model.s2_patch_size, # used by transformer
    s2_bands=len(args.model.s2_bands))
  
  model = model.to(args.train.device)
  
  
  exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
  include = lambda n, p: not exclude(n, p)
  named_parameters = list(model.named_parameters())
  gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
  rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
  data = get_data(args)
  total_steps = (data["train"].dataloader.num_batches // args.train.accum_freq) * args.train.epochs
  
  optimizer= optim.AdamW(
    [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.train.wd},
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
    
    logging.info(f'Start epoch {epoch}')

    train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer)
    completed_epoch = epoch + 1

    # print(data['train'].dataloader)
    # print(data['val'].dataloader)
    # print("IF statement", 'val' in data)
    if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        evaluate(model, data, completed_epoch, args, tb_writer)

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

        if completed_epoch == args.train.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(args.io.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )
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
