import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

from train import train_one_epoch, evaluate
from data import get_data
from ..model_ciip import CIIP
from ..loss import CiipLoss

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

def main(args, start_epoch=0):
  # loss = create_loss(args)
  loss = CiipLoss(local_loss=False,
    gather_with_grad=False,
    cache_labels=False,
    rank=0,
    world_size=1,
    use_horovod=False)

  model = CIIP(embed_dim=args.embed_dim,
    s1_resolution=args.s1_resolution,
    s1_layers=args.resnet_layers,
    s1_width=args.width,
    s1_patch_size=args.s1_patch_size, # used by transformer 
    s1_bands=args.s1_bands,
    s2_resolution=args.s2_resolution,
    s2_layers=args.resnet_layers, #Resnet-34
    s2_width=args.width,
    s2_patch_size=args.s2_patch_size, # used by transformer
    s2_bands=args.s2_bands)
  

  data = get_data(args, epoch=start_epoch)
  total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
  scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
  dist_model = None 
  tb_writer = None
  # TODO(behzad): alternatively we might need to use what is in the original code:
  # scaler = GradScaler() if args.precision == "amp" else None
  scaler = None 

  
  exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
  include = lambda n, p: not exclude(n, p)
  named_parameters = list(model.named_parameters())
  gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
  rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

  optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

  for epoch in range(start_epoch, args.epochs):
    
    logging.info(f'Start epoch {epoch}')

    train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer)
    completed_epoch = epoch + 1

    if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        evaluate(model, data, completed_epoch, args, tb_writer)

    original_model = model
    
    # Saving checkpoints.
    if args.save_logs:
        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.name,
            "state_dict": original_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )
        if args.delete_previous_checkpoint:
            previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)

        if args.save_most_recent:
            # try not to corrupt the latest checkpoint if save fails
            tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
            latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
            torch.save(checkpoint_dict, tmp_save_path)
            os.replace(tmp_save_path, latest_save_path)