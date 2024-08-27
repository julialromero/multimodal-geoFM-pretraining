import logging
import os
from configparser import ConfigParser
from types import SimpleNamespace
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

def main(args, start_epoch=0):
  
  log_base_path = os.path.join(args.log_path, args.name)
  args.log_path = None
  os.makedirs(log_base_path, exist_ok=True)
  log_filename = f'out-{args.name}{time.time()}_.log'
  args.log_path = os.path.join(log_base_path, log_filename)
  print(f'Logging to {args.log_path}')
    # if os.path.exists(args.log_path) and not resume_latest:
    #     print(
    #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
    #     )
    #     return -1 

  args.log_level = logging.DEBUG if args.debug else logging.INFO
  setup_logging(args.log_path, args.log_level)
  # if args.log_path does not exist create it
#   os.makedirs(args.log_path, exist_ok=True)
  # loss = create_loss(args)
  loss = CiipLoss(local_loss=False,
    gather_with_grad=False,
    cache_labels=False,
    rank=0,
    world_size=1,
    use_horovod=False)

  model = CIIP(embed_dim=args.embed_dim,
    s1_resolution=args.s1_resolution,
    s1_layers=args.s1_layers,
    s1_width=args.width,
    s1_patch_size=args.s1_patch_size, # used by transformer 
    s1_bands=len(args.s1_bands),
    s2_resolution=args.s2_resolution,
    s2_layers=args.s2_layers, #Resnet-34
    s2_width=args.width,
    s2_patch_size=args.s2_patch_size, # used by transformer
    s2_bands=len(args.s2_bands))
  
  model = model.to(args.device)
  
  
  exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
  include = lambda n, p: not exclude(n, p)
  named_parameters = list(model.named_parameters())
  gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
  rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
  data = get_data(args)
  total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
  
  optimizer= optim.AdamW(
    [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.wd},
    ],
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
  )

  scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

  dist_model = None 
  tb_writer = None
  # TODO(behzad): alternatively we might need to use what is in the original code:
  # scaler = GradScaler() if args.precision == "amp" else None
  scaler = None 

  # check if checkpoint outdir exists
  os.makedirs(args.checkpoint_path, exist_ok=True)

  for epoch in range(start_epoch, args.epochs):
    
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

def parse_config(config):
    config_dict = {
        'embed_dim': config.getint('model', 'embed_dim'),
        's1_resolution': config.getint('model', 's1_resolution'),
        's1_layers': eval(config.get('model', 's1_layers')),
        'width': config.getint('model', 'width'),
        's1_patch_size': config.getint('model', 's1_patch_size'),
        's1_bands': eval(config.get('model', 's1_bands')),
        's2_resolution': config.getint('model', 's2_resolution'),
        's2_layers': eval(config.get('model', 's2_layers')),
        'width': config.getint('model', 'width'),
        's2_patch_size': config.getint('model', 's2_patch_size'),
        's2_bands': eval(config.get('model', 's2_bands')),
        'lr': config.getfloat('train', 'lr'),
        'wd': config.getfloat('train', 'wd'),
        'beta1': config.getfloat('train', 'beta1'),
        'beta2': config.getfloat('train', 'beta2'),
        'eps': config.getfloat('train', 'eps'),
        'warmup': config.getint('train', 'warmup'),
        'accum_freq': config.getint('train', 'accum_freq'),
        'epochs': config.getint('train', 'epochs'),
        'save_logs': config.getboolean('train', 'save_logs'),
        'name': config.get('train', 'name'),
        'checkpoint_path': config.get('io', 'checkpoint_path'),
        # 'delete_previous_checkpoint': config.getboolean('train', 'delete_previous_checkpoint'),
        # 'save_most_recent': config.getboolean('train', 'save_most_recent'),
        'save_frequency': config.getint('io', 'save_frequency'),
        'batch_size': config.getint('datamodule', 'batch_size'),
        'workers': config.getint('model', 'workers'),
        'precision': config.get('model', 'precision'),
        'dataset_type': config.get('dataset', 'dataset_type'),
        'train_data': config.getboolean('dataset', 'train_data'),
        'debug': config.getboolean('train', 'debug'),
        'root': config.get('dataset', 'root'),
        'distributed': config.getboolean('model', 'distributed'),
        'distill': config.get('model', 'distill'),
        'skip_scheduler': config.getboolean('model', 'skip_scheduler'),
        'delete_previous_checkpoint': config.getboolean('train', 'delete_previous_checkpoint'),
        'save_most_recent': config.getboolean('train', 'save_most_recent'),
        'log_path': config.get('io', 'log_path'),
        'grad_clip_norm': eval(config.get('model', 'grad_clip_norm')),
        'rank': config.getint('datamodule', 'rank'),
        'log_every_n_steps': config.getint('io', 'log_every_n_steps'),
        'world_size': config.getint('datamodule', 'world_size'),
        'wandb': config.getboolean('io', 'wandb'),
        'val_frac': config.getfloat('train', 'val_frac'),
        'use_val': config.getboolean('train', 'use_val'),
        'val_frequency': config.getint('train', 'val_frequency'),
        
    }

    config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = SimpleNamespace(**config_dict)
    return args


if __name__ == "__main__":

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config_file', default='ciip/open_clip_train/config_train.ini')
  command_line_args = parser.parse_args()

  if os.path.isfile(command_line_args.config_file):
    config = ConfigParser()
    config.read(command_line_args.config_file)

    args = parse_config(config)
    main(args)

  else:
    print('Please provide a valid configuration file.')