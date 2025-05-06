import os
from configparser import ConfigParser
from types import SimpleNamespace
import sys
import time
import json
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from train import train_one_epoch, evaluate
from data import get_data

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)
from model_ciip import CIIP
from open_clip import get_input_dtype
from open_clip_train.precision import get_autocast
from open_clip_train.distributed import is_master

from open_clip_train.distributed import is_master, init_distributed_device
from utils import create_model

CONF = "prod_default"
LATEST_CHECKPOINT_NAME = "epoch_100.pt"
CHECKPOINT_EPOCH = '100'
EXPERIMENT_NAME = '2025_03_31-MoCoInit-bs256'
 
def extract_embeddings(model, data, args, embedding_output_path):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.datamodule.device)
    model.eval()

    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    # metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.model.precision)
    input_dtype = get_input_dtype(args.model.precision)

    with torch.inference_mode():
        for idx in range(len(data)):
            uid, filepath = data.get_sample_uid(idx) # TODO: i think this is correct?
            sample = data[idx]
            s1, s2 = sample
            s1 = torch.tensor(s1).unsqueeze(0).to(device=device, dtype=input_dtype, non_blocking=True)
            s2 = torch.tensor(s2).unsqueeze(0).to(device=device, dtype=input_dtype, non_blocking=True)

            with autocast():
                model_out = model.compute_embeddings(s1, s2)
                s1_features = model_out["s1_features"].cpu()
                s2_features = model_out["s2_features"].cpu()

            # read metadata
            # read json at filepath
            fp = os.path.join(os.path.join(filepath), os.listdir(filepath)[0], 'metadata.json')
            metadata = json.load(open(fp))

            # for k in metadata:
            #     print(f'{k}: {metadata[k]}')

            # print('------')
            # # print(metadata['VV'])
            metadata = dict((k, metadata[k]) for k in ['system:bands', 'system:footprint'] if k in metadata)
            embed = {'s1_filepath': filepath, 'uid': uid, 's1': s1_features, 's2': s2_features, 'metadata': metadata}

            # print(metadata)
            # quit()
            torch.save(embed, os.path.join(embedding_output_path, f"{uid}.pt"))
            
            if idx % 10000 == 0:
                print(f"Saved {idx} embeddings")


@hydra.main(config_path="configs", config_name=CONF)
# def main(args):
def main(args: DictConfig, start_epoch=0):
    

    device = init_distributed_device(args.datamodule)
    model = create_model(
        args,
        device=device,
        # jit=args.torchscript,
        # force_quick_gelu=args.force_quick_gelu,
        # force_custom_text=args.force_custom_text,
        # force_patch_dropout=args.force_patch_dropout,
        # force_image_size=args.force_image_size,
        # image_mean=args.image_mean,
        # image_std=args.image_std,
        # image_interpolation=args.image_interpolation,
        # image_resize_mode=args.image_resize_mode,  # only effective for inference
        # aug_cfg=args.aug_cfg,
        # pretrained_image=args.pretrained_image,
        # output_dict=True,
        # **model_kwargs,
    )

    # if args.datamodule.distributed and not args.datamodule.horovod:
    #     if args.use_bn_sync:
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     ddp_args = {}
    #     if args.ddp_static_graph:
    #         # this doesn't exist in older PyTorch, arg only added if enabled
    #         ddp_args['static_graph'] = True
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    #     print(f"Using DDP with {torch.cuda.device_count()} GPUs")
    #     raise NotImplementedError("This code is not yet compatible with DDP")
    
    #     if args.distill:
    #         dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    

    # load checkpoint
    # chkpt_path = '/local/ms-data/SSL4EO/model/2025-03-23/21-12-02/log/2025_03_23-21_12_02-model_resnet50-lr_0.0001-b_128-j_6-p_fp16/checkpoints/epoch_5.pt'
    # chkpt_path = '/local/ms-data/SSL4EO/model/2025-03-18/17-19-02/log/2025_03_18-17_19_03-model_resnet50-lr_0.0001-b_128-j_6-p_fp16/checkpoints/epoch_30.pt'
    chkpt_path = '/local/ms-data/SSL4EO/model/2025_03_31-MoCoInit-bs256/checkpoints/epoch_100.pt'
    checkpoint = torch.load(chkpt_path)


    checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded model from {chkpt_path}")

    model = model.to(args.datamodule.device)


#   exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
#   include = lambda n, p: not exclude(n, p)
#   named_parameters = list(model.named_parameters())
#   gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
#   rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
#   val_dataset = SSL4EODataset(
#         root, # root file path
#         transforms=transforms, # transforms
#         s2_bands=args.s2_bands if hasattr(args, 's2_bands') else default_bands  # from config file
#     )

    data = get_data(args)

    train_dataset = data['train'].dataloader.dataset
    # val_dataset = data['val'].dataloader.dataset

    print(f"Train dataset: {len(train_dataset)} samples")
    # print(f"Val dataset: {len(val_dataset)} samples")


    dist_model = None 
    tb_writer = None
    # TODO(behzad): alternatively we might need to use what is in the original code:
    # scaler = GradScaler() if args.precision == "amp" else None
    scaler = None 


    embedding_output_path = os.path.join(args.dataset.root, 'extracted_embeddings', EXPERIMENT_NAME, f'{CHECKPOINT_EPOCH}', 'val')
    os.makedirs(embedding_output_path, exist_ok=True)
    print(f'Saving embeddings to {embedding_output_path}')
    extract_embeddings(model, train_dataset, args, embedding_output_path=embedding_output_path)


    original_model = model
    
    # Saving embeddings
    # torch.save(
    #     checkpoint_dict,
    #     os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
    # )

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
        'checkpoint_path': config.get('io', 'checkpoint_path'),
        'batch_size': config.getint('datamodule', 'batch_size'),
        'workers': config.getint('model', 'workers'),
        'precision': config.get('model', 'precision'),
        'dataset_type': config.get('dataset', 'dataset_type'),
        'train_data': config.getboolean('dataset', 'train_data'),
        'root': config.get('dataset', 'root'),
        'distributed': config.getboolean('model', 'distributed'),
        'rank': config.getint('datamodule', 'rank'),
        'val_frac': config.getfloat('train', 'val_frac'),
        'use_val': config.getboolean('train', 'use_val'),
        
    }

    config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = SimpleNamespace(**config_dict)
    return args


if __name__ == "__main__":
    main()
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config_file', default='ciip/open_clip_train/config_train.ini')
    # command_line_args = parser.parse_args()

    # if os.path.isfile(command_line_args.config_file):
    #     config = ConfigParser()
    #     config.read(command_line_args.config_file)

    #     args = parse_config(config)

    #     main(args)

    # else:
    #     print('Please provide a valid configuration file.')