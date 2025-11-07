import json
import os
import time
from configparser import ConfigParser
from types import SimpleNamespace

import numpy as np
import timm
import torch
from open_clip import get_input_dtype
from torchgeo.models import ResNet50_Weights

from ciip.open_clip_train.data import get_data
from ciip.open_clip_train.distributed import is_master
from ciip.open_clip_train.precision import get_autocast
from ciip.open_clip_train.train import evaluate, train_one_epoch

MODEL_NAME = "MoCo"

### band statistics: mean & std
# calculated from 50k data
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]
S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]
S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value)
    # img = np.clip(img, 0, 1).astype(np.float32)
    return img

def extract_embeddings(model, data, args, embedding_output_path):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    # Set eval mode
    model['s1'].eval()
    model['s2'].eval()
    # zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    # metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        for idx in range(len(data)):
            uid, filepath = data.get_sample_uid(idx) # TODO: i think this is correct?
            sample = data[idx]
            s1, s2 = sample
            # Normalize
            # s1 = [normalize(img, mean, std) for img, mean, std in zip(s1, S1_MEAN, S1_STD)]
            s1 = [np.clip(img / 10000, 0, 1) for img in s1]
            # s2 = [normalize(img, mean, std) for img, mean, std in zip(s2, S2C_MEAN, S2C_STD)]
            s2 = [np.clip(img / 10000, 0, 1) for img in s2]
            # To Tensor
            s1 = torch.tensor(s1).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
            s2 = torch.tensor(s2).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)

            with autocast():
                s1_features = model["s1"](s1).cpu()
                s2_features = model["s2"](s2).cpu()

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


def main(args, start_epoch=0):
    # Load Model
    model_s1 = timm.create_model("resnet50", pretrained=False, in_chans=2, num_classes=10)
    model_s2 = timm.create_model("resnet50", pretrained=False, in_chans=13, num_classes=10)
    if MODEL_NAME is not None:
        if MODEL_NAME == "MoCo":
            weights_s1 = ResNet50_Weights.SENTINEL1_ALL_MOCO
            weights_s2 = ResNet50_Weights.SENTINEL2_ALL_MOCO
        elif MODEL_NAME == "DINO":
            weights_s1 = ResNet50_Weights.SENTINEL1_ALL_DECUR
            weights_s2 = ResNet50_Weights.SENTINEL2_ALL_DINO
        model_s1.load_state_dict(weights_s1.get_state_dict(progress=True), strict=False)
        model_s2.load_state_dict(weights_s2.get_state_dict(progress=True), strict=False)
    else:
        print("Continue with Randomly Initialized Weights.")

    # Drop Head
    model_s1 = torch.nn.Sequential(*list(model_s1.children())[:-1])
    model_s2 = torch.nn.Sequential(*list(model_s2.children())[:-1])

    model_s1 = model_s1.to(args.device)
    model_s2 = model_s2.to(args.device)

    model = {'s1': model_s1, 's2': model_s2}
  
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
    val_dataset = data['val'].dataloader.dataset

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")


    dist_model = None 
    tb_writer = None
    # TODO(behzad): alternatively we might need to use what is in the original code:
    # scaler = GradScaler() if args.precision == "amp" else None
    scaler = None 


    embedding_output_path = os.path.join(args.root, 'extracted_embeddings', f'SSL4EO_{MODEL_NAME}_embeddings', 'val')
    os.makedirs(embedding_output_path, exist_ok=True)
    print(f'Saving embeddings to {embedding_output_path}')
    extract_embeddings(model, val_dataset, args, embedding_output_path=embedding_output_path)


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

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config_file', default='ciip/open_clip_train/config_train.ini')
  # For debug mode
  # parser.add_argument('-c', '--config_file', default='/home/zhwa7649/ciip/ciip/open_clip_train/config_train.ini')

  command_line_args = parser.parse_args()

  if os.path.isfile(command_line_args.config_file):
    config = ConfigParser()
    config.read(command_line_args.config_file)

    args = parse_config(config)

    main(args)

  else:
    print('Please provide a valid configuration file.')