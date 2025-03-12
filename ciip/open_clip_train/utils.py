import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)
from model_ciip import CIIP
from loss import CiipLoss
from torch import nn
from omegaconf import DictConfig, OmegaConf


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype

def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        else:
            raise NotImplementedError(f"Conversion of {type(l)} weights to {dtype} not supported")

        # if isinstance(l, (nn.MultiheadAttention, Attention)):
        #     for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
        #         tensor = getattr(l, attr)
        #         if tensor is not None:
        #             tensor.data = tensor.data.to(dtype)

        # if isinstance(l, (CLIP, TextTransformer)):
        #     # convert text nn.Parameter projections
        #     attr = getattr(l, "text_projection", None)
        #     if attr is not None:
        #         attr.data = attr.data.to(dtype)

        # if isinstance(l, VisionTransformer):
        #     # convert vision nn.Parameter projections
        #     attr = getattr(l, "proj", None)
        #     if attr is not None:
        #         attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)

convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat



def create_loss(args):
    if args.distill:
        raise NotImplementedError("DistillClipLoss not currently supported")
        # return DistillClipLoss(
        #     local_loss=args.local_loss,
        #     gather_with_grad=args.gather_with_grad,
        #     cache_labels=True,
        #     rank=args.rank,
        #     world_size=args.world_size,
        #     use_horovod=args.horovod,
        # )
    # elif "coca" in args.model.lower():
    #     raise NotImplementedError("CoCa not currently supported")
        # return CoCaLoss(
        #     caption_loss_weight=args.coca_caption_loss_weight,
        #     clip_loss_weight=args.coca_contrastive_loss_weight,
        #     local_loss=args.local_loss,
        #     gather_with_grad=args.gather_with_grad,
        #     cache_labels=True,
        #     rank=args.rank,
        #     world_size=args.world_size,
        #     use_horovod=args.horovod,
        # )
    elif args.siglip:
        raise NotImplementedError("SigLip not currently supported")
        # assert not args.horovod, "Horovod not currently supported for SigLip"
        # return SigLipLoss(
        #     rank=args.rank,
        #     world_size=args.world_size,
        # )
    return CiipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


def create_model(args, device):

# (
#         model_name: str,
#         pretrained: Optional[str] = None,
#         precision: str = 'fp32',
#         device: Union[str, torch.device] = 'cpu',
#         jit: bool = False,
#         force_quick_gelu: bool = False,
#         # force_custom_text: bool = False,
#         force_patch_dropout: Optional[float] = None,
#         force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
#         force_preprocess_cfg: Optional[Dict[str, Any]] = None,
#         pretrained_image: bool = False,
#         pretrained_hf: bool = True,
#         cache_dir: Optional[str] = None,
#         output_dict: Optional[bool] = None,
#         require_pretrained: bool = False,
#         **model_kwargs,
# ):
    model_name = "CIIP"
    precision = args.model.precision
    pretrained = args.model.pretrain.load

    # force_preprocess_cfg = force_preprocess_cfg or {}
    # preprocess_cfg = asdict(PreprocessCfg())
    # has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    # if has_hf_hub_prefix:
    #     model_id = model_name[len(HF_HUB_PREFIX):]
    #     checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
    #     config = _get_hf_config(model_id, cache_dir)
    #     preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
    #     model_cfg = config['model_cfg']
    #     pretrained_hf = False  # override, no need to load original HF text weights
    # else:
    # model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    checkpoint_path = args.io.checkpoint_path
    # model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)


    # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
    cast_dtype = get_cast_dtype(precision)

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
        s2_bands=len(args.model.s2_bands),
        framework=args.model.framework,
        pretrain=args.model.pretrain.load,
        s1_weights=args.model.pretrain.s1_weights,
        s2_weights=args.model.pretrain.s2_weights)
    # , 
        # cast_dtype=cast_dtype)

    # if precision in ("fp16", "bf16"):
    #     dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        
    #     model.to(device=device)
    #     convert_weights_to_lp(model, dtype=dtype)
    # elif precision in ("pure_fp16", "pure_bf16"):
    #     dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
    #     model.to(device=device, dtype=dtype)
    # else:
    #     model.to(device=device)
    model.to(device=device)

        # pretrained_loaded = False
        # if pretrained:
        #     checkpoint_path = ''
        #     pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        #     if pretrained_cfg:
        #         checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
        #         preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
        #     elif os.path.exists(pretrained):
        #         checkpoint_path = pretrained

        #     if checkpoint_path:
        #         logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
        #         load_checkpoint(model, checkpoint_path)
        #     else:
        #         error_str = (
        #             f'Pretrained weights ({pretrained}) not found for model {model_name}.'
        #             f' Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
        #         logging.warning(error_str)
        #         raise RuntimeError(error_str)
        #     pretrained_loaded = True
        # elif has_hf_hub_prefix:
        #     logging.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
        #     load_checkpoint(model, checkpoint_path)
        #     pretrained_loaded = True

        # if require_pretrained and not pretrained_loaded:
        #     # callers of create_model_from_pretrained always expect pretrained weights
        #     raise RuntimeError(
        #         f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    # if output_dict and hasattr(model, "output_dict"):
    #     model.output_dict = True

    # if jit:
    #     model = torch.jit.script(model)

    

    return model