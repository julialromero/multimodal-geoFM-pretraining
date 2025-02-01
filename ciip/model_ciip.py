from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchgeo.models import resnet50, resnet18

from model import ModifiedResNet#, S1Transformer, S2Transformer
# VisionTransformer



##############################################################
############ START CIIP MODEL IMPLEMENTATION #################
##############################################################


class CIIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # s1
                 s1_resolution: int,
                 s1_layers: Union[Tuple[int, int, int, int], int],
                 s1_width: int,
                 s1_patch_size: int, # transformer parameter
                 s1_bands: int,
                 # s2
                 s2_resolution: int,
                 s2_layers: Union[Tuple[int, int, int, int], int],
                 s2_width: int,
                 s2_patch_size: int,
                 s2_bands: int,
                 framework: str = "modified_resnet"
                #  # text
                #  context_length: int,
                #  vocab_size: int,
                #  transformer_width: int,
                #  transformer_heads: int,
                #  transformer_layers: int
                 ):
        super().__init__()

        # self.context_length = context_length


        # Create s1 encoder model
        if framework == "modified_resnet":
            print("Using Modified ResNet for S1")
            s1_heads = s1_width * 32 // 64
            self.encoder_s1 = ModifiedResNet(
                layers=s1_layers,
                output_dim=embed_dim,
                heads=s1_heads,
                num_bands=s1_bands,
                input_resolution=s1_resolution,
                width=s1_width
            )
        elif framework == "transformer":
            s1_heads = s1_width // 64
            self.encoder_s1 = S1Transformer(
                input_resolution=s1_resolution,
                patch_size=s1_patch_size,
                width=s1_width,
                layers=s1_layers,
                heads=s1_heads,
                output_dim=embed_dim
            )
        elif framework == "resnet18":
            self.encoder_s1 = resnet18(
                    in_chans=s1_bands,
                    num_classes = embed_dim
                    )
        elif framework == "resent50":
            self.encoder_s1 = resnet50(
                in_chans=s1_bands,
                num_classes=embed_dim
            )
        else:
            print("Framework not supported for S1")


        # Same for s2
        if framework == "modified_resnet":
            print("Using ResNet for S2")
            s2_heads = s2_width * 32 // 64
            self.encoder_s2 = ModifiedResNet(  ## adapt this to s2
                layers=s2_layers,
                output_dim=embed_dim,
                heads=s2_heads,
                num_bands=s2_bands,
                input_resolution=s2_resolution,
                width=s2_width
            )
        elif framework == "transformer":
            s2_heads = s2_width // 64
            self.encoder_s2 = S2Transformer(
                input_resolution=s2_resolution,
                patch_size=s2_patch_size,
                width=s2_width,
                layers=s2_layers,
                heads=s2_heads,
                output_dim=embed_dim
            )
        elif framework == "resnet18":
            self.encoder_s2 = resnet18(
                in_chans=s2_bands,
                num_classes=embed_dim
            )
        elif framework == "resnet50":
            self.encoder_s2 = resnet50(
                in_chans=s2_bands,
                num_classes=embed_dim
            )
        else:
            print("Framework not supported for S1")


        ##### Text transformer is removed #####
        # self.transformer = Transformer(
        #     width=transformer_width,
        #     layers=transformer_layers,
        #     heads=transformer_heads,
        #     attn_mask=self.build_attention_mask()
        # )

        # self.vocab_size = vocab_size
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # self.ln_final = LayerNorm(transformer_width)

        # self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        print("Final - S1 Encoder Parameters: ", self.count_parameters_encoder1())
        print("Final - S2 Encoder Parameters: ", self.count_parameters_encoder2())

    def initialize_parameters(self):
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.encoder_s1, ModifiedResNet):
            if self.encoder_s1.attnpool is not None:
                std = self.encoder_s1.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.encoder_s1.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.encoder_s1.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.encoder_s1.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.encoder_s1.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.encoder_s1.layer1, self.encoder_s1.layer2, self.encoder_s1.layer3, self.encoder_s1.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        ## Same For s2 Modality
        if isinstance(self.encoder_s2, ModifiedResNet):
            if self.encoder_s2.attnpool is not None:
                std = self.encoder_s2.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.encoder_s2.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.encoder_s2.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.encoder_s2.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.encoder_s2.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.encoder_s2.layer1, self.encoder_s2.layer2, self.encoder_s2.layer3, self.encoder_s2.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # def build_attention_mask(self):
    #     # lazily create causal attention mask, with full attention between the vision tokens
    #     # pytorch uses additive attention mask; fill with -inf
    #     mask = torch.empty(self.context_length, self.context_length)
    #     mask.fill_(float("-inf"))
    #     mask.triu_(1)  # zero out the lower diagonal
    #     return mask

    @property
    def dtype_s1(self):
        return self.encoder_s1.conv1.weight.dtype
    
    @property
    def dtype_s2(self):
        return self.encoder_s2.conv1.weight.dtype

    def encode_s1(self, s1):
        return self.encoder_s1(s1.type(self.dtype_s1))

    def encode_s2(self, s2):
        return self.encoder_s2(s2.type(self.dtype_s2))
    
    # def encode_text(self, text):
    #     x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

    #     x = x + self.positional_embedding.type(self.dtype)
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x).type(self.dtype)

    #     # x.shape = [batch_size, n_ctx, transformer.width]
    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    #     return x

    def forward(self, s1, s2):
        s1_features = self.encode_s1(s1)
        s2_features = self.encode_s2(s2)

        # normalized features
        s1_features  = s1_features  / s1_features.norm(dim=1, keepdim=True)
        s2_features = s2_features / s2_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_s1 = logit_scale * s1_features @ s2_features.t()   # May have to check that these dimensions align
        logits_per_s2 = logits_per_s1.t()

        # shape = [global_batch_size, global_batch_size]
 
        out_dict = {
            "s1_features": logits_per_s1,
            "s2_features": logits_per_s2,
            "logit_scale": logit_scale
            }

    

        return out_dict
    
    # write definition to print number of parameters in encoder1 
    def count_parameters_encoder1(self):
        return sum(p.numel() for p in self.encoder_s1.parameters() if p.requires_grad)
    
    def count_parameters_encoder2(self):
        return sum(p.numel() for p in self.encoder_s2.parameters() if p.requires_grad)



## TODO: adjust this function
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

## TODO: adjust this function
def build_model(state_dict: dict):
    vit = "s1.proj" in state_dict

    if vit:
        s1_width = state_dict["s1.conv1.weight"].shape[0]
        s1_layers = len([k for k in state_dict.keys() if k.startswith("s1.") and k.endswith(".attn.in_proj_weight")])
        s1_patch_size = state_dict["s1.conv1.weight"].shape[-1]
        grid_size = round((state_dict["s1.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = s1_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"s1.layer{b}"))) for b in [1, 2, 3, 4]]
        s1_layers = tuple(counts)
        s1_width = state_dict["s1.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["s1.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        s1_patch_size = None
        assert output_width ** 2 + 1 == state_dict["s1.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32


    ## Added s2 parameters
    vit = "s2.proj" in state_dict
    if vit:
        s2_width = state_dict["s2.conv1.weight"].shape[0]
        s2_layers = len([k for k in state_dict.keys() if k.startswith("s2.") and k.endswith(".attn.in_proj_weight")])
        s2_patch_size = state_dict["s2.conv1.weight"].shape[-1]
        grid_size = round((state_dict["s2.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = s1_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"s2.layer{b}"))) for b in [1, 2, 3, 4]]
        s2_layers = tuple(counts)
        s2_width = state_dict["s2.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["s2.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        s2_patch_size = None
        assert output_width ** 2 + 1 == state_dict["s2.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    # embed_dim = state_dict["text_projection"].shape[1]
    # context_length = state_dict["positional_embedding"].shape[0]
    # vocab_size = state_dict["token_embedding.weight"].shape[0]
    # transformer_width = state_dict["ln_final.weight"].shape[0]
    # transformer_heads = transformer_width // 64
    # transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # ## TODO: change this line
    # model = CIIP(
    #     embed_dim,
    #     image_resolution, vision_layers, vision_width, vision_patch_size,
    #     context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    # )

    # for key in ["input_resolution"]: # ?
    #     if key in state_dict:
    #         del state_dict[key]

    # convert_weights(model)
    # model.load_state_dict(state_dict)
    # return model.eval()

if __name__ =="__main__":
    
    s1_resolution = 224
    s1_bands = 3

    s2_resolution = 224
    s2_bands = 12

    batch_size = 8
    model = CIIP(
        embed_dim=512,
        s1_resolution=s1_resolution,
        s1_layers=(3, 4, 6, 3), #Resnet-34
        s1_width=512,
        s1_patch_size=16, # used by transformer 
        s1_bands=s1_bands,
        s2_resolution=s2_resolution,
        s2_layers=(3, 4, 6, 3), #Resnet-34
        s2_width=512,
        s2_patch_size=16,
        s2_bands=s2_bands,
    )


    s1 = torch.randn(batch_size, s1_bands, s1_resolution, s1_resolution)
    s2 = torch.randn(batch_size, s2_bands, s2_resolution, s2_resolution)

    # print(model)

    logits_per_s1, logits_per_s2 = model(s1, s2)
    print(logits_per_s1.shape, logits_per_s2.shape)