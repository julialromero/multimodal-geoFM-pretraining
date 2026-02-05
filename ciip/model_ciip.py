from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

from torchgeo.models import resnet18, ResNet18_Weights, ResNet50_Weights, resnet50
from .model import ModifiedResNet, VisionTransformer  # , ResNet50 #, S1Transformer, S2Transformer
# VisionTransformer
import logging
# import lorentz as L
from . import lorentz as L
##############################################################
############ START CIIP MODEL IMPLEMENTATION #################
##############################################################

def compute_optimal_orthogonal_mapping(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the orthogonal matrix W that best aligns Y to X (i.e., X ≈ YW).
    Args:
        X: [N, D] tensor of image embeddings
        Y: [N, D] tensor of text embeddings
    Returns:
        W: [D, D] orthogonal matrix
    """

    # Ensure inputs are float tensors and on same device
    X = X.to(dtype=torch.float32)
    Y = Y.to(dtype=torch.float32)

    X = X.T
    Y = Y.T

    # Note, these are already be normalized when input, but maybe it doesnt hurt to keep tihs code?
    # #  normalize to unit norm
    # X = X / X.norm(dim=0, keepdim=True).clamp(min=1e-8)
    # Y = Y / Y.norm(dim=0, keepdim=True).clamp(min=1e-8)

    # Compute cross-covariance matrix
    A = Y @ X.T  # shape: [dim, dim]

    # SVD decomposition
    U, _, Vt = torch.linalg.svd(A)

    # Compute the orthogonal matrix R
    R = U @ Vt

    # Optional: ensure det(R) == 1 to prevent reflection
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    print(f"Orthogonal matrix R shape: {R.shape}, det(R): {torch.linalg.det(R)}")

    return R

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
                 framework: None,
                #  pre_projection_dim: Optional[int] = None,
                #  # text
                #  context_length: int,
                #  vocab_size: int,
                #  transformer_width: int,
                #  transformer_heads: int,
                #  transformer_layers: int
                 pretrain: bool = False,
                 s1_weights: str = "MOCO",
                 s2_weights: str = "MOCO",
                 patch_masking: bool = False,
                 patch_mask_ratio: float = 0.0,
                 init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None,
                 ):
        super().__init__()
        if framework is None:
            raise ValueError("Framework must be specified. Options: 'modified_resnet', 'transformer', 'resnet18', 'resnet50'.")

        # self.context_length = context_length

        self.embed_dim = embed_dim
        # self.pre_projection_dim = pre_projection_dim or embed_dim


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
            self.encoder_s1 = VisionTransformer(
                input_resolution=s1_resolution,
                patch_size=s1_patch_size,
                width=s1_width,
                layers=s1_layers,
                heads=s1_heads,
                output_dim=embed_dim,
                in_channels=s1_bands,
                patch_masking=patch_masking,
                patch_mask_ratio=patch_mask_ratio,
            )
        elif framework == "resnet18":
            self.encoder_s1 = resnet18(
                    in_chans=s1_bands,
                    num_classes=self.embed_dim
                    )
            if pretrain:
                print("Warning: Pretrained weights are not supported for ResNet18 (S1). Ignoring pretrain flag for S1.")
        elif framework == "resnet50":
            if not pretrain:
                self.encoder_s1 = resnet50(
                    in_chans=s1_bands,
                    num_classes=self.embed_dim
                )
                # logging.info("Using ResNet50 for S1 without pretrained weights.")
            else:
                if s1_weights == "MOCO":
                    self.encoder_s1 = resnet50(
                        in_chans=s1_bands,
                        num_classes=self.embed_dim,
                        weights=ResNet50_Weights.SENTINEL1_ALL_MOCO
                    )
                elif s1_weights == "DINO":
                    self.encoder_s1 = resnet50(
                        in_chans=s1_bands,
                        num_classes=self.embed_dim,
                        weights=ResNet50_Weights.SENTINEL1_ALL_DINO
                    )
                else:
                    raise ValueError(f"Unsupported S1 weights: {s1_weights}. Use 'MOCO' or 'DINO'.")
                logging.info(f"Loaded pretrained weights for S1 encoder: {s1_weights}")
            # self.encoder_s1 = ResNet50(
            #     in_chans=s1_bands,
            #     num_classes=embed_dim
            # )
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
            self.encoder_s2 = VisionTransformer(
                input_resolution=s2_resolution,
                patch_size=s2_patch_size,
                width=s2_width,
                layers=s2_layers,
                heads=s2_heads,
                output_dim=embed_dim,
                in_channels=s2_bands,
                patch_masking=patch_masking,
                patch_mask_ratio=patch_mask_ratio,
            )
        elif framework == "resnet18":
            self.encoder_s2 = resnet18(
                in_chans=s2_bands,
                num_classes=self.embed_dim
            )
            if pretrain:
                print("Warning: Pretrained weights are not supported for ResNet18 (S1). Ignoring pretrain flag for S1.")
        elif framework == "resnet50":
            if not pretrain:
                self.encoder_s2 = resnet50(
                    in_chans=s2_bands,
                    num_classes=self.embed_dim
                )
                # logging.info("Using ResNet50 for S2 without pretrained weights.")
            else:
                if s2_weights == "MOCO":
                    self.encoder_s2 = resnet50(
                        in_chans=s2_bands,
                        num_classes=self.embed_dim,
                        weights=ResNet50_Weights.SENTINEL2_ALL_MOCO
                    )
                elif s2_weights == "DINO":
                    self.encoder_s2 = resnet50(
                        in_chans=s2_bands,
                        num_classes=self.embed_dim,
                        weights=ResNet50_Weights.SENTINEL2_ALL_DINO
                    )
                else:
                    raise ValueError(f"Unsupported S2 weights: {s2_weights}. Use 'MOCO' or 'DINO'.")
                logging.info(f"Loaded pretrained weights for S2 encoder: {s2_weights}")
            # self.encoder_s2 = ResNet50(
            #     in_chans=s2_bands,
            #     num_classes=embed_dim
            # )
        else:
            print("Framework not supported for S1")

        # if framework in {"resnet18", "resnet50"} and self.pre_projection_dim != embed_dim:
        #     self.encoder_s1.add_module("proj", nn.Linear(self.pre_projection_dim, embed_dim))
        #     self.encoder_s2.add_module("proj", nn.Linear(self.pre_projection_dim, embed_dim))

        # # Load pretrained weights
        # if pretrain:
        #     if framework != "resnet50":
        #         print("Warning: Pretrained weights only support for ResNet50 framework.")
        #     else:
        #         # Load S1 pretrained weights (only "MOCO" is allowed)
        #         if self.encoder_s1 is not None:
        #             if s1_weights != "MOCO":
        #                 raise ValueError("For ResNet50 S1 encoder, only 'MOCO' pretrained weights are supported.")
        #             else:
        #                 # Load S1 weights using ResNet50_Weights (for example, SENTINEL1_ALL_MOCO)
        #                 weights_obj = ResNet50_Weights.SENTINEL1_ALL_MOCO
        #                 self.encoder_s1.load_state_dict(weights_obj.get_state_dict(progress=True), strict=True)
        #                 logging.info(f"Loaded pretrained weights for S1 encoder: {s1_weights}")
        #         # Load S2 pretrained weights ("MOCO" or "DINO" are allowed)
        #         if self.encoder_s2 is not None:
        #             if s2_weights not in ["MOCO", "DINO"]:
        #                 raise ValueError(
        #                     "For ResNet50 S2 encoder, pretrained weights should be either 'MOCO' or 'DINO'.")
        #             else:
        #                 if s2_weights == "MOCO":
        #                     weights_obj = ResNet50_Weights.SENTINEL2_ALL_MOCO
        #                 else:
        #                     weights_obj = ResNet50_Weights.SENTINEL2_ALL_DINO
        #                 self.encoder_s2.load_state_dict(weights_obj.get_state_dict(progress=True), strict=True)
        #                 logging.info(f"Loaded pretrained weights for S2 encoder: {s2_weights}")


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
        nonscalar_logit_scale = False
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

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

        for encoder in (self.encoder_s1, self.encoder_s2):
            proj_layer = getattr(encoder, "proj", None)
            if isinstance(proj_layer, nn.Linear):
                nn.init.normal_(proj_layer.weight, std=proj_layer.in_features ** -0.5)
                if proj_layer.bias is not None:
                    nn.init.zeros_(proj_layer.bias)

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

  
    def encode_s1(self, s1, normalize, post_head: bool = True, keep_mask: torch.Tensor = None):
        if not post_head and normalize:
            raise ValueError("Cannot normalize before projection head. Set post_head=True when normalize=True.")
        # print("encoding s1 with shape: {}".format(s1.shape))
        if keep_mask is not None and isinstance(self.encoder_s1, VisionTransformer):
            features = self.encoder_s1(s1.type(self.dtype_s1), keep_mask=keep_mask)
        else:
            features = self.encoder_s1(s1.type(self.dtype_s1))
        if post_head:
            proj_layer = getattr(self.encoder_s1, "proj", None)
            if isinstance(proj_layer, nn.Module):
                features = proj_layer(features)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_s2(self, s2, normalize, post_head: bool = True, keep_mask: torch.Tensor = None):
        if not post_head and normalize:
            raise ValueError("Cannot normalize before projection head. Set post_head=True when normalize=True.")
        if keep_mask is not None and isinstance(self.encoder_s2, VisionTransformer):
            features = self.encoder_s2(s2.type(self.dtype_s2), keep_mask=keep_mask)
        else:
            features = self.encoder_s2(s2.type(self.dtype_s2))
        if post_head:
            proj_layer = getattr(self.encoder_s2, "proj", None)
            if isinstance(proj_layer, nn.Module):
                features = proj_layer(features)
        return F.normalize(features, dim=-1) if normalize else features
    
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
        # # returns normalized logits along with the raw encoder outputs
        # s1_features_vc = self.encode_s1(s1, normalize=False)
        # s2_features_vc = self.encode_s2(s2, normalize=False)

        # s1_features = F.normalize(s1_features_vc, dim=-1)
        # s2_features = F.normalize(s2_features_vc, dim=-1)

        keep_s1 = None
        keep_s2 = None
        if (
            self.training
            and isinstance(self.encoder_s1, VisionTransformer)
            and isinstance(self.encoder_s2, VisionTransformer)
            and getattr(self.encoder_s1, "patch_masking", False)
        ):
            keep_s1 = self.encoder_s1.sample_patch_keep_mask(s1.shape[0], s1.device)
            keep_s2 = self.encoder_s2.sample_patch_keep_mask(s2.shape[0], s2.device)

        s1_features = self.encode_s1(s1, normalize=True, keep_mask=keep_s1) # normalize after projection
        s2_features = self.encode_s2(s2, normalize=True, keep_mask=keep_s2)

        # # cosine similarity as logits
        logit_scale = self.logit_scale.exp()


        out_dict = {
            "s1_features": s1_features,
            "s2_features": s2_features,
            # "s1_features_vc": s1_features_vc,
            # "s2_features_vc": s2_features_vc,
            "logit_scale": logit_scale
            }

        if self.logit_bias is not None:
            out_dict['logit_bias'] = self.logit_bias

        return out_dict

    # def get_logits(self, s1, s2, lorentz: bool = False):
    #     s1_features = self.encode_s1(s1, normalize=True)
    #     s2_features = self.encode_s2(s2, normalize=True)
    #     s1_logits = self.logit_scale.exp() * s1_features @ s2_features.T
    #     if self.logit_bias is not None:
    #         s1_logits += self.logit_bias
    #     s2_logits = s1_logits.T
    #     return s1_logits, s2_logits
    

    # def compute_embeddings(self, s1, s2):
    #     s1_features = self.encode_s1(s1, normalize=False)
    #     s2_features = self.encode_s2(s2, normalize=False)

    #     # these should be normalized already
    #     # # # normalized features
    #     # s1_features  = s1_features  / s1_features.norm(dim=1, keepdim=True)
    #     # s2_features = s2_features / s2_features.norm(dim=1, keepdim=True)

    #     out_dict = {
    #         "s1_features": s1_features,
    #         "s2_features": s2_features,
    #         }
    
    #     return out_dict

    # def compute_orthogonal_matrix(self, s1, s2):
    #     self.encoder_s1.compute_orthogonal_matrix = True
    #     self.encoder_s2.compute_orthogonal_matrix = True

        
    #     s1_layer1_features = self.encode_s1(s1, normalize=True)
    #     s2_layer1_features = self.encode_s2(s2, normalize=True)


    #     # Compute centroids BEFORE alignment (mean over batch and spatial dims)
    #     centroid_s1 = s1_layer1_features.mean(dim=0) # shape (num_samples, num_feats) ## #.mean(dim=[0, 2, 3])  # shape: (feat_dim,)
    #     centroid_s2 = s2_layer1_features.mean(dim=0) #.mean(dim=[0, 2, 3])

    #     # normalize the centroids as well (to compare directions)
    #     centroid_s1 = centroid_s1 / centroid_s1.norm()
    #     centroid_s2 = centroid_s2 / centroid_s2.norm()
  
    #     l2_before = torch.norm(centroid_s1 - centroid_s2, p=2).item()
    #     cos_before = F.cosine_similarity(centroid_s1.unsqueeze(0), centroid_s2.unsqueeze(0)).item()
    #     logging.info(f"--- Before Orthogonal Transformation ---")
    #     logging.info(f"L2 norm between centroids: {l2_before:.6f}")
    #     logging.info(f"Cosine similarity between centroids: {cos_before:.6f}")


    #     W = compute_optimal_orthogonal_mapping(s2_layer1_features, s1_layer1_features)
    #     W = W.to(device=s1.device, dtype=torch.float32, non_blocking=True)

    #     s1_layer1_features = s1_layer1_features.to(dtype=torch.float32)
    #     s1_aligned = s1_layer1_features @ W
    

    #     # Normalize the aligned features
    #     s1_aligned = s1_aligned / s1_aligned.norm(dim=1, keepdim=True)

    #     # Compute centroids AFTER alignment
    #     centroid_s1_aligned = s1_aligned.mean(dim=0) #.mean(dim=[0, 2, 3])

    #     l2_after = torch.norm(centroid_s1_aligned - centroid_s2, p=2).item()
    #     cos_after = F.cosine_similarity(centroid_s1_aligned.unsqueeze(0), centroid_s2.unsqueeze(0)).item()

    #     logging.info(f"--- After Orthogonal Transformation ---")
    #     logging.info(f"L2 norm between centroids: {l2_after:.6f}")
    #     logging.info(f"Cosine similarity between centroids: {cos_after:.6f}")


    #     self.encoder_s1.compute_orthogonal_matrix = False
    #     self.encoder_s2.compute_orthogonal_matrix = False
    #     self.encoder_s1.apply_orthogonal_matrix = True
    #     # self.encoder_s1.W = W
    #     self.encoder_s1.register_buffer("W", W)
        


    #     # create dictionary to return
    #     out_dict = {
    #     "l2_before": float(l2_before),
    #     "cos_before": float(cos_before),
    #     "l2_after": float(l2_after),
    #     "cos_after": float(cos_after)
    #     }

    #     return W.detach().cpu(), out_dict



    # write definition to print number of parameters in encoder1 
    def count_parameters_encoder1(self):
        return sum(p.numel() for p in self.encoder_s1.parameters() if p.requires_grad)
    
    def count_parameters_encoder2(self):
        return sum(p.numel() for p in self.encoder_s2.parameters() if p.requires_grad)


class LorentzCIIP(CIIP):
    def __init__(self, 
        # Base CIIP parameters first
        embed_dim: int,
        s1_resolution: int,
        s1_layers: Union[Tuple[int, int, int, int], int],
        s1_width: int,
        s1_patch_size: int,
        s1_bands: int,
        s2_resolution: int,
        s2_layers: Union[Tuple[int, int, int, int], int],
        s2_width: int,
        s2_patch_size: int,
        s2_bands: int,
        framework: None,
        # Lorentz-specific parameters
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        **kwargs):
        
        # Remove Lorentz-specific params before passing to parent
        super().__init__(
            embed_dim=embed_dim,
            s1_resolution=s1_resolution,
            s1_layers=s1_layers,
            s1_width=s1_width,
            s1_patch_size=s1_patch_size,
            s1_bands=s1_bands,
            s2_resolution=s2_resolution,
            s2_layers=s2_layers,
            s2_width=s2_width,
            s2_patch_size=s2_patch_size,
            s2_bands=s2_bands,
            framework=framework,
            **kwargs
        )

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 5),
            "min": math.log(curv_init / 5),
        }
        self.entail_weight = entail_weight

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.s2_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.s1_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())


    def encode_s2(
        self,
        s2: torch.Tensor,
        lorentz: bool,
        normalize: bool = False,
        post_head: bool = True,
        keep_mask: torch.Tensor = None,
    ):
        """
        Args:
            project: Lift features from the encoder onto the Hyperboloid.

        """
        # if normalize:
        #     print("Warning: normalize=True is ignored in LorentzCIIP.encode_s2")
        if post_head is False and lorentz:
            raise ValueError("Cannot project to Lorentzian space before projection head. Set post_head=True when lorentz=True.")

        # Get Euclidean features from the encoder (without L2 normalization).
        s2_feats = super().encode_s2(
            s2,
            normalize=False,
            post_head=post_head,
            keep_mask=keep_mask,
        ) # get post-head un-norm euclidean feats

        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if lorentz:
            s2_feats = s2_feats * self.s2_alpha.exp()
            with torch.autocast(s2_feats.device.type, dtype=torch.float32):
                s2_feats = L.exp_map0(s2_feats, self.curv.exp())

        return s2_feats

    def encode_s1(
        self,
        s1: list[torch.Tensor],
        lorentz: bool,
        normalize: bool = False,
        post_head: bool = True,
        keep_mask: torch.Tensor = None,
    ):
        # if normalize:
        #     print("Warning: normalize=True is ignored in LorentzCIIP.encode_s2")
        if post_head is False and lorentz:
            raise ValueError("Cannot project to Lorentzian space before projection head. Set post_head=True when lorentz=True.")
            
        # Get Euclidean features from the encoder (without L2 normalization).

        s1_feats = super().encode_s1(
            s1,
            normalize=False,
            post_head=post_head,
            keep_mask=keep_mask,
        )

        if lorentz:
            s1_feats = s1_feats * self.s1_alpha.exp()
            with torch.autocast(s1_feats.device.type, dtype=torch.float32):
                s1_feats = L.exp_map0(s1_feats, self.curv.exp())

        return s1_feats


    def einstein_loss(self, features, dist, curv):
   #     print(features.shape)
        feature_norm = torch.sum(features ** 2, dim=-1, keepdim= True)
        features_time = torch.sqrt(1 / curv + feature_norm)
        klein_features = features/features_time
        lorenz_factors = 1/torch.sqrt(1+ curv * feature_norm**2)
        klein_average = torch.sum(features * lorenz_factors, dim = 0, keepdim= True)/torch.sum(lorenz_factors, dim = 0, keepdim= True)
       # print(klein_average.shape)

        klein_norm = torch.sum(klein_average ** 2, dim=-1, keepdim= True)
        avg_time = 1/torch.sqrt(1 + curv * klein_norm)

        avg_lorenz = klein_average/torch.sqrt(1 + curv * klein_norm)

        geo_dist = L.pairwise_dist(avg_lorenz, torch.zeros_like(avg_lorenz))
        
        return  geo_dist # torch.abs(geo_dist - dist) 
    


    def forward(
        self, s1: torch.Tensor, s2: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        # _curv = _curv.unsqueeze(0)
        # # assert that there is a batch dim
        # assert _curv.dim() == 1

        self.s2_alpha.data = torch.clamp(self.s2_alpha.data, max=0.0)
        self.s1_alpha.data = torch.clamp(self.s1_alpha.data, max=0.0)

        keep_s1 = None
        keep_s2 = None
        if (
            self.training
            and isinstance(self.encoder_s1, VisionTransformer)
            and isinstance(self.encoder_s2, VisionTransformer)
            and getattr(self.encoder_s1, "patch_masking", False)
        ):
            keep_s1 = self.encoder_s1.sample_patch_keep_mask(s1.shape[0], s1.device)
            keep_s2 = self.encoder_s2.sample_patch_keep_mask(s2.shape[0], s2.device)

        # shape: (batch_size, embed_dim)
        s2_feats = self.encode_s2(s2, lorentz=True, keep_mask=keep_s2)
        s1_feats = self.encode_s1(s1, lorentz=True, keep_mask=keep_s1)

        logit_scale = self.logit_scale.exp()

        # Return dictionary similar to base CIIP class
        out_dict = {
            "s1_features": s1_feats,
            "s2_features": s2_feats,
            "logit_scale": logit_scale,
            "curv": _curv,  # Include curvature for loss computation
        }

        if self.logit_bias is not None:
            out_dict['logit_bias'] = self.logit_bias

        return out_dict


        ## ATMG seems like they compute the contrsative loss in forward here, but do they use einstein loss at all? 
        # all_s2_feats = dist.gather_across_processes(s2_feats)
        # all_s1_feats = dist.gather_across_processes(s1_feats)

        # all_s2_feats = torch.cat(all_s2_feats, dim=0)
        # all_s1_feats = torch.cat(all_s1_feats, dim=0)
        # with torch.autocast(self.device.type, dtype=torch.float32):
        #     # Compute logits for hyperbolic angle based contrastive loss.
        #     text_logits = -L.pairwise_oxy_angle(s1_feats, all_s2_feats, _curv)
        #     image_logits = L.pairwise_oxy_angle(s2_feats, all_s1_feats, _curv)
        #     batch_size = s2_feats.shape[0]
        #     targets = torch.arange(batch_size, device=text_logits.device)
        #     targets = targets + batch_size * self._rank

        #     # Clamp temperature such that logits are not scaled more than 100x.
        #     # ln(100) = ~4.6052
        #     self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        #     _scale = self.logit_scale.exp()

        #     contrastive_loss =  0.5*(
        #         nn.functional.cross_entropy(_scale * image_logits, targets) + 
        #         nn.functional.cross_entropy(_scale * text_logits, targets)
        #     )

        #   #  loss = contrastive_loss
        # return {
        #     "loss": contrastive_loss,
        #     "logging": {
        #         "contrastive_loss": contrastive_loss,
        #         "logit_scale": _scale,
        #         "curv": _curv,
        #     },
        # }




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
                if attr is None:
                    continue
                if isinstance(attr, nn.Module):
                    attr.half()
                elif torch.is_tensor(attr):
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

# ## TODO: adjust this function
# def build_model(state_dict: dict):
#     vit = "s1.proj" in state_dict

#     if vit:
#         s1_width = state_dict["s1.conv1.weight"].shape[0]
#         s1_layers = len([k for k in state_dict.keys() if k.startswith("s1.") and k.endswith(".attn.in_proj_weight")])
#         s1_patch_size = state_dict["s1.conv1.weight"].shape[-1]
#         grid_size = round((state_dict["s1.positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = s1_patch_size * grid_size
#     else:
#         counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"s1.layer{b}"))) for b in [1, 2, 3, 4]]
#         s1_layers = tuple(counts)
#         s1_width = state_dict["s1.layer1.0.conv1.weight"].shape[0]
#         output_width = round((state_dict["s1.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
#         s1_patch_size = None
#         assert output_width ** 2 + 1 == state_dict["s1.attnpool.positional_embedding"].shape[0]
#         image_resolution = output_width * 32


#     ## Added s2 parameters
#     vit = "s2.proj" in state_dict
#     if vit:
#         s2_width = state_dict["s2.conv1.weight"].shape[0]
#         s2_layers = len([k for k in state_dict.keys() if k.startswith("s2.") and k.endswith(".attn.in_proj_weight")])
#         s2_patch_size = state_dict["s2.conv1.weight"].shape[-1]
#         grid_size = round((state_dict["s2.positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = s1_patch_size * grid_size
#     else:
#         counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"s2.layer{b}"))) for b in [1, 2, 3, 4]]
#         s2_layers = tuple(counts)
#         s2_width = state_dict["s2.layer1.0.conv1.weight"].shape[0]
#         output_width = round((state_dict["s2.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
#         s2_patch_size = None
#         assert output_width ** 2 + 1 == state_dict["s2.attnpool.positional_embedding"].shape[0]
#         image_resolution = output_width * 32

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
