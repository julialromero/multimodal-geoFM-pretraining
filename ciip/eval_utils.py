import torch
import torch.nn as nn
from collections import OrderedDict
from model_ciip import CIIP  # Make sure this is defined correctly  
import torch.nn.functional as F


class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample



def create_ciip_model(embed_dim=512, pre_projection_dim=1024):
    s1_bands = [1, 2]
    s2_bands = list(range(1, 14))  # Bands 1 through 13
    model = CIIP(
        framework="resnet50",
        embed_dim=embed_dim,
        pre_projection_dim=pre_projection_dim,
        s1_resolution=224,
        s1_layers=(3, 4, 6, 3),
        s1_width=32,
        s1_patch_size=16,  # used by transformer
        s1_bands=len(s1_bands),
        s2_resolution=224,
        s2_layers=(3, 4, 6, 3),  # ResNet-34
        s2_width=32,
        s2_patch_size=16,  # used by transformer
        s2_bands=len(s2_bands),
    )
    return model

# def clean_ciip_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
#     """
#     Cleans the state dictionary by removing keys that are not needed for the model.
#     """
#     keys_to_remove = []
#     for k, v in state_dict.items():
#         if "encoder_s1" in k:
#             keys_to_remove.append(k)
    
#     for key in keys_to_remove:
#         if key in state_dict:
#             del state_dict[key]

#     # remove prefix
#     state_dict = {k.replace("encoder_s2.", ""): v for k, v in state_dict.items() if "encoder_s2." in k}
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    
#     return state_dict

def load_ciip_model_checkpoint(checkpoint_path):
    model = create_ciip_model()
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu') # , map_location='cuda'

    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # remove fcs
    new_state_dict = {k: v for k, v in new_state_dict.items() if "fc" not in k}

    # print the incompatible keys
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)
    assert set(missing_keys) <= {'encoder_s1.fc.weight', 'encoder_s1.fc.bias', 'encoder_s2.fc.weight', 'encoder_s2.fc.bias'}
    assert not unexpected_keys


    print("Checkpoint loaded successfully, loaded S1 and S2 weights.")

    return model


# def modify_ciip_for_eurosat(model, num_classes=10, freeze_encoder=False):
#     """
#     Modify the CIIP model for the EuroSAT dataset by adjusting the encoder_s2 to include
#     a classification head and adjust positional embedding.
#     """
#     if not hasattr(model, "encoder_s2"):
#         raise AttributeError("The model does not have an 'encoder_s2' attribute.")

#     encoder_s2 = model.encoder_s2

#     # Freeze encoder parameters if requested
#     if freeze_encoder:
#         for param in encoder_s2.parameters():
#             param.requires_grad = False

#     # Add a classification head
#     encoder_s2.fc = nn.Linear(512, num_classes)

#     if freeze_encoder:
#         for param in encoder_s2.fc.parameters():
#             param.requires_grad = True

#     # Adjust positional embedding for the new resolution, currently commented out
#     # adjust_positional_embedding(encoder_s2, resolution=264, patch_size=16)

#     # Wrap the forward method
#     original_forward = encoder_s2.forward

#     def new_forward(x):
#         x = original_forward(x)
#         x = encoder_s2.fc(x)
#         return x

#     encoder_s2.forward = new_forward

#     print("Model modified for EuroSAT dataset with a classification head.")
#     return model



# def adjust_positional_embedding(encoder, resolution=264, patch_size=16):
#     """
#     Adjust the positional embedding in the attention pooling layer to match the new resolution and patch size.
#     """
#     if not hasattr(encoder, "attnpool") or encoder.attnpool is None:
#         print("No attention pooling layer found; skipping positional embedding adjustment.")
#         return

#     # Calculate the expected number of patches
#     num_patches = (resolution // patch_size) ** 2  # For a 264x264 image with 16x16 patches, this should be 256
#     expected_embedding_size = num_patches + 1  # +1 for the [CLS] token

#     # Get the current positional embedding
#     positional_embedding = encoder.attnpool.positional_embedding
#     current_size = positional_embedding.size(0)

#     if current_size != expected_embedding_size:
#         print(f"Adjusting positional embeddings: {current_size} -> {expected_embedding_size}")
#         # Interpolate to the new size
#         new_positional_embedding = F.interpolate(
#             positional_embedding.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
#             size=(expected_embedding_size,),
#             mode="linear",
#             align_corners=False,
#         ).squeeze(0).squeeze(0)  # Remove added dimensions

#         # Update the positional embedding in the model
#         encoder.attnpool.positional_embedding = nn.Parameter(new_positional_embedding)
#     else:
#         print("Positional embeddings already match the expected size; no adjustment needed.")

