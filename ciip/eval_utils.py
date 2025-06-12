import torch
import torch.nn as nn
from collections import OrderedDict
from model_ciip import CIIP  # Make sure this is defined correctly
from model import ModifiedResNet   
import torch.nn.functional as F


class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample



def create_ciip_model():
    s1_bands = [1, 2]
    s2_bands = list(range(1, 14))  # Bands 1 through 13
    model = CIIP(
        embed_dim=512,
        s1_resolution=264,
        s1_layers=(3, 4, 6, 3),
        s1_width=32,
        s1_patch_size=16,  # used by transformer
        s1_bands=len(s1_bands),
        s2_resolution=264,
        s2_layers=(3, 4, 6, 3),  # ResNet-34
        s2_width=32,
        s2_patch_size=16,  # used by transformer
        s2_bands=len(s2_bands),
    )
    return model

def load_ciip_model_checkpoint(checkpoint_path):
    model = create_ciip_model()
    checkpoint = torch.load(checkpoint_path)

    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # Skip loading weights for encoder_s1
        if "encoder_s1" in k:
            continue
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    print("Checkpoint loaded successfully, excluding Sentinel-1 weights.")

    #If needed to use only 12 pands, these next two blocks need to be uncommented.
    # Modify the first convolutional layer to accept 13 input channels
    # original_conv1 = model.encoder_s2.conv1
    # new_conv1 = nn.Conv2d(
    #     in_channels=13,
    #     out_channels=original_conv1.out_channels,
    #     kernel_size=original_conv1.kernel_size,
    #     stride=original_conv1.stride,
    #     padding=original_conv1.padding,
    #     bias=original_conv1.bias,
    # )


    # with torch.no_grad():
    #     for i, idx in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, None, 9, 10, 11]):  # Match bands 1-9, initialize B10, map 11-13
    #         if i == 9:  # Initialize B10 (index 9 in the new weights)
    #             new_conv1.weight[:, i, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)[:, 0, :, :] #maybe better to initialize instead of mean
    #         elif idx is not None:  # Map directly for other bands (excluding B10)
    #             new_conv1.weight[:, i, :, :] = original_conv1.weight[:, idx, :, :]

    
    # model.encoder_s2.conv1 = new_conv1

    return model


def modify_ciip_for_eurosat(model, num_classes=10, freeze_encoder=False):
    """
    Modify the CIIP model for the EuroSAT dataset by adjusting the encoder_s2 to include
    a classification head and adjust positional embedding.
    """
    if not hasattr(model, "encoder_s2"):
        raise AttributeError("The model does not have an 'encoder_s2' attribute.")

    encoder_s2 = model.encoder_s2

    # Freeze encoder parameters if requested
    if freeze_encoder:
        for param in encoder_s2.parameters():
            param.requires_grad = False

    # Add a classification head
    encoder_s2.fc = nn.Linear(512, num_classes)

    if freeze_encoder:
        for param in encoder_s2.fc.parameters():
            param.requires_grad = True

    # Adjust positional embedding for the new resolution, currently commented out
    # adjust_positional_embedding(encoder_s2, resolution=264, patch_size=16)

    # Wrap the forward method
    original_forward = encoder_s2.forward

    def new_forward(x):
        x = original_forward(x)
        x = encoder_s2.fc(x)
        return x

    encoder_s2.forward = new_forward

    print("Model modified for EuroSAT dataset with a classification head.")
    return model



def adjust_positional_embedding(encoder, resolution=264, patch_size=16):
    """
    Adjust the positional embedding in the attention pooling layer to match the new resolution and patch size.
    """
    if not hasattr(encoder, "attnpool") or encoder.attnpool is None:
        print("No attention pooling layer found; skipping positional embedding adjustment.")
        return

    # Calculate the expected number of patches
    num_patches = (resolution // patch_size) ** 2  # For a 264x264 image with 16x16 patches, this should be 256
    expected_embedding_size = num_patches + 1  # +1 for the [CLS] token

    # Get the current positional embedding
    positional_embedding = encoder.attnpool.positional_embedding
    current_size = positional_embedding.size(0)

    if current_size != expected_embedding_size:
        print(f"Adjusting positional embeddings: {current_size} -> {expected_embedding_size}")
        # Interpolate to the new size
        new_positional_embedding = F.interpolate(
            positional_embedding.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            size=(expected_embedding_size,),
            mode="linear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # Remove added dimensions

        # Update the positional embedding in the model
        encoder.attnpool.positional_embedding = nn.Parameter(new_positional_embedding)
    else:
        print("Positional embeddings already match the expected size; no adjustment needed.")


# def create_ciip_model():
#     s1_bands = [1, 2]
#     s2_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#     model = CIIP(
#         embed_dim=512,
#         s1_resolution=264,
#         s1_layers= (3, 4, 6, 3),
#         s1_width=32,
#         s1_patch_size=16, # used by transformer 
#         s1_bands=len(s1_bands),
#         s2_resolution=264,
#         s2_layers=(3, 4, 6, 3), #Resnet-34
#         s2_width=32,
#         s2_patch_size=16, # used by transformer
#         s2_bands=len(s2_bands)
#     )
#     return model  

# def load_ciip_model_checkpoint(checkpoint_path):
#     model = create_ciip_model()
#     checkpoint = torch.load(checkpoint_path)

#     state_dict = checkpoint['state_dict']
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k.replace("module.", "")  # remove `module.`
#         new_state_dict[name] = v
#     # load params
#     model.load_state_dict(new_state_dict)

#     print("Checkpoint loaded successfully.")
    
#     return model

# def modify_ciip_for_eurosat(model, num_classes=10, freeze_encoder=False):
#     # grab just the s2_encoder part of the model
#     encoder_s2 = model.encoder_s2

#     # Freeze the parameters of the original encoder
#     if freeze_encoder:
#         for param in encoder_s2.parameters():
#             param.requires_grad = False
        
#     # add in another layer to match the number of classes in the EuroSAT dataset
#     encoder_s2.fc = nn.Linear(512, num_classes)

#     # unfreeze the last layer
#     # only need to do this if the encoder was frozen
#     if freeze_encoder:
#         for param in encoder_s2.fc.parameters():
#             param.requires_grad = True

#     # Wrap the forward function
#     original_forward = encoder_s2.forward

#     def new_forward(x):
#         x = original_forward(x)  # Get the 512-dim embedding from the existing forward pass
#         x = encoder_s2.fc(x)  # Pass through the new FC layer for 19-class output
#         return x

#     # Replace the model's forward with the new forward function
#     encoder_s2.forward = new_forward
#     print("Model modified for EuroSAT dataset.")
    
#     return encoder_s2