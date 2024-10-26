import torch
from torchgeo.datasets import BigEarthNet
from torch.utils.data import DataLoader
from test import test_model
# from open_clip_train import run_train_val
from torchgeo.models import resnet50
from torch import nn
from model_ciip import CIIP

# Load the BigEarthNet dataset
# https://torchgeo.readthedocs.io/en/stable/api/datasets.html#bigearthnet
# https://doi.org/10.1109/IGARSS.2019.8900532
def download_data(root_path="local/ms-data/bigearthnet"):
    print('Loading BigEarthNet dataset...')

    print('--Loading train dataset...')
    _ = BigEarthNet(root=root_path, split="train", download=True)

    print('--Loading validation dataset...')
    _ = BigEarthNet(root=root_path, split="val", download=True)

    print('--Loading test dataset...')
    _ = BigEarthNet(root=root_path, split="test", download=True)
    
    print('BigEarthNet dataset downloaded!')

def load_data(root_path="/local/ms-data/bigearthnet", bands="all", batch_size=64):
    dataset_train = BigEarthNet(root_path, bands=bands, split="train")
    dataset_val = BigEarthNet(root_path, bands=bands, split="val")
    dataset_test = BigEarthNet(root_path, bands=bands, split="test")

    # define a dataloader to iterate over the dataset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)
    
    return dataloader_train, dataloader_val, dataloader_test

def load_ciip_model_s2_weights():
    # TODO FIGURE OUT WHAT IS GETTING LOADED AND HOW TO ACTUALLY LOAD IT
    model = torch.load("/local/ms-data/SSL4EO/model/epoch_50.pt", weights_only=False)
    # model.eval()
    # grab just the s2_encoder part of the model
    encoder_s2 = model.encoder_s2

    # add in another layer to match the number of classes in the BigEarthNet dataset
    encoder_s2.fc = nn.Linear(512, 19)
    return encoder_s2

# Test the model on the BigEarthNet dataset
root_path = "/local/ms-data/bigearthnet"
_, _, dataloader_test = load_data(root_path, bands="s2", batch_size=64)
# model = resnet50(weights=None, num_classes=19, in_chans=12) # .to(device)
model = load_ciip_model_s2_weights()
# change the output layer to match the number of classes in the BigEarthNet dataset
model.fc = nn.Linear(2048, 19)
# test the model
test_model(dataloader_test, model, nn.CrossEntropyLoss(), val=False)

# TODO load the ciip model s2 weights