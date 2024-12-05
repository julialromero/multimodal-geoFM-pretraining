import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchgeo.datasets import EuroSAT
from torchgeo.models import resnet50, ResNet50_Weights
from torchgeo.models import resnet18, ResNet18_Weights
from model_ciip import CIIP
from sklearn.metrics import f1_score
from collections import OrderedDict
from datetime import datetime

# from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout, Linear

# # define a simple MLP head for the CIIP model
# class MLPHead(Module):
#     def __init__(self, input_dim, final_dim):
#         hidden_size = 256
#         super(MLPHead, self).__init__()
#         self.fc1 = Linear(input_dim, hidden_size)  # First fully connected layer
#         self.relu = ReLU()                          # ReLU activation function
#         self.fc2 = Linear(hidden_size, final_dim)
        
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# load the dataset
def download_data(data_path):
    # ensure data_path is not empty
    if not data_path:
        raise ValueError("data_path cannot be empty")
    # check if data_path exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # download
    print('Downloading dataset...')
    _ = EuroSAT(root=data_path, download=True)
    print('Dataset downloaded!')

def load_data(data_path, bands, batch_size, num_workers, transforms):
    print("Loading data...")
    dataset_train = EuroSAT(data_path, bands=bands, split="train", transforms=transforms)
    dataset_val = EuroSAT(data_path, bands=bands, split="val", transforms=transforms)
    dataset_test = EuroSAT(data_path, bands=bands, split="test", transforms=transforms)

    # define a dataloader to iterate over the dataset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return dataloader_train, dataloader_val, dataloader_test

# Custom transform function to handle the dictionary structure of torchgeo dataset
class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample

def create_ciip_model():
    s1_bands = [1, 2]
    s2_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    model = CIIP(
        embed_dim=512,
        s1_resolution=264,
        s1_layers= (3, 4, 6, 3),
        s1_width=32,
        s1_patch_size=16, # used by transformer 
        s1_bands=len(s1_bands),
        s2_resolution=264,
        s2_layers=(3, 4, 6, 3), #Resnet-34
        s2_width=32,
        s2_patch_size=16, # used by transformer
        s2_bands=len(s2_bands)
    )
    return model  

def load_ciip_model_checkpoint(checkpoint_path):
    model = create_ciip_model()
    checkpoint = torch.load(checkpoint_path)

    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    print("Checkpoint loaded successfully.")
    
    return model

def modify_ciip_for_eurosat(model, num_classes=10, freeze_encoder=False):
    # grab just the s2_encoder part of the model
    encoder_s2 = model.encoder_s2

    # Freeze the parameters of the original encoder
    if freeze_encoder:
        for param in encoder_s2.parameters():
            param.requires_grad = False
        
    # add in another layer to match the number of classes in the EuroSAT dataset
    encoder_s2.fc = nn.Linear(512, num_classes)

    # unfreeze the last layer
    # only need to do this if the encoder was frozen
    if freeze_encoder:
        for param in encoder_s2.fc.parameters():
            param.requires_grad = True

    # Wrap the forward function
    original_forward = encoder_s2.forward

    def new_forward(x):
        x = original_forward(x)  # Get the 512-dim embedding from the existing forward pass
        x = encoder_s2.fc(x)  # Pass through the new FC layer for 19-class output
        return x

    # Replace the model's forward with the new forward function
    encoder_s2.forward = new_forward
    print("Model modified for EuroSAT dataset.")
    
    return encoder_s2

def test_model(dataloader, model, loss_fn, file, val=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sample in dataloader:
            X, y = sample['image'], sample['label']
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Collect predictions and labels
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # calculate metrics
    test_loss /= num_batches
    correct /= size
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Use 'macro' or 'weighted' depending on your dataset

    # report on metrics
    prefix = "Validation" if val else "Test"
    output_general((f"-- {prefix} Error -- Avg loss: {test_loss:>8f}, Accuracy: {(100*correct):>0.1f}%, F1 Score: {100*f1:>0.1f}% \n"), file)


def train_model(dataloader, model, loss_fn, lr, file):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for _, sample in enumerate(dataloader):
        X, y = sample['image'], sample['label']
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # calculate avg training loss
    train_loss /= num_batches
    output_general(f"-- Training Error -- Avg Loss: {train_loss:>8f}", file)

def get_s2_encoder_model(model_name, num_bands, ciip_model_checkpoint_path):
    if model_name == "resnet50":
        model = resnet50(weights=None, in_chans=num_bands, num_classes=10)
    elif model_name == "resnet50_pretrained":
        model = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO, in_chans=num_bands, num_classes=10)
    elif model_name == "resnet18":
        model = resnet18(weights=None, in_chans=num_bands, num_classes=10)
    elif model_name == "resnet18_pretrained":
        model = resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO, in_chans=num_bands, num_classes=10)
    elif model_name == "resnet32_modified":
        full_ciip_model = create_ciip_model()
        model = modify_ciip_for_eurosat(full_ciip_model, num_classes=10, freeze_encoder=False)
    elif model_name == "ciip":
        ciip_model = load_ciip_model_checkpoint(ciip_model_checkpoint_path)
        model = modify_ciip_for_eurosat(ciip_model, num_classes=10, freeze_encoder=False)
    else:
        raise ValueError("Model not found.")
    return model

def output_general(your_string, file):
    print(your_string)
    file.write(your_string + "\n")

# run this code if calling this file directly to load the model run inferences on the EuroSAT dataset
if __name__ =="__main__":

    today = str(datetime.now())
    filename = "benchmarks/eurosat/" + today + ".txt"
    file = open(filename, "w")

    # set run params
    root_path = "/ADrive/data"
    data_path = os.path.join(root_path, "eurosat")
    model_type = "ciip" # choose from "resnet50", "resnet50_pretrained", "resnet18", "resnet18_pretrained", "resnet32_modified", "ciip"
    bands = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')  # drop B10, not included in SSL4EO level 2a 
    num_bands = len(bands)
    input_dim = (264, 264)
    batch_size = 256
    num_workers = 32
    epochs = 30
    lr = 1e-4
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(44)
    which_gpu = 1
    experiment_group = "bs_512_11-2024"
    experiment_name = "ciip" + "_lr" + str(lr) + "_bs" + str(batch_size) + "_norm" + "_e" + str(epochs) 
    ciip_model_name = "epoch_35"
    model_checkpoint_path = os.path.join(root_path, "ciip_model", experiment_group, experiment_name + ".pt")
    ciip_model_checkpoint_path = os.path.join(root_path, "ciip_model", experiment_group, ciip_model_name + ".pt") 

    # configure and load the data
    tx = transforms.Compose([
        transforms.Resize(input_dim),  # Resizes the images, 224 for ResNet, 264 for CIIP
        # normalize values according to -> https://d-nb.info/1239826591/34
        # remember to exclude B10 cirrus band
        transforms.Normalize(mean=[1353.439, 1117.253, 1042.253, 947.128, 1199.404, 2002.936, 2373.488, 2300.642, 732.159, 12.113, 1820.932, 1119.173, 2598.82], 
                              std=[65.571, 154.376, 188.262, 278.926, 228.244, 355.633, 454.901, 530.549, 98.718, 1.187, 378.496, 304.439, 501.747])
    ])
    custom_tx = CustomTransform(tx)
    download_data(data_path)
    dataloader_train, dataloader_val, dataloader_test = load_data(
        data_path, 
        bands=bands,
        batch_size=batch_size, 
        transforms=tx,
        num_workers=num_workers
    )
    
    # define and load the model, configure devices
    model = get_s2_encoder_model(model_type, num_bands, ciip_model_checkpoint_path)
    output_general(("Model loaded successfully. Using model:", model_type), file)
    output_general(("Using bands:", bands), file)
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.cuda.set_device(which_gpu)
    model.to(device)
    output_general(("Device set to:", device), file)
    output_general(("Image preprocessing steps:", tx), file)

    # test and train the model
    output_general("Testing model before training...", file)
    test_model(dataloader_test, model, loss_fn, val=False)
    output_general(("Training model for" , epochs, "epochs with a batch size of", batch_size, "and learning rate of", lr), file)
    for t in range(epochs):
        output_general(f"Epoch {t + 1} -------------------------------", file)
        train_model(dataloader_train, model, loss_fn, lr, file)
        test_model(dataloader_val, model, loss_fn, file, val=True)
    test_model(dataloader_test, model, loss_fn, file, val=False)
    torch.save(model.state_dict(), model_checkpoint_path)

# TODO track loss in a variable > change output to a .txt or .csv file
# TODO add early stopping
# TODO try with other heads for the CIIP model
# TODO try extracting the embeddings for all of the eurosat images then training just the classifier
# TODO try with pretrained weights from other models
# TODO try with 50 epochs