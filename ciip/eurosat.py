import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchgeo.datasets import EuroSATSpatial
from torchgeo.models import resnet50, ResNet50_Weights
from torchgeo.models import resnet18, ResNet18_Weights
from model_ciip import CIIP
from sklearn.metrics import f1_score

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
def download_data(root_path):
    # ensure root_path is not empty
    if not root_path:
        raise ValueError("root_path cannot be empty")
    # check if root_path exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    # download
    print('Downloading dataset...')
    _ = EuroSATSpatial(root=root_path, download=True)
    print('Dataset downloaded!')

def load_data(root_path="/local/ms-data/eurosat", bands="all", batch_size=64, num_workers=16, transforms=None):
    print("Loading data...")
    dataset_train = EuroSATSpatial(root_path, bands=bands, split="train", transforms=transforms)
    dataset_val = EuroSATSpatial(root_path, bands=bands, split="val", transforms=transforms)
    dataset_test = EuroSATSpatial(root_path, bands=bands, split="test", transforms=transforms)

    # define a dataloader to iterate over the dataset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers)
    
    return dataloader_train, dataloader_val, dataloader_test

# Custom transform function to handle the dictionary structure of torchgeo dataset
class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample

def load_ciip_model_checkpoint(checkpoint_path):
    s1_bands = [1, 2, 3]
    s2_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
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
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("Checkpoint loaded successfully.")
    
    return model

def modify_ciip_for_eurosat(model, num_classes=10):
    # grab just the s2_encoder part of the model
    encoder_s2 = model.encoder_s2

    # Freeze the parameters of the original encoder
    for param in encoder_s2.parameters():
        param.requires_grad = False
        
    # add in another layer to match the number of classes in the EuroSAT dataset
    encoder_s2.fc = nn.Linear(512, num_classes)

    # encoder_s2.fc = MLPHead(input_dim=512, final_dim=num_classes)
    for param in encoder_s2.fc.parameters():
        param.requires_grad = True

    # # Wrap the forward function
    original_forward = encoder_s2.forward

    def new_forward(x):
        x = original_forward(x)  # Get the 512-dim embedding from the existing forward pass
        x = encoder_s2.fc(x)  # Pass through the new FC layer for 19-class output
        return x

    # Replace the model's forward with the new forward function
    encoder_s2.forward = new_forward
    print("Model modified for EuroSAT dataset.")
    
    return encoder_s2

def test_model(dataloader, model, loss_fn=nn.CrossEntropyLoss(), val=False):
    # print("Running test_model function...")
    model.to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        i = 0
        for sample in dataloader:
            i += 1
            X, y = sample['image'], sample['label']
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Collect predictions and labels
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # if i % 2 == 0:
            #     print("Sample number", i, "of", num_batches, "-- Test loss:", test_loss, "-- Correct:", correct)
    
    test_loss /= num_batches
    correct /= size

    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Use 'macro' or 'weighted' depending on your dataset

    prefix = "Validation" if val else "Test"
    print(f"-- {prefix} Error -- Avg loss: {test_loss:>8f}, Accuracy: {(100*correct):>0.1f}%, F1 Score: {100*f1:>0.1f}% \n")


def train_model(dataloader, model, loss_fn=nn.CrossEntropyLoss(), lr=1e-3):
    # print("Training model...")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, sample in enumerate(dataloader):
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

        # # print intermediate results every n batches
        # n = 4
        # if batch % n == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"Batch {batch:d} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # calculate avg training loss
    train_loss /= num_batches
    print(f"-- Training Error -- Avg Loss: {train_loss:>8f}")

def get_s2_encoder_model(model_name):
    if model_name == "resnet50":
        model = resnet50(weights=None, in_chans=12, num_classes=10)
    elif model_name == "resnet50_pretrained":
        model = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO, in_chans=12, num_classes=10)
    elif model_name == "resnet18":
        model = resnet18(weights=None, in_chans=12, num_classes=10)
    elif model_name == "resnet18_pretrained":
        model = resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO, in_chans=12, num_classes=10)
    elif model_name == "ciip":
        path_to_ciip_model = "/local/ms-data/SSL4EO/model/bs_128_09-2024/epoch_50.pt"
        ciip_model = load_ciip_model_checkpoint(path_to_ciip_model)
        model = modify_ciip_for_eurosat(ciip_model)
    else:
        raise ValueError("Model not found.")
    return model

# run this code if calling this file directly to load the model run inferences on the EuroSAT dataset
if __name__ =="__main__":
    root_path = "/ADrive/data/eurosat"
    model_type = "resnet50" # choose from "resnet50", "resnet50_pretrained", "resnet18", "resnet18_pretrained", "ciip"
    batch_size = 128
    num_workers = 32
    epochs = 50
    lr = 1e-3
    torch.manual_seed(7)
    tx = transforms.Compose([
        transforms.Resize((264, 264))  # Resizes the images
        # normalize values according to -> https://d-nb.info/1239826591/34
        # remember to exclude B10 cirrus band
        # transforms.Normalize(mean=[1353.439, 1117.253, 1042.253, 947.128, 1199.404, 2002.936, 2373.488, 2300.642, 732.159, 12.113, 1119.173, 2598.82], 
                            #   std=[65.571, 154.376, 188.262, 278.926, 228.244, 355.633, 454.901, 530.549, 98.718, 1.187, 304.439, 501.747])
    ])
    custom_tx = CustomTransform(tx)
    download_data(root_path)
    dataloader_train, dataloader_val, dataloader_test = load_data(
        root_path, 
        bands=('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'), # drop B10, not included in SSL4EO level 2a 
        batch_size=batch_size, 
        transforms=tx,
        num_workers=num_workers
    )
    encoder_s2 = get_s2_encoder_model(model_type)
    print("Model loaded successfully. Using model:", model_type)
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.cuda.set_device(1)
    print("Device set to:", device)
    print("Testing model before training...")
    test_model(dataloader_test, encoder_s2, val=False)
    print("Training model for" , epochs, "epochs with a batch size of", batch_size, "and learning rate of", lr)
    for t in range(epochs):
        print(f"Epoch {t+1} -------------------------------")
        train_model(dataloader_train, encoder_s2, lr=lr)
        test_model(dataloader_val, encoder_s2, val=True)
    test_model(dataloader_test, encoder_s2, val=False)

# TODO traverse the learning rates (already tried 1e-3, 1e-2, 1e-1)
# TODO try with more epochs once one seems to do okay with 5 epochs

# TODO try with the model weights from epoch 5 and epoch 25 of CIIP weights
# TODO try with other heads for the CIIP model
# TODO try extracting the embeddings for all of the eurosat images then training just the classifier
# TODO try with pretrained weights from other models
# TODO try with 50 epochs
# TODO try with other learning rates for the randomly initialized model baseline (already tried 1e-3, 1e-1)