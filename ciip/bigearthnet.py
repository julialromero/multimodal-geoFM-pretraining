import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchgeo.datasets import BigEarthNet
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

def load_data(root_path="/local/ms-data/bigearthnet", bands="all", batch_size=64, transforms=None):
    dataset_train = BigEarthNet(root_path, bands=bands, split="train", transforms=transforms)
    dataset_val = BigEarthNet(root_path, bands=bands, split="val", transforms=transforms)
    dataset_test = BigEarthNet(root_path, bands=bands, split="test", transforms=transforms)

    # define a dataloader to iterate over the dataset
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)
    
    return dataloader_train, dataloader_val, dataloader_test

# Custom transform function to handle the dictionary structure of torchgeo dataset
class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample
    

def load_ciip_model_checkpoint(checkpoint_path):
    # TODO FIGURE OUT WHAT IS GETTING LOADED AND HOW TO ACTUALLY LOAD IT
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
    return model

def modify_ciip_for_bigearthnet(model):
    # grab just the s2_encoder part of the model
    encoder_s2 = model.encoder_s2
    # add in another layer to match the number of classes in the BigEarthNet dataset
    encoder_s2.fc = nn.Linear(512, 19)
    # Wrap the forward function
    original_forward = encoder_s2.forward

    def new_forward(x):
        x = original_forward(x)  # Get the 512-dim embedding from the existing forward pass
        x = encoder_s2.fc(x)  # Pass through the new FC layer for 19-class output
        return x

    # Replace the model's forward with the new forward function
    encoder_s2.forward = new_forward
    return encoder_s2

def test_model(dataloader, model, loss_fn=nn.CrossEntropyLoss(), val=False):
    print("Running test_model function...")
    model.to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        i = 0
        # calculate the number of iterations the following loop will run
        for sample in dataloader:
            i += 1
            X, y = sample['image'], sample['label']
            X, y = X.to(device), y.float().to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            if i % 24 == 0:
                print("Sample number", i, "of", num_batches, "-- Test loss:", test_loss, "-- Correct:", correct, "of")
    test_loss /= num_batches
    correct /= size
    prefix = "Validation" if val else "Test"
    print(f"{prefix} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_model(dataloader, model, loss_fn=nn.CrossEntropyLoss(), lr=1e-3):
    print("Running train_model function...")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, sample in enumerate(dataloader):
        X, y = sample['image'], sample['label']
        X, y = X.to(device), y.float().to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print intermediate results every n batches
        n = 24
        if batch % n == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Batch {batch:d} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # calculate avg training loss
    train_loss /= num_batches
    print(f"Training Avg Loss: {train_loss:>8f} \n")
    


# run this code if calling this file directly to load the model run inferences on the BigEarthNet dataset
if __name__ =="__main__":
    # Test the model on the BigEarthNet dataset
    root_path = "/local/ms-data/bigearthnet"
    batch_size = 256
    tx = transforms.Resize((264, 264))  # Resizes the images to 224x224
    custom_tx = CustomTransform(tx)
    print("Loading data...")
    dataloader_train, dataloader_val, dataloader_test = load_data(root_path, bands="s2", batch_size=batch_size, transforms=tx)
    model = load_ciip_model_checkpoint("/local/ms-data/SSL4EO/model/epoch_50.pt")
    print("Checkpoint loaded successfully.")
    # grab just the s2_encoder part of the model
    encoder_s2 = modify_ciip_for_bigearthnet(model)
    print("Model modified for BigEarthNet dataset.")
    torch.save(encoder_s2.state_dict(), "/local/ms-data/bigearthnet/model/s2_encoder_epoch0.pt")
    # configure gpu
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.cuda.set_device(1)
    print("Device set to:", device)
    # train the model
    epochs = 5
    lr = 1e-4
    print("Testing model before training...")
    test_model(dataloader_test, encoder_s2, val=False)
    print("Training model for" , epochs, "epochs...")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_model(dataloader_train, encoder_s2, lr=lr)
        test_model(dataloader_val, encoder_s2, val=True)
        # checkpoint the model
        torch.save(encoder_s2.state_dict(), f"/local/ms-data/bigearthnet/model/s2_encoder_lr1e-4_epoch{t+1}.pt")
    test_model(dataloader_test, encoder_s2, val=False)