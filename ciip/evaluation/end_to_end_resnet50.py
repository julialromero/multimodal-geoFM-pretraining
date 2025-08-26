import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

from torchgeo.datasets import EuroSAT
from torchgeo.models import resnet50

# Mean and std for each Sentinel-2 band used in EuroSAT
MEAN = {
    'B01': 1354.40546513,
    'B02': 1118.24399958,
    'B03': 1042.92983953,
    'B04': 947.62620298,
    'B05': 1199.47283961,
    'B06': 1999.79090914,
    'B07': 2369.22292565,
    'B08': 2296.82608323,
    'B09': 732.08340178,
    'B10': 12.11327804,
    'B11': 1819.01027855,
    'B12': 1118.92391149,
    'B8A': 2594.14080798,
}

STD = {
    'B01': 245.71762908,
    'B02': 333.00778264,
    'B03': 395.09249139,
    'B04': 593.75055589,
    'B05': 566.4170017,
    'B06': 861.18399006,
    'B07': 1086.63139075,
    'B08': 1117.98170791,
    'B09': 404.91978886,
    'B10': 4.77584468,
    'B11': 1002.58768311,
    'B12': 761.30323499,
    'B8A': 1231.58581042,
}

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")

class CustomTransform:
    """Wrapper to apply torchvision transforms to dict samples."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])
        return sample

def get_dataloaders(data_path, batch_size=64):
    mean_list = [MEAN[b] for b in BANDS]
    std_list = [STD[b] for b in BANDS]

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean_list, std=std_list),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=mean_list, std=std_list),
        ]),
    }

    train_ds = EuroSAT(data_path, split="train", bands=BANDS, transforms=CustomTransform(data_transforms["train"]), download=True)
    val_ds = EuroSAT(data_path, split="val", bands=BANDS, transforms=CustomTransform(data_transforms["val"]), download=True)
    test_ds = EuroSAT(data_path, split="test", bands=BANDS, transforms=CustomTransform(data_transforms["val"]), download=True)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4),
    }
    return loaders, len(train_ds.classes)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss / len(loader.dataset), acc, f1

def eval_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss / len(loader.dataset), acc, f1

def train_resnet50_end_to_end(data_path, epochs=10, batch_size=64, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, num_classes = get_dataloaders(data_path, batch_size)

    model = resnet50(weights=None, in_chans=len(BANDS), num_classes=num_classes, pretrained=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, val_f1 = eval_epoch(model, loaders["val"], criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} acc: {train_acc:.3f} f1: {train_f1:.3f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} f1: {val_f1:.3f}")

    test_loss, test_acc, test_f1 = eval_epoch(model, loaders["test"], criterion, device)
    print(f"Test - loss: {test_loss:.4f} acc: {test_acc:.3f} f1: {test_f1:.3f}")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="End-to-end supervised training of ResNet50 on EuroSAT")
    parser.add_argument("data_path", help="Path to EuroSAT dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_resnet50_end_to_end(args.data_path, args.epochs, args.batch_size, args.lr)
