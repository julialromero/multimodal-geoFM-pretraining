import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, num_samples=8, img_size1=(3, 32, 32), img_size2=(12, 32, 32)):
        self.num_samples = num_samples
        self.img_size1 = img_size1
        self.img_size2 = img_size2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img1 = torch.randn(self.img_size1)
        img2 = torch.randn(self.img_size2)
        return img1, img2

def get_dummy_dataloader(batch_size=8, num_samples=8, img_size1=(3, 32, 32), img_size2=(12, 32, 32)):
    dataset = DummyDataset(num_samples, img_size1, img_size2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader.num_batches =  batch_size
    dataloader.num_samples = num_samples
    return dataloader
