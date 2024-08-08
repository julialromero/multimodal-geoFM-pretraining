import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import os
import sys


from test_data import get_dummy_dataloader
# from ..loss import ClipLoss
from train import train_one_epoch

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Now you can import your function
from loss import CiipLoss
from model_ciip import CIIP

dummy_model = CIIP(
    embed_dim=512,
    s1_resolution=32,
    s1_layers=(3, 4, 6, 3), #Resnet-34
    s1_width=512,
    s1_patch_size=16,
    s1_bands=3,
    s2_resolution=32,
    s2_layers=(3, 4, 6, 3), #Resnet-34
    s2_width=512,
    s2_patch_size=16,
    s2_bands=12,
)

class DummyModel(nn.Module):
    def __init__(self, feature_dim=512):
        super(DummyModel, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, feature_dim),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, feature_dim),
            nn.ReLU()
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, x1, x2):
        f1 = self.encoder1(x1)
        f2 = self.encoder2(x2)
        return {"s1_features": f1, "s2_features": f2, "logit_scale": self.logit_scale}

def main():
    # Hyperparameters and arguments
    class Args:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        precision = 'fp16' if torch.cuda.is_available() else 'fp32'
        accum_freq = 1
        log_every_n_steps = 10
        grad_clip_norm = None
        skip_scheduler = False
        save_logs = False
        wandb = False
        distill = False
        epochs = 1
        val_frequency = 1
        checkpoint_path = './'
        world_size = 1
        rank = 0
        batch_size = 32

    args = Args()

    # Initialize model, loss, optimizer, and scaler
    model = dummy_model.to(args.device)
    loss_fn = CiipLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler() if args.precision == 'fp16' else None
    scheduler = lambda step: None  # Dummy scheduler for testing

    # Initialize data loader
    dataloader = get_dummy_dataloader(batch_size=32)
    data = {'train': type('dummy', (object,), {'dataloader': dataloader, 'set_epoch': lambda x: None})}
    
    # Initialize tensorboard writer
    tb_writer = None

    # Train for one epoch
    train_one_epoch(model, data, loss_fn, 0, optimizer, scaler, scheduler, None, args, tb_writer)

if __name__ == "__main__":
    main()
