import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import os
import sys

from data import get_data
# from test_data import get_dummy_dataloader
# from ..loss import ClipLoss
from train import train_one_epoch

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Now you can import your function
from loss import CiipLoss
from model_ciip import CIIP

ciip_model = CIIP(
    embed_dim=512,
    pre_projection_dim=1024,
    s1_resolution=264,
    s1_layers=(3, 4, 6, 3), #Resnet-34
    s1_width=32,
    s1_patch_size=16,
    s1_bands=3,
    s2_resolution=264,
    s2_layers=(3, 4, 6, 3), #Resnet-34
    s2_width=32,
    s2_patch_size=16,
    s2_bands=12,
)


def main():
    # Hyperparameters and arguments
    class Args:
        root = 'data/'
        dataset_type = 'ssl4eo'
        train_data = True
        val_data = False
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
        batch_size = 2
        horovod = False
        distributed = False
        workers = 0
        s2_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] #[2, 3, 4]

    args = Args()

    # Initialize model, loss, optimizer, and scaler
    model = ciip_model.to(args.device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the number of parameters
    print(f"Total number of parameters: {count_parameters(model)}")

    count_parameters_encoder1 = model.count_parameters_encoder1()
    count_parameters_encoder2 = model.count_parameters_encoder2()

    print(f"Total number of parameters in encoder1: {count_parameters_encoder1}")
    print(f"Total number of parameters in encoder2: {count_parameters_encoder2}")

    # print(model)

    loss_fn = CiipLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler() if args.precision == 'fp16' else None
    scheduler = lambda step: None  # Dummy scheduler for testing

    # get dummy dtaloader
    # dataloader = get_dummy_dataloader(batch_size=args.batch_size, img_size1=(3, 264, 264), img_size2=(3, 264, 264))

    # Initialize data loader
    data_info = get_data(args, (None, None))
    # data_in = data_info['train']
    
    # print(f'DATALOADER NUM EPOCHS: {data_in.num_epochs}')
    # data_info = {'train': type('ssl4eo', (object,), {'dataloader': dataloader, 'set_epoch': lambda x: None})}
    
    # Initialize tensorboard writer
    
    tb_writer = None

    # Train for one epoch
    train_one_epoch(model, data_info, loss_fn, 0, optimizer, scaler, scheduler, None, args, tb_writer)

if __name__ == "__main__":
    main()
