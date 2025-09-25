import os
import sys
from datetime import datetime
import torch
import glob
import logging
import re
import subprocess
import time
import random
import hydra
from datetime import datetime
from functools import partial
from omegaconf import DictConfig, OmegaConf
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

######## added
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from utils import create_loss, create_model
from torchvision.transforms import v2
###

# from open_clip import create_model_and_transforms #, trace_model, get_tokenizer, create_loss
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object, world_info_from_env
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync



def allocate_gpu_memory_for_fragmentation(device: torch.device, current_rank: int = 0):
    """
    Attempts to allocate a large tensor to manage GPU memory fragmentation.
    The allocated tensor is returned and *must be kept in scope* by the caller
    for the memory to remain reserved.
    """
    allocated_tensor = None
    allocation_sizes_gb = [80, 70, 64, 55] # Example sizes

    print(f"Rank {current_rank}: Attempting to pre-allocate GPU memory on {device} for fragmentation management.")

    for size_gb in allocation_sizes_gb:
        try:
            num_elems = size_gb * 1024**3 // 4 # Convert GB to number of float32 elements
            print(f"Rank {current_rank}: Attempting to allocate {size_gb:.2f} GB ({num_elems * 4 / 1024**3:.2f} GiB) on {device}...")
            
            # Allocate on the specific device for this process
            temp_tensor = torch.empty(num_elems, dtype=torch.float32, device=device)
            # Perform a small operation to ensure memory is truly committed, if desired
            temp_tensor.zero_() 

            allocated_tensor = temp_tensor # Store the reference
            print(f"Rank {current_rank}: Successfully allocated {size_gb:.2f} GB on {device}. Holding memory.")
            
            # Run nvidia-smi after successful allocation
            print(f"Rank {current_rank}: nvidia-smi after successful allocation:")
            if os.environ.get("SLURM_PROCID", "0") == "0":
                os.system("nvidia-smi")
            break # Exit loop if allocation is successful
        except torch.cuda.OutOfMemoryError:
            print(f"Rank {current_rank}: Failed to allocate {size_gb:.2f} GB on {device}. GPU out of memory (fragmentation likely). Trying smaller size.")
        except Exception as e:
            print(f"Rank {current_rank}: An unexpected error occurred during allocation: {e}")
            break # Exit loop on other unexpected errors

    if allocated_tensor is None:
        print(f"Rank {current_rank}: Unable to allocate any substantial memory on {device}. Proceeding without explicit large reservation.")
        # os.system("nvidia-smi")
 
    if os.environ.get("SLURM_PROCID", "0") == "0":
        print(f"Rank {current_rank} | Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB | Reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    
    return allocated_tensor # Return the tensor so it stays in scope in the main function


torch.cuda.current_device()  # <-- or torch.empty(1, device='cuda')
torch.cuda.synchronize()


def setup_slurm_logging():
    local_rank, global_rank, world_size = world_info_from_env()
    date_dir = os.environ.get('MY_LOG_BASE_DIR')
    log_dir = os.path.join(date_dir, 'torchrun_logs')
    os.makedirs(log_dir, exist_ok=True)
    stdout_log = os.path.join(log_dir, f"stdout_rank{global_rank}.log")
    stderr_log = os.path.join(log_dir, f"stderr_rank{global_rank}.log")

    if global_rank == 0:
        # Tee output to both SLURM .out and file
        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self.streams:
                    s.flush()
        sys.stdout = Tee(open(stdout_log, "w", buffering=1), sys.__stdout__)
        sys.stderr = Tee(open(stderr_log, "w", buffering=1), sys.__stderr__)
    else:
        # Only log to file for other ranks
        sys.stdout = open(stdout_log, "w", buffering=1)
        sys.stderr = open(stderr_log, "w", buffering=1)

    print(f"[Rank {global_rank}] Logging started.", flush=True)


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
CONF = "prod_default"

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

@hydra.main(config_path="configs", config_name=CONF)
# def main(args):
def main(args: DictConfig, start_epoch=0):

#     # record memory
#     MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT=10000
#     torch.cuda.memory._record_memory_history(
#        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
#    )


    # args = parse_args(args)
    args.io.logs = os.environ.get('MY_LOG_BASE_DIR')
    device = init_distributed_device(args.datamodule)
    local_rank, global_rank, world_size = world_info_from_env()

    setup_slurm_logging()
          
    # Setup text logger
    args.log_level = logging.DEBUG if args.train.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        
    _reserved_memory_tensor = allocate_gpu_memory_for_fragmentation(device, current_rank=global_rank)
    file_monitor_dir = os.environ.get('MY_LOG_BASE_DIR')
    check_path = os.path.join(file_monitor_dir, 'data_extraction_done.txt')
    temp_path = f'/tmp/jromero5/allocated_{local_rank}.txt'
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    with open(temp_path, 'w') as f:
        f.write(f"Allocated {os.environ.get('SLURM_PROCID', '-1')} {os.environ.get('CUDA_VISIBLE_DEVICES', '-1')}")
    
    extracting_data = True
    while extracting_data:
        time.sleep(30)
        now =datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if os.path.exists(check_path):
            extracting_data = False
            print(f'{now} | File found, continuing with training')
            logging.info(f'{now} | File found, continuing with training')
        else:
            print(f'{now} | File not found, waiting for data extraction to finish: {check_path}')
            logging.info(f'{now} | File not found, waiting for data extraction to finish: {check_path}')
            # os.system("nvidia-smi")

    # delete the allocatd tensor to free up memory
    if os.environ.get("SLURM_PROCID", "0") == "0":
        print(f"BEFORE FREEING TENSOR: Rank {args.datamodule.rank} | Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB | Reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
        print('####### DELETING 80GB ALLOCATED TENSOR #######')
        logging.info('####### DELETING 80GB ALLOCATED TENSOR #######')
    del _reserved_memory_tensor
    # torch.cuda.empty_cache()

    

    # get the name of the experiments
    if True: #args.train.name is None:
        # raise NotImplementedError('Name not yet supported.')
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.framework.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.datamodule.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args.datamodule, date_str)
        args.train.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.train.lr}",
            f"b_{args.datamodule.batch_size}",
            f"j_{args.model.workers}",
            f"p_{args.model.precision}",
        ])
        print(f'Experiment name using {args.train.name}.')

    resume_latest = args.io.resume == 'latest'
    log_base_path = os.path.join(args.io.logs, args.train.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.datamodule.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

        #setup comet_ml logging
        if(args.io.comet_ml):
            experiment = Experiment(
                api_key=args.comet.api_key,
                project_name=args.comet.project_name,
                workspace=args.comet.workspace
            )
        



    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = False #'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = False #'tensorboard' in args.report_to or 'all' in args.report_to
    args.io.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.io.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.io.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.train.name, "checkpoints")
            if args.train.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.train.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.datamodule.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args.datamodule, resume_from)
        args.io.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.io.logs, args.train.name), 
            os.path.join(args.remote_sync, args.train.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.io.logs, args.train.name), 
            os.path.join(args.remote_sync, args.train.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.model.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.datamodule.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.datamodule.device}.'
            f'Process (global: {args.datamodule.rank}, local {args.datamodule.local_rank}), total {args.datamodule.world_size}.')
    elif args.datamodule.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.datamodule.device}.'
            f'Process (global: {args.datamodule.rank}, local {args.datamodule.local_rank}), total {args.datamodule.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.datamodule.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.train.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.framework.lower()

    # if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
    #     # arg is nargs, single (square) image size list -> int
    #     args.force_image_size = args.force_image_size[0]
    random_seed(args.train.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    if model_kwargs:
        print(model_kwargs)
        # raise NotImplementedError('Model kwargs not yet supported.')
    model = create_model(
        args,
        device=device,
        # jit=args.torchscript,
        # force_quick_gelu=args.force_quick_gelu,
        # force_custom_text=args.force_custom_text,
        # force_patch_dropout=args.force_patch_dropout,
        # force_image_size=args.force_image_size,
        # image_mean=args.image_mean,
        # image_std=args.image_std,
        # image_interpolation=args.image_interpolation,
        # image_resize_mode=args.image_resize_mode,  # only effective for inference
        # aug_cfg=args.aug_cfg,
        # pretrained_image=args.pretrained_image,
        # output_dict=True,
        **model_kwargs,
    )
    if args.distill:
        raise NotImplementedError('Distillation not yet supported.')
        # # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        # dist_model, _, _ = create_model_and_transforms(
        #     args.distill_model, 
        #     args.distill_pretrained,
        #     device=device,
        #     precision=args.precision,
        #     output_dict=True,
        # )
    
    random_seed(args.train.seed, args.datamodule.rank)

    # if args.trace:
    #     model = trace_model(model, batch_size=args.batch_size, device=device)

    # if args.lock_image:
    #     # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
    #     model.lock_image_tower(
    #         unlocked_groups=args.lock_image_unlocked_groups,
    #         freeze_bn_stats=args.lock_image_freeze_bn_stats)
    # if args.lock_text:
    #     model.lock_text_tower(
    #         unlocked_layers=args.lock_text_unlocked_layers,
    #         freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.io.logs, args.train.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.datamodule.distributed and not args.datamodule.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.dataset.train_data or args.dataset.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer= optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.train.wd},
        ],
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        eps=args.train.eps,
        )

        if args.datamodule.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.model.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.io.resume is not None:
        # raise NotImplementedError('Resume not yet supported.')
        checkpoint = pt_load(args.io.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.datamodule.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.io.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.io.resume}' (epoch {start_epoch})")
    else:
        logging.info(f"=> no checkpoint found at '{args.io.resume}', starting from scratch.")
        # save the omegaconf config to a yaml file
        if is_master(args):
            config_file = os.path.join(args.io.logs, "prod_default.yaml")
            with open(config_file, "w") as f:
                OmegaConf.save(args, f)
            logging.info(f"Saved config to {config_file}")


    if args.dataset.use_transforms:
      transforms = v2.Compose([
          v2.RandomResizedCrop(size=(args.model.s1_resolution, args.model.s2_resolution), antialias=True),
          v2.RandomHorizontalFlip(p=0.5),
          v2.RandomVerticalFlip(p=0.5),
          v2.GaussianBlur(3),
      ])
    else:
        transforms = None

    data = get_data(args, transforms)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.train.accum_freq) * args.train.epochs
        # if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.train.lr, args.train.warmup, total_steps)
        # elif args.lr_scheduler == "const":
        #     scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        # elif args.lr_scheduler == "const-cooldown":
        #     assert args.epochs_cooldown is not None,\
        #         "Please specify the number of cooldown epochs for this lr schedule."
        #     cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
        #     scheduler = const_lr_cooldown(
        #         optimizer, args.lr, args.warmup, total_steps,
        #         cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        # else:
        #     logging.error(
        #         f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
        #     exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.io.save_logs and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        raise NotImplementedError('Wandb not yet supported.')
        # assert wandb is not None, 'Please install wandb.'
        # logging.debug('Starting wandb.')
        # args.train_sz = data["train"].dataloader.num_samples
        # if args.val_data is not None:
        #     args.val_sz = data["val"].dataloader.num_samples
        # # you will have to configure this for your project!
        # wandb.init(
        #     project=args.wandb_project_name,
        #     name=args.train.name,
        #     id=args.train.name,
        #     notes=args.wandb_notes,
        #     tags=[],
        #     resume='auto' if args.io.resume == "latest" else None,
        #     config=vars(args),
        # )
        # if args.debug:
        #     wandb.watch(model, log='all')
        # wandb.save(params_file)
        # logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)

    if 'train' not in data:
        raise NotImplementedError('Validation is not yet supported.')
        # # If using int8, convert to inference mode.
        # if args.use_bnb_linear is not None:
        #     from open_clip.utils import convert_int8_model_to_inference_mode
        #     convert_int8_model_to_inference_mode(model)
        # # Evaluate.
        # evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        # return

    loss = create_loss(args)

    
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M")
    if is_master(args):
        print(f'Before training: Rank {args.datamodule.rank} {os.system("nvidia-smi")}', flush=True)

    
    # MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT=100000
    # torch.cuda.memory._record_memory_history(
    #    max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    # )

    for epoch in range(start_epoch, args.train.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        # Right before training each epoch
        # torch.cuda.empty_cache()

        # print(torch.cuda.memory_summary())
        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        # print time, rank
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Rank {args.datamodule.rank} finished epoch {epoch} |  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB | Reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB | Memory summary: {torch.cuda.memory_summary(device=0, abbreviated=True)}', flush=True)
       
   
        # if epoch < 10: 
        #     try:
        #         save_dir = os.path.join(args.io.mem_snapshots, date_str)
        #         os.makedirs(save_dir, exist_ok=True)
        #         save_path = os.path.join(save_dir, F'rank{args.datamodule.rank}_gpu{args.datamodule.local_rank}_epoch{epoch}.pickle')
        #         torch.cuda.memory._dump_snapshot(save_path)
        #         logging.info(f"Captured memory snapshot {save_path}")
        #         print(f"Captured memory snapshot {save_path}")
        #     except Exception as e:
        #         logging.error(f"Failed to capture memory snapshot {e}")
        #         print(f'Failed to capture memory snapshot {e}')
        # else:
        #     torch.cuda.memory._record_memory_history(enabled=None)
        completed_epoch = epoch + 1

        # if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        #     evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        if args.io.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.train.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.train.epochs \
                    or (args.io.save_frequency > 0 and (completed_epoch % args.io.save_frequency) == 0) \
                        or completed_epoch == 1:
                # save checkpoints within outputs file
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.io.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )

                # log out via comet
                # TBD if we want the whole checkpoint dict or just some specific hyper-params . . .
                if is_master(args, local=args.log_local):
                    if(args.io.comet_ml):
                        experiment.log_parameters(checkpoint_dict)
                        log_model(experiment, model=original_model, model_name="CIIP!")

        if args.train.delete_previous_checkpoint:
            previous_checkpoint = os.path.join(args.io.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)

        if args.train.save_most_recent:
            # try not to corrupt the latest checkpoint if save fails
            tmp_save_path = os.path.join(args.io.checkpoint_path, "tmp.pt")
            latest_save_path = os.path.join(args.io.checkpoint_path, LATEST_CHECKPOINT_NAME)
            torch.save(checkpoint_dict, tmp_save_path)
            os.replace(tmp_save_path, latest_save_path)


    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.io.logs, args.train.name), 
            os.path.join(args.remote_sync, args.train.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

# def copy_codebase(args):
#     from shutil import copytree, ignore_patterns
#     new_code_path = os.path.join(args.logs, args.name, "code")
#     if os.path.exists(new_code_path):
#         print(
#             f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
#         )
#         return -1
#     print(f"Copying codebase to {new_code_path}")
#     current_code_path = os.path.realpath(__file__)
#     for _ in range(3):
#         current_code_path = os.path.dirname(current_code_path)
#     copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
#     print("Done copying code.")
#     return 1


if __name__ == "__main__":
    main()
    # main(sys.argv[1:])
