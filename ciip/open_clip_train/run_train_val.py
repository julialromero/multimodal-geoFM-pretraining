import logging
import os
import hydra
from omegaconf import DictConfig
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model


from ciip.open_clip_train.checkpointing import (
    build_training_checkpoint,
    remove_checkpoint,
    save_checkpoint,
)
from ciip.open_clip_train.data import get_data
from ciip.open_clip_train.config_validation import validate_training_config
from ciip.open_clip_train.train import train_one_epoch
from ciip.open_clip_train.optimizer import create_optimizer
from ciip.open_clip_train.scheduler import cosine_lr
from ciip.open_clip_train.utils import create_loss, create_model

from torchvision.transforms import v2
# from torchvision.transforms import *

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
# Change this to local_default for local testing
# CONF = "local_default"
CONF = "prod_default"

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name=CONF)
def main(args: DictConfig, start_epoch=0):

  validate_training_config(args, runner="single")

  print("LOCAL_RANK", os.environ.get("LOCAL_RANK"))
  # Get the Local Rank
  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  args.train.device = "cuda:%d" % local_rank

  loss = create_loss(args)

  model = create_model(args, device=args.train.device)

  #setup comet_ml logging
  if(args.io.comet_ml):
      experiment = Experiment(
          api_key=args.comet.api_key,
          project_name=args.comet.project_name,
          workspace=args.comet.workspace
      )


  transforms = None
  if args.dataset.use_transforms:
      transforms = v2.Compose([
          v2.RandomResizedCrop(
              size=(args.model.s1_resolution, args.model.s2_resolution),
              antialias=True,
          ),
          v2.RandomHorizontalFlip(p=0.5),
          v2.RandomVerticalFlip(p=0.5),
          v2.GaussianBlur(3),
      ])

  data = get_data(args, transforms)
  steps_per_epoch = data["train"].dataloader.num_batches // args.train.accum_freq
  total_steps = steps_per_epoch * args.train.epochs

  optimizer = create_optimizer(model, loss, args.train, getattr(args, "loss", None))

  scheduler = cosine_lr(optimizer, args.train.lr, args.train.warmup, total_steps)

  dist_model = None
  tb_writer = None
  # TODO(behzad): alternatively we might need to use what is in the original code:
  # scaler = GradScaler() if args.precision == "amp" else None
  scaler = None

  # check if checkpoint outdir exists
  os.makedirs(args.io.checkpoint_path, exist_ok=True)

  for epoch in range(start_epoch, args.train.epochs):

    logger.info(f'Start epoch {epoch}')

    train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer)
    completed_epoch = epoch + 1

    # print(data['train'].dataloader)
    # print(data['val'].dataloader)
    # print("IF statement", 'val' in data)
    # if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
    #     evaluate(model, data, completed_epoch, args, tb_writer)

    original_model = model

    # Saving checkpoints.
    if args.io.save_logs:
        checkpoint_dict = build_training_checkpoint(
            model,
            optimizer,
            loss,
            epoch=completed_epoch,
            name=args.train.name,
            scaler=scaler,
        )

        if completed_epoch == args.train.epochs \
                or (args.io.save_frequency > 0 and (completed_epoch % args.io.save_frequency) == 0):
            # save checkpoints within outputs file
            save_checkpoint(
                checkpoint_dict,
                os.path.join(args.io.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )

            # log out via comet
            # TBD if we want the whole checkpoint dict or just some specific hyper-params . . .
            if(args.io.comet_ml):
                # Extract only the scalar items
                experiment.log_parameters({
                    "epoch": checkpoint_dict["epoch"],
                    "name": checkpoint_dict["name"],
                    # Add any other scalar or string fields here...
                })
                log_model(experiment, model=original_model, model_name="CIIP!")

        if args.train.delete_previous_checkpoint:
            remove_checkpoint(
                os.path.join(args.io.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
            )

        if args.train.save_most_recent:
            save_checkpoint(
                checkpoint_dict,
                os.path.join(args.io.checkpoint_path, LATEST_CHECKPOINT_NAME),
            )



if __name__ == "__main__":
    main()
