import json
import logging
import math
import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)
from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast

import random
from torch.utils.data import Dataset, DataLoader

class Subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        # TODO: make sure the same keys are used in all models
        "s1_features": model_out[0],
        "s2_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # TODO: figure out what dist_model is
    device = torch.device(args.datamodule.device)
    autocast = get_autocast(args.model.precision)
    input_dtype = get_input_dtype(args.model.precision)

    model.train()
    
    ## for using pretrained model, these pretrained models are distributedparallel models
    ## this means we have to either use a distributedparallel model, otherwise i think we have to rename the state_dict keys 
    ## (as the naming convention is slightly different for regular vs distributed parallel)
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.train.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.train.accum_freq > 1:
        accum_s1, accum_s2, accum_features = [], [], {}


    with torch.no_grad():
        if epoch == 0 and args.train.apply_orthogonal_mapping:
            #log
            logging.info("Computing optimal orthogonal mapping W for s1 to s2...")
            base_dataset = dataloader.dataset
            random.seed(42)
            sample_indices = random.sample(range(len(base_dataset)), 2_000)
            subset = Subset(base_dataset, sample_indices)
            loader = DataLoader(subset, batch_size=300, shuffle=False)
            batch = next(iter(loader))
            s1, s2 = batch
            s1 = s1.to(device=device, dtype=input_dtype, non_blocking=True)
            s2 = s2.to(device=device, dtype=input_dtype, non_blocking=True)
            if hasattr(model, "module"):
                W, stats = model.module.compute_orthogonal_matrix(s1, s2)
            else:
                W, stats = model.compute_orthogonal_matrix(s2, s1)
            logging.info(f"Computed W: {W.shape}")
            logging.info(f"Orthogonal matrix stats: {stats}")
            # save stats to log directory
            if args.io.save_logs:
                with open(os.path.join(args.io.checkpoint_path, "orthogonal_matrix_stats.json"), "w") as f:
                    json.dump(stats, f, indent=4)
                # save W matrix to log directory 
                W_path = os.path.join(args.io.checkpoint_path, "W.pt")
                torch.save(W, W_path)
                logging.info(f"Saved orthogonal matrix W to {W_path}")


    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.train.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.model.skip_scheduler:
            scheduler(step)

        s1, s2 = batch
        s1 = s1.to(device=device, dtype=input_dtype, non_blocking=True)
        s2 = s2.to(device=device, dtype=input_dtype, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.train.accum_freq == 1:
            with autocast():
                model_out = model(s1, s2)

                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(s1, s2)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(s1, s2)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_s1.append(s1)
                accum_s2.append(s2)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.train.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.train.accum_freq):
                s1 = accum_s1[j]
                s2 = accum_s2[j]
                with autocast():
                    model_out = model(s1, s2)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.datamodule.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.model.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.model.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.model.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.model.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.model.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.model.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.train.accum_freq > 1:
            accum_s1, accum_s2, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.io.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(s1)
            num_samples = batch_count * batch_size * args.train.accum_freq * args.datamodule.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.train.accum_freq * args.datamodule.batch_size * args.datamodule.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.train.accum_freq * args.datamodule.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.io.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.train.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.model.precision)
    input_dtype = get_input_dtype(args.model.precision)

    if 'val' in data and (args.train.val_frequency and ((epoch % args.train.val_frequency) == 0 or epoch == args.train.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_image_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_s1_features, all_s2_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                s1, s2 = batch
                s1 = s1.to(device=device, dtype=input_dtype, non_blocking=True)
                s2 = s2.to(device=device, dtype=input_dtype, non_blocking=True)

                with autocast():
                    model_out = model(s1, s2)
                    s1_features = model_out["s1_features"]
                    s2_features = model_out["s2_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_s1_features.append(s1_features.cpu())
                    all_s2_features.append(s2_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_s1 = logit_scale * s1_features @ s2_features.t()
                    logits_per_s2 = logits_per_s1.t()

                    batch_size = s1.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_s1, labels) +
                        F.cross_entropy(logits_per_s2, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"CIIP Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            # print(f's1: {all_s1_features[0].shape, all_s1_features[-1].shape}')
            # print(f's2: {all_s2_features[0].shape, all_s2_features[-1].shape}')

            val_metrics = get_clip_metrics(
                s1_features=torch.cat(all_s1_features),
                s2_features=torch.cat(all_s2_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "CIIP_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.io.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.io.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.io.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.train.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(s1_features, s2_features, logit_scale):
    metrics = {}
    logits_per_s1 = (logit_scale * s1_features @ s2_features.t()).detach().cpu()
    logits_per_s2 = logits_per_s1.t().detach().cpu()

    logits = {"s1_to_s2": logits_per_s1, "s2_to_s1": logits_per_s2}
    ground_truth = torch.arange(len(s2_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

def main(args):
    start_epoch = 0
    model = CLIP(**vars(args))
    original_model = model

    for epoch in range(start_epoch, args.train.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        if args.io.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.train.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.io.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            # if args.delete_previous_checkpoint:
            #     previous_checkpoint = os.path.join(args.io.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
            #     if os.path.exists(previous_checkpoint):
            #         os.remove(previous_checkpoint)

            # if args.save_most_recent:
            #     # try not to corrupt the latest checkpoint if save fails
            #     tmp_save_path = os.path.join(args.io.checkpoint_path, "tmp.pt")
            #     latest_save_path = os.path.join(args.io.checkpoint_path, LATEST_CHECKPOINT_NAME)
            #     torch.save(checkpoint_dict, tmp_save_path)
            #     os.replace(tmp_save_path, latest_save_path)

if __name__ == "__main__":
    main(sys.argv[1:])    