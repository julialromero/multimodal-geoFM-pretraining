import json
import logging
import math
import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F

from typing import Dict, Tuple

from torch.nn.parallel.distributed import DistributedDataParallel
from torch.autograd.profiler import record_function

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
from loss import gather_features

import random
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

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


def _compute_vc_geometry(
        features: torch.Tensor,
        gamma: float,
        eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """Compute diagnostic statistics for the variance-covariance regularizer."""
    if features.ndim != 2 or features.shape[0] < 2:
        nan = float("nan")
        return nan, nan, nan, nan

    autocast_off = (
        torch.cuda.amp.autocast(enabled=False)
        if features.is_cuda and torch.cuda.is_available()
        else nullcontext()
    )

    with autocast_off:
        feats = features.float()
        variances = torch.var(feats, dim=0, unbiased=False)
        std = torch.sqrt(variances + 1e-4)
        std_min = torch.min(std)
        gamma_tensor = torch.as_tensor(gamma, dtype=std.dtype, device=std.device)
        std_diff = std_min - gamma_tensor

        centered = feats - feats.mean(dim=0, keepdim=True)
        cov = centered.t().matmul(centered) / (centered.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        cov_fro = torch.linalg.norm(off_diag, ord="fro")

        eigvals = torch.linalg.eigvalsh(cov).real
        participation = (eigvals.sum() ** 2) / (eigvals.pow(2).sum() + eps)

    return (
        std_min.item(),
        std_diff.item(),
        cov_fro.item(),
        participation.item(),
    )


def _update_vc_metric_meter(meters: Dict[str, AverageMeter], name: str, value: float, weight: int) -> None:
    meter = meters.setdefault(name, AverageMeter())
    if math.isnan(value):
        meter.val = value
        meter.avg = value
    else:
        meter.update(value, weight)


def _accumulate_vc_metrics(
        loss,
        model_out: Dict[str, torch.Tensor],
        vc_metrics_m: Dict[str, AverageMeter],
        batch_size: int,
) -> None:
    if not getattr(loss, "vc_reg_enabled", False):
        return
    with torch.no_grad():
        gathered_features = {}
        for modality in ("s1", "s2"):
            raw_key = f"{modality}_features_vc"
            feature_key = f"{modality}_features"
            features = model_out.get(raw_key)
            if features is None:
                features = model_out.get(feature_key)
            gathered_features[modality] = features

        world_size = getattr(loss, "world_size", 1)
        if world_size > 1:
            local_loss = getattr(loss, "local_loss", False)
            rank = getattr(loss, "rank", 0)
            use_horovod = getattr(loss, "use_horovod", False)

            s1_tensor = gathered_features.get("s1")
            s2_tensor = gathered_features.get("s2")
            if s1_tensor is not None or s2_tensor is not None:
                if s1_tensor is not None and s2_tensor is not None:
                    s1_tensor, s2_tensor = gather_features(
                        s1_tensor,
                        s2_tensor,
                        local_loss,
                        False,
                        rank,
                        world_size,
                        use_horovod,
                    )
                else:
                    if s1_tensor is not None:
                        s1_tensor, _ = gather_features(
                            s1_tensor,
                            s1_tensor,
                            local_loss,
                            False,
                            rank,
                            world_size,
                            use_horovod,
                        )
                    if s2_tensor is not None:
                        s2_tensor, _ = gather_features(
                            s2_tensor,
                            s2_tensor,
                            local_loss,
                            False,
                            rank,
                            world_size,
                            use_horovod,
                        )
                gathered_features["s1"] = s1_tensor
                gathered_features["s2"] = s2_tensor

        for modality, features in gathered_features.items():
            if features is None:
                continue
            features = features.detach()
            std_min, std_diff, cov_fro, participation = _compute_vc_geometry(features, loss.vc_gamma)
            _update_vc_metric_meter(vc_metrics_m, f"vc_std_min_{modality}", std_min, batch_size)
            _update_vc_metric_meter(vc_metrics_m, f"vc_std_diff_{modality}", std_diff, batch_size)
            _update_vc_metric_meter(vc_metrics_m, f"vc_cov_fro_{modality}", cov_fro, batch_size)
            _update_vc_metric_meter(vc_metrics_m, f"vc_participation_ratio_{modality}", participation, batch_size)


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

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
from datetime import datetime, timedelta
def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f'/home/juro4948/ciip/mem_snapshots/{timestamp}_{prof.step}'

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


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
            loader = DataLoader(base_dataset, batch_size=1000, shuffle=False, num_workers=args.model.workers, pin_memory=True)
            it = iter(loader)
            batch = next(it)
            s1, s2 = batch
            s1 = s1.to(device=device, dtype=input_dtype, non_blocking=True)
            s2 = s2.to(device=device, dtype=input_dtype, non_blocking=True)


            
            ###### COMPUTE TRAINING ORTHOGONAL MAPPING ######
            ####### WARMUP FOR ORTHOGONAL MAPPING #######
            logging.info("Computing orthogonal matrix for **TRAIN**...")
            NUM_WARMUP_BATCHES = 100
            model.train()
            for i, (s1, s2) in enumerate(loader):
                s1 = s1.to(device)
                s2 = s2.to(device)
                if hasattr(model, "module"):
                    _ = model.module.encode_s1(s1)
                    _ = model.module.encode_s2(s2)
                else:
                    _ = model.encode_s1(s1)
                    _ = model.encode_s2(s2)
                if i >= NUM_WARMUP_BATCHES:
                    print(f"Warmup for orthogonal mapping completed after {i} batches.")
                    break

            
            if hasattr(model, "module"):
                W, stats = model.module.compute_orthogonal_matrix(s1, s2)
            else:
                W, stats = model.compute_orthogonal_matrix(s2, s1)
            logging.info(f"Orthogonal matrix train stats: {stats}")

            # save stats to log directory
            if args.io.save_logs:
                with open(os.path.join(args.io.checkpoint_path, "orthogonal_matrix_stats_train.json"), "w") as f:
                    json.dump(stats, f, indent=4)
                W_path = os.path.join(args.io.checkpoint_path, "W_train.pt")
                torch.save(W, W_path)
                logging.info(f"Saved orthogonal matrix W to {W_path}")

            # apply orthogonal mapping to modeltorch.save(
            

            if hasattr(model, "module"):
                del model.module.encoder_s1._buffers["W"]
                model.module.encoder_s1.register_buffer("W", None)
                model.module.encoder_s1.apply_orthogonal_matrix = False
                
            else:
                del model.encoder_s1._buffers["W"]
                model.encoder_s1.register_buffer("W", None)
                model.encoder_s1.apply_orthogonal_matrix = False
            print("W is set to None in the model, so that it does not apply orthogonal mapping during training.")


            ###### EVAL ORTHOGOANL MATRIX COMPUTATION ######
            logging.info("Computing orthogonal matrix for **EVAL**...")
            model.eval()
            if hasattr(model, "module"):
                W, stats = model.module.compute_orthogonal_matrix(s1, s2)
            else:
                W, stats = model.compute_orthogonal_matrix(s2, s1)
            logging.info(f"Orthogonal matrix eval stats: {stats}")

            # save stats to log directory
            if args.io.save_logs:
                with open(os.path.join(args.io.checkpoint_path, "orthogonal_matrix_stats.json"), "w") as f:
                    json.dump(stats, f, indent=4)
                # save W matrix to log directory 
                W_path = os.path.join(args.io.checkpoint_path, "W.pt")
                torch.save(W, W_path)
                logging.info(f"Saved orthogonal matrix W to {W_path}")



            

            # delete  dataloader
            del loader
    # save the model weights after the warm up phase -> random weights with appropriate batch norm stats
    torch.save(
    model.state_dict(),
        os.path.join(args.io.checkpoint_path, f"epoch_0.pt"),
    )
    logging.info(f'Saved initial model weights to {args.io.checkpoint_path}.')

            
    model.train()
    losses_m = {}
    vc_metrics_m: Dict[str, AverageMeter] = {}
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

                _accumulate_vc_metrics(loss, model_out, vc_metrics_m, s1.shape[0])

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(s1, s2)

                    _accumulate_vc_metrics(loss, model_out, vc_metrics_m, s1.shape[0])

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
            accum_features_current_loop = {key: list(val) for key, val in accum_features.items()} # Ensure a copy

            for j in range(args.train.accum_freq):
                s1 = accum_s1[j]
                s2 = accum_s2[j]

                is_last_backward_pass = (j == args.train.accum_freq - 1)
                sync_context = model.no_sync() if not is_last_backward_pass else nullcontext()
                with sync_context:
                    with autocast():
                        
                        model_out = model(s1, s2)

                        inputs_no_accum = {}
                        inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                        if "logit_bias" in model_out:
                            inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                        # inputs = {}
                        # for key, val in accum_features.items():
                        #     accumulated = accum_features[key]
                        #     inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])
                        inputs = {}
                        for key in accum_features_current_loop.keys(): # Iterate over keys
                            # Use the copied list for concatenation, and replace the j-th element
                            temp_accumulated = list(accum_features_current_loop[key]) # Create a copy for modification
                            temp_accumulated[j] = model_out[key] # Replace with the current grad-tracked feature
                            inputs[key] = torch.cat(temp_accumulated)

                        losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                        # logging.info(f"Losses for batch {i_accum}, inner pass {j + 1}/{args.train.accum_freq}: {losses}")
                        
                        total_loss = sum(losses.values()) #/ args.train.accum_freq
                        losses["loss"] = total_loss

                    backward(total_loss, scaler)

                del inputs
                del inputs_no_accum

        # log gradients
        # logit_scale_param = unwrap_model(model).logit_scale
        # if logit_scale_param.grad is not None:
        #     grad_norm = logit_scale_param.grad.data.norm().item()
        #     logit_scale_val = logit_scale_param.item()
        #     logging.info(f"Logit scale: {logit_scale_val:.3f}, grad norm: {grad_norm:.3e}")
        # else:
        #     logging.warning("Logit scale grad is None!")

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
            metrics_log_parts = [
                f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                for loss_name, loss_m in losses_m.items()
            ]
            if vc_metrics_m:
                metrics_log_parts.extend(
                    f"{name}: {meter.val:#.5g} ({meter.avg:#.5g})"
                    for name, meter in vc_metrics_m.items()
                )
            metrics_log = " ".join(metrics_log_parts)
            samples_per_second = args.train.accum_freq * args.datamodule.batch_size * args.datamodule.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.train.accum_freq * args.datamodule.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + metrics_log
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
            log_data.update({name: val.val for name, val in losses_m.items()})
            if vc_metrics_m:
                log_data.update({name: meter.val for name, meter in vc_metrics_m.items()})

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