import json
import logging
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from open_clip import get_input_dtype
from torch.utils.data import Dataset

try:
    import wandb
except ImportError:
    wandb = None

from ciip.loss import gather_features
from ciip.open_clip_train.accumulation import (
    build_accumulated_loss_inputs,
    cache_model_features,
)
from ciip.open_clip_train.distributed import is_master
from ciip.open_clip_train.batches import prepare_training_batch
from ciip.open_clip_train.objectives import (
    backward_loss,
    compose_training_loss,
    reconstruction_weight,
    run_training_step,
    scalar_output,
)
from ciip.open_clip_train.optimizer import step_optimizer
from ciip.open_clip_train.precision import get_autocast
from ciip.open_clip_train.zero_shot import zero_shot_eval

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


def _compute_vc_geometry(features: torch.Tensor, vc_gamma: float, eps: float = 1e-6):
    std = torch.std(features, dim=0)
    std_min = std.min()
    std_diff = (vc_gamma - std).relu().mean()

    cov_fro = torch.tensor(0.0, device=features.device)
    participation = torch.tensor(0.0, device=features.device)
    if features.shape[0] > 1:
        centered = features - features.mean(dim=0)
        cov = centered.T @ centered / (centered.shape[0] - 1)
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


TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
from datetime import datetime
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
    model_cfg = getattr(args, "model", None)
    encoder_pair = getattr(model_cfg, "encoder_pair", "s1s2")

    model.train()
    
    ## for using pretrained model, these pretrained models are distributedparallel models
    ## this means we have to either use a distributedparallel model, otherwise i think we have to rename the state_dict keys 
    ## (as the naming convention is slightly different for regular vs distributed parallel)
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.train.accum_freq
    max_train_steps = getattr(args.train, "total_steps", None)
    if max_train_steps is not None:
        max_train_steps = max(int(max_train_steps), 1)
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    rank0 = args.datamodule.rank == 0
    logged_accum_boundary = False
    logged_first_backward = False
    stop_training = False

    if rank0:
        logging.debug(
            "Rank0: train_one_epoch start epoch=%s batches=%s accum_freq=%s workers=%s",
            epoch,
            dataloader.num_batches,
            args.train.accum_freq,
            args.model.workers,
        )

    if args.train.accum_freq > 1:
        accumulated_batches, accum_features = [], {}

    model.train()
    losses_m = {}
    vc_metrics_m: Dict[str, AverageMeter] = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        if rank0 and i == 0:
            logging.debug("Rank0: fetched batch 0 from dataloader")
        i_accum = i // args.train.accum_freq
        step = num_batches_per_epoch * epoch + i_accum
        if max_train_steps is not None and step >= max_train_steps:
            stop_training = True
            break
        if hasattr(loss, "set_gather_context"):
            loss.set_gather_context(
                {
                    "epoch": epoch,
                    "batch": i,
                    "accum": i_accum,
                    "micro": None,
                    "step": step,
                }
            )

        if not args.model.skip_scheduler:
            scheduler(step)

        prepared = prepare_training_batch(
            batch,
            encoder_pair=encoder_pair,
            device=device,
            input_dtype=input_dtype,
        )
        s1 = prepared.s1

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        curv_scalar: Optional[float] = None

        if args.train.accum_freq == 1:
            if rank0 and i == 0:
                logging.debug("Rank0: forward start batch 0")
            model_out, losses, total_loss = run_training_step(
                model,
                prepared.model_inputs,
                loss,
                autocast=autocast,
                reconstruction_config=getattr(args, "recon", None),
                epoch=epoch,
                step=step,
                scaler=scaler,
                distillation_model=dist_model if args.distill else None,
            )
            if rank0 and i == 0:
                logging.debug("Rank0: forward and backward done for batch 0")
            if "curv" in model_out:
                curv_scalar = float(model_out["curv"].detach().item())
            _accumulate_vc_metrics(loss, model_out, vc_metrics_m, s1.shape[0])
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    if rank0 and i == 0:
                        logging.debug("Rank0: forward(no_grad) start batch 0")
                    model_out = model(*prepared.model_inputs)
                    if rank0 and i == 0:
                        logging.debug("Rank0: forward(no_grad) done batch 0")

                    _accumulate_vc_metrics(loss, model_out, vc_metrics_m, s1.shape[0])
                    cache_model_features(accum_features, model_out)

                accumulated_batches.append(prepared)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.train.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue
            if rank0 and not logged_accum_boundary:
                logging.debug("Rank0: reached accum boundary at batch %s", i)
                logged_accum_boundary = True

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            accum_features_current_loop = {key: list(val) for key, val in accum_features.items()} # Ensure a copy

            for j in range(args.train.accum_freq):
                accumulated = accumulated_batches[j]
                s1 = accumulated.s1

                is_last_backward_pass = j == args.train.accum_freq - 1
                sync_context = (
                    model.no_sync()
                    if not is_last_backward_pass and hasattr(model, "no_sync")
                    else nullcontext()
                )
                with sync_context:
                    with autocast():
                        if rank0 and not logged_first_backward and j == 0:
                            logging.debug("Rank0: forward(backward) start at accum boundary")
                        
                        model_out = model(*accumulated.model_inputs)
                        if rank0 and not logged_first_backward and j == 0:
                            logging.debug("Rank0: forward(backward) done, computing loss")

                        loss_inputs = build_accumulated_loss_inputs(
                            accum_features_current_loop, model_out, j
                        )

                        if "curv" in model_out:
                            curv_scalar = float(model_out["curv"].detach().item())

                        if hasattr(loss, "set_gather_context"):
                            loss.set_gather_context(
                                {
                                    "epoch": epoch,
                                    "batch": i,
                                    "accum": i_accum,
                                    "micro": j,
                                    "step": step,
                                }
                            )
                        losses, raw_total_loss = compose_training_loss(
                            loss,
                            loss_inputs,
                            reconstruction=model_out.get("recon_loss"),
                            reconstruction_config=getattr(args, "recon", None),
                            epoch=epoch,
                            step=step,
                        )

                    scaled_loss = raw_total_loss / args.train.accum_freq
                    backward_loss(scaled_loss, scaler)

        logit_scale = scalar_output(model_out["logit_scale"])

        step_optimizer(
            model,
            optimizer,
            scaler=scaler,
            grad_clip_norm=args.model.grad_clip_norm,
            horovod=args.datamodule.horovod,
        )

        # reset gradient accum, if enabled
        if args.train.accum_freq > 1:
            accumulated_batches, accum_features = [], {}

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
            loss_param_state = {}
            if hasattr(loss, "get_loggable_state"):
                try:
                    loss_param_state = loss.get_loggable_state()
                except Exception:
                    logging.exception("Failed to read CiipLoss loggable state")
                    loss_param_state = {}

            metrics_log_parts = [
                f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                for loss_name, loss_m in losses_m.items()
            ]
            if vc_metrics_m:
                metrics_log_parts.extend(
                    f"{name}: {meter.val:#.5g} ({meter.avg:#.5g})"
                    for name, meter in vc_metrics_m.items()
                )
            if loss_param_state:
                metrics_log_parts.extend(
                    f"[ParamState]{name}: {value:#.5g}" for name, value in loss_param_state.items()
                )
            if curv_scalar is None and loss_param_state:
                curv_scalar = loss_param_state.get("curvature")
            if curv_scalar is not None:
                metrics_log_parts.append(f"curv: {curv_scalar:#.5g}")
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
            recon_lambda = reconstruction_weight(getattr(args, "recon", None), epoch=epoch, step=step)
            if recon_lambda:
                log_data["recon_lambda"] = recon_lambda
            log_data.update({name:val.val for name,val in losses_m.items()})
            if vc_metrics_m:
                log_data.update({name: meter.val for name, meter in vc_metrics_m.items()})
            if loss_param_state:
                log_data.update({f"loss/{name}": value for name, value in loss_param_state.items()})
            if curv_scalar is not None:
                log_data["curv"] = curv_scalar
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
            for meter in vc_metrics_m.values():
                meter.reset()
        if max_train_steps is not None and (step + 1) >= max_train_steps:
            stop_training = True
            break
    # end for
    return stop_training


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
