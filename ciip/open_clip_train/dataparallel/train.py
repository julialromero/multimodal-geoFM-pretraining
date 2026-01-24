import logging
import math
import time
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from open_clip import get_input_dtype

from ciip.open_clip_train.distributed import is_master
from ciip.open_clip_train.precision import get_autocast
from ciip.open_clip_train.train import (
    AverageMeter,
    _accumulate_vc_metrics,
    backward,
    unwrap_model,
)


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    """DataParallel-friendly training loop (no no_sync usage)."""
    device = torch.device(args.datamodule.device)
    autocast = get_autocast(args.model.precision)
    input_dtype = get_input_dtype(args.model.precision)
    model_cfg = getattr(args, "model", None)
    encoder_pair = getattr(model_cfg, "encoder_pair", "s1s2")
    triple_modal = encoder_pair == "s1s2_text"

    model.train()
    if args.distill:
        dist_model.eval()

    data["train"].set_epoch(epoch)
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.train.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.train.accum_freq > 1:
        accum_s1, accum_s2, accum_features = [], [], {}
        accum_text = [] if triple_modal else None

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

        text = None
        if triple_modal:
            s1 = batch["s1"].to(device=device, dtype=input_dtype, non_blocking=True)
            s2 = batch["s2"].to(device=device, dtype=input_dtype, non_blocking=True)
            text = batch["text"].to(device=device, non_blocking=True)
        elif encoder_pair == "s2_text":
            s1 = batch["s2"].to(device=device, dtype=input_dtype, non_blocking=True)
            s2 = batch["text"].to(device=device, non_blocking=True)
        else:
            s1, s2 = batch["s1"], batch["s2"]
            s1 = s1.to(device=device, dtype=input_dtype, non_blocking=True)
            s2 = s2.to(device=device, dtype=input_dtype, non_blocking=True)
        # print(args.model.encoder_pair)
        # print(f'Batch loaded: s1 shape: {s1.shape}, s2 shape: {s2.shape}')

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        curv_scalar: Optional[float] = None

        if args.train.accum_freq == 1:
            with autocast():
                if triple_modal:
                    model_out = model(s1, s2, text)
                else:
                    model_out = model(s1, s2)

                logit_scale = model_out["logit_scale"]
                if isinstance(logit_scale, torch.Tensor) and logit_scale.ndim > 0:
                    logit_scale = logit_scale.mean()
                model_out["logit_scale"] = logit_scale
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(s1, s2)
                    model_out.update({f"dist_{k}": v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            if "curv" in model_out:
                curv_scalar = float(model_out["curv"].detach().item())

            _accumulate_vc_metrics(loss, model_out, vc_metrics_m, s1.shape[0])
            backward(total_loss, scaler)
        else:
            with torch.no_grad():
                with autocast():
                    if triple_modal:
                        model_out = model(s1, s2, text)
                    else:
                        model_out = model(s1, s2)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    _accumulate_vc_metrics(loss, model_out, vc_metrics_m, s1.shape[0])

                    for key, val in model_out.items():
                        if val.ndim == 0:
                            continue
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_s1.append(s1)
                accum_s2.append(s2)
                if triple_modal:
                    accum_text.append(text)

            if ((i + 1) % args.train.accum_freq) > 0:
                continue

            optimizer.zero_grad()
            accum_features_current_loop = {key: list(val) for key, val in accum_features.items()}

            for j in range(args.train.accum_freq):
                s1 = accum_s1[j]
                s2 = accum_s2[j]
                text_j = accum_text[j] if triple_modal else None

                with nullcontext():
                    with autocast():
                        if triple_modal:
                            model_out = model(s1, s2, text_j)
                        else:
                            model_out = model(s1, s2)
                        # print(model_out.keys())
                        # print(f"out: {model_out['s1_features'].shape}, {model_out['s2_features'].shape}")
                        # print(f'logit scale shape: {model_out["logit_scale"].shape}')

                        inputs_no_accum = {}
                        logit_scale = model_out.pop("logit_scale")
                        if isinstance(logit_scale, torch.Tensor) and logit_scale.ndim > 0:
                            logit_scale = logit_scale.mean()
                        inputs_no_accum["logit_scale"] = logit_scale
                        if "logit_bias" in model_out:
                            logit_bias = model_out.pop("logit_bias")
                            if isinstance(logit_bias, torch.Tensor) and logit_bias.ndim > 0:
                                logit_bias = logit_bias.mean()
                            inputs_no_accum["logit_bias"] = logit_bias

                        inputs = {}
                        for key in accum_features_current_loop.keys():
                            temp_accumulated = list(accum_features_current_loop[key])
                            temp_accumulated[j] = model_out[key]
                            inputs[key] = torch.cat(temp_accumulated)

                        for key, val in model_out.items():
                            if key in inputs or key in inputs_no_accum:
                                continue
                            inputs[key] = val

                        if "curv" in model_out:
                            curv_scalar = float(model_out["curv"].detach().item())

                        losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                        raw_total_loss = sum(losses.values())
                        losses["loss"] = raw_total_loss

                    scaled_loss = raw_total_loss / args.train.accum_freq
                    backward(scaled_loss, scaler)

                del inputs
                del inputs_no_accum

        if scaler is not None:
            if args.model.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.model.grad_clip_norm, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.model.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.model.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        if args.train.accum_freq > 1:
            accum_s1, accum_s2, accum_features = [], [], {}
            if triple_modal:
                accum_text = []

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
            samples_per_second = (
                args.train.accum_freq * args.datamodule.batch_size * args.datamodule.world_size / batch_time_m.val
            )
            samples_per_second_per_gpu = args.train.accum_freq * args.datamodule.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"Loss: {metrics_log}"
            )


__all__ = ["train_one_epoch"]
