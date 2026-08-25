import logging
import math
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# import lorentz as L
from . import lorentz as L


def _setup_gather_logger() -> logging.Logger:
    logger = logging.getLogger("ciip.gather_debug")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


_GATHER_LOG = _setup_gather_logger()
_GATHER_CONTEXT: Dict[str, int] = {}


def _set_gather_context(ctx: Optional[Dict[str, int]]) -> None:
    if ctx is None:
        _GATHER_CONTEXT.clear()
        return
    _GATHER_CONTEXT.clear()
    _GATHER_CONTEXT.update(ctx)


def gather_features(
        s1_features,
        s2_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    debug_first = int(os.environ.get("CIIP_GATHER_DEBUG_FIRST", "5"))
    debug_every = int(os.environ.get("CIIP_GATHER_DEBUG_EVERY", "50"))
    call_idx = getattr(gather_features, "_debug_counter", 0) + 1
    setattr(gather_features, "_debug_counter", call_idx)
    do_log = (
        rank == 0
        and (call_idx <= debug_first or (debug_every > 0 and call_idx % debug_every == 0))
    )
    ctx = _GATHER_CONTEXT if _GATHER_CONTEXT else {}
    ctx_epoch = ctx.get("epoch")
    ctx_batch = ctx.get("batch")
    ctx_accum = ctx.get("accum")
    ctx_micro = ctx.get("micro")
    ctx_step = ctx.get("step")

    if do_log:
        _GATHER_LOG.info(
            "Rank0 gather_features enter call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s s1_shape=%s s2_shape=%s local_loss=%s gather_with_grad=%s world_size=%s dist_init=%s",
            call_idx,
            ctx_epoch,
            ctx_batch,
            ctx_accum,
            ctx_micro,
            ctx_step,
            tuple(s1_features.shape),
            tuple(s2_features.shape),
            local_loss,
            gather_with_grad,
            world_size,
            dist.is_initialized() if has_distributed else False,
        )
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_s1_features = hvd.allgather(s1_features)
            all_s2_features = hvd.allgather(s2_features)
        else:
            with torch.no_grad():
                all_s1_features = hvd.allgather(s1_features)
                all_s2_features = hvd.allgather(s2_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_s1_features = list(all_s1_features.chunk(world_size, dim=0))
                gathered_s2_features = list(all_s2_features.chunk(world_size, dim=0))
                gathered_s1_features[rank] = s1_features
                gathered_s2_features[rank] = s2_features
                all_s1_features = torch.cat(gathered_s1_features, dim=0)
                all_s2_features = torch.cat(gathered_s2_features, dim=0)
    else:
        # We gather tensors from all gpus
        bs = torch.tensor([s1_features.shape[0]], device=s1_features.device)
        if do_log:
            _GATHER_LOG.info(
                "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s before all_reduce MIN",
                call_idx,
                ctx_epoch,
                ctx_batch,
                ctx_accum,
                ctx_micro,
                ctx_step,
            )
        try:
            torch.distributed.all_reduce(bs, op=torch.distributed.ReduceOp.MIN)
        except Exception:
            if rank == 0:
                _GATHER_LOG.exception(
                    "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s all_reduce MIN failed",
                    call_idx,
                    ctx_epoch,
                    ctx_batch,
                    ctx_accum,
                    ctx_micro,
                    ctx_step,
                )
            raise
        bs_min = int(bs.item())
        if do_log:
            _GATHER_LOG.info(
                "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s after all_reduce MIN bs_min=%s",
                call_idx,
                ctx_epoch,
                ctx_batch,
                ctx_accum,
                ctx_micro,
                ctx_step,
                bs_min,
            )

        bs2 = torch.tensor([s1_features.shape[0]], device=s1_features.device)
        if do_log:
            _GATHER_LOG.info(
                "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s before all_reduce MAX",
                call_idx,
                ctx_epoch,
                ctx_batch,
                ctx_accum,
                ctx_micro,
                ctx_step,
            )
        try:
            torch.distributed.all_reduce(bs2, op=torch.distributed.ReduceOp.MAX)
        except Exception:
            if rank == 0:
                _GATHER_LOG.exception(
                    "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s all_reduce MAX failed",
                    call_idx,
                    ctx_epoch,
                    ctx_batch,
                    ctx_accum,
                    ctx_micro,
                    ctx_step,
                )
            raise
        bs_max = int(bs2.item())
        if do_log:
            _GATHER_LOG.info(
                "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s after all_reduce MAX bs_max=%s",
                call_idx,
                ctx_epoch,
                ctx_batch,
                ctx_accum,
                ctx_micro,
                ctx_step,
                bs_max,
            )

        if bs_min != bs_max:
            if rank == 0:
                _GATHER_LOG.error(
                    "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s unequal batch sizes across ranks: min=%s max=%s",
                    call_idx,
                    ctx_epoch,
                    ctx_batch,
                    ctx_accum,
                    ctx_micro,
                    ctx_step,
                    bs_min,
                    bs_max,
                )
            raise RuntimeError(f"Unequal batch sizes across ranks: min={bs_min}, max={bs_max}")


        if gather_with_grad:
            if do_log:
                _GATHER_LOG.info(
                    "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s before all_gather (with grad)",
                    call_idx,
                    ctx_epoch,
                    ctx_batch,
                    ctx_accum,
                    ctx_micro,
                    ctx_step,
                )
            all_s1_features = torch.cat(torch.distributed.nn.all_gather(s1_features), dim=0)
            all_s2_features = torch.cat(torch.distributed.nn.all_gather(s2_features), dim=0)
            if do_log:
                _GATHER_LOG.info(
                    "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s after all_gather (with grad)",
                    call_idx,
                    ctx_epoch,
                    ctx_batch,
                    ctx_accum,
                    ctx_micro,
                    ctx_step,
                )
        else:
            gathered_s1_features = [torch.zeros_like(s1_features) for _ in range(world_size)]
            gathered_s2_features = [torch.zeros_like(s2_features) for _ in range(world_size)]
            if do_log:
                _GATHER_LOG.info(
                    "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s before all_gather (no grad)",
                    call_idx,
                    ctx_epoch,
                    ctx_batch,
                    ctx_accum,
                    ctx_micro,
                    ctx_step,
                )
            dist.all_gather(gathered_s1_features, s1_features)
            dist.all_gather(gathered_s2_features, s2_features)
            if do_log:
                _GATHER_LOG.info(
                    "Rank0 gather_features call=%s epoch=%s batch=%s accum=%s micro=%s train_step=%s after all_gather (no grad)",
                    call_idx,
                    ctx_epoch,
                    ctx_batch,
                    ctx_accum,
                    ctx_micro,
                    ctx_step,
                )
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_s1_features[rank] = s1_features
                gathered_s2_features[rank] = s2_features
            all_s1_features = torch.cat(gathered_s1_features, dim=0)
            all_s2_features = torch.cat(gathered_s2_features, dim=0)

    return all_s1_features, all_s2_features


class CiipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            contrastive_weight=1.0,
            vc_reg_enabled=False,
            vc_weight=0.0,
            vc_gamma=1.0,
            vc_covariance_weights=None,
            batch_uniformity_enabled=True,
            batch_uniformity_weight=0.05,
            hyperbolic=True,
            hyperbolic_normalize=False,
            hyperbolic_curvature_init=1e-2,
            hyperbolic_eps=1e-5,
            matryoshka_enabled=False,
            matryoshka_weight=1.0,
            matryoshka_dims=None,
            matryoshka_relative_weights=None,
            matryoshka_normalize=True,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.contrastive_weight = float(contrastive_weight)

        self.vc_reg_enabled = bool(vc_reg_enabled)
        self.vc_weight = float(vc_weight)
        self.vc_gamma = float(vc_gamma)

        if vc_covariance_weights is None:
            self.vc_covariance_weights = (1.0, 1.0)
        else:
            if isinstance(vc_covariance_weights, torch.Tensor):
                vc_covariance_weights = vc_covariance_weights.tolist()
            if not isinstance(vc_covariance_weights, (list, tuple)):
                vc_covariance_weights = [float(vc_covariance_weights)]
            if len(vc_covariance_weights) == 1:
                vc_covariance_weights = [vc_covariance_weights[0], vc_covariance_weights[0]]
            elif len(vc_covariance_weights) >= 2:
                vc_covariance_weights = vc_covariance_weights[:2]
            self.vc_covariance_weights = tuple(float(w) for w in vc_covariance_weights)

        self.batch_uniformity_enabled = bool(batch_uniformity_enabled)
        self.batch_uniformity_weight = float(batch_uniformity_weight)

        self.use_hyperbolic = bool(hyperbolic)
        # self.hyperbolic_normalize = bool(hyperbolic_normalize)
        # self.hyperbolic_eps = float(hyperbolic_eps)
        # curvature_init = max(float(hyperbolic_curvature_init), self.hyperbolic_eps)

        # # Learnable positive scale on Euclidean features BEFORE Lorentz lift
        # # (stabilizes early training without destroying radial info)
        # self.hyp_scale = nn.Parameter(torch.tensor(0.5))
        # curvature_alpha = math.log(math.expm1(curvature_init))
        # self.curvature_alpha = nn.Parameter(torch.tensor(curvature_alpha))

        self.matryoshka_enabled = bool(matryoshka_enabled)
        self.matryoshka_weight = float(matryoshka_weight)
        self.matryoshka_normalize = bool(matryoshka_normalize)
        self._matryoshka_dims = matryoshka_dims
        self._matryoshka_relative_weights = matryoshka_relative_weights

    def set_gather_context(self, ctx: Optional[Dict[str, int]]) -> None:
        _set_gather_context(ctx)
        self.matryoshka_dims = self._normalize_matryoshka_dims(self._matryoshka_dims)
        self.matryoshka_dim_weights = self._normalize_matryoshka_weights(
            self._matryoshka_relative_weights,
            len(self.matryoshka_dims),
        )

        # cache state
        self.prev_num_logits = 0
        self.labels = {}


    def _variance_regularizer(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] <= 1:
            return torch.zeros((), device=features.device, dtype=features.dtype)
        variances = torch.var(features, dim=0, unbiased=False)
        std = torch.sqrt(variances + 1e-4)
        return torch.mean(F.relu(self.vc_gamma - std))

    def _covariance_regularizer(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] <= 1:
            return torch.zeros((), device=features.device, dtype=features.dtype)
        features = features - features.mean(dim=0)
        cov = features.T @ features / (features.shape[0] - 1)
        cov = cov - torch.diag(torch.diagonal(cov))
        return cov.pow(2).sum() / cov.shape[0]

    def _batch_uniformity_loss(
        self,
        features: torch.Tensor,
        alpha: float = 2.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] <= 1:
            return torch.zeros((), device=features.device, dtype=features.dtype)

        z = F.normalize(features, dim=1)
        sim = torch.matmul(z, z.t())
        sq_dist = 2.0 * (1.0 - sim)

        mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        vals = torch.exp(-alpha * sq_dist[mask])
        loss = torch.log(vals.mean() + eps)
        return loss

    @staticmethod
    def _to_iterable(value):
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, (int, float)):
            return [value]
        if isinstance(value, dict):
            raise ValueError("Matryoshka parameters must be a list of scalars")
        if isinstance(value, str):
            raise ValueError("Matryoshka parameters must be a list of scalars")
        if hasattr(value, "__iter__"):
            return list(value)
        return [value]

    def _normalize_matryoshka_dims(self, dims):
        raw = self._to_iterable(dims)
        normalized: List[int] = []
        for dim in raw:
            try:
                dim = int(dim)
            except (TypeError, ValueError):
                continue
            if dim <= 0:
                continue
            normalized.append(dim)
        normalized = sorted(set(normalized))
        return tuple(normalized)

    def _normalize_matryoshka_weights(self, weights, dim_count):
        if dim_count <= 0:
            return tuple()
        raw = self._to_iterable(weights)
        if not raw:
            raw = [1.0]
        normalized = [float(w) for w in raw]
        if len(normalized) == 1:
            normalized = normalized * dim_count
        if len(normalized) != dim_count:
            raise ValueError(
                "Number of Matryoshka weights must match the number of nestings"
            )
        return tuple(normalized)

    def _should_apply_matryoshka(self):
        return (
            self.matryoshka_enabled
            and self.matryoshka_weight != 0
            and len(self.matryoshka_dims) > 0
        )

    def _compute_matryoshka_loss(
        self,
        anchor_s1: torch.Tensor,
        anchor_s2: torch.Tensor,
        all_s1: torch.Tensor,
        all_s2: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if (
            anchor_s1 is None
            or anchor_s2 is None
            or all_s1 is None
            or all_s2 is None
        ):
            if torch.is_tensor(logit_scale):
                zero_device = logit_scale.device
                zero_dtype = logit_scale.dtype
            else:
                zero_device = anchor_s1.device if anchor_s1 is not None else torch.device("cpu")
                zero_dtype = anchor_s1.dtype if anchor_s1 is not None else torch.float32
            return torch.zeros((), device=zero_device, dtype=zero_dtype), {}

        max_dim = min(
            anchor_s1.shape[1],
            anchor_s2.shape[1],
            all_s1.shape[1],
            all_s2.shape[1],
        )
        applicable = [
            (dim, weight)
            for dim, weight in zip(self.matryoshka_dims, self.matryoshka_dim_weights)
            if 0 < dim <= max_dim
        ]
        if not applicable:
            return (
                torch.zeros((), device=anchor_s1.device, dtype=anchor_s1.dtype),
                {},
            )

        labels = self.get_ground_truth(anchor_s1.device, anchor_s1.shape[0])
        mat_loss = torch.zeros((), device=anchor_s1.device, dtype=anchor_s1.dtype)
        per_dim_losses: Dict[int, torch.Tensor] = {}
        for dim, weight in applicable:
            anchor_s1_slice = anchor_s1[:, :dim]
            anchor_s2_slice = anchor_s2[:, :dim]
            all_s1_slice = all_s1[:, :dim]
            all_s2_slice = all_s2[:, :dim]
            if self.matryoshka_normalize:
                anchor_s1_slice = F.normalize(anchor_s1_slice, dim=1, eps=1e-12)
                anchor_s2_slice = F.normalize(anchor_s2_slice, dim=1, eps=1e-12)
                all_s1_slice = F.normalize(all_s1_slice, dim=1, eps=1e-12)
                all_s2_slice = F.normalize(all_s2_slice, dim=1, eps=1e-12)

            logits_s1 = logit_scale * anchor_s1_slice @ all_s2_slice.T
            logits_s2 = logit_scale * anchor_s2_slice @ all_s1_slice.T
            if logit_bias is not None:
                logits_s1 = logits_s1 + logit_bias
                logits_s2 = logits_s2 + logit_bias

            per_dim = (
                F.cross_entropy(logits_s1, labels)
                + F.cross_entropy(logits_s2, labels)
            ) / 2
            weighted_per_dim = float(weight) * per_dim
            mat_loss = mat_loss + weighted_per_dim
            per_dim_losses[dim] = weighted_per_dim
        mat_loss = mat_loss * self.matryoshka_weight
        dim_losses = {
            f"matryoshka_dim_{dim}": loss * self.matryoshka_weight
            for dim, loss in per_dim_losses.items()
        }
        return mat_loss, dim_losses

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(
            self,
            s1_features,
            s2_features,
            logit_scale,
            logit_bias=None,
            return_gathered=False,
            curv=None,
    ):
        with torch.cuda.amp.autocast(enabled=False):
            if self.use_hyperbolic:
                assert curv is not None, "curv must be provided for hyperbolic logits"
                return self._get_hyperbolic_logits(
                    s1_features,
                    s2_features,
                    logit_scale,
                    logit_bias=logit_bias,
                    return_gathered=return_gathered,
                    curv=curv,
                )

        if self.world_size > 1:
            all_s1_features, all_s2_features = gather_features(
                s1_features, s2_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_s1 = logit_scale * s1_features @ all_s2_features.T
                logits_per_s2 = logit_scale * s2_features @ all_s1_features.T
            else:
                logits_per_s1 = logit_scale * all_s1_features @ all_s2_features.T
                logits_per_s2 = logits_per_s1.T
        else:
            logits_per_s1 = logit_scale * s1_features @ s2_features.T
            logits_per_s2 = logit_scale * s2_features @ s1_features.T
            all_s1_features = s1_features
            all_s2_features = s2_features

        if logit_bias is not None:
            logits_per_s1 = logits_per_s1 + logit_bias
            logits_per_s2 = logits_per_s2 + logit_bias

        if return_gathered:
            return logits_per_s1, logits_per_s2, all_s1_features, all_s2_features

        return logits_per_s1, logits_per_s2

    def forward(
            self,
            s1_features,
            s2_features,
            logit_scale,
        text_features=None,
        logit_bias=None,
        output_dict=False,
        s1_features_vc=None,
        s2_features_vc=None,
        compute_contrastive=True,
        **kwargs,
    ):
        if text_features is not None:
            return self._forward_three_modalities(
                s1_features=s1_features,
                s2_features=s2_features,
                text_features=text_features,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                output_dict=output_dict,
                s1_features_vc=s1_features_vc,
                s2_features_vc=s2_features_vc,
            )

        device = s1_features.device
        need_vc = self.vc_reg_enabled and self.vc_weight != 0
        need_batch_uniformity = self.batch_uniformity_enabled and self.batch_uniformity_weight != 0
        gather_for_vc = need_vc and self.world_size > 1
        use_matryoshka = self._should_apply_matryoshka()
        if use_matryoshka and self.use_hyperbolic:
            raise NotImplementedError("Matryoshka loss is not supported for hyperbolic embeddings")
        return_gathered = gather_for_vc or use_matryoshka
        if use_matryoshka:
            compute_contrastive = False

        contrastive_loss = torch.zeros((), device=device, dtype=s1_features.dtype)
        gathered_s1_features = None
        gathered_s2_features = None
        matryoshka_loss = torch.zeros((), device=device, dtype=s1_features.dtype)
        matryoshka_logging: Dict[str, torch.Tensor] = {}

        if compute_contrastive:
            logits_outputs = self.get_logits(
                s1_features,
                s2_features,
                logit_scale,
                logit_bias=logit_bias,
                return_gathered=return_gathered,
                curv=kwargs.get("curv", None)
            )
            if self.use_hyperbolic:
                if gather_for_vc:
                    raise NotImplementedError("gather_for_vc not implemented for hyperbolic loss yet")

                s1_logits, s2_logits, targets = logits_outputs
                gathered_s1_features = None
                gathered_s2_features = None

                contrastive_loss = 0.5 * (
                    F.cross_entropy(logit_scale * s1_logits, targets)
                    + F.cross_entropy(logit_scale * s2_logits, targets)
                )
            else:
                if return_gathered:
                    logits_per_s1, logits_per_s2, gathered_s1_features, gathered_s2_features = logits_outputs
                else:
                    logits_per_s1, logits_per_s2 = logits_outputs
                    gathered_s1_features = None
                    gathered_s2_features = None

                if use_matryoshka:
                    if self.world_size > 1 and self.local_loss:
                        anchor_s1 = s1_features
                        anchor_s2 = s2_features
                    else:
                        anchor_s1 = gathered_s1_features if gathered_s1_features is not None else s1_features
                        anchor_s2 = gathered_s2_features if gathered_s2_features is not None else s2_features
                    all_s1 = gathered_s1_features if gathered_s1_features is not None else s1_features
                    all_s2 = gathered_s2_features if gathered_s2_features is not None else s2_features
                    matryoshka_loss, matryoshka_logging = self._compute_matryoshka_loss(
                        anchor_s1,
                        anchor_s2,
                        all_s1,
                        all_s2,
                        logit_scale,
                        logit_bias,
                    )
                    contrastive_loss = matryoshka_loss
                else:
                    labels = self.get_ground_truth(device, logits_per_s1.shape[0])
                    contrastive_loss = (
                        F.cross_entropy(logits_per_s1, labels) +
                        F.cross_entropy(logits_per_s2, labels)
                    ) / 2
        else:
            if use_matryoshka or gather_for_vc:
                if self.world_size > 1:
                    gathered_s1_features, gathered_s2_features = gather_features(
                        s1_features,
                        s2_features,
                        self.local_loss,
                        self.gather_with_grad,
                        self.rank,
                        self.world_size,
                        self.use_horovod,
                    )
                elif use_matryoshka:
                    gathered_s1_features = s1_features
                    gathered_s2_features = s2_features
            if use_matryoshka:
                if self.world_size > 1 and self.local_loss:
                    anchor_s1 = s1_features
                    anchor_s2 = s2_features
                else:
                    anchor_s1 = gathered_s1_features if gathered_s1_features is not None else s1_features
                    anchor_s2 = gathered_s2_features if gathered_s2_features is not None else s2_features
                all_s1 = gathered_s1_features if gathered_s1_features is not None else s1_features
                all_s2 = gathered_s2_features if gathered_s2_features is not None else s2_features
                matryoshka_loss, matryoshka_logging = self._compute_matryoshka_loss(
                    anchor_s1,
                    anchor_s2,
                    all_s1,
                    all_s2,
                    logit_scale,
                    logit_bias,
                )
                contrastive_loss = matryoshka_loss

        weighted_contrastive_loss = self.contrastive_weight * contrastive_loss

        losses = {}
        if output_dict or self.contrastive_weight != 0:
            losses["contrastive_loss"] = weighted_contrastive_loss

        if use_matryoshka:
            losses["matryoshka_loss"] = matryoshka_loss
            losses.update(matryoshka_logging)



        if need_vc:
            s1_vc_local = s1_features_vc if s1_features_vc is not None else s1_features
            s2_vc_local = s2_features_vc if s2_features_vc is not None else s2_features

            if self.world_size > 1:
                reuse_info_nce_gather = (
                    gathered_s1_features is not None and gathered_s2_features is not None
                )
                if reuse_info_nce_gather:
                    s1_vc = gathered_s1_features
                    s2_vc = gathered_s2_features
                else:
                    s1_vc, s2_vc = gather_features(
                        s1_vc_local,
                        s2_vc_local,
                        self.local_loss,
                        True,
                        self.rank,
                        self.world_size,
                        self.use_horovod,
                    )
            else:
                s1_vc = s1_vc_local
                s2_vc = s2_vc_local

            variance_loss = self._variance_regularizer(s1_vc) + self._variance_regularizer(s2_vc)
            cov_w_s1, cov_w_s2 = self.vc_covariance_weights
            covariance_loss = (
                cov_w_s1 * self._covariance_regularizer(s1_vc) +
                cov_w_s2 * self._covariance_regularizer(s2_vc)
            )
            vc_loss = self.vc_weight * (variance_loss + covariance_loss)
            losses["vc_loss"] = vc_loss

        if need_batch_uniformity:
            bu_loss = (
                self._batch_uniformity_loss(s1_features)
                + self._batch_uniformity_loss(s2_features)
            )
            bu_loss = self.batch_uniformity_weight * bu_loss
            losses["batch_uniformity_loss"] = bu_loss

        total_loss = weighted_contrastive_loss
        if need_vc:
            total_loss = total_loss + vc_loss
        if need_batch_uniformity:
            total_loss = total_loss + bu_loss
        return losses if output_dict else total_loss

    def _gather_single_feature(self, feature: torch.Tensor) -> torch.Tensor:
        assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
        if self.world_size <= 1:
            return feature

        if self.use_horovod:
            assert hvd is not None, 'Please install horovod'
            if self.gather_with_grad:
                all_feature = hvd.allgather(feature)
            else:
                with torch.no_grad():
                    all_feature = hvd.allgather(feature)
                if not self.local_loss:
                    gathered = list(all_feature.chunk(self.world_size, dim=0))
                    gathered[self.rank] = feature
                    all_feature = torch.cat(gathered, dim=0)
            return all_feature

        if self.gather_with_grad:
            return torch.cat(torch.distributed.nn.all_gather(feature), dim=0)

        gathered = [torch.zeros_like(feature) for _ in range(self.world_size)]
        dist.all_gather(gathered, feature)
        if not self.local_loss:
            gathered[self.rank] = feature
        return torch.cat(gathered, dim=0)

    def _forward_three_modalities(
        self,
        *,
        s1_features: torch.Tensor,
        s2_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        output_dict: bool = False,
        s1_features_vc: Optional[torch.Tensor] = None,
        s2_features_vc: Optional[torch.Tensor] = None,
    ):
        device = s1_features.device
        need_vc = self.vc_reg_enabled and self.vc_weight != 0
        need_batch_uniformity = self.batch_uniformity_enabled and self.batch_uniformity_weight != 0

        if self.world_size > 1:
            all_s1_features = self._gather_single_feature(s1_features)
            all_s2_features = self._gather_single_feature(s2_features)
            all_text_features = self._gather_single_feature(text_features)
        else:
            all_s1_features = s1_features
            all_s2_features = s2_features
            all_text_features = text_features

        if self.world_size > 1 and self.local_loss:
            anchor_s1 = s1_features
            anchor_s2 = s2_features
            anchor_text = text_features
        else:
            anchor_s1 = all_s1_features
            anchor_s2 = all_s2_features
            anchor_text = all_text_features

        logits_s1_s2 = logit_scale * anchor_s1 @ all_s2_features.T
        logits_s2_s1 = logit_scale * anchor_s2 @ all_s1_features.T
        logits_s1_txt = logit_scale * anchor_s1 @ all_text_features.T
        logits_txt_s1 = logit_scale * anchor_text @ all_s1_features.T
        logits_s2_txt = logit_scale * anchor_s2 @ all_text_features.T
        logits_txt_s2 = logit_scale * anchor_text @ all_s2_features.T

        if logit_bias is not None:
            logits_s1_s2 = logits_s1_s2 + logit_bias
            logits_s2_s1 = logits_s2_s1 + logit_bias
            logits_s1_txt = logits_s1_txt + logit_bias
            logits_txt_s1 = logits_txt_s1 + logit_bias
            logits_s2_txt = logits_s2_txt + logit_bias
            logits_txt_s2 = logits_txt_s2 + logit_bias

        labels = self.get_ground_truth(device, logits_s1_s2.shape[0])
        loss_s1_s2 = (F.cross_entropy(logits_s1_s2, labels) + F.cross_entropy(logits_s2_s1, labels)) / 2
        loss_s1_txt = (F.cross_entropy(logits_s1_txt, labels) + F.cross_entropy(logits_txt_s1, labels)) / 2
        loss_s2_txt = (F.cross_entropy(logits_s2_txt, labels) + F.cross_entropy(logits_txt_s2, labels)) / 2

        contrastive_loss = (loss_s1_s2 + loss_s1_txt + loss_s2_txt) / 3
        weighted_contrastive_loss = self.contrastive_weight * contrastive_loss

        losses = {"contrastive_loss": weighted_contrastive_loss} if output_dict or self.contrastive_weight != 0 else {}

        if need_vc:
            s1_vc_local = s1_features_vc if s1_features_vc is not None else s1_features
            s2_vc_local = s2_features_vc if s2_features_vc is not None else s2_features
            if self.world_size > 1:
                s1_vc = all_s1_features
                s2_vc = all_s2_features
            else:
                s1_vc = s1_vc_local
                s2_vc = s2_vc_local

            variance_loss = self._variance_regularizer(s1_vc) + self._variance_regularizer(s2_vc)
            cov_w_s1, cov_w_s2 = self.vc_covariance_weights
            covariance_loss = (
                cov_w_s1 * self._covariance_regularizer(s1_vc) +
                cov_w_s2 * self._covariance_regularizer(s2_vc)
            )
            vc_loss = self.vc_weight * (variance_loss + covariance_loss)
            losses["vc_loss"] = vc_loss

        if need_batch_uniformity:
            bu_loss = (
                self._batch_uniformity_loss(s1_features)
                + self._batch_uniformity_loss(s2_features)
            )
            bu_loss = self.batch_uniformity_weight * bu_loss
            losses["batch_uniformity_loss"] = bu_loss

        total_loss = sum(losses.values())
        return losses if output_dict else total_loss


    def _get_hyperbolic_logits(
        self,
        s1_features: torch.Tensor,
        s2_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias=None,
        return_gathered=False,
        curv=None,
    ):
        return self._get_hyperbolic_logits_atmg(
            s1_features,
            s2_features,
            logit_scale,
            logit_bias=logit_bias,
            return_gathered=return_gathered,
            curv=curv,
        )

    def _get_hyperbolic_logits_atmg(
        self,
        s1_features: torch.Tensor,
        s2_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias=None,
        return_gathered=False,
        curv=None
    ):
        if curv is None:
            raise ValueError("curv must be provided for hyperbolic logits")

        if torch.is_tensor(curv):
            curv = curv.to(device=s1_features.device, dtype=s1_features.dtype)
        else:
            curv = torch.tensor(curv, device=s1_features.device, dtype=s1_features.dtype)

        if self.world_size > 1:
            all_s1_features, all_s2_features = gather_features(
                s1_features,
                s2_features,
                local_loss=False,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
        else:
            all_s1_features = s1_features
            all_s2_features = s2_features

        device_type = s1_features.device.type
        autocast_enabled = device_type in {"cuda", "cpu", "xpu", "mps"}
        autocast_context = (
            torch.autocast(device_type=device_type, dtype=torch.float32)
            if autocast_enabled
            else nullcontext()
        )

        anchor_s1 = s1_features if self.local_loss or self.world_size > 1 else all_s1_features
        anchor_s2 = s2_features if self.local_loss or self.world_size > 1 else all_s2_features

        with autocast_context:
            s1_logits = -L.pairwise_oxy_angle(anchor_s1, all_s2_features, curv)
            s2_logits = L.pairwise_oxy_angle(anchor_s2, all_s1_features, curv)

        if logit_bias is not None:
            s1_logits = s1_logits + logit_bias
            s2_logits = s2_logits + logit_bias

        batch_size = s1_features.shape[0]
        targets = torch.arange(batch_size, device=s1_features.device, dtype=torch.long)
        if self.world_size > 1:
            targets = targets + batch_size * self.rank

        return s1_logits, s2_logits, targets

      


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, s1_features, s2_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(s1_features, s2_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            text_features_to_right = s2_features
            for _ in range(self.world_size - 1):
                text_features_from_left = neighbour_exchange_with_grad(
                    left_rank, right_rank, text_features_to_right
                )
                loss += self._loss(
                    s1_features,
                    text_features_from_left,
                    logit_scale,
                    logit_bias,
                    negative_only=True,
                )
                text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
