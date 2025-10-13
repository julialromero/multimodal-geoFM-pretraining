# CUDA Memory Regressions After e96a0f0

## Key changes increasing peak memory
- Default production config now doubles the projection head to 1024 dims and enables VICReg-style regularization, raising activation and covariance sizes.
- The CIIP forward path now returns both normalized and raw features so the loss can compute the variance/covariance penalty, keeping an extra copy of encoder outputs resident on GPU each step.
- The CIIP loss adds variance/covariance regularizers that build `D x D` covariance matrices when enabled, which is especially costly after the embedding dimension jump.

## Likely impact
These configuration + model changes raise per-batch memory substantially versus commit e96a0f0 and can push previously stable runs into OOM territory even though the training loop itself became more OOM-resilient.
