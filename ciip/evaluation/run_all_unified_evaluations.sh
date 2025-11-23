#!/usr/bin/env bash
set -euo pipefail

# Batch runner for unified_evaluation.py across the supported model variants.
# Each invocation is executed sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON:-python}

# Optional: set CROMA_WEIGHTS to the path of the pretrained CROMA checkpoint
# before running this script. The script will error if it is required but
# not provided.
CROMA_WEIGHTS_PATH=${CROMA_WEIGHTS:-}

run_eval() {
  local description=$1
  shift
  echo "=== Running unified_evaluation.py for ${description} ==="
  "${PYTHON_BIN}" "${SCRIPT_DIR}/unified_evaluation.py" "$@"
}

# Backbone-only models (Sentinel-2 encoders only)
run_eval "RCF_13ch" \
  --model-type backbone_only \
  --model-weights rcf_13ch \
  --model-in-channels 13

run_eval "DOFA_base_S2_13ch" \
  --model-type backbone_only \
  --model-weights dofa_base_s2_13ch \
  --model-in-channels 13

run_eval "ScaleMAE_large_RGB" \
  --model-type backbone_only \
  --model-weights scalemae_large_rgb \
  --model-in-channels 3

run_eval "ResNet18_S2_ALL_MOCO" \
  --model-type backbone_only \
  --model-weights resnet18_s2_all_moco \
  --model-in-channels 13

run_eval "ResNet18_S2_RGB_MOCO" \
  --model-type backbone_only \
  --model-weights resnet18_s2_rgb_moco \
  --model-in-channels 3

run_eval "ResNet50_S2_RGB_MOCO" \
  --model-type backbone_only \
  --model-weights resnet50_s2_rgb_moco \
  --model-in-channels 3

run_eval "ResNet152_ImageNet_RGB" \
  --model-type backbone_only \
  --model-weights resnet152_imagenet_rgb \
  --model-in-channels 3

run_eval "ViTSmall16_S2_ALL_MOCO" \
  --model-type backbone_only \
  --model-weights vitsmall16_s2_all_moco \
  --model-in-channels 13

# TorchGeo ResNet50 baselines
run_eval "TorchGeo ResNet50 MoCo" \
  --model-type torchgeo_resnet50 \
  --model-weights moco \
  --model-in-channels 13

run_eval "TorchGeo ResNet50 DINO" \
  --model-type torchgeo_resnet50 \
  --model-weights dino \
  --model-in-channels 13

# CROMA model (requires a weights path)
if [[ -z "${CROMA_WEIGHTS_PATH}" ]]; then
  echo "CROMA_WEIGHTS environment variable is not set; skipping CROMA run." >&2
else
  run_eval "CROMA" \
    --model-type croma \
    --croma-weights "${CROMA_WEIGHTS_PATH}"
fi

# CIIP checkpoint (defaults are resolved inside unified_evaluation.py)
run_eval "CIIP checkpoint" \
  --model-type ciip_checkpoint
