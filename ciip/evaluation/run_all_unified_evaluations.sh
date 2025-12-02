#!/usr/bin/env bash
set -euo pipefail

# Batch runner for unified_evaluation.py across the supported model variants.
# Each invocation is executed sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON:-python}

# Optional: set CROMA_WEIGHTS to the path of the pretrained CROMA checkpoint
# before running this script. The script will error if it is required but
# not provided.
CROMA_WEIGHTS_PATH=/home/juro4948/ciip/comparison/CROMA-main/CROMA_base.pt

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
  --model-in-channels 13 \
  --ssl4eo-subset-size 50 \
  --neuco-modalities s2l1c \
  --disable-neuco \
  --disable-ssl4eo 

  

# run_eval "DOFA_base_S2_13ch" \
#   --model-type backbone_only \
#   --model-weights dofa_base_s2_13ch \
#   --model-in-channels 13 \
#   --ssl4eo-subset-size 50 \
#   --disable-neuco \
#   --disable-ssl4eo \
#   --neuco-modalities s2l1c 
#   # --disable-eurosat

# # TorchGeo ResNet50 baselines
# run_eval "TorchGeo ResNet50 MoCo" \
#   --model-type torchgeo_resnet50 \
#   --model-weights moco \
#   --model-in-channels 13 \
#   --ssl4eo-subset-size 50 \
#   --neuco-modalities s2l1c s1 \
#   --disable-neuco \
#   --disable-ssl4eo

# run_eval "ScaleMAE_large_RGB" \
#   --model-type backbone_only \
#   --model-weights scalemae_large_rgb \
#   --model-in-channels 3 \
#   --ssl4eo-subset-size 50 \
#   --neuco-modalities s2l2a \
#   --disable-neuco \
#   --disable-ssl4eo

# run_eval "ResNet18_S2_ALL_MOCO" \
#   --model-type backbone_only \
#   --model-weights resnet18_s2_all_moco \
#   --model-in-channels 13 \
#   --ssl4eo-subset-size 70 \
#   --disable-neuco \
#   --disable-ssl4eo \
#   --neuco-modalities s2l1c 
#   # --disable-eurosat

# run_eval "ResNet18_S2_RGB_MOCO" \
#   --model-type backbone_only \
#   --model-weights resnet18_s2_rgb_moco \
#   --model-in-channels 3 \
#   --ssl4eo-subset-size 70 \
#   --neuco-modalities s2l2a \
#   --disable-neuco \
#   --disable-ssl4eo

# run_eval "ResNet50_S2_RGB_MOCO" \
#   --model-type backbone_only \
#   --model-weights resnet50_s2_rgb_moco \
#   --model-in-channels 3 \
#   --ssl4eo-subset-size 70 \
#   --neuco-modalities s2l2a \
#   --disable-neuco \
#   --disable-ssl4eo

# run_eval "ResNet152_ImageNet_RGB" \
#   --model-type backbone_only \
#   --model-weights resnet152_imagenet_rgb \
#   --model-in-channels 3 \
#   --ssl4eo-subset-size 70 \
#   --neuco-modalities s2l2a \
#   --disable-neuco \
#   --disable-ssl4eo

# run_eval "ViTSmall16_S2_ALL_MOCO" \
#   --model-type backbone_only \
#   --model-weights vitsmall16_s2_all_moco \
#   --model-in-channels 13 \
#   --ssl4eo-subset-size 50 \
#   --disable-neuco \
#   --disable-ssl4eo \
#   --neuco-modalities s2l1c 
#   # --disable-eurosat


# run_eval "TorchGeo ResNet50 DINO" \
#   --model-type torchgeo_resnet50 \
#   --model-weights dino \
#   --model-in-channels 13 \
#   --ssl4eo-subset-size 70 \
#   --neuco-modalities s2l1c \
#   --disable-neuco \
#   --disable-ssl4eo


# # // TODO run croma
# # CROMA model (requires a weights path)
# if [[ -z "${CROMA_WEIGHTS_PATH}" ]]; then
#   echo "CROMA_WEIGHTS environment variable is not set; skipping CROMA run." >&2
# else
#   run_eval "CROMA" \
#     --model-type croma \
#     --croma-weights "${CROMA_WEIGHTS_PATH}" \
#     --model-in-channels 12 \
#     --ssl4eo-subset-size 70 \
#     --neuco-modalities s2l2a s1 \
#     --disable-ssl4eo \
#     --disable-neuco \
#     # --disable-eurosat

# fi



# # CIIP checkpoint (defaults are resolved inside unified_evaluation.py)
# run_eval "CIIP checkpoint" \
#   --model-type ciip_checkpoint \
#   --ssl4eo-subset-size 70 \
#   --model-in-channels 12 \
#   --ciip-epoch 115 \
#   --model-path 2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \
#   --disable-ssl4eo \
#   --disable-neuco \
#   --neuco-modalities s2l2a s1 
# #   --disable-eurosat

# # CIIP checkpoint (defaults are resolved inside unified_evaluation.py)
# run_eval "11-22 Random ResNet (CIIP), v1.1/12bands, Epoch0" \
#   --model-type ciip_checkpoint \
#   --ssl4eo-subset-size 70 \
#   --model-in-channels 12 \
#   --ciip-epoch 0 \
#   --model-path 2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \
#   --neuco-modalities s2l2a s1 \
#   --disable-neuco \
#   --disable-ssl4eo

  # CIIP checkpoint (defaults are resolved inside unified_evaluation.py)
  #  vanilla, clamped logist, bs: 2025_11_22-08_31_28-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \

run_eval "CIIP checkpoint epoch 20" \
  --model-type ciip_checkpoint \
  --ssl4eo-subset-size 70 \
  --model-in-channels 12 \
  --ciip-epoch 20 \
  --model-path 2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \
  --neuco-modalities s2l2a s1 

run_eval "CIIP checkpoint epoch 10" \
  --model-type ciip_checkpoint \
  --ssl4eo-subset-size 70 \
  --model-in-channels 12 \
  --ciip-epoch 10 \
  --model-path 2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \
  --neuco-modalities s2l2a s1 

run_eval "CIIP checkpoint epoch 30" \
  --model-type ciip_checkpoint \
  --ssl4eo-subset-size 70 \
  --model-in-channels 12 \
  --ciip-epoch 30 \
  --model-path 2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \
  --neuco-modalities s2l2a s1 

run_eval "CIIP checkpoint epoch 40" \
  --model-type ciip_checkpoint \
  --ssl4eo-subset-size 70 \
  --model-in-channels 12 \
  --ciip-epoch 40 \
  --model-path 2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \
  --neuco-modalities s2l2a s1 

run_eval "CIIP checkpoint epoch 50" \
  --model-type ciip_checkpoint \
  --ssl4eo-subset-size 70 \
  --model-in-channels 12 \
  --ciip-epoch 50 \
  --model-path 2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \
  --neuco-modalities s2l2a s1 

run_eval "CIIP checkpoint epoch 60" \
  --model-type ciip_checkpoint \
  --ssl4eo-subset-size 70 \
  --model-in-channels 12 \
  --ciip-epoch 60 \
  --model-path 2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \
  --neuco-modalities s2l2a s1 

run_eval "CIIP checkpoint epoch 70" \
  --model-type ciip_checkpoint \
  --ssl4eo-subset-size 70 \
  --model-in-channels 12 \
  --ciip-epoch 70 \
  --model-path 2025_11_29-20_57_06-model_resnet50-lr_0.001-b_6-j_6-p_amp_bfloat16 \
  --neuco-modalities s2l2a s1 
