#!/bin/bash

export CUDA_VISIBLE_DEVICES=""


set -e  # Exit on any error
echo "Calling s1 and s2 extraction scripts"

# echo "$(date) | Task $SLURM_PROCID | Running on node $SLURM_NODEID with local ID $SLURM_LOCALID"

# scontrol show job to check GPU allocation
# print this scontrol to console
echo "$(date) | Task $SLURM_PROCID with LocalID: $SLURM_LOCALID | scontrol show job $SLURM_JOBID" 


# Determine which script to run based on the global task ID (SLURM_PROCID)
if [ $SLURM_LOCALID -lt 4 ]; then
    # First two tasks (SLURM_PROCID=0,1) run extract_data_s1.sh
    echo "$(date) | Task $SLURM_PROCID running extract_data_s1.sh"
    /projects/bekj/jromero5/ciip/ciip/open_clip_train/extract_data_s1.sh
else
    # Last two tasks (SLURM_PROCID=2,3) run extract_data_s2.sh
    echo "$(date) | Task $SLURM_PROCID running extract_data_s2.sh"
    /projects/bekj/jromero5/ciip/ciip/open_clip_train/extract_data_s2.sh
fi

# nvidia-smi

# pkill -P $$

# echo "$(date) | Task $SLURM_PROCID | Cleaning up any leftover processes."
# nvidia-smi