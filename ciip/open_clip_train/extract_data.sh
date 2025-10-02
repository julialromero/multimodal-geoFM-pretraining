#!/bin/bash

export CUDA_VISIBLE_DEVICES=""


set -e  # Exit on any error
echo "Calling s1 and s2 extraction scripts"

# echo "$(date) | Task $SLURM_PROCID | Running on node $SLURM_NODEID with local ID $SLURM_LOCALID"

# scontrol show job to check GPU allocation
# print this scontrol to console
echo "$(date) | Task $SLURM_PROCID with LocalID: $SLURM_LOCALID | scontrol show job $SLURM_JOBID" 


# Determine the S1/S2 split dynamically so job scripts can increase --ntasks-per-node
# without having to update this file. We honour an explicit override (S1_S2_SPLIT_OVERRIDE
# or S1_S2_SPLIT) first, then fall back to SLURM_NTASKS_PER_NODE, and finally split the
# total SLURM_NTASKS in half when per-node information is unavailable.
S1_S2_SPLIT_OVERRIDE=${S1_S2_SPLIT_OVERRIDE:-${S1_S2_SPLIT:-}}

if [ -n "$S1_S2_SPLIT_OVERRIDE" ]; then
    split=$S1_S2_SPLIT_OVERRIDE
    split_source="S1_S2_SPLIT override"
else
    split_source="SLURM_NTASKS_PER_NODE"
    tasks_for_split=${SLURM_NTASKS_PER_NODE:-}

    if [ -z "$tasks_for_split" ]; then
        split_source="SLURM_NTASKS"
        tasks_for_split=${SLURM_NTASKS:-}
    fi

    if [ -n "$tasks_for_split" ]; then
        if [ $((tasks_for_split % 2)) -ne 0 ]; then
            echo "$(date) | Warning: $split_source reported an odd worker count ($tasks_for_split). Rounding up when splitting between S1 and S2." >&2
            split=$(((tasks_for_split + 1) / 2))
        else
            split=$((tasks_for_split / 2))
        fi
    fi
fi

case ${split:-} in
    ''|*[!0-9]*)
        echo "$(date) | Warning: Unable to determine a numeric S1/S2 split. Defaulting to 1." >&2
        split=1
        ;;
esac

if [ "$split" -le 0 ]; then
    echo "$(date) | Warning: Derived S1/S2 split ($split) is not positive. Defaulting to 1." >&2
    split=1
fi

split_source=${split_source:-default}

# Determine which script to run based on the dynamically derived split.
if [ "$SLURM_LOCALID" -lt "$split" ]; then
    echo "$(date) | Task $SLURM_PROCID running extract_data_s1.sh (split source: $split_source, threshold: $split)"
    /projects/bekj/jromero5/ciip/ciip/open_clip_train/extract_data_s1.sh
else
    echo "$(date) | Task $SLURM_PROCID running extract_data_s2.sh (split source: $split_source, threshold: $split)"
    /projects/bekj/jromero5/ciip/ciip/open_clip_train/extract_data_s2.sh
fi

# nvidia-smi

# pkill -P $$

# echo "$(date) | Task $SLURM_PROCID | Cleaning up any leftover processes."
# nvidia-smi