#!/bin/bash

export CUDA_VISIBLE_DEVICES=""


set -e  # Exit on any error
echo "Calling s1 and s2 extraction scripts"

sanitize_positive_integer() {
    local raw_value=$1
    local label=$2
    local sanitized=""

    if [ -n "$raw_value" ]; then
        sanitized=$(printf '%s' "$raw_value" | sed -n -E 's/^[[:space:]]*([0-9]+).*$/\1/p')
        if [ -z "$sanitized" ]; then
            echo "$(date) | Warning: Unable to parse a positive integer from $label ('$raw_value')." >&2
        fi
    fi

    printf '%s' "$sanitized"
}

# echo "$(date) | Task $SLURM_PROCID | Running on node $SLURM_NODEID with local ID $SLURM_LOCALID"

# scontrol show job to check GPU allocation
# print this scontrol to console
echo "$(date) | Task $SLURM_PROCID with LocalID: $SLURM_LOCALID | scontrol show job $SLURM_JOBID"


# Determine the S1/S2 split dynamically so job scripts can increase --ntasks-per-node
# without having to update this file. We honour an explicit override (S1_S2_SPLIT_OVERRIDE
# or S1_S2_SPLIT) first, then fall back to SLURM_NTASKS_PER_NODE, and finally split the
# total SLURM_NTASKS in half when per-node information is unavailable.
#
# The resulting split represents the number of S1 workers per node. We also derive
# contiguous worker identifiers/counts for the S1 and S2 pools so the dataset scripts can
# shard over the true worker ranges rather than relying on SLURM globals that include both
# pools.
S1_S2_SPLIT_OVERRIDE=${S1_S2_SPLIT_OVERRIDE:-${S1_S2_SPLIT:-}}

if [ -n "$S1_S2_SPLIT_OVERRIDE" ]; then
    split=$S1_S2_SPLIT_OVERRIDE
    split_source="S1_S2_SPLIT override"
else
    split_source=""
    tasks_for_split=""

    for candidate in SLURM_STEP_NTASKS_PER_NODE SLURM_NTASKS_PER_NODE SLURM_STEP_NTASKS SLURM_NTASKS; do
        raw_value=${!candidate:-}
        if [ -n "$raw_value" ]; then
            tasks_for_split=$(sanitize_positive_integer "$raw_value" "$candidate")
            if [ -n "$tasks_for_split" ]; then
                split_source="$candidate"
                break
            fi
        fi
    done

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

# Normalise SLURM_NTASKS_PER_NODE into a numeric per-node worker count when available.
per_node_total=""
for candidate in SLURM_STEP_NTASKS_PER_NODE SLURM_NTASKS_PER_NODE; do
    raw_value=${!candidate:-}
    if [ -n "$raw_value" ]; then
        per_node_total=$(sanitize_positive_integer "$raw_value" "$candidate")
        if [ -n "$per_node_total" ]; then
            break
        fi
    fi
done

per_node_s1=$split
per_node_s2=0
if [ -n "$per_node_total" ]; then
    per_node_s2=$((per_node_total - per_node_s1))
    if [ "$per_node_s2" -lt 0 ]; then
        per_node_s2=0
    fi
fi

# Determine the total number of nodes participating in the job.
num_nodes=""
for candidate in SLURM_STEP_NUM_NODES SLURM_JOB_NUM_NODES SLURM_NNODES; do
    raw_value=${!candidate:-}
    if [ -n "$raw_value" ]; then
        num_nodes=$(sanitize_positive_integer "$raw_value" "$candidate")
        if [ -n "$num_nodes" ]; then
            break
        fi
    fi
done

if [ -z "$num_nodes" ] && [ -n "$per_node_total" ]; then
    total_tasks=""
    for candidate in SLURM_STEP_NTASKS SLURM_NTASKS; do
        raw_value=${!candidate:-}
        if [ -n "$raw_value" ]; then
            total_tasks=$(sanitize_positive_integer "$raw_value" "$candidate")
            if [ -n "$total_tasks" ]; then
                break
            fi
        fi
    done

    if [ -n "$total_tasks" ] && [ "$per_node_total" -gt 0 ]; then
        num_nodes=$(((total_tasks + per_node_total - 1) / per_node_total))
    fi
fi

case ${num_nodes:-} in
    ''|*[!0-9]*)
        num_nodes=""
        ;;
    0)
        num_nodes=""
        ;;
esac

total_tasks=""
for candidate in SLURM_STEP_NTASKS SLURM_NTASKS; do
    raw_value=${!candidate:-}
    if [ -n "$raw_value" ]; then
        total_tasks=$(sanitize_positive_integer "$raw_value" "$candidate")
        if [ -n "$total_tasks" ]; then
            break
        fi
    fi
done

s1_worker_count=""
s2_worker_count=""
if [ -n "$num_nodes" ]; then
    s1_worker_count=$((per_node_s1 * num_nodes))
    if [ -n "$per_node_total" ]; then
        s2_worker_count=$((per_node_s2 * num_nodes))
    elif [ -n "$total_tasks" ]; then
        s2_worker_count=$((total_tasks - s1_worker_count))
    fi
fi

if [ -z "$s1_worker_count" ]; then
    if [ -n "$total_tasks" ]; then
        # Assume the split partitions workers as evenly as possible.
        s1_worker_count=$(((total_tasks + 1) / 2))
    else
        s1_worker_count=$per_node_s1
    fi
fi

if [ -z "$s2_worker_count" ]; then
    if [ -n "$total_tasks" ]; then
        s2_worker_count=$((total_tasks - s1_worker_count))
    else
        s2_worker_count=$per_node_s2
    fi
fi

if [ -n "$total_tasks" ] && [ "$s1_worker_count" -gt "$total_tasks" ]; then
    s1_worker_count=$total_tasks
fi
if [ "$s1_worker_count" -lt 0 ]; then
    s1_worker_count=0
fi
if [ "$s2_worker_count" -lt 0 ]; then
    s2_worker_count=0
fi
if [ -n "$total_tasks" ] && [ "$s2_worker_count" -gt "$total_tasks" ]; then
    s2_worker_count=$total_tasks
fi

export CIIP_S1_WORKERS_PER_NODE=$per_node_s1
export CIIP_S2_WORKERS_PER_NODE=$per_node_s2

# Derive contiguous worker identifiers for this process within its pool.
node_id=${SLURM_NODEID:-0}
case ${node_id:-} in
    ''|*[!0-9]*)
        node_id=0
        ;;
esac

local_id_raw=${SLURM_LOCALID:-0}
local_id=$(sanitize_positive_integer "$local_id_raw" "SLURM_LOCALID")
if [ -z "$local_id" ]; then
    echo "$(date) | Warning: SLURM_LOCALID ('$local_id_raw') is not numeric. Defaulting to 0." >&2
    local_id=0
fi

if [ "$local_id" -lt "$split" ]; then
    local_s1_id=$local_id
    worker_id=$local_s1_id
    export CIIP_S1_WORKER_ID=$worker_id
    unset CIIP_S2_WORKER_ID
else
    local_s2_id=$((local_id - per_node_s1))
    if [ "$local_s2_id" -lt 0 ]; then
        local_s2_id=0
    fi
    worker_id=$local_s2_id
    export CIIP_S2_WORKER_ID=$worker_id
    unset CIIP_S1_WORKER_ID
fi

# Determine which script to run based on the dynamically derived split.
if [ "$local_id" -lt "$split" ]; then
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