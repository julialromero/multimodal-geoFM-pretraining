#!/bin/bash

export CUDA_VISIBLE_DEVICES=""


set -e  # Exit on any error
echo "Calling s1 and s2 extraction scripts"

# echo "$(date) | Task $SLURM_PROCID | Running on node $SLURM_NODEID with local ID $SLURM_LOCALID"

# scontrol show job to check GPU allocation
# print this scontrol to console
echo "$(date) | Task $SLURM_PROCID with LocalID: $SLURM_LOCALID | scontrol show job $SLURM_JOBID" 


# Helper to normalise Slurm "tasks per node" strings (e.g. "8(x2),4") into a single
# numeric value representing the primary per-node task count for the current step/job.
normalise_tasks_per_node() {
    local raw=$1
    local value=""

    case ${raw:-} in
        *"(x"*)
            value=${raw%%(x*}
            ;;
        *","*)
            value=${raw%%,*}
            ;;
        *)
            value=$raw
            ;;
    esac

    case ${value:-} in
        ''|*[!0-9]*)
            value=""
            ;;
    esac

    echo "$value"
}

# Determine the S1/S2 split dynamically so job scripts can increase --ntasks-per-node
# without having to update this file. We honour an explicit override (S1_S2_SPLIT_OVERRIDE
# or S1_S2_SPLIT) first, then fall back to step-scoped information
# (SLURM_STEP_TASKS_PER_NODE/SLURM_STEP_NUM_TASKS) before considering the job-level
# variables. This ensures the split reflects the actual srun launch even when the job was
# submitted with different settings.
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
    split_source="SLURM_STEP_TASKS_PER_NODE"
    tasks_for_split=$(normalise_tasks_per_node "${SLURM_STEP_TASKS_PER_NODE:-}")

    if [ -z "$tasks_for_split" ]; then
        split_source="SLURM_NTASKS_PER_NODE"
        tasks_for_split=$(normalise_tasks_per_node "${SLURM_NTASKS_PER_NODE:-}")
    fi

    if [ -z "$tasks_for_split" ]; then
        split_source="SLURM_STEP_NUM_TASKS"
        tasks_for_split=${SLURM_STEP_NUM_TASKS:-}

        case ${tasks_for_split:-} in
            ''|*[!0-9]*)
                tasks_for_split=""
                ;;
        esac
    fi

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

# Normalise tasks-per-node into a numeric per-node worker count when available, preferring
# step-scoped values.
per_node_total=""
per_node_total=$(normalise_tasks_per_node "${SLURM_STEP_TASKS_PER_NODE:-}")
if [ -z "$per_node_total" ]; then
    per_node_total=$(normalise_tasks_per_node "${SLURM_NTASKS_PER_NODE:-}")
fi

per_node_s1=$split
per_node_s2=0
if [ -n "$per_node_total" ]; then
    per_node_s2=$((per_node_total - per_node_s1))
    if [ "$per_node_s2" -lt 0 ]; then
        per_node_s2=0
    fi
fi

# Determine the total number of nodes participating in the job/step.
num_nodes=${SLURM_STEP_NUM_NODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}}
case ${num_nodes:-} in
    ''|*[!0-9]*)
        num_nodes=""
        ;;
esac

if [ -z "$num_nodes" ] && [ -n "$per_node_total" ] && [ -n "${SLURM_NTASKS:-}" ]; then
    total_tasks=$SLURM_NTASKS
    if [ "$per_node_total" -gt 0 ]; then
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
case ${SLURM_STEP_NUM_TASKS:-} in
    ''|*[!0-9]*)
        ;;
    *)
        total_tasks=$SLURM_STEP_NUM_TASKS
        ;;
esac

if [ -z "$total_tasks" ]; then
    case ${SLURM_NTASKS:-} in
        ''|*[!0-9]*)
            ;;
        *)
            total_tasks=$SLURM_NTASKS
            ;;
    esac
fi

s1_worker_count=""
s2_worker_count=""
if [ -n "$num_nodes" ]; then
    s1_worker_count=$((per_node_s1 * num_nodes))
    if [ -n "$per_node_total" ]; then
        s2_worker_count=$((per_node_s2 * num_nodes))
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

if [ -n "$total_tasks" ] && [ -n "$s2_worker_count" ] && [ "$s2_worker_count" -lt 0 ]; then
    s2_worker_count=0
fi

if [ -n "$total_tasks" ] && [ "$s1_worker_count" -gt "$total_tasks" ]; then
    s1_worker_count=$total_tasks
fi
if [ "$s1_worker_count" -lt 0 ]; then
    s1_worker_count=0
fi
if [ -n "$total_tasks" ] && [ "$s2_worker_count" -gt "$total_tasks" ]; then
    s2_worker_count=$total_tasks
fi

# Ensure the per-node S2 worker count does not exceed the number of S2 tasks that actually
# start. We cap it to the maximum possible per-node allocation given the total S2 tasks for
# the step.
if [ -n "$num_nodes" ] && [ "$num_nodes" -gt 0 ] && [ -n "$s2_worker_count" ]; then
    max_per_node_s2=$(((s2_worker_count + num_nodes - 1) / num_nodes))
    if [ "$per_node_s2" -gt "$max_per_node_s2" ]; then
        per_node_s2=$max_per_node_s2
    fi
    if [ "$s2_worker_count" -gt $((per_node_s2 * num_nodes)) ]; then
        s2_worker_count=$((per_node_s2 * num_nodes))
    fi
fi

export CIIP_S1_WORKERS_PER_NODE=$per_node_s1
export CIIP_S2_WORKERS_PER_NODE=$per_node_s2

echo "$(date) | Derived worker layout: split=$split (source: $split_source), per_node_total=${per_node_total:-unknown}, per_node_s1=$per_node_s1, per_node_s2=$per_node_s2, total_s1=${s1_worker_count:-unknown}, total_s2=${s2_worker_count:-unknown}, num_nodes=${num_nodes:-unknown}"

# Derive contiguous worker identifiers for this process within its pool.
node_id=${SLURM_NODEID:-0}
case ${node_id:-} in
    ''|*[!0-9]*)
        node_id=0
        ;;
esac

if [ "$SLURM_LOCALID" -lt "$split" ]; then
    local_s1_id=$SLURM_LOCALID
    worker_id=$local_s1_id
    export CIIP_S1_WORKER_ID=$worker_id
    unset CIIP_S2_WORKER_ID
else
    local_s2_id=$((SLURM_LOCALID - per_node_s1))
    if [ "$local_s2_id" -lt 0 ]; then
        local_s2_id=0
    fi
    worker_id=$local_s2_id
    export CIIP_S2_WORKER_ID=$worker_id
    unset CIIP_S1_WORKER_ID
fi

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