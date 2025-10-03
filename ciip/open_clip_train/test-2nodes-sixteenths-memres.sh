#!/bin/bash

#SBATCH --job-name=test_batch_size_512 # series__startrun
#SBATCH --output=series_%j_node-%N_task%t.out
#SBATCH --error=series_%j_node-%N_task%t.err
#SBATCH --partition=ghx4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=12 # 8x + 1x#GPUs # 8 tasks for each node for data extraction, + 1task per gpu for memory grab
#SBATCH --cpus-per-task=6 # use x CPUs for each one GPU
#SBATCH --account=bekj-dtai-gh
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=julia.romero@colorado.edu
#SBATCH --mail-type="BEGIN,END"
#SBATCH --exclusive
#SBATCH --mem=0
# NOTE: The extraction helpers now stage *all* chunk_XX.tar.gz archives on each
# node (/tmp/$USER/s1 and /tmp/$USER/s2c). Ensure every node has enough local
# scratch space for the full, extracted datasets before launching the job.


###NOTE: make sure to run the following to launch the script:
# module load cuda/12.6.1  # will error if not run
# conda activate cii
# sbatch /projects/bekj/jromero5/ciip/ciip/open_clip_train/test-2nodes-sixteenths-memres.sh

CUDA_MALLOC_SCRIPT="/projects/bekj/jromero5/ciip/ciip/open_clip_train/cuda_malloc.py" # Adjust path
EXTRACT_DATA_SCRIPT="/projects/bekj/jromero5/ciip/ciip/open_clip_train/extract_data.sh" # Adjust path

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')


STATE_DIR="/work/nvme/bekj/jromero5/ciip/logs/run_state"
mkdir -p "$STATE_DIR"
STATE_FILE="${STATE_DIR}/${SLURM_JOB_ID}.env"

if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
fi

if [ -z "$LOG_BASE_DIR" ]; then
    DATE=$(date "+%Y-%m-%d_%H-%M-%S")
    LOG_BASE_DIR="/work/nvme/bekj/jromero5/ciip/logs/4nodes-tests/${DATE}-test-compute"
    echo "LOG_BASE_DIR=$LOG_BASE_DIR" > "$STATE_FILE"
fi

mkdir -p "$LOG_BASE_DIR"
export MY_LOG_BASE_DIR="$LOG_BASE_DIR"
LATEST_CHECKPOINT_TRACKER="${LOG_BASE_DIR}/latest_checkpoint.txt"
if [ -f "$LATEST_CHECKPOINT_TRACKER" ]; then
    export RESUME_FROM_CHECKPOINT="$(cat "$LATEST_CHECKPOINT_TRACKER")"
fi

# Export the LOG_BASE_DIR so Python scripts can access it
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO #WARN  #INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=hsn
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.6
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "Running on $(hostname)"
echo "SLURM_NODEID: $SLURM_NODEID, SLURM_PROCID: $SLURM_PROCID"
echo "SLURM NUM TASKS: $SLURM_NTASKS, SLURM NODES: $SLURM_NNODES"


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=29500
echo "MASTER_ADDR: $MASTER_ADDR"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
cd /projects/bekj/jromero5/ciip/ciip/open_clip_train/


# echio log base dir
echo "Log base directory: $LOG_BASE_DIR"
echo "My log base directory: $MY_LOG_BASE_DIR"
echo "$(date) | Global rank $SLURM_PROCID | Starting training script."

run_training_attempt() {
    local LOG_PIDS=()
    local skip_data=${SKIP_DATA_EXTRACTION:-0}

    srun --ntasks=4 --nodes=4 --overlap --accel-bind=gn torchrun \
      --nnodes=4 \
      --nproc_per_node=4 \
      --node_rank=$SLURM_NODEID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      run_train_val_distributed.py &

    local TORCHRUN_PID=$!

    # --- Step 1.5: Wait for all PyTorch processes to signal memory allocation ---
    echo "$(date) | Global rank $SLURM_PROCID | Waiting for all PyTorch processes to complete memory allocation."

    sleep 120
    ALL_ALLOCATED=false
    MAX_WAIT_TIME=600 # Max 10 minutes (600 seconds)
    CHECK_INTERVAL=15 # Check every 15 seconds
    ELAPSED_TIME=0
    while [ "$ALL_ALLOCATED" == false ] && [ "$ELAPSED_TIME" -lt "$MAX_WAIT_TIME" ]; do
        ALL_ALLOCATED=true # Assume true, then check if any are missing

        # Loop through each node
        for NODE in "${nodes_array[@]}"; do
            NUM_PROCS_PER_NODE=$SLURM_GPUS_PER_NODE
            CHECK_CMD='missing=0; for i in $(seq 0 $((NUM_PROCS_PER_NODE-1))); do [ -f "/tmp/jromero5/allocated_${i}.txt" ] || missing=1; done; exit $missing'

            if srun --nodes=1 --ntasks=1 --overlap --export=ALL,NUM_PROCS_PER_NODE=$NUM_PROCS_PER_NODE \
                -w "$NODE" bash -c "$CHECK_CMD"; then
                echo "$(date) | Global rank $SLURM_PROCID | All allocation files found on node $NODE."
            else
                echo "$(date) | Global rank $SLURM_PROCID | Waiting: Missing allocation files on node $NODE."
                ALL_ALLOCATED=false
                break
            fi
        done

        if [ "$ALL_ALLOCATED" == false ]; then
            sleep "$CHECK_INTERVAL"
            ELAPSED_TIME=$((ELAPSED_TIME + CHECK_INTERVAL))
        fi
    done

    echo "$(date) | Global rank $SLURM_PROCID | All PyTorch processes have completed memory allocation."

    # echo nvidia-smi contents to SLURM .out file on process 0
    if [ $SLURM_PROCID -eq 0 ]; then
        echo "$(date) | Global rank $SLURM_PROCID | nvidia-smi output on $(hostname):"
        nvidia-smi
        echo "============================================"

    fi

    DATA_EXTRACTION_DONE_FILE="${LOG_BASE_DIR}/data_extraction_done.txt"
    if [ "$skip_data" -eq 0 ] && [ ! -f "$DATA_EXTRACTION_DONE_FILE" ]; then
        ########## Run the data extraction script on all nodes - run in the foreground
        echo "$(date) | Global rank $SLURM_PROCID | Starting data extraction script."
        # run_train_val_distributed.py will continue to block on
        # ${LOG_BASE_DIR}/data_extraction_done.txt, so all ranks must finish
        # extraction before training begins.
        # echo the full path to the output file extract_data_%j_%n.out

        srun --nodes=$SLURM_NNODES --ntasks-per-node=8 --cpus-per-task=6 --gpus-per-task=0 \
             --job-name=data_extract \
             --output=/work/nvme/bekj/jromero5/ciip/logs/extract_data_%j.out \
             --error=/work/nvme/bekj/jromero5/ciip/logs/extract_data_%j.err \
             --export=ALL \
             --overlap \
             $EXTRACT_DATA_SCRIPT
        echo "$(date) | Global rank $SLURM_PROCID | Done with file extraction."


        # create file to indicate that the data extraction is done, torch script will read and continue processing
        srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 \
             --job-name=data_extract_file \
             --export=ALL \
             --overlap \
             bash -c "echo \$(date) Data extraction done > ${DATA_EXTRACTION_DONE_FILE}"

        echo "$(date) | Global rank $SLURM_PROCID | Data extraction done file created."

        # add file check to ensure that the file was created
        srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 \
             --job-name=data_extract \
             --export=ALL \
             --overlap \
             bash -c "test -f ${DATA_EXTRACTION_DONE_FILE} && echo 'File exists' || echo 'File does not exist'"
        echo "$(date) | Global rank $SLURM_PROCID | Data extraction done file checked."
    else
        echo "$(date) | Global rank $SLURM_PROCID | Skipping data extraction (already completed)."
    fi

    # start 1 logging process on each node - looping nvidia-smi
    for NODE in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
        srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 \
             --job-name=gpu_monitor \
             --output="${LOG_BASE_DIR}/gpu_usage_monitor_node_${NODE}.out" \
             --overlap \
             --export=ALL \
             -w $NODE \
             bash -c "END_TIME=\$((\$(date +%s) + 1200)); \
                      while [ \$(date +%s) -lt \$END_TIME ]; do \
                          TIME=\$(date +%H:%M:%S); \
                          echo \"=== \${TIME} === (Node $NODE)\"; \
                          nvidia-smi; \
                          sleep 30; \
                      done" &
        LOG_PIDS+=($!) # Capture the PID of the background srun command
        echo "$(date) | nvidia-smi monitor srun started on $NODE with PID: $!"
    done
    echo "$(date) | nvidia-smi monitor srun started on all nodes with PIDs: ${LOG_PIDS[@]}"

    wait $TORCHRUN_PID
    TORCHRUN_EXIT=$?

    for pid in "${LOG_PIDS[@]}"; do
        echo "$(date) | Terminating logging process with PID: $pid"
        kill "$pid" 2>/dev/null || true
    done

    wait || true

    return $TORCHRUN_EXIT
}

MAX_RETRIES=${MAX_RETRIES:-3}
ATTEMPT=0
EXIT_CODE=0

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    if [ $ATTEMPT -gt 0 ]; then
        export SKIP_DATA_EXTRACTION=1
        echo "$(date) | Retry attempt $ATTEMPT after failure."
    else
        if [ -f "${LOG_BASE_DIR}/data_extraction_done.txt" ]; then
            export SKIP_DATA_EXTRACTION=1
        else
            export SKIP_DATA_EXTRACTION=0
        fi
    fi

    run_training_attempt
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 42 ]; then
        echo "$(date) | torchrun exited due to CUDA OOM (code 42). Preparing to resume."
        ATTEMPT=$((ATTEMPT + 1))
        sleep 60
        continue
    fi

    break
done

exit $EXIT_CODE

echo "== End of Job =="

