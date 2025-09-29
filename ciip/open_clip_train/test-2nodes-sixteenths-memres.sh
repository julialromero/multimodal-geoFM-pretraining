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


DATE=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_BASE_DIR="/work/nvme/bekj/jromero5/ciip/logs/4nodes-tests/${DATE}-test-compute"
mkdir -p "$LOG_BASE_DIR" 
export MY_LOG_BASE_DIR="$LOG_BASE_DIR"

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


# launch script to reserve memory --> should this be --overlap or --exclusive??
# srun --nodes=4 --ntasks=16 --cpus-per-task=6 --cpu_bind=cores \
# torchrun \
#   --nnodes=4 \
#   --nproc_per_node=4 \
#   --node_rank=$SLURM_NODEID \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#   run_train_val_distributed.py


srun --ntasks=4 --nodes=4 --overlap --accel-bind=gn torchrun \
  --nnodes=4 \
  --nproc_per_node=4 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  run_train_val_distributed.py &



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
    for NODE in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
        NUM_PROCS_PER_NODE=$SLURM_GPUS_PER_NODE 
        
        for i in $(seq 0 $((NUM_PROCS_PER_NODE-1))); do
            FILE_TO_CHECK="/tmp/jromero5/allocated_${i}.txt"
            CHECK_RESULT=$(srun --nodes=1 --ntasks=1 --overlap --export=ALL -w "$NODE" bash -c "test -f \"$FILE_TO_CHECK\" && echo 'found' || echo 'not_found'")
            echo "$(date) | Global rank $SLURM_PROCID | Checking file $FILE_TO_CHECK on node $NODE: $CHECK_RESULT"
            if [ "$CHECK_RESULT" == "found" ]; then
                echo "$(date) | Global rank $SLURM_PROCID | File $FILE_TO_CHECK found on node $NODE."
            elif [ "$CHECK_RESULT" == "not_found" ]; then
                echo "$(date) | Global rank $SLURM_PROCID | Waiting: File $FILE_TO_CHECK NOT FOUND on node $NODE."
                ALL_ALLOCATED=false
                break 2 # Break out of inner and outer loops (node and process loops)
            fi
           
        done
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
     bash -c "echo \$(date) Data extraction done > ${LOG_BASE_DIR}/data_extraction_done.txt"

echo "$(date) | Global rank $SLURM_PROCID | Data extraction done file created."

# add file check to ensure that the file was created
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 \
     --job-name=data_extract \
     --export=ALL \
     --overlap \
     bash -c "test -f ${LOG_BASE_DIR}/data_extraction_done.txt && echo 'File exists' || echo 'File does not exist'"
echo "$(date) | Global rank $SLURM_PROCID | Data extraction done file checked."


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


# terminate the logging processes after the training script is done
wait

for pid in "${LOG_PIDS[@]}"; do
    echo "$(date) | Terminating logging process with PID: $pid"
    kill "$pid"
done

echo "== End of Job =="

