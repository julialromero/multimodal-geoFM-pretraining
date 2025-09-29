#!/bin/bash
set -e  # Exit on any error
echo "Starting data extraction - S1"

RANKFILE=$SLURM_NODELIST
NODE_RANK=$SLURM_NODEID

# Assign chunk name based on rank
SLURM_ADJUSTED_LOCALID=$((SLURM_LOCALID % 4))
GLOBAL_TASK_ID=$((SLURM_NODEID * 4 + SLURM_ADJUSTED_LOCALID))
CHUNK_ID=$(printf "%02d" $GLOBAL_TASK_ID)
# CHUNK_ID=$(printf "%02d" $NODE_RANK)


CHUNK_FILE_S1=/work/nvme/bekj/jromero5/tarballs/s1/chunk_${CHUNK_ID}.tar.gz
TMP_CHUNK_FILE_S1=/tmp/$USER/s1/chunk_${CHUNK_ID}.tar.gz
EXTRACT_DIR_S1=/tmp/$USER/s1

echo "S1 Local ID: $SLURM_LOCALID"
echo "S1 Node rank: $NODE_RANK"
echo "S1 Global task ID: $GLOBAL_TASK_ID"
echo "S1 Chunk ID: $CHUNK_ID"
echo "S1 Number of CPUs per task: $SLURM_CPUS_PER_TASK"

mkdir -p /tmp/$USER/s1
mkdir -p $EXTRACT_DIR_S1

####### PROCESS S1 #######
# COPY S1 TAR FILE
echo "$(date) S1 | Global task ID $GLOBAL_TASK_ID | Copying chunk file $CHUNK_FILE_S1 to $TMP_CHUNK_FILE_S1"
cp $CHUNK_FILE_S1 $TMP_CHUNK_FILE_S1
echo "$(date) S1 | Global task ID $GLOBAL_TASK_ID | Finished copying chunk file $CHUNK_FILE_S1 to $TMP_CHUNK_FILE_S1"

# Parallel extraction using all allocated CPUs
echo "$(date) S1 | Global task ID $GLOBAL_TASK_ID | Extracting $TMP_CHUNK_FILE_S1 to $EXTRACT_DIR_S1"
tar -I pigz -xf $TMP_CHUNK_FILE_S1 -C $EXTRACT_DIR_S1
rm $TMP_CHUNK_FILE_S1
echo "$(date) S1 | Global task ID $GLOBAL_TASK_ID | Finished extracting S1 chunk $CHUNK_ID"




