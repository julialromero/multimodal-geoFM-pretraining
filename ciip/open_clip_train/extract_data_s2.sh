#!/bin/bash
set -e  # Exit on any error
echo "Starting data extraction - S2"

RANKFILE=$SLURM_NODELIST
NODE_RANK=$SLURM_NODEID

# Assign chunk name based on rank
SLURM_ADJUSTED_LOCALID=$((SLURM_LOCALID % 4))
GLOBAL_TASK_ID=$((SLURM_NODEID * 4 + SLURM_ADJUSTED_LOCALID))
CHUNK_ID=$(printf "%02d" $GLOBAL_TASK_ID)
# CHUNK_ID=$(printf "%02d" $NODE_RANK)


CHUNK_FILE_S2=/work/nvme/bekj/jromero5/tarballs/s2c/chunk_${CHUNK_ID}.tar.gz
TMP_CHUNK_FILE_S2=/tmp/$USER/s2c/chunk_${CHUNK_ID}.tar.gz
EXTRACT_DIR_S2=/tmp/$USER/s2c

# print local id
echo "S2 Local ID: $SLURM_LOCALID"
echo "S2 Node rank: $NODE_RANK"
echo "S2 Global task ID: $GLOBAL_TASK_ID"
echo "S2 Chunk ID: $CHUNK_ID"
echo "S2 Number of CPUs per task: $SLURM_CPUS_PER_TASK"

mkdir -p /tmp/$USER/s2c
mkdir -p $EXTRACT_DIR_S2


####### PROCESS S2 #######
# COPY S2 TAR FILE
echo "$(date) S2 | Global task ID $GLOBAL_TASK_ID | Copying chunk file $CHUNK_FILE_S2 to $TMP_CHUNK_FILE_S2"
cp $CHUNK_FILE_S2 $TMP_CHUNK_FILE_S2
echo "$(date) S2 | Global task ID $GLOBAL_TASK_ID | Finished copying chunk file $CHUNK_FILE_S2 to $TMP_CHUNK_FILE_S2"

# Parallel extraction using all allocated CPUs
echo "$(date) S2 | Global task ID $GLOBAL_TASK_ID | Extracting $TMP_CHUNK_FILE_S2 to $EXTRACT_DIR_S2"
tar -I pigz -xf $TMP_CHUNK_FILE_S2 -C $EXTRACT_DIR_S2
rm $TMP_CHUNK_FILE_S2
echo "$(date) S2 | Global task ID $GLOBAL_TASK_ID | Finished extracting S2 chunk $CHUNK_ID"
