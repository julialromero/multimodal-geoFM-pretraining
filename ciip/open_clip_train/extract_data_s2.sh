#!/bin/bash
set -euo pipefail

pigz_threads="${SLURM_CPUS_PER_TASK:-$(nproc)}"

echo "Starting data extraction - S2"
echo "$(date) S2 | Using pigz threads: ${pigz_threads}"

SOURCE_DIR="/work/nvme/bekj/jromero5/tarballs/s2c"
EXTRACT_DIR="/tmp/$USER/s2c"
COORD_DIR="/tmp/$USER/.ciip_s2"
MARKER_DIR="$COORD_DIR/markers"
LOCK_DIR="$COORD_DIR/locks"
TMP_DIR="$COORD_DIR/tmp"

mkdir -p "$EXTRACT_DIR" "$MARKER_DIR" "$LOCK_DIR" "$TMP_DIR"

echo "$(date) S2 | Running on node ${SLURM_NODEID:-unknown} with local task ${SLURM_LOCALID:-unknown}"
echo "$(date) S2 | Copying all chunks from $SOURCE_DIR to $EXTRACT_DIR"

shopt -s nullglob
tarballs=("$SOURCE_DIR"/chunk_*.tar.gz)
if [ ${#tarballs[@]} -eq 0 ]; then
    echo "$(date) S2 | No chunk tarballs were found in $SOURCE_DIR"
fi

worker_id=${SLURM_PROCID:-0}
worker_count=${SLURM_NTASKS:-1}
if [ "$worker_count" -lt 1 ]; then
    worker_count=1
fi
worker_id=$((worker_id % worker_count))
echo "$(date) S2 | Worker sharding id $worker_id of $worker_count"

for idx in "${!tarballs[@]}"; do
    tarball="${tarballs[$idx]}"
    chunk_basename=$(basename "$tarball")
    chunk_name="${chunk_basename%.tar.gz}"
    marker="$MARKER_DIR/$chunk_name"
    tmp_tar="$EXTRACT_DIR/$chunk_basename"
    lock_path="$LOCK_DIR/$chunk_name.lock"

    if [ $((idx % worker_count)) -ne $worker_id ]; then
        continue
    fi

    if [ -f "$marker" ]; then
        echo "$(date) S2 | $chunk_name already extracted, skipping"
        continue
    fi

    exec {lock_fd}>"$lock_path"
    if ! flock -n "$lock_fd"; then
        # Another process is already working on this chunk despite sharding; skipping.
        echo "$(date) S2 | $chunk_name locked by another worker despite sharding; contention should be rare"
        exec {lock_fd}>&-
        continue
    fi

    if [ -f "$marker" ]; then
        echo "$(date) S2 | $chunk_name extracted by another worker after sharding; contention should be rare, skipping"
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        rm -f "$lock_path"
        continue
    fi

    echo "$(date) S2 | Copying $tarball to $tmp_tar"
    rm -f "$tmp_tar"
    rsync -a "$tarball" "$tmp_tar"

    tmp_extract_dir=$(mktemp -d "$TMP_DIR/${chunk_name}_XXXXXX")
    echo "$(date) S2 | Extracting $tmp_tar into $tmp_extract_dir"
    tar --use-compress-program="pigz -p ${pigz_threads}" -xf "$tmp_tar" -C "$tmp_extract_dir"
    rm -f "$tmp_tar"

    echo "$(date) S2 | Installing contents of $chunk_name into $EXTRACT_DIR"
    rsync -a "$tmp_extract_dir"/ "$EXTRACT_DIR"/

    rm -rf "$tmp_extract_dir"
    touch "$marker"
    echo "$(date) S2 | Finished processing $chunk_name"

    flock -u "$lock_fd"
    exec {lock_fd}>&-
    rm -f "$lock_path"
done
shopt -u nullglob

echo "$(date) S2 | Data extraction complete"
