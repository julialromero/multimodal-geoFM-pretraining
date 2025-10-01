#!/bin/bash
set -euo pipefail

echo "Starting data extraction - S1"

SOURCE_DIR="/work/nvme/bekj/jromero5/tarballs/s1"
EXTRACT_DIR="/tmp/$USER/s1"
COORD_DIR="/tmp/$USER/.ciip_s1"
MARKER_DIR="$COORD_DIR/markers"
LOCK_DIR="$COORD_DIR/locks"
TMP_DIR="$COORD_DIR/tmp"

mkdir -p "$EXTRACT_DIR" "$MARKER_DIR" "$LOCK_DIR" "$TMP_DIR"

echo "$(date) S1 | Running on node ${SLURM_NODEID:-unknown} with local task ${SLURM_LOCALID:-unknown}"
echo "$(date) S1 | Copying all chunks from $SOURCE_DIR to $EXTRACT_DIR"

shopt -s nullglob
tarballs=("$SOURCE_DIR"/chunk_*.tar.gz)
if [ ${#tarballs[@]} -eq 0 ]; then
    echo "$(date) S1 | No chunk tarballs were found in $SOURCE_DIR"
fi

for tarball in "${tarballs[@]}"; do
    chunk_basename=$(basename "$tarball")
    chunk_name="${chunk_basename%.tar.gz}"
    marker="$MARKER_DIR/$chunk_name"
    tmp_tar="$EXTRACT_DIR/$chunk_basename"
    lock_path="$LOCK_DIR/$chunk_name.lock"

    if [ -f "$marker" ]; then
        echo "$(date) S1 | $chunk_name already extracted, skipping"
        continue
    fi

    exec {lock_fd}>"$lock_path"
    if ! flock -n "$lock_fd"; then
        # Another process is already working on this chunk; skip it.
        exec {lock_fd}>&-
        continue
    fi

    if [ -f "$marker" ]; then
        echo "$(date) S1 | $chunk_name extracted by another rank, skipping"
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        rm -f "$lock_path"
        continue
    fi

    echo "$(date) S1 | Copying $tarball to $tmp_tar"
    rm -f "$tmp_tar"
    rsync -a "$tarball" "$tmp_tar"

    tmp_extract_dir=$(mktemp -d "$TMP_DIR/${chunk_name}_XXXXXX")
    echo "$(date) S1 | Extracting $tmp_tar into $tmp_extract_dir"
    tar -I pigz -xf "$tmp_tar" -C "$tmp_extract_dir"
    rm -f "$tmp_tar"

    echo "$(date) S1 | Installing contents of $chunk_name into $EXTRACT_DIR"
    rsync -a "$tmp_extract_dir"/ "$EXTRACT_DIR"/

    rm -rf "$tmp_extract_dir"
    touch "$marker"
    echo "$(date) S1 | Finished processing $chunk_name"

    flock -u "$lock_fd"
    exec {lock_fd}>&-
    rm -f "$lock_path"
done
shopt -u nullglob

echo "$(date) S1 | Data extraction complete"




