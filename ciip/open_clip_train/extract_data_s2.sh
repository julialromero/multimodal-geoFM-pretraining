#!/bin/bash
set -euo pipefail

echo "Starting data extraction - S2"

SOURCE_DIR="/work/nvme/bekj/jromero5/tarballs/s2c"
EXTRACT_DIR="/tmp/$USER/s2c"
MARKER_DIR="$EXTRACT_DIR/.extracted"
LOCK_DIR="$EXTRACT_DIR/.locks"

mkdir -p "$EXTRACT_DIR" "$MARKER_DIR" "$LOCK_DIR"

echo "$(date) S2 | Running on node ${SLURM_NODEID:-unknown} with local task ${SLURM_LOCALID:-unknown}"
echo "$(date) S2 | Copying all chunks from $SOURCE_DIR to $EXTRACT_DIR"

shopt -s nullglob
tarballs=("$SOURCE_DIR"/chunk_*.tar.gz)
if [ ${#tarballs[@]} -eq 0 ]; then
    echo "$(date) S2 | No chunk tarballs were found in $SOURCE_DIR"
fi

for tarball in "${tarballs[@]}"; do
    chunk_basename=$(basename "$tarball")
    chunk_name="${chunk_basename%.tar.gz}"
    marker="$MARKER_DIR/$chunk_name"
    tmp_tar="$EXTRACT_DIR/$chunk_basename"

    if [ -f "$marker" ]; then
        echo "$(date) S2 | $chunk_name already extracted, skipping"
        continue
    fi

    (
        flock 9

        if [ -f "$marker" ]; then
            echo "$(date) S2 | $chunk_name extracted by another rank, skipping"
        else
            echo "$(date) S2 | Copying $tarball to $tmp_tar"
            rm -f "$tmp_tar"
            rsync -a "$tarball" "$tmp_tar"

            tmp_extract_dir=$(mktemp -d "$EXTRACT_DIR/.${chunk_name}_XXXXXX")
            echo "$(date) S2 | Extracting $tmp_tar into $tmp_extract_dir"
            tar -I pigz -xf "$tmp_tar" -C "$tmp_extract_dir"
            rm -f "$tmp_tar"

            shopt -s dotglob
            for extracted_item in "$tmp_extract_dir"/*; do
                [ -e "$extracted_item" ] || continue
                dest="$EXTRACT_DIR/$(basename "$extracted_item")"
                echo "$(date) S2 | Installing $(basename "$extracted_item") into $EXTRACT_DIR"
                rm -rf "$dest"
                mv "$extracted_item" "$dest"
            done
            shopt -u dotglob

            rmdir "$tmp_extract_dir"
            touch "$marker"
            echo "$(date) S2 | Finished processing $chunk_name"
        fi

    ) 9>"$LOCK_DIR/$chunk_name.lock"
done
shopt -u nullglob

echo "$(date) S2 | Data extraction complete"
