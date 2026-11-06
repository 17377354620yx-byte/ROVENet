#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=/home/yangx/miniconda3/envs/geo_v2/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export IRCADB_V3_RUN_NAME="${IRCADB_V3_RUN_NAME:-full_v1}"
EPOCHS="${IRCADB_V3_EPOCHS:-30}"
RUN_DIR="$PROJECT_ROOT/output/geotransformer.ircadb_v3.$IRCADB_V3_RUN_NAME"
mkdir -p "$RUN_DIR"

TRAIN_ARGS=(--max_epoch "$EPOCHS" --log_steps 50)
if [[ -f "$RUN_DIR/snapshots/snapshot.pth.tar" ]]; then
  TRAIN_ARGS+=(--resume)
fi

"$PYTHON" experiments/geotransformer.ircadb_v3/trainval.py \
  "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/train_console.log"

SNAPSHOT="$RUN_DIR/snapshots/best.pth.tar"
if [[ ! -f "$SNAPSHOT" ]]; then
  echo "Missing best validation checkpoint: $SNAPSHOT" >&2
  exit 1
fi
"$PYTHON" experiments/geotransformer.ircadb_v3/test.py \
  --snapshot "$SNAPSHOT" --output "$RUN_DIR/test_summary.json" \
  2>&1 | tee -a "$RUN_DIR/test_console.log"
