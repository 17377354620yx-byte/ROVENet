#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/train_cooperative_seeded.sh SEED [trainval options]" >&2
  exit 2
fi
SEED="$1"
shift
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export P2P_RUN_NAME="${P2P_RUN_NAME:-rtor_a3_cooperative_cc68_seed${SEED}}"
RUN_DIR="output/geotransformer.p2p_liver.${P2P_RUN_NAME}"
RESUME=false
for ARG in "$@"; do
  if [[ "$ARG" == --resume ]]; then
    RESUME=true
  fi
done
if [[ "$RESUME" == false && -f "$RUN_DIR/snapshots/snapshot.pth.tar" ]]; then
  echo "Existing training snapshot in $RUN_DIR; choose another P2P_RUN_NAME or pass --resume" >&2
  exit 2
fi
exec bash scripts/train_cooperative.sh --seed "$SEED" "$@"
