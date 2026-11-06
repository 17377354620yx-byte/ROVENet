#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/test_cooperative_seeded.sh SEED [EPOCH]" >&2
  exit 2
fi
SEED="$1"
EPOCH="${2:-150}"
if [[ "$EPOCH" != best && ! "$EPOCH" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCH must be a positive integer or best" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_RUN_NAME="${P2P_TRAIN_RUN_NAME:-rtor_a3_cooperative_cc68_seed${SEED}}"
export P2P_RUN_NAME="${P2P_TEST_RUN_NAME:-${TRAIN_RUN_NAME}_epoch${EPOCH}}"
if [[ "$EPOCH" == best ]]; then
  SNAPSHOT_NAME="best.pth.tar"
else
  SNAPSHOT_NAME="epoch-${EPOCH}.pth.tar"
fi
export P2P_SNAPSHOT="${P2P_SNAPSHOT:-${PROJECT_ROOT}/output/geotransformer.p2p_liver.${TRAIN_RUN_NAME}/snapshots/${SNAPSHOT_NAME}}"

if [[ ! -f "$P2P_SNAPSHOT" ]]; then
  echo "Missing trained checkpoint: $P2P_SNAPSHOT" >&2
  exit 1
fi

echo "Checkpoint: $P2P_SNAPSHOT"
echo "Results: output/geotransformer.p2p_liver.${P2P_RUN_NAME}/"
exec bash scripts/test_cooperative.sh
