#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! "$2" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/run_module_ablation.sh {baseline|rtor|a3|a3_geometry_off|joint|geometry_off} SEED" >&2
  exit 2
fi
ARM="$1"
SEED="$2"
case "$ARM" in
  baseline) ARCHITECTURE=geotransformer; GEOMETRY_WEIGHT=1 ;;
  rtor) ARCHITECTURE=rtor_only; GEOMETRY_WEIGHT=1 ;;
  a3) ARCHITECTURE=a3_only; GEOMETRY_WEIGHT=1 ;;
  a3_geometry_off) ARCHITECTURE=a3_only; GEOMETRY_WEIGHT=0 ;;
  joint) ARCHITECTURE=rtor_a3; GEOMETRY_WEIGHT=1 ;;
  geometry_off) ARCHITECTURE=rtor_a3; GEOMETRY_WEIGHT=0 ;;
  *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
unset P2P_TEST_LIMIT P2P_TEST_INDICES

RUN_NAME="ablate_coop_${ARM}_seed${SEED}"
RUN_DIR="output/geotransformer.p2p_liver.${RUN_NAME}"
CHECKPOINT="${RUN_DIR}/snapshots/epoch-150.pth.tar"
if [[ ! -f "$CHECKPOINT" ]]; then
  TRAIN_ARGS=(--architecture "$ARCHITECTURE" --interaction_profile cooperative
              --seed "$SEED" --max_epoch 150 --lr 1e-4 --log_steps 10
              --geometry_weight "$GEOMETRY_WEIGHT")
  if [[ -f "${RUN_DIR}/snapshots/snapshot.pth.tar" ]]; then
    TRAIN_ARGS+=(--resume)
  fi
  P2P_RUN_NAME="$RUN_NAME" conda run --no-capture-output -n geo_v2 \
    python experiments/geotransformer.p2p_liver/trainval.py "${TRAIN_ARGS[@]}"
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training did not produce $CHECKPOINT" >&2
  exit 1
fi

RESULT_DIR="output/geotransformer.p2p_liver.${RUN_NAME}_epoch150"
mkdir -p "$RESULT_DIR"
for NOISE in none 2 4; do
  P2P_RUN_NAME="${RUN_NAME}_epoch150" conda run --no-capture-output -n geo_v2 \
    python experiments/geotransformer.p2p_liver/test.py \
    --architecture "$ARCHITECTURE" --interaction_profile cooperative \
    --snapshot "$CHECKPOINT" --dataset in_silico --noise "$NOISE" \
    --output "${RESULT_DIR}/in_silico_noise_${NOISE}.json"
done
P2P_RUN_NAME="${RUN_NAME}_epoch150" conda run --no-capture-output -n geo_v2 \
  python experiments/geotransformer.p2p_liver/test.py \
  --architecture "$ARCHITECTURE" --interaction_profile cooperative \
  --snapshot "$CHECKPOINT" --dataset in_vitro --noise none \
  --output "${RESULT_DIR}/in_vitro.json"
echo "Completed $ARM seed $SEED: $RESULT_DIR"
