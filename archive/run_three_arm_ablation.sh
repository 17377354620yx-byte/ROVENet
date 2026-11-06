#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/run_three_arm_ablation.sh SEED" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for ARM in rtor a3 a3_geometry_off; do
  echo "=== Seed $1, arm $ARM ==="
  bash "$PROJECT_ROOT/scripts/run_module_ablation.sh" "$ARM" "$1"
done
