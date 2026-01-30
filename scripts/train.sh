#!/usr/bin/env bash
set -euo pipefail

# Tanítás indítása (RandLA-Net) a virtuális környezetből.
# Használat:
#   source .venv/bin/activate
#   bash scripts/train.sh [--dataset_path <útvonal>]
# Alapértelmezett dataset_path a configs/randla_hungary.yaml-ban: exp/tiles

DATASET_PATH=""
if [[ "${1:-}" == "--dataset_path" ]]; then
  DATASET_PATH="$2"
  shift 2
fi

# biztosítsuk, hogy az Open3D-ML forrás elérhető legyen
export PYTHONPATH=$(pwd)/Open3D-ML:$(pwd):${PYTHONPATH:-}

CMD="python -m ml3d.tools.run_pipeline --cfg configs/randla_hungary.yaml --device cuda"
if [[ -n "$DATASET_PATH" ]]; then
  CMD+=" --override dataset.dataset_path=${DATASET_PATH}"
fi

echo ">> $CMD"
eval "$CMD"
