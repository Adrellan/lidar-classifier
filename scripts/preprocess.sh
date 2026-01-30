#!/usr/bin/env bash
set -euo pipefail

# Csempézés futtatása a virtuális környezetből.
# Használat:
#   source .venv/bin/activate
#   bash scripts/preprocess.sh train_data exp/tiles

INPUT_DIR=${1:-train_data}
OUTPUT_DIR=${2:-exp/tiles}

python scripts/preprocess_lidar.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --tile 50 \
  --max_points 200000
