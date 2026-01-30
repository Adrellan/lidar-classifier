#!/usr/bin/env bash
set -euo pipefail

# Tanítás indítása (RandLA-Net) a virtuális környezetből.
# Használat:
#   source .venv/bin/activate
#   bash scripts/train.sh

python -m ml3d.tools.run_pipeline \
  --cfg configs/randla_hungary.yaml \
  --device cuda
