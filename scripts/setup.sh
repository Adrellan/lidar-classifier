#!/usr/bin/env bash
set -euo pipefail

# Egyszerű helyi környezet telepítő CUDA-s szerverre (Docker nélkül).
# Futás: bash scripts/setup.sh

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

if [ ! -d "$VENV_DIR" ]; then
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# Torch + CUDA páros és egyéb ML függőségek az Open3D-ML saját listájából
pip install -r Open3D-ML/requirements-torch-cuda.txt

# Open3D kerék
pip install open3d

# Open3D-ML kód a lokális repóból
pip install -e Open3D-ML/ml3d

# LAZ/LAS kezelő és hasznos libek + TensorBoard
pip install "laspy[lazrs]" numpy pyyaml tqdm shapely tensorboard

echo "Kész. Aktiváld: source $VENV_DIR/bin/activate"
