#!/usr/bin/env bash
set -euo pipefail

# Egyszerű helyi környezet telepítő CUDA-s szerverre (Docker nélkül).
# Futás: bash scripts/setup.sh

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}
O3D_ML_DIR=${O3D_ML_DIR:-Open3D-ML}

if [ ! -d "$VENV_DIR" ]; then
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# Ha nincs meg az Open3D-ML repo, klónozzuk
if [ ! -d "$O3D_ML_DIR" ]; then
  echo ">>> Cloning Open3D-ML into $O3D_ML_DIR"
  git clone --depth 1 https://github.com/isl-org/Open3D-ML.git "$O3D_ML_DIR"
fi

# Torch + CUDA páros és egyéb ML függőségek az Open3D-ML saját listájából
pip install -r "$O3D_ML_DIR/requirements-torch-cuda.txt"

# Open3D kerék
pip install open3d

# Open3D-ML kód a lokális repóból
pip install -e "$O3D_ML_DIR/ml3d"

# LAZ/LAS kezelő és hasznos libek + TensorBoard
pip install "laspy[lazrs]" numpy pyyaml tqdm shapely tensorboard

echo "Kész. Aktiváld: source $VENV_DIR/bin/activate"
