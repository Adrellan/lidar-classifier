#!/usr/bin/env python3
"""
Lokális train launcher (Open3D-ML), ami futáskor regisztrálja a custom HungaryLidar
datasetet, így nem kell a vendég Open3D-ML repo-t módosítani.

Használat:
  source .venv/bin/activate
  DATASET_PATH=exp/tiles_pcd python scripts/train_local.py
  # vagy LAS/LAZ csempékre:
  # DATASET_PATH=exp/tiles python scripts/train_local.py
"""
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
custom_paths = [
    REPO_ROOT,
    REPO_ROOT / "Open3D-ML",
]

# Ensure local sources shadow any globally installed ml3d package.
for path in reversed(custom_paths):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

from datasets.hungary_lidar import HungaryLidar

# Regisztráljuk a custom datasetet az összes elérhető registry-be
dataset_registries = []
try:
    from ml3d import utils as local_utils
except ImportError:
    local_utils = None
else:
    dataset_registries.append(local_utils.DATASET)

try:
    import open3d.ml as o3d_ml
except ImportError:
    o3d_ml = None
else:
    dataset_registries.append(o3d_ml.utils.DATASET)

if not dataset_registries:
    raise ImportError("Nem található Open3D-ML dataset registry a HungaryLidar regisztrálásához.")

for registry in dataset_registries:
    registry._register_module(HungaryLidar, module_name="HungaryLidar")

ds_path = os.getenv("DATASET_PATH", "exp/tiles")

sys.argv = [
    "run_pipeline.py",
    "torch",
    "-c",
    "configs/randla_hungary.yaml",
    "--device",
    "cuda",
    "--dataset.dataset_path",
    ds_path,
]
runpy.run_path(
    str(REPO_ROOT / "Open3D-ML" / "scripts" / "run_pipeline.py"),
    run_name="__main__",
)
