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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(REPO_ROOT), str(REPO_ROOT / "Open3D-ML")])  # látja a saját kódot és az O3D-ML-t

from ml3d.utils import DATASET  # Registry
from datasets.hungary_lidar import HungaryLidar
from ml3d.tools import run_pipeline

# Regisztráljuk a custom datasetet a torch registry-be
DATASET.setdefault("torch", {})["HungaryLidar"] = HungaryLidar

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

run_pipeline.main()
