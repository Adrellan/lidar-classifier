#!/usr/bin/env python3
"""
Lokális train launcher, ami regisztrálja a HungaryLidar datasetet
és a meglévő Open3D-ML run_pipeline-t használja.

Használat:
  source .venv/bin/activate
  DATASET_PATH=exp/tiles_pcd python scripts/train_local.py
  # vagy a LAS/LAZ csempékre:
  # DATASET_PATH=exp/tiles python scripts/train_local.py
"""
import sys
import os

sys.path.extend([".", "Open3D-ML"])

from ml3d.datasets import DATASET
from datasets.hungary_lidar import HungaryLidar
from ml3d.tools import run_pipeline

# Regisztráljuk a custom datasetet a torch registry-be
DATASET.setdefault("torch", {})["HungaryLidar"] = HungaryLidar

# Építsük fel az argv-t a run_pipeline számára
args = [
    "run_pipeline.py",
    "torch",
    "-c",
    "configs/randla_hungary.yaml",
    "--device",
    "cuda",
]

ds_path = os.getenv("DATASET_PATH", "exp/tiles")
args += ["--dataset.dataset_path", ds_path]

sys.argv = args
run_pipeline.main()
