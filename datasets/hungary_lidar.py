"""
Egyszerű Open3D-ML dataset wrapper a preprocesselt .npy csempékhez.
"""
import json
from pathlib import Path

import numpy as np
# A lokális Open3D-ML forrásból importálunk
from ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit


class HungaryLidar(BaseDataset):
    def __init__(self, dataset_path, name="HungaryLidar", cache_dir="cache", **kwargs):
        self.dataset_path = Path(dataset_path)
        with open(self.dataset_path / "meta.json") as f:
            meta = json.load(f)
        self.label_to_names = {i: n for i, n in enumerate(meta["class_names"])}
        self.name = name
        super().__init__(dataset_path=str(self.dataset_path),
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=False,
                         **kwargs)

    def get_split(self, split):
        return HungaryLidarSplit(self, split=split)

    def get_split_list(self, split):
        fname = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}[split]
        return [l.strip() for l in (self.dataset_path / fname).read_text().splitlines() if l.strip()]

    @staticmethod
    def get_label_to_names():
        return {0: "ground", 1: "vegetation", 2: "building", 3: "noise", 4: "other"}

    # Tesztelés/eredmény mentés nem használjuk; helykitöltő.
    def is_tested(self, attr):
        return False

    def save_test_result(self, results, attr):
        return


class HungaryLidarSplit(BaseDatasetSplit):
    def __init__(self, dataset, split="train"):
        super().__init__(dataset, split=split)
        self.split = split
        self.items = dataset.get_split_list(split)

    def __len__(self):
        return len(self.items)

    def get_data(self, idx):
        name = self.items[idx]
        base = self.dataset.dataset_path
        pts = np.load(base / f"{name}_points.npy")  # (N,3) float32
        labels = np.load(base / f"{name}_labels.npy")  # (N,) int64
        return {
            "points": pts,
            "labels": labels,
            "feat": None,
        }

    def get_attr(self, idx):
        name = self.items[idx]
        return {"name": name, "path": str(self.dataset.dataset_path)}
