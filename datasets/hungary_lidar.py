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
        canonical = {
            "train": "train",
            "training": "train",
            "val": "val",
            "validation": "val",
            "test": "test",
        }
        key = canonical.get(split)
        if key is None:
            raise KeyError(f"Ismeretlen split: {split}")
        fname = {
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
        }[key]
        return [l.strip() for l in (self.dataset_path / fname).read_text().splitlines() if l.strip()]

    @staticmethod
    def get_label_to_names():
        return {0: "ground", 1: "vegetation", 2: "building", 3: "noise", 4: "other"}

    # Tesztelés/eredmény mentés a pred label-ekkel; a kimenet npy és (ha elérhető) színezett PLY.
    def is_tested(self, attr):
        return False

    def save_test_result(self, results, attr):
        name = attr["name"]
        base = self.dataset_path

        # results lehet dict a predict_labels kulccsal vagy közvetlen array
        pred = results.get("predict_labels", results)
        pred = np.asarray(pred)
        np.save(base / f"{name}_pred.npy", pred)

        # Próbáljunk színezett PLY-t is írni, ha van open3d
        try:
            import open3d as o3d  # type: ignore

            pts = np.load(base / f"{name}_points.npy")
            palette = np.array([
                [0, 176, 80],    # ground
                [0, 112, 255],   # vegetation
                [255, 0, 0],     # building
                [255, 192, 0],   # noise
                [160, 160, 160], # other
            ], dtype=np.float32) / 255.0
            colors = palette[pred % len(palette)]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(base / f"{name}_pred.ply"), pcd, compressed=True)
        except Exception:
            # Ha nincs open3d vagy nem sikerült, legalább a pred.npy megvan.
            pass


class HungaryLidarSplit(BaseDatasetSplit):
    def __init__(self, dataset, split="train"):
        super().__init__(dataset, split=split)
        self.split = split
        self.items = list(self.path_list)

    def __len__(self):
        return len(getattr(self, "items", self.path_list))

    def get_data(self, idx):
        name = self.items[idx]
        base = self.dataset.dataset_path
        pts = np.load(base / f"{name}_points.npy")  # (N,3) float32
        labels = np.load(base / f"{name}_labels.npy")  # (N,) int64
        return {
            "point": pts,
            "label": labels,
            "feat": None,
        }

    def get_attr(self, idx):
        name = self.items[idx]
        return {
            "name": name,
            "path": str(self.dataset.dataset_path),
            "split": self.split,
        }
