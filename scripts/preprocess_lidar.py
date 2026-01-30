#!/usr/bin/env python3
"""
LAZ/LAS csempézés és Open3D-ML kompatibilis .npy mentés.
Egyszerű rács-alapú 2D tile-olás fix mérettel (pl. 50 m).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import laspy
import numpy as np

# ASPRS -> belső 0..4 osztálykódok
CLASS_MAP = {
    2: 0,             # ground
    3: 1, 4: 1, 5: 1, # vegetation
    6: 2,             # building
    7: 3,             # noise
    14: 4, 16: 4, 17: 4, 18: 4, 19: 4  # other
}
DEFAULT_CLASS_NAMES = ["ground", "vegetation", "building", "noise", "other"]


def read_bounds(las_paths):
    mins_x, mins_y, maxs_x, maxs_y = [], [], [], []
    for p in las_paths:
        with laspy.open(p) as fh:
            h = fh.header
            mins_x.append(h.mins[0]); maxs_x.append(h.maxs[0])
            mins_y.append(h.mins[1]); maxs_y.append(h.maxs[1])
    return min(mins_x), min(mins_y), max(maxs_x), max(maxs_y)


def map_classes(raw_cls):
    mapped = np.full(raw_cls.shape, 4, dtype=np.int64)  # default: other
    for k, v in CLASS_MAP.items():
        mapped[raw_cls == k] = v
    return mapped


def main(args):
    las_files = sorted([p for p in Path(args.input).glob("*.las")] +
                       [p for p in Path(args.input).glob("*.laz")])
    if not las_files:
        raise SystemExit("Nincs .las vagy .laz bemenet a megadott könyvtárban.")

    xmin, ymin, xmax, ymax = read_bounds(las_files)
    tile_size = float(args.tile)
    gx = math.ceil((xmax - xmin) / tile_size)
    gy = math.ceil((ymax - ymin) / tile_size)

    # tile_id -> list of chunks
    buckets = defaultdict(lambda: {"points": [], "labels": []})

    for path in las_files:
        las = laspy.read(path)  # laspy.read nem context manager
        x, y, z = las.x, las.y, las.z
        raw_cls = las.classification
        ix = np.floor((x - xmin) / tile_size).astype(int)
        iy = np.floor((y - ymin) / tile_size).astype(int)
        # pontok a legszélén: klippeljük rácsra
        ix = np.clip(ix, 0, gx - 1)
        iy = np.clip(iy, 0, gy - 1)
        tile_id = ix + iy * gx

        mapped = map_classes(raw_cls)
        pts = np.stack((x, y, z), axis=1)

        uniq_ids, inverse = np.unique(tile_id, return_inverse=True)
        for uid in uniq_ids:
            m = tile_id == uid
            buckets[int(uid)]["points"].append(pts[m])
            buckets[int(uid)]["labels"].append(mapped[m])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tile_names = []

    for uid, data in buckets.items():
        pts = np.concatenate(data["points"], axis=0)
        lbl = np.concatenate(data["labels"], axis=0)
        if args.max_points and pts.shape[0] > args.max_points:
            idx = np.random.choice(pts.shape[0], args.max_points, replace=False)
            pts = pts[idx]; lbl = lbl[idx]

        ix = uid % gx
        iy = uid // gx
        name = f"tile_{ix:04d}_{iy:04d}"
        np.save(out_dir / f"{name}_points.npy", pts.astype(np.float32))
        np.save(out_dir / f"{name}_labels.npy", lbl.astype(np.int64))
        all_tile_names.append(name)

    # egyszerű shuffle + split
    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_tile_names)
    n = len(all_tile_names)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train = all_tile_names[:n_train]
    val = all_tile_names[n_train:n_train + n_val]
    test = all_tile_names[n_train + n_val:]

    for fname, items in [("train.txt", train), ("val.txt", val), ("test.txt", test)]:
        (out_dir / fname).write_text("\n".join(items))

    meta = {
        "dataset": "hungary_lidar",
        "num_classes": len(DEFAULT_CLASS_NAMES),
        "class_names": DEFAULT_CLASS_NAMES,
        "class_map": CLASS_MAP,
        "tile_size_m": tile_size,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Csempék száma: {len(all_tile_names)} | train {len(train)}, val {len(val)}, test {len(test)}")
    print(f"Kimenet: {out_dir.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Könyvtár, benne .las/.laz fájlok")
    p.add_argument("--output", required=True, help="Kimeneti könyvtár a csempéknek")
    p.add_argument("--tile", type=float, default=50.0, help="Csempe méret méterben (alap: 50)")
    p.add_argument("--max_points", type=int, default=200000,
                   help="Max pont/csempe (random mintavételezés). 0 = korlátlan")
    p.add_argument("--train_ratio", type=float, default=0.7, help="Train arány")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Val arány")
    args = p.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio < 1.0 legyen, a maradék a test.")
    main(args)
