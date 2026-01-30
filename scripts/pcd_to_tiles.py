#!/usr/bin/env python3
"""
Gyors PCD -> Open3D-ML csempe konverter.
Feltételezés: PCD-ben oszlopok: x y z label (ASCII vagy bináris).
"""
import argparse
import numpy as np
from pathlib import Path
import open3d as o3d
import json

CLASS_MAP = {2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 14: 4, 16: 4, 17: 4, 18: 4, 19: 4}
CLASS_NAMES = ["ground", "vegetation", "building", "noise", "other"]


def load_pcd(path: Path):
    pc = o3d.io.read_point_cloud(str(path))
    if not pc.has_points():
        raise ValueError(f"Üres PCD: {path}")
    pts = np.asarray(pc.points)
    # Label olvasás: feltételezzük, hogy a 'label' mezőt intensity-ként tárolják
    # Ha a PCD-ben tényleges scalar field van, open3d nem mindig tölti be.
    # Megoldás: ha van külön text label fájl, illeszteni kell.
    # Itt: ha nincs label, akkor hibázunk.
    if pc.has_point_colors():
        labels = np.asarray(pc.colors)[:, 0]  # gyakori trükk, de ritkán igaz
    else:
        raise ValueError("A PCD-ben nem található label mező. Add meg, hogyan tárolod a címkéket.")
    labels = labels.astype(int)
    labels = np.vectorize(CLASS_MAP.get)(labels)
    return pts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="PCD mappa")
    ap.add_argument("--output", required=True, help="Csempe kimenet")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcds = sorted(in_dir.glob("*.pcd"))
    if not pcds:
        raise SystemExit("Nincs PCD a bemeneti mappában.")

    names = []
    for i, p in enumerate(pcds):
        pts, lbl = load_pcd(p)
        name = f"pcd_{i:05d}"
        np.save(out_dir / f"{name}_points.npy", pts.astype(np.float32))
        np.save(out_dir / f"{name}_labels.npy", lbl.astype(np.int64))
        names.append(name)

    # egyszerű 80/10/10 split
    rng = np.random.default_rng(42)
    rng.shuffle(names)
    n = len(names)
    ntr = int(n * 0.8)
    nv = int(n * 0.1)
    (out_dir / "train.txt").write_text("\n".join(names[:ntr]))
    (out_dir / "val.txt").write_text("\n".join(names[ntr:ntr + nv]))
    (out_dir / "test.txt").write_text("\n".join(names[ntr + nv:]))

    meta = {
        "dataset": "pcd_tiles",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "class_map": CLASS_MAP,
        "source": "PCD"
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Kész: {len(names)} csempe -> {out_dir}")


if __name__ == "__main__":
    main()
