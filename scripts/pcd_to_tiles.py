#!/usr/bin/env python3
"""
Gyors PCD -> Open3D-ML csempe konverter.
Feltételezés: PCD-ben oszlopok: x y z label (ASCII vagy bináris).
"""
import argparse
import numpy as np
from pathlib import Path
import json

CLASS_MAP = {2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 14: 4, 16: 4, 17: 4, 18: 4, 19: 4}
CLASS_NAMES = ["ground", "vegetation", "building", "noise", "other"]


def load_pcd_ascii(path: Path):
    """ASCII PCD olvasó: FIELDS header alapján keresi az x,y,z és label/class mezőket."""
    lines = path.read_text().splitlines()
    header = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        header.append(line)
        if line.lower().startswith("data"):
            data_start = i + 1
            data_type = line.split()[1].lower()
            break
        i += 1
    else:
        raise ValueError(f"DATA sor nem található: {path}")

    if data_type != "ascii":
        raise ValueError(f"Csak ASCII PCD támogatott most: {path}")

    fields_line = next((l for l in header if l.lower().startswith("fields")), None)
    if not fields_line:
        raise ValueError(f"FIELDS sor hiányzik: {path}")
    fields = fields_line.split()[1:]

    try:
        x_idx = fields.index("x")
        y_idx = fields.index("y")
        z_idx = fields.index("z")
        if "label" in fields:
            l_idx = fields.index("label")
        elif "class" in fields:
            l_idx = fields.index("class")
        elif "classification" in fields:
            l_idx = fields.index("classification")
        else:
            raise ValueError("Nincs label/classification mező a FIELDS-ben")
    except ValueError as e:
        raise ValueError(f"Hiányzó mező: {e}")

    data = np.loadtxt(lines[data_start:])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    pts = data[:, [x_idx, y_idx, z_idx]]
    labels = data[:, l_idx].astype(int)
    labels = np.vectorize(CLASS_MAP.get, otypes=[int])(labels)
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
        pts, lbl = load_pcd_ascii(p)
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
