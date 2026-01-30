#!/usr/bin/env python3
"""
PCD -> Open3D-ML Numpy csempe konverter.
Támogat: ASCII, binary, binary_compressed PCD (PCL konverzióval).
Mezők: x, y, z, classification.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np

CLASS_MAP = {2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 14: 4, 16: 4, 17: 4, 18: 4, 19: 4}
CLASS_NAMES = ["ground", "vegetation", "building", "noise", "other"]


def detect_data_type(path: Path) -> str:
    with open(path, "rb") as f:
        for line in f:
            low = line.strip().lower()
            if low.startswith(b"data"):
                parts = low.split()
                if len(parts) >= 2:
                    return parts[1].decode("ascii")
                return "ascii"
            if f.tell() > 4096:
                break
    return "ascii"


def convert_to_ascii(src: Path) -> Path:
    """PCL-lel ASCII-ra konvertál (fmt=1)."""
    pcl = subprocess.run(["which", "pcl_convert_pcd_ascii_binary"], capture_output=True, text=True)
    if pcl.returncode != 0:
        raise SystemExit("pcl_convert_pcd_ascii_binary nem elérhető; telepítsd: sudo apt-get install pcl-tools")
    tmpdir = Path(tempfile.mkdtemp())
    dst = tmpdir / (src.stem + "_ascii.pcd")
    subprocess.check_call(["pcl_convert_pcd_ascii_binary", str(src), str(dst), "1"])
    return dst


def load_ascii(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    hdr = {}
    data_start = None
    for i, l in enumerate(lines):
        low = l.lower()
        if low.startswith("data"):
            data_start = i + 1
            break
        parts = l.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "fields":
            hdr["fields"] = [p.lower() for p in parts[1:]]
    if data_start is None:
        raise ValueError("DATA sor hiányzik")
    fields = hdr.get("fields", [])
    xi = fields.index("x")
    yi = fields.index("y")
    zi = fields.index("z")
    li = fields.index("classification")

    data = np.loadtxt(lines[data_start:])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    pts = data[:, [xi, yi, zi]].astype(np.float32)
    lbl = data[:, li].astype(np.int64)
    lbl = np.array([CLASS_MAP.get(int(v), 4) for v in lbl], dtype=np.int64)
    return pts, lbl


def load_pcd(path: Path):
    dtype = detect_data_type(path)
    if dtype == "ascii":
        return load_ascii(path)
    else:
        ascii_path = convert_to_ascii(path)
        return load_ascii(ascii_path)


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
        np.save(out_dir / f"{name}_points.npy", pts)
        np.save(out_dir / f"{name}_labels.npy", lbl)
        names.append(name)

    rng = np.random.default_rng(42)
    rng.shuffle(names)
    n = len(names)
    ntr = int(0.8 * n)
    nv = int(0.1 * n)
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
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Kész: {len(names)} csempe -> {out_dir}")


if __name__ == "__main__":
    main()
