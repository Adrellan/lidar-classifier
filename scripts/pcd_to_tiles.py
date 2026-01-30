#!/usr/bin/env python3
"""
PCD -> Open3D-ML Numpy csempe konverter.
Támogat: ASCII, binary, binary_compressed PCD (pcl_convert_pcd_ascii_binary segítségével).
Elvárt mezők: x y z és classification (ASPRS kód).
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np

CLASS_MAP = {2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 14: 4, 16: 4, 17: 4, 18: 4, 19: 4}
CLASS_NAMES = ["ground", "vegetation", "building", "noise", "other"]


def convert_to_ascii(src: Path) -> Path:
    """Binary vagy binary_compressed PCD-t ASCII-ra konvertál PCL eszközzel."""
    pcl = subprocess.run(["which", "pcl_convert_pcd_ascii_binary"], capture_output=True, text=True)
    if pcl.returncode != 0:
        raise SystemExit("pcl_convert_pcd_ascii_binary nem elérhető; telepítsd: sudo apt-get install pcl-tools")
    tmpdir = Path(tempfile.mkdtemp())
    dst = tmpdir / (src.stem + "_ascii.pcd")
    subprocess.check_call(["pcl_convert_pcd_ascii_binary", str(src), str(dst), "1"])
    return dst


def load_ascii(path: Path):
    raw = path.read_bytes()
    lines = raw.decode("utf-8", errors="ignore").splitlines()
    hdr = {}
    data_start = None
    for i, l in enumerate(lines):
        if l.lower().startswith("data"):
            data_start = i + 1
            break
        parts = l.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "fields":
            hdr["fields"] = [p.lower() for p in parts[1:]]
    if data_start is None:
        raise ValueError(f"DATA sor hiányzik: {path}")
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
    # Döntsük el a DATA típust a header első 200 byte-jából
    head = path.read_bytes()[:200].lower()
    if b"data ascii" in head:
        return load_ascii(path)
    elif b"data binary" in head or b"data binary_compressed" in head:
        ascii_path = convert_to_ascii(path)
        return load_ascii(ascii_path)
    else:
        raise ValueError(f"Ismeretlen DATA formátum: {path}")


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

    # 80/10/10 split
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
