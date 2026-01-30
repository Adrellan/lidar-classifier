#!/usr/bin/env python3
"""
PCD -> Open3D-ML Numpy csempe konverter.
Kezeli: ASCII, binary, binary_compressed PCD (pcl_convert_pcd_ascii_binary segítségével).
Elvárt mezők: x, y, z, classification (ASPRS kódok).
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


def convert_pcd(src: Path, to_ascii: bool) -> Path:
    """PCL konvertálás ascii-ra (to_ascii=True) vagy binárisra (to_ascii=False)."""
    pcl = subprocess.run(["which", "pcl_convert_pcd_ascii_binary"], capture_output=True, text=True)
    if pcl.returncode != 0:
        raise SystemExit("pcl_convert_pcd_ascii_binary nem elérhető; telepítsd: sudo apt-get install pcl-tools")
    tmpdir = Path(tempfile.mkdtemp())
    dst = tmpdir / (src.stem + ("_ascii.pcd" if to_ascii else "_bin.pcd"))
    fmt = "1" if to_ascii else "0"
    subprocess.check_call(["pcl_convert_pcd_ascii_binary", str(src), str(dst), fmt])
    return dst


def make_struct_dtype(fields, sizes, types, counts):
    dt = []
    for name, sz, tp, cnt in zip(fields, sizes, types, counts):
        if tp.lower() == "f":
            base = {4: "f4", 8: "f8"}.get(int(sz))
        elif tp.lower() == "u":
            base = {1: "u1", 2: "u2", 4: "u4"}.get(int(sz))
        elif tp.lower() == "i":
            base = {1: "i1", 2: "i2", 4: "i4"}.get(int(sz))
        else:
            base = None
        if base is None:
            raise ValueError(f"Unsupported type/size: {tp}{sz}")
        cnt = int(cnt)
        if cnt == 1:
            dt.append((name, base))
        else:
            dt.append((name, base, cnt))
    return np.dtype(dt)


def parse_header_ascii(lines):
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
        elif key == "size":
            hdr["size"] = parts[1:]
        elif key == "type":
            hdr["type"] = parts[1:]
        elif key == "count":
            hdr["count"] = parts[1:]
    if data_start is None:
        raise ValueError("DATA sor hiányzik")
    return hdr, data_start


def load_ascii(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    hdr, data_start = parse_header_ascii(lines)
    fields = hdr.get("fields", [])
    xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
    li = fields.index("classification")
    data = np.loadtxt(lines[data_start:])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    pts = data[:, [xi, yi, zi]].astype(np.float32)
    lbl = data[:, li].astype(np.int64)
    lbl = np.array([CLASS_MAP.get(int(v), 4) for v in lbl], dtype=np.int64)
    return pts, lbl


def load_binary(path: Path):
    raw = path.read_bytes()
    # parse header bytes
    lines = raw.splitlines()
    header_lines = []
    data_offset = 0
    for i, ln in enumerate(lines):
        header_lines.append(ln)
        if ln.lower().startswith(b"data"):
            data_offset = len(b"\n".join(header_lines)) + 1
            break
    meta = {}
    for l in header_lines:
        parts = l.decode("ascii", errors="ignore").strip().split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "fields":
            meta["fields"] = [p.lower() for p in parts[1:]]
        elif key == "size":
            meta["size"] = parts[1:]
        elif key == "type":
            meta["type"] = parts[1:]
        elif key == "count":
            meta["count"] = parts[1:]
    fields = meta["fields"]
    xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
    li = fields.index("classification")
    sizes = list(map(int, meta["size"]))
    types = meta["type"]
    counts = list(map(int, meta.get("count", ["1"] * len(fields))))
    dt = make_struct_dtype(fields, sizes, types, counts)
    arr = np.frombuffer(raw, offset=data_offset, dtype=dt)
    x = np.asarray(arr[fields[xi]], dtype=np.float32)
    y = np.asarray(arr[fields[yi]], dtype=np.float32)
    z = np.asarray(arr[fields[zi]], dtype=np.float32)
    lbl_raw = np.asarray(arr[fields[li]])
    if lbl_raw.dtype.kind == "f":
        lbl_raw = np.rint(lbl_raw)
    lbl = np.array([CLASS_MAP.get(int(v), 4) for v in lbl_raw], dtype=np.int64)
    pts = np.vstack([x, y, z]).T
    return pts, lbl


def load_pcd(path: Path):
    dtype = detect_data_type(path)
    if dtype == "ascii":
        return load_ascii(path)
    elif dtype in ("binary", "binary_compressed"):
        # próbáljuk ASCII-ra konvertálni; ha az is binary marad, olvassuk binárisan
        ascii_path = convert_pcd(path, to_ascii=True)
        new_type = detect_data_type(ascii_path)
        if new_type == "ascii":
            return load_ascii(ascii_path)
        else:
            bin_path = convert_pcd(path, to_ascii=False)
            return load_binary(bin_path)
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
