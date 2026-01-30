#!/usr/bin/env python3
"""
Gyors PCD -> Open3D-ML csempe konverter.
Feltételezés: PCD-ben oszlopok: x y z label (ASCII vagy bináris).
"""
import argparse
import json
from pathlib import Path
import numpy as np

CLASS_MAP = {2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 14: 4, 16: 4, 17: 4, 18: 4, 19: 4}
CLASS_NAMES = ["ground", "vegetation", "building", "noise", "other"]


def parse_header_bytes(b: bytes):
    """Parses PCD header from raw bytes. Returns (meta, data_offset)."""
    lines = b.splitlines()
    header_lines = []
    data_offset = None
    for i, ln in enumerate(lines):
        if ln.startswith(b"DATA") or ln.startswith(b"data"):
            header_lines = lines[: i + 1]
            data_offset = len(b"\n".join(lines[: i + 1])) + 1
            break
    if data_offset is None:
        raise ValueError("DATA line not found")

    meta = {}
    for l in header_lines:
        parts = l.decode("ascii").strip().split()
        if not parts:
            continue
        key = parts[0].lower()
        meta[key] = parts[1:]
    return meta, data_offset


def header_indices(meta):
    fields = meta.get("fields", [])
    if not fields:
        raise ValueError("FIELDS missing")
    try:
        xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError:
        raise ValueError("x/y/z field missing")
    label_keys = ["label", "class", "classification"]
    li = None
    for k in label_keys:
        if k in fields:
            li = fields.index(k)
            break
    if li is None:
        raise ValueError("No label/classification field in FIELDS")
    return fields, xi, yi, zi, li


def make_dtype(meta):
    fields = meta["fields"]
    sizes = list(map(int, meta.get("size", [])))
    types = meta.get("type", [])
    counts = list(map(int, meta.get("count", ["1"] * len(fields))))
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("Header length mismatch in FIELDS/SIZE/TYPE/COUNT")
    np_types = []
    for s, t, c in zip(sizes, types, counts):
        if t.lower() == "f":
            base = {4: "f4", 8: "f8"}.get(s)
        elif t.lower() == "u":
            base = {1: "u1", 2: "u2", 4: "u4"}.get(s)
        elif t.lower() == "i":
            base = {1: "i1", 2: "i2", 4: "i4"}.get(s)
        else:
            base = None
        if base is None:
            raise ValueError(f"Unsupported type/size: {t}{s}")
        if c == 1:
            np_types.append(base)
        else:
            np_types.append((base, c))
    return np_types


def load_pcd(path: Path):
    raw = path.read_bytes()
    meta, data_offset = parse_header_bytes(raw)
    fields, xi, yi, zi, li = header_indices(meta)
    data_type = meta.get("data", ["ascii"])[0].lower()
    if data_type == "ascii":
        lines = raw[data_offset:].decode("utf-8").splitlines()
        data = np.loadtxt(lines)
        if data.ndim == 1:
            data = data.reshape(1, -1)
    elif data_type == "binary":
        dtype = make_dtype(meta)
        data = np.frombuffer(raw, offset=data_offset, dtype=np.dtype({"names": fields, "formats": dtype}))
    else:
        raise ValueError(f"DATA {data_type} nem támogatott (binary_compressed esetén konvertálni kell)")

    if data_type == "binary":
        x = np.asarray(data[fields[xi]], dtype=np.float32)
        y = np.asarray(data[fields[yi]], dtype=np.float32)
        z = np.asarray(data[fields[zi]], dtype=np.float32)
        lbl = np.asarray(data[fields[li]], dtype=np.int64)
        pts = np.vstack([x, y, z]).T
    else:
        pts = data[:, [xi, yi, zi]]
        lbl = data[:, li].astype(np.int64)

    lbl = np.vectorize(CLASS_MAP.get, otypes=[int])(lbl)
    return pts, lbl


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
