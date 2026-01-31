#!/usr/bin/env python3
"""
One-stop script to:
 - pick raw data from train_data/las_laz or train_data/pcd
 - convert PCD -> LAS if needed
 - tile the LAS/LAZ files into Open3D-ML ready .npy chunks
 - launch Open3D-ML training with the supplied config.yaml

Usage:
  python scripts/train.py --las
  python scripts/train.py --pcd
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import laspy
import numpy as np
import runpy
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_ROOT = REPO_ROOT / "train_data"
RUNS_ROOT = REPO_ROOT / "runs"

# ASPRS -> internal 0..4 class map
CLASS_MAP = {
    2: 0,             # ground
    3: 1, 4: 1, 5: 1, # vegetation
    6: 2,             # building
    7: 3,             # noise
    14: 4, 16: 4, 17: 4, 18: 4, 19: 4  # other
}
CLASS_NAMES = ["ground", "vegetation", "building", "noise", "other"]


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def ensure_data_roots():
    for sub in ("pcd", "las_laz"):
        (TRAIN_DATA_ROOT / sub).mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def find_files(folder: Path, exts):
    files = []
    for ext in exts:
        files.extend(folder.glob(f"*.{ext}"))
    return sorted(files)


def convert_pcd_if_needed(pcd_dir: Path) -> Path:
    """Return a directory that contains LAS files for training.

    Reuses earlier conversions and only converts new/changed PCD-k.
    """
    pcd_files = find_files(pcd_dir, ("pcd",))
    las_in_root = find_files(pcd_dir, ("las", "laz"))
    if not pcd_files and not las_in_root:
        raise FileNotFoundError("Nincs .pcd vagy .las/.laz a train_data/pcd alatt.")

    if shutil.which("pdal") is None:
        raise SystemExit("PDAL CLI is required for PCD->LAS conversion (missing `pdal`).")

    out_dir = pcd_dir / "_converted_las"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Másold át a már létező LAS/LAZ-okat a gyökérből (ha vannak).
    copied = 0
    for src in las_in_root:
        dst = out_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    # 2) Konvertáld azokat a PCD-ket, amelyekhez még nincs LAS.
    converted = 0
    for src in pcd_files:
        dst = out_dir / f"{src.stem}.las"
        if dst.exists():
            continue
        print(f"[pcd->las] {src.name} -> {dst.name}")
        ok, err = pdal_translate(src, dst)
        if not ok:
            if "Binary compressed PCD is not supported" in err:
                ok = convert_via_ascii(src, dst)
            else:
                raise SystemExit(f"PDAL translate failed for {src.name}: {err}")
        if ok:
            converted += 1

    las_ready = find_files(out_dir, ("las", "laz"))
    if not las_ready:
        raise SystemExit("Conversion produced no LAS files; check PCD inputs.")

    print(f"LAS-ok a használathoz: {len(las_ready)} | újonnan konvertált: {converted} | átmásolt: {copied}")
    return out_dir


def pdal_translate(src: Path, dst: Path):
    proc = subprocess.run(["pdal", "translate", str(src), str(dst)],
                          capture_output=True, text=True)
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "").strip()


def convert_via_ascii(src: Path, dst: Path) -> bool:
    """Fallback: decompress binary-compressed PCD with PCL, then PDAL translate."""
    if shutil.which("pcl_convert_pcd_ascii_binary") is None:
        raise SystemExit(
            "Binary compressed PCD és nincs pcl_convert_pcd_ascii_binary. Telepítsd a pcl-tools csomagot."
        )
    tmpdir = Path(tempfile.mkdtemp(prefix="pcd_ascii_"))
    ascii_path = tmpdir / f"{src.stem}_ascii.pcd"
    try:
        subprocess.run(
            ["pcl_convert_pcd_ascii_binary", str(src), str(ascii_path), "1"],
            check=True,
            capture_output=True,
            text=True,
        )
        ok, err = pdal_translate(ascii_path, dst)
        if not ok:
            raise SystemExit(f"PDAL translate (ASCII) failed for {src.name}: {err}")
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def map_classes(raw_cls: np.ndarray) -> np.ndarray:
    mapped = np.full(raw_cls.shape, 4, dtype=np.int64)  # default: other
    for k, v in CLASS_MAP.items():
        mapped[raw_cls == k] = v
    return mapped


def preprocess_las_dir(las_dir: Path, out_dir: Path, tile_size: float,
                       max_points: int, train_ratio: float, val_ratio: float):
    las_files = find_files(las_dir, ("las", "laz"))
    if not las_files:
        raise FileNotFoundError(f"No .las/.laz files found under {las_dir}")

    print(f"Tiling {len(las_files)} LAS/LAZ files from {las_dir} ...")

    # fresh output
    if out_dir.exists() and out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # bounds across all files
    mins_x, mins_y, maxs_x, maxs_y = [], [], [], []
    for p in las_files:
        with laspy.open(p) as fh:
            h = fh.header
            mins_x.append(h.mins[0]); maxs_x.append(h.maxs[0])
            mins_y.append(h.mins[1]); maxs_y.append(h.maxs[1])
    xmin, ymin, xmax, ymax = min(mins_x), min(mins_y), max(maxs_x), max(maxs_y)

    gx = max(1, math.ceil((xmax - xmin) / tile_size))
    gy = max(1, math.ceil((ymax - ymin) / tile_size))

    buckets = defaultdict(lambda: {"points": [], "labels": []})

    for path in las_files:
        las = laspy.read(path)  # laspy.read is not a context manager
        x, y, z = las.x, las.y, las.z
        raw_cls = las.classification
        ix = np.floor((x - xmin) / tile_size).astype(int)
        iy = np.floor((y - ymin) / tile_size).astype(int)
        ix = np.clip(ix, 0, gx - 1)
        iy = np.clip(iy, 0, gy - 1)
        tile_id = ix + iy * gx

        mapped = map_classes(raw_cls)
        pts = np.stack((x, y, z), axis=1)

        uniq_ids = np.unique(tile_id)
        for uid in uniq_ids:
            m = tile_id == uid
            buckets[int(uid)]["points"].append(pts[m])
            buckets[int(uid)]["labels"].append(mapped[m])

    all_tile_names = []
    for uid, data in buckets.items():
        pts = np.concatenate(data["points"], axis=0)
        lbl = np.concatenate(data["labels"], axis=0)
        if max_points and pts.shape[0] > max_points:
            idx = np.random.default_rng(seed=42).choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]; lbl = lbl[idx]

        ix = uid % gx
        iy = uid // gx
        name = f"tile_{ix:04d}_{iy:04d}"
        np.save(out_dir / f"{name}_points.npy", pts.astype(np.float32))
        np.save(out_dir / f"{name}_labels.npy", lbl.astype(np.int64))
        all_tile_names.append(name)

    rng = np.random.default_rng(seed=42)
    rng.shuffle(all_tile_names)
    n = len(all_tile_names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = all_tile_names[:n_train]
    val = all_tile_names[n_train:n_train + n_val]
    test = all_tile_names[n_train + n_val:]

    for fname, items in [("train.txt", train), ("val.txt", val), ("test.txt", test)]:
        (out_dir / fname).write_text("\n".join(items))

    meta = {
        "dataset": "hungary_lidar",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "class_map": CLASS_MAP,
        "tile_size_m": tile_size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Tiles written to {out_dir} | train {len(train)}, val {len(val)}, test {len(test)}")


def register_dataset():
    custom_paths = [REPO_ROOT, REPO_ROOT / "Open3D-ML"]
    for path in reversed(custom_paths):
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)

    from datasets.hungary_lidar import HungaryLidar

    registries = []
    try:
        from ml3d import utils as local_utils  # type: ignore
    except ImportError:
        local_utils = None
    else:
        registries.append(local_utils.DATASET)

    try:
        import open3d.ml as o3d_ml  # type: ignore
    except ImportError:
        o3d_ml = None
    else:
        registries.append(o3d_ml.utils.DATASET)

    if not registries:
        raise ImportError("Open3D-ML dataset registry not found; check environment.")

    for registry in registries:
        registry._register_module(HungaryLidar, module_name="HungaryLidar")


def start_tensorboard(log_dir: Path, host: str = "0.0.0.0", port: int = 6006):
    if shutil.which("tensorboard") is None:
        print("TensorBoard nem elérhető (parancs: tensorboard). Telepítsd a csomagot, ha kell.")
        return
    tb_dir = log_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "tensorboard",
        "--logdir",
        str(tb_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"TensorBoard elindítva: http://{host}:{port}  (logdir: {tb_dir})")


def launch_training(config_path: Path, tiles_dir: Path, cache_dir: Path,
                    log_dir: Path, device: str, start_tb: bool = False,
                    tb_port: int = 6006):
    register_dataset()

    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

    if start_tb:
        start_tensorboard(log_dir, port=tb_port)

    sys.argv = [
        "run_pipeline.py",
        "torch",
        "-c",
        str(config_path),
        "--device",
        device,
        "--dataset.dataset_path",
        str(tiles_dir),
        "--dataset.cache_dir",
        str(cache_dir),
        "--pipeline.main_log_dir",
        str(log_dir),
        "--pipeline.train_sum_dir",
        str(log_dir / "tensorboard"),
    ]

    runpy.run_path(
        str(REPO_ROOT / "Open3D-ML" / "scripts" / "run_pipeline.py"),
        run_name="__main__",
    )


def main():
    parser = argparse.ArgumentParser(description="Single entrypoint for preprocessing + training.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--las", action="store_true", help="Use train_data/las_laz")
    group.add_argument("--pcd", action="store_true", help="Use train_data/pcd (auto-convert to LAS if needed)")
    parser.add_argument("--config", default="config.yaml", help="Training config file")
    parser.add_argument("--device", default="cuda", help="Device for training (cuda or cpu)")
    parser.add_argument("--tensorboard", action="store_true", help="Indítsd el automatikusan a TensorBoard-ot (port 6006).")
    parser.add_argument("--tb-port", type=int, default=6006, help="TensorBoard port (alap: 6006)")
    args = parser.parse_args()

    ensure_data_roots()
    cfg = load_config(Path(args.config))
    preprocess_cfg = cfg.get("preprocess", {})
    tile_size = float(preprocess_cfg.get("tile_size_m", 50.0))
    max_points = int(preprocess_cfg.get("max_points_per_tile", 200000))
    train_ratio = float(preprocess_cfg.get("train_ratio", 0.7))
    val_ratio = float(preprocess_cfg.get("val_ratio", 0.15))
    if train_ratio + val_ratio >= 1.0:
        raise SystemExit("preprocess.train_ratio + preprocess.val_ratio must be < 1.0")

    mode = "pcd" if args.pcd else "las"
    raw_dir = TRAIN_DATA_ROOT / ("pcd" if args.pcd else "las_laz")
    if args.pcd:
        las_dir = convert_pcd_if_needed(raw_dir)
    else:
        las_dir = raw_dir

    tiles_dir = RUNS_ROOT / mode / "tiles"
    cache_dir = RUNS_ROOT / mode / "cache"
    log_dir = RUNS_ROOT / mode / "logs"

    preprocess_las_dir(las_dir, tiles_dir, tile_size, max_points, train_ratio, val_ratio)
    launch_training(Path(args.config), tiles_dir, cache_dir, log_dir, args.device,
                    start_tb=args.tensorboard, tb_port=args.tb_port)


if __name__ == "__main__":
    main()
