#!/usr/bin/env python3
"""Render a colorized prediction PLY to PNG (offscreen).

Usage:
  python scripts/render_pred.py exp/tiles_pcd/tile_0003_0002_pred.ply out.png
If out.png is omitted, saves next to the PLY as <name>_view.png.
"""
import sys
from pathlib import Path
import numpy as np
import open3d as o3d


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ply_path = Path(sys.argv[1]).expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2]).expanduser().resolve()
    else:
        out_path = ply_path.with_name(ply_path.stem + "_view.png")

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty: {ply_path}")

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = np.max(bbox.get_extent())

    width, height = 1280, 960
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("pcd", pcd, mat)

    cam = renderer.scene.camera
    cam.look_at(center, center + [0, 0, extent], [0, 1, 0])
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    cam.set_projection(60.0, width / height, 0.1, 1000.0, fov_type)

    img = renderer.render_to_image()
    o3d.io.write_image(str(out_path), img)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
