#!/usr/bin/env bash
set -euo pipefail

# PCD -> LAS batch konverzió PDAL-lal.
# Használat:
#   bash scripts/pcd_to_las.sh pcd_data converted_las
# Alapértelmezés:
#   in_dir=pcd_data , out_dir=train_data_pcd_las

IN_DIR=${1:-pcd_data}
OUT_DIR=${2:-train_data_pcd_las}

mkdir -p "$OUT_DIR"

shopt -s nullglob
count=0
for f in "$IN_DIR"/*.pcd; do
  base=$(basename "${f%.pcd}")
  pdal translate "$f" "$OUT_DIR/$base.las"
  count=$((count+1))
done

echo "Kész: $count fájl konvertálva -> $OUT_DIR"
