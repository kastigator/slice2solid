from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PointCloudExportResult:
    points_total: int
    points_written: int
    sampled: bool


def export_voxel_centers_csv(
    occupied: np.ndarray,
    *,
    origin_xyz_mm: np.ndarray,
    voxel_size_mm: float,
    out_csv: str | Path,
    max_points: int = 250_000,
    seed: int = 0,
    include_header: bool = False,
) -> PointCloudExportResult:
    """
    Exports occupied voxel centers as a point cloud CSV.

    This is intended as a generic interchange format for external CAD/mesh tools that
    can consume a point list / point cloud.
    For large models, the output is randomly sampled to `max_points`.
    """

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if occupied.dtype != bool:
        occupied = occupied.astype(bool, copy=False)

    points_total = int(occupied.sum())
    if points_total == 0:
        out_csv.write_text("x,y,z\n" if include_header else "", encoding="utf-8")
        return PointCloudExportResult(points_total=0, points_written=0, sampled=False)

    flat = occupied.ravel()
    flat_idx = np.flatnonzero(flat)
    sampled = points_total > int(max_points)
    if sampled:
        rng = np.random.default_rng(int(seed))
        take = rng.choice(flat_idx, size=int(max_points), replace=False)
    else:
        take = flat_idx

    ix, iy, iz = np.unravel_index(take, occupied.shape)
    pts = np.column_stack([ix, iy, iz]).astype(np.float64, copy=False)
    pts = origin_xyz_mm.reshape(1, 3) + (pts + 0.5) * float(voxel_size_mm)

    with out_csv.open("w", encoding="utf-8", newline="\n") as f:
        if include_header:
            f.write("x,y,z\n")
        chunk = 50_000
        for i in range(0, pts.shape[0], chunk):
            sub = pts[i : i + chunk]
            for x, y, z in sub:
                f.write(f"{x:.6f}, {y:.6f}, {z:.6f}\n")

    return PointCloudExportResult(points_total=points_total, points_written=int(pts.shape[0]), sampled=sampled)
