from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclasses.dataclass(frozen=True)
class BoundaryLoop:
    vertex_count: int
    edge_count: int
    perimeter_mm: float
    approx_diameter_mm: float


@dataclasses.dataclass(frozen=True)
class MeshStats:
    path: str | None
    vertices: int
    faces: int
    bbox_min: tuple[float, float, float] | None
    bbox_max: tuple[float, float, float] | None
    bbox_diag_mm: float | None
    watertight: bool | None
    winding_consistent: bool | None
    volume_mm3: float | None
    boundary_edges: int | None
    boundary_loops: list[BoundaryLoop] | None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "vertices": int(self.vertices),
            "faces": int(self.faces),
            "bbox_min": list(self.bbox_min) if self.bbox_min is not None else None,
            "bbox_max": list(self.bbox_max) if self.bbox_max is not None else None,
            "bbox_diag_mm": float(self.bbox_diag_mm) if self.bbox_diag_mm is not None else None,
            "watertight": bool(self.watertight) if self.watertight is not None else None,
            "winding_consistent": bool(self.winding_consistent) if self.winding_consistent is not None else None,
            "volume_mm3": float(self.volume_mm3) if self.volume_mm3 is not None else None,
            "boundary_edges": int(self.boundary_edges) if self.boundary_edges is not None else None,
            "boundary_loops": [dataclasses.asdict(x) for x in (self.boundary_loops or [])]
            if self.boundary_loops is not None
            else None,
        }


def _boundary_edges_and_loops(mesh: trimesh.Trimesh) -> tuple[int, list[BoundaryLoop]]:
    if mesh.faces.size == 0 or mesh.vertices.size == 0:
        return 0, []

    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)

    if edges.size == 0:
        return 0, []

    edges_view = edges.view([("a", edges.dtype), ("b", edges.dtype)]).reshape(-1)
    uniq, counts = np.unique(edges_view, return_counts=True)
    boundary = uniq[counts == 1]
    boundary_edges = int(boundary.shape[0])
    if boundary_edges == 0:
        return 0, []

    bedges = np.column_stack([boundary["a"], boundary["b"]]).astype(np.int64, copy=False)

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return parent.get(x, x)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        parent[rb] = ra

    for a, b in bedges:
        parent.setdefault(int(a), int(a))
        parent.setdefault(int(b), int(b))
        union(int(a), int(b))

    comp_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, b in bedges:
        r = find(int(a))
        comp_edges[r].append((int(a), int(b)))

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    loops: list[BoundaryLoop] = []
    for _root, elist in comp_edges.items():
        vset: set[int] = set()
        perim = 0.0
        for a, b in elist:
            vset.add(a)
            vset.add(b)
            pa = verts[a]
            pb = verts[b]
            perim += float(np.linalg.norm(pb - pa))
        approx_d = float(perim / math.pi) if perim > 0 else 0.0
        loops.append(
            BoundaryLoop(
                vertex_count=int(len(vset)),
                edge_count=int(len(elist)),
                perimeter_mm=float(perim),
                approx_diameter_mm=float(approx_d),
            )
        )

    loops.sort(key=lambda x: x.perimeter_mm, reverse=True)
    return boundary_edges, loops


def compute_mesh_stats(path: str | Path, *, include_boundary: bool = True) -> MeshStats:
    p = Path(path)
    mesh = trimesh.load_mesh(str(p), force="mesh", process=False)
    vertices = int(mesh.vertices.shape[0])
    faces = int(mesh.faces.shape[0])

    bbox_min = bbox_max = None
    bbox_diag = None
    try:
        if vertices > 0:
            b = np.asarray(mesh.bounds, dtype=float)
            bbox_min = (float(b[0, 0]), float(b[0, 1]), float(b[0, 2]))
            bbox_max = (float(b[1, 0]), float(b[1, 1]), float(b[1, 2]))
            bbox_diag = float(np.linalg.norm(b[1] - b[0]))
    except Exception:
        bbox_min = bbox_max = None
        bbox_diag = None

    watertight = winding = None
    try:
        watertight = bool(mesh.is_watertight)
        winding = bool(mesh.is_winding_consistent)
    except Exception:
        watertight = winding = None

    volume = None
    try:
        if watertight:
            volume = float(mesh.volume)
    except Exception:
        volume = None

    boundary_edges = None
    boundary_loops = None
    if include_boundary:
        try:
            boundary_edges, boundary_loops = _boundary_edges_and_loops(mesh)
        except Exception:
            boundary_edges, boundary_loops = None, None

    return MeshStats(
        path=str(p),
        vertices=vertices,
        faces=faces,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_diag_mm=bbox_diag,
        watertight=watertight,
        winding_consistent=winding,
        volume_mm3=volume,
        boundary_edges=boundary_edges,
        boundary_loops=boundary_loops,
    )


def estimate_close_holes_max_edges(
    stats: MeshStats, *, close_holes_max_mm: float, fallback_edges: int = 30
) -> tuple[int, dict[str, Any]]:
    """
    MeshLab/pymeshlab 'Close Holes' uses `maxholesize` as an edge-count limit.

    The project-level API accepts `close_holes_max_mm` in millimeters; we convert it to an
    approximate edge-count limit using mean boundary edge length + perimeter approximation.
    """
    close_mm = float(close_holes_max_mm)
    if close_mm <= 0:
        return int(fallback_edges), {"reason": "close_holes_max_mm<=0", "fallback_edges": int(fallback_edges)}

    loops = stats.boundary_loops or []
    if not loops or stats.boundary_edges in (None, 0):
        return int(fallback_edges), {"reason": "no_boundary_detected", "fallback_edges": int(fallback_edges)}

    # Estimate mean boundary edge length from loop perimeters.
    total_edges = sum(int(l.edge_count) for l in loops)
    total_perim = sum(float(l.perimeter_mm) for l in loops)
    mean_edge = (total_perim / float(total_edges)) if total_edges > 0 and total_perim > 0 else 0.0
    if mean_edge <= 1e-9:
        return int(fallback_edges), {"reason": "mean_boundary_edge_too_small", "fallback_edges": int(fallback_edges)}

    # Interpret close_holes_max_mm as approximate hole diameter.
    max_perim = math.pi * close_mm
    max_edges = int(max(3, math.ceil(max_perim / float(mean_edge))))
    return max_edges, {
        "close_holes_max_mm": float(close_mm),
        "interpreted_as": "approx_diameter_mm",
        "mean_boundary_edge_len_mm": float(mean_edge),
        "maxholesize_edges": int(max_edges),
    }

