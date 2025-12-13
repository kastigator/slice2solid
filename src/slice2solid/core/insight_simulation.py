from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class SimulationHeader:
    raw: dict[str, str]
    stl_to_cmb: np.ndarray  # 4x4, row-vector convention (translation in last row)

    @property
    def slice_height_mm(self) -> float | None:
        v = self.raw.get("Slice height")
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    @property
    def segment_filter_length_mm(self) -> float | None:
        v = self.raw.get("Segment filter length")
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None


@dataclasses.dataclass(frozen=True)
class ToolpathPoint:
    x: float
    y: float
    z: float
    time_s: float
    bead_area: float
    factor: float
    type: int
    bead_mode: int


_ROW_RE = re.compile(
    r"^\s*([+-]?[0-9]*\.?[0-9]+)\s+"
    r"([+-]?[0-9]*\.?[0-9]+)\s+"
    r"([+-]?[0-9]*\.?[0-9]+)\s+"
    r"([+-]?[0-9]*\.?[0-9]+)\s+"
    r"([+-]?[0-9]*\.?[0-9]+)\s+"
    r"([+-]?[0-9]*\.?[0-9]+)\s+"
    r"([0-9]+)\s+"
    r"([0-9]+)\s*$"
)


def read_simulation_export(path: str | Path) -> tuple[SimulationHeader, Iterator[ToolpathPoint]]:
    p = Path(path)
    header: dict[str, str] = {}
    stl_to_cmb: np.ndarray | None = None

    table_start_line: int | None = None
    matrix_start_line: int | None = None

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    for i, line in enumerate(lines[:500]):
        if line.strip().startswith("-------------"):
            table_start_line = i
            break
        if line.strip().lower().startswith("stl to cmb transformation matrix"):
            matrix_start_line = i
        if ":" in line and not line.strip().startswith("---"):
            k, v = line.split(":", 1)
            header[k.strip()] = v.strip()

    if matrix_start_line is not None:
        rows: list[list[float]] = []
        for j in range(matrix_start_line + 1, matrix_start_line + 5):
            nums = [float(x) for x in lines[j].split()]
            rows.append(nums)
        stl_to_cmb = np.array(rows, dtype=float)

    if stl_to_cmb is None:
        raise ValueError("Could not parse STL to CMB transformation matrix from simulation export.")
    if table_start_line is None:
        raise ValueError("Could not locate toolpath table start in simulation export.")

    header_obj = SimulationHeader(raw=header, stl_to_cmb=stl_to_cmb)

    def iter_rows() -> Iterator[ToolpathPoint]:
        for line in lines[table_start_line + 1 :]:
            m = _ROW_RE.match(line)
            if not m:
                continue
            x, y, z, t, area, factor, typ, mode = m.groups()
            yield ToolpathPoint(
                x=float(x),
                y=float(y),
                z=float(z),
                time_s=float(t),
                bead_area=float(area),
                factor=float(factor),
                type=int(typ),
                bead_mode=int(mode),
            )

    return header_obj, iter_rows()


def invert_rowvec_matrix(m: np.ndarray) -> np.ndarray:
    """
    Inverts a 4x4 homogeneous transform used in Insight exports where translation is stored in the last row
    and points are treated as row-vectors: p' = p @ M.
    """
    if m.shape != (4, 4):
        raise ValueError("Expected a 4x4 matrix.")
    return np.linalg.inv(m)


def transform_points_rowvec(points_xyz: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 row-vector transform: [x y z 1] @ M.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be (N,3).")
    ones = np.ones((points_xyz.shape[0], 1), dtype=float)
    homo = np.hstack([points_xyz.astype(float), ones])
    out = homo @ m
    return out[:, :3]


def chunked(iterable: Iterable[ToolpathPoint], n: int) -> Iterator[list[ToolpathPoint]]:
    buf: list[ToolpathPoint] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

