from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np


_EXPORT_VERSION_RE = re.compile(r"^\s*Version\s+(.+?)\s*$", re.IGNORECASE)
_INSIGHT_VERSION_RE = re.compile(r"^\s*Insight\s+version\s+(.+?)\s*$", re.IGNORECASE)


def _parse_float_relaxed(token: str) -> float:
    """
    Parses a float token robustly across locales.

    Insight may export decimals with a comma in some locales, while other parts of the toolchain use a dot.
    """
    s = token.strip()
    if not s:
        raise ValueError("Empty float token.")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    return float(s)


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
            return _parse_float_relaxed(v)
        except ValueError:
            return None

    @property
    def segment_filter_length_mm(self) -> float | None:
        v = self.raw.get("Segment filter length")
        if v is None:
            return None
        try:
            return _parse_float_relaxed(v)
        except ValueError:
            return None

    @property
    def export_version(self) -> str | None:
        return self.raw.get("Export version")

    @property
    def insight_version(self) -> str | None:
        return self.raw.get("Insight version")

    @property
    def stl_units(self) -> str | None:
        v = self.raw.get("STL units")
        if v is None:
            return None
        s = v.strip().lower()
        return s or None

    @property
    def build_mode(self) -> str | None:
        v = self.raw.get("Build mode")
        if v is None:
            return None
        s = v.strip().lower()
        return s or None

    @property
    def modeler_type(self) -> str | None:
        v = self.raw.get("Modeler type")
        if v is None:
            return None
        s = v.strip()
        return s or None

    @property
    def model_shrink_xyz(self) -> tuple[float, float, float] | None:
        try:
            sx = _parse_float_relaxed(self.raw.get("Model X-shrink", ""))
            sy = _parse_float_relaxed(self.raw.get("Model Y-shrink", ""))
            sz = _parse_float_relaxed(self.raw.get("Model Z-shrink", ""))
        except ValueError:
            return None
        return (float(sx), float(sy), float(sz))

    @property
    def support_shrink_xyz(self) -> tuple[float, float, float] | None:
        try:
            sx = _parse_float_relaxed(self.raw.get("Support X-shrink", ""))
            sy = _parse_float_relaxed(self.raw.get("Support Y-shrink", ""))
            sz = _parse_float_relaxed(self.raw.get("Support Z-shrink", ""))
        except ValueError:
            return None
        return (float(sx), float(sy), float(sz))

    def validation_warnings(self) -> list[str]:
        return validate_stl_to_cmb_matrix(self.stl_to_cmb, expected_model_shrink=self.model_shrink_xyz)


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


def validate_stl_to_cmb_matrix(
    m: np.ndarray,
    *,
    expected_model_shrink: tuple[float, float, float] | None = None,
    atol: float = 1e-6,
) -> list[str]:
    """
    Performs basic sanity checks for the STL→CMB transform matrix found in Simulation Data exports.

    The project assumes Insight's "row-vector" convention:
        [x y z 1] @ M
    with translation stored in the last row (M[3,0:3]).
    """
    warnings: list[str] = []

    if not isinstance(m, np.ndarray):
        warnings.append("STL→CMB matrix is not a numpy array.")
        return warnings
    if m.shape != (4, 4):
        warnings.append(f"STL→CMB matrix has shape {m.shape}, expected (4,4).")
        return warnings
    if not np.isfinite(m).all():
        warnings.append("STL→CMB matrix contains NaN/Inf values.")
        return warnings

    # Affine bottom-right element sanity.
    if abs(float(m[3, 3]) - 1.0) > float(atol):
        warnings.append(f"STL→CMB matrix M[3,3] is {float(m[3,3]):.6g}, expected ~1.0.")

    # Heuristic: detect column-vector convention (translation in last column).
    t_row = float(np.linalg.norm(m[3, 0:3]))
    t_col = float(np.linalg.norm(m[0:3, 3]))
    if t_col > 10.0 * float(atol) and t_row < 10.0 * float(atol):
        warnings.append(
            "STL→CMB matrix looks like a column-vector transform (translation in last column). "
            "Expected Insight row-vector convention (translation in last row)."
        )

    # If the matrix is close to diagonal scaling, compare to shrink factors.
    if expected_model_shrink is not None:
        lin = m[0:3, 0:3]
        off_diag = lin - np.diag(np.diag(lin))
        diag = np.diag(lin)
        if float(np.linalg.norm(off_diag, ord=np.inf)) <= 1e-3:
            sx, sy, sz = (float(diag[0]), float(diag[1]), float(diag[2]))
            ex, ey, ez = (float(expected_model_shrink[0]), float(expected_model_shrink[1]), float(expected_model_shrink[2]))
            if max(abs(sx - ex), abs(sy - ey), abs(sz - ez)) > 0.02:
                warnings.append(
                    "STL→CMB matrix scale does not match header shrink factors "
                    f"(matrix≈[{sx:.4f},{sy:.4f},{sz:.4f}] vs header≈[{ex:.4f},{ey:.4f},{ez:.4f}])."
                )

    # Numerical stability.
    try:
        det = float(np.linalg.det(m[0:3, 0:3]))
        if abs(det) < 1e-9:
            warnings.append("STL→CMB linear part is near-singular (det≈0).")
    except Exception:
        pass

    return warnings


def read_simulation_export(path: str | Path) -> tuple[SimulationHeader, Iterator[ToolpathPoint]]:
    p = Path(path)
    header: dict[str, str] = {}
    stl_to_cmb: np.ndarray | None = None

    table_start_line: int | None = None
    matrix_start_line: int | None = None

    # Stream header scanning to avoid loading huge exports into memory.
    # We only need the header key/value pairs, the STL->CMB matrix, and the table delimiter line number.
    matrix_rows: list[list[float]] = []
    need_matrix_rows = 0

    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f):
            line = raw.rstrip("\n")

            if need_matrix_rows > 0:
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(
                        f"Could not parse STL→CMB matrix row at line {i+1}: expected 4 numbers, got {len(parts)}."
                    )
                nums = [_parse_float_relaxed(x) for x in parts]
                matrix_rows.append([float(x) for x in nums])
                need_matrix_rows -= 1
                if need_matrix_rows == 0:
                    stl_to_cmb = np.array(matrix_rows, dtype=float)
                continue

            s = line.strip()
            m_ver = _EXPORT_VERSION_RE.match(s)
            if m_ver:
                header.setdefault("Export version", m_ver.group(1).strip())
                continue
            m_ins = _INSIGHT_VERSION_RE.match(s)
            if m_ins:
                header.setdefault("Insight version", m_ins.group(1).strip())
                continue

            if s.lower().startswith("stl to cmb transformation matrix"):
                matrix_start_line = i
                matrix_rows = []
                need_matrix_rows = 4
                continue

            if ":" in line and not s.startswith("---"):
                k, v = line.split(":", 1)
                header[k.strip()] = v.strip()

            if s.startswith("-------------"):
                table_start_line = i
                break

    if stl_to_cmb is None:
        raise ValueError("Could not parse STL to CMB transformation matrix from simulation export.")
    if table_start_line is None:
        raise ValueError("Could not locate toolpath table start in simulation export.")

    header_obj = SimulationHeader(raw=header, stl_to_cmb=stl_to_cmb)

    def iter_rows() -> Iterator[ToolpathPoint]:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(int(table_start_line) + 1):
                next(f, None)
            for line in f:
                parts = line.split()
                if len(parts) != 8:
                    continue
                try:
                    x, y, z, t, area, factor = (_parse_float_relaxed(x) for x in parts[0:6])
                    typ = int(parts[6])
                    mode = int(parts[7])
                except ValueError:
                    continue
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


def transform_toolpath_points_rowvec(
    points: Iterable[ToolpathPoint],
    m: np.ndarray,
    *,
    chunk_size: int = 200_000,
) -> Iterator[ToolpathPoint]:
    """
    Streams ToolpathPoint items through a 4x4 row-vector transform.

    This keeps memory usage low for large Insight exports by transforming points in chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    buf_xyz: list[list[float]] = []
    buf_rest: list[tuple[float, float, float, int, int]] = []

    def flush() -> Iterator[ToolpathPoint]:
        nonlocal buf_xyz, buf_rest
        if not buf_xyz:
            return
        arr = np.array(buf_xyz, dtype=float)
        out = transform_points_rowvec(arr, m)
        for i in range(out.shape[0]):
            time_s, bead_area, factor, typ, mode = buf_rest[i]
            yield ToolpathPoint(
                x=float(out[i, 0]),
                y=float(out[i, 1]),
                z=float(out[i, 2]),
                time_s=float(time_s),
                bead_area=float(bead_area),
                factor=float(factor),
                type=int(typ),
                bead_mode=int(mode),
            )
        buf_xyz = []
        buf_rest = []
        return

    for pt in points:
        buf_xyz.append([pt.x, pt.y, pt.z])
        buf_rest.append((pt.time_s, pt.bead_area, pt.factor, pt.type, pt.bead_mode))
        if len(buf_xyz) >= int(chunk_size):
            yield from flush()

    yield from flush()


def chunked(iterable: Iterable[ToolpathPoint], n: int) -> Iterator[list[ToolpathPoint]]:
    buf: list[ToolpathPoint] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
