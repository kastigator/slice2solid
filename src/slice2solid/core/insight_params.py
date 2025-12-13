from __future__ import annotations

import dataclasses
import re
from pathlib import Path


_SETVAR_PAT = re.compile(r"^\s*setVersionVar\s+([^\s]+)\s+\{(.*?)\}\s+[0-9.]+;\s*$")


@dataclasses.dataclass(frozen=True)
class InsightParams:
    slice_params: dict[str, str]
    toolpath_params: dict[str, str]
    support_params: dict[str, str]


def parse_setversionvar_file(path: str | Path) -> dict[str, str]:
    p = Path(path)
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _SETVAR_PAT.match(line)
        if not m:
            continue
        out[m.group(1)] = m.group(2).strip()
    return out


def load_job_params(job_dir: str | Path) -> InsightParams:
    """
    Loads Insight parameter snapshots from a ssys_* folder.
    Prefers *.new over *.cur when present.
    """
    root = Path(job_dir)
    if not root.exists():
        raise FileNotFoundError(str(root))

    def pick(name: str) -> Path | None:
        p_new = root / f"{name}.new"
        if p_new.exists():
            return p_new
        p_cur = root / f"{name}.cur"
        if p_cur.exists():
            return p_cur
        return None

    slice_path = pick("sliceParams")
    toolpath_path = pick("toolpathParams")
    support_path = pick("supportParams")

    return InsightParams(
        slice_params=parse_setversionvar_file(slice_path) if slice_path else {},
        toolpath_params=parse_setversionvar_file(toolpath_path) if toolpath_path else {},
        support_params=parse_setversionvar_file(support_path) if support_path else {},
    )


def infer_stl_path_from_job(job_dir: str | Path) -> Path | None:
    """
    Tries to infer the STL path from a job folder by reading *.sjb.
    """
    root = Path(job_dir)
    sjb = next(root.glob("*.sjb"), None)
    if sjb is None:
        return None

    stl_directory: str | None = None
    stl_path: str | None = None

    pat = re.compile(r'^\s*setVersionVar\s+([^\s]+)\s+"(.*)";\s*$')
    for line in sjb.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if key == "stlDirectory":
            stl_directory = value
        elif key == "stlPath":
            stl_path = value

    if stl_path:
        p = Path(stl_path)
        if p.exists():
            return p
        # try stlDirectory + basename
        if stl_directory:
            candidate = Path(stl_directory) / Path(stl_path).name
            if candidate.exists():
                return candidate
    return None


def _parse_float(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        return float(v.strip())
    except ValueError:
        return None


def infer_params_length_scale(
    *,
    params_slice_height: float | None,
    sim_slice_height_mm: float | None,
) -> float | None:
    """
    Infers length unit conversion for Insight params files based on slice height.

    - Many Insight params files use inches (e.g., sliceHeight 0.0100 = 0.254 mm).
    - Simulation export uses mm for Slice height.

    Returns multiplier 'k' such that: value_mm = value_in_params * k.
    """
    if params_slice_height is None or sim_slice_height_mm is None:
        return None
    if params_slice_height <= 0 or sim_slice_height_mm <= 0:
        return None

    ratio = sim_slice_height_mm / params_slice_height
    if abs(ratio - 25.4) < 0.5:
        return 25.4
    if abs(ratio - 1.0) < 0.05:
        return 1.0
    return ratio


def estimate_bead_width_mm(
    params: InsightParams,
    *,
    sim_slice_height_mm: float | None,
) -> float | None:
    """
    Estimates bead width (mm) from Insight params.
    Uses the largest plausible bead width among common keys (including custom infill groups),
    because under-estimating radius can create gaps in voxelization.
    """
    slice_h_params = _parse_float(params.slice_params.get("sliceHeight"))
    k = infer_params_length_scale(params_slice_height=slice_h_params, sim_slice_height_mm=sim_slice_height_mm) or 25.4

    tp = params.toolpath_params
    widths_in_params: list[float] = []

    # Common/global widths (not all will be present for all printers/modes).
    common_keys = (
        "contourWidth",
        "openCurveWidth",
        "perimeterWidth",
        "rasterWidth",
        "sparseRasterWidth",
        "main:base:width",
        "alt:base:width",
        "main:base:topWidth",
        "alt:base:topWidth",
        "main:part:prefContourWidth",
        "alt:part:prefContourWidth",
    )
    for key in common_keys:
        w = _parse_float(tp.get(key))
        if w is not None and w > 0:
            widths_in_params.append(w)

    # Custom group widths (e.g. when using custom infill groups in Insight).
    # Keys typically look like: custom:1:rasterWidth, custom:1:sparseRasterWidth, custom:1:prefContourWidth, etc.
    custom_width_re = re.compile(r"^custom:\d+:(?:prefContourWidth|openCurveWidth|rasterWidth|sparseRasterWidth)$")
    for key, value in tp.items():
        if not custom_width_re.match(key):
            continue
        w = _parse_float(value)
        if w is not None and w > 0:
            widths_in_params.append(w)

    if not widths_in_params:
        return None

    return max(widths_in_params) * k
