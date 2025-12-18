from __future__ import annotations

import argparse
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slice2solid.core.insight_simulation import read_simulation_export


@dataclass(frozen=True)
class Match:
    dtype: str
    unit: str
    float_index: int
    byte_offset: int


def _find_desktop_file(name: str) -> Path:
    base = Path.home() / "Desktop"
    for p in base.rglob(name):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Not found under Desktop: {name}")


def _extract_ascii_strings(data: bytes, *, min_len: int = 6) -> list[str]:
    pat = re.compile(rb"[\x20-\x7E]{%d,}" % int(min_len))
    out: list[str] = []
    for m in pat.finditer(data):
        try:
            out.append(m.group(0).decode("ascii", errors="replace"))
        except Exception:
            continue
    return out


def _collect_points(sim_path: Path, *, n: int) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    header, it = read_simulation_export(sim_path)
    pts: list[tuple[float, float, float]] = []
    for p in it:
        # Use only model points (Type=1) with deposition (Factor>0, BeadArea>0)
        if int(p.type) != 1:
            continue
        if float(p.factor) <= 0 or float(p.bead_area) <= 0:
            continue
        pts.append((float(p.x), float(p.y), float(p.z)))
        if len(pts) >= int(n):
            break
    if not pts:
        raise RuntimeError("No Type=1 deposition points found in simulation export.")
    return header.stl_to_cmb, pts


def _rowvec_transform(points_xyz: list[tuple[float, float, float]], m: np.ndarray) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for x, y, z in points_xyz:
        v = np.array([float(x), float(y), float(z), 1.0], dtype=np.float64)
        r = v @ m
        out.append((float(r[0]), float(r[1]), float(r[2])))
    return out


def _search_triplet_f32(arr: np.ndarray, x: float, y: float, z: float, tol: float) -> list[int]:
    # Find indices i such that arr[i:i+3] ~= (x,y,z)
    if arr.size < 3:
        return []
    mask = np.isfinite(arr)
    arr = np.where(mask, arr, np.nan)
    x_hits = np.where(np.abs(arr[:-2] - x) <= tol)[0]
    out: list[int] = []
    for i in x_hits[:20000]:  # cap for speed
        if abs(float(arr[i + 1]) - y) <= tol and abs(float(arr[i + 2]) - z) <= tol:
            out.append(int(i))
            if len(out) >= 50:
                break
    return out


def _scan_matches(cmb: bytes, pts_mm: list[tuple[float, float, float]]) -> list[Match]:
    matches: list[Match] = []

    # Try interpreting the entire buffer as float32/float64 streams (little-endian).
    f32 = np.frombuffer(cmb, dtype="<f4", count=(len(cmb) // 4))
    f64 = np.frombuffer(cmb, dtype="<f8", count=(len(cmb) // 8))

    # Two unit hypotheses: mm or inches (Insight params often use inches internally).
    unit_sets: list[tuple[str, float]] = [("mm", 1.0), ("inch", 1.0 / 25.4)]
    tol_sets = {
        ("<f4", "mm"): 1e-3,
        ("<f4", "inch"): 5e-5,
        ("<f8", "mm"): 1e-6,
        ("<f8", "inch"): 1e-7,
    }

    sample_pts = pts_mm[: min(20, len(pts_mm))]
    for unit_name, k in unit_sets:
        for (dtype_name, arr, stride_bytes) in (("<f4", f32, 4), ("<f8", f64, 8)):
            tol = float(tol_sets[(dtype_name, unit_name)])
            for (x_mm, y_mm, z_mm) in sample_pts:
                x = x_mm * k
                y = y_mm * k
                z = z_mm * k
                hits = _search_triplet_f32(arr, x, y, z, tol) if dtype_name == "<f4" else _search_triplet_f32(arr.astype(np.float32), x, y, z, tol)  # noqa: PLR2004
                for idx in hits[:3]:
                    matches.append(
                        Match(
                            dtype=dtype_name,
                            unit=unit_name,
                            float_index=idx,
                            byte_offset=int(idx) * stride_bytes,
                        )
                    )
                if len(matches) >= 30:
                    return matches
    return matches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmb", type=str, default=r"C:\Users\alexa\Desktop\тест проект\ssys_part\part.cmb")
    ap.add_argument("--sim", type=str, default=r"C:\Users\alexa\Desktop\тест проект\part-simulation-data.txt")
    ap.add_argument("--n", type=int, default=200, help="How many Type=1 deposition points to sample from sim export.")
    ns = ap.parse_args()

    cmb_path = Path(ns.cmb)
    sim_path = Path(ns.sim)
    if not cmb_path.exists():
        cmb_path = _find_desktop_file("part.cmb")
    if not sim_path.exists():
        sim_path = _find_desktop_file("part-simulation-data.txt")

    cmb = cmb_path.read_bytes()
    print("cmb:", cmb_path, "bytes:", len(cmb))
    print("sim:", sim_path)

    # Quick strings to confirm we're looking at the right file.
    strings = _extract_ascii_strings(cmb, min_len=6)
    print("\n-- printable strings (sample) --")
    for s in strings[:40]:
        print(" ", s)

    stl_to_cmb, pts_cmb = _collect_points(sim_path, n=int(ns.n))
    print("\n-- sim points sample (CMB coords, from export) --")
    for p in pts_cmb[:5]:
        print(" ", p)

    # Many Insight internals store toolpaths closer to STL/local coordinates.
    # Convert CMB->STL by inverting STL->CMB (row-vector convention still holds: v @ inv(M)).
    try:
        inv = np.linalg.inv(np.asarray(stl_to_cmb, dtype=np.float64))
        pts_stl = _rowvec_transform(pts_cmb[: min(200, len(pts_cmb))], inv)
    except Exception as e:
        inv = None
        pts_stl = []
        print("WARN: failed to invert STL->CMB matrix:", e)

    def _run(label: str, pts: list[tuple[float, float, float]]) -> list[Match]:
        m = _scan_matches(cmb, pts)
        print(f"\n-- candidate coordinate matches: {label} --")
        if not m:
            print("No obvious float32/float64 triplet matches found (mm/inch).")
            return []
        for mm in m[:20]:
            print(f"dtype={mm.dtype} unit={mm.unit} float_index={mm.float_index} byte_offset=0x{mm.byte_offset:08X}")
        return m

    m1 = _run("CMB coords (export)", pts_cmb)
    m2 = _run("STL coords (from inverse matrix)", pts_stl) if pts_stl else []

    if not m1 and not m2:
        print(
            "\nLikely causes: fixed-point encoding, record packing (not pure XYZ float stream), "
            "or additional transforms (e.g., envelope-center offsets) before storage."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
