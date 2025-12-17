from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from .stats import MeshStats, compute_mesh_stats, estimate_close_holes_max_edges


HealPreset = Literal["safe", "aggressive"]
HealBackend = Literal["auto", "pymeshlab", "meshlabserver"]


@dataclasses.dataclass(frozen=True)
class HealResult:
    input_path: str
    output_path: str
    preset: HealPreset
    backend: str
    close_holes_max_mm: float
    close_holes_max_edges: int
    conversion: dict[str, Any]
    before: MeshStats | None
    after: MeshStats | None
    warnings: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "preset": self.preset,
            "backend": self.backend,
            "close_holes_max_mm": float(self.close_holes_max_mm),
            "close_holes_max_edges": int(self.close_holes_max_edges),
            "conversion": self.conversion,
            "before": self.before.to_json() if self.before is not None else None,
            "after": self.after.to_json() if self.after is not None else None,
            "warnings": list(self.warnings),
        }


def _default_out_path(inp: Path) -> Path:
    return inp.with_name(f"{inp.stem}_healed{inp.suffix}")


def _try_import_pymeshlab() -> Any | None:
    try:
        import pymeshlab  # type: ignore

        return pymeshlab
    except Exception:
        return None


def _heal_with_pymeshlab(
    *,
    inp: Path,
    out: Path,
    preset: HealPreset,
    close_holes_max_edges: int,
    logger: logging.Logger,
) -> None:
    pymeshlab = _try_import_pymeshlab()
    if pymeshlab is None:
        raise RuntimeError("pymeshlab is not available")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(inp))

    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_re_orient_faces_coherently")

    if preset == "aggressive":
        ms.apply_filter("compute_selection_by_self_intersections_per_face")
        ms.apply_filter("meshing_remove_selected_faces")
        ms.apply_filter("meshing_remove_unreferenced_vertices")

    ms.apply_filter(
        "meshing_close_holes",
        maxholesize=int(close_holes_max_edges),
        selected=False,
        newfaceselected=True,
        selfintersection=True,
        refinehole=False,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(out))
    logger.info("Healed mesh saved: %s", out)


def _meshlabserver_path() -> str | None:
    return shutil.which("meshlabserver") or shutil.which("MeshLabServer")


def _write_mlx(
    *,
    out_mlx: Path,
    preset: HealPreset,
    close_holes_max_edges: int,
) -> None:
    # NOTE: MeshLab 'Close Holes' max size is typically edge-count, not mm.
    # This script is generated to match the pymeshlab pipeline as close as possible.
    # Filter and parameter names can vary across MeshLab versions.
    out_mlx.parent.mkdir(parents=True, exist_ok=True)
    aggressive_block = ""
    if preset == "aggressive":
        aggressive_block = """
 <filter name="Remove Self Intersections"/>
"""

    xml = f"""<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Remove Duplicate Vertices"/>
 <filter name="Remove Duplicate Faces"/>
 <filter name="Remove Unreferenced Vertices"/>
 <filter name="Remove Zero Area Faces"/>
 <filter name="Re-Orient all faces coherently"/>
{aggressive_block.strip()}
 <filter name="Close Holes">
  <Param type="RichInt" name="maxholesize" value="{int(close_holes_max_edges)}"/>
  <Param type="RichBool" name="selected" value="false"/>
  <Param type="RichBool" name="newfaceselected" value="true"/>
  <Param type="RichBool" name="selfintersection" value="true"/>
  <Param type="RichBool" name="refinehole" value="false"/>
 </filter>
</FilterScript>
"""
    out_mlx.write_text(xml, encoding="utf-8", newline="\n")


def _heal_with_meshlabserver(
    *,
    inp: Path,
    out: Path,
    preset: HealPreset,
    close_holes_max_edges: int,
    logger: logging.Logger,
) -> None:
    exe = _meshlabserver_path()
    if not exe:
        raise RuntimeError("meshlabserver not found in PATH")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s2s_meshlab_") as td:
        mlx = Path(td) / "heal.mlx"
        _write_mlx(out_mlx=mlx, preset=preset, close_holes_max_edges=close_holes_max_edges)
        cmd = [exe, "-i", str(inp), "-o", str(out), "-s", str(mlx)]
        logger.info("Running meshlabserver: %s", " ".join(cmd))
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"meshlabserver failed (code {p.returncode}):\n{p.stdout}")


def heal_mesh_file(
    input_path: str | Path,
    *,
    out_path: str | Path | None = None,
    preset: HealPreset = "safe",
    close_holes_max_mm: float = 2.0,
    report_path: str | Path | None = None,
    backend: HealBackend = "auto",
    logger: logging.Logger | None = None,
) -> HealResult:
    """
    Heal an STL/OBJ/PLY mesh file and write a repaired mesh suitable for CAD mesh import.

    The implementation aims to be "safe": no remeshing/simplification, and only small-hole closing.
    """
    log = logger or logging.getLogger("slice2solid.mesh_heal")

    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    out = Path(out_path) if out_path is not None else _default_out_path(inp)
    preset_v: HealPreset = "aggressive" if str(preset).lower().strip() == "aggressive" else "safe"

    warnings: list[str] = []
    before: MeshStats | None = None
    after: MeshStats | None = None

    try:
        before = compute_mesh_stats(inp)
    except Exception as e:
        warnings.append(f"Failed to compute pre-stats: {e}")

    if before is not None:
        max_edges, conversion = estimate_close_holes_max_edges(before, close_holes_max_mm=float(close_holes_max_mm))
    else:
        max_edges, conversion = int(30), {"reason": "no_pre_stats", "fallback_edges": 30}

    used_backend = "auto"
    last_err: Exception | None = None

    def _try(name: str, fn) -> bool:
        nonlocal used_backend, last_err
        try:
            fn()
            used_backend = name
            return True
        except Exception as e:
            last_err = e
            return False

    if backend == "pymeshlab":
        ok = _try(
            "pymeshlab",
            lambda: _heal_with_pymeshlab(
                inp=inp, out=out, preset=preset_v, close_holes_max_edges=max_edges, logger=log
            ),
        )
        if not ok:
            raise RuntimeError(f"pymeshlab healing failed: {last_err}")
    elif backend == "meshlabserver":
        ok = _try(
            "meshlabserver",
            lambda: _heal_with_meshlabserver(
                inp=inp, out=out, preset=preset_v, close_holes_max_edges=max_edges, logger=log
            ),
        )
        if not ok:
            raise RuntimeError(f"meshlabserver healing failed: {last_err}")
    else:
        # auto: prefer pymeshlab, fallback to meshlabserver
        if _try_import_pymeshlab() is not None:
            _try(
                "pymeshlab",
                lambda: _heal_with_pymeshlab(
                    inp=inp, out=out, preset=preset_v, close_holes_max_edges=max_edges, logger=log
                ),
            )
        if used_backend == "auto":
            _try(
                "meshlabserver",
                lambda: _heal_with_meshlabserver(
                    inp=inp, out=out, preset=preset_v, close_holes_max_edges=max_edges, logger=log
                ),
            )
        if used_backend == "auto":
            hint = (
                "Mesh healing backend not available. Install pymeshlab (`pip install pymeshlab`) "
                "or install MeshLab and add `meshlabserver` to PATH."
            )
            raise RuntimeError(f"{hint}\nLast error: {last_err}")

    try:
        after = compute_mesh_stats(out)
    except Exception as e:
        warnings.append(f"Failed to compute post-stats: {e}")

    result = HealResult(
        input_path=str(inp),
        output_path=str(out),
        preset=preset_v,
        backend=str(used_backend),
        close_holes_max_mm=float(close_holes_max_mm),
        close_holes_max_edges=int(max_edges),
        conversion=conversion,
        before=before,
        after=after,
        warnings=warnings,
    )

    if report_path is not None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(result.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    return result
