from __future__ import annotations

import html
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import trimesh
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

try:  # optional: 3D preview
    import pyqtgraph.opengl as gl
except Exception:  # pragma: no cover
    gl = None

pv = None
QtInteractor = None
_PYVISTA_IMPORT_ERROR: Exception | None = None


def _lazy_import_pyvista() -> bool:
    global pv, QtInteractor, _PYVISTA_IMPORT_ERROR
    if pv is not None and QtInteractor is not None:
        return True
    if _PYVISTA_IMPORT_ERROR is not None:
        return False
    try:  # optional: high-quality 3D preview (VTK)
        import pyvista as _pv
        from pyvistaqt import QtInteractor as _QtInteractor

        pv = _pv
        QtInteractor = _QtInteractor
        return True
    except Exception as e:  # pragma: no cover
        _PYVISTA_IMPORT_ERROR = e
        pv = None
        QtInteractor = None
        return False


def _preferred_preview_backend() -> str:
    """
    Returns one of: "auto", "vtk", "gl", "2d".

    Environment override:
      S2S_PREVIEW_BACKEND=auto|vtk|gl|2d
    """
    try:
        v = str(os.environ.get("S2S_PREVIEW_BACKEND", "")).strip().lower()
    except Exception:
        v = ""
    if v in ("auto", "vtk", "gl", "2d"):
        return v
    return "auto"


def _select_view_cls() -> type[QtWidgets.QWidget]:
    backend = _preferred_preview_backend()
    if backend == "vtk":
        if _lazy_import_pyvista():
            return _MeshVTKView
        return _Mesh3DView if gl is not None else _Mesh2DView
    if backend == "gl":
        return _Mesh3DView if gl is not None else (_MeshVTKView if _lazy_import_pyvista() else _Mesh2DView)
    if backend == "2d":
        return _Mesh2DView
    # auto: prefer VTK, but will fall back at runtime if init fails.
    if _lazy_import_pyvista():
        return _MeshVTKView
    if gl is not None:
        return _Mesh3DView
    return _Mesh2DView


from slice2solid.core.insight_simulation import (
    read_simulation_export,
    transform_points_rowvec,
)
from slice2solid.core.insight_params import (
    estimate_auto_max_jump_mm,
    estimate_bead_width_mm,
    estimate_toolpath_thresholds_mm,
    infer_stl_path_from_job,
    load_job_params,
)
from slice2solid.core.insight_sgm import extract_sgm_to_folder
from slice2solid.core.cae_orientation import compute_layer_orientations_toolpath
from slice2solid.core.cad_bundle import export_voxel_centers_csv
from slice2solid.core.voxelize import mesh_from_voxels_configured, voxelize_toolpath
from slice2solid.app_info import APP_DISPLAY_NAME, AUTHOR, CONTACT_EMAIL, DEPARTMENT, ORGANIZATION, VERSION
from slice2solid.mesh_heal import heal_mesh_file


def infer_simulation_txt_from_job(job_dir: str | Path) -> Path | None:
    root = Path(job_dir)
    if not root.exists():
        return None

    pats = (
        "*-simulation-data.txt",
        "*simulation-data*.txt",
        "*simulation*data*.txt",
        "*simulation*.txt",
    )
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pat in pats:
        for p in root.glob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                candidates.append(p)

    if not candidates:
        return None

    # Prefer the most explicit conventional name.
    candidates.sort(key=lambda p: (("simulation-data" not in p.name.lower()), len(p.name), p.name.lower()))
    return candidates[0]


def infer_stl_from_job_folder(job_dir: str | Path) -> Path | None:
    root = Path(job_dir)
    if not root.exists():
        return None
    stls = [p for p in root.glob("*.stl") if p.is_file()]
    stls += [p for p in root.glob("*.STL") if p.is_file()]
    stls = list({p.resolve() for p in stls})
    if len(stls) == 1:
        return stls[0]
    return infer_stl_path_from_job(root)


def has_toolpath_params(job_dir: str | Path) -> bool:
    root = Path(job_dir)
    try:
        return (root / "toolpathParams.new").exists() or (root / "toolpathParams.cur").exists()
    except Exception:
        return False


def extract_toolpath_segments_for_layer(
    simulation_txt: str | Path,
    *,
    z_center_mm: float,
    slice_height_mm: float,
    max_jump_mm: float | None,
    max_segments: int = 250_000,
) -> np.ndarray:
    """
    Builds line segments approximating the toolpath at a given Z layer.

    This is meant for a slicer-like preview (Cura-style) and is intentionally lightweight:
    - only Type=1
    - filters to a thin slab around z_center_mm
    - breaks segments on large jumps
    """
    tol = max(1e-6, 0.45 * float(slice_height_mm))
    max_jump = float(max_jump_mm) if (max_jump_mm is not None and float(max_jump_mm) > 0) else None

    _hdr, rows = read_simulation_export(simulation_txt)
    last: tuple[float, float, float] | None = None
    seg: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    for pt in rows:
        if pt.type != 1:
            last = None
            continue
        try:
            if float(pt.factor) <= 0 or float(pt.bead_area) <= 0:
                last = None
                continue
        except Exception:
            pass
        z = float(pt.z)
        if abs(z - float(z_center_mm)) > tol:
            last = None
            continue
        cur = (float(pt.x), float(pt.y), z)
        if last is not None:
            dx = cur[0] - last[0]
            dy = cur[1] - last[1]
            dz = cur[2] - last[2]
            d = float((dx * dx + dy * dy + dz * dz) ** 0.5)
            if max_jump is None or d <= max_jump:
                seg.append((last, cur))
                if len(seg) >= int(max_segments):
                    break
            else:
                last = cur
                continue
        last = cur

    if not seg:
        return np.zeros((0, 2, 3), dtype=np.float32)
    arr = np.asarray(seg, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (2, 3):
        return np.zeros((0, 2, 3), dtype=np.float32)
    return arr


def extract_toolpath_segments_for_range(
    simulation_txt: str | Path,
    *,
    z0_mm: float,
    slice_height_mm: float,
    max_layer_id_inclusive: int | None,
    max_jump_mm: float | None,
    max_segments: int = 600_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (range_segments, current_layer_segments).

    - range_segments: segments from layer 0..max_layer_id_inclusive (or all if None)
    - current_layer_segments: segments only at max_layer_id_inclusive (or empty if None)
    """
    if slice_height_mm <= 0:
        return (np.zeros((0, 2, 3), dtype=np.float32), np.zeros((0, 2, 3), dtype=np.float32))

    max_jump = float(max_jump_mm) if (max_jump_mm is not None and float(max_jump_mm) > 0) else None
    max_lid = int(max_layer_id_inclusive) if max_layer_id_inclusive is not None else None

    _hdr, rows = read_simulation_export(simulation_txt)

    last_pt: tuple[float, float, float] | None = None
    last_lid: int | None = None

    seg_range: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    seg_cur: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    for pt in rows:
        if pt.type != 1:
            last_pt = None
            last_lid = None
            continue
        try:
            if float(pt.factor) <= 0 or float(pt.bead_area) <= 0:
                last_pt = None
                last_lid = None
                continue
        except Exception:
            pass

        z = float(pt.z)
        lid = int(round((z - float(z0_mm)) / float(slice_height_mm)))
        if lid < 0:
            last_pt = None
            last_lid = None
            continue
        if max_lid is not None and lid > max_lid:
            last_pt = None
            last_lid = None
            continue

        cur = (float(pt.x), float(pt.y), z)
        if last_pt is not None and last_lid == lid:
            dx = cur[0] - last_pt[0]
            dy = cur[1] - last_pt[1]
            dz = cur[2] - last_pt[2]
            d = float((dx * dx + dy * dy + dz * dz) ** 0.5)
            if max_jump is None or d <= max_jump:
                seg_range.append((last_pt, cur))
                if max_lid is not None and lid == max_lid:
                    seg_cur.append((last_pt, cur))
        last_pt = cur
        last_lid = lid

        if len(seg_range) >= int(max_segments):
            break

    def _to_arr(seg: list[tuple[tuple[float, float, float], tuple[float, float, float]]]) -> np.ndarray:
        if not seg:
            return np.zeros((0, 2, 3), dtype=np.float32)
        arr = np.asarray(seg, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (2, 3):
            return np.zeros((0, 2, 3), dtype=np.float32)
        return arr

    return (_to_arr(seg_range), _to_arr(seg_cur))


@dataclass
class JobConfig:
    simulation_txt: str
    job_dir: str | None
    placed_stl: str
    output_dir: str
    voxel_size_mm: float
    max_radius_mm: float | None
    max_jump_mm: float | None
    min_component_voxels: int
    min_mesh_component_faces: int
    volume_smooth_sigma_vox: float
    meshing_downsample_factor: int
    smooth_iterations: int
    export_cae_layers: bool
    export_geometry_preview: bool
    export_cad_bundle: bool = True
    ansys_min_confidence: float = 0.2
    ansys_group_size_layers: int = 1
    ansys_create_named_selections: bool = True
    ansys_create_coordinate_systems: bool = True
    heal_enabled: bool = False
    heal_preset: str = "safe"
    heal_close_holes_max_mm: float = 2.0
    heal_report_enabled: bool = False
    heal_report_path: str | None = None
    heal_backend: str = "auto"


class Worker(QtCore.QObject):
    progress = QtCore.Signal(int)
    log = QtCore.Signal(str)
    meshes_ready = QtCore.Signal(object, object, object)
    finished = QtCore.Signal(bool, str, object)

    def __init__(self, cfg: JobConfig):
        super().__init__()
        self.cfg = cfg

    @QtCore.Slot()
    def run(self) -> None:
        try:
            t0 = time.time()
            scratch_run: Path | None = None
            scratch_note_logged = False

            def _scratch_size_bytes(p: Path) -> int:
                total = 0
                try:
                    if not p.exists():
                        return 0
                    for e in os.scandir(p):
                        try:
                            if e.is_file():
                                total += int(e.stat().st_size)
                        except Exception:
                            pass
                except Exception:
                    return 0
                return int(total)

            def _fmt_gb(n: int) -> str:
                try:
                    return f"{float(n) / float(1024**3):.2f} GB"
                except Exception:
                    return f"{n} bytes"

            sim_header, rows_iter = read_simulation_export(self.cfg.simulation_txt)
            for w in sim_header.validation_warnings():
                self.log.emit(f"ПРЕДУПРЕЖДЕНИЕ: {w}")
            slice_h = sim_header.slice_height_mm or 0.254

            # Single supported mode: CMB (Insight build coordinates).
            coord_space = "cmb"

            if not self.cfg.export_cae_layers and not self.cfg.export_geometry_preview:
                raise ValueError("Не выбраны выходные файлы: включите Геометрию и/или экспорт для ANSYS.")

            if self.cfg.export_geometry_preview:
                self.log.emit("Загрузка STL...")
                mesh = trimesh.load_mesh(self.cfg.placed_stl, force="mesh")
                bounds_min = mesh.bounds[0]
                bounds_max = mesh.bounds[1]
            else:
                bounds_min = None
                bounds_max = None
                try:
                    p_stl = Path(self.cfg.placed_stl) if self.cfg.placed_stl else None
                    if p_stl is not None and p_stl.exists():
                        self.log.emit("Загрузка STL (только bbox):")
                        _m = trimesh.load_mesh(str(p_stl), force="mesh")
                        bounds_min = _m.bounds[0]
                        bounds_max = _m.bounds[1]
                except Exception:
                    bounds_min = None
                    bounds_max = None

            self.log.emit("Подготовка преобразования координат (STL -> CMB)...")
            stl_to_cmb = sim_header.stl_to_cmb
            # CMB-only pipeline: toolpath table is already in CMB coordinates.

            # Full pipeline in Insight build coordinates. Toolpath table is already in CMB.
            layer_axis_xyz = (0.0, 0.0, 1.0)
            self.log.emit("Координаты: CMB (Insight). Ось печати: Z+")

            # Transform STL bounds to CMB so voxelization bounds are in the same space as toolpath.
            try:
                if bounds_min is not None and bounds_max is not None:
                    bmin = np.asarray(bounds_min, dtype=float)
                    bmax = np.asarray(bounds_max, dtype=float)
                    corners = np.array(
                        [
                            [bmin[0], bmin[1], bmin[2]],
                            [bmin[0], bmin[1], bmax[2]],
                            [bmin[0], bmax[1], bmin[2]],
                            [bmin[0], bmax[1], bmax[2]],
                            [bmax[0], bmin[1], bmin[2]],
                            [bmax[0], bmin[1], bmax[2]],
                            [bmax[0], bmax[1], bmin[2]],
                            [bmax[0], bmax[1], bmax[2]],
                        ],
                        dtype=float,
                    )
                    corners_cmb = transform_points_rowvec(corners, np.asarray(stl_to_cmb, dtype=float))
                    bounds_min = corners_cmb.min(axis=0)
                    bounds_max = corners_cmb.max(axis=0)
            except Exception:
                pass

            total = 0
            kept = 0
            counted = False

            def _count_points(points):
                nonlocal total, kept
                for p in points:
                    total += 1
                    if p.type == 1:
                        kept += 1
                    yield p

            bbox_min = np.array([np.inf, np.inf, np.inf], dtype=float)
            bbox_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
            bbox_count = 0

            def _track_bbox(points):
                nonlocal bbox_min, bbox_max, bbox_count
                for p in points:
                    if p.type == 1 and float(p.factor) > 0 and float(p.bead_area) > 0:
                        bbox_min[0] = min(float(bbox_min[0]), float(p.x))
                        bbox_min[1] = min(float(bbox_min[1]), float(p.y))
                        bbox_min[2] = min(float(bbox_min[2]), float(p.z))
                        bbox_max[0] = max(float(bbox_max[0]), float(p.x))
                        bbox_max[1] = max(float(bbox_max[1]), float(p.y))
                        bbox_max[2] = max(float(bbox_max[2]), float(p.z))
                        bbox_count += 1
                    yield p

            def _check_bbox_against_stl() -> None:
                if bounds_min is None or bounds_max is None:
                    return
                if not np.isfinite(bbox_min).all() or not np.isfinite(bbox_max).all() or bbox_count <= 0:
                    return

                stl_min = bounds_min.astype(float)
                stl_max = bounds_max.astype(float)
                diag = float(np.linalg.norm(stl_max - stl_min))
                tol = max(1.0, 2.0 * float(slice_h), 0.02 * diag)
                err = max(10.0, 10.0 * tol)

                exceed_low = stl_min - bbox_min
                exceed_high = bbox_max - stl_max
                exceed = np.maximum(exceed_low, exceed_high)
                max_exceed = float(np.max(exceed))

                if max_exceed <= tol:
                    return

                msg = (
                    "Габариты траекторий (CMB->STL) не совпадают с габаритами STL "
                    f"(макс. расхождение {max_exceed:.3f} мм, допуск {tol:.3f} мм). "
                    "Возможна неверная матрица, единицы или усадка."
                )
                if max_exceed >= err:
                    raise ValueError(msg)
                self.log.emit(f"ПРЕДУПРЕЖДЕНИЕ: {msg}")

            layers = []
            z0 = None

            if self.cfg.export_cae_layers:
                if bounds_min is not None and bounds_max is not None:
                    # CMB-only: build axis is always Z+.
                    z0 = float(bounds_min[2])
                else:
                    self.log.emit("Сканирование Z0 (min Z) для CAE...")
                    z0_val = None
                    for pt in _count_points(_track_bbox(rows_iter)):
                        if pt.type != 1:
                            continue
                        coord = float(
                            layer_axis_xyz[0] * float(pt.x) + layer_axis_xyz[1] * float(pt.y) + layer_axis_xyz[2] * float(pt.z)
                        )
                        z0_val = coord if z0_val is None else min(z0_val, coord)
                    counted = True
                    z0 = float(z0_val) if z0_val is not None else 0.0
                    sim_header, rows_iter = read_simulation_export(self.cfg.simulation_txt)

                self.log.emit("Расчёт ориентации печати по слоям (CAE)...")
                pts = _track_bbox(rows_iter)
                if bounds_min is None and not counted:
                    pts = _count_points(pts)
                    counted = True
                layers = compute_layer_orientations_toolpath(
                    pts,
                    slice_height_mm=float(slice_h),
                    z0_mm=float(z0),
                    layer_axis_xyz=layer_axis_xyz,
                    max_jump_mm=self.cfg.max_jump_mm,
                    weight_by_bead_area=True,
                )
                _check_bbox_against_stl()

            preview_mesh = None
            mesh_before = None
            vox = None
            if self.cfg.export_geometry_preview:
                # Expand bounds slightly to avoid clipping
                pad = float(self.cfg.max_radius_mm or 0.0) * 2.0
                bmin = bounds_min - pad
                bmax = bounds_max + pad

                self.log.emit("Построение объёма (вокселизация)...")
                self.progress.emit(35)
                _header2, rows2 = read_simulation_export(self.cfg.simulation_txt)
                pts_iter = _track_bbox(_count_points(rows2))
                counted = True

                scratch = None
                try:
                    base = os.environ.get("S2S_SCRATCH_DIR", "").strip()
                    scratch_base = Path(base) if base else (diag_dir / "scratch")
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    scratch_run = scratch_base / f"run_{stamp}_{os.getpid()}"
                    scratch_run.mkdir(parents=True, exist_ok=True)
                    scratch = scratch_run
                except Exception:
                    scratch = None
                if scratch is not None and not scratch_note_logged:
                    scratch_note_logged = True
                    keep = os.environ.get("S2S_KEEP_SCRATCH") == "1"
                    thr = os.environ.get("S2S_MEMMAP_THRESHOLD_MB", "256")
                    self.log.emit(f"Временная папка (scratch): {scratch}")
                    self.log.emit(f"Порог перехода на диск: {thr} MB (S2S_MEMMAP_THRESHOLD_MB)")
                    if keep:
                        self.log.emit("Временные файлы: НЕ удалять (S2S_KEEP_SCRATCH=1)")
                    else:
                        self.log.emit("Временные файлы: автоудаление после прогона (S2S_KEEP_SCRATCH=1 - не удалять)")

                # Preflight disk check: quality-first out-of-core uses scratch heavily.
                try:
                    if scratch is not None:
                        try:
                            thr_mb = int(os.environ.get("S2S_MEMMAP_THRESHOLD_MB", "256"))
                            thr_mb = max(0, thr_mb)
                        except Exception:
                            thr_mb = 256
                        threshold = int(thr_mb) * 1024 * 1024

                        size = (bmax - bmin).astype(float)
                        shape = tuple(int(math.ceil(float(s) / float(self.cfg.voxel_size_mm))) + 1 for s in size)
                        voxels = int(np.prod(shape, dtype=np.int64))
                        occ_bytes = int(voxels)  # bool memmap: ~1 byte / voxel
                        need = 0
                        if occ_bytes >= threshold:
                            need += occ_bytes
                        if float(self.cfg.volume_smooth_sigma_vox or 0.0) > 0:
                            vol_bytes = int(voxels) * 4  # float32
                            if vol_bytes >= threshold:
                                # Conservative: vol + blur + transpose
                                need += 3 * vol_bytes

                        if need > 0:
                            free = int(shutil.disk_usage(str(scratch)).free)
                            if need > int(free * 0.80):
                                need_gb = need / (1024**3)
                                free_gb = free / (1024**3)
                                raise RuntimeError(
                                    "Недостаточно места на диске для временных файлов.\n"
                                    f"Оценка: нужно ~{need_gb:.1f} GB, доступно ~{free_gb:.1f} GB.\n\n"
                                    "Решение (без потери качества геометрии):\n"
                                    " - Укажите папку на диске с большим свободным местом через `S2S_SCRATCH_DIR`.\n"
                                    " - Освободите место на диске.\n\n"
                                    "Если согласны на компромисс качества: увеличьте Voxel size или Downsample."
                                )
                except Exception:
                    # Fail early with clear message.
                    raise
                vox = voxelize_toolpath(
                    pts_iter,
                    voxel_size=self.cfg.voxel_size_mm,
                    bounds_min=bmin,
                    bounds_max=bmax,
                    slice_height=slice_h,
                    type_filter=1,
                    max_radius_mm=self.cfg.max_radius_mm,
                    max_jump_mm=self.cfg.max_jump_mm,
                    min_component_voxels=self.cfg.min_component_voxels,
                    scratch_dir=scratch,
                )
                _check_bbox_against_stl()
                self.progress.emit(70)

                self.log.emit("Построение поверхности (marching cubes)...")
                preview_mesh = mesh_from_voxels_configured(
                    vox,
                    volume_smooth_sigma_vox=self.cfg.volume_smooth_sigma_vox,
                    min_component_faces=self.cfg.min_mesh_component_faces,
                    downsample_factor=int(self.cfg.meshing_downsample_factor)
                    if int(self.cfg.meshing_downsample_factor) > 1
                    else None,
                    scratch_dir=scratch,
                )
                try:
                    ds = int(preview_mesh.metadata.get("meshing_downsample_factor", 1))
                    eff = float(preview_mesh.metadata.get("meshing_voxel_size_mm", float(self.cfg.voxel_size_mm)))
                    if ds > 1:
                        self.log.emit(f"Разрежение при построении поверхности: x{ds} (эффективный шаг {eff:.3f} мм)")
                except Exception:
                    pass
                mesh_before = preview_mesh.copy()
                if self.cfg.smooth_iterations > 0:
                    self.log.emit(f"Сглаживание сетки ({self.cfg.smooth_iterations} итераций)...")
                    trimesh.smoothing.filter_laplacian(preview_mesh, iterations=int(self.cfg.smooth_iterations))
                mesh_after = preview_mesh
                try:
                    target_preview_faces = 600_000

                    def _build_display_mesh(base_mesh: trimesh.Trimesh, *, post_smooth: bool) -> tuple[trimesh.Trimesh, int]:
                        if base_mesh.faces is not None and len(base_mesh.faces) <= target_preview_faces:
                            return base_mesh, int(base_mesh.metadata.get("meshing_downsample_factor", 1) or 1)
                        if vox is None:
                            return base_mesh, int(base_mesh.metadata.get("meshing_downsample_factor", 1) or 1)
                        base_ds = max(1, int(self.cfg.meshing_downsample_factor))
                        faces_count = int(len(base_mesh.faces)) if base_mesh.faces is not None else target_preview_faces + 1
                        ratio = max(1.0, faces_count / float(target_preview_faces))
                        mul = int(math.ceil(math.sqrt(ratio)))
                        mul_pow2 = 1 << int(max(0, mul - 1)).bit_length()
                        ds = min(64, base_ds * mul_pow2)
                        display = mesh_from_voxels_configured(
                            vox,
                            volume_smooth_sigma_vox=self.cfg.volume_smooth_sigma_vox,
                            min_component_faces=self.cfg.min_mesh_component_faces,
                            downsample_factor=int(ds) if int(ds) > 1 else None,
                        )
                        if post_smooth and int(self.cfg.smooth_iterations) > 0:
                            trimesh.smoothing.filter_laplacian(display, iterations=int(self.cfg.smooth_iterations))
                        return display, int(ds)

                    disp_before, disp_ds_before = _build_display_mesh(mesh_before, post_smooth=False)
                    disp_after, disp_ds_after = _build_display_mesh(mesh_after, post_smooth=True)

                    stats = {
                        "before": {"vertices": int(mesh_before.vertices.shape[0]), "faces": int(mesh_before.faces.shape[0])},
                        "after": {"vertices": int(mesh_after.vertices.shape[0]), "faces": int(mesh_after.faces.shape[0])},
                        "display_before": {
                            "vertices": int(disp_before.vertices.shape[0]),
                            "faces": int(disp_before.faces.shape[0]),
                            "ds": int(disp_ds_before),
                        },
                        "display_after": {
                            "vertices": int(disp_after.vertices.shape[0]),
                            "faces": int(disp_after.faces.shape[0]),
                            "ds": int(disp_ds_after),
                        },
                    }
                    self.meshes_ready.emit(disp_before, disp_after, stats)
                except Exception:
                    pass
                self.progress.emit(85)

            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Logical output structure:
            #   out/
            #     CAD/  (explicit infill geometry + CAD bundle)
            #     CAE/  (layer orientation tables + ANSYS Mechanical scripts)
            #     DIAG/ (optional slicer diagnostics)
            cad_dir = out_dir / "CAD"
            cae_dir = out_dir / "CAE"
            diag_dir = out_dir / "DIAG"
            cad_dir.mkdir(parents=True, exist_ok=True)
            cae_dir.mkdir(parents=True, exist_ok=True)
            diag_dir.mkdir(parents=True, exist_ok=True)

            # Prefix geometry outputs to reduce confusion with the user-provided STL.
            preview_stem = _preview_mesh_stem(self.cfg)
            out_stl = cad_dir / f"{preview_stem}.stl"
            # PLY is an optional CAD-bundle artifact; name it explicitly to avoid confusion with the STL.
            out_ply = cad_dir / f"{preview_stem}_mesh.ply"
            out_notes = cad_dir / "cad_import_notes.txt"
            out_points = cad_dir / "voxel_points.csv"
            out_json = out_dir / "metadata.json"
            out_layers_json = cae_dir / "ansys_layers.json"
            out_layers_csv = cae_dir / "ansys_layers.csv"
            out_ansys_script = cae_dir / "ansys_mechanical_import_layers.py"
            out_section_planes_script = cae_dir / "ansys_mechanical_section_planes.py"
            out_cae_notes = cae_dir / "cae_import_notes.txt"

            outputs: list[str] = []
            bundle_written = False

            # In CMB mode, export the input STL transformed into CMB coordinates for easy CAE import/validation.
            try:
                # Intentionally not exported: we keep a single geometry output (`*_s2s_preview_structure.stl`) to avoid confusion.
                pass
            except Exception:
                pass

            if self.cfg.export_geometry_preview and preview_mesh is not None:
                self.log.emit(f"Запись {out_stl}...")
                preview_mesh.export(out_stl)
                outputs.append(str(out_stl))
                if bool(self.cfg.heal_enabled):
                    try:
                        healed_stl = cad_dir / f"{out_stl.stem}_healed{out_stl.suffix}"
                        report_path = None
                        if bool(self.cfg.heal_report_enabled):
                            if self.cfg.heal_report_path:
                                report_path = Path(self.cfg.heal_report_path)
                            else:
                                report_path = cad_dir / f"{out_stl.stem}_healed_report.json"
                        self.log.emit(
                            f"Mesh Healer: preset={self.cfg.heal_preset}, close_holes_max={self.cfg.heal_close_holes_max_mm} мм"
                        )
                        heal_mesh_file(
                            out_stl,
                            out_path=healed_stl,
                            preset=str(self.cfg.heal_preset),
                            close_holes_max_mm=float(self.cfg.heal_close_holes_max_mm),
                            report_path=report_path,
                            backend=str(self.cfg.heal_backend),
                        )
                        outputs.append(str(healed_stl))
                        if report_path is not None:
                            outputs.append(str(report_path))
                    except Exception as e:
                        self.log.emit(f"ПРЕДУПРЕЖДЕНИЕ: исправление сетки (Mesh Healer) не удалось: {e}")
                if self.cfg.export_cad_bundle:
                    try:
                        self.log.emit(f"Запись {out_ply}...")
                        preview_mesh.export(out_ply)
                        outputs.append(str(out_ply))

                        notes = _render_cad_import_notes(self.cfg)
                        out_notes.write_text(notes, encoding="utf-8")
                        outputs.append(str(out_notes))

                        if vox is not None:
                            self.log.emit(f"Запись {out_points}...")
                            res = export_voxel_centers_csv(
                                vox.occupied,
                                origin_xyz_mm=vox.origin,
                                voxel_size_mm=float(vox.voxel_size),
                                out_csv=out_points,
                                max_points=250_000,
                                include_header=False,
                            )
                            note = "sampled" if res.sampled else "all"
                            self.log.emit(f"Точки (voxel): {res.points_written:,}/{res.points_total:,} ({note})")
                            outputs.append(str(out_points))
                        bundle_written = True
                    except Exception as e:
                        self.log.emit(f"ПРЕДУПРЕЖДЕНИЕ: пакет для CAD пропущен: {e}")

            if self.cfg.export_cae_layers:
                self.log.emit(f"Запись {out_layers_json}...")
                out_layers_json.write_text(
                    json.dumps(
                        {
                            "slice_height_mm": float(slice_h),
                            "z0_mm": float(z0) if z0 is not None else None,
                            "build_axis": "z",
                            "build_axis_sign": 1,
                            "layer_axis_xyz": [0.0, 0.0, 1.0],
                            "layers": [
                                {
                                    "layer_id": l.layer_id,
                                    "z_min": l.z_min,
                                    "z_max": l.z_max,
                                    "z_center": l.z_center,
                                    "dir_xyz": list(l.dir_xyz) if l.dir_xyz is not None else None,
                                    "angle_deg": l.angle_deg,
                                    "confidence": l.confidence,
                                    "segments_used": l.segments_used,
                                    "total_weight": l.total_weight,
                                }
                                for l in layers
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                outputs.append(str(out_layers_json))
                self.log.emit(f"Запись {out_layers_csv}...")
                lines = ["layer_id,z_min_mm,z_max_mm,angle_deg,dx,dy,dz,confidence,segments_used,total_weight"]
                for l in layers:
                    if l.dir_xyz is None:
                        dx = dy = dz = ""
                        ang = ""
                    else:
                        dx, dy, dz = l.dir_xyz
                        ang = f"{l.angle_deg:.6f}" if l.angle_deg is not None else ""
                    lines.append(
                        f"{l.layer_id},{l.z_min:.6f},{l.z_max:.6f},{ang},{dx},{dy},{dz},{l.confidence:.6f},{l.segments_used},{l.total_weight:.6f}"
                    )
                out_layers_csv.write_text("\n".join(lines), encoding="utf-8")
                outputs.append(str(out_layers_csv))

                self.log.emit(f"Запись {out_ansys_script}...")
                out_ansys_script.write_text(
                    # CAE default: import nominal CAD/STL (original coordinates) and apply STL->CMB inside the script.
                    # User can set APPLY_STL_TO_CMB=False in the script if their model is already in CMB.
                    _render_ansys_mechanical_script(self.cfg, stl_to_cmb=np.asarray(stl_to_cmb, dtype=float), apply_stl_to_cmb=True),
                    encoding="utf-8",
                )
                outputs.append(str(out_ansys_script))

                # Mechanical helper: create a Section Plane (and optionally export PNGs per layer).
                out_section_planes_script.write_text(
                    _render_ansys_mechanical_section_planes_script(
                        self.cfg,
                        stl_to_cmb=np.asarray(stl_to_cmb, dtype=float),
                        apply_stl_to_cmb=True,
                    ),
                    encoding="utf-8",
                )
                outputs.append(str(out_section_planes_script))

                out_cae_notes.write_text(_render_cae_import_notes(self.cfg), encoding="utf-8")
                outputs.append(str(out_cae_notes))

                # Optional: extract Insight SGM (slice geometry) for diagnostics / validation.
                try:
                    if self.cfg.job_dir and Path(self.cfg.job_dir).exists():
                        extracted = extract_sgm_to_folder(self.cfg.job_dir, diag_dir)
                        if extracted is not None:
                            outputs.append(str(extracted.extracted_path))
                except Exception:
                    pass

            meta = {
                "inputs": asdict(self.cfg),
                "coordinate_space": "cmb",
                "simulation_header": sim_header.raw,
                "stl_to_cmb_matrix": sim_header.stl_to_cmb.tolist(),
                "outputs": outputs,
                "mesh": None,
                "voxel": None,
                "stats": {
                    "rows_total": total,
                    "rows_type1": kept,
                    "elapsed_s": time.time() - t0,
                },
            }
            if vox is not None:
                meta["voxel"] = {
                    "voxel_size_mm": vox.voxel_size,
                    "origin": vox.origin.tolist(),
                    "shape": list(vox.shape),
                    "occupied_voxels": int(vox.occupied.sum()),
                }
            if self.cfg.export_geometry_preview and preview_mesh is not None:
                try:
                    meta["mesh"] = {
                        "vertices": int(preview_mesh.vertices.shape[0]),
                        "faces": int(preview_mesh.faces.shape[0]),
                        "meshing_downsample_factor": int(preview_mesh.metadata.get("meshing_downsample_factor", 1)),
                        "meshing_voxel_size_mm": float(
                            preview_mesh.metadata.get("meshing_voxel_size_mm", float(self.cfg.voxel_size_mm))
                        ),
                        "estimated_binary_stl_size_bytes": int(84 + 50 * int(preview_mesh.faces.shape[0])),
                    }
                except Exception:
                    meta["mesh"] = None
            out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            outputs.append(str(out_json))

            # Cleanup scratch by default: memmap files can be huge.
            try:
                if scratch_run is not None and os.environ.get("S2S_KEEP_SCRATCH") != "1":
                    before = _scratch_size_bytes(scratch_run)
                    shutil.rmtree(str(scratch_run), ignore_errors=False)
                    self.log.emit(f"Временные файлы удалены: освобождено ~{_fmt_gb(before)}")
            except Exception:
                try:
                    if scratch_run is not None and scratch_run.exists():
                        sz = _scratch_size_bytes(scratch_run)
                        self.log.emit(
                            f"ПРЕДУПРЕЖДЕНИЕ: не удалось удалить временные файлы; осталось ~{_fmt_gb(sz)} в {scratch_run}"
                        )
                except Exception:
                    self.log.emit("ПРЕДУПРЕЖДЕНИЕ: не удалось удалить временные файлы.")

            self.progress.emit(100)
            extra = ""
            if self.cfg.export_cae_layers:
                extra = f", {out_layers_json.name}, {out_layers_csv.name}, {out_ansys_script.name}"
            if self.cfg.export_geometry_preview:
                bundle_part = f", {out_ply.name}, {out_notes.name}, {out_points.name}" if bundle_written else ""
                base = f"{out_stl.name}{bundle_part}, {out_json.name}{extra}"
            else:
                base = f"{out_json.name}{extra}"
            self.finished.emit(True, f"Готово. Файлы: {base}", outputs)
        except Exception as e:
            try:
                if scratch_run is not None and os.environ.get("S2S_KEEP_SCRATCH") != "1":
                    shutil.rmtree(str(scratch_run), ignore_errors=True)
            except Exception:
                pass
            self.finished.emit(False, f"Ошибка: {e}", [])


def _load_app_icon() -> QtGui.QIcon:
    try:
        svg_bytes = resources.files("slice2solid.gui.assets").joinpath("herb.svg").read_bytes()
    except Exception:
        return QtGui.QIcon()

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg_bytes))
    if not renderer.isValid():
        return QtGui.QIcon()

    icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
        img.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(img)
        renderer.render(p, QtCore.QRectF(0, 0, size, size))
        p.end()
        icon.addPixmap(QtGui.QPixmap.fromImage(img))
    return icon


def _load_logo_pixmap(size: int = 72) -> QtGui.QPixmap | None:
    try:
        svg_bytes = resources.files("slice2solid.gui.assets").joinpath("herb.svg").read_bytes()
    except Exception:
        return None

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg_bytes))
    if not renderer.isValid():
        return None

    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    renderer.render(p, QtCore.QRectF(0, 0, size, size))
    p.end()
    return QtGui.QPixmap.fromImage(img)


def _create_splash_pixmap(width: int = 640, height: int = 360) -> QtGui.QPixmap:
    img = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.transparent)

    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

    bg = QtGui.QLinearGradient(0.0, 0.0, float(width), float(height))
    bg.setColorAt(0.0, QtGui.QColor("#061A2D"))
    bg.setColorAt(0.55, QtGui.QColor("#0B2A4A"))
    bg.setColorAt(1.0, QtGui.QColor("#0F3D66"))
    p.fillRect(QtCore.QRectF(0.0, 0.0, float(width), float(height)), bg)

    # Subtle "layers" motif.
    layer_color = QtGui.QColor(255, 255, 255, 18)
    p.setPen(QtGui.QPen(layer_color, 1.0))
    for i in range(18):
        y = int(height * 0.18 + i * (height * 0.035))
        p.drawLine(int(width * 0.08), y, int(width * 0.92), y)

    # Logo mark: S2S + stacked layers.
    mark_r = int(min(width, height) * 0.16)
    mark_cx = int(width * 0.18)
    mark_cy = int(height * 0.42)
    mark_rect = QtCore.QRect(mark_cx - mark_r, mark_cy - mark_r, 2 * mark_r, 2 * mark_r)

    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(QtGui.QColor("#0E9AA7"))
    p.drawEllipse(mark_rect)

    p.setBrush(QtGui.QColor(255, 255, 255, 235))
    inner = mark_rect.adjusted(int(mark_r * 0.14), int(mark_r * 0.14), -int(mark_r * 0.14), -int(mark_r * 0.14))
    p.drawEllipse(inner)

    p.setPen(QtGui.QPen(QtGui.QColor("#0B2A4A"), max(2, int(mark_r * 0.06))))
    y0 = inner.center().y() - int(mark_r * 0.22)
    for j in range(3):
        yy = y0 + j * int(mark_r * 0.22)
        p.drawLine(inner.left() + int(mark_r * 0.18), yy, inner.right() - int(mark_r * 0.18), yy)

    font = QtGui.QFont()
    font.setBold(True)
    font.setPixelSize(int(mark_r * 0.44))
    p.setFont(font)
    p.setPen(QtGui.QPen(QtGui.QColor("#0B2A4A")))
    p.drawText(inner, int(QtCore.Qt.AlignmentFlag.AlignCenter), "S2S")

    # University crest (existing SVG icon) to keep the academic identity visible.
    crest = _load_logo_pixmap(int(min(width, height) * 0.16))
    if crest is not None and not crest.isNull():
        crest_size = int(min(width, height) * 0.14)
        crest = crest.scaled(
            crest_size,
            crest_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        crest_x = int(width * 0.90) - crest.width()
        crest_y = int(height * 0.12)
        p.setOpacity(0.92)
        p.drawPixmap(crest_x, crest_y, crest)
        p.setOpacity(1.0)

    # Title block.
    title = "slice2solid"
    subtitle = "Восстановление структуры"
    version = f"v{VERSION}"

    title_font = QtGui.QFont()
    title_font.setBold(True)
    title_font.setPixelSize(int(height * 0.11))
    p.setFont(title_font)
    p.setPen(QtGui.QColor("#FFFFFF"))
    p.drawText(QtCore.QRectF(width * 0.30, height * 0.28, width * 0.66, height * 0.20), title)

    sub_font = QtGui.QFont()
    sub_font.setPixelSize(int(height * 0.055))
    p.setFont(sub_font)
    p.setPen(QtGui.QColor(235, 244, 255, 225))
    p.drawText(QtCore.QRectF(width * 0.30, height * 0.47, width * 0.66, height * 0.12), subtitle)

    ver_font = QtGui.QFont()
    ver_font.setPixelSize(int(height * 0.045))
    p.setFont(ver_font)
    p.setPen(QtGui.QColor(255, 255, 255, 175))
    p.drawText(QtCore.QRectF(width * 0.30, height * 0.58, width * 0.66, height * 0.10), version)

    footer_font = QtGui.QFont()
    footer_font.setPixelSize(int(height * 0.035))
    p.setFont(footer_font)
    p.setPen(QtGui.QColor(255, 255, 255, 150))
    org = ORGANIZATION or ""
    dept = DEPARTMENT or ""
    author = AUTHOR or ""
    footer = " • ".join([x for x in (org, dept, author) if x])
    p.drawText(QtCore.QRectF(width * 0.08, height * 0.84, width * 0.84, height * 0.14), footer)

    p.end()
    return QtGui.QPixmap.fromImage(img)


def _apply_app_style(app: QtWidgets.QApplication) -> None:
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    try:
        font = QtGui.QFont("Segoe UI", 10)
        app.setFont(font)
    except Exception:
        pass

    try:
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#F6F7FB"))
        pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#111827"))
        pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#FFFFFF"))
        pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#F3F4F6"))
        pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#111827"))
        pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#FFFFFF"))
        pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#111827"))
        pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#FFFFFF"))
        pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#111827"))
        pal.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#EF4444"))
        pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#2563EB"))
        pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#FFFFFF"))
        app.setPalette(pal)
    except Exception:
        pass

    qss = """
        QWidget { color: #111827; }

        QGroupBox {
            border: 1px solid #D7DBE5;
            border-radius: 10px;
            margin-top: 10px;
            background: rgba(255,255,255,0.75);
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
            color: #111827;
            font-weight: 600;
        }

        QTabWidget::pane {
            border: 1px solid #D7DBE5;
            border-radius: 10px;
            top: -1px;
            background: #FFFFFF;
        }
        QTabBar::tab {
            background: #EEF2FF;
            border: 1px solid #D7DBE5;
            border-bottom: none;
            padding: 8px 12px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            margin-right: 4px;
        }
        QTabBar::tab:selected {
            background: #FFFFFF;
            font-weight: 600;
        }

        QPushButton {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 10px;
            padding: 7px 12px;
        }
        QPushButton:hover { border-color: #94A3B8; background: #F8FAFC; }
        QPushButton:pressed { background: #EEF2FF; }
        QPushButton:disabled { color: #9CA3AF; background: #F3F4F6; border-color: #E5E7EB; }

        QPushButton#primaryButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563EB, stop:1 #06B6D4);
            color: #FFFFFF;
            border: 1px solid #1D4ED8;
            font-weight: 600;
        }
        QPushButton#primaryButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1D4ED8, stop:1 #0891B2);
        }
        QPushButton#primaryButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E40AF, stop:1 #0E7490);
        }

        QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 10px;
            padding: 6px 10px;
            selection-background-color: #2563EB;
            selection-color: white;
        }
        QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QTextBrowser:focus {
            border-color: #2563EB;
        }

        QComboBox, QSpinBox, QDoubleSpinBox {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 10px;
            padding: 6px 10px;
        }
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #2563EB; }

        QProgressBar {
            border: 1px solid #CBD5E1;
            border-radius: 9px;
            background: #FFFFFF;
            text-align: center;
            color: #111827;
        }
        QProgressBar::chunk {
            border-radius: 8px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #2563EB, stop:1 #06B6D4);
        }

        QMenuBar { background: #FFFFFF; border-bottom: 1px solid #E5E7EB; }
        QMenuBar::item { padding: 6px 10px; background: transparent; }
        QMenuBar::item:selected { background: #EEF2FF; border-radius: 8px; }
        QMenu { background: #FFFFFF; border: 1px solid #D7DBE5; }
        QMenu::item { padding: 6px 14px; }
        QMenu::item:selected { background: #EEF2FF; }
    """
    try:
        app.setStyleSheet(qss)
    except Exception:
        pass


def _about_html() -> str:
    title = html.escape(APP_DISPLAY_NAME)
    version = html.escape(VERSION)
    author = html.escape(AUTHOR)
    org = html.escape(ORGANIZATION)
    dept = html.escape(DEPARTMENT)
    email = html.escape(CONTACT_EMAIL)
    return (
        f"<div style='font-size: 14px'>"
        f"<b>{title}</b> <span style='color: #666'>v{version}</span><br>"
        f"<span style='color: #444'>{author}</span><br>"
        f"<span style='color: #444'>{org}</span><br>"
        f"<span style='color: #444'>{dept}</span><br><br>"
        f"Контакты: <a href='mailto:{email}'>{email}</a>"
        f"</div>"
    )


_ANSYS_MECHANICAL_SCRIPT_TEMPLATE = r'''# slice2solid -> ANSYS Mechanical (Workbench) import helper
#
# Tested target: ANSYS Mechanical (Workbench) scripting API.
#
# This script tries to:
#   1) Load layer orientation table from ansys_layers.json
#   2) Create Named Selections of MESH ELEMENTS per layer (by element centroid along build axis)
#   3) Create Coordinate Systems per layer (X along print direction, Z along build axis)
#
# IMPORTANT:
# - Mechanical scripting APIs can differ. If this script errors, copy the error text to chat and we will adapt it.
# - Creating one Named Selection per layer may be heavy for tall parts. If needed we can add grouping (e.g., 5 layers per group).
#
# How to use:
#   Mechanical -> Automation -> Scripting -> Run Script -> select this .py file.
#
# Expected files in the SAME folder:
#   - ansys_layers.json
#
#
# This file is auto-generated by slice2solid. You can tweak the CONFIG values below and re-run in Mechanical.
#
import sys

_IS_IRONPYTHON = (getattr(sys, "platform", "") == "cli") or ("IronPython" in getattr(sys, "version", ""))
if not _IS_IRONPYTHON:
    try:
        import json
        _HAVE_JSON = True
    except Exception:
        json = None
        _HAVE_JSON = False
else:
    json = None
    _HAVE_JSON = False
import bisect
import math
import os

LOG_PATH = None

try:
    BUILD_AXIS
except Exception:
    BUILD_AXIS = "z"
try:
    BUILD_SIGN
except Exception:
    BUILD_SIGN = 1

try:
    APPLY_STL_TO_CMB
except Exception:
    # When True: element centroids are transformed using STL->CMB before layer selection/CS orientation.
    # Use this when your imported geometry is in the original STL/CAD coordinates (not already in CMB).
    APPLY_STL_TO_CMB = True
try:
    STL_TO_CMB
except Exception:
    # 4x4 row-vector matrix (translation in last row). Injected by slice2solid GUI.
    STL_TO_CMB = None

LAYER_TO_ELEM = 1.0


def _rowvec_mul3(v, m):
    # out = v @ m  (row-vector convention)
    return (
        float(v[0]) * float(m[0][0]) + float(v[1]) * float(m[1][0]) + float(v[2]) * float(m[2][0]),
        float(v[0]) * float(m[0][1]) + float(v[1]) * float(m[1][1]) + float(v[2]) * float(m[2][1]),
        float(v[0]) * float(m[0][2]) + float(v[1]) * float(m[1][2]) + float(v[2]) * float(m[2][2]),
    )


def _mat3_inv(a):
    # Invert a 3x3 matrix (lists) in pure Python.
    a00, a01, a02 = float(a[0][0]), float(a[0][1]), float(a[0][2])
    a10, a11, a12 = float(a[1][0]), float(a[1][1]), float(a[1][2])
    a20, a21, a22 = float(a[2][0]), float(a[2][1]), float(a[2][2])
    b00 = a11 * a22 - a12 * a21
    b01 = a02 * a21 - a01 * a22
    b02 = a01 * a12 - a02 * a11
    b10 = a12 * a20 - a10 * a22
    b11 = a00 * a22 - a02 * a20
    b12 = a02 * a10 - a00 * a12
    b20 = a10 * a21 - a11 * a20
    b21 = a01 * a20 - a00 * a21
    b22 = a00 * a11 - a01 * a10
    det = a00 * b00 + a01 * b10 + a02 * b20
    if abs(det) <= 1e-18:
        return None
    inv_det = 1.0 / det
    return [
        [b00 * inv_det, b01 * inv_det, b02 * inv_det],
        [b10 * inv_det, b11 * inv_det, b12 * inv_det],
        [b20 * inv_det, b21 * inv_det, b22 * inv_det],
    ]


_LIN_INV = None


def _lin_inv():
    global _LIN_INV
    if _LIN_INV is not None:
        return _LIN_INV
    if not STL_TO_CMB:
        _LIN_INV = None
        return None
    try:
        lin = [
            [float(STL_TO_CMB[0][0]), float(STL_TO_CMB[0][1]), float(STL_TO_CMB[0][2])],
            [float(STL_TO_CMB[1][0]), float(STL_TO_CMB[1][1]), float(STL_TO_CMB[1][2])],
            [float(STL_TO_CMB[2][0]), float(STL_TO_CMB[2][1]), float(STL_TO_CMB[2][2])],
        ]
        _LIN_INV = _mat3_inv(lin)
        return _LIN_INV
    except Exception:
        _LIN_INV = None
        return None


def _stl_to_cmb_point(xyz):
    if not (bool(APPLY_STL_TO_CMB) and STL_TO_CMB):
        return xyz
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    m = STL_TO_CMB
    # Translation is given in layer units (typically mm). Scale to element units using LAYER_TO_ELEM.
    tx = float(m[3][0]) * float(LAYER_TO_ELEM)
    ty = float(m[3][1]) * float(LAYER_TO_ELEM)
    tz = float(m[3][2]) * float(LAYER_TO_ELEM)
    return (
        x * float(m[0][0]) + y * float(m[1][0]) + z * float(m[2][0]) + tx,
        x * float(m[0][1]) + y * float(m[1][1]) + z * float(m[2][1]) + ty,
        x * float(m[0][2]) + y * float(m[1][2]) + z * float(m[2][2]) + tz,
    )


def _cmb_vec_to_model(v):
    if not (bool(APPLY_STL_TO_CMB) and STL_TO_CMB):
        return v
    inv = _lin_inv()
    if inv is None:
        return v
    return _rowvec_mul3(v, inv)

def _axis_index(axis):
    a = str(axis or "z").strip().lower()
    if a not in ("x", "y", "z"):
        a = "z"
    return {"x": 0, "y": 1, "z": 2}[a]

def _build_sign():
    try:
        return 1 if int(BUILD_SIGN) >= 0 else -1
    except Exception:
        return 1

def _build_coord(xyz):
    # build_height = sign * coord_axis
    xyz = _stl_to_cmb_point(xyz)
    idx = _axis_index(BUILD_AXIS)
    s = _build_sign()
    return float(s) * float(xyz[idx])

def _build_axis_vec_xyz():
    idx = _axis_index(BUILD_AXIS)
    s = _build_sign()
    v = [0.0, 0.0, 0.0]
    v[idx] = float(s)
    vv = (float(v[0]), float(v[1]), float(v[2]))
    return _unit(_cmb_vec_to_model(vv))

def _log(*args):
    msg = " ".join([str(a) for a in args])
    global LOG_PATH
    try:
        if LOG_PATH is None:
            LOG_PATH = os.path.join(os.path.dirname(__file__), "ansys_mechanical_import_log.txt")
        try:
            f = open(LOG_PATH, "a")
            try:
                f.write(msg + "\n")
            finally:
                f.close()
        except Exception:
            pass
    except Exception:
        pass
    try:
        if "ExtAPI" in globals() and hasattr(ExtAPI, "Log") and hasattr(ExtAPI.Log, "WriteMessage"):
            ExtAPI.Log.WriteMessage(msg)
            return
    except Exception:
        pass
    try:
        print(msg)
    except Exception:
        pass

def _get_settable_props(obj):
    try:
        if "System" in globals() and System is not None:
            t = obj.GetType()
            props = t.GetProperties()
            out = []
            for p in props:
                try:
                    if p.CanWrite:
                        out.append((p.Name, str(p.PropertyType)))
                except Exception:
                    pass
            return out
    except Exception:
        pass
    return []

try:
    import System
    from System.IO import File

    def _read_text(path):
        return File.ReadAllText(path)
except Exception:
    System = None

    def _read_text(path):
        f = open(path, "r")
        try:
            return f.read()
        finally:
            f.close()

HERE = os.path.dirname(__file__)
LAYERS_JSON = os.path.join(HERE, "ansys_layers.json")
LAYERS_CSV = os.path.join(HERE, "ansys_layers.csv")

# ---- CONFIG (can be edited) ----
# Values are injected by slice2solid GUI, but remain editable here.
try:
    MIN_CONFIDENCE
except NameError:
    MIN_CONFIDENCE = 0.2
try:
    GROUP_SIZE_LAYERS
except NameError:
    GROUP_SIZE_LAYERS = 1  # 1 = per-layer; 5 = one NS/CS per 5 layers
try:
    CREATE_NAMED_SELECTIONS
except NameError:
    CREATE_NAMED_SELECTIONS = True
try:
    CREATE_COORDINATE_SYSTEMS
except NameError:
    CREATE_COORDINATE_SYSTEMS = True


def _unit(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _safe_get_mesh(model):
    # Mechanical scripting APIs differ across versions (and IronPython builds).
    # Try a few common locations for MeshData.
    candidates = []
    try:
        if model is not None:
            candidates.append(model)
    except Exception:
        pass

    try:
        if hasattr(ExtAPI, "DataModel") and hasattr(ExtAPI.DataModel, "Project"):
            m = ExtAPI.DataModel.Project.Model
            candidates.append(m)
            if hasattr(m, "Analyses"):
                try:
                    for a in m.Analyses:
                        candidates.append(a)
                        if hasattr(a, "Analysis"):
                            candidates.append(a.Analysis)
                except Exception:
                    pass
    except Exception:
        pass

    # Direct MeshData property
    for obj in candidates:
        try:
            if hasattr(obj, "MeshData"):
                return obj.MeshData
        except Exception:
            pass

    # Some builds expose MeshData under Mesh object or via GetMeshData()
    for obj in candidates:
        try:
            if hasattr(obj, "Mesh"):
                mo = obj.Mesh
                if hasattr(mo, "MeshData"):
                    return mo.MeshData
                if hasattr(mo, "GetMeshData"):
                    return mo.GetMeshData()
        except Exception:
            pass

    # Emit diagnostics to help adapt to this Mechanical API.
    try:
        _log("DEBUG: MeshData not found; dumping mesh-related attributes.")
        for obj in candidates:
            try:
                names = [n for n in dir(obj) if "Mesh" in n or "mesh" in n]
            except Exception:
                names = []
            _log("  candidate=", type(obj), "names=", names[:50])
    except Exception:
        pass

    raise RuntimeError(
        "Could not find MeshData on Model. Make sure mesh is generated, then report your ANSYS version + this error."
    )


def _iter_element_ids(mesh):
    # Try common properties.
    if hasattr(mesh, "ElementIds"):
        for eid in mesh.ElementIds:
            yield int(eid)
        return
    if hasattr(mesh, "Elements"):
        for e in mesh.Elements:
            if hasattr(e, "Id"):
                yield int(e.Id)
            else:
                yield int(e)
        return
    raise RuntimeError("Could not iterate element IDs from MeshData (no ElementIds/Elements).")


def _element_centroid(mesh, eid):
    e = mesh.ElementById(eid)
    # Try common centroid attribute names.
    for name in ("Centroid", "CentroidCoordinate", "Center", "CenterOfGravity"):
        if hasattr(e, name):
            c = getattr(e, name)
            # c may be an object with X/Y/Z or a tuple/list
            if hasattr(c, "X") and hasattr(c, "Y") and hasattr(c, "Z"):
                return (float(c.X), float(c.Y), float(c.Z))
            if isinstance(c, (tuple, list)) and len(c) >= 3:
                return (float(c[0]), float(c[1]), float(c[2]))
    # Fallback: compute from element nodes (some Mechanical APIs don't expose element centroid directly).
    node_ids = None
    for nname in ("NodeIds", "NodeIDs", "Nodes", "Connectivity", "CornerNodeIds", "CornerNodes"):
        if not hasattr(e, nname):
            continue
        v = getattr(e, nname)
        try:
            if isinstance(v, (list, tuple)):
                node_ids = v
                break
            node_ids = list(v)
            break
        except Exception:
            pass

    if node_ids is not None and len(node_ids) > 0:
        xs = 0.0
        ys = 0.0
        zs = 0.0
        n = 0.0
        for nid in node_ids:
            try:
                if hasattr(nid, "Id"):
                    nid2 = int(nid.Id)
                else:
                    nid2 = int(nid)
                nd = mesh.NodeById(nid2) if hasattr(mesh, "NodeById") else None
                if nd is None and hasattr(mesh, "Nodes"):
                    nd = mesh.Nodes[nid2]
                if nd is None:
                    continue
                if hasattr(nd, "X") and hasattr(nd, "Y") and hasattr(nd, "Z"):
                    xs += float(nd.X)
                    ys += float(nd.Y)
                    zs += float(nd.Z)
                elif hasattr(nd, "Coordinate"):
                    c = nd.Coordinate
                    xs += float(c.X)
                    ys += float(c.Y)
                    zs += float(c.Z)
                elif hasattr(nd, "Coordinates"):
                    c = nd.Coordinates
                    xs += float(c[0])
                    ys += float(c[1])
                    zs += float(c[2])
                else:
                    continue
                n += 1.0
            except Exception:
                continue
        if n > 0:
            return (xs / n, ys / n, zs / n)

    raise RuntimeError("Could not read element centroid for element id={}".format(eid))


def _create_named_selection_by_ids(model, name, ids):
    # Create a MeshElements selection and assign to a Named Selection.
    sel = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshElements)

    # In some Mechanical/IronPython builds, direct assignment to `sel.Ids` can fail
    # or produce opaque internal errors. Prefer mutating the underlying collection.
    try:
        if hasattr(sel, "Ids") and hasattr(sel.Ids, "Clear") and hasattr(sel.Ids, "Add"):
            try:
                sel.Ids.Clear()
            except Exception:
                pass
            for _eid in ids:
                try:
                    sel.Ids.Add(int(_eid))
                except Exception:
                    pass
        else:
            sel.Ids = list(ids)
    except Exception:
        # Last-resort fallback.
        try:
            sel.Ids = list(ids)
        except Exception:
            pass
    def _with_transaction(fn):
        t = globals().get("Transaction", None)
        if t is None:
            return fn()
        try:
            with t():
                return fn()
        except Exception as e:
            _log("WARN: Transaction wrapper failed:", e)
            return fn()

    # Defensive: some builds can throw opaque internal errors (e.g. ObjectState)
    # when creating or scoping Named Selections. Prefer logging + skipping over hard crash.
    try:
        ns = _with_transaction(lambda: model.AddNamedSelection())
    except Exception as e:
        _log("ОШИБКА: AddNamedSelection не удалось для", name, ":", e)
        return None

    if ns is None:
        _log("ОШИБКА: AddNamedSelection вернул None для", name)
        return None

    try:
        ns.Name = name
    except Exception as e:
        _log("WARN: failed to set NamedSelection.Name for", name, ":", e)

    try:
        ns.Location = sel
    except Exception as e:
        _log("ОШИБКА: не удалось установить NamedSelection.Location для", name, ":", e)
        return None

    return ns


def _create_coordinate_system(model, name, origin_xyz, x_axis_xyz, z_axis_xyz=(0.0, 0.0, 1.0)):
    # Create a coordinate system with specified axes.
    x = _unit(x_axis_xyz)
    z = _unit(z_axis_xyz)
    y = _unit(_cross(z, x))
    x = _unit(_cross(y, z))  # re-orthogonalize

    cs = model.CoordinateSystems.AddCoordinateSystem()
    cs.Name = name
    _log("DEBUG: created CS", name, "settable_props=", _get_settable_props(cs)[:40])
    # Mechanical often expects Quantity for coordinates (not plain float).
    if "Quantity" in globals():
        cs.OriginX = Quantity(float(origin_xyz[0]), "mm")
        cs.OriginY = Quantity(float(origin_xyz[1]), "mm")
        cs.OriginZ = Quantity(float(origin_xyz[2]), "mm")
    else:
        cs.OriginX = float(origin_xyz[0])
        cs.OriginY = float(origin_xyz[1])
        cs.OriginZ = float(origin_xyz[2])

    if bool(_IS_IRONPYTHON):
        _log("INFO: IronPython detected; skipping CS axis orientation (not supported in this build).")
        return cs
    def _enum_pick(prop_name, preferred):
        try:
            if "System" not in globals() or System is None:
                return None
            t = cs.GetType().GetProperty(prop_name).PropertyType
            if not t.IsEnum:
                return None
            names = list(System.Enum.GetNames(t))
            for want in preferred:
                wl = str(want).lower()
                for n in names:
                    nl = str(n).lower()
                    if nl == wl or wl in nl:
                        return System.Enum.Parse(t, n)
        except Exception as e:
            _log("WARN: enum pick failed for", prop_name, ":", e)
        return None

    def _make_v3(vec):
        if "System" not in globals() or System is None:
            return None
        try:
            from Ansys.ACT.Math import Vector3D

            return Vector3D(float(vec[0]), float(vec[1]), float(vec[2]))
        except Exception:
            pass
        try:
            Vector3D = System.Type.GetType("Ansys.ACT.Math.Vector3D, Ansys.ACT.Common")
            if Vector3D is None:
                Vector3D = System.Type.GetType("Ansys.ACT.Math.Vector3D")
            if Vector3D is not None:
                return System.Activator.CreateInstance(Vector3D, float(vec[0]), float(vec[1]), float(vec[2]))
        except Exception as e:
            _log("WARN: cannot create Vector3D:", e)
        return None

    def _try_set_primary_secondary():
        settable = [p[0] for p in _get_settable_props(cs)]
        if "PrimaryAxisDirection" not in settable or "SecondaryAxisDirection" not in settable:
            return False
        vx = _make_v3(x)
        vy = _make_v3(y)
        if vx is None or vy is None:
            return False
        try:
            v = _enum_pick("PrimaryAxisDefineBy", ["direction", "vector", "directionvector"])
            if v is not None:
                cs.PrimaryAxisDefineBy = v
            v = _enum_pick("SecondaryAxisDefineBy", ["direction", "vector", "directionvector"])
            if v is not None:
                cs.SecondaryAxisDefineBy = v
            v = _enum_pick("PrimaryAxis", ["x"])
            if v is not None:
                cs.PrimaryAxis = v
            v = _enum_pick("SecondaryAxis", ["y"])
            if v is not None:
                cs.SecondaryAxis = v

            try:
                cs.PrimaryAxisDirection = vx
            except System.Exception as e:
                _log("WARN: failed to set PrimaryAxisDirection (System.Exception):", e)
            except Exception as e:
                _log("WARN: failed to set PrimaryAxisDirection:", e)
            except BaseException as e:
                _log("WARN: failed to set PrimaryAxisDirection:", e)
            except:
                _log("WARN: failed to set PrimaryAxisDirection: unknown error")
            try:
                cs.SecondaryAxisDirection = vy
            except System.Exception as e:
                _log("WARN: failed to set SecondaryAxisDirection (System.Exception):", e)
            except Exception as e:
                _log("WARN: failed to set SecondaryAxisDirection:", e)
            except BaseException as e:
                _log("WARN: failed to set SecondaryAxisDirection:", e)
            except:
                _log("WARN: failed to set SecondaryAxisDirection: unknown error")
            return True
        except System.Exception as e:
            _log("WARN: failed to set Primary/Secondary axis directions (System.Exception):", e)
        except Exception as e:
            _log("WARN: failed to set Primary/Secondary axis directions:", e)
        except BaseException as e:
            _log("WARN: failed to set Primary/Secondary axis directions:", e)
        return False

    ok = _try_set_primary_secondary()
    if not ok:
        _log("WARN: could not set CS orientation for", name, "- leaving default orientation.")
    return cs


def _group_layers(layers, group_size):
    if group_size is None:
        group_size = 1
    g = int(group_size)
    if g <= 1:
        # one group per layer_id
        out = {}
        for l in layers:
            gid = int(l.get("layer_id", 0))
            out.setdefault(gid, []).append(l)
        return out
    out = {}
    for l in layers:
        lid = int(l.get("layer_id", 0))
        gid = lid // g
        out.setdefault(gid, []).append(l)
    return out


def _pick_group_direction(group_layers):
    # Choose a representative direction for a group. Prefer higher confidence, then higher total_weight.
    best = None
    best_key = None
    for l in group_layers:
        d = l.get("dir_xyz", None)
        if d is None:
            continue
        conf = float(l.get("confidence", 0.0))
        if conf < float(MIN_CONFIDENCE):
            continue
        w = float(l.get("total_weight", 0.0))
        key = (conf, w)
        if best_key is None or key > best_key:
            best_key = key
            best = l
    return best


def main():
    if "ExtAPI" not in globals():
        print("ОШИБКА: этот скрипт нужно запускать внутри ANSYS Mechanical (Workbench).")
        return

    def _load_layers():
        # Prefer JSON when available (CPython). Fall back to CSV for IronPython builds where json is incomplete.
        if bool(_HAVE_JSON) and os.path.exists(LAYERS_JSON):
            try:
                f = open(LAYERS_JSON, "r")
                try:
                    data = json.load(f)
                finally:
                    f.close()
                layers = data.get("layers", [])
                if layers:
                    return layers
            except Exception as e:
                print("WARN: failed to read ansys_layers.json:", e)

        if not os.path.exists(LAYERS_CSV):
            print("ОШИБКА: файл ansys_layers.csv не найден, а JSON загрузить не удалось.")
            return []

        txt = _read_text(LAYERS_CSV)
        lines = [x.strip() for x in txt.splitlines() if x.strip()]
        if len(lines) < 2:
            return []
        header = [h.strip() for h in lines[0].split(",")]
        idx = {}
        for i, h in enumerate(header):
            idx[h] = i

        def _get(parts, name, default=None):
            if name not in idx:
                return default
            i = idx[name]
            if i < 0 or i >= len(parts):
                return default
            return parts[i]

        out = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            try:
                layer_id = int(_get(parts, "layer_id", 0) or 0)
                z_min = float(_get(parts, "z_min_mm", 0.0) or 0.0)
                z_max = float(_get(parts, "z_max_mm", 0.0) or 0.0)
                dx = float(_get(parts, "dx", 0.0) or 0.0)
                dy = float(_get(parts, "dy", 0.0) or 0.0)
                dz = float(_get(parts, "dz", 0.0) or 0.0)
                conf = float(_get(parts, "confidence", 0.0) or 0.0)
                seg = int(float(_get(parts, "segments_used", 0.0) or 0.0))
                tw = float(_get(parts, "total_weight", 0.0) or 0.0)
            except Exception:
                continue
            out.append(
                {
                    "layer_id": layer_id,
                    "z_min": z_min,
                    "z_max": z_max,
                    "z_center": 0.5 * (z_min + z_max),
                    "dir_xyz": [dx, dy, dz],
                    "angle_deg": None,
                    "confidence": conf,
                    "segments_used": seg,
                    "total_weight": tw,
                }
            )
        return out

    layers = _load_layers()
    if not layers:
        print("No layers found in ansys_layers.json/csv")
        return

    model = ExtAPI.DataModel.Project.Model
    mesh = _safe_get_mesh(model)

    # Collect element centroids once.
    print("Collecting element centroids...")
    elem_z = {}
    elem_xyz = {}
    ids = list(_iter_element_ids(mesh))
    x0 = y0 = z0 = 1e300
    x1 = y1 = z1 = -1e300
    for i, eid in enumerate(ids):
        c = _element_centroid(mesh, eid)
        elem_xyz[eid] = c
        try:
            x0 = min(x0, float(c[0]))
            y0 = min(y0, float(c[1]))
            z0 = min(z0, float(c[2]))
            x1 = max(x1, float(c[0]))
            y1 = max(y1, float(c[1]))
            z1 = max(z1, float(c[2]))
        except Exception:
            pass
        if (i + 1) % 50000 == 0:
            print("  processed", i + 1, "elements")

    print("Elements:", len(ids))

    try:
        layers_z_min = min(float(l.get("z_min", 0.0)) for l in layers)
        layers_z_max = max(float(l.get("z_max", 0.0)) for l in layers)
    except Exception:
        layers_z_min = 0.0
        layers_z_max = 0.0

    # Scale factor from layer-units -> element-units.
    # If layer Z-range is ~1000x larger, element coords are likely meters while layers are mm.
    layer_to_elem = 1.0
    try:
        elem_extent = max(float(x1 - x0), float(y1 - y0), float(z1 - z0))
    except Exception:
        elem_extent = 0.0
    layers_extent = max(0.0, float(layers_z_max) - float(layers_z_min))
    if elem_extent > 1e-12 and layers_extent > 1e-12:
        ratio = float(layers_extent) / float(elem_extent)
        if ratio > 200.0 and ratio < 5000.0:
            layer_to_elem = 1.0 / ratio
            _log(
                "INFO: detected unit mismatch; scaling layer Z by",
                layer_to_elem,
                "(layer-units -> element-units).",
                "elem_extent=",
                elem_extent,
                "layers_extent=",
                layers_extent,
            )
        else:
            _log("INFO: unit check ok. elem_extent=", elem_extent, "layers_extent=", layers_extent)
    else:
        _log("WARN: cannot detect units (zero range). elem_extent=", elem_extent, "layers_extent=", layers_extent)

    # Apply unit scaling for STL->CMB translation.
    global LAYER_TO_ELEM
    LAYER_TO_ELEM = float(layer_to_elem)

    # Now compute element Z in build space.
    for eid in ids:
        elem_z[eid] = _build_coord(elem_xyz[eid])

    # Pre-sort elements by Z to avoid O(layers * elements) scanning.
    print("Indexing elements by Z...")
    z_eid = sorted((elem_z[eid], eid) for eid in ids)
    zs = [ze[0] for ze in z_eid]
    eids_sorted = [ze[1] for ze in z_eid]

    # Create outputs per layer/group.
    created_ns = 0
    created_cs = 0

    groups = _group_layers(layers, GROUP_SIZE_LAYERS)
    group_keys = sorted(groups.keys())
    print("Groups:", len(group_keys), "(group_size_layers={})".format(GROUP_SIZE_LAYERS))

    for gid in group_keys:
        glayers = groups[gid]
        lids = [int(x.get("layer_id", 0)) for x in glayers]
        lid0 = min(lids) if lids else int(gid)
        lid1 = max(lids) if lids else int(gid)
        z_min = min(float(x.get("z_min", 0.0)) for x in glayers) * float(layer_to_elem)
        z_max = max(float(x.get("z_max", 0.0)) for x in glayers) * float(layer_to_elem)

        # Find elements in Z band using binary search.
        i0 = bisect.bisect_left(zs, z_min)
        i1 = bisect.bisect_left(zs, z_max)
        band_eids = eids_sorted[i0:i1]
        if not band_eids:
            continue

        name = "L_{:04d}".format(lid0) if int(GROUP_SIZE_LAYERS) <= 1 else "L_{:04d}_{:04d}".format(lid0, lid1)

        if bool(CREATE_NAMED_SELECTIONS):
            ns = _create_named_selection_by_ids(model, name, band_eids)
            if ns is not None:
                created_ns += 1

        if bool(CREATE_COORDINATE_SYSTEMS):
            try:
                pick = _pick_group_direction(glayers)
                if pick is None:
                    # no reliable direction; still keep Named Selection if requested
                    continue
                d = pick.get("dir_xyz", None)
                if d is None:
                    continue

                cx = sum(elem_xyz[eid][0] for eid in band_eids) / float(len(band_eids))
                cy = sum(elem_xyz[eid][1] for eid in band_eids) / float(len(band_eids))
                cz = sum(elem_xyz[eid][2] for eid in band_eids) / float(len(band_eids))
                # Convert origin from element-units back to layer-units for Quantity (usually mm).
                inv = 1.0 / float(layer_to_elem) if float(layer_to_elem) > 1e-18 else 1.0
                _create_coordinate_system(
                    model,
                    "CS_" + name,
                    (cx * inv, cy * inv, cz * inv),
                    _unit(_cmb_vec_to_model((float(d[0]), float(d[1]), float(d[2])))),
                    z_axis_xyz=_build_axis_vec_xyz(),
                )
                created_cs += 1
            except System.Exception as e:
                _log("WARN: failed to create coordinate system for", name, "(System.Exception):", e)
            except Exception as e:
                _log("WARN: failed to create coordinate system for", name, ":", e)
            except BaseException as e:
                _log("WARN: failed to create coordinate system for", name, ":", e)
            except:
                _log("WARN: failed to create coordinate system for", name, ": unknown error")

    print("Created Named Selections:", created_ns)
    print("Created Coordinate Systems:", created_cs)
    print("")
    print("Next step in Mechanical:")
    print(" - Assign orthotropic material to the body")
    print(" - Use the created coordinate systems (CS_L_XXXX) as material orientation references")


main()
'''


_ANSYS_MECHANICAL_SECTION_PLANES_TEMPLATE = r"""# -*- coding: utf-8 -*-
# slice2solid: Mechanical helper (Section Plane aligned to Insight/CMB)
#
# What it does:
# - Creates (or reuses) a single active Section Plane named 'S2S_Slice'
# - Moves it to a requested build height Z_MM (mm) in Insight/CMB coordinates (build axis = Z+)
# - Optional: exports one PNG per layer for quick visual comparison with the slicer preview
#
# Why STL->CMB matters:
# Insight exports toolpaths in CMB coordinates, but the STL/CAD you import into Mechanical is usually in "original" STL coords.
# This script applies the inverse of STL->CMB to position the Section Plane correctly over the imported geometry.
#
# How to use (Mechanical):
# - Automation -> Scripting -> Open Script... -> select this file -> Run
# - Edit Z_MM / EXPORT_IMAGES below and re-run as needed

import csv
import math
import os

from Ansys.Mechanical.Graphics import GraphicsImageExportFormat, GraphicsImageExportSettings
from Ansys.Mechanical.Graphics import Point, SectionPlane, SectionPlaneType, Vector3D


HERE = os.path.dirname(__file__)
LAYERS_CSV = os.path.join(HERE, "ansys_layers.csv")

# User knobs
Z_MM = {z_mm:.6f}  # build height in CMB (mm)
EXPORT_IMAGES = {export_images}  # True to export a PNG per layer
EXPORT_DIR = os.path.join(HERE, "s2s_slices_png")
PLANE_NAME = "S2S_Slice"

# Geometry placement: apply inverse(STL->CMB) to map CMB plane positions onto imported STL/CAD coordinates.
APPLY_STL_TO_CMB = {apply_stl_to_cmb}
STL_TO_CMB = {stl_to_cmb_json}


def _mat3_inv(a):
    a00, a01, a02 = float(a[0][0]), float(a[0][1]), float(a[0][2])
    a10, a11, a12 = float(a[1][0]), float(a[1][1]), float(a[1][2])
    a20, a21, a22 = float(a[2][0]), float(a[2][1]), float(a[2][2])
    b00 = a11 * a22 - a12 * a21
    b01 = a02 * a21 - a01 * a22
    b02 = a01 * a12 - a02 * a11
    b10 = a12 * a20 - a10 * a22
    b11 = a00 * a22 - a02 * a20
    b12 = a02 * a10 - a00 * a12
    b20 = a10 * a21 - a11 * a20
    b21 = a01 * a20 - a00 * a21
    b22 = a00 * a11 - a01 * a10
    det = a00 * b00 + a01 * b10 + a02 * b20
    if abs(det) <= 1e-18:
        return None
    inv_det = 1.0 / det
    return [
        [b00 * inv_det, b01 * inv_det, b02 * inv_det],
        [b10 * inv_det, b11 * inv_det, b12 * inv_det],
        [b20 * inv_det, b21 * inv_det, b22 * inv_det],
    ]


_LIN_INV = None


def _lin_inv():
    global _LIN_INV
    if _LIN_INV is not None:
        return _LIN_INV
    if not STL_TO_CMB:
        _LIN_INV = None
        return None
    try:
        lin = [
            [float(STL_TO_CMB[0][0]), float(STL_TO_CMB[0][1]), float(STL_TO_CMB[0][2])],
            [float(STL_TO_CMB[1][0]), float(STL_TO_CMB[1][1]), float(STL_TO_CMB[1][2])],
            [float(STL_TO_CMB[2][0]), float(STL_TO_CMB[2][1]), float(STL_TO_CMB[2][2])],
        ]
        _LIN_INV = _mat3_inv(lin)
        return _LIN_INV
    except Exception:
        _LIN_INV = None
        return None


def _cmb_to_model_point(xyz):
    if not (bool(APPLY_STL_TO_CMB) and STL_TO_CMB):
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    inv = _lin_inv()
    if inv is None:
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    tx = float(STL_TO_CMB[3][0])
    ty = float(STL_TO_CMB[3][1])
    tz = float(STL_TO_CMB[3][2])
    x = float(xyz[0]) - tx
    y = float(xyz[1]) - ty
    z = float(xyz[2]) - tz
    # out = (xyz - t) @ inv  (row-vector convention)
    return (
        x * float(inv[0][0]) + y * float(inv[1][0]) + z * float(inv[2][0]),
        x * float(inv[0][1]) + y * float(inv[1][1]) + z * float(inv[2][1]),
        x * float(inv[0][2]) + y * float(inv[1][2]) + z * float(inv[2][2]),
    )


def _cmb_to_model_vec(v):
    inv = _lin_inv() if (bool(APPLY_STL_TO_CMB) and STL_TO_CMB) else None
    if inv is None:
        return (float(v[0]), float(v[1]), float(v[2]))
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return (
        x * float(inv[0][0]) + y * float(inv[1][0]) + z * float(inv[2][0]),
        x * float(inv[0][1]) + y * float(inv[1][1]) + z * float(inv[2][1]),
        x * float(inv[0][2]) + y * float(inv[1][2]) + z * float(inv[2][2]),
    )


def _unit(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _load_layers_csv(path):
    layers = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lid = int(row["layer_id"])
            zmin = float(row["z_min_mm"])
            zmax = float(row["z_max_mm"])
            layers.append((lid, zmin, zmax))
    return layers


def _get_or_create_plane(name):
    # CMB build axis is always Z+.
    d_model = _unit(_cmb_to_model_vec((0.0, 0.0, 1.0)))

    for p in Graphics.SectionPlanes:
        try:
            if p.Name == name:
                p.Active = True
                return p
        except Exception:
            pass

    sp = SectionPlane()
    sp.Name = name
    sp.Active = True
    sp.Type = SectionPlaneType.AgainstDirection
    sp.Direction = Vector3D(float(d_model[0]), float(d_model[1]), float(d_model[2]))
    sp.Center = Point([0.0, 0.0, 0.0], "mm")
    Graphics.SectionPlanes.Add(sp)
    return sp


def _set_plane_z_cmb(plane, z_mm):
    # CMB build coordinate: (0,0,z_mm)
    p_model = _cmb_to_model_point((0.0, 0.0, float(z_mm)))
    plane.Center = Point([float(p_model[0]), float(p_model[1]), float(p_model[2])], "mm")


def _export_png(path):
    settings = GraphicsImageExportSettings()
    settings.Width = 1600
    settings.Height = 900
    settings.CurrentGraphicsDisplay = True
    Graphics.ExportImage(path, GraphicsImageExportFormat.PNG, settings)


plane = _get_or_create_plane(PLANE_NAME)
_set_plane_z_cmb(plane, Z_MM)

if EXPORT_IMAGES:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    layers = _load_layers_csv(LAYERS_CSV)
    for lid, zmin, zmax in layers:
        zmid = 0.5 * (zmin + zmax)
        _set_plane_z_cmb(plane, zmid)
        out = os.path.join(EXPORT_DIR, "slice_L_%04d_z%.3fmm.png" % (lid, zmid))
        _export_png(out)

print("S2S: Section plane '%s' set to CMB Z=%.3f mm (APPLY_STL_TO_CMB=%s)" % (PLANE_NAME, Z_MM, APPLY_STL_TO_CMB))  # noqa: T201
"""


def _render_ansys_mechanical_script(
    cfg: JobConfig,
    *,
    stl_to_cmb: np.ndarray,
    apply_stl_to_cmb: bool,
) -> str:
    # Embed matrix so the script can work even in IronPython builds without json.
    m = np.asarray(stl_to_cmb, dtype=float)
    if m.shape != (4, 4):
        m = np.eye(4, dtype=float)
    m_list = [[float(m[r, c]) for c in range(4)] for r in range(4)]
    header = (
        "# -*- coding: utf-8 -*-\n"
        "# Generated by slice2solid\n"
        "BUILD_AXIS = \"z\"\n"
        "BUILD_SIGN = 1\n"
        f"APPLY_STL_TO_CMB = {str(bool(apply_stl_to_cmb))}\n"
        f"STL_TO_CMB = {json.dumps(m_list, ensure_ascii=False)}\n"
        f"MIN_CONFIDENCE = {float(cfg.ansys_min_confidence):.6g}\n"
        f"GROUP_SIZE_LAYERS = {int(cfg.ansys_group_size_layers)}\n"
        f"CREATE_NAMED_SELECTIONS = {str(bool(cfg.ansys_create_named_selections))}\n"
        f"CREATE_COORDINATE_SYSTEMS = {str(bool(cfg.ansys_create_coordinate_systems))}\n"
        "\n"
    )
    return header + _ANSYS_MECHANICAL_SCRIPT_TEMPLATE


def _render_ansys_mechanical_section_planes_script(
    cfg: JobConfig,
    *,
    stl_to_cmb: np.ndarray,
    apply_stl_to_cmb: bool,
) -> str:
    # Default: Z=0 mm (user can edit Z_MM in the script).
    z_mm = 0.0
    m = np.asarray(stl_to_cmb, dtype=float)
    if m.shape != (4, 4):
        m = np.eye(4, dtype=float)
    m_list = [[float(m[r, c]) for c in range(4)] for r in range(4)]
    return _ANSYS_MECHANICAL_SECTION_PLANES_TEMPLATE.format(
        z_mm=z_mm,
        export_images="False",
        apply_stl_to_cmb=str(bool(apply_stl_to_cmb)),
        stl_to_cmb_json=json.dumps(m_list, ensure_ascii=False),
    )


def _safe_filename_stem(name: str) -> str:
    # Keep it Windows-friendly and readable.
    name = name.strip()
    if not name:
        return "part"
    name = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "part"


def _format_token(x: float, *, decimals: int = 3) -> str:
    s = f"{float(x):.{int(decimals)}f}"
    s = s.rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _preview_mesh_stem(cfg: JobConfig) -> str:
    part = ""
    try:
        if cfg.placed_stl:
            part = _safe_filename_stem(Path(cfg.placed_stl).stem)
    except Exception:
        part = ""
    if not part:
        try:
            if cfg.job_dir:
                part = _safe_filename_stem(Path(cfg.job_dir).name)
        except Exception:
            part = ""
    if not part:
        part = "part"
    vox = _format_token(cfg.voxel_size_mm, decimals=3)
    sig = _format_token(cfg.volume_smooth_sigma_vox, decimals=3)
    ds = int(getattr(cfg, "meshing_downsample_factor", 1) or 1)
    it = int(cfg.smooth_iterations)
    ds_part = f"_ds{ds}" if ds > 1 else ""
    return f"{part}_vox{vox}{ds_part}_sig{sig}_it{it}_s2s_preview_structure"


def _render_cad_import_notes(cfg: JobConfig) -> str:
    v = float(cfg.voxel_size_mm)
    suggested = max(0.5 * v, 0.02)
    ds = int(getattr(cfg, "meshing_downsample_factor", 1) or 1)
    v_mesh = v * float(ds)
    suggested_mesh = max(0.5 * v_mesh, 0.02)
    preview_stem = _preview_mesh_stem(cfg)
    return (
        "slice2solid: заметки для импорта/конвертации в CAD\n"
        "\n"
        "Важно про единицы:\n"
        " - STL/PLY не хранят единицы. При импорте выбирайте миллиметры (mm).\n"
        " - Если CAD импортировал как inches, масштаб будет ~25.4x.\n"
        "\n"
        "Файлы в этой папке (CAD/):\n"
        f" - {preview_stem}.stl            (mesh; основной файл)\n"
        f" - {preview_stem}_mesh.ply       (mesh; альтернативный импорт)\n"
        " - voxel_points.csv              (point cloud из занятых вокселей; x,y,z в мм; без заголовка; может быть sampled)\n"
        "\n"
        "Параметры расчёта:\n"
        " - ../metadata.json              (в корне папки результата)\n"
        "\n"
        "Быстрый старт (типовой путь в CAD/mesh-инструменте):\n"
        f" 1) Импортируйте {preview_stem}.stl (или .ply) в единицах mm\n"
        " 2) Если импорт/конвертация ругается: Repair/Close Holes/Remove self-intersections/Orient Normals\n"
        " 3) Если нужно тело (solid/B-Rep): конвертируйте mesh/implicit -> solid\n"
        " 4) Экспортируйте STEP/Parasolid (или другой CAD-формат)\n"
        "\n"
        "Альтернатива (иногда лучше для решёток/заполнений и «сложных» сеток):\n"
        " - Импорт point cloud (voxel_points.csv) -> построение implicit/volume -> solidify -> экспорт STEP\n"
        "\n"
        "Подсказка по шагу/разрешению (если инструмент просит spacing/resolution):\n"
        f" - slice2solid voxel_size_mm = {v:.3f} мм\n"
        f" - mesh effective voxel (после downsample при мешинге): {v_mesh:.3f} мм (ds={ds})\n"
        f" - стартовый spacing (по voxels/points): ~{suggested:.3f} мм (~ 0.5 * voxel_size)\n"
        f" - стартовый spacing (по mesh): ~{suggested_mesh:.3f} мм (~ 0.5 * mesh effective voxel)\n"
        "   Если слишком медленно: увеличьте spacing. Если теряются детали: уменьшите spacing.\n"
    )


def _render_cae_import_notes(cfg: JobConfig) -> str:
    preview_stem = _preview_mesh_stem(cfg)
    return (
        "slice2solid: заметки для CAE (ANSYS Mechanical)\n"
        "\n"
        "Что лежит в этой папке (CAE/):\n"
        " - ansys_layers.json / ansys_layers.csv  (ориентация печати по слоям + Z-границы слоёв)\n"
        " - ansys_mechanical_import_layers.py    (скрипт для Mechanical: Named Selections/Coordinate Systems по слоям)\n"
        " - ansys_mechanical_section_planes.py   (скрипт для Mechanical: сечения по Z и визуальная проверка)\n"
        "\n"
        "Типовой рабочий процесс в ANSYS Mechanical:\n"
        " 1) Импортируйте геометрию детали (CAD-solid или обычный STL) и убедитесь, что единицы mm.\n"
        " 2) Постройте сетку (Mesh) обычным способом.\n"
        " 3) Automation -> Scripting -> Run Script -> запустите ansys_mechanical_import_layers.py.\n"
        " 4) (Опционально) Для визуальной проверки: запустите ansys_mechanical_section_planes.py и меняйте Z_MM.\n"
        "\n"
        "Важно про координаты (CMB vs исходный STL):\n"
        " - Скрипты содержат матрицу STL->CMB из прогона slice2solid и по умолчанию APPLY_STL_TO_CMB=True.\n"
        " - Если вы импортировали модель уже в координатах CMB, выставьте APPLY_STL_TO_CMB=False внутри скрипта.\n"
        "\n"
        "Производительность:\n"
        " - Если слоёв много, создание Named Selection на каждый слой может быть тяжёлым.\n"
        " - Увеличьте GROUP_SIZE_LAYERS или выключите CREATE_NAMED_SELECTIONS/CREATE_COORDINATE_SYSTEMS в скрипте.\n"
        "\n"
        "Геометрия инфилла из slice2solid (если включали предпросмотр):\n"
        f" - CAD/{preview_stem}.stl — обычно НЕ нужна для CAE, если цель — учёт направления печати по слоям.\n"
        " - Обычно для CAE используют «номинальную» геометрию детали, а слойные данные берут из CAE/.\n"
    )


def _render_wireframe_preview(
    mesh: trimesh.Trimesh,
    *,
    width: int,
    height: int,
    max_edges: int = 200_000,
    fg: QtGui.QColor,
    bg: QtGui.QColor,
) -> QtGui.QImage:
    img = QtGui.QImage(max(1, int(width)), max(1, int(height)), QtGui.QImage.Format.Format_ARGB32)
    img.fill(bg)

    if mesh.faces is None or mesh.vertices is None:
        return img
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return img

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    forward = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-12)
    up2 = np.cross(forward, right)
    up2 = up2 / (np.linalg.norm(up2) + 1e-12)

    basis = np.stack([right, up2], axis=1)  # (3,2)
    proj = verts @ basis  # (N,2)

    pmin = proj.min(axis=0)
    pmax = proj.max(axis=0)
    span = np.maximum(pmax - pmin, 1e-6)

    pad = 12.0
    sx = (float(width) - 2.0 * pad) / float(span[0])
    sy = (float(height) - 2.0 * pad) / float(span[1])
    scale = float(min(sx, sy))
    xy = (proj - pmin[None, :]) * scale + pad
    xy[:, 1] = float(height) - xy[:, 1]

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    try:
        edges = np.unique(edges, axis=0)
    except Exception:
        pass
    if max_edges > 0 and edges.shape[0] > int(max_edges):
        idx = np.linspace(0, edges.shape[0] - 1, num=int(max_edges), dtype=int)
        edges = edges[idx]

    painter = QtGui.QPainter(img)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    pen = QtGui.QPen(fg)
    pen.setWidthF(1.0)
    painter.setPen(pen)

    path = QtGui.QPainterPath()
    pts = xy[edges.reshape(-1)].reshape((-1, 2, 2))
    for a, b in pts:
        path.moveTo(float(a[0]), float(a[1]))
        path.lineTo(float(b[0]), float(b[1]))
    painter.drawPath(path)
    painter.end()
    return img


class _Mesh2DView(QtWidgets.QWidget):
    def __init__(self, *, title: str):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._title = QtWidgets.QLabel(title)
        self._title.setStyleSheet("font-weight: 600;")
        self._stats = QtWidgets.QLabel("")
        self._stats.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.addWidget(self._title, 0)
        header_layout.addStretch(1)
        header_layout.addWidget(self._stats, 0)
        layout.addWidget(header, 0)

        self._image = QtWidgets.QLabel("")
        self._image.setMinimumHeight(200)
        self._image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._image.setStyleSheet("background-color: #F3F4F6; border: 1px solid #D1D5DB;")
        layout.addWidget(self._image, 1)

    def set_mesh(self, mesh: trimesh.Trimesh | None, *, stats_text: str = "", color: str = "#6BCB77") -> None:
        self._stats.setText(stats_text)
        if mesh is None or mesh.faces is None or mesh.vertices is None or len(mesh.faces) == 0:
            self._image.setText("Нет сетки для отображения")
            self._image.setPixmap(QtGui.QPixmap())
            return

        w = max(320, int(self._image.width()))
        h = max(240, int(self._image.height()))
        img = _render_wireframe_preview(
            mesh,
            width=w,
            height=h,
            fg=QtGui.QColor(color),
            bg=QtGui.QColor("#F3F4F6"),
        )
        self._image.setText("")
        self._image.setPixmap(QtGui.QPixmap.fromImage(img))


class _Mesh3DView(QtWidgets.QWidget):
    def __init__(self, *, title: str):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._title = QtWidgets.QLabel(title)
        self._title.setStyleSheet("font-weight: 600;")
        self._stats = QtWidgets.QLabel("")
        self._stats.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.addWidget(self._title, 0)
        header_layout.addStretch(1)
        header_layout.addWidget(self._stats, 0)
        layout.addWidget(header, 0)

        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 0, 8, 6)
        self._faces_cb = QtWidgets.QCheckBox("Поверхность")
        self._faces_cb.setChecked(True)
        self._edges_cb = QtWidgets.QCheckBox("Рёбра")
        self._edges_cb.setChecked(True)
        self._light_bg_cb = QtWidgets.QCheckBox("Светлый фон")
        self._light_bg_cb.setChecked(True)
        self._auto_fit_btn = QtWidgets.QPushButton("Вписать")
        self._auto_fit_btn.setToolTip("Подогнать камеру под модель")
        self._hint = QtWidgets.QLabel("ЛКМ: вращение · колесо: зум · ПКМ: панорама")
        self._hint.setStyleSheet("color: #6B7280;")
        toolbar_layout.addWidget(self._faces_cb, 0)
        toolbar_layout.addWidget(self._edges_cb, 0)
        toolbar_layout.addWidget(self._light_bg_cb, 0)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self._auto_fit_btn, 0)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self._hint, 0)
        layout.addWidget(toolbar, 0)

        self._gl = gl.GLViewWidget()
        self._gl.setBackgroundColor("#F3F4F6")
        layout.addWidget(self._gl, 1)
        self._mesh_item: object | None = None
        self._edges_item: object | None = None
        self._edges_seg: np.ndarray | None = None
        self._slice_item: object | None = None
        self._toolpath_item_range: object | None = None
        self._toolpath_item_layer: object | None = None
        self._mesh_for_slice: trimesh.Trimesh | None = None
        self._zmin: float | None = None
        self._zmax: float | None = None
        self._z_phys_min: float | None = None
        self._z_phys_max: float | None = None
        # Display offset: place "build plate" at Z=0 and center XY (slicer-like).
        # The slider/labels stay in original CMB coordinates; views convert to display coords internally.
        self._disp_offset = np.zeros((3,), dtype=float)
        # CMB-only: build axis is always Z+.
        self._build_min: float | None = None
        self._build_max: float | None = None
        self._slice_enabled: bool = False
        self._slice_t: float = 0.5
        self._radius: float = 1.0
        self._edge_rgba = (0.0, 0.0, 0.0, 0.18)
        self._slice_rgba = (0.93, 0.11, 0.14, 0.95)

        self._axis = gl.GLAxisItem()
        self._axis.setSize(10, 10, 10)
        self._gl.addItem(self._axis)

        self._grid = gl.GLGridItem()
        self._grid.setSize(50, 50)
        self._grid.setSpacing(10, 10)
        self._grid.translate(0, 0, 0)
        self._gl.addItem(self._grid)

        self._faces_cb.toggled.connect(self._apply_visibility)
        self._edges_cb.toggled.connect(self._apply_visibility)
        self._light_bg_cb.toggled.connect(self._apply_theme)
        self._auto_fit_btn.clicked.connect(self._fit_camera)
        self._apply_theme()

    def _apply_theme(self) -> None:
        light = bool(self._light_bg_cb.isChecked())
        bg = "#F3F4F6" if light else "#111317"
        grid = QtGui.QColor("#D1D5DB" if light else "#2B2F36")
        self._edge_rgba = (0.0, 0.0, 0.0, 0.18) if light else (1.0, 1.0, 1.0, 0.22)
        try:
            self._gl.setBackgroundColor(bg)
        except Exception:
            pass
        try:
            self._grid.setColor(grid)
        except Exception:
            pass
        try:
            self._hint.setStyleSheet("color: #374151;" if light else "color: #9CA3AF;")
        except Exception:
            pass
        if self._edges_item is not None and self._edges_seg is not None:
            try:
                self._edges_item.setData(pos=self._edges_seg, color=self._edge_rgba)
            except Exception:
                pass

    def _fit_camera(self) -> None:
        r = float(self._radius or 1.0)
        try:
            self._gl.setCameraPosition(distance=max(10.0, 2.6 * r), elevation=25, azimuth=-45)
        except Exception:
            self._gl.opts["distance"] = max(10.0, 2.6 * r)
            self._gl.opts["elevation"] = 25
            self._gl.opts["azimuth"] = -45

    def _apply_visibility(self) -> None:
        if self._mesh_item is not None:
            try:
                self._mesh_item.setVisible(bool(self._faces_cb.isChecked()))
            except Exception:
                pass
        if self._edges_item is not None:
            try:
                self._edges_item.setVisible(bool(self._edges_cb.isChecked()))
            except Exception:
                pass
        if self._slice_item is not None:
            try:
                self._slice_item.setVisible(bool(self._slice_enabled))
            except Exception:
                pass

    def set_slice(self, *, enabled: bool, t: float) -> None:
        self._slice_enabled = bool(enabled)
        self._slice_t = float(max(0.0, min(1.0, float(t))))
        self._update_slice()

    def set_slice_z(self, *, enabled: bool, z_mm: float) -> None:
        self._slice_enabled = bool(enabled)
        if self._build_min is None or self._build_max is None:
            return
        z0 = float(self._build_min)
        z1 = float(self._build_max)
        if abs(z1 - z0) <= 1e-12:
            return
        t = (float(z_mm) - z0) / (z1 - z0)
        t = float(max(0.0, min(1.0, t)))
        self._slice_t = float(t)
        self._update_slice()

    def _update_slice(self) -> None:
        if not bool(self._slice_enabled):
            if self._slice_item is not None:
                try:
                    self._gl.removeItem(self._slice_item)
                except Exception:
                    pass
                self._slice_item = None
            return
        if self._mesh_for_slice is None or self._build_min is None or self._build_max is None:
            return
        z0 = float(self._build_min)
        z1 = float(self._build_max)
        t = float(self._slice_t)
        build_z = z0 + t * (z1 - z0)

        axis_i = 2
        sign = 1
        axis_coord = float(build_z)
        axis_coord_centered = axis_coord - float(self._disp_offset[axis_i])
        origin = [0.0, 0.0, 0.0]
        origin[axis_i] = float(axis_coord_centered)
        normal = [0.0, 0.0, 0.0]
        normal[axis_i] = float(sign)

        try:
            sec = self._mesh_for_slice.section(plane_origin=origin, plane_normal=normal)
        except Exception:
            sec = None
        if sec is None:
            if self._slice_item is not None:
                try:
                    self._gl.removeItem(self._slice_item)
                except Exception:
                    pass
                self._slice_item = None
            return

        try:
            polylines = sec.discrete
        except Exception:
            polylines = []
        if not polylines:
            if self._slice_item is not None:
                try:
                    self._gl.removeItem(self._slice_item)
                except Exception:
                    pass
                self._slice_item = None
            return

        segs: list[np.ndarray] = []
        seg_budget = 200_000
        seg_count = 0
        for pl in polylines:
            arr = np.asarray(pl, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 2:
                continue
            a = arr[:-1]
            b = arr[1:]
            s = np.empty((a.shape[0] * 2, 3), dtype=np.float32)
            s[0::2] = a
            s[1::2] = b
            segs.append(s)
            seg_count += int(a.shape[0])
            if seg_count >= seg_budget:
                break
        if not segs:
            return
        seg = np.vstack(segs)

        if self._slice_item is not None:
            try:
                self._gl.removeItem(self._slice_item)
            except Exception:
                pass
            self._slice_item = None
        try:
            item = gl.GLLinePlotItem(pos=seg, mode="lines", color=self._slice_rgba, width=2, antialias=True)
            item.setGLOptions("translucent")
            self._gl.addItem(item)
            self._slice_item = item
            self._apply_visibility()
        except Exception:
            self._slice_item = None

    @staticmethod
    def _unique_edges(faces: np.ndarray, *, max_edges: int) -> np.ndarray:
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64, copy=False)
        edges.sort(axis=1)
        try:
            edges = np.unique(edges, axis=0)
        except Exception:
            pass
        if max_edges > 0 and edges.shape[0] > int(max_edges):
            idx = np.linspace(0, edges.shape[0] - 1, num=int(max_edges), dtype=int)
            edges = edges[idx]
        return edges

    def set_mesh(self, mesh: trimesh.Trimesh | None, *, stats_text: str = "", color: str = "#6BCB77") -> None:
        self._stats.setText(stats_text)
        if self._mesh_item is not None:
            try:
                self._gl.removeItem(self._mesh_item)
            except Exception:
                pass
            self._mesh_item = None
        if self._edges_item is not None:
            try:
                self._gl.removeItem(self._edges_item)
            except Exception:
                pass
            self._edges_item = None
        if self._slice_item is not None:
            try:
                self._gl.removeItem(self._slice_item)
            except Exception:
                pass
            self._slice_item = None
        if self._toolpath_item_range is not None:
            try:
                self._gl.removeItem(self._toolpath_item_range)
            except Exception:
                pass
            self._toolpath_item_range = None
        if self._toolpath_item_layer is not None:
            try:
                self._gl.removeItem(self._toolpath_item_layer)
            except Exception:
                pass
            self._toolpath_item_layer = None
        self._mesh_for_slice = None
        self._zmin = None
        self._zmax = None
        self._build_min = None
        self._build_max = None

        if mesh is None or mesh.faces is None or mesh.vertices is None or len(mesh.faces) == 0:
            return

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if verts.size == 0 or faces.size == 0:
            return

        try:
            bounds = np.asarray(mesh.bounds, dtype=np.float32)
            self._z_phys_min = float(bounds[0, 2])
            self._z_phys_max = float(bounds[1, 2])
            # Slicer-like placement:
            # - build plate is Z=0 (so the part is above XY plane)
            # - XY is centered for nicer navigation
            disp = bounds.mean(axis=0).astype(float)
            disp[2] = float(bounds[0, 2])
            self._disp_offset = np.asarray(disp, dtype=float)
            verts = verts - self._disp_offset[None, :]
            ext = bounds[1] - bounds[0]
            radius = float(np.linalg.norm(ext) * 0.6 + 1e-6)
        except Exception:
            self._z_phys_min = None
            self._z_phys_max = None
            self._disp_offset = np.zeros((3,), dtype=float)
            radius = 1.0
        self._radius = float(radius)

        try:
            # Build range stays in original CMB coordinates.
            self._build_min = float(bounds[0, 2])
            self._build_max = float(bounds[1, 2])
        except Exception:
            self._build_min = None
            self._build_max = None

        try:
            self._mesh_for_slice = trimesh.Trimesh(vertices=verts.astype(np.float64), faces=faces.astype(np.int64), process=False)
            bb = np.asarray(self._mesh_for_slice.bounds, dtype=float)
            self._zmin = float(bb[0, 2])
            self._zmax = float(bb[1, 2])
        except Exception:
            self._mesh_for_slice = None
            self._zmin = None
            self._zmax = None

        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        have_nan = False
        try:
            n = meshdata.vertexNormals()
            have_nan = (n is None) or (not np.isfinite(np.asarray(n)).all())
        except Exception:
            have_nan = True
        item = gl.GLMeshItem(
            meshdata=meshdata,
            smooth=not bool(have_nan),
            shader="shaded",
            color=QtGui.QColor(color).getRgbF(),
            drawFaces=True,
            drawEdges=False,
        )
        try:
            item.setGLOptions("opaque")
        except Exception:
            pass
        self._gl.addItem(item)
        self._mesh_item = item

        # Wireframe overlay (brighter than GLMeshItem edges and easier to read).
        try:
            edges = self._unique_edges(faces, max_edges=200_000)
            seg = verts[edges.reshape(-1)].reshape((-1, 3))
            self._edges_seg = seg
            edges_item = gl.GLLinePlotItem(pos=seg, mode="lines", color=self._edge_rgba, width=1, antialias=True)
            edges_item.setGLOptions("translucent")
            self._gl.addItem(edges_item)
            self._edges_item = edges_item
        except Exception:
            self._edges_item = None
            self._edges_seg = None

        # Fit helpers.
        grid_size = max(20.0, 3.0 * float(radius))
        try:
            self._axis.setSize(grid_size * 0.6, grid_size * 0.6, grid_size * 0.6)
            self._grid.setSize(grid_size, grid_size)
            step = max(1.0, grid_size / 10.0)
            self._grid.setSpacing(step, step)
        except Exception:
            pass

        self._fit_camera()
        self._apply_visibility()
        self._update_slice()

    def _set_toolpath_item(self, attr: str, segments: np.ndarray | None, *, rgba: tuple[float, float, float, float]) -> None:
        item = getattr(self, attr, None)
        if item is not None:
            try:
                self._gl.removeItem(item)
            except Exception:
                pass
            setattr(self, attr, None)
        if segments is None:
            return
        try:
            seg = np.asarray(segments, dtype=np.float32)
            if seg.ndim != 3 or seg.shape[1:] != (2, 3) or seg.shape[0] == 0:
                return
            # Convert from CMB coords to display coords.
            try:
                seg = seg - self._disp_offset.reshape((1, 1, 3)).astype(np.float32)
            except Exception:
                pass
            pts = seg.reshape((-1, 3))
            # Use thicker lines for better readability on dense parts.
            width = 3 if attr.endswith("_layer") else 1
            item = gl.GLLinePlotItem(pos=pts, mode="lines", color=rgba, width=width, antialias=True)
            item.setGLOptions("translucent")
            self._gl.addItem(item)
            setattr(self, attr, item)
        except Exception:
            setattr(self, attr, None)

    def set_toolpath_segments(self, segments: np.ndarray | None) -> None:
        # Backward-compatible: one overlay (current layer).
        self.set_toolpath_layers(range_segments=None, layer_segments=segments)

    def set_toolpath_layers(self, *, range_segments: np.ndarray | None, layer_segments: np.ndarray | None) -> None:
        # Range overlay (faint) + current layer (red).
        self._set_toolpath_item("_toolpath_item_range", range_segments, rgba=(0.10, 0.70, 0.25, 0.12))
        # Highly visible highlight for the current layer.
        self._set_toolpath_item("_toolpath_item_layer", layer_segments, rgba=(1.00, 0.00, 1.00, 1.00))


class _MeshVTKView(QtWidgets.QWidget):
    def __init__(self, *, title: str):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._title = QtWidgets.QLabel(title)
        self._title.setStyleSheet("font-weight: 600;")
        self._stats = QtWidgets.QLabel("")
        self._stats.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.addWidget(self._title, 0)
        header_layout.addStretch(1)
        header_layout.addWidget(self._stats, 0)
        layout.addWidget(header, 0)

        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 0, 8, 6)
        self._edges_cb = QtWidgets.QCheckBox("Рёбра")
        self._edges_cb.setChecked(True)
        self._light_bg_cb = QtWidgets.QCheckBox("Светлый фон")
        self._light_bg_cb.setChecked(True)
        self._fit_btn = QtWidgets.QPushButton("Вписать")
        self._fit_btn.setToolTip("Подогнать камеру под модель")
        hint = QtWidgets.QLabel("ЛКМ: вращение · колесо: зум · ПКМ: панорама")
        hint.setStyleSheet("color: #6B7280;")
        toolbar_layout.addWidget(self._edges_cb, 0)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self._light_bg_cb, 0)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self._fit_btn, 0)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(hint, 0)
        layout.addWidget(toolbar, 0)

        self._plot = None
        try:
            self._plot = QtInteractor(self)
            self._plot.set_background("#F3F4F6")
            layout.addWidget(self._plot.interactor, 1)
        except Exception as e:
            # Fallback placeholder; the higher-level widget may choose another backend on next run,
            # or user can set S2S_PREVIEW_BACKEND=gl to force a stable backend.
            lbl = QtWidgets.QLabel(
                "VTK preview backend is unavailable on this system/session.\n"
                "Set environment variable S2S_PREVIEW_BACKEND=gl to use the OpenGL backend.\n\n"
                f"Error: {e}"
            )
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #B91C1C; padding: 12px;")
            layout.addWidget(lbl, 1)

        self._poly: pv.PolyData | None = None
        self._actor = None
        self._toolpath_actor = None
        self._disp_offset = np.zeros((3,), dtype=float)
        self._bounds: np.ndarray | None = None
        self._clip_enabled: bool = False
        self._clip_t: float = 0.5

        self._edges_cb.toggled.connect(self._update_render)
        self._light_bg_cb.toggled.connect(self._update_render)
        self._fit_btn.clicked.connect(self._fit_camera)

    @staticmethod
    def _to_poly(mesh: trimesh.Trimesh, *, disp_offset: np.ndarray) -> pv.PolyData:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        try:
            verts = verts - np.asarray(disp_offset, dtype=np.float32).reshape((1, 3))
        except Exception:
            pass
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if verts.size == 0 or faces.size == 0:
            return pv.PolyData()
        faces_vtk = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        poly = pv.PolyData(verts, faces_vtk)
        poly.clean(inplace=True)
        return poly

    def _fit_camera(self) -> None:
        if self._plot is None:
            return
        try:
            self._plot.reset_camera()
        except Exception:
            pass
        # Make Z the "up" axis (matches Insight/CMB build coordinates).
        try:
            cam = self._plot.camera
            cam.SetViewUp(0.0, 0.0, 1.0)
        except Exception:
            pass

    def set_mesh(self, mesh: trimesh.Trimesh | None, *, stats_text: str = "", color: str = "#6BCB77") -> None:
        self._stats.setText(stats_text)
        if self._plot is None:
            return
        self._plot.clear()
        self._poly = None
        self._toolpath_actor = None
        self._bounds = None
        self._disp_offset = np.zeros((3,), dtype=float)
        if mesh is None or mesh.faces is None or mesh.vertices is None or len(mesh.faces) == 0:
            self._plot.render()
            return
        try:
            b = np.asarray(mesh.bounds, dtype=float)
            disp = b.mean(axis=0)
            disp[2] = float(b[0, 2])  # build plate at Z=0 (slicer-like)
            self._disp_offset = np.asarray(disp, dtype=float)
        except Exception:
            self._disp_offset = np.zeros((3,), dtype=float)
        poly = self._to_poly(mesh, disp_offset=self._disp_offset)
        self._poly = poly
        try:
            self._bounds = np.array(poly.bounds, dtype=float)
        except Exception:
            self._bounds = None
        self._color = color
        self._update_render()
        self._fit_camera()

    def _set_toolpath_actor(
        self,
        attr: str,
        segments: np.ndarray | None,
        *,
        color: str,
        opacity: float,
        line_width: int,
        tubes: bool,
    ) -> None:
        actor = getattr(self, attr, None)
        if actor is not None:
            try:
                self._plot.remove_actor(actor)
            except Exception:
                pass
            setattr(self, attr, None)
        if segments is None:
            return
        try:
            seg = np.asarray(segments, dtype=np.float32)
            if seg.ndim != 3 or seg.shape[1:] != (2, 3) or seg.shape[0] == 0:
                return
            try:
                seg = seg - self._disp_offset.reshape((1, 1, 3)).astype(np.float32)
            except Exception:
                pass
            pts = seg.reshape((-1, 3))
            n = int(pts.shape[0])
            lines = np.empty((int(n // 2), 3), dtype=np.int64)
            lines[:, 0] = 2
            lines[:, 1] = np.arange(0, n, 2, dtype=np.int64)
            lines[:, 2] = np.arange(1, n, 2, dtype=np.int64)
            poly = pv.PolyData(pts)
            poly.lines = lines.ravel()
            actor = self._plot.add_mesh(
                poly,
                color=color,
                opacity=float(opacity),
                line_width=int(line_width),
                render_lines_as_tubes=bool(tubes),
                lighting=False,
            )
            setattr(self, attr, actor)
        except Exception:
            setattr(self, attr, None)

    def set_toolpath_segments(self, segments: np.ndarray | None) -> None:
        # Backward-compatible: one overlay (current layer).
        self.set_toolpath_layers(range_segments=None, layer_segments=segments)

    def set_toolpath_layers(self, *, range_segments: np.ndarray | None, layer_segments: np.ndarray | None) -> None:
        if self._plot is None:
            return
        if self._poly is None:
            return
        self._set_toolpath_actor("_toolpath_actor_range", range_segments, color="#16A34A", opacity=0.12, line_width=1, tubes=False)
        # Highly visible highlight for the current layer.
        self._set_toolpath_actor("_toolpath_actor_layer", layer_segments, color="#FF00FF", opacity=1.0, line_width=4, tubes=True)
        try:
            self._plot.render()
        except Exception:
            pass

    def _update_render(self) -> None:
        if self._plot is None:
            return
        if self._poly is None:
            return
        poly = self._poly
        if bool(self._clip_enabled) and self._bounds is not None:
            axis_i = 2
            a0 = float(self._bounds[axis_i * 2 + 0])
            a1 = float(self._bounds[axis_i * 2 + 1])
            z0 = a0
            z1 = a1
            t = float(self._clip_t)
            z = z0 + t * (z1 - z0)
            axis_coord = float(z)
            origin = [0.0, 0.0, 0.0]
            origin[axis_i] = float(axis_coord)
            normal = [0.0, 0.0, 0.0]
            normal[axis_i] = 1.0
            try:
                poly = poly.clip(normal=tuple(normal), origin=tuple(origin), invert=False)
            except Exception:
                poly = self._poly

        self._plot.clear()
        self._plot.add_mesh(
            poly,
            color=self._color,
            smooth_shading=True,
            show_edges=bool(self._edges_cb.isChecked()),
            edge_color="#1F2937" if self._light_bg_cb.isChecked() else "#E5E7EB",
            ambient=0.45 if self._light_bg_cb.isChecked() else 0.30,
            diffuse=0.85,
            specular=0.25,
            specular_power=35.0,
        )
        # Slicer-like contour overlay at the clipping plane.
        if bool(self._clip_enabled) and self._bounds is not None:
            try:
                axis_i = 2
                a0 = float(self._bounds[axis_i * 2 + 0])
                a1 = float(self._bounds[axis_i * 2 + 1])
                z0 = a0
                z1 = a1
                t = float(self._clip_t)
                z = z0 + t * (z1 - z0)
                axis_coord = float(z)
                origin = [0.0, 0.0, 0.0]
                origin[axis_i] = float(axis_coord)
                normal = [0.0, 0.0, 0.0]
                normal[axis_i] = 1.0
                contour = self._poly.slice(normal=tuple(normal), origin=tuple(origin))
                self._plot.add_mesh(contour, color="#DC2626", line_width=2)
            except Exception:
                pass
        try:
            self._plot.set_background("#F3F4F6" if self._light_bg_cb.isChecked() else "#111317")
        except Exception:
            pass
        try:
            # Improves depth perception for dense meshes (best-effort).
            self._plot.enable_eye_dome_lighting()
        except Exception:
            pass
        self._plot.show_axes()
        self._plot.render()

    def set_slice(self, *, enabled: bool, t: float) -> None:
        self._clip_enabled = bool(enabled)
        self._clip_t = float(max(0.0, min(1.0, float(t))))
        self._update_render()

    def set_slice_z(self, *, enabled: bool, z_mm: float) -> None:
        self._clip_enabled = bool(enabled)
        if self._bounds is None:
            return
        # Incoming z_mm is in CMB coordinates; display is shifted so build plate is at z=0.
        try:
            z_mm = float(z_mm) - float(self._disp_offset[2])
        except Exception:
            z_mm = float(z_mm)
        axis_i = 2
        a0 = float(self._bounds[axis_i * 2 + 0])
        a1 = float(self._bounds[axis_i * 2 + 1])
        z0 = a0
        z1 = a1
        if abs(z1 - z0) <= 1e-12:
            return
        t = (float(z_mm) - z0) / (z1 - z0)
        t = float(max(0.0, min(1.0, t)))
        self._clip_t = float(t)
        self._update_render()


class MeshSingleWidget(QtWidgets.QWidget):
    def __init__(self, *, title: str = "Модель"):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        view_cls = _select_view_cls()
        # VTK can fail to create an OpenGL context on some systems (drivers / Remote Desktop / debugger terminals).
        # In that case, fall back to OpenGL (pyqtgraph) or 2D.
        try:
            self.view = view_cls(title=title)
        except Exception:
            fallback = _Mesh3DView if gl is not None else _Mesh2DView
            self.view = fallback(title=title)
        layout.addWidget(self.view, 1)

    @staticmethod
    def _fmt_stats(stats: dict, key: str) -> str:
        try:
            v = int(stats[key]["vertices"])
            f = int(stats[key]["faces"])
            est = 84 + 50 * f
            mb = est / (1024.0 * 1024.0)
            extra = ""
            dkey = "display_" + key
            if isinstance(stats.get(dkey), dict):
                pv = int(stats[dkey].get("vertices", 0) or 0)
                pf = int(stats[dkey].get("faces", 0) or 0)
                ds = int(stats[dkey].get("ds", 1) or 1)
                if pv and pf:
                    extra = f"  (preview: V={pv:,} F={pf:,} ds={ds})"
            return f"V={v:,}  F={f:,}  ~{mb:.1f} MiB STL{extra}"
        except Exception:
            return ""

    def set_mesh(self, mesh: trimesh.Trimesh | None, *, stats: dict | None = None, key: str = "after") -> None:
        stats = stats or {}
        self.view.set_mesh(mesh, stats_text=self._fmt_stats(stats, key), color="#6BCB77")

    def set_slice_z(self, *, enabled: bool, z_mm: float) -> None:
        fn = getattr(self.view, "set_slice_z", None)
        if callable(fn):
            fn(enabled=bool(enabled), z_mm=float(z_mm))

    def set_slice(self, *, enabled: bool, t: float) -> None:
        fn = getattr(self.view, "set_slice", None)
        if callable(fn):
            fn(enabled=bool(enabled), t=float(t))

    def set_toolpath_layers(self, *, range_segments: np.ndarray | None, layer_segments: np.ndarray | None) -> None:
        fn = getattr(self.view, "set_toolpath_layers", None)
        if callable(fn):
            fn(range_segments=range_segments, layer_segments=layer_segments)
            return
        fn2 = getattr(self.view, "set_toolpath_segments", None)
        if callable(fn2):
            fn2(layer_segments)


def _build_lightweight_display_mesh(mesh: trimesh.Trimesh, *, target_faces: int = 600_000) -> trimesh.Trimesh:
    """
    Builds a lightweight display mesh that remains renderable in Qt OpenGL/VTK views.

    Naive face sub-sampling can look like a "point cloud" and may generate NaN normals in pyqtgraph OpenGL shaders
    (degenerate triangles). We use a simple deterministic vertex-clustering decimation instead.
    """
    if target_faces <= 0:
        return mesh
    if mesh.faces is None or mesh.vertices is None:
        return mesh

    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        return mesh
    if verts.ndim != 2 or verts.shape[1] != 3:
        return mesh

    n_faces = int(faces.shape[0])
    if n_faces <= int(target_faces):
        return mesh

    # Rough target vertex count for stable shading.
    target_vertices = max(50_000, min(int(0.55 * float(target_faces)), 1_200_000))

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    diag = float(np.linalg.norm(vmax - vmin))
    if not np.isfinite(diag) or diag <= 1e-9:
        return mesh
    ext = vmax - vmin
    # Use bbox volume heuristic too (diag-only tends to over-decimate for thin parts).
    vol = float(max(ext[0] * ext[1] * ext[2], 1e-12))
    cell_diag = diag / (float(target_vertices) ** (1.0 / 3.0))
    cell_vol = (vol / float(target_vertices)) ** (1.0 / 3.0)
    cell = max(min(float(cell_diag), float(cell_vol)), 1e-6)

    q = np.floor((verts - vmin[None, :]) / float(cell)).astype(np.int64)
    q0 = q.min(axis=0)
    q = q - q0[None, :]
    spans = q.max(axis=0) + 1
    spans = np.maximum(spans, 1)
    keys = q[:, 0] + spans[0] * (q[:, 1] + spans[1] * q[:, 2])
    uniq, inv = np.unique(keys, return_inverse=True)
    if uniq.size < 3:
        return mesh

    sums = np.zeros((uniq.size, 3), dtype=np.float64)
    counts = np.zeros((uniq.size,), dtype=np.int64)
    np.add.at(sums, inv, verts)
    np.add.at(counts, inv, 1)
    counts = np.maximum(counts, 1)
    verts2 = sums / counts[:, None]

    faces2 = inv[faces.reshape(-1)].reshape((-1, 3)).astype(np.int64, copy=False)
    a = faces2[:, 0]
    b = faces2[:, 1]
    c = faces2[:, 2]
    ok = (a != b) & (b != c) & (a != c)
    faces2 = faces2[ok]
    if faces2.shape[0] < 10:
        return mesh

    tri = verts2[faces2]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.einsum("ij,ij->i", n, n)
    eps = (1e-10 * diag) ** 2
    faces2 = faces2[area2 > eps]
    if faces2.shape[0] < 10:
        return mesh

    used = np.unique(faces2.reshape(-1))
    remap = np.full((verts2.shape[0],), -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    verts3 = verts2[used]
    faces3 = remap[faces2].astype(np.int64, copy=False)

    out = trimesh.Trimesh(vertices=verts3, faces=faces3, process=False)
    out.metadata.update(getattr(mesh, "metadata", {}) or {})
    out.metadata["s2s_preview_decimated_from_faces"] = n_faces
    out.metadata["s2s_preview_decimated_to_faces"] = int(faces3.shape[0])
    out.metadata["s2s_preview_decimated_from_vertices"] = int(verts.shape[0])
    out.metadata["s2s_preview_decimated_to_vertices"] = int(verts3.shape[0])
    out.metadata["s2s_preview_decimation_cell"] = float(cell)
    return out


_HELP_HTML = """
<style>
  body { font-family: 'Segoe UI', sans-serif; font-size: 13px; color: #111827; }
  h2 { margin: 0 0 10px 0; }
  h3 { margin: 18px 0 8px 0; }
  h4 { margin: 14px 0 8px 0; }
  p, li { line-height: 1.35; }
  code { background: #F3F4F6; padding: 1px 5px; border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  th, td { border: 1px solid #E5E7EB; padding: 8px; vertical-align: top; text-align: left; }
  th { background: #F3F4F6; }
  tr:nth-child(even) td { background: #FAFAFB; }
</style>
<h2>slice2solid - Справка</h2>

<p><b>Подсказки в интерфейсе:</b> наведите курсор на параметр, чтобы увидеть подсказку. Также можно нажать <b>Shift+F1</b> и кликнуть по элементу, чтобы открыть "What's This?".</p>

<h3>1) Что выбрать в Insight</h3>
<ol>
  <li><b>Папка после слайсинга (ssys_*)</b>: папка задания Insight. Внутри должны быть <code>*-simulation-data.txt</code>, <code>toolpathParams.*</code> и (для геометрии) <code>*.stl</code>.</li>
  <li><b>Результат</b>: пишется в <code>&lt;ssys_*&gt;/slice2solid_out</code> (папка создаётся автоматически).</li>
</ol>

<h3>2) Вкладка “CAD / Геометрия”</h3>
<ul>
  <li>Выход (в папке результата): <code>CAD/*_s2s_preview_structure.stl</code> + (опционально) <code>CAD/*_s2s_preview_structure_mesh.ply</code>, <code>CAD/voxel_points.csv</code>, <code>CAD/cad_import_notes.txt</code> + <code>metadata.json</code>.</li>
  <li><b>Пресеты</b> — быстрые наборы настроек. Выберите пресет — настройки применятся сразу. При ручных изменениях режим станет <b>Custom</b>.</li>
  <li><b>Размер вокселя (Voxel size)</b> — главный компромисс «качество/скорость». Меньше -&gt; точнее, но заметно тяжелее по RAM/времени и размеру STL.</li>
  <li><b>Разрежение (downsample)</b> — построение поверхности (marching cubes) по разреженному объёму (каждый N-й воксель): ускоряет расчёт и уменьшает STL, но может “съесть” тонкие элементы.</li>
  <li><b>Сглаживание объёма</b> (sigma, vox) сглаживает воксельный объём перед построением поверхности — уменьшает “ступеньки”.</li>
  <li><b>Сглаживание сетки</b> (Laplacian iterations) сглаживает уже готовую сетку — убирает “рывки”, но может замыливать мелкие детали.</li>
  <li>Если появляются "лишние перемычки/линии" между отдельными дорожками: включите <b>Фильтр перемещений (travel jumps)</b> и/или увеличьте <b>Удаление шума</b>.</li>
  <li>Если вокруг есть мелкие "островки" сетки: увеличьте <b>Удаление островков сетки</b>.</li>
</ul>

<h4>2.0 Файлы результата: что чем является и что куда грузить</h4>
<table>
  <tr><th>Файл</th><th>Что это</th><th>Когда использовать (CAD/CAE)</th></tr>
  <tr>
    <td><code>CAD/*_s2s_preview_structure.stl</code></td>
    <td>Сетка восстановленной <b>явной</b> структуры печати (периметры/заполнение). Обычно очень большой файл.</td>
    <td><b>CAD</b>: визуализация/архив/сеточное тело (mesh body), попытка конвертации в B-Rep. <b>CAE</b>: только если нужна именно явная геометрия инфилла (иначе слишком тяжело).</td>
  </tr>
  <tr>
    <td><code>CAD/*_s2s_preview_structure_mesh.ply</code></td>
    <td>То же самое, что STL, но в PLY (часто быстрее/стабильнее импорт в некоторых инструментах).</td>
    <td><b>CAD/CAE</b>: пробовать вместо STL, если импорт STL проблемный/медленный.</td>
  </tr>
  <tr>
    <td><code>CAD/*_s2s_preview_structure_healed.stl</code></td>
    <td>Результат <b>Mesh Healer</b>: попытка сделать сетку более "чистой" для импорта.</td>
    <td>Выбирать, если исходный STL плохо импортируется (дырки/нормали/мусор).</td>
  </tr>
  <tr>
    <td><code>CAE/ansys_layers.json</code> / <code>CAE/ansys_layers.csv</code></td>
    <td>Ориентация/угол дорожек по слоям Z + confidence (для анизотропии).</td>
    <td><b>CAE (ANSYS)</b>: ключевой результат. Обычно используется вместе с геометрией детали (CAD-solid / обычный STL) — без явного инфилла.</td>
  </tr>
  <tr>
    <td><code>CAE/ansys_mechanical_import_layers.py</code></td>
    <td>Скрипт импорта слоёв/ориентаций в ANSYS Mechanical.</td>
    <td><b>CAE (ANSYS)</b>: запускать в Mechanical, чтобы создать нужные сущности/настройки по слоям.</td>
  </tr>
  <tr>
    <td><code>CAE/ansys_mechanical_section_planes.py</code></td>
    <td>Визуальная проверка: Section Plane по высоте (Z в CMB) + опциональный экспорт PNG по слоям.</td>
    <td><b>CAE (ANSYS)</b>: удобно “глазами” сравнить слои/ориентацию с предпросмотром слайсера.</td>
  </tr>
  <tr>
    <td><code>CAD/voxel_points.csv</code></td>
    <td>Облако точек занятых вокселей (x,y,z), без заголовка.</td>
    <td><b>CAD</b>: иногда лучше для решёток (implicit/volume -&gt; solidify). <b>CAE</b>: редко, скорее для альтернативной реконструкции.</td>
  </tr>
  <tr>
    <td><code>CAD/cad_import_notes.txt</code></td>
    <td>Короткая памятка по импорту и стартовым параметрам spacing/resolution.</td>
    <td><b>CAD</b>: открыть и следовать рекомендациям.</td>
  </tr>
  <tr>
    <td><code>metadata.json</code></td>
    <td>Параметры запуска, матрица (STL-&gt;CMB), статистика сетки/вокселей.</td>
    <td>Для контроля единиц/матрицы/параметров, воспроизводимости и диагностики.</td>
  </tr>
</table>

<p><b>Как получить STEP/твердое тело в стороннем CAD или инструменте работы с сеткой:</b><br/>
Импортируйте сетку (STL/PLY) -&gt; при необходимости Repair/Close/Orient Normals -&gt; затем (если поддерживается) Convert to Solid (B-Rep) -&gt; Export STEP.<br/>
Если включён <b>пакет для CAD (CAD bundle)</b>, файлы лежат в подпапке <code>CAD/</code> (например, <code>CAD/cad_import_notes.txt</code>).</p>

<h4>2.1 Mesh Healer (CAD)</h4>
<ul>
  <li><b>Зачем:</b> некоторые CAD-системы плохо импортируют "грязные" STL (дырки, дубликаты, нулевые грани, проблемы ориентации). Mesh Healer пытается автоматически исправить типовые дефекты.</li>
  <li><b>Что делает (safe):</b> удаляет дубликаты вершин/граней, удаляет неиспользуемые вершины, удаляет нулевые грани, переориентирует грани, закрывает небольшие отверстия.</li>
  <li><b>Профиль:</b> <code>safe</code> (по умолчанию, без ремешинга/упрощения) и <code>aggressive</code> (доп. попытки убрать self-intersections; использовать только если safe не помогает).</li>
  <li><b>Порог дырок (мм):</b> <code>close_holes_max</code> задаёт максимальный размер дырок для закрытия. Примечание: в MeshLab/pymeshlab это обычно лимит по числу рёбер контура, поэтому мм переводятся в рёбра по оценке (это видно в JSON-отчёте).</li>
  <li><b>Выход:</b> рядом с STL появляется <code>*_healed.stl</code> (и опционально <code>*_healed_report.json</code>).</li>
</ul>

<h3>Быстрый гайд по параметрам (что крутить)</h3>
<table>
  <tr><th>Параметр</th><th>Эффект</th><th>Плюсы</th><th>Минусы</th><th>Стартовые значения</th></tr>
  <tr>
    <td><b>Пакет для CAD (CAD bundle)</b></td>
    <td>Пишет доп. файлы для удобного импорта: <code>*.ply</code>, <code>voxel_points.csv</code>, <code>cad_import_notes.txt</code>.</td>
    <td>Упрощает импорт и подбор шага/разрешения (spacing/resolution), даёт облако точек (point cloud) для альтернативного восстановления.</td>
    <td>Доп. файлы в папке результата.</td>
    <td>Включено</td>
  </tr>
  <tr>
    <td><b>Mesh Healer (CAD)</b></td>
    <td>Автоматически исправляет типовые дефекты сетки после экспорта STL.</td>
    <td>Повышает шанс корректного импорта и получения замкнутой (watertight) сетки.</td>
    <td>Может не помочь при очень сложной/самопересекающейся сетке; aggressive может удалять проблемные области.</td>
    <td>Выключено; включать при проблемах импорта</td>
  </tr>
  <tr>
    <td><b>Размер вокселя (Voxel size, mm)</b></td>
    <td>Размер ячейки сетки, из которой строится поверхность.</td>
    <td>Меньше - более гладкая/точная поверхность.</td>
    <td>Меньше - нагрузка по RAM/времени растёт очень резко (примерно кубически), STL тяжелее.</td>
    <td>0.10-0.25 (если грубо - 0.07-0.05)</td>
  </tr>
  <tr>
    <td><b>Ограничение радиуса дорожки (Bead radius limit)</b></td>
    <td>Ограничивает “толщину” дорожки при вокселизации.</td>
    <td>Стабилизирует результат, убирает случайные завышения.</td>
    <td>Слишком мало - "худые" ребра/разрывы.</td>
    <td>Auto; вручную обычно 1.0-2.5 мм</td>
  </tr>
  <tr>
    <td><b>Фильтр перемещений (travel jumps)</b></td>
    <td>Не заполнять материал по длинным перемещениям (travel) между разорванными траекториями.</td>
    <td>Убирает ложные перемычки.</td>
    <td>Если порог слишком строгий - могут появиться разрывы (редко).</td>
    <td>Включено (recommended)</td>
  </tr>
  <tr>
    <td><b>Удаление шума (min voxels)</b></td>
    <td>Удаляет маленькие “пятна” вокселей до построения сетки.</td>
    <td>Убирает мусор, ускоряет marching cubes.</td>
    <td>Слишком много - можно потерять тонкие элементы.</td>
    <td>100-500</td>
  </tr>
  <tr>
    <td><b>Удаление островков сетки (min faces)</b></td>
    <td>Удаляет мелкие куски сетки после построения поверхности.</td>
    <td>Убирает островки/пылинки.</td>
    <td>Слишком много - удалит полезные мелкие детали.</td>
    <td>1000-10000</td>
  </tr>
  <tr>
    <td><b>Сглаживание объёма (sigma, vox)</b></td>
    <td>Гауссово сглаживание объёма (в вокселях) перед marching cubes.</td>
    <td>Лучше “убирает ступеньки” без сильной потери формы.</td>
    <td>Слишком много - тонкие стенки могут "съесться".</td>
    <td>0.8-1.5 (начать с 1.0)</td>
  </tr>
  <tr>
    <td><b>Сглаживание сетки (iterations)</b></td>
    <td>Laplacian smoothing по вершинам после marching cubes.</td>
    <td>Убирает "рывки", делает поверхность приятнее для импорта/исправления в CAD и инструментах работы с сеткой.</td>
    <td>Слишком много - усадка/замыливание деталей.</td>
    <td>10-30 (начать с 15)</td>
  </tr>
  <tr>
    <td><b>Разрежение (downsample)</b></td>
    <td>Строит поверхность по разреженному объёму (каждый N-й воксель).</td>
    <td>Сильно ускоряет marching cubes и уменьшает STL.</td>
    <td>Может “съесть” тонкие элементы и огрубить поверхность.</td>
    <td>1; для ускорения 2-4</td>
  </tr>
</table>

<h3>3) Вкладка “Просмотр”</h3>
<ul>
  <li>Показывает результат <b>последнего запуска</b>: восстановленную модель (после сглаживания).</li>
  <li>Если сетка слишком большая, для интерактивности она автоматически прореживается (в статистике видно <code>preview: ... ds=N</code>).</li>
  <li>Есть <b>сечение</b> по оси печати (Z+ в CMB), контур на плоскости и <b>траектория</b> (один слой / до слоя / все слои).</li>
</ul>

<h3>4) Вкладка “ANSYS / CAE”</h3>
<ul>
  <li>Выход (в папке результата): <code>CAE/ansys_layers.json</code>, <code>CAE/ansys_layers.csv</code>, <code>CAE/ansys_mechanical_import_layers.py</code>, <code>CAE/ansys_mechanical_section_planes.py</code>.</li>
  <li>Идея: назначить ортотропию по слоям (X вдоль печати, Z - ось построения (build direction)).</li>
  <li><b>Пресеты</b> на вкладке ANSYS меняют параметры генерируемого Mechanical-скрипта (группировка слоёв, порог confidence, создавать ли NS/CS).</li>
</ul>
<ol>
  <li>Откройте ANSYS Mechanical, импортируйте геометрию, постройте сетку (Mesh).</li>
  <li>Mechanical -&gt; Automation -&gt; Scripting -&gt; <b>Run Script...</b></li>
  <li>Выберите <code>CAE/ansys_mechanical_import_layers.py</code> из папки результата.</li>
  <li>По умолчанию скрипт применяет матрицу STL -&gt; CMB (как в Insight). Если ваша модель уже в CMB - установите в скрипте <code>APPLY_STL_TO_CMB = False</code>.</li>
  <li>После выполнения появятся Named Selections <code>L_0000</code>, <code>L_0001</code>... и Coordinate Systems <code>CS_L_0000</code>... (если API доступен в вашей конфигурации).</li>
  <li>Для визуальной проверки по слоям: запустите <code>CAE/ansys_mechanical_section_planes.py</code> и меняйте <code>Z_MM</code>.</li>
</ol>

<h3>Блок “Результаты”</h3>
<ul>
  <li>После запуска список файлов заполняется автоматически.</li>
  <li>Двойной клик или кнопка <b>Открыть выбранный</b> открывают файл; <b>Копировать путь</b> кладёт путь в буфер.</li>
  <li><b>Открыть папку результата</b> открывает директорию с выходными файлами.</li>
</ul>

<p><b>Важно:</b> механическая прочность/разрушение задаются в ANSYS материалом. Мы экспортируем “карту печати” (ориентацию по слоям).</p>
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("slice2solid - Восстановление структуры (MVP)")
        self.setWindowIcon(_load_app_icon())
        self.resize(1500, 920)
        self.setMinimumSize(1200, 780)
        self._settings = QtCore.QSettings()

        def _set_help(widget: QtWidgets.QWidget, *, title: str, body: str, pros: str = "", cons: str = "", tip: str = "") -> None:
            parts = [f"<b>{title}</b><br>{body}"]
            if pros:
                parts.append(f"<br><b>Плюсы</b>: {pros}")
            if cons:
                parts.append(f"<br><b>Минусы</b>: {cons}")
            if tip:
                parts.append(f"<br><b>Совет</b>: {tip}")
            html = "".join(parts)
            widget.setToolTip(html)
            widget.setWhatsThis(html)

        def _with_hint(widget: QtWidgets.QWidget, hint: str) -> QtWidgets.QWidget:
            wrap = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            row.addWidget(widget, 0)
            if isinstance(hint, QtWidgets.QLabel):
                lab = hint
            else:
                lab = QtWidgets.QLabel(str(hint))
                lab.setWordWrap(True)
                lab.setStyleSheet("QLabel { color: #64748B; font-size: 12px; }")
            row.addWidget(lab, 1)
            return wrap

        # Справка достаточно подробно расписана в интерфейсе (tooltips + вкладка "Справка"),
        # поэтому отдельные actions/диалоги в меню не добавляем.

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        layout.addWidget(main_splitter, 1)
        try:
            main_splitter.setSizes([650, 250])
        except Exception:
            pass

        scrollbar_css = """
            QScrollBar:vertical { width: 14px; }
            QScrollBar:horizontal { height: 14px; }
            QScrollBar::handle { min-height: 28px; min-width: 28px; background: #9A9A9A; border-radius: 6px; }
            QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; }
            QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
        """

        top_panel = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter.addWidget(top_panel)

        header = QtWidgets.QLabel(
            "Цель: получить `*_s2s_preview_structure.stl`, который можно импортировать как сетку (mesh) в CAD/CAE и, при необходимости,\n"
            "преобразовать в твердое тело (STEP) средствами стороннего CAD/инструмента для работы с сетками.\n"
            "Поддержки/подложка (`Type=0`) игнорируются; используется только траектория модели (`Type=1`).\n"
            "Подсказки: наведите курсор на параметр (или Shift+F1 -> клик).\n"
            "Шаги: Import Mesh -> Repair/Close/Orient Normals -> (если поддерживается) Convert to Solid (B-Rep) -> Export STEP."
        )
        header.setWordWrap(True)
        header.setStyleSheet(
            "QLabel { background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 10px; padding: 10px; color: #334155; font-size: 12px; }"
        )
        top_layout.addWidget(header)

        io_group = QtWidgets.QGroupBox("Шаг 1 - Входные файлы")
        io_form = QtWidgets.QFormLayout(io_group)
        top_layout.addWidget(io_group)

        self.job_edit = QtWidgets.QLineEdit()
        self.job_edit.setPlaceholderText(r"Например: ...\ssys_part-table")
        self.job_edit.setToolTip(
            "Папка задания Stratasys Insight (ssys_*).\n"
            "Внутри должны быть:\n"
            "- `*-simulation-data.txt` (Insight: Toolpaths -> Simulation data export)\n"
            "- `*.stl` (копия исходной геометрии)\n"
            "- `toolpathParams.*` (для авто-радиуса дорожки)\n\n"
            "Дальше программа сама создаст `<ssys_*>/slice2solid_out` и запишет результаты туда."
        )
        self.job_btn = QtWidgets.QPushButton("Обзор...")
        job_row = QtWidgets.QHBoxLayout()
        job_row.addWidget(self.job_edit, 1)
        job_row.addWidget(self.job_btn)
        io_form.addRow("Папка после слайсинга (ssys_*):", job_row)

        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setReadOnly(True)
        self.out_edit.setPlaceholderText(r"Автоматически: <ssys_*>\\slice2solid_out")
        _set_help(
            self.out_edit,
            title="Папка результата",
            body="Результаты автоматически пишутся в `<ssys_*>/slice2solid_out` (папка создаётся при запуске).",
            pros="Не нужно выбирать выходную папку — меньше шансов перепутать.",
            cons="Большие STL могут занимать много места.",
            tip="Если нужно разделять эксперименты — используйте разные папки ssys_* (или копии).",
        )
        io_form.addRow("Папка результата:", self.out_edit)

        tabs = QtWidgets.QTabWidget()
        self.main_tabs = tabs
        top_layout.addWidget(tabs, 1)

        # --- Tab: CAD / Geometry preview ---
        geometry_tab = QtWidgets.QWidget()
        tabs.addTab(geometry_tab, "CAD / Геометрия")
        geo_outer = QtWidgets.QVBoxLayout(geometry_tab)
        geo_outer.setContentsMargins(10, 8, 10, 10)

        geo_scroll = QtWidgets.QScrollArea()
        geo_scroll.setWidgetResizable(True)
        geo_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        geo_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        geo_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        geo_scroll.setStyleSheet(scrollbar_css)
        geo_outer.addWidget(geo_scroll, 1)

        geo_scroll_content = QtWidgets.QWidget()
        geo_scroll.setWidget(geo_scroll_content)
        geo_layout = QtWidgets.QVBoxLayout(geo_scroll_content)
        geo_layout.setContentsMargins(10, 10, 10, 10)
        geo_layout.setSpacing(10)

        geo_intro = QtWidgets.QLabel(
            "Режим CAD: восстановление внутренней структуры и экспорт `*_s2s_preview_structure.stl`.\n"
            "Дальше: импорт в сторонний CAD или инструмент работы с сеткой -> исправление сетки (если нужно) -> преобразование в тело (если поддерживается) -> экспорт STEP."
        )
        geo_intro.setWordWrap(True)
        geo_intro.setStyleSheet(
            "QLabel { background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 10px; padding: 10px; color: #1E3A8A; }"
        )
        geo_layout.addWidget(geo_intro)

        geo_basic_group = QtWidgets.QGroupBox("Основные параметры")
        geo_form = QtWidgets.QFormLayout(geo_basic_group)
        geo_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        geo_form.setHorizontalSpacing(14)
        geo_form.setVerticalSpacing(10)
        geo_layout.addWidget(geo_basic_group)

        self.export_geometry = QtWidgets.QCheckBox("Сгенерировать STL предпросмотра")
        self.export_geometry.setChecked(True)
        self.export_geometry.setToolTip(
            "Пишет `CAD/*_s2s_preview_structure.stl` (обычно большой mesh).\n"
            "Отключите, если нужен только экспорт для ANSYS (карта слоёв)."
        )
        self.export_geometry.setWhatsThis(self.export_geometry.toolTip())
        geo_form.addRow("Выходная геометрия:", self.export_geometry)

        self.export_bundle = QtWidgets.QCheckBox("Экспортировать пакет для CAD (PLY + точки + заметки)")
        self.export_bundle.setChecked(True)
        self.export_bundle.setToolTip(
            "Доп. универсальные файлы для внешних CAD/mesh-инструментов:\n"
            "- `CAD/*_s2s_preview_structure_mesh.ply` (mesh)\n"
            "- `CAD/voxel_points.csv` (point cloud из вокселей)\n"
            "- `CAD/cad_import_notes.txt` (подсказки по импорту/spacing)"
        )
        self.export_bundle.setWhatsThis(self.export_bundle.toolTip())
        geo_form.addRow("Пакет для CAD:", self.export_bundle)

        self.geo_advanced_toggle = QtWidgets.QToolButton()
        self.geo_advanced_toggle.setText("Продвинутые параметры")
        self.geo_advanced_toggle.setCheckable(True)
        self.geo_advanced_toggle.setChecked(False)
        self.geo_advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.geo_advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        geo_layout.addWidget(self.geo_advanced_toggle, 0)

        self.geo_advanced_panel = QtWidgets.QWidget()
        geo_adv_layout = QtWidgets.QVBoxLayout(self.geo_advanced_panel)
        geo_adv_layout.setContentsMargins(0, 0, 0, 0)
        geo_adv_layout.setSpacing(10)
        self.geo_advanced_panel.setVisible(False)
        geo_layout.addWidget(self.geo_advanced_panel, 0)

        geo_adv_group = QtWidgets.QGroupBox("Продвинутые параметры")
        geo_adv_form = QtWidgets.QFormLayout(geo_adv_group)
        geo_adv_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        geo_adv_form.setHorizontalSpacing(14)
        geo_adv_form.setVerticalSpacing(10)
        geo_adv_layout.addWidget(geo_adv_group, 0)

        heal_group = QtWidgets.QGroupBox("Mesh Healer (CAD)")
        heal_form = QtWidgets.QFormLayout(heal_group)
        heal_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        heal_form.setHorizontalSpacing(14)
        heal_form.setVerticalSpacing(10)
        geo_adv_layout.addWidget(heal_group, 0)

        self.heal_enable = QtWidgets.QCheckBox("Автоматически исправить сетку после экспорта STL (*_healed.stl)")
        self.heal_enable.setChecked(False)
        self.heal_enable.setToolTip(
            "Исправляет типовые проблемы сетки (дубликаты, нулевые грани, ориентация, небольшие дырки).\n"
            "Без ремешинга/упрощения, чтобы не разрушить инфилл.\n"
            "Backend по умолчанию: pymeshlab (если доступен), иначе meshlabserver."
        )
        self.heal_enable.setWhatsThis(self.heal_enable.toolTip())
        heal_form.addRow("Включить:", self.heal_enable)

        self.heal_preset_combo = QtWidgets.QComboBox()
        self.heal_preset_combo.addItems(["safe", "aggressive"])
        self.heal_preset_combo.setCurrentText("safe")
        self.heal_preset_combo.setToolTip("safe: без агрессивного удаления; aggressive: доп. попытки удалить self-intersections.")
        heal_form.addRow("Профиль:", self.heal_preset_combo)

        self.close_holes_max = QtWidgets.QDoubleSpinBox()
        self.close_holes_max.setRange(0.0, 100.0)
        self.close_holes_max.setDecimals(2)
        self.close_holes_max.setSingleStep(0.5)
        self.close_holes_max.setValue(2.0)
        self.close_holes_max.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.close_holes_max.setFixedWidth(140)
        self.close_holes_max.setToolTip(
            "Максимальный размер дырки (мм) для закрытия.\n"
            "Примечание: MeshLab использует лимит по числу рёбер контура; программа переводит мм в рёбра по оценке."
        )
        heal_form.addRow("Закрывать дырки до (мм):", self.close_holes_max)

        self.heal_report = QtWidgets.QCheckBox("Записать JSON-отчёт (до/после)")
        self.heal_report.setChecked(False)
        heal_form.addRow("Отчёт:", self.heal_report)

        self.heal_report_path_edit = QtWidgets.QLineEdit()
        self.heal_report_path_edit.setPlaceholderText("Путь (опционально). Пусто = рядом со STL")
        self.heal_report_path_btn = QtWidgets.QPushButton("Обзор...")
        report_row = QtWidgets.QHBoxLayout()
        report_row.addWidget(self.heal_report_path_edit, 1)
        report_row.addWidget(self.heal_report_path_btn)
        heal_form.addRow("Файл отчёта:", report_row)

        self.voxel_size = QtWidgets.QDoubleSpinBox()
        self.voxel_size.setRange(0.05, 5.0)
        self.voxel_size.setSingleStep(0.05)
        self.voxel_size.setValue(0.25)
        self.voxel_size.setDecimals(3)
        self.voxel_size.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.voxel_size.setFixedWidth(140)
        self._voxel_rec_label = QtWidgets.QLabel()
        self._voxel_rec_label.setWordWrap(True)
        self._voxel_rec_label.setStyleSheet("QLabel { color: #64748B; font-size: 12px; }")
        _set_help(
            self.voxel_size,
            title="Размер вокселя (Voxel size), мм",
            body="Размер вокселя (мм): из этой сетки строится поверхность (marching cubes).",
            pros="Меньше - более гладкая/точная поверхность.",
            cons="Меньше - нагрузка по RAM/времени и размер STL растут очень резко (примерно кубически).",
            tip="Если поверхность \"ступеньками\": сначала включите сглаживание объёма (Volume smoothing, примерно 1.0), и только потом уменьшайте voxel size.",
        )
        geo_form.addRow(
            "Размер вокселя (мм):",
            _with_hint(self.voxel_size, self._voxel_rec_label),
        )

        self.auto_radius = QtWidgets.QCheckBox("Авто (из параметров слайсера)")
        self.auto_radius.setChecked(True)
        _set_help(
            self.auto_radius,
            title="Авто-радиус дорожки",
            body="Автоматически берёт радиус дорожки из параметров слайсера (папка ssys_*).",
            pros="Обычно даёт правильную толщину дорожек без ручной настройки.",
            cons="Если ssys_* не выбран/не распознан - авто-режим недоступен.",
            tip="Если авто-режим не сработал или есть \"жирные\" дорожки - снимите Auto и задайте лимит вручную.",
        )
        self.max_radius = QtWidgets.QDoubleSpinBox()
        self.max_radius.setRange(0.1, 10.0)
        self.max_radius.setSingleStep(0.1)
        self.max_radius.setValue(1.5)
        self.max_radius.setDecimals(2)
        self.max_radius.setEnabled(False)
        self.max_radius.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.max_radius.setFixedWidth(140)
        _set_help(
            self.max_radius,
            title="Ограничение радиуса дорожки (мм)",
            body="Ограничение максимального радиуса «сферы» при вокселизации (из Bead Area).",
            pros="Убирает выбросы, делает толщину дорожек стабильнее.",
            cons="Слишком низко - тонкие стенки/ребра могут исчезнуть.",
            tip="Типичные значения: 1.0-2.5 мм. Если модель \"разваливается\" - увеличьте.",
        )
        self.radius_hint = QtWidgets.QLabel("Авто: неизвестно (выберите папку ssys_*)")
        _set_help(
            self.radius_hint,
            title="Авто-радиус: статус",
            body="Подсказка, получилось ли определить радиус автоматически.",
            tip="Выберите папку ssys_* (папка задания слайсера), чтобы авто-режим стал доступен.",
        )
        radius_row = QtWidgets.QHBoxLayout()
        radius_row.addWidget(self.auto_radius)
        radius_row.addWidget(self.max_radius)
        radius_row.addWidget(self.radius_hint, 1)
        geo_adv_form.addRow("Ограничение радиуса дорожки:", radius_row)

        self.estimate = QtWidgets.QLabel("Оценка: -")
        self.estimate.setWordWrap(True)
        _set_help(
            self.estimate,
            title="Оценка нагрузки",
            body="Прикидка размеров воксельной сетки и ожидаемой нагрузки.",
            tip="Если оценка \"слишком большая\" - увеличьте Voxel size или ограничьте область/геометрию.",
        )
        geo_form.addRow("Оценка нагрузки:", self.estimate)

        # --- Presets ---
        self._applying_preset = False
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(["Custom", "Быстро (черновик)", "Баланс", "Качество"])
        self.preset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.preset_combo.setMinimumContentsLength(18)
        preset_wrap = self.preset_combo
        _set_help(
            self.preset_combo,
            title="Пресеты",
            body="Готовые наборы параметров для быстрого старта.",
            pros="Ускоряет настройку для новичков.",
            cons="Не учитывает все особенности детали/траектории.",
            tip="Выберите пресет - настройки применятся сразу. При ручных изменениях режим станет Custom.",
        )
        geo_form.addRow("Пресеты:", preset_wrap)

        self.jump_filter = QtWidgets.QCheckBox("Игнорировать перемещения (travel jumps) между траекториями (рекомендуется)")
        self.jump_filter.setChecked(True)
        _set_help(
            self.jump_filter,
            title="Игнорирование перемещений (travel jumps)",
            body="Игнорирует длинные перемещения между разорванными траекториями (travel/jump), чтобы не ‘заливать’ материал по воздуху.",
            pros="Убирает ложные перемычки и внутренние ‘нитки’.",
            cons="Если траектория реально разорвана короткими прыжками, можно получить разрывы (редко).",
            tip="Обычно держите включённым. Если появились неожиданные дырки - попробуйте временно выключить и сравнить.",
        )
        geo_adv_form.addRow("Фильтр траектории:", self.jump_filter)

        self.min_island = QtWidgets.QSpinBox()
        self.min_island.setRange(0, 10000)
        self.min_island.setValue(150)
        self.min_island.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.min_island.setFixedWidth(140)
        _set_help(
            self.min_island,
            title="Удаление шума (мин. вокселей)",
            body="Удаляет маленькие компоненты вокселей до построения сетки (3D связность).",
            pros="Убирает ‘мусор’ и ускоряет построение сетки.",
            cons="Слишком большое значение может удалить тонкие элементы.",
            tip="Начните с 150-300. Если вокруг много мелких точек - увеличьте; если теряются тонкие элементы - уменьшите.",
        )
        geo_adv_form.addRow("Удаление шума (мин. вокселей):", self.min_island)

        self.min_mesh_faces = QtWidgets.QSpinBox()
        self.min_mesh_faces.setRange(0, 50_000_000)
        self.min_mesh_faces.setValue(2000)
        self.min_mesh_faces.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.min_mesh_faces.setFixedWidth(140)
        _set_help(
            self.min_mesh_faces,
            title="Удаление островков (мин. граней)",
            body="После построения поверхности удаляет куски сетки, у которых меньше указанного числа граней.",
            pros="Убирает мелкие ‘островки’/пылинки вокруг структуры.",
            cons="Слишком большое значение может удалить полезные мелкие детали.",
            tip="Если много мусора вокруг - увеличьте. Если пропадают нужные мелкие элементы - уменьшите.",
        )
        geo_adv_form.addRow("Удаление островков (мин. граней):", self.min_mesh_faces)

        self.vol_sigma = QtWidgets.QDoubleSpinBox()
        self.vol_sigma.setRange(0.0, 5.0)
        self.vol_sigma.setSingleStep(0.1)
        self.vol_sigma.setValue(0.0)
        self.vol_sigma.setDecimals(2)
        self.vol_sigma.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.vol_sigma.setFixedWidth(140)
        self._sigma_rec_label = QtWidgets.QLabel()
        self._sigma_rec_label.setWordWrap(True)
        self._sigma_rec_label.setStyleSheet("QLabel { color: #64748B; font-size: 12px; }")
        _set_help(
            self.vol_sigma,
            title="Сглаживание объёма (sigma, vox)",
            body="Гауссово сглаживание воксельного объёма перед marching cubes (sigma в вокселях).",
            pros="Сильно уменьшает “ступеньки/пилу” без большого роста времени.",
            cons="Слишком большое sigma может ‘съесть’ тонкие стенки и сгладить мелкие детали.",
            tip="Для более гладкой поверхности начните с 1.0. Если тонкие элементы размываются - снизьте до 0.6-0.8.",
        )
        geo_form.addRow("Сглаживание объёма (sigma, vox):", _with_hint(self.vol_sigma, self._sigma_rec_label))

        self.meshing_downsample = QtWidgets.QSpinBox()
        self.meshing_downsample.setRange(1, 64)
        self.meshing_downsample.setValue(1)
        self.meshing_downsample.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.meshing_downsample.setFixedWidth(140)
        self._downsample_rec_label = QtWidgets.QLabel()
        self._downsample_rec_label.setWordWrap(True)
        self._downsample_rec_label.setStyleSheet("QLabel { color: #64748B; font-size: 12px; }")
        _set_help(
            self.meshing_downsample,
            title="Разрежение перед построением поверхности (downsample)",
            body="Строит поверхность по разреженному объёму (каждый N-й воксель) на этапе marching cubes.",
            pros="Сильно ускоряет построение поверхности и уменьшает размер STL.",
            cons="Тонкие элементы могут исчезнуть; поверхность станет грубее.",
            tip="Начните с 2 или 4. Если детали теряются - уменьшайте. Если STL слишком большой - увеличивайте.",
        )
        geo_form.addRow("Разрежение (downsample):", _with_hint(self.meshing_downsample, self._downsample_rec_label))

        self.smooth = QtWidgets.QSpinBox()
        self.smooth.setRange(0, 200)
        self.smooth.setValue(0)
        self.smooth.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.smooth.setFixedWidth(140)
        _set_help(
            self.smooth,
            title="Сглаживание сетки (итерации)",
            body="Сглаживание уже готовой сетки (Laplacian).",
            pros="Убирает ‘рывки’ и делает поверхность приятнее для последующей постобработки/конвертации в CAD.",
            cons="Может вызывать усадку/замыливание деталей при больших значениях.",
            tip="10-30 обычно достаточно. Если форма начинает \"плыть\" - уменьшите.",
        )
        geo_adv_form.addRow("Сглаживание сетки (итерации):", self.smooth)

        geo_layout.addStretch(1)

        # --- Tab: Preview ---
        preview_tab = QtWidgets.QWidget()
        tabs.addTab(preview_tab, "Просмотр")
        self._preview_tab_index = tabs.indexOf(preview_tab)
        prev_layout = QtWidgets.QVBoxLayout(preview_tab)
        prev_layout.setContentsMargins(10, 8, 10, 10)
        prev_layout.setSpacing(10)
        prev_hint = QtWidgets.QLabel(
            "Просмотр результата последнего запуска.\n"
            "Показывает восстановленную геометрию (после сглаживания) и сечение по оси печати (Z+).\n"
            "Если сетка слишком большая, для предпросмотра она автоматически прореживается.\n"
            "Также можно включить траекторию слоя (как в слайсере)."
        )
        prev_hint.setWordWrap(True)
        prev_hint.setStyleSheet(
            "QLabel { background: #F0FDFA; border: 1px solid #99F6E4; border-radius: 10px; padding: 10px; color: #0F766E; }"
        )
        prev_layout.addWidget(prev_hint, 0)
        prev_btn_row = QtWidgets.QHBoxLayout()
        prev_layout.addLayout(prev_btn_row)
        self.preview_reload_btn = QtWidgets.QPushButton("Загрузить из папки результата")
        self.preview_open_folder_btn = QtWidgets.QPushButton("Открыть папку результата")
        prev_btn_row.addWidget(self.preview_reload_btn)
        prev_btn_row.addStretch(1)
        prev_btn_row.addWidget(self.preview_open_folder_btn)

        # Preview controls: slicer-like Z plane (CMB space, build axis is Z+).
        controls_row = QtWidgets.QHBoxLayout()
        prev_layout.addLayout(controls_row, 0)
        self.preview_slice_cb = QtWidgets.QCheckBox("Сечение по оси печати")
        self.preview_slice_cb.setChecked(False)
        self.preview_slice_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.preview_slice_slider.setRange(0, 1000)
        self.preview_slice_slider.setValue(500)
        self.preview_slice_slider.setEnabled(False)
        self.preview_slice_label = QtWidgets.QLabel("Z: -")
        self.preview_slice_label.setStyleSheet("color: #6B7280;")
        self.preview_slice_snap_cb = QtWidgets.QCheckBox("Привязка к слоям")
        self.preview_slice_snap_cb.setChecked(True)
        self.preview_slice_snap_cb.setToolTip("Если есть ansys_layers.csv, слайдер привязывается к ближайшему слою.")
        self.preview_toolpath_cb = QtWidgets.QCheckBox("Траектория слоя")
        self.preview_toolpath_cb.setChecked(False)
        self.preview_toolpath_cb.setToolTip("Показывает линии траектории (Type=1) как в слайсере.")
        self.preview_toolpath_range = QtWidgets.QComboBox()
        self.preview_toolpath_range.addItems(["Один слой", "До слоя", "Все слои"])
        self.preview_toolpath_range.setCurrentIndex(0)
        self.preview_toolpath_range.setEnabled(False)
        self.preview_toolpath_range.setToolTip("Диапазон отображения траекторий: один слой / до слоя / все слои.")
        controls_row.addWidget(self.preview_slice_cb, 0)
        controls_row.addWidget(self.preview_slice_slider, 1)
        controls_row.addWidget(self.preview_slice_label, 0)
        controls_row.addWidget(self.preview_slice_snap_cb, 0)
        controls_row.addWidget(self.preview_toolpath_cb, 0)
        controls_row.addWidget(self.preview_toolpath_range, 0)

        self._preview_mesh_holder = QtWidgets.QWidget()
        self._preview_mesh_holder_layout = QtWidgets.QVBoxLayout(self._preview_mesh_holder)
        self._preview_mesh_holder_layout.setContentsMargins(0, 0, 0, 0)
        prev_layout.addWidget(self._preview_mesh_holder, 1)

        self.mesh_preview = MeshSingleWidget(title="Модель (после сглаживания)")
        self._preview_mesh_holder_layout.addWidget(self.mesh_preview, 1)

        # --- Tab: ANSYS / CAE ---
        ansys_tab = QtWidgets.QWidget()
        tabs.addTab(ansys_tab, "ANSYS / CAE")
        ansys_outer = QtWidgets.QVBoxLayout(ansys_tab)
        ansys_outer.setContentsMargins(10, 8, 10, 10)

        ansys_scroll = QtWidgets.QScrollArea()
        ansys_scroll.setWidgetResizable(True)
        ansys_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        ansys_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ansys_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ansys_scroll.setStyleSheet(scrollbar_css)
        ansys_outer.addWidget(ansys_scroll, 1)

        ansys_scroll_content = QtWidgets.QWidget()
        ansys_scroll.setWidget(ansys_scroll_content)
        ansys_layout = QtWidgets.QVBoxLayout(ansys_scroll_content)
        ansys_layout.setContentsMargins(10, 10, 10, 10)
        ansys_layout.setSpacing(10)

        ansys_intro = QtWidgets.QLabel(
            "Режим ANSYS: экспорт ориентации печати по слоям для назначения ортотропии в Mechanical.\n"
            "Выход: ansys_layers.json/csv + ansys_mechanical_import_layers.py."
        )
        ansys_intro.setWordWrap(True)
        ansys_intro.setStyleSheet(
            "QLabel { background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 10px; padding: 10px; color: #1E3A8A; }"
        )
        ansys_layout.addWidget(ansys_intro)

        ansys_intro2 = QtWidgets.QLabel(
            "Важно: slice2solid работает в координатах CMB (Insight) - ось печати всегда Z+ (как в Insight/слайсере).\n"
            "ANSYS (обычный CAE-режим): импортируйте внешнюю геометрию детали (CAD-solid или обычный STL), "
            "сгенерируйте mesh, затем запустите `ansys_mechanical_import_layers.py` из папки результата.\n"
            "Явную структуру `*_s2s_preview_structure.stl` загружайте только если нужно считать реальный инфилл (файл очень тяжёлый)."
        )
        ansys_intro2.setWordWrap(True)
        ansys_intro2.setStyleSheet(
            "QLabel { background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 10px; padding: 10px; color: #334155; }"
        )
        ansys_layout.addWidget(ansys_intro2)

        ansys_group = QtWidgets.QGroupBox("Параметры CAE")
        ansys_form = QtWidgets.QFormLayout(ansys_group)
        ansys_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        ansys_form.setHorizontalSpacing(14)
        ansys_form.setVerticalSpacing(10)
        ansys_layout.addWidget(ansys_group)

        self.export_cae = QtWidgets.QCheckBox("Экспортировать карту ориентации слоёв (ANSYS)")
        self.export_cae.setChecked(True)
        ansys_form.addRow("Выход CAE:", self.export_cae)

        # --- Mechanical script presets/options ---
        self._applying_ansys_preset = False
        self.ansys_preset_combo = QtWidgets.QComboBox()
        self.ansys_preset_combo.addItems(
            ["Custom", "Подробно (по слоям)", "Быстро (группы по 5 слоёв)", "Только CS (группы по 5)"]
        )
        self.ansys_preset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.ansys_preset_combo.setMinimumContentsLength(22)
        ansys_preset_wrap = self.ansys_preset_combo
        _set_help(
            self.ansys_preset_combo,
            title="Пресеты ANSYS",
            body="Наборы настроек для генерируемого Mechanical-скрипта.",
            pros="Ускоряет старт: можно уменьшить число Named Selections/CS за счёт группировки слоёв.",
            cons="Группировка снижает 'детальность' ориентации по высоте.",
            tip="Выберите пресет - настройки применятся сразу. При ручных изменениях режим станет Custom.",
        )
        ansys_form.addRow("Пресеты:", ansys_preset_wrap)

        self.ansys_min_conf = QtWidgets.QDoubleSpinBox()
        self.ansys_min_conf.setRange(0.0, 1.0)
        self.ansys_min_conf.setSingleStep(0.05)
        self.ansys_min_conf.setDecimals(2)
        self.ansys_min_conf.setValue(0.20)
        self.ansys_min_conf.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.ansys_min_conf.setFixedWidth(140)
        _set_help(
            self.ansys_min_conf,
            title="Min confidence",
            body="Порог ‘confidence’ слоя для создания Coordinate System в Mechanical-скрипте.",
            pros="Отсекает слои с неопределённой/шумной ориентацией.",
            cons="Слишком высокий порог - больше пропусков CS по высоте.",
            tip="Обычно 0.15-0.30. Если CS создаются \"странные\" - поднимите; если CS мало - опустите.",
        )
        ansys_form.addRow("Порог confidence:", self.ansys_min_conf)

        self.ansys_group_layers = QtWidgets.QSpinBox()
        self.ansys_group_layers.setRange(1, 200)
        self.ansys_group_layers.setValue(1)
        self.ansys_group_layers.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.ansys_group_layers.setFixedWidth(140)
        _set_help(
            self.ansys_group_layers,
            title="Group size (layers)",
            body="Сколько слоёв объединять в одну Named Selection/CS в Mechanical-скрипте.",
            pros="Меньше объектов в дереве Mechanical - быстрее и удобнее.",
            cons="Потеря детальности ориентации по высоте.",
            tip="1 = по слоям. Для ускорения попробуйте 5 или 10.",
        )
        ansys_form.addRow("Группировка (слоёв):", self.ansys_group_layers)

        self.ansys_create_ns = QtWidgets.QCheckBox("Создать Named Selections (элементы сетки)")
        self.ansys_create_ns.setChecked(True)
        _set_help(
            self.ansys_create_ns,
            title="Create Named Selections",
            body="Создаёт Named Selection (Mesh Elements) на каждый слой/группу по Z-диапазону.",
            pros="Удобно назначать материалы/постпроцессинг по высоте.",
            cons="Много слоёв - много объектов (может быть тяжело).",
            tip="Если Mechanical \"тяжёлый\" - включите группировку или выключите NS, оставив только CS.",
        )
        ansys_form.addRow("Mechanical:", self.ansys_create_ns)

        self.ansys_create_cs = QtWidgets.QCheckBox("Создать Coordinate Systems")
        self.ansys_create_cs.setChecked(True)
        _set_help(
            self.ansys_create_cs,
            title="Create Coordinate Systems",
            body="Создаёт Coordinate System на каждый слой/группу (X вдоль печати, Z вверх).",
            pros="Можно использовать как ориентацию материала для ортотропии.",
            cons="При низком confidence возможны ‘скачки’ ориентации; тогда помогает Min confidence/Group size.",
            tip="Обычно включено. Если нужны только Named Selections — можно выключить.",
        )
        ansys_form.addRow("", self.ansys_create_cs)

        ansys_hint = QtWidgets.QLabel(
            "ANSYS Mechanical:\n"
            "1) Импортируйте геометрию детали (обычный CAD-solid/STL в исходных координатах), постройте сетку (Mesh).\n"
            "   Размещать/поворачивать вручную не нужно: скрипты используют матрицу STL -> CMB из Insight.\n"
            "2) Mechanical -> Automation -> Scripting -> Run Script...\n"
            "3) Запустите `CAE/ansys_mechanical_import_layers.py` из папки результата.\n"
            "   По умолчанию скрипт сам применяет матрицу STL -> CMB (компенсация усадки/ориентация Insight).\n"
            "   Если ваша модель уже импортирована в CMB - откройте скрипт и установите APPLY_STL_TO_CMB = False.\n"
            "4) Для визуальной проверки по слоям: запустите `CAE/ansys_mechanical_section_planes.py` и меняйте Z_MM.\n"
        )
        ansys_hint.setWordWrap(True)
        ansys_layout.addWidget(ansys_hint)
        ansys_layout.addStretch(1)

        # --- Tab: Help ---
        help_tab = QtWidgets.QWidget()
        tabs.addTab(help_tab, "Справка")
        help_layout = QtWidgets.QVBoxLayout(help_tab)
        help_layout.setContentsMargins(10, 8, 10, 10)
        self.help_view = QtWidgets.QTextBrowser()
        self.help_view.setOpenExternalLinks(True)
        self.help_view.setHtml(_HELP_HTML)
        help_layout.addWidget(self.help_view, 1)

        # --- Run + Outputs (compact bottom panel; collapsible) ---
        bottom_panel = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter.addWidget(bottom_panel)
        self._main_splitter = main_splitter
        self._bottom_tabs: QtWidgets.QTabWidget | None = None
        self._bottom_collapsed = False
        self._bottom_last_height = 260

        bottom_header = QtWidgets.QWidget()
        bottom_header_layout = QtWidgets.QHBoxLayout(bottom_header)
        bottom_header_layout.setContentsMargins(6, 6, 6, 6)
        self.run_btn = QtWidgets.QPushButton("Запуск")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setMinimumHeight(32)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedHeight(18)
        self.bottom_toggle_btn = QtWidgets.QToolButton()
        self.bottom_toggle_btn.setText("Свернуть")
        self.bottom_toggle_btn.setCheckable(True)
        self.bottom_toggle_btn.setChecked(False)
        self.bottom_toggle_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.bottom_toggle_btn.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        bottom_header_layout.addWidget(self.run_btn, 0)
        bottom_header_layout.addSpacing(10)
        bottom_header_layout.addWidget(self.progress, 1)
        bottom_header_layout.addSpacing(10)
        bottom_header_layout.addWidget(self.bottom_toggle_btn, 0)
        bottom_layout.addWidget(bottom_header, 0)

        bottom_tabs = QtWidgets.QTabWidget()
        self._bottom_tabs = bottom_tabs
        bottom_layout.addWidget(bottom_tabs, 1)

        results_tab = QtWidgets.QWidget()
        bottom_tabs.addTab(results_tab, "Результаты")
        out_layout = QtWidgets.QVBoxLayout(results_tab)

        self.outputs_list = QtWidgets.QListWidget()
        out_layout.addWidget(self.outputs_list, 1)

        out_btn_row = QtWidgets.QHBoxLayout()
        out_layout.addLayout(out_btn_row)
        self.open_selected_btn = QtWidgets.QPushButton("Открыть выбранный")
        self.copy_selected_btn = QtWidgets.QPushButton("Копировать путь")
        self.open_out_btn = QtWidgets.QPushButton("Открыть папку результата")
        self.open_out_btn.setEnabled(False)
        self.open_selected_btn.setEnabled(False)
        self.copy_selected_btn.setEnabled(False)
        out_btn_row.addWidget(self.open_selected_btn)
        out_btn_row.addWidget(self.copy_selected_btn)
        out_btn_row.addStretch(1)
        out_btn_row.addWidget(self.open_out_btn)

        log_tab = QtWidgets.QWidget()
        bottom_tabs.addTab(log_tab, "Лог")
        log_layout = QtWidgets.QVBoxLayout(log_tab)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box, 1)

        # Prefer the upper area visually; keep bottom compact but resizable by dragging the splitter.
        main_splitter.setStretchFactor(0, 5)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([1000, 220])
        self._bottom_last_height = 220

        self.thread: QtCore.QThread | None = None
        self.worker: Worker | None = None
        self._last_outputs: list[str] = []
        self._preview_zmin: float | None = None
        self._preview_zmax: float | None = None
        self._preview_layer_z: list[float] | None = None
        self._preview_sim_path: str | None = None
        self._preview_slice_height_mm: float | None = None
        # Cache: (mode, layer_id) -> (range_segments, layer_segments)
        self._preview_toolpath_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._preview_toolpath_last_key: tuple[str, int] | None = None
        # CMB-only: build axis is always Z+.
        self._preview_last_after: trimesh.Trimesh | None = None
        self._preview_last_stats: dict | None = None

        self.job_btn.clicked.connect(self._pick_job)
        self.geo_advanced_toggle.toggled.connect(self._on_geo_advanced_toggled)
        self.run_btn.clicked.connect(self._run)
        self.bottom_toggle_btn.toggled.connect(self._on_bottom_toggle)
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)
        self.auto_radius.toggled.connect(self._update_radius_widgets)
        self.job_edit.textChanged.connect(self._sync_output_dir_from_job)
        self.job_edit.editingFinished.connect(lambda: self._load_last_run_from_output_dir(load_meshes=False))
        self.job_edit.textChanged.connect(self._recompute_auto_radius)
        self.job_edit.textChanged.connect(self._recompute_estimate)
        self.voxel_size.valueChanged.connect(self._recompute_estimate)
        self.jump_filter.toggled.connect(self._recompute_estimate)
        self.min_island.valueChanged.connect(self._recompute_estimate)
        self.min_mesh_faces.valueChanged.connect(self._recompute_estimate)
        self.vol_sigma.valueChanged.connect(self._recompute_estimate)
        self.smooth.valueChanged.connect(self._recompute_estimate)
        self.export_geometry.toggled.connect(self._recompute_estimate)
        self.export_geometry.toggled.connect(self._update_step_widgets)
        self.heal_enable.toggled.connect(self._update_step_widgets)
        self.heal_report.toggled.connect(self._update_step_widgets)
        self.heal_report_path_btn.clicked.connect(self._pick_heal_report_path)
        self.open_out_btn.clicked.connect(self._open_output_folder)
        self.outputs_list.itemSelectionChanged.connect(self._update_output_buttons)
        self.outputs_list.itemDoubleClicked.connect(self._open_selected_output)
        self.open_selected_btn.clicked.connect(self._open_selected_output)
        self.copy_selected_btn.clicked.connect(self._copy_selected_output_path)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selection_changed)
        self.ansys_preset_combo.currentIndexChanged.connect(self._on_ansys_preset_selection_changed)
        self.preview_open_folder_btn.clicked.connect(self._open_output_folder)
        self.preview_reload_btn.clicked.connect(lambda: self._load_last_run_from_output_dir(load_meshes=True))
        self.preview_slice_cb.toggled.connect(self._on_preview_slice_toggled)
        self.preview_slice_slider.valueChanged.connect(self._on_preview_slice_changed)
        self.preview_slice_snap_cb.toggled.connect(self._on_preview_slice_changed)
        self.preview_toolpath_cb.toggled.connect(self._on_preview_toolpath_toggled)
        self.preview_toolpath_range.currentIndexChanged.connect(self._on_preview_slice_changed)

        # If user edits any parameter manually -> switch preset to Custom.
        for w in (
            self.export_geometry,
            self.export_bundle,
            self.heal_enable,
            self.heal_preset_combo,
            self.close_holes_max,
            self.heal_report,
            self.heal_report_path_edit,
            self.voxel_size,
            self.auto_radius,
            self.max_radius,
            self.jump_filter,
            self.min_island,
            self.min_mesh_faces,
            self.vol_sigma,
            self.smooth,
        ):
            self._connect_any_change(w, self._mark_preset_custom)

        for w in (self.ansys_min_conf, self.ansys_group_layers, self.ansys_create_ns, self.ansys_create_cs):
            self._connect_any_change(w, self._mark_ansys_preset_custom)

        self._update_cad_recommendations()

        self._restore_settings()
        self._update_step_widgets()

    def ensure_visible_on_screen(self) -> None:
        try:
            screens = QtGui.QGuiApplication.screens()
            if not screens:
                return

            frame = self.frameGeometry()
            if frame.isNull():
                return

            for s in screens:
                if s.availableGeometry().intersects(frame):
                    return

            primary = QtGui.QGuiApplication.primaryScreen() or screens[0]
            avail = primary.availableGeometry()
            margin = 40

            width = min(max(frame.width(), 920), max(320, avail.width() - margin * 2))
            height = min(max(frame.height(), 640), max(240, avail.height() - margin * 2))
            self.resize(width, height)

            center = avail.center()
            self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)

            self.setWindowState(
                (self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized) | QtCore.Qt.WindowState.WindowActive
            )
            self.raise_()
            self.activateWindow()
        except Exception:
            return

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._settings.setValue("window/geometry", self.saveGeometry())
            self._settings.setValue("window/state", self.saveState())
            self._settings.setValue("paths/job", self.job_edit.text().strip())
            # Output path is derived from the selected job folder.
            try:
                self._settings.remove("paths/out")
            except Exception:
                pass
        except Exception:
            pass
        super().closeEvent(event)

    def _restore_settings(self) -> None:
        try:
            geo = self._settings.value("window/geometry", None)
            state = self._settings.value("window/state", None)
            if isinstance(geo, QtCore.QByteArray):
                self.restoreGeometry(geo)
            if isinstance(state, QtCore.QByteArray):
                self.restoreState(state)

            job = self._settings.value("paths/job", "", type=str) or ""

            if job and not self.job_edit.text().strip():
                self.job_edit.setText(job)
        except Exception:
            return
        self._sync_output_dir_from_job()
        self._load_last_run_from_output_dir(load_meshes=False)

    # Preview backend switching removed: keep a single CMB-oriented preview (Auto => prefer VTK).

    def _update_preview_slider_enabled(self) -> None:
        enabled = bool(self.preview_slice_cb.isChecked()) or bool(self.preview_toolpath_cb.isChecked())
        try:
            self.preview_slice_slider.setEnabled(enabled)
        except Exception:
            pass

    def _on_preview_slice_toggled(self, on: bool) -> None:
        self._update_preview_slider_enabled()
        self._apply_preview_slice()

    def _on_preview_toolpath_toggled(self, on: bool) -> None:
        try:
            self.preview_toolpath_range.setEnabled(bool(on))
        except Exception:
            pass
        self._update_preview_slider_enabled()
        self._apply_preview_slice()

    def _on_preview_slice_changed(self, *_args: object) -> None:
        self._apply_preview_slice()

    def _update_preview_slice_label(self) -> None:
        if not (bool(self.preview_slice_cb.isChecked()) or bool(self.preview_toolpath_cb.isChecked())):
            self.preview_slice_label.setText("Z: -")
            return
        if self._preview_zmin is None or self._preview_zmax is None:
            self.preview_slice_label.setText("Z: -")
            return
        z = self._preview_slice_z_mm()
        self.preview_slice_label.setText(f"Z: {z:.3f}")

    def _preview_slice_z_mm(self) -> float:
        z0 = float(self._preview_zmin or 0.0)
        z1 = float(self._preview_zmax or z0)
        t = float(self.preview_slice_slider.value()) / 1000.0
        z = z0 + t * (z1 - z0)
        if bool(self.preview_slice_snap_cb.isChecked()) and self._preview_layer_z:
            # Snap to closest layer center when available.
            try:
                arr = np.asarray(self._preview_layer_z, dtype=float)
                idx = int(np.argmin(np.abs(arr - z)))
                z = float(arr[idx])
            except Exception:
                pass
        return float(z)

    def _apply_preview_slice(self) -> None:
        self._update_preview_slice_label()
        enabled = bool(self.preview_slice_cb.isChecked())
        z = self._preview_slice_z_mm()
        try:
            self.mesh_preview.set_slice_z(enabled=enabled, z_mm=z)
        except Exception:
            pass
        # Fallback for older view types: use normalized t.
        try:
            self.mesh_preview.set_slice(enabled=enabled, t=float(self.preview_slice_slider.value()) / 1000.0)
        except Exception:
            pass

        self._apply_preview_toolpath(z_mm=z)

    def _apply_preview_toolpath(self, *, z_mm: float) -> None:
        if not hasattr(self, "preview_toolpath_cb"):
            return
        if not bool(self.preview_toolpath_cb.isChecked()):
            try:
                self.mesh_preview.set_toolpath_layers(range_segments=None, layer_segments=None)
            except Exception:
                pass
            self._preview_toolpath_last_key = None
            return
        if not self._preview_sim_path or not Path(self._preview_sim_path).exists():
            return
        if self._preview_slice_height_mm is None or float(self._preview_slice_height_mm) <= 0:
            return

        z0 = float(self._preview_zmin or 0.0)
        slice_h = float(self._preview_slice_height_mm)
        layer_id = int(round((float(z_mm) - z0) / slice_h))
        layer_id = max(0, layer_id)
        mode_idx = 0
        try:
            mode_idx = int(self.preview_toolpath_range.currentIndex())
        except Exception:
            mode_idx = 0
        mode = "layer" if mode_idx == 0 else ("upto" if mode_idx == 1 else "all")

        # max jump for "travel jump" filtering
        max_jump = None
        try:
            max_jump = float(self._last_max_jump_mm) if hasattr(self, "_last_max_jump_mm") else None
        except Exception:
            max_jump = None

        # Layer-only cache (used by all modes to draw current layer in red).
        layer_key = ("layer", int(layer_id))
        if layer_key not in self._preview_toolpath_cache:
            try:
                seg_layer = extract_toolpath_segments_for_layer(
                    self._preview_sim_path,
                    z_center_mm=float(z_mm),
                    slice_height_mm=slice_h,
                    max_jump_mm=max_jump,
                    max_segments=250_000,
                )
                self._preview_toolpath_cache[layer_key] = (np.zeros((0, 2, 3), dtype=np.float32), seg_layer)
            except Exception:
                return

        range_seg = None
        layer_seg = self._preview_toolpath_cache[layer_key][1]

        if mode == "layer":
            range_seg = None
            apply_key = layer_key
        elif mode == "upto":
            upto_key = ("upto", int(layer_id))
            if upto_key not in self._preview_toolpath_cache:
                try:
                    seg_range, seg_cur = extract_toolpath_segments_for_range(
                        self._preview_sim_path,
                        z0_mm=z0,
                        slice_height_mm=slice_h,
                        max_layer_id_inclusive=int(layer_id),
                        max_jump_mm=max_jump,
                        max_segments=600_000,
                    )
                    self._preview_toolpath_cache[upto_key] = (seg_range, seg_cur)
                except Exception:
                    return
            range_seg = self._preview_toolpath_cache[upto_key][0]
            layer_seg = self._preview_toolpath_cache[upto_key][1]
            apply_key = upto_key
        else:
            all_key = ("all", 0)
            if all_key not in self._preview_toolpath_cache:
                try:
                    seg_range, _seg_cur = extract_toolpath_segments_for_range(
                        self._preview_sim_path,
                        z0_mm=z0,
                        slice_height_mm=slice_h,
                        max_layer_id_inclusive=None,
                        max_jump_mm=max_jump,
                        max_segments=1_200_000,
                    )
                    self._preview_toolpath_cache[all_key] = (seg_range, np.zeros((0, 2, 3), dtype=np.float32))
                except Exception:
                    return
            range_seg = self._preview_toolpath_cache[all_key][0]
            # current layer still highlighted in red
            layer_seg = self._preview_toolpath_cache[layer_key][1]
            # Track the current layer to refresh the red highlight when the slider moves.
            apply_key = ("all", int(layer_id))

        if self._preview_toolpath_last_key == apply_key:
            # Avoid excessive re-rendering when slider sends repeated events.
            return
        try:
            self.mesh_preview.set_toolpath_layers(range_segments=range_seg, layer_segments=layer_seg)
            self._preview_toolpath_last_key = apply_key
        except Exception:
            pass

    def _load_last_run_from_output_dir(self, *, load_meshes: bool) -> None:
        out = self.out_edit.text().strip()
        if not out:
            return
        out_dir = Path(out)
        meta_path = out_dir / "metadata.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return

        # Toolpath overlay inputs.
        try:
            inputs = meta.get("inputs") if isinstance(meta, dict) else None
            if isinstance(inputs, dict):
                sp = inputs.get("simulation_txt")
                self._preview_sim_path = str(sp) if isinstance(sp, str) and sp else None
            else:
                self._preview_sim_path = None
        except Exception:
            self._preview_sim_path = None

        try:
            sh = None
            if isinstance(meta.get("simulation_header"), dict):
                sh = meta["simulation_header"].get("Slice height")
            self._preview_slice_height_mm = float(sh) if sh is not None and str(sh).strip() else None
        except Exception:
            self._preview_slice_height_mm = None

        try:
            self._preview_toolpath_cache.clear()
            self._preview_toolpath_last_key = None
        except Exception:
            pass

        outputs = meta.get("outputs")
        if isinstance(outputs, list):
            try:
                self._set_outputs([str(x) for x in outputs])
            except Exception:
                pass

        after_path = None
        if isinstance(outputs, list):
            for x in outputs:
                p = str(x)
                pl = p.lower()
                if pl.endswith("_mesh.ply") and "_s2s_preview_structure" in pl:
                    after_path = p
                if pl.endswith(".ply") and "_before" not in pl and "_s2s_preview_structure" in pl and after_path is None:
                    after_path = p
                if pl.endswith(".stl") and "_s2s_preview_structure" in pl and after_path is None:
                    after_path = p

        if after_path is None:
            candidates = sorted(
                (out_dir / "CAD").glob("*_s2s_preview_structure*_mesh.ply"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if candidates:
                after_path = str(candidates[0])
        if after_path is None:
            # Backward compatibility: older runs used `{stem}.ply`.
            candidates = sorted((out_dir / "CAD").glob("*_s2s_preview_structure*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                after_path = str(candidates[0])

        # Keep these paths local: opening meshes is done via the Results list.

        # Preferred slicing build-height range: from ansys_layers.csv if present (aligns with CAE layers).
        self._preview_layer_z = None
        self._preview_zmin = None
        self._preview_zmax = None
        try:
            layers_csv = (out_dir / "CAE") / "ansys_layers.csv"
            if layers_csv.exists():
                txt = layers_csv.read_text(encoding="utf-8", errors="replace").splitlines()
                if len(txt) >= 2:
                    header = [h.strip() for h in txt[0].split(",")]
                    idx = {h: i for i, h in enumerate(header)}
                    zmin_i = idx.get("z_min_mm")
                    zmax_i = idx.get("z_max_mm")
                    if zmin_i is not None and zmax_i is not None:
                        zmins = []
                        zmaxs = []
                        zc = []
                        for line in txt[1:]:
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) <= max(zmin_i, zmax_i):
                                continue
                            try:
                                zmin = float(parts[zmin_i])
                                zmax = float(parts[zmax_i])
                            except Exception:
                                continue
                            zmins.append(zmin)
                            zmaxs.append(zmax)
                            zc.append(0.5 * (zmin + zmax))
                        if zmins and zmaxs:
                            self._preview_zmin = float(min(zmins))
                            self._preview_zmax = float(max(zmaxs))
                            self._preview_layer_z = sorted(float(x) for x in zc)
        except Exception:
            self._preview_layer_z = None

        if not load_meshes or after_path is None:
            # Fallback slicing range: from last in-memory mesh bounds (if available).
            if self._preview_zmin is None or self._preview_zmax is None:
                try:
                    if isinstance(self._preview_last_after, trimesh.Trimesh):
                        bb = np.asarray(self._preview_last_after.bounds, dtype=float)
                        self._preview_zmin = float(bb[0, 2])
                        self._preview_zmax = float(bb[1, 2])
                except Exception:
                    self._preview_zmin = None
                    self._preview_zmax = None
            self._apply_preview_slice()
            return

        try:
            after_mesh = trimesh.load_mesh(after_path, force="mesh")
        except Exception:
            return

        # Fallback slicing Z-range: from mesh bounds.
        if self._preview_zmin is None or self._preview_zmax is None:
            try:
                bb = np.asarray(after_mesh.bounds, dtype=float)
                self._preview_zmin = float(bb[0, 2])
                self._preview_zmax = float(bb[1, 2])
            except Exception:
                self._preview_zmin = None
                self._preview_zmax = None

        stats = {"after": {"vertices": int(after_mesh.vertices.shape[0]), "faces": int(after_mesh.faces.shape[0])}}

        # Some preview backends (notably Qt OpenGL) can silently fail to render multi-million-face meshes.
        # Use a lightweight deterministic face subsample for interactive preview while keeping original files intact.
        try:
            target_faces = 600_000
            disp_after = _build_lightweight_display_mesh(after_mesh, target_faces=target_faces)
            stats["display_after"] = {"vertices": int(disp_after.vertices.shape[0]), "faces": int(disp_after.faces.shape[0]), "ds": 1}
            after_mesh = disp_after
        except Exception:
            pass
        try:
            self.mesh_preview.set_mesh(after_mesh, stats=stats, key="after")
        except Exception:
            pass
        self._preview_last_after = after_mesh
        self._preview_last_stats = stats
        self._apply_preview_slice()

    def _connect_any_change(self, widget: QtWidgets.QWidget, cb) -> None:
        # Best-effort connections for common widget types.
        try:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(cb)  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        try:
            if hasattr(widget, "toggled"):
                widget.toggled.connect(cb)  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        try:
            if hasattr(widget, "textChanged"):
                widget.textChanged.connect(cb)  # type: ignore[attr-defined]
                return
        except Exception:
            pass

    def _mark_preset_custom(self, *_args: object) -> None:
        if getattr(self, "_applying_preset", False):
            return
        if self.preset_combo.currentText() != "Custom":
            self.preset_combo.setCurrentText("Custom")
        self._update_cad_recommendations()

    def _mark_ansys_preset_custom(self, *_args: object) -> None:
        if getattr(self, "_applying_ansys_preset", False):
            return
        if self.ansys_preset_combo.currentText() != "Custom":
            self.ansys_preset_combo.setCurrentText("Custom")

    def _update_cad_recommendations(self) -> None:
        def _set(label: QtWidgets.QLabel | None, text: str, *, tone: str = "muted") -> None:
            if label is None:
                return
            label.setText(text)
            if tone == "ok":
                label.setStyleSheet("QLabel { color: #0F766E; font-size: 12px; }")
            elif tone == "warn":
                label.setStyleSheet("QLabel { color: #B45309; font-size: 12px; }")
            else:
                label.setStyleSheet("QLabel { color: #64748B; font-size: 12px; }")

        preset = ""
        try:
            preset = str(self.preset_combo.currentText()).strip()
        except Exception:
            preset = ""

        rec_map = {
            "Быстро (черновик)": dict(voxel=0.25, sigma=0.0, downsample=4),
            "Баланс": dict(voxel=0.10, sigma=1.0, downsample=2),
            "Качество": dict(voxel=0.05, sigma=1.0, downsample=1),
            # legacy/compat keys
            "Fast (draft)": dict(voxel=0.25, sigma=0.0, downsample=4),
            "Balanced": dict(voxel=0.10, sigma=1.0, downsample=2),
            "Quality": dict(voxel=0.05, sigma=1.0, downsample=1),
        }

        if preset in rec_map:
            r = rec_map[preset]
            v = float(getattr(self.voxel_size, "value")())
            s = float(getattr(self.vol_sigma, "value")())
            d = int(getattr(self.meshing_downsample, "value")())

            tol_v = 1e-6
            ok_v = abs(v - float(r["voxel"])) <= tol_v
            ok_s = abs(s - float(r["sigma"])) <= 1e-6
            ok_d = int(d) == int(r["downsample"])

            _set(self._voxel_rec_label, f"Реком. для пресета: {r['voxel']:.2f} мм", tone="ok" if ok_v else "warn")
            _set(self._sigma_rec_label, f"Реком. для пресета: {r['sigma']:.1f}", tone="ok" if ok_s else "warn")
            _set(self._downsample_rec_label, f"Реком. для пресета: {int(r['downsample'])}", tone="ok" if ok_d else "warn")
            return

        # Custom / no preset selected: show general guidance.
        _set(self._voxel_rec_label, "Реком.: 0.25 (быстро) / 0.10 (баланс) / 0.05 (качество)", tone="muted")
        _set(self._sigma_rec_label, "Реком.: 0.6-1.0 (для сглаживания); 0.0 если важны тонкие элементы", tone="muted")
        _set(self._downsample_rec_label, "Реком.: 2-4 (если STL слишком большой/медленно)", tone="muted")

    def _on_preset_selection_changed(self, *_args: object) -> None:
        self._update_cad_recommendations()
        # Apply immediately to keep the UI simple (no extra "Apply" button).
        try:
            if self.preset_combo.currentText() != "Custom":
                self._apply_selected_preset()
        except Exception:
            pass

    def _apply_selected_preset(self) -> None:
        preset = self.preset_combo.currentText()
        if preset == "Custom":
            return

        presets = {
            "Fast (draft)": dict(
                voxel_size_mm=0.25,
                meshing_downsample_factor=4,
                volume_smooth_sigma_vox=0.0,
                smooth_iterations=0,
                min_component_voxels=150,
                min_mesh_component_faces=2000,
            ),
            "Balanced": dict(
                voxel_size_mm=0.10,
                meshing_downsample_factor=2,
                volume_smooth_sigma_vox=1.0,
                smooth_iterations=15,
                min_component_voxels=150,
                min_mesh_component_faces=2000,
            ),
            "Quality": dict(
                voxel_size_mm=0.05,
                meshing_downsample_factor=1,
                volume_smooth_sigma_vox=1.0,
                smooth_iterations=25,
                min_component_voxels=150,
                min_mesh_component_faces=2000,
            ),
            # RU aliases (current UI)
            "Быстро (черновик)": dict(
                voxel_size_mm=0.25,
                meshing_downsample_factor=4,
                volume_smooth_sigma_vox=0.0,
                smooth_iterations=0,
                min_component_voxels=150,
                min_mesh_component_faces=2000,
            ),
            "Баланс": dict(
                voxel_size_mm=0.10,
                meshing_downsample_factor=2,
                volume_smooth_sigma_vox=1.0,
                smooth_iterations=15,
                min_component_voxels=150,
                min_mesh_component_faces=2000,
            ),
            "Качество": dict(
                voxel_size_mm=0.05,
                meshing_downsample_factor=1,
                volume_smooth_sigma_vox=1.0,
                smooth_iterations=25,
                min_component_voxels=150,
                min_mesh_component_faces=2000,
            ),
        }
        cfg = presets.get(preset)
        if not cfg:
            return

        self._applying_preset = True
        try:
            self.voxel_size.setValue(float(cfg["voxel_size_mm"]))
            self.meshing_downsample.setValue(int(cfg.get("meshing_downsample_factor", int(self.meshing_downsample.value()))))
            self.vol_sigma.setValue(float(cfg["volume_smooth_sigma_vox"]))
            self.smooth.setValue(int(cfg["smooth_iterations"]))
            self.min_island.setValue(int(cfg["min_component_voxels"]))
            self.min_mesh_faces.setValue(int(cfg["min_mesh_component_faces"]))
        finally:
            self._applying_preset = False

        self._recompute_estimate()
        self._update_cad_recommendations()

    def _apply_selected_ansys_preset(self) -> None:
        preset = self.ansys_preset_combo.currentText()
        if preset == "Custom":
            return

        presets = {
            "Detailed (per layer)": dict(group_size_layers=1, min_conf=0.2, create_ns=True, create_cs=True),
            "Fast (group 5 layers)": dict(group_size_layers=5, min_conf=0.2, create_ns=True, create_cs=True),
            "CS only (group 5)": dict(group_size_layers=5, min_conf=0.2, create_ns=False, create_cs=True),
            # RU aliases (current UI)
            "Подробно (по слоям)": dict(group_size_layers=1, min_conf=0.2, create_ns=True, create_cs=True),
            "Быстро (группы по 5 слоёв)": dict(group_size_layers=5, min_conf=0.2, create_ns=True, create_cs=True),
            "Только CS (группы по 5)": dict(group_size_layers=5, min_conf=0.2, create_ns=False, create_cs=True),
        }
        cfg = presets.get(preset)
        if not cfg:
            return

        self._applying_ansys_preset = True
        try:
            self.ansys_group_layers.setValue(int(cfg["group_size_layers"]))
            self.ansys_min_conf.setValue(float(cfg["min_conf"]))
            self.ansys_create_ns.setChecked(bool(cfg["create_ns"]))
            self.ansys_create_cs.setChecked(bool(cfg["create_cs"]))
        finally:
            self._applying_ansys_preset = False

    def _on_ansys_preset_selection_changed(self, *_args: object) -> None:
        # Apply immediately to keep the UI simple (no extra "Apply" button).
        try:
            if self.ansys_preset_combo.currentText() != "Custom":
                self._apply_selected_ansys_preset()
        except Exception:
            pass
    def _pick_job(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку после слайсинга (ssys_*)")
        if path:
            self.job_edit.setText(path)
            self._auto_fill_from_job(path)

    def _on_geo_advanced_toggled(self, checked: bool) -> None:
        checked = bool(checked)
        try:
            self.geo_advanced_panel.setVisible(checked)
        except Exception:
            pass
        try:
            self.geo_advanced_toggle.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow
            )
        except Exception:
            pass

    def _set_bottom_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if getattr(self, "_bottom_collapsed", False) == collapsed:
            return
        self._bottom_collapsed = collapsed
        try:
            if self._bottom_tabs is not None:
                self._bottom_tabs.setVisible(not collapsed)
        except Exception:
            pass
        try:
            self.bottom_toggle_btn.setText("Развернуть" if collapsed else "Свернуть")
            self.bottom_toggle_btn.setArrowType(QtCore.Qt.ArrowType.RightArrow if collapsed else QtCore.Qt.ArrowType.DownArrow)
        except Exception:
            pass
        try:
            if self._main_splitter is not None:
                sizes = list(self._main_splitter.sizes())
                if len(sizes) >= 2:
                    if not collapsed:
                        sizes[1] = int(getattr(self, "_bottom_last_height", 220) or 220)
                    else:
                        # Keep a thin strip for Run + progress.
                        self._bottom_last_height = int(sizes[1])
                        sizes[1] = 54
                    # Increase top accordingly.
                    sizes[0] = max(200, int(sizes[0]) + int(sizes[1]))
                    self._main_splitter.setSizes(sizes[:2])
        except Exception:
            pass

    def _on_bottom_toggle(self, checked: bool) -> None:
        self._set_bottom_collapsed(bool(checked))

    def _on_main_tab_changed(self, idx: int) -> None:
        # Auto-collapse bottom panel on Preview tab to maximize the graphics area.
        try:
            if hasattr(self, "_preview_tab_index") and int(idx) == int(self._preview_tab_index):
                self._set_bottom_collapsed(True)
        except Exception:
            pass

    def _append_log(self, msg: str) -> None:
        self.log_box.appendPlainText(msg)

    def _set_outputs(self, outputs: list[str]) -> None:
        self._last_outputs = [str(p) for p in outputs]
        self.outputs_list.clear()
        for p in self._last_outputs:
            item = QtWidgets.QListWidgetItem(Path(p).name)
            item.setToolTip(p)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
            self.outputs_list.addItem(item)
        self._update_output_buttons()

    def _selected_output_path(self) -> str | None:
        item = self.outputs_list.currentItem()
        if item is None:
            return None
        p = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return str(p) if p else None

    def _update_output_buttons(self) -> None:
        has = self._selected_output_path() is not None
        self.open_selected_btn.setEnabled(has)
        self.copy_selected_btn.setEnabled(has)

    def _open_selected_output(self, *_args: object) -> None:
        p = self._selected_output_path()
        if not p:
            return
        try:
            os.startfile(p)  # type: ignore[attr-defined]
        except Exception:
            # fallback: open containing folder
            try:
                os.startfile(str(Path(p).parent))  # type: ignore[attr-defined]
            except Exception:
                pass

    def _copy_selected_output_path(self) -> None:
        p = self._selected_output_path()
        if not p:
            return
        QtWidgets.QApplication.clipboard().setText(p)

    def _show_help(self) -> None:
        msg = (
            "1) Выберите папку после слайсинга (ssys_*).\n"
            "   Внутри должен быть `*-simulation-data.txt` (Insight: Toolpaths -> Simulation data export).\n\n"
            "2) Для восстановления геометрии нужен `*.stl` в этой же папке.\n"
            "   Если STL не находится: включите в Insight сохранение копии STL в папку задания (Save STL copy / Save STL in job folder)\n"
            "   или скопируйте исходный STL в папку ssys_* вручную.\n\n"
            "3) Нажмите Запуск. Результаты пишутся в `<ssys_*>/slice2solid_out` (папка создаётся автоматически).\n\n"
            "После запуска:\n"
            " - вкладка 'Просмотр' показывает модель (после сглаживания), сечение по Z и траекторию слоёв\n"
            " - блок 'Результаты' позволяет открыть файлы/папку и скопировать путь\n\n"
            "Основные выходные файлы:\n"
            " - `metadata.json` (параметры/матрица/статистика)\n"
            " - `CAD/*_s2s_preview_structure.stl` (явная геометрия инфилла — тяжёлая)\n"
            " - `CAD/*_s2s_preview_structure_mesh.ply`, `CAD/voxel_points.csv`, `CAD/cad_import_notes.txt` (опционально)\n"
            " - `CAE/ansys_layers.json`, `CAE/ansys_layers.csv` + `CAE/ansys_mechanical_import_layers.py`\n"
            " - `CAE/ansys_mechanical_section_planes.py` (визуальная проверка слоёв в Mechanical)\n\n"
            " - *_healed.stl (+ *_healed_report.json), если включён Mesh Healer (CAD)\n\n"
            "Подробности: вкладка 'Справка' и docs/cad_import_guide_ru.md."
        )
        QtWidgets.QMessageBox.information(self, "Как пользоваться", msg)

    def _open_output_folder(self) -> None:
        out = self.out_edit.text().strip()
        if not out:
            return
        try:
            os.startfile(out)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _auto_fill_from_job(self, job_dir: str) -> None:
        self.out_edit.setText(str(Path(job_dir) / "slice2solid_out"))
        # Reset preview toolpath cache when switching jobs.
        try:
            self._preview_sim_path = str(infer_simulation_txt_from_job(job_dir) or "")
        except Exception:
            self._preview_sim_path = None
        try:
            self._preview_toolpath_cache.clear()
            self._preview_toolpath_last_key = None
        except Exception:
            pass
        self._recompute_auto_radius()
        self._recompute_estimate()

    def _sync_output_dir_from_job(self) -> None:
        job_dir = self.job_edit.text().strip()
        if not job_dir:
            if self.out_edit.text().strip():
                self.out_edit.setText("")
            return
        try:
            out = str(Path(job_dir) / "slice2solid_out")
        except Exception:
            return
        if self.out_edit.text().strip() != out:
            self.out_edit.setText(out)

    def _update_radius_widgets(self) -> None:
        manual = not self.auto_radius.isChecked()
        self.max_radius.setEnabled(manual)
        self.radius_hint.setVisible(not manual)

    def _update_step_widgets(self) -> None:
        enabled = bool(self.export_geometry.isChecked())
        self.export_bundle.setEnabled(enabled)
        if not enabled:
            self.export_bundle.setChecked(False)
        heal_master_enabled = enabled and bool(self.heal_enable.isChecked())
        self.heal_preset_combo.setEnabled(heal_master_enabled)
        self.close_holes_max.setEnabled(heal_master_enabled)
        self.heal_report.setEnabled(heal_master_enabled)
        report_enabled = heal_master_enabled and bool(self.heal_report.isChecked())
        self.heal_report_path_edit.setEnabled(report_enabled)
        self.heal_report_path_btn.setEnabled(report_enabled)

    def _pick_heal_report_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Файл JSON-отчёта", "", "JSON (*.json)")
        if path:
            self.heal_report_path_edit.setText(path)

    def _recompute_auto_radius(self) -> None:
        if not self.auto_radius.isChecked():
            return

        job_dir = self.job_edit.text().strip()
        if not job_dir:
            self.radius_hint.setText("Авто: неизвестно (выберите папку ssys_*)")
            return
        if not Path(job_dir).exists():
            self.radius_hint.setText("Авто: папка не найдена")
            return

        try:
            header = None
            sim_path = infer_simulation_txt_from_job(job_dir)
            if sim_path is not None and sim_path.exists():
                header, _it = read_simulation_export(str(sim_path))
            params = load_job_params(job_dir)
            bead_w = estimate_bead_width_mm(params, sim_slice_height_mm=header.slice_height_mm if header else None)
            if bead_w is None:
                self.radius_hint.setText("Авто: не удалось определить ширину дорожки")
                return
            r = bead_w / 2.0
            self.radius_hint.setText(f"Авто: ширина {bead_w:.3f} мм, радиус {r:.3f} мм")
        except Exception as e:
            self.radius_hint.setText(f"Авто: ошибка ({e})")

    def _recompute_estimate(self) -> None:
        if hasattr(self, "export_geometry") and not self.export_geometry.isChecked():
            self.estimate.setText("Оценка: геометрия отключена (только ANSYS/CAE).")
            return
        job_dir = self.job_edit.text().strip()
        if not job_dir or not Path(job_dir).exists():
            self.estimate.setText("Оценка: выберите папку ssys_*, чтобы посчитать габариты/сетку.")
            return
        stl_path = infer_stl_from_job_folder(job_dir)
        if stl_path is None or not stl_path.exists():
            self.estimate.setText(
                "Оценка: STL не найден в папке ssys_*.\n"
                "В Insight включите сохранение копии STL в папку задания (Save STL copy / Save STL in job folder) "
                "или скопируйте STL в эту папку вручную."
            )
            return
        try:
            mesh = trimesh.load_mesh(str(stl_path), force="mesh")
            bmin = mesh.bounds[0]
            bmax = mesh.bounds[1]
            v = float(self.voxel_size.value())
            size = bmax - bmin
            nx, ny, nz = (int(np.ceil(s / v)) + 1 for s in size)
            voxels = nx * ny * nz
            occ_mb = float(voxels) / (1024 * 1024)  # bool ~1 byte worst-case
            sigma = float(self.vol_sigma.value()) if hasattr(self, "vol_sigma") else 0.0
            ds = int(self.meshing_downsample.value()) if hasattr(self, "meshing_downsample") else 1

            # Rough workload indicator (quality-first; long runtime is acceptable).
            if voxels < 20_000_000:
                level = "Легко"
            elif voxels < 80_000_000:
                level = "Средне"
            elif voxels < 200_000_000:
                level = "Тяжело"
            else:
                level = "Очень тяжело"

            extra = ""
            if sigma > 0:
                # Conservative estimate: float32 volume copies during smoothing/meshing.
                vol_gb = (float(voxels) * 4.0 * 3.0) / float(1024**3)
                extra = f"\nСглаживание объёма: может потребоваться временно до ~{vol_gb:.1f} GB (RAM/диск)."

            self.estimate.setText(
                "Оценка нагрузки:\n"
                f" - Габариты STL: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} мм\n"
                f" - Воксельная сетка: {nx} x {ny} x {nz} (примерно {voxels:,} ячеек)\n"
                f" - Память под объём (минимум): ~{occ_mb:.0f} MB\n"
                f" - Построение поверхности: downsample={ds}, sigma={sigma:.2f}\n"
                f" - Итог: {level}{extra}"
            )
        except Exception as e:
            self.estimate.setText(f"Оценка: ошибка чтения STL ({e})")

    def _run(self) -> None:
        job_dir = self.job_edit.text().strip() or None
        do_geo = bool(self.export_geometry.isChecked())
        do_bundle = bool(self.export_bundle.isChecked()) and do_geo
        do_cae = bool(self.export_cae.isChecked())

        if not do_geo and not do_cae:
            QtWidgets.QMessageBox.warning(self, "Нечего делать", "Включите выходную геометрию и/или экспорт для ANSYS.")
            return

        if not job_dir:
            QtWidgets.QMessageBox.warning(self, "Не хватает данных", "Выберите папку после слайсинга (ssys_*).")
            return
        if not Path(job_dir).exists():
            QtWidgets.QMessageBox.warning(self, "Папка не найдена", "Указанная папка ssys_* не существует.")
            return

        # Output folder is always derived from job_dir to avoid using a stale path.
        out = str(Path(job_dir) / "slice2solid_out")
        self.out_edit.setText(out)

        missing: list[str] = []

        sim_path = infer_simulation_txt_from_job(job_dir)
        if sim_path is None or not sim_path.exists():
            missing.append("`*-simulation-data.txt` (Insight: Toolpaths -> Simulation data export)")

        stl_path = infer_stl_from_job_folder(job_dir)
        if do_geo and (stl_path is None or not stl_path.exists()):
            missing.append("`*.stl` (копия STL в папке ssys_*)")

        if do_geo and self.auto_radius.isChecked() and not has_toolpath_params(job_dir):
            missing.append("`toolpathParams.new` или `toolpathParams.cur` (для авто-радиуса)")

        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "Не хватает файлов в папке ssys_*",
                "В выбранной папке не хватает файлов:\n- " + "\n- ".join(missing),
            )
            return

        max_r: float | None
        max_jump: float | None
        header = None
        if self.jump_filter.isChecked() or (do_geo and self.auto_radius.isChecked()):
            header, _it = read_simulation_export(str(sim_path))

        # For slicer-like preview overlay.
        try:
            self._preview_sim_path = str(sim_path)
        except Exception:
            self._preview_sim_path = None
        try:
            self._preview_slice_height_mm = float(header.slice_height_mm) if header is not None and header.slice_height_mm else None
        except Exception:
            self._preview_slice_height_mm = None
        try:
            self._preview_toolpath_cache.clear()
            self._preview_toolpath_last_key = None
        except Exception:
            pass

        params = None
        bead_w: float | None = None
        thresholds = None

        if do_geo:
            if self.auto_radius.isChecked():
                params = load_job_params(job_dir)
                bead_w = estimate_bead_width_mm(params, sim_slice_height_mm=header.slice_height_mm if header else None)
                if bead_w is None:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Авто-радиус не сработал",
                        "Не удалось вычислить ширину дорожки из `toolpathParams.*`.\n"
                        "Отключите авто-радиус и задайте радиус вручную, либо проверьте содержимое папки ssys_*.",
                    )
                    return
                max_r = bead_w / 2.0
            else:
                max_r = float(self.max_radius.value())
        else:
            max_r = None

        # jump threshold: derived from segment filter length when enabled
        if self.jump_filter.isChecked():
            seg = (header.segment_filter_length_mm if header else None)
            if job_dir and params is None:
                try:
                    params = load_job_params(job_dir)
                except Exception:
                    params = None

            if params is not None and bead_w is None:
                bead_w = estimate_bead_width_mm(params, sim_slice_height_mm=header.slice_height_mm if header else None)

            if params is not None:
                thresholds = estimate_toolpath_thresholds_mm(params, sim_slice_height_mm=header.slice_height_mm if header else None)

            max_jump = estimate_auto_max_jump_mm(
                header_segment_filter_length_mm=seg,
                bead_width_mm=bead_w,
                thresholds_mm=thresholds,
                fallback_mm=3.0 * float((seg or 0.508)),
            )
            if max_r is not None:
                max_jump = max(float(max_jump), 3.0 * float(max_r))
        else:
            max_jump = None

        # Cache for preview overlay (toolpath lines).
        try:
            self._last_max_jump_mm = float(max_jump) if max_jump is not None else None
        except Exception:
            self._last_max_jump_mm = None

        cfg = JobConfig(
            simulation_txt=str(sim_path),
            job_dir=job_dir,
            placed_stl=str(stl_path) if stl_path is not None else "",
            output_dir=out,
            voxel_size_mm=float(self.voxel_size.value()),
            max_radius_mm=max_r,
            max_jump_mm=max_jump,
            min_component_voxels=int(self.min_island.value()),
            min_mesh_component_faces=int(self.min_mesh_faces.value()),
            volume_smooth_sigma_vox=float(self.vol_sigma.value()),
            meshing_downsample_factor=int(self.meshing_downsample.value()),
            smooth_iterations=int(self.smooth.value()),
            export_cae_layers=do_cae,
            export_geometry_preview=do_geo,
            export_cad_bundle=do_bundle,
            ansys_min_confidence=float(self.ansys_min_conf.value()),
            ansys_group_size_layers=int(self.ansys_group_layers.value()),
            ansys_create_named_selections=bool(self.ansys_create_ns.isChecked()),
            ansys_create_coordinate_systems=bool(self.ansys_create_cs.isChecked()),
            heal_enabled=bool(do_geo and self.heal_enable.isChecked()),
            heal_preset=str(self.heal_preset_combo.currentText()).strip().lower() or "safe",
            heal_close_holes_max_mm=float(self.close_holes_max.value()),
            heal_report_enabled=bool(do_geo and self.heal_enable.isChecked() and self.heal_report.isChecked()),
            heal_report_path=str(self.heal_report_path_edit.text().strip()) or None,
            heal_backend="auto",
        )

        self.progress.setValue(0)
        self.log_box.clear()
        self._append_log("Запуск...")
        self.open_out_btn.setEnabled(False)
        try:
            self.mesh_preview.set_mesh(None, stats={}, key="after")
        except Exception:
            pass

        self.run_btn.setEnabled(False)
        self.thread = QtCore.QThread()
        self.worker = Worker(cfg)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self._append_log)
        self.worker.meshes_ready.connect(self._update_preview)
        self.worker.finished.connect(self._done)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _update_preview(self, before: object, after: object, stats: object) -> None:
        # Keep last preview mesh for the Preview tab (single-view).
        try:
            if isinstance(after, trimesh.Trimesh):
                self._preview_last_after = after
                self._preview_last_stats = stats if isinstance(stats, dict) else {}
                try:
                    bb = np.asarray(after.bounds, dtype=float)
                    self._preview_zmin = float(bb[0, 2])
                    self._preview_zmax = float(bb[1, 2])
                except Exception:
                    self._preview_zmin = None
                    self._preview_zmax = None
        except Exception:
            pass
        try:
            self.mesh_preview.set_mesh(after if isinstance(after, trimesh.Trimesh) else None, stats=stats if isinstance(stats, dict) else {}, key="after")
        except Exception:
            pass
        self._apply_preview_slice()

    def _done(self, ok: bool, message: str, outputs: object) -> None:
        self.run_btn.setEnabled(True)
        self._append_log(message)
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Error", message)
            return
        if isinstance(outputs, list):
            self._set_outputs([str(x) for x in outputs])
            # Keep preview snappy: we already have meshes in-memory via `_update_preview`.
            self._load_last_run_from_output_dir(load_meshes=False)
        self.open_out_btn.setEnabled(True)

    def _show_about(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("О программе")
        dlg.setWindowIcon(_load_app_icon())
        layout = QtWidgets.QVBoxLayout(dlg)

        header = QtWidgets.QWidget()
        header.setStyleSheet("background-color: #0B2A4A;")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(12)

        logo = _load_logo_pixmap(56)
        if logo is not None and not logo.isNull():
            logo_label = QtWidgets.QLabel()
            logo_label.setPixmap(logo)
            logo_label.setFixedSize(56, 56)
            logo_label.setScaledContents(True)
            header_layout.addWidget(logo_label, 0, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        header_title = QtWidgets.QLabel("Белорусский государственный технологический университет")
        header_title.setStyleSheet("color: white; font-weight: 600;")
        header_title.setWordWrap(True)
        header_layout.addWidget(header_title, 1)

        layout.addWidget(header)

        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setHtml(_about_html())
        layout.addWidget(view)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)

        dlg.setMinimumWidth(520)
        dlg.exec()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    app = QtWidgets.QApplication(sys.argv)
    _apply_app_style(app)
    try:
        app.setOrganizationName(ORGANIZATION or "slice2solid")
        app.setApplicationName(APP_DISPLAY_NAME or "slice2solid")
    except Exception:
        pass

    if "--reset-ui" in argv or os.environ.get("S2S_RESET_UI") == "1":
        try:
            settings = QtCore.QSettings()
            settings.clear()
            settings.sync()
        except Exception:
            pass
    splash: QtWidgets.QSplashScreen | None = None
    timer = QtCore.QElapsedTimer()
    try:
        timer.start()
        splash = QtWidgets.QSplashScreen(_create_splash_pixmap())
        splash.setWindowIcon(_load_app_icon())
        splash.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        try:
            splash.setWindowOpacity(0.0)
        except Exception:
            pass
        splash.show()
        splash.showMessage(
            "Загрузка...",
            QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignHCenter,
            QtGui.QColor(255, 255, 255, 200),
        )
        try:
            anim_in = QtCore.QPropertyAnimation(splash, b"windowOpacity")
            anim_in.setDuration(550)
            anim_in.setStartValue(0.0)
            anim_in.setEndValue(1.0)
            anim_in.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
            setattr(splash, "_s2s_anim_in", anim_in)
        except Exception:
            pass
        app.processEvents()
    except Exception:
        splash = None

    w = MainWindow()

    if splash is not None:
        try:
            try:
                min_ms = int(os.environ.get("S2S_SPLASH_MS", "6000"))
            except Exception:
                min_ms = 6000
            elapsed = int(timer.elapsed()) if timer.isValid() else min_ms
            remaining = max(0, int(min_ms - elapsed))
            if remaining > 0:
                loop = QtCore.QEventLoop()
                QtCore.QTimer.singleShot(remaining, loop.quit)
                loop.exec()
        except Exception:
            pass

        # Fade-out splash BEFORE showing the main window (prevents the splash from "hanging" on top).
        try:
            anim_out = QtCore.QPropertyAnimation(splash, b"windowOpacity")
            anim_out.setDuration(550)
            anim_out.setStartValue(1.0)
            anim_out.setEndValue(0.0)
            anim_out.finished.connect(splash.close)
            anim_out.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

            loop = QtCore.QEventLoop()
            anim_out.finished.connect(loop.quit)
            loop.exec()
        except Exception:
            try:
                splash.close()
            except Exception:
                pass

    w.show()
    try:
        app.processEvents()
        w.ensure_visible_on_screen()
    except Exception:
        pass
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
