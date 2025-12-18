from __future__ import annotations

import html
import json
import os
import re
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

from slice2solid.core.insight_simulation import (
    invert_rowvec_matrix,
    read_simulation_export,
    transform_toolpath_points_rowvec,
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
            sim_header, rows_iter = read_simulation_export(self.cfg.simulation_txt)
            for w in sim_header.validation_warnings():
                self.log.emit(f"WARNING: {w}")
            slice_h = sim_header.slice_height_mm or 0.254

            if not self.cfg.export_cae_layers and not self.cfg.export_geometry_preview:
                raise ValueError("Не выбраны выходные файлы: включите Геометрию и/или экспорт для ANSYS.")

            if self.cfg.export_geometry_preview:
                self.log.emit("Загрузка placed STL…")
                mesh = trimesh.load_mesh(self.cfg.placed_stl, force="mesh")
                bounds_min = mesh.bounds[0]
                bounds_max = mesh.bounds[1]
            else:
                bounds_min = None
                bounds_max = None
                try:
                    p_stl = Path(self.cfg.placed_stl) if self.cfg.placed_stl else None
                    if p_stl is not None and p_stl.exists():
                        self.log.emit("Загрузка placed STL (только bbox):")
                        _m = trimesh.load_mesh(str(p_stl), force="mesh")
                        bounds_min = _m.bounds[0]
                        bounds_max = _m.bounds[1]
                except Exception:
                    bounds_min = None
                    bounds_max = None

            self.log.emit("Подготовка преобразования координат (CMB → placed STL)…")
            stl_to_cmb = sim_header.stl_to_cmb
            cmb_to_stl = invert_rowvec_matrix(stl_to_cmb)

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
                    "Toolpath bbox (CMB→STL) does not match placed STL bounds "
                    f"(max exceed {max_exceed:.3f} mm, tol {tol:.3f} mm). "
                    "Likely wrong transform / units / shrink."
                )
                if max_exceed >= err:
                    raise ValueError(msg)
                self.log.emit(f"WARNING: {msg}")

            layers = []
            z0 = None

            if self.cfg.export_cae_layers:
                if bounds_min is not None:
                    z0 = float(bounds_min[2])
                else:
                    self.log.emit("Сканирование Z0 (min Z) для CAE…")
                    z0_val = None
                    for pt in _count_points(_track_bbox(transform_toolpath_points_rowvec(rows_iter, cmb_to_stl))):
                        if pt.type != 1:
                            continue
                        z0_val = float(pt.z) if z0_val is None else min(z0_val, float(pt.z))
                    counted = True
                    z0 = float(z0_val) if z0_val is not None else 0.0
                    sim_header, rows_iter = read_simulation_export(self.cfg.simulation_txt)

                self.log.emit("Computing per-layer print orientation (CAE export)…")
                pts = _track_bbox(transform_toolpath_points_rowvec(rows_iter, cmb_to_stl))
                if bounds_min is None and not counted:
                    pts = _count_points(pts)
                    counted = True
                layers = compute_layer_orientations_toolpath(
                    pts,
                    slice_height_mm=float(slice_h),
                    z0_mm=float(z0),
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

                self.log.emit("Вокселизация…")
                self.progress.emit(35)
                _header2, rows2 = read_simulation_export(self.cfg.simulation_txt)
                pts_iter = _track_bbox(_count_points(transform_toolpath_points_rowvec(rows2, cmb_to_stl)))
                counted = True
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
                )
                _check_bbox_against_stl()
                self.progress.emit(70)

                self.log.emit("Построение сетки (marching cubes)…")
                preview_mesh = mesh_from_voxels_configured(
                    vox,
                    volume_smooth_sigma_vox=self.cfg.volume_smooth_sigma_vox,
                    min_component_faces=self.cfg.min_mesh_component_faces,
                    downsample_factor=int(self.cfg.meshing_downsample_factor)
                    if int(self.cfg.meshing_downsample_factor) > 1
                    else None,
                )
                try:
                    ds = int(preview_mesh.metadata.get("meshing_downsample_factor", 1))
                    eff = float(preview_mesh.metadata.get("meshing_voxel_size_mm", float(self.cfg.voxel_size_mm)))
                    if ds > 1:
                        self.log.emit(f"Meshing downsample: x{ds} (effective voxel {eff:.3f} mm)")
                except Exception:
                    pass
                mesh_before = preview_mesh.copy()
                if self.cfg.smooth_iterations > 0:
                    self.log.emit(f"Сглаживание сетки ({self.cfg.smooth_iterations} итераций)…")
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
            # Prefix geometry outputs to reduce confusion with the user-provided "placed STL".
            preview_stem = _preview_mesh_stem(self.cfg)
            out_stl = out_dir / f"{preview_stem}.stl"
            # PLY is an optional CAD-bundle artifact; name it explicitly to avoid confusion with the STL.
            out_ply = out_dir / f"{preview_stem}_mesh.ply"
            out_ply_before = out_dir / f"{preview_stem}_mesh_before.ply"
            out_notes = out_dir / "cad_import_notes.txt"
            out_points = out_dir / "voxel_points.csv"
            out_json = out_dir / "metadata.json"
            out_layers_json = out_dir / "ansys_layers.json"
            out_layers_csv = out_dir / "ansys_layers.csv"
            out_ansys_script = out_dir / "ansys_mechanical_import_layers.py"
            out_mapdl_script = out_dir / "ansys_mapdl_layers.mac"
            out_section_planes_script = out_dir / "ansys_mechanical_section_planes.py"
            out_insight_sgm = out_dir / "insight_part.sgm"

            outputs: list[str] = []
            bundle_written = False

            if self.cfg.export_geometry_preview and preview_mesh is not None:
                self.log.emit(f"Запись {out_stl}…")
                preview_mesh.export(out_stl)
                outputs.append(str(out_stl))
                if bool(self.cfg.heal_enabled):
                    try:
                        healed_stl = out_dir / f"{out_stl.stem}_healed{out_stl.suffix}"
                        report_path = None
                        if bool(self.cfg.heal_report_enabled):
                            if self.cfg.heal_report_path:
                                report_path = Path(self.cfg.heal_report_path)
                            else:
                                report_path = out_dir / f"{out_stl.stem}_healed_report.json"
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
                        self.log.emit(f"WARNING: Mesh Healer failed: {e}")
                try:
                    if mesh_before is not None:
                        self.log.emit(f"Запись {out_ply_before}:")
                        mesh_before.export(out_ply_before)
                        outputs.append(str(out_ply_before))
                except Exception:
                    pass
                if self.cfg.export_cad_bundle:
                    try:
                        self.log.emit(f"Запись {out_ply}…")
                        preview_mesh.export(out_ply)
                        outputs.append(str(out_ply))

                        notes = _render_cad_import_notes(self.cfg)
                        out_notes.write_text(notes, encoding="utf-8")
                        outputs.append(str(out_notes))

                        if vox is not None:
                            self.log.emit(f"Запись {out_points}…")
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
                        self.log.emit(f"CAD bundle пропущен: {e}")

            if self.cfg.export_cae_layers:
                self.log.emit(f"Запись {out_layers_json}…")
                out_layers_json.write_text(
                    json.dumps(
                        {
                            "slice_height_mm": float(slice_h),
                            "z0_mm": float(z0) if z0 is not None else None,
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
                self.log.emit(f"Запись {out_layers_csv}…")
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

                self.log.emit(f"Запись {out_ansys_script}…")
                out_ansys_script.write_text(_render_ansys_mechanical_script(self.cfg), encoding="utf-8")
                outputs.append(str(out_ansys_script))

                # Preferred (path B): MAPDL snippet for layer components + ESYS assignment.
                out_mapdl_script.write_text(
                    _render_ansys_mapdl_script(
                        layers=[
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
                        cfg=self.cfg,
                    ),
                    encoding="utf-8",
                )
                outputs.append(str(out_mapdl_script))

                # Mechanical helper: create a Section Plane (and optionally export PNGs per layer).
                out_section_planes_script.write_text(
                    _render_ansys_mechanical_section_planes_script(self.cfg),
                    encoding="utf-8",
                )
                outputs.append(str(out_section_planes_script))

                # Optional: extract Insight SGM (slice geometry) for diagnostics / validation.
                try:
                    if self.cfg.job_dir and Path(self.cfg.job_dir).exists():
                        extracted = extract_sgm_to_folder(self.cfg.job_dir, out_dir)
                        if extracted is not None:
                            try:
                                out_insight_sgm.write_bytes(Path(extracted.extracted_path).read_bytes())
                                outputs.append(str(out_insight_sgm))
                            except Exception:
                                outputs.append(str(extracted.extracted_path))
                except Exception:
                    pass

            meta = {
                "inputs": asdict(self.cfg),
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
#   2) Create Named Selections of MESH ELEMENTS per layer (by element centroid Z)
#   3) Create Coordinate Systems per layer (X along print direction, Z global +Z)
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
        _log("ERROR: AddNamedSelection failed for", name, ":", e)
        return None

    if ns is None:
        _log("ERROR: AddNamedSelection returned None for", name)
        return None

    try:
        ns.Name = name
    except Exception as e:
        _log("WARN: failed to set NamedSelection.Name for", name, ":", e)

    try:
        ns.Location = sel
    except Exception as e:
        _log("ERROR: failed to set NamedSelection.Location for", name, ":", e)
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
        print("ERROR: This script must be run inside ANSYS Mechanical (Workbench).")
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
            print("ERROR: ansys_layers.csv not found and JSON could not be loaded.")
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
    for i, eid in enumerate(ids):
        c = _element_centroid(mesh, eid)
        elem_xyz[eid] = c
        elem_z[eid] = c[2]
        if (i + 1) % 50000 == 0:
            print("  processed", i + 1, "elements")

    print("Elements:", len(ids))

    # Detect unit mismatch between layer Z (typically mm) and Mechanical API coordinates (often meters).
    try:
        elem_z_min = min(elem_z.values())
        elem_z_max = max(elem_z.values())
    except Exception:
        elem_z_min = 0.0
        elem_z_max = 0.0

    try:
        layers_z_min = min(float(l.get("z_min", 0.0)) for l in layers)
        layers_z_max = max(float(l.get("z_max", 0.0)) for l in layers)
    except Exception:
        layers_z_min = 0.0
        layers_z_max = 0.0

    # Scale factor from layer-units -> element-units.
    # If layer Z-range is ~1000x larger, element coords are likely meters while layers are mm.
    layer_to_elem = 1.0
    if elem_z_max > 1e-12 and layers_z_max > 1e-12:
        ratio = float(layers_z_max) / float(elem_z_max)
        if ratio > 200.0 and ratio < 5000.0:
            layer_to_elem = 1.0 / ratio
            _log(
                "INFO: detected unit mismatch; scaling layer Z by",
                layer_to_elem,
                "(layer-units -> element-units).",
                "elem_z_max=",
                elem_z_max,
                "layers_z_max=",
                layers_z_max,
            )
        else:
            _log("INFO: unit check ok. elem_z_max=", elem_z_max, "layers_z_max=", layers_z_max)
    else:
        _log("WARN: cannot detect units (zero range). elem_z_max=", elem_z_max, "layers_z_max=", layers_z_max)

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
                    (float(d[0]), float(d[1]), float(d[2])),
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


_ANSYS_MAPDL_SCRIPT_TEMPLATE = r"""! slice2solid -> MAPDL layer helper (for Workbench Mechanical)
!
! Goal:
! - Create element components per layer (or layer-group) by centroid Z
! - Assign element coordinate systems (ESYS) per layer using in-plane toolpath angle
! - Emit a small per-layer report (element counts, angle, confidence)
!
! How to use (Mechanical):
! - Static Structural -> Environment -> Commands
! - Paste this file contents, or use "Read Input File" if available in your version.
!
! Notes:
! - Units follow your Mechanical model units (recommended: mm). This script uses Z ranges in mm.
! - This avoids Mechanical API Named Selections (which can be unstable on some builds).
! - This snippet temporarily enters /PREP7 and then returns to /SOLU so Mechanical can continue solving.
!
/PREP7
CSYS,0
!
! Write a small CSV-like report next to solver working directory.
*CFOPEN,ansys_mapdl_layers_report,txt
*VWRITE,'name,zmin_mm,zmax_mm,nelem,angle_deg,confidence'
(A)
!
! ---- CONFIG injected by slice2solid ----
! MIN_CONFIDENCE: {min_conf}
! GROUP_SIZE_LAYERS: {group_size}
! CS_BASE_ID: {cs_base}
!
! ---- BEGIN LAYERS ----
{layers_block}
! ---- END LAYERS ----
!
ALLSEL,ALL
CSYS,0
*CFCLOS
/SOLU
"""


_ANSYS_MECHANICAL_SECTION_PLANES_TEMPLATE = r"""# -*- coding: utf-8 -*-
# slice2solid: Mechanical helper (Section Planes)
#
# What it does:
# - Creates (or reuses) a single active Section Plane named 'S2S_Slice'
# - Moves it to a requested Z (mm)
# - Optional: exports one PNG per layer for comparison with slicer
#
# How to use (Mechanical):
# - Automation -> Scripting -> Open Script... -> select this file -> Run
# - Edit Z_MM / EXPORT_IMAGES below and re-run as needed

import csv
import os

from Ansys.Mechanical.Graphics import GraphicsImageExportFormat, GraphicsImageExportSettings
from Ansys.Mechanical.Graphics import Point, SectionPlane, SectionPlaneType, Vector3D


HERE = os.path.dirname(__file__)
LAYERS_CSV = os.path.join(HERE, "ansys_layers.csv")

# User knobs
Z_MM = {z_mm:.6f}  # move slice here (mm)
EXPORT_IMAGES = {export_images}  # True to export a PNG per layer
EXPORT_DIR = os.path.join(HERE, "s2s_slices_png")
PLANE_NAME = "S2S_Slice"


def _load_layers_csv(path):
    layers = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lid = int(row[\"layer_id\"])
            zmin = float(row[\"z_min_mm\"])
            zmax = float(row[\"z_max_mm\"])
            layers.append((lid, zmin, zmax))
    return layers


def _get_or_create_plane(name):
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
    sp.Direction = Vector3D(0, 0, 1)
    sp.Center = Point([0.0, 0.0, 0.0], \"mm\")
    Graphics.SectionPlanes.Add(sp)
    return sp


def _set_plane_z(plane, z_mm):
    plane.Center = Point([0.0, 0.0, float(z_mm)], \"mm\")


def _export_png(path):
    settings = GraphicsImageExportSettings()
    settings.Width = 1600
    settings.Height = 900
    settings.CurrentGraphicsDisplay = True
    Graphics.ExportImage(path, GraphicsImageExportFormat.PNG, settings)


plane = _get_or_create_plane(PLANE_NAME)
_set_plane_z(plane, Z_MM)

if EXPORT_IMAGES:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    layers = _load_layers_csv(LAYERS_CSV)
    for lid, zmin, zmax in layers:
        zmid = 0.5 * (zmin + zmax)
        _set_plane_z(plane, zmid)
        out = os.path.join(EXPORT_DIR, f\"slice_L_{lid:04d}_z{zmid:.3f}mm.png\")
        _export_png(out)

print(f\"S2S: Section plane '{PLANE_NAME}' set to Z={Z_MM} mm\")  # noqa: T201
"""


def _render_ansys_mapdl_script(*, layers: list[dict[str, object]], cfg: JobConfig) -> str:
    min_conf = float(cfg.ansys_min_confidence)
    group_size = int(cfg.ansys_group_size_layers) if int(cfg.ansys_group_size_layers) > 0 else 1
    cs_base = 1000  # avoid conflicts with user-defined CS

    def _grouped() -> list[tuple[int, int, float, float, float | None, float, float]]:
        groups: dict[int, list[dict[str, object]]] = {}
        if group_size <= 1:
            for l in layers:
                gid = int(l.get("layer_id", 0) or 0)
                groups.setdefault(gid, []).append(l)
        else:
            for l in layers:
                lid = int(l.get("layer_id", 0) or 0)
                gid = lid // group_size
                groups.setdefault(gid, []).append(l)

        out: list[tuple[int, int, float, float, float | None, float, float]] = []
        for gid in sorted(groups.keys()):
            gl = groups[gid]
            lids = [int(x.get("layer_id", 0) or 0) for x in gl]
            lid0 = min(lids) if lids else int(gid)
            lid1 = max(lids) if lids else int(gid)
            zmin = min(float(x.get("z_min", 0.0) or 0.0) for x in gl)
            zmax = max(float(x.get("z_max", 0.0) or 0.0) for x in gl)

            best_angle = None
            best_key = None
            best_conf = 0.0
            best_w = 0.0
            for x in gl:
                conf = float(x.get("confidence", 0.0) or 0.0)
                w = float(x.get("total_weight", 0.0) or 0.0)
                a = x.get("angle_deg", None)
                if a is None:
                    continue
                key = (conf, w)
                if best_key is None or key > best_key:
                    best_key = key
                    best_angle = float(a)
                    best_conf = conf
                    best_w = w

            out.append((lid0, lid1, zmin, zmax, best_angle, best_conf, best_w))
        return out

    lines: list[str] = []
    lines.append("! Each block: select elements by centroid Z; create CM; optionally set ESYS.")
    for i, (lid0, lid1, zmin, zmax, angle, conf, w) in enumerate(_grouped()):
        name = f"L_{lid0:04d}" if group_size <= 1 else f"L_{lid0:04d}_{lid1:04d}"
        csid = cs_base + i
        a_out = float(angle) if angle is not None else 0.0
        conf_out = float(conf)
        lines.append("! ---- " + name + " ----")
        lines.append("ALLSEL,ALL")
        lines.append(f"ESEL,S,CENT,Z,{zmin:.6f},{max(zmin, zmax - 1e-9):.6f}")
        lines.append("*GET,S2S_NELEM,ELEM,0,COUNT")
        lines.append(f"CM,{name},ELEM")
        lines.append(f"*SET,S2S_NAME,'{name}'")
        lines.append(f"*VWRITE,S2S_NAME,{zmin:.6f},{zmax:.6f},S2S_NELEM,{a_out:.6f},{conf_out:.6f}")
        lines.append("(A20,',',F12.6,',',F12.6,',',I10,',',F10.4,',',F8.6)")
        lines.append(f"/COM, S2S {name}  Z={zmin:.6f}..{zmax:.6f}  (see ansys_mapdl_layers_report.txt)")

        if angle is not None and float(conf) >= float(min_conf):
            lines.append(f"! angle_deg={float(angle):.6f} conf={float(conf):.6f} weight={float(w):.6f}")
            lines.append(f"LOCAL,{csid},0,0,0,0,{float(angle):.6f},0,0")
            lines.append(f"EMODIF,ALL,ESYS,{csid}")
        else:
            lines.append(f"! No ESYS assigned (angle missing or conf<{min_conf:.3f}). conf={float(conf):.6f}")

    return _ANSYS_MAPDL_SCRIPT_TEMPLATE.format(
        min_conf=f"{min_conf:.6g}",
        group_size=str(group_size),
        cs_base=str(cs_base),
        layers_block="\n".join(lines),
    )


def _render_ansys_mechanical_script(cfg: JobConfig) -> str:
    header = (
        "# -*- coding: utf-8 -*-\n"
        "# Generated by slice2solid\n"
        f"MIN_CONFIDENCE = {float(cfg.ansys_min_confidence):.6g}\n"
        f"GROUP_SIZE_LAYERS = {int(cfg.ansys_group_size_layers)}\n"
        f"CREATE_NAMED_SELECTIONS = {str(bool(cfg.ansys_create_named_selections))}\n"
        f"CREATE_COORDINATE_SYSTEMS = {str(bool(cfg.ansys_create_coordinate_systems))}\n"
        "\n"
    )
    return header + _ANSYS_MECHANICAL_SCRIPT_TEMPLATE


def _render_ansys_mechanical_section_planes_script(cfg: JobConfig) -> str:
    # Default: Z=0 mm (user can edit Z_MM in the script).
    z_mm = 0.0
    return _ANSYS_MECHANICAL_SECTION_PLANES_TEMPLATE.format(z_mm=z_mm, export_images="False")


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
    part = _safe_filename_stem(Path(cfg.placed_stl).stem)
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
        "slice2solid: заметки для импорта/конвертации в CAD (универсально)\n"
        "\n"
        "Файлы в папке результата:\n"
        f" - {preview_stem}.stl            (mesh)\n"
        f" - {preview_stem}_mesh.ply       (mesh; alternative import)\n"
        " - voxel_points.csv              (point cloud from occupied voxels; x,y,z; no header; may be sampled)\n"
        " - metadata.json                 (параметры/статистика)\n"
        "\n"
        "Типовой путь в стороннем CAD/mesh-инструменте:\n"
        f" 1) Импортируйте mesh ({preview_stem}.stl / .ply) в единицах mm\n"
        " 2) При необходимости выполните Repair/Close Holes/Orient Normals\n"
        " 3) Если инструмент поддерживает: преобразуйте mesh/implicit в solid (B-Rep)\n"
        " 4) Экспортируйте STEP/Parasolid (или другой CAD-формат)\n"
        "\n"
        "Альтернатива (иногда лучше для решёток/заполнений):\n"
        " - Импорт point cloud (voxel_points.csv) -> построение implicit/volume -> затем solidify -> экспорт STEP\n"
        "\n"
        "Подсказка по шагу/разрешению (если инструмент просит spacing/resolution):\n"
        f" - slice2solid voxel_size_mm = {v:.3f}\n"
        f" - mesh effective voxel (after meshing downsample): {v_mesh:.3f} mm (ds={ds})\n"
        f" - starting spacing (from points/voxels): ~{suggested:.3f} mm (~ 0.5 * voxel_size)\n"
        f" - starting spacing (from mesh): ~{suggested_mesh:.3f} mm (~ 0.5 * mesh effective voxel)\n"
        "   Если слишком медленно: увеличьте spacing. Если теряются детали: уменьшите spacing.\n"
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
        self._auto_fit_btn = QtWidgets.QPushButton("Fit")
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
        self._radius: float = 1.0
        self._edge_rgba = (0.0, 0.0, 0.0, 0.18)

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

        if mesh is None or mesh.faces is None or mesh.vertices is None or len(mesh.faces) == 0:
            return

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if verts.size == 0 or faces.size == 0:
            return

        try:
            bounds = np.asarray(mesh.bounds, dtype=np.float32)
            center = bounds.mean(axis=0)
            verts = verts - center[None, :]
            ext = bounds[1] - bounds[0]
            radius = float(np.linalg.norm(ext) * 0.6 + 1e-6)
        except Exception:
            radius = 1.0
        self._radius = float(radius)

        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        try:
            meshdata.vertexNormals()
        except Exception:
            pass
        item = gl.GLMeshItem(
            meshdata=meshdata,
            smooth=True,
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
        self._clip_cb = QtWidgets.QCheckBox("Сечение")
        self._clip_cb.setChecked(False)
        self._clip_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._clip_slider.setRange(0, 1000)
        self._clip_slider.setValue(500)
        self._clip_slider.setEnabled(False)
        self._clip_slider.setToolTip("Плоскость сечения вдоль Z (для осмотра внутренности)")
        self._fit_btn = QtWidgets.QPushButton("Fit")
        self._fit_btn.setToolTip("Подогнать камеру под модель")
        hint = QtWidgets.QLabel("ЛКМ: вращение · колесо: зум · ПКМ: панорама")
        hint.setStyleSheet("color: #6B7280;")
        toolbar_layout.addWidget(self._edges_cb, 0)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self._light_bg_cb, 0)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self._clip_cb, 0)
        toolbar_layout.addWidget(self._clip_slider, 1)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self._fit_btn, 0)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(hint, 0)
        layout.addWidget(toolbar, 0)

        self._plot = QtInteractor(self)
        self._plot.set_background("#F3F4F6")
        layout.addWidget(self._plot.interactor, 1)

        self._poly: pv.PolyData | None = None
        self._actor = None
        self._bounds: np.ndarray | None = None

        self._edges_cb.toggled.connect(self._update_render)
        self._light_bg_cb.toggled.connect(self._update_render)
        self._clip_cb.toggled.connect(self._toggle_clip)
        self._clip_slider.valueChanged.connect(self._update_render)
        self._fit_btn.clicked.connect(self._fit_camera)

    def _toggle_clip(self, on: bool) -> None:
        self._clip_slider.setEnabled(bool(on))
        self._update_render()

    @staticmethod
    def _to_poly(mesh: trimesh.Trimesh) -> pv.PolyData:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if verts.size == 0 or faces.size == 0:
            return pv.PolyData()
        faces_vtk = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        poly = pv.PolyData(verts, faces_vtk)
        poly.clean(inplace=True)
        return poly

    def _fit_camera(self) -> None:
        try:
            self._plot.reset_camera()
        except Exception:
            pass

    def set_mesh(self, mesh: trimesh.Trimesh | None, *, stats_text: str = "", color: str = "#6BCB77") -> None:
        self._stats.setText(stats_text)
        self._plot.clear()
        self._poly = None
        self._bounds = None
        if mesh is None or mesh.faces is None or mesh.vertices is None or len(mesh.faces) == 0:
            self._plot.render()
            return
        poly = self._to_poly(mesh)
        self._poly = poly
        try:
            self._bounds = np.array(poly.bounds, dtype=float)
        except Exception:
            self._bounds = None
        self._color = color
        self._update_render()
        self._fit_camera()

    def _update_render(self) -> None:
        if self._poly is None:
            return
        poly = self._poly
        if self._clip_cb.isChecked() and self._bounds is not None:
            z0, z1 = float(self._bounds[4]), float(self._bounds[5])
            t = float(self._clip_slider.value()) / 1000.0
            z = z0 + t * (z1 - z0)
            try:
                poly = poly.clip(normal=(0, 0, 1), origin=(0, 0, z), invert=False)
            except Exception:
                poly = self._poly

        self._plot.clear()
        self._plot.add_mesh(
            poly,
            color=self._color,
            smooth_shading=True,
            show_edges=bool(self._edges_cb.isChecked()),
            edge_color="#1F2937" if self._light_bg_cb.isChecked() else "#E5E7EB",
            ambient=0.35 if self._light_bg_cb.isChecked() else 0.25,
            diffuse=0.8,
            specular=0.15,
            specular_power=20.0,
        )
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


class MeshCompareWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        backend = os.environ.get("S2S_PREVIEW_BACKEND", "auto").strip().lower()
        if backend == "vtk":
            view_cls = _MeshVTKView if _lazy_import_pyvista() else _Mesh2DView
        elif backend == "gl":
            view_cls = _Mesh3DView if gl is not None else _Mesh2DView
        elif backend == "2d":
            view_cls = _Mesh2DView
        else:
            # Default to pyqtgraph OpenGL when available: it's typically more robust than VTK in Qt layouts.
            if gl is not None:
                view_cls = _Mesh3DView
            elif _lazy_import_pyvista():
                view_cls = _MeshVTKView
            else:
                view_cls = _Mesh2DView
        self.before = view_cls(title="До (после marching cubes)")
        self.after = view_cls(title="После (после сглаживания)")
        layout.addWidget(self.before, 1)
        layout.addWidget(self.after, 1)

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

    def set_meshes(self, before: trimesh.Trimesh | None, after: trimesh.Trimesh | None, stats: dict | None = None) -> None:
        stats = stats or {}
        self.before.set_mesh(before, stats_text=self._fmt_stats(stats, "before"), color="#4D96FF")
        self.after.set_mesh(after, stats_text=self._fmt_stats(stats, "after"), color="#6BCB77")


_HELP_HTML = """
<h2>slice2solid - Справка</h2>

<p><b>Подсказки в интерфейсе:</b> наведите курсор на параметр, чтобы увидеть tooltip. Также можно нажать <b>Shift+F1</b> и кликнуть по элементу, чтобы открыть “What’s This?”.</p>

<h3>1) Что выбрать в Insight</h3>
<ol>
  <li><b>Simulation export (*.txt)</b>: Insight -&gt; Toolpaths -&gt; Simulation data export.</li>
  <li><b>Slicer job folder (ssys_*)</b>: папка задания, где лежат <code>toolpathParams.*</code>, <code>sliceParams.*</code> (нужно для auto-ширины дорожки).</li>
  <li><b>Placed STL (*.stl)</b>: STL после ориентации/размещения на столе в Insight.</li>
  <li><b>Папка результата (Output folder)</b>: сюда пишутся файлы.</li>
</ol>

<h3>2) Вкладка “CAD / Геометрия”</h3>
<ul>
  <li>Выход: <code>*_s2s_preview_structure.stl</code> + (опционально) <code>*_s2s_preview_structure_mesh.ply</code>, <code>voxel_points.csv</code>, <code>cad_import_notes.txt</code> + <code>metadata.json</code> (в имени mesh-файлов есть имя детали и параметры).</li>
  <li><b>Пресеты</b> — быстрые наборы настроек. Выберите пресет и нажмите <b>Применить</b>. При ручных изменениях режим становится <b>Custom</b>.</li>
  <li><b>Voxel size</b> - главный "качество&lt;-&gt;скорость". Меньше -&gt; точнее, но сильно тяжелее по RAM/времени и размеру STL.</li>
  <li><b>Downsample для meshing</b> — упрощает построение поверхности (marching cubes) по разреженному объёму (каждый N-й воксель): резко ускоряет meshing и уменьшает STL, но может “съесть” тонкие элементы.</li>
  <li><b>Volume smoothing</b> (sigma, vox) сглаживает сам воксельный объём перед построением поверхности — уменьшает “ступеньки”.</li>
  <li><b>Mesh smoothing</b> сглаживает уже готовую сетку (Laplacian) — убирает “рывки”, но может замыливать мелкие детали.</li>
  <li>Если появляются "лишние перемычки/линии" между отдельными дорожками: включите <b>Ignore travel jumps</b> (порог подбирается автоматически и зависит от bead radius) и/или увеличьте <b>Remove noise</b>.</li>
  <li>Если есть мелкие "островки" сетки вокруг: увеличьте <b>Remove mesh islands</b>.</li>
</ul>

<h4>2.0 Файлы результата: что чем является и что куда грузить</h4>
<table border="1" cellpadding="6" cellspacing="0">
  <tr><th>Файл</th><th>Что это</th><th>Когда использовать (CAD/CAE)</th></tr>
  <tr>
    <td><code>*_s2s_preview_structure.stl</code></td>
    <td>Mesh восстановленной <b>явной</b> структуры печати (периметры/заполнение). Обычно очень большой.</td>
    <td><b>CAD</b>: визуализация/архив/mesh-body, попытка конвертировать в B-Rep. <b>CAE</b>: только если нужна именно явная геометрия инфилла (иначе слишком тяжело).</td>
  </tr>
  <tr>
    <td><code>*_s2s_preview_structure_mesh.ply</code></td>
    <td>То же самое, что STL, но в PLY (часто быстрее/стабильнее импорт в некоторых инструментах).</td>
    <td><b>CAD/CAE</b>: пробовать вместо STL, если импорт STL проблемный/медленный.</td>
  </tr>
  <tr>
    <td><code>*_s2s_preview_structure_mesh_before.ply</code></td>
    <td>Служебный экспорт mesh <b>до</b> сглаживания/лечения.</td>
    <td>Для сравнения/диагностики, обычно не нужен для импорта.</td>
  </tr>
  <tr>
    <td><code>*_s2s_preview_structure_healed.stl</code></td>
    <td>Результат <b>Mesh Healer</b>: попытка сделать сетку более "чистой" для импорта.</td>
    <td>Выбирать, если исходный STL плохо импортируется (дырки/нормали/мусор).</td>
  </tr>
  <tr>
    <td><code>ansys_layers.json</code> / <code>ansys_layers.csv</code></td>
    <td>Ориентация/угол дорожек по слоям Z + confidence (для анизотропии).</td>
    <td><b>CAE (ANSYS)</b>: ключевой результат. Обычно используется вместе с геометрией детали (placed STL или CAD-solid) — без явного инфилла.</td>
  </tr>
  <tr>
    <td><code>ansys_mechanical_import_layers.py</code></td>
    <td>Скрипт импорта слоёв/ориентаций в ANSYS Mechanical.</td>
    <td><b>CAE (ANSYS)</b>: запускать в Mechanical, чтобы создать нужные сущности/настройки по слоям.</td>
  </tr>
  <tr>
    <td><code>voxel_points.csv</code></td>
    <td>Облако точек занятых вокселей (x,y,z), без заголовка.</td>
    <td><b>CAD</b>: иногда лучше для решёток (implicit/volume -&gt; solidify). <b>CAE</b>: редко, скорее для альтернативной реконструкции.</td>
  </tr>
  <tr>
    <td><code>cad_import_notes.txt</code></td>
    <td>Короткая памятка по импорту и стартовым параметрам spacing/resolution.</td>
    <td><b>CAD</b>: открыть и следовать рекомендациям.</td>
  </tr>
  <tr>
    <td><code>metadata.json</code></td>
    <td>Параметры запуска, матрица (STL-&gt;CMB), статистика mesh/вокселей.</td>
    <td>Для контроля единиц/матрицы/параметров, воспроизводимости и диагностики.</td>
  </tr>
</table>

<p><b>Как получить STEP/твердое тело в стороннем CAD/mesh-инструменте:</b><br/>
Импортируйте mesh (STL/PLY) -&gt; при необходимости Repair/Close/Orient Normals -&gt; затем (если поддерживается) Convert to Solid (B-Rep) -&gt; Export STEP.<br/>
Если включён <b>CAD bundle</b>, рядом с STL будет файл <code>cad_import_notes.txt</code> с подсказками по стартовым параметрам (spacing/resolution) на основе параметров slice2solid.</p>

<h4>2.1 Mesh Healer (CAD)</h4>
<ul>
  <li><b>Зачем:</b> некоторые CAD-системы плохо импортируют "грязные" STL (дырки, дубликаты, нулевые грани, проблемы ориентации). Mesh Healer пытается автоматически исправить типовые дефекты.</li>
  <li><b>Что делает (safe):</b> удаляет дубликаты вершин/граней, удаляет неиспользуемые вершины, удаляет нулевые грани, переориентирует грани, закрывает небольшие отверстия.</li>
  <li><b>Профиль:</b> <code>safe</code> (по умолчанию, без ремешинга/упрощения) и <code>aggressive</code> (доп. попытки убрать self-intersections; использовать только если safe не помогает).</li>
  <li><b>Порог дырок (мм):</b> <code>close_holes_max</code> задаёт максимальный размер дырок для закрытия. Примечание: в MeshLab/pymeshlab это обычно лимит по числу рёбер контура, поэтому мм переводятся в рёбра по оценке (это видно в JSON-отчёте).</li>
  <li><b>Выход:</b> рядом с STL появляется <code>*_healed.stl</code> (и опционально <code>*_healed_report.json</code>).</li>
</ul>

<h3>Быстрый гайд по параметрам (что крутить)</h3>
<table border="1" cellpadding="6" cellspacing="0">
  <tr><th>Параметр</th><th>Эффект</th><th>Плюсы</th><th>Минусы</th><th>Стартовые значения</th></tr>
  <tr>
    <td><b>CAD bundle</b></td>
    <td>Пишет доп. файлы для удобного импорта: <code>*.ply</code>, <code>voxel_points.csv</code>, <code>cad_import_notes.txt</code>.</td>
    <td>Упрощает импорт/подбор spacing, даёт point cloud для альтернативного восстановления.</td>
    <td>Доп. файлы в папке результата.</td>
    <td>Включено</td>
  </tr>
  <tr>
    <td><b>Mesh Healer (CAD)</b></td>
    <td>Автоматически исправляет типовые дефекты сетки после экспорта STL.</td>
    <td>Повышает шанс корректного импорта и "watertight" сетки.</td>
    <td>Может не помочь при очень сложной/самопересекающейся сетке; aggressive может удалять проблемные области.</td>
    <td>Выключено; включать при проблемах импорта</td>
  </tr>
  <tr>
    <td><b>Voxel size (mm)</b></td>
    <td>Размер ячейки сетки, из которой строится поверхность.</td>
    <td>Меньше → более гладкая/точная поверхность.</td>
    <td>Меньше → RAM/время растут очень резко (≈ кубически), STL тяжелее.</td>
    <td>0.10–0.25 (если грубо → 0.07 → 0.05)</td>
  </tr>
  <tr>
    <td><b>Bead radius limit</b></td>
    <td>Ограничивает “толщину” дорожки при вокселизации.</td>
    <td>Стабилизирует результат, убирает случайные завышения.</td>
    <td>Слишком мало → “худые” ребра/разрывы.</td>
    <td>Auto; вручную обычно 1.0–2.5 мм</td>
  </tr>
  <tr>
    <td><b>Ignore travel jumps</b></td>
    <td>Не заполнять материал по длинным перемещениям (travel) между разорванными траекториями.</td>
    <td>Убирает ложные перемычки.</td>
    <td>Если порог слишком строгий → могут появиться разрывы (редко).</td>
    <td>Включено (recommended)</td>
  </tr>
  <tr>
    <td><b>Remove noise (min voxels)</b></td>
    <td>Удаляет маленькие “пятна” вокселей до построения сетки.</td>
    <td>Убирает мусор, ускоряет marching cubes.</td>
    <td>Слишком много → можно потерять тонкие элементы.</td>
    <td>100–500</td>
  </tr>
  <tr>
    <td><b>Remove mesh islands (min faces)</b></td>
    <td>Удаляет мелкие куски сетки после построения поверхности.</td>
    <td>Убирает островки/пылинки.</td>
    <td>Слишком много → удалит полезные мелкие детали.</td>
    <td>1000–10000</td>
  </tr>
  <tr>
    <td><b>Volume smoothing (sigma, vox)</b></td>
    <td>Гауссово сглаживание объёма (в вокселях) перед marching cubes.</td>
    <td>Лучше “убирает ступеньки” без сильной потери формы.</td>
    <td>Слишком много → тонкие стенки могут “съесться”.</td>
    <td>0.8–1.5 (начать с 1.0)</td>
  </tr>
  <tr>
    <td><b>Mesh smoothing (iterations)</b></td>
    <td>Laplacian smoothing по вершинам после marching cubes.</td>
    <td>Убирает "рывки", делает поверхность приятнее для импорта/ремонта в CAD/mesh-инструментах.</td>
    <td>Слишком много → усадка/замыливание деталей.</td>
    <td>10–30 (начать с 15)</td>
  </tr>
  <tr>
    <td><b>Downsample для meshing</b></td>
    <td>Строит поверхность по разреженному объёму (каждый N-й воксель).</td>
    <td>Сильно ускоряет marching cubes и уменьшает STL.</td>
    <td>Может “съесть” тонкие элементы и огрубить поверхность.</td>
    <td>1; для ускорения 2–4</td>
  </tr>
</table>

<h3>3) Вкладка “Просмотр”</h3>
<ul>
  <li>Показывает результат <b>последнего запуска</b>: слева сетка сразу после marching cubes, справа — после сглаживания.</li>
  <li>Если сетка слишком большая, для интерактивности она автоматически прореживается (в статистике видно <code>preview: ... ds=N</code>).</li>
  <li>Если доступен VTK/pyvista: есть режим <b>Сечение</b> по Z и отображение <b>рёбер</b>; кнопка <b>Fit</b> подгоняет камеру.</li>
</ul>

<h3>4) Вкладка “ANSYS / CAE”</h3>
<ul>
  <li>Выход: <code>ansys_layers.json</code>, <code>ansys_layers.csv</code>, <code>ansys_mechanical_import_layers.py</code>, <code>ansys_mapdl_layers.mac</code>.</li>
  <li>Идея: назначить ортотропию по слоям (X вдоль печати, Z — build direction).</li>
  <li><b>Пресеты</b> на вкладке ANSYS меняют параметры генерируемого Mechanical-скрипта (группировка слоёв, порог confidence, создавать ли NS/CS).</li>
</ul>
<ol>
  <li>Откройте ANSYS Mechanical, импортируйте геометрию, сделайте <b>Mesh</b>.</li>
  <li>Mechanical → Automation → Scripting → <b>Run Script…</b></li>
  <li>Выберите <code>ansys_mechanical_import_layers.py</code> из папки результата.</li>
  <li>После выполнения появятся Named Selections <code>L_0000</code>, <code>L_0001</code>… и Coordinate Systems <code>CS_L_0000</code>… (если API доступен в вашей конфигурации).</li>
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
        self.setWindowTitle("slice2solid — Восстановление структуры (MVP)")
        self.setWindowIcon(_load_app_icon())
        self.resize(920, 640)

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

        help_menu = self.menuBar().addMenu("Справка")
        about_action = QtGui.QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        howto_action = QtGui.QAction("Как пользоваться", self)
        howto_action.triggered.connect(self._show_help)
        help_menu.addAction(howto_action)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        layout.addWidget(main_splitter, 1)

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
            "Цель: получить `*_s2s_preview_structure.stl`, который можно импортировать как mesh в CAD/CAE и, при необходимости,\n"
            "сконвертировать в твердое тело (STEP) средствами стороннего CAD/mesh-инструмента.\n"
            "Поддержки/подложка (`Type=0`) игнорируются; используется только траектория модели (`Type=1`).\n"
            "Подсказки: наведите курсор на параметр (или Shift+F1 → клик)."
            "Пайплайн: Import Mesh → Repair/Close/Orient Normals → (если поддерживается) Convert to Solid (B-Rep) → Export STEP."
        )
        header.setWordWrap(True)
        top_layout.addWidget(header)

        io_group = QtWidgets.QGroupBox("Шаг 1 - Входные файлы")
        io_form = QtWidgets.QFormLayout(io_group)
        top_layout.addWidget(io_group)

        self.sim_edit = QtWidgets.QLineEdit()
        self.sim_edit.setPlaceholderText(r"Например: ...\part-table-simulation-data.txt")
        self.sim_edit.setToolTip(
            "Файл экспорта Stratasys Insight: Toolpaths → Simulation data export.\n"
            "Внутри есть таблица X/Y/Z/Bead Area/Type/BeadMode и матрица STL↔CMB."
        )
        self.sim_btn = QtWidgets.QPushButton("Обзор…")
        sim_row = QtWidgets.QHBoxLayout()
        sim_row.addWidget(self.sim_edit, 1)
        sim_row.addWidget(self.sim_btn)
        io_form.addRow("Экспорт симуляции (*.txt):", sim_row)

        self.job_edit = QtWidgets.QLineEdit()
        self.job_edit.setPlaceholderText(r"Например: ...\ssys_part-table")
        self.job_edit.setToolTip(
            "Папка задания Stratasys Insight (ssys_*).\n"
            "Нужна для чтения `toolpathParams.*` и автоматического определения ширины дорожки."
        )
        self.job_btn = QtWidgets.QPushButton("Обзор…")
        job_row = QtWidgets.QHBoxLayout()
        job_row.addWidget(self.job_edit, 1)
        job_row.addWidget(self.job_btn)
        io_form.addRow("Папка задания (ssys_*):", job_row)

        self.stl_edit = QtWidgets.QLineEdit()
        self.stl_edit.setPlaceholderText(r"Например: ...\part-table.stl (placed STL)")
        self.stl_edit.setToolTip(
            "Placed STL — STL после ориентации/размещения на столе в Insight.\n"
            "Используется как эталонная геометрия/габариты для вокселизации и проверки координат."
        )
        self.stl_btn = QtWidgets.QPushButton("Обзор…")
        stl_row = QtWidgets.QHBoxLayout()
        stl_row.addWidget(self.stl_edit, 1)
        stl_row.addWidget(self.stl_btn)
        io_form.addRow("Placed STL (*.stl):", stl_row)

        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText(
            r"Папка результата (*_s2s_preview_structure.stl/ply, cad_import_notes.txt, metadata.json, ...)"
        )
        _set_help(
            self.out_edit,
            title="Папка результата",
            body="Папка, куда программа запишет результаты.",
            pros="Можно запускать несколько раз в разные папки и сравнивать параметры.",
            cons="Большие STL могут занимать много места.",
            tip="Для экспериментов создайте отдельную папку на каждый прогон (например, out_0p10_sigma1).",
        )
        self.out_btn = QtWidgets.QPushButton("Обзор…")
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(self.out_btn)
        io_form.addRow("Папка результата:", out_row)

        tabs = QtWidgets.QTabWidget()
        top_layout.addWidget(tabs, 1)

        # --- Tab: CAD / Geometry preview ---
        geometry_tab = QtWidgets.QWidget()
        tabs.addTab(geometry_tab, "CAD / Геометрия")
        geo_outer = QtWidgets.QVBoxLayout(geometry_tab)
        geo_outer.setContentsMargins(0, 0, 0, 0)

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
        geo_layout.setContentsMargins(0, 0, 0, 0)

        geo_intro = QtWidgets.QLabel(
            "Режим CAD: восстановление внутренней структуры и экспорт `*_s2s_preview_structure.stl`.\n"
            "Дальше: импорт в сторонний CAD/mesh-инструмент → repair/solidify (если нужно) → экспорт STEP."
        )
        geo_intro.setWordWrap(True)
        geo_layout.addWidget(geo_intro)

        geo_group = QtWidgets.QGroupBox("Параметры геометрии")
        geo_form = QtWidgets.QFormLayout(geo_group)
        geo_layout.addWidget(geo_group)

        self.export_geometry = QtWidgets.QCheckBox("Сгенерировать STL (предпросмотр) (*_s2s_preview_structure.stl)")
        self.export_geometry.setChecked(True)
        self.export_geometry.setToolTip("Отключите, если нужен только экспорт для ANSYS (карта слоёв).")
        self.export_geometry.setWhatsThis(self.export_geometry.toolTip())
        geo_form.addRow("Выходная геометрия:", self.export_geometry)

        self.export_bundle = QtWidgets.QCheckBox(
            "Экспортировать CAD bundle (PLY + voxel_points.csv + cad_import_notes.txt)"
        )
        self.export_bundle.setChecked(True)
        self.export_bundle.setToolTip(
            "Доп. универсальные файлы для внешних CAD/mesh-инструментов: PLY (mesh), voxel_points.csv (point cloud из вокселей)\n"
            "и cad_import_notes.txt (краткие подсказки по импорту/spacing на основе параметров slice2solid)."
        )
        self.export_bundle.setWhatsThis(self.export_bundle.toolTip())
        geo_form.addRow("CAD bundle:", self.export_bundle)

        heal_group = QtWidgets.QGroupBox("Mesh Healer (CAD)")
        heal_form = QtWidgets.QFormLayout(heal_group)
        geo_layout.addWidget(heal_group)

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
        self.heal_report_path_btn = QtWidgets.QPushButton("Обзор…")
        report_row = QtWidgets.QHBoxLayout()
        report_row.addWidget(self.heal_report_path_edit, 1)
        report_row.addWidget(self.heal_report_path_btn)
        heal_form.addRow("Файл отчёта:", report_row)

        self.voxel_size = QtWidgets.QDoubleSpinBox()
        self.voxel_size.setRange(0.05, 5.0)
        self.voxel_size.setSingleStep(0.05)
        self.voxel_size.setValue(0.25)
        self.voxel_size.setDecimals(3)
        _set_help(
            self.voxel_size,
            title="Размер вокселя (Voxel size), мм",
            body="Размер вокселя (мм): из этой сетки строится поверхность (marching cubes).",
            pros="Меньше → более гладкая/точная поверхность.",
            cons="Меньше → очень сильный рост RAM/времени и размера STL (приблизительно кубически).",
            tip="Если поверхность “ступеньками”: сначала включите сглаживание объёма (Volume smoothing, ≈1.0), и только потом уменьшайте voxel size.",
        )
        geo_form.addRow("Размер вокселя (мм):", self.voxel_size)

        self.auto_radius = QtWidgets.QCheckBox("Авто (из параметров слайсера)")
        self.auto_radius.setChecked(True)
        _set_help(
            self.auto_radius,
            title="Bead radius limit — Auto",
            body="Автоматически берёт радиус дорожки из параметров слайсера (папка ssys_*).",
            pros="Обычно даёт правильную толщину дорожек без ручной настройки.",
            cons="Если ssys_* не выбран/не распознан → auto недоступен.",
            tip="Если auto не сработал или есть “жирные” дорожки — снимите Auto и задайте лимит вручную.",
        )
        self.max_radius = QtWidgets.QDoubleSpinBox()
        self.max_radius.setRange(0.1, 10.0)
        self.max_radius.setSingleStep(0.1)
        self.max_radius.setValue(1.5)
        self.max_radius.setDecimals(2)
        self.max_radius.setEnabled(False)
        _set_help(
            self.max_radius,
            title="Bead radius limit (mm)",
            body="Ограничение максимального радиуса ‘сферы’ при вокселизации (из Bead Area).",
            pros="Убирает выбросы, делает толщину дорожек стабильнее.",
            cons="Слишком низко → тонкие стенки/ребра могут исчезнуть.",
            tip="Типичные значения: 1.0–2.5 мм. Если модель ‘разваливается’ — увеличьте.",
        )
        self.radius_hint = QtWidgets.QLabel("Авто: неизвестно (выберите папку ssys_*)")
        _set_help(
            self.radius_hint,
            title="Bead radius (auto) status",
            body="Подсказка, получилось ли определить радиус автоматически.",
            tip="Выберите папку ssys_* (Slicer job folder), чтобы auto стало доступно.",
        )
        radius_row = QtWidgets.QHBoxLayout()
        radius_row.addWidget(self.auto_radius)
        radius_row.addWidget(self.max_radius)
        radius_row.addWidget(self.radius_hint, 1)
        geo_form.addRow("Ограничение радиуса дорожки:", radius_row)

        self.estimate = QtWidgets.QLabel("Оценка: —")
        self.estimate.setWordWrap(True)
        _set_help(
            self.estimate,
            title="Оценка сетки",
            body="Прикидка размеров воксельной сетки и ожидаемой нагрузки.",
            tip="Если оценка ‘слишком большая’ — увеличьте Voxel size или ограничьте область/геометрию.",
        )
        geo_form.addRow("Оценка сетки:", self.estimate)

        # --- Presets ---
        self._applying_preset = False
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(["Custom", "Fast (draft)", "Balanced", "Quality"])
        self.preset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.preset_combo.setMinimumContentsLength(18)
        self.apply_preset_btn = QtWidgets.QPushButton("Применить")
        self.apply_preset_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.apply_preset_btn)
        preset_wrap = QtWidgets.QWidget()
        preset_wrap.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        preset_wrap.setLayout(preset_row)
        _set_help(
            self.preset_combo,
            title="Пресеты",
            body="Готовые наборы параметров для быстрого старта.",
            pros="Ускоряет настройку для новичков.",
            cons="Не учитывает все особенности детали/траектории.",
            tip="Выберите пресет и нажмите Применить. При ручных изменениях режим станет Custom.",
        )
        self.apply_preset_btn.setToolTip("Применить выбранный пресет к параметрам геометрии.")
        self.apply_preset_btn.setWhatsThis(self.apply_preset_btn.toolTip())
        geo_form.addRow("Пресеты:", preset_wrap)

        self.jump_filter = QtWidgets.QCheckBox("Игнорировать перемещения (travel jumps) между траекториями (рекомендуется)")
        self.jump_filter.setChecked(True)
        _set_help(
            self.jump_filter,
            title="Игнорирование перемещений (travel jumps)",
            body="Игнорирует длинные перемещения между разорванными траекториями (travel/jump), чтобы не ‘заливать’ материал по воздуху.",
            pros="Убирает ложные перемычки и внутренние ‘нитки’.",
            cons="Если траектория реально разорвана короткими прыжками, можно получить разрывы (редко).",
            tip="Обычно держите включённым. Если появились неожиданные дырки — попробуйте временно выключить и сравнить.",
        )
        geo_form.addRow("Фильтр траектории:", self.jump_filter)

        self.min_island = QtWidgets.QSpinBox()
        self.min_island.setRange(0, 10000)
        self.min_island.setValue(150)
        _set_help(
            self.min_island,
            title="Удаление шума (мин. вокселей)",
            body="Удаляет маленькие компоненты вокселей до построения сетки (3D связность).",
            pros="Убирает ‘мусор’ и ускоряет построение сетки.",
            cons="Слишком большое значение может удалить тонкие элементы.",
            tip="Начните с 150–300. Если вокруг много мелких точек — увеличьте; если теряются тонкие элементы — уменьшите.",
        )
        geo_form.addRow("Удаление шума (мин. вокселей):", self.min_island)

        self.min_mesh_faces = QtWidgets.QSpinBox()
        self.min_mesh_faces.setRange(0, 50_000_000)
        self.min_mesh_faces.setValue(2000)
        _set_help(
            self.min_mesh_faces,
            title="Удаление островков (мин. граней)",
            body="После построения поверхности удаляет куски сетки, у которых меньше указанного числа граней.",
            pros="Убирает мелкие ‘островки’/пылинки вокруг структуры.",
            cons="Слишком большое значение может удалить полезные мелкие детали.",
            tip="Если много мусора вокруг — увеличьте. Если пропадают нужные мелкие элементы — уменьшите.",
        )
        geo_form.addRow("Удаление островков (мин. граней):", self.min_mesh_faces)

        self.vol_sigma = QtWidgets.QDoubleSpinBox()
        self.vol_sigma.setRange(0.0, 5.0)
        self.vol_sigma.setSingleStep(0.1)
        self.vol_sigma.setValue(0.0)
        self.vol_sigma.setDecimals(2)
        _set_help(
            self.vol_sigma,
            title="Сглаживание объёма (sigma, vox)",
            body="Гауссово сглаживание воксельного объёма перед marching cubes (sigma в вокселях).",
            pros="Сильно уменьшает “ступеньки/пилу” без большого роста времени.",
            cons="Слишком большое sigma может ‘съесть’ тонкие стенки и сгладить мелкие детали.",
            tip="Для более гладкой поверхности начните с 1.0. Если тонкие элементы размываются — снизьте до 0.6–0.8.",
        )
        geo_form.addRow("Сглаживание объёма (sigma, vox):", self.vol_sigma)

        self.meshing_downsample = QtWidgets.QSpinBox()
        self.meshing_downsample.setRange(1, 64)
        self.meshing_downsample.setValue(1)
        _set_help(
            self.meshing_downsample,
            title="Downsample для meshing (factor)",
            body="Упрощает сетку ещё на этапе marching cubes: строит поверхность по разреженному объёму (каждый N-й воксель).",
            pros="Очень сильно уменьшает размер STL и ускоряет meshing (примерно ~N² по числу граней).",
            cons="Тонкие элементы могут исчезнуть; поверхность станет грубее.",
            tip="Начните с 2 или 4. Если детали теряются — уменьшайте. Если STL слишком большой — увеличивайте.",
        )
        geo_form.addRow("Downsample для meshing:", self.meshing_downsample)

        self.smooth = QtWidgets.QSpinBox()
        self.smooth.setRange(0, 200)
        self.smooth.setValue(0)
        _set_help(
            self.smooth,
            title="Сглаживание сетки (итерации)",
            body="Сглаживание уже готовой сетки (Laplacian).",
            pros="Убирает ‘рывки’ и делает поверхность приятнее для последующей постобработки/конвертации в CAD.",
            cons="Может вызывать усадку/замыливание деталей при больших значениях.",
            tip="10–30 обычно достаточно. Если форма начинает ‘плыть’ — уменьшите.",
        )
        geo_form.addRow("Сглаживание сетки (итерации):", self.smooth)

        geo_layout.addStretch(1)

        # --- Tab: Preview ---
        preview_tab = QtWidgets.QWidget()
        tabs.addTab(preview_tab, "Просмотр")
        prev_layout = QtWidgets.QVBoxLayout(preview_tab)
        prev_hint = QtWidgets.QLabel(
            "Просмотр результата последнего запуска.\n"
            "Слева: сетка сразу после marching cubes. Справа: после сглаживания.\n"
            "Если сетка слишком большая, для предпросмотра она автоматически прореживается.\n"
            "После закрытия программы можно заново загрузить meshes из папки результата."
        )
        prev_hint.setWordWrap(True)
        prev_layout.addWidget(prev_hint, 0)
        prev_btn_row = QtWidgets.QHBoxLayout()
        prev_layout.addLayout(prev_btn_row)
        self.preview_reload_btn = QtWidgets.QPushButton("Загрузить из папки результата")
        self.preview_open_after_btn = QtWidgets.QPushButton("Открыть mesh (после)")
        self.preview_open_before_btn = QtWidgets.QPushButton("Открыть mesh (до)")
        self.preview_open_folder_btn = QtWidgets.QPushButton("Открыть папку результата")
        prev_btn_row.addWidget(self.preview_reload_btn)
        prev_btn_row.addStretch(1)
        prev_btn_row.addWidget(self.preview_open_before_btn)
        prev_btn_row.addWidget(self.preview_open_after_btn)
        prev_btn_row.addWidget(self.preview_open_folder_btn)
        self.mesh_preview = MeshCompareWidget()
        prev_layout.addWidget(self.mesh_preview, 1)

        # --- Tab: ANSYS / CAE ---
        ansys_tab = QtWidgets.QWidget()
        tabs.addTab(ansys_tab, "ANSYS / CAE")
        ansys_outer = QtWidgets.QVBoxLayout(ansys_tab)
        ansys_outer.setContentsMargins(0, 0, 0, 0)

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
        ansys_layout.setContentsMargins(0, 0, 0, 0)

        ansys_intro = QtWidgets.QLabel(
            "Режим ANSYS: экспорт ориентации печати по слоям для назначения ортотропии в Mechanical.\n"
            "Выход: ansys_layers.json/csv + ansys_mechanical_import_layers.py + ansys_mapdl_layers.mac."
        )
        ansys_intro.setWordWrap(True)
        ansys_layout.addWidget(ansys_intro)

        ansys_group = QtWidgets.QGroupBox("Параметры CAE")
        ansys_form = QtWidgets.QFormLayout(ansys_group)
        ansys_layout.addWidget(ansys_group)

        self.export_cae = QtWidgets.QCheckBox("Экспортировать карту ориентации слоёв (ANSYS)")
        self.export_cae.setChecked(True)
        ansys_form.addRow("Выход CAE:", self.export_cae)

        # --- Mechanical script presets/options ---
        self._applying_ansys_preset = False
        self.ansys_preset_combo = QtWidgets.QComboBox()
        self.ansys_preset_combo.addItems(["Custom", "Detailed (per layer)", "Fast (group 5 layers)", "CS only (group 5)"])
        self.ansys_preset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.ansys_preset_combo.setMinimumContentsLength(22)
        self.apply_ansys_preset_btn = QtWidgets.QPushButton("Применить")
        self.apply_ansys_preset_btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        ansys_preset_row = QtWidgets.QHBoxLayout()
        ansys_preset_row.setContentsMargins(0, 0, 0, 0)
        ansys_preset_row.addWidget(self.ansys_preset_combo, 1)
        ansys_preset_row.addWidget(self.apply_ansys_preset_btn)
        ansys_preset_wrap = QtWidgets.QWidget()
        ansys_preset_wrap.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        ansys_preset_wrap.setLayout(ansys_preset_row)
        _set_help(
            self.ansys_preset_combo,
            title="Пресеты ANSYS",
            body="Наборы настроек для генерируемого Mechanical-скрипта.",
            pros="Ускоряет старт: можно уменьшить число Named Selections/CS за счёт группировки слоёв.",
            cons="Группировка снижает ‘детальность’ ориентации по высоте.",
            tip="Выберите пресет и нажмите Применить. При ручных изменениях режим станет Custom.",
        )
        self.apply_ansys_preset_btn.setToolTip("Применить выбранный пресет к параметрам Mechanical-скрипта.")
        self.apply_ansys_preset_btn.setWhatsThis(self.apply_ansys_preset_btn.toolTip())
        ansys_form.addRow("Пресеты:", ansys_preset_wrap)

        self.ansys_min_conf = QtWidgets.QDoubleSpinBox()
        self.ansys_min_conf.setRange(0.0, 1.0)
        self.ansys_min_conf.setSingleStep(0.05)
        self.ansys_min_conf.setDecimals(2)
        self.ansys_min_conf.setValue(0.20)
        _set_help(
            self.ansys_min_conf,
            title="Min confidence",
            body="Порог ‘confidence’ слоя для создания Coordinate System в Mechanical-скрипте.",
            pros="Отсекает слои с неопределённой/шумной ориентацией.",
            cons="Слишком высокий порог → больше пропусков CS по высоте.",
            tip="Обычно 0.15–0.30. Если CS создаются ‘странные’ — поднимите; если CS мало — опустите.",
        )
        ansys_form.addRow("Min confidence:", self.ansys_min_conf)

        self.ansys_group_layers = QtWidgets.QSpinBox()
        self.ansys_group_layers.setRange(1, 200)
        self.ansys_group_layers.setValue(1)
        _set_help(
            self.ansys_group_layers,
            title="Group size (layers)",
            body="Сколько слоёв объединять в одну Named Selection/CS в Mechanical-скрипте.",
            pros="Меньше объектов в дереве Mechanical → быстрее и удобнее.",
            cons="Потеря детальности ориентации по высоте.",
            tip="1 = по слоям. Для ускорения попробуйте 5 или 10.",
        )
        ansys_form.addRow("Group size (layers):", self.ansys_group_layers)

        self.ansys_create_ns = QtWidgets.QCheckBox("Create Named Selections (mesh elements)")
        self.ansys_create_ns.setChecked(True)
        _set_help(
            self.ansys_create_ns,
            title="Create Named Selections",
            body="Создаёт Named Selection (Mesh Elements) на каждый слой/группу по Z-диапазону.",
            pros="Удобно назначать материалы/постпроцессинг по высоте.",
            cons="Много слоёв → много объектов (может быть тяжело).",
            tip="Если Mechanical ‘тяжёлый’ — включите группировку или выключите NS, оставив только CS.",
        )
        ansys_form.addRow("Mechanical:", self.ansys_create_ns)

        self.ansys_create_cs = QtWidgets.QCheckBox("Create Coordinate Systems")
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
            "1) Импортируйте геометрию, сгенерируйте mesh.\n"
            "2) Mechanical → Automation → Scripting → Run Script…\n"
            "3) Для пути A: запустите ansys_mechanical_import_layers.py из папки результата.\n"
            "   Для пути B (рекомендуется): вставьте ansys_mapdl_layers.mac в Static Structural → Environment → Commands.\n"
        )
        ansys_hint.setWordWrap(True)
        ansys_layout.addWidget(ansys_hint)
        ansys_layout.addStretch(1)

        # --- Tab: Help ---
        help_tab = QtWidgets.QWidget()
        tabs.addTab(help_tab, "Справка")
        help_layout = QtWidgets.QVBoxLayout(help_tab)
        help_btn_row = QtWidgets.QHBoxLayout()
        self.about_btn = QtWidgets.QPushButton("О программе")
        self.howto_btn = QtWidgets.QPushButton("Как пользоваться")
        help_btn_row.addWidget(self.about_btn)
        help_btn_row.addWidget(self.howto_btn)
        help_btn_row.addStretch(1)
        help_layout.addLayout(help_btn_row)
        self.help_view = QtWidgets.QTextBrowser()
        self.help_view.setOpenExternalLinks(True)
        self.help_view.setHtml(_HELP_HTML)
        help_layout.addWidget(self.help_view, 1)

        # --- Run + Outputs (compact bottom panel) ---
        bottom_panel = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter.addWidget(bottom_panel)

        self.run_btn = QtWidgets.QPushButton("Запуск")
        self.run_btn.setMinimumHeight(36)
        bottom_layout.addWidget(self.run_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        bottom_layout.addWidget(self.progress)

        bottom_tabs = QtWidgets.QTabWidget()
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
        main_splitter.setSizes([1000, 260])

        self.thread: QtCore.QThread | None = None
        self.worker: Worker | None = None
        self._last_outputs: list[str] = []
        self._preview_before_path: str | None = None
        self._preview_after_path: str | None = None
        self._settings = QtCore.QSettings()

        self.sim_btn.clicked.connect(self._pick_sim)
        self.job_btn.clicked.connect(self._pick_job)
        self.stl_btn.clicked.connect(self._pick_stl)
        self.out_btn.clicked.connect(self._pick_out)
        self.run_btn.clicked.connect(self._run)
        self.auto_radius.toggled.connect(self._update_radius_widgets)
        self.sim_edit.textChanged.connect(self._recompute_auto_radius)
        self.job_edit.textChanged.connect(self._recompute_auto_radius)
        self.stl_edit.textChanged.connect(self._recompute_estimate)
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
        self.apply_preset_btn.clicked.connect(self._apply_selected_preset)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selection_changed)
        self.apply_ansys_preset_btn.clicked.connect(self._apply_selected_ansys_preset)
        self.about_btn.clicked.connect(self._show_about)
        self.howto_btn.clicked.connect(self._show_help)
        self.preview_open_folder_btn.clicked.connect(self._open_output_folder)
        self.preview_reload_btn.clicked.connect(lambda: self._load_last_run_from_output_dir(load_meshes=True))
        self.preview_open_after_btn.clicked.connect(self._open_preview_after)
        self.preview_open_before_btn.clicked.connect(self._open_preview_before)
        self.out_edit.editingFinished.connect(lambda: self._load_last_run_from_output_dir(load_meshes=False))

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

        self._restore_settings()
        self._update_step_widgets()
        self._update_preview_buttons()

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
            self._settings.setValue("paths/sim", self.sim_edit.text().strip())
            self._settings.setValue("paths/job", self.job_edit.text().strip())
            self._settings.setValue("paths/stl", self.stl_edit.text().strip())
            self._settings.setValue("paths/out", self.out_edit.text().strip())
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

            sim = self._settings.value("paths/sim", "", type=str) or ""
            job = self._settings.value("paths/job", "", type=str) or ""
            stl = self._settings.value("paths/stl", "", type=str) or ""
            out = self._settings.value("paths/out", "", type=str) or ""

            if sim and not self.sim_edit.text().strip():
                self.sim_edit.setText(sim)
            if job and not self.job_edit.text().strip():
                self.job_edit.setText(job)
            if stl and not self.stl_edit.text().strip():
                self.stl_edit.setText(stl)
            if out and not self.out_edit.text().strip():
                self.out_edit.setText(out)
        except Exception:
            return
        self._load_last_run_from_output_dir(load_meshes=False)

    def _update_preview_buttons(self) -> None:
        self.preview_open_before_btn.setEnabled(bool(self._preview_before_path))
        self.preview_open_after_btn.setEnabled(bool(self._preview_after_path))

    def _open_preview_after(self) -> None:
        if not self._preview_after_path:
            return
        try:
            os.startfile(self._preview_after_path)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _open_preview_before(self) -> None:
        if not self._preview_before_path:
            return
        try:
            os.startfile(self._preview_before_path)  # type: ignore[attr-defined]
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

        outputs = meta.get("outputs")
        if isinstance(outputs, list):
            try:
                self._set_outputs([str(x) for x in outputs])
            except Exception:
                pass

        before_path = None
        after_path = None
        if isinstance(outputs, list):
            for x in outputs:
                p = str(x)
                pl = p.lower()
                if pl.endswith("_mesh_before.ply") or pl.endswith("_before.ply"):
                    before_path = p
                if pl.endswith("_mesh.ply") and "_s2s_preview_structure" in pl:
                    after_path = p
                if pl.endswith(".ply") and "_before" not in pl and "_s2s_preview_structure" in pl and after_path is None:
                    after_path = p
                if pl.endswith(".stl") and "_s2s_preview_structure" in pl and after_path is None:
                    after_path = p

        if after_path is None:
            candidates = sorted(
                out_dir.glob("*_s2s_preview_structure*_mesh.ply"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if candidates:
                after_path = str(candidates[0])
        if after_path is None:
            # Backward compatibility: older runs used `{stem}.ply`.
            candidates = sorted(out_dir.glob("*_s2s_preview_structure*.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                after_path = str(candidates[0])
        if before_path is None:
            candidates = sorted(
                out_dir.glob("*_s2s_preview_structure*_mesh_before.ply"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if candidates:
                before_path = str(candidates[0])
        if before_path is None:
            # Backward compatibility: older runs used `{stem}_before.ply`.
            candidates = sorted(out_dir.glob("*_s2s_preview_structure*_before.ply"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                before_path = str(candidates[0])

        self._preview_before_path = before_path
        self._preview_after_path = after_path
        self._update_preview_buttons()

        if not load_meshes or after_path is None:
            return
        try:
            after_mesh = trimesh.load_mesh(after_path, force="mesh")
        except Exception:
            return
        before_mesh = None
        if before_path is not None:
            try:
                before_mesh = trimesh.load_mesh(before_path, force="mesh")
            except Exception:
                before_mesh = None
        if before_mesh is None:
            before_mesh = after_mesh

        stats = {
            "before": {"vertices": int(before_mesh.vertices.shape[0]), "faces": int(before_mesh.faces.shape[0])},
            "after": {"vertices": int(after_mesh.vertices.shape[0]), "faces": int(after_mesh.faces.shape[0])},
        }
        try:
            self.mesh_preview.set_meshes(before_mesh, after_mesh, stats)
        except Exception:
            pass

    def _pick_sim(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите экспорт симуляции", "", "Text (*.txt)")
        if path:
            self.sim_edit.setText(path)
            self._auto_fill_from_sim(path)

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

    def _mark_ansys_preset_custom(self, *_args: object) -> None:
        if getattr(self, "_applying_ansys_preset", False):
            return
        if self.ansys_preset_combo.currentText() != "Custom":
            self.ansys_preset_combo.setCurrentText("Custom")

    def _on_preset_selection_changed(self, *_args: object) -> None:
        # Do not auto-apply on selection to avoid surprising changes; user clicks Apply.
        pass

    def _apply_selected_preset(self) -> None:
        preset = self.preset_combo.currentText()
        if preset == "Custom":
            return

        presets = {
            "Fast (draft)": dict(
                voxel_size_mm=0.25,
                volume_smooth_sigma_vox=0.0,
                smooth_iterations=0,
                min_component_voxels=150,
                min_mesh_component_faces=2000,
            ),
            "Balanced": dict(
                voxel_size_mm=0.10,
                volume_smooth_sigma_vox=1.0,
                smooth_iterations=15,
                min_component_voxels=150,
                min_mesh_component_faces=2000,
            ),
            "Quality": dict(
                voxel_size_mm=0.05,
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
            self.vol_sigma.setValue(float(cfg["volume_smooth_sigma_vox"]))
            self.smooth.setValue(int(cfg["smooth_iterations"]))
            self.min_island.setValue(int(cfg["min_component_voxels"]))
            self.min_mesh_faces.setValue(int(cfg["min_mesh_component_faces"]))
        finally:
            self._applying_preset = False

        self._recompute_estimate()

    def _apply_selected_ansys_preset(self) -> None:
        preset = self.ansys_preset_combo.currentText()
        if preset == "Custom":
            return

        presets = {
            "Detailed (per layer)": dict(group_size_layers=1, min_conf=0.2, create_ns=True, create_cs=True),
            "Fast (group 5 layers)": dict(group_size_layers=5, min_conf=0.2, create_ns=True, create_cs=True),
            "CS only (group 5)": dict(group_size_layers=5, min_conf=0.2, create_ns=False, create_cs=True),
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
    def _pick_job(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку задания ssys_*")
        if path:
            self.job_edit.setText(path)
            self._auto_fill_from_job(path)

    def _pick_stl(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите placed STL", "", "STL (*.stl *.STL)")
        if path:
            self.stl_edit.setText(path)

    def _pick_out(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку результата")
        if path:
            self.out_edit.setText(path)

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
            "1) Выберите Simulation export (*.txt):\n"
            "   Insight -> Toolpaths -> Simulation data export.\n\n"
            "2) Выберите Slicer job folder (ssys_*):\n"
            "   Нужен для авто-определения ширины дорожки / bead radius.\n\n"
            "3) Выберите Placed STL (*.stl):\n"
            "   STL после размещения на столе (placed).\n\n"
            "4) Выберите папку результата и нажмите Запуск.\n\n"
            "После запуска:\n"
            " - вкладка 'Просмотр' показывает сетку до/после сглаживания\n"
            " - блок 'Результаты' позволяет открыть файлы/папку и скопировать путь\n\n"
            "Основные выходные файлы:\n"
            " - Для CAE (ANSYS): обычно берут геометрию детали (placed STL или CAD-solid) + ansys_layers.*\n"
            " - *_s2s_preview_structure.stl (если включён экспорт геометрии)\n"
            " - metadata.json (параметры/матрица/статистика)\n"
            " - ansys_layers.json/csv + ansys_mechanical_import_layers.py + ansys_mapdl_layers.mac (если включён экспорт ANSYS)\n"
            " - *_s2s_preview_structure_mesh.ply, voxel_points.csv, cad_import_notes.txt (если включён CAD bundle)\n\n"
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

    def _auto_fill_from_sim(self, sim_path: str) -> None:
        p = Path(sim_path)
        # auto output folder
        if not self.out_edit.text().strip():
            self.out_edit.setText(str(p.parent / "slice2solid_out"))

        # try infer ssys_* folder from proximity
        if not self.job_edit.text().strip():
            candidates: list[Path] = []
            for base in [p.parent, p.parent.parent]:
                if base is None or not base.exists():
                    continue
                for d in base.iterdir():
                    if d.is_dir() and d.name.lower().startswith("ssys_"):
                        if (d / "toolpathParams.cur").exists() or (d / "toolpathParams.new").exists():
                            candidates.append(d)
            if candidates:
                self.job_edit.setText(str(candidates[0]))

        self._recompute_auto_radius()
        self._recompute_estimate()

    def _auto_fill_from_job(self, job_dir: str) -> None:
        # try infer placed STL from sjb
        if not self.stl_edit.text().strip():
            p = infer_stl_path_from_job(job_dir)
            if p is not None:
                self.stl_edit.setText(str(p))
        self._recompute_auto_radius()
        self._recompute_estimate()

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
        sim_path = self.sim_edit.text().strip()
        if not job_dir:
            self.radius_hint.setText("Авто: неизвестно (выберите папку ssys_*)")
            return
        if not Path(job_dir).exists():
            self.radius_hint.setText("Авто: папка не найдена")
            return

        try:
            header = None
            if sim_path and Path(sim_path).exists():
                header, _it = read_simulation_export(sim_path)
            params = load_job_params(job_dir)
            bead_w = estimate_bead_width_mm(params, sim_slice_height_mm=header.slice_height_mm if header else None)
            if bead_w is None:
                self.radius_hint.setText("Авто: не удалось определить ширину дорожки")
                return
            r = bead_w / 2.0
            self.radius_hint.setText(f"Авто: ширина≈{bead_w:.3f} мм → радиус≈{r:.3f} мм")
        except Exception as e:
            self.radius_hint.setText(f"Авто: ошибка ({e})")

    def _recompute_estimate(self) -> None:
        if hasattr(self, "export_geometry") and not self.export_geometry.isChecked():
            self.estimate.setText("Оценка: геометрия отключена (только ANSYS/CAE).")
            return
        stl_path = self.stl_edit.text().strip()
        if not stl_path or not Path(stl_path).exists():
            self.estimate.setText("Оценка: выберите Placed STL, чтобы посчитать габариты/сетку.")
            return
        try:
            mesh = trimesh.load_mesh(stl_path, force="mesh")
            bmin = mesh.bounds[0]
            bmax = mesh.bounds[1]
            v = float(self.voxel_size.value())
            size = bmax - bmin
            nx, ny, nz = (int(np.ceil(s / v)) + 1 for s in size)
            voxels = nx * ny * nz
            approx_mb = voxels / (1024 * 1024)  # bool ~1 byte worst-case
            self.estimate.setText(
                f"bbox≈{size[0]:.1f}×{size[1]:.1f}×{size[2]:.1f} мм; сетка≈{nx}×{ny}×{nz} ({voxels:,} вокселей), "
                f"память грубо ~{approx_mb:.1f} MB (+ оверхед)."
            )
        except Exception as e:
            self.estimate.setText(f"Оценка: ошибка чтения STL ({e})")

    def _run(self) -> None:
        sim = self.sim_edit.text().strip()
        job_dir = self.job_edit.text().strip() or None
        stl = self.stl_edit.text().strip()
        out = self.out_edit.text().strip()
        do_geo = bool(self.export_geometry.isChecked())
        do_bundle = bool(self.export_bundle.isChecked()) and do_geo
        do_cae = bool(self.export_cae.isChecked())
        if not sim or not out or (do_geo and not stl):
            QtWidgets.QMessageBox.warning(
                self,
                "Missing input",
                "Please select simulation export and output folder. Placed STL is required for geometry export.",
            )
            return
        if not do_geo and not do_cae:
            QtWidgets.QMessageBox.warning(self, "Нечего делать", "Включите выходную геометрию и/или экспорт для ANSYS.")
            return

        max_r: float | None
        max_jump: float | None
        header = None
        if self.jump_filter.isChecked() or (do_geo and self.auto_radius.isChecked()):
            header, _it = read_simulation_export(sim)

        params = None
        bead_w: float | None = None
        thresholds = None

        if do_geo:
            if self.auto_radius.isChecked():
                if not job_dir:
                    QtWidgets.QMessageBox.warning(
                        self, "Missing job folder", "Auto radius requires selecting the ssys_* folder (toolpathParams)."
                    )
                    return
                params = load_job_params(job_dir)
                bead_w = estimate_bead_width_mm(params, sim_slice_height_mm=header.slice_height_mm if header else None)
                if bead_w is None:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Auto radius failed",
                        "Could not infer bead width from slicer params. Set radius manually or check job folder.",
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

        cfg = JobConfig(
            simulation_txt=sim,
            job_dir=job_dir,
            placed_stl=stl,
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
        self._append_log("Starting…")
        self.open_out_btn.setEnabled(False)
        try:
            self.mesh_preview.set_meshes(None, None, {})
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
        try:
            self.mesh_preview.set_meshes(before, after, stats if isinstance(stats, dict) else {})
        except Exception:
            pass

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
    w = MainWindow()
    w.show()
    try:
        app.processEvents()
        w.ensure_visible_on_screen()
    except Exception:
        pass
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
