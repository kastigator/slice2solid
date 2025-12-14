from __future__ import annotations

import html
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import trimesh
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from slice2solid.core.insight_simulation import (
    invert_rowvec_matrix,
    read_simulation_export,
    transform_points_rowvec,
)
from slice2solid.core.insight_params import estimate_bead_width_mm, infer_stl_path_from_job, load_job_params
from slice2solid.core.cae_orientation import compute_layer_orientations
from slice2solid.core.ntop_bundle import export_voxel_centers_csv
from slice2solid.core.voxelize import mesh_from_voxels_configured, voxelize_toolpath
from slice2solid.app_info import APP_DISPLAY_NAME, AUTHOR, CONTACT_EMAIL, DEPARTMENT, ORGANIZATION, VERSION


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
    smooth_iterations: int
    export_cae_layers: bool
    export_geometry_preview: bool
    export_ntop_bundle: bool = True
    ansys_min_confidence: float = 0.2
    ansys_group_size_layers: int = 1
    ansys_create_named_selections: bool = True
    ansys_create_coordinate_systems: bool = True


class Worker(QtCore.QObject):
    progress = QtCore.Signal(int)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str, object)

    def __init__(self, cfg: JobConfig):
        super().__init__()
        self.cfg = cfg

    @QtCore.Slot()
    def run(self) -> None:
        try:
            t0 = time.time()
            sim_header, rows_iter = read_simulation_export(self.cfg.simulation_txt)
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

            self.log.emit("Подготовка преобразования координат (CMB → placed STL)…")
            stl_to_cmb = sim_header.stl_to_cmb
            cmb_to_stl = invert_rowvec_matrix(stl_to_cmb)

            self.log.emit("Чтение точек траектории (Type=1)…")
            # Transform on the fly: we will buffer points to transform in chunks for speed.
            buffer_xyz: list[list[float]] = []
            buffer_area: list[float] = []
            buffer_time: list[float] = []
            buffer_factor: list[float] = []
            buffer_type: list[int] = []
            buffer_mode: list[int] = []

            transformed_points = []
            total = 0
            kept = 0
            chunk_size = 200_000

            def flush() -> None:
                nonlocal transformed_points, kept
                if not buffer_xyz:
                    return
                arr = np.array(buffer_xyz, dtype=float)
                out = transform_points_rowvec(arr, cmb_to_stl)
                for i in range(out.shape[0]):
                    transformed_points.append(
                        (
                            float(out[i, 0]),
                            float(out[i, 1]),
                            float(out[i, 2]),
                            float(buffer_time[i]),
                            float(buffer_area[i]),
                            float(buffer_factor[i]),
                            int(buffer_type[i]),
                            int(buffer_mode[i]),
                        )
                    )
                kept += out.shape[0]
                buffer_xyz.clear()
                buffer_area.clear()
                buffer_time.clear()
                buffer_factor.clear()
                buffer_type.clear()
                buffer_mode.clear()

            for pt in rows_iter:
                total += 1
                if pt.type != 1:
                    continue
                buffer_xyz.append([pt.x, pt.y, pt.z])
                buffer_time.append(pt.time_s)
                buffer_area.append(pt.bead_area)
                buffer_factor.append(pt.factor)
                buffer_type.append(pt.type)
                buffer_mode.append(pt.bead_mode)

                if len(buffer_xyz) >= chunk_size:
                    flush()
                    self.progress.emit(min(30, int(30 * kept / max(kept, 1))))

            flush()

            self.log.emit(f"Type=1 points: {kept} (from {total} rows).")

            if self.cfg.export_cae_layers and kept >= 2:
                self.log.emit("Computing per-layer print orientation (CAE export)…")
                arr_xyz = np.array([[p[0], p[1], p[2]] for p in transformed_points], dtype=float)
                areas = np.array([p[4] for p in transformed_points], dtype=float)
                z0 = float(np.min(arr_xyz[:, 2])) if arr_xyz.size else 0.0
                layers = compute_layer_orientations(
                    arr_xyz,
                    slice_height_mm=float(slice_h),
                    z0_mm=z0,
                    max_jump_mm=self.cfg.max_jump_mm,
                    weights=areas,
                )
            else:
                layers = []

            preview_mesh = None
            vox = None
            if self.cfg.export_geometry_preview:
                # Build iterable of ToolpathPoint-like tuples for voxelization
                class _PT:
                    __slots__ = ("x", "y", "z", "time_s", "bead_area", "factor", "type", "bead_mode")

                    def __init__(self, t):
                        self.x, self.y, self.z, self.time_s, self.bead_area, self.factor, self.type, self.bead_mode = t

                pts_iter = (_PT(t) for t in transformed_points)

                # Expand bounds slightly to avoid clipping
                pad = float(self.cfg.max_radius_mm or 0.0) * 2.0
                bmin = bounds_min - pad
                bmax = bounds_max + pad

                self.log.emit("Вокселизация…")
                self.progress.emit(35)
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
                self.progress.emit(70)

                self.log.emit("Построение сетки (marching cubes)…")
                preview_mesh = mesh_from_voxels_configured(
                    vox,
                    volume_smooth_sigma_vox=self.cfg.volume_smooth_sigma_vox,
                    min_component_faces=self.cfg.min_mesh_component_faces,
                )
                if self.cfg.smooth_iterations > 0:
                    self.log.emit(f"Сглаживание сетки ({self.cfg.smooth_iterations} итераций)…")
                    trimesh.smoothing.filter_laplacian(preview_mesh, iterations=int(self.cfg.smooth_iterations))
                self.progress.emit(85)

            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Prefix geometry outputs to reduce confusion with the user-provided "placed STL".
            preview_stem = _preview_mesh_stem(self.cfg)
            out_stl = out_dir / f"{preview_stem}.stl"
            out_ply = out_dir / f"{preview_stem}.ply"
            out_recipe = out_dir / "ntop_recipe.txt"
            out_points = out_dir / "ntop_points.csv"
            out_json = out_dir / "metadata.json"
            out_layers_json = out_dir / "ansys_layers.json"
            out_layers_csv = out_dir / "ansys_layers.csv"
            out_ansys_script = out_dir / "ansys_mechanical_import_layers.py"

            outputs: list[str] = []
            ntop_written = False

            if self.cfg.export_geometry_preview and preview_mesh is not None:
                self.log.emit(f"Запись {out_stl}…")
                preview_mesh.export(out_stl)
                outputs.append(str(out_stl))
                if self.cfg.export_ntop_bundle:
                    try:
                        self.log.emit(f"Запись {out_ply}…")
                        preview_mesh.export(out_ply)
                        outputs.append(str(out_ply))

                        recipe = _render_ntop_recipe(self.cfg)
                        out_recipe.write_text(recipe, encoding="utf-8")
                        outputs.append(str(out_recipe))

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
                            self.log.emit(f"nTop точки: {res.points_written:,}/{res.points_total:,} ({note})")
                            outputs.append(str(out_points))
                        ntop_written = True
                    except Exception as e:
                        self.log.emit(f"nTop bundle пропущен: {e}")

            if self.cfg.export_cae_layers:
                self.log.emit(f"Запись {out_layers_json}…")
                out_layers_json.write_text(
                    json.dumps(
                        {
                            "slice_height_mm": float(slice_h),
                            "z0_mm": float(np.min(arr_xyz[:, 2])) if kept else None,
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

            meta = {
                "inputs": asdict(self.cfg),
                "simulation_header": sim_header.raw,
                "stl_to_cmb_matrix": sim_header.stl_to_cmb.tolist(),
                "outputs": outputs,
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
            out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            outputs.append(str(out_json))

            self.progress.emit(100)
            extra = ""
            if self.cfg.export_cae_layers:
                extra = f", {out_layers_json.name}, {out_layers_csv.name}, {out_ansys_script.name}"
            if self.cfg.export_geometry_preview:
                ntop_part = f", {out_ply.name}, {out_recipe.name}" if ntop_written else ""
                if ntop_written:
                    ntop_part = f", {out_ply.name}, {out_points.name}, {out_recipe.name}"
                base = f"{out_stl.name}{ntop_part}, {out_json.name}{extra}"
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


_ANSYS_MECHANICAL_SCRIPT_TEMPLATE = r'''# slice2solid → ANSYS Mechanical (Workbench) import helper
#
# Tested target: ANSYS 2025 R2 (Mechanical scripting).
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
#   Mechanical → Automation → Scripting → Run Script… → select this .py file.
#
# Expected files in the SAME folder:
#   - ansys_layers.json
#
#
# This file is auto-generated by slice2solid. You can tweak the CONFIG values below and re-run in Mechanical.
#
import json
import bisect
import math
import os

HERE = os.path.dirname(__file__)
LAYERS_JSON = os.path.join(HERE, "ansys_layers.json")

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
    # Some versions expose MeshData at different locations.
    if hasattr(model, "MeshData"):
        return model.MeshData
    if hasattr(ExtAPI, "DataModel") and hasattr(ExtAPI.DataModel, "Project"):
        m = ExtAPI.DataModel.Project.Model
        if hasattr(m, "MeshData"):
            return m.MeshData
    raise RuntimeError("Could not find MeshData on Model. Please report your ANSYS version + API error.")


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
    raise RuntimeError("Could not read element centroid for element id={}".format(eid))


def _create_named_selection_by_ids(model, name, ids):
    # Create a MeshElements selection and assign to a Named Selection.
    sel = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshElements)
    sel.Ids = list(ids)
    ns = model.AddNamedSelection()
    ns.Name = name
    ns.Location = sel
    return ns


def _create_coordinate_system(model, name, origin_xyz, x_axis_xyz, z_axis_xyz=(0.0, 0.0, 1.0)):
    # Create a coordinate system with specified axes.
    x = _unit(x_axis_xyz)
    z = _unit(z_axis_xyz)
    y = _unit(_cross(z, x))
    x = _unit(_cross(y, z))  # re-orthogonalize

    cs = model.CoordinateSystems.AddCoordinateSystem()
    cs.Name = name
    cs.OriginX = float(origin_xyz[0])
    cs.OriginY = float(origin_xyz[1])
    cs.OriginZ = float(origin_xyz[2])
    cs.XAxisX = float(x[0])
    cs.XAxisY = float(x[1])
    cs.XAxisZ = float(x[2])
    cs.YAxisX = float(y[0])
    cs.YAxisY = float(y[1])
    cs.YAxisZ = float(y[2])
    cs.ZAxisX = float(z[0])
    cs.ZAxisY = float(z[1])
    cs.ZAxisZ = float(z[2])
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

    with open(LAYERS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data.get("layers", [])
    if not layers:
        print("No layers found in ansys_layers.json")
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
        z_min = min(float(x.get("z_min", 0.0)) for x in glayers)
        z_max = max(float(x.get("z_max", 0.0)) for x in glayers)

        # Find elements in Z band using binary search.
        i0 = bisect.bisect_left(zs, z_min)
        i1 = bisect.bisect_left(zs, z_max)
        band_eids = eids_sorted[i0:i1]
        if not band_eids:
            continue

        name = "L_{:04d}".format(lid0) if int(GROUP_SIZE_LAYERS) <= 1 else "L_{:04d}_{:04d}".format(lid0, lid1)

        if bool(CREATE_NAMED_SELECTIONS):
            _create_named_selection_by_ids(model, name, band_eids)
            created_ns += 1

        if bool(CREATE_COORDINATE_SYSTEMS):
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
            _create_coordinate_system(model, "CS_" + name, (cx, cy, cz), (float(d[0]), float(d[1]), float(d[2])))
            created_cs += 1

    print("Created Named Selections:", created_ns)
    print("Created Coordinate Systems:", created_cs)
    print("")
    print("Next step in Mechanical:")
    print(" - Assign orthotropic material to the body")
    print(" - Use the created coordinate systems (CS_L_XXXX) as material orientation references")


main()
'''


def _render_ansys_mechanical_script(cfg: JobConfig) -> str:
    header = (
        "# Generated by slice2solid\n"
        f"MIN_CONFIDENCE = {float(cfg.ansys_min_confidence):.6g}\n"
        f"GROUP_SIZE_LAYERS = {int(cfg.ansys_group_size_layers)}\n"
        f"CREATE_NAMED_SELECTIONS = {str(bool(cfg.ansys_create_named_selections))}\n"
        f"CREATE_COORDINATE_SYSTEMS = {str(bool(cfg.ansys_create_coordinate_systems))}\n"
        "\n"
    )
    return header + _ANSYS_MECHANICAL_SCRIPT_TEMPLATE


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
    it = int(cfg.smooth_iterations)
    return f"{part}_vox{vox}_sig{sig}_it{it}_s2s_preview_structure"


def _render_ntop_recipe(cfg: JobConfig) -> str:
    v = float(cfg.voxel_size_mm)
    suggested = max(0.5 * v, 0.02)
    preview_stem = _preview_mesh_stem(cfg)
    return (
        "slice2solid → nTop: рекомендуемый workflow\n"
        "\n"
        "Файлы в папке результата:\n"
        f" - {preview_stem}.stl  (mesh)\n"
        f" - {preview_stem}.ply  (mesh; alternative import)\n"
        " - ntop_points.csv        (point cloud from occupied voxels; format: x, y, z; no header; may be sampled)\n"
        " - metadata.json          (параметры/статистика)\n"
        "\n"
        "Шаги в nTop (типовой путь):\n"
        f" 1) Utilities → Import Mesh → {preview_stem}.stl (Units: mm)\n"
        " 2) Search: \"Implicit Body from Mesh\" → convert mesh to implicit\n"
        " 3) (Optional) Smooth/Close/Repair on the implicit body\n"
        " 4) Convert implicit to CAD/solid body\n"
        " 5) Export STEP\n"
        "\n"
        "Альтернатива (иногда лучше для решёток/заполнений):\n"
        " - Import point list (CSV) → create implicit from points → then Convert to CAD/Solid → Export STEP\n"
        "\n"
        "Стартовые параметры (подсказка):\n"
        f" - slice2solid voxel_size_mm = {v:.3f}\n"
        f" - nTop implicit resolution/spacing: ~{suggested:.3f} mm (≈ 0.5 * voxel_size)\n"
        "   Если слишком медленно: увеличьте spacing. Если теряются детали: уменьшите spacing.\n"
    )


_HELP_HTML = """
<h2>slice2solid — Справка</h2>

<p><b>Подсказки в интерфейсе:</b> наведите курсор на параметр, чтобы увидеть tooltip. Также можно нажать <b>Shift+F1</b> и кликнуть по элементу, чтобы открыть “What’s This?”.</p>

<h3>1) Что выбрать в Insight</h3>
<ol>
  <li><b>Simulation export (*.txt)</b>: Insight → Toolpaths → Simulation data export.</li>
  <li><b>Slicer job folder (ssys_*)</b>: папка задания, где лежат <code>toolpathParams.*</code>, <code>sliceParams.*</code> (нужно для auto-ширины дорожки).</li>
  <li><b>Placed STL (*.stl)</b>: STL после ориентации/размещения на столе в Insight.</li>
  <li><b>Папка результата (Output folder)</b>: сюда пишутся файлы.</li>
</ol>

<h3>2) Вкладка “nTop / Geometry”</h3>
<ul>
  <li>Выход: <code>*_s2s_preview_structure.stl</code> + (опционально) <code>*_s2s_preview_structure.ply</code>, <code>ntop_points.csv</code>, <code>ntop_recipe.txt</code> + <code>metadata.json</code> (в имени mesh-файлов есть имя детали и параметры).</li>
  <li><b>Пресеты</b> — быстрые наборы настроек. Выберите пресет и нажмите <b>Применить</b>. При ручных изменениях режим становится <b>Custom</b>.</li>
  <li><b>Voxel size</b> — главный “качество↔скорость”. Меньше → точнее, но сильно тяжелее по RAM/времени и размеру STL.</li>
  <li><b>Volume smoothing</b> (sigma, vox) сглаживает сам воксельный объём перед построением поверхности — уменьшает “ступеньки”.</li>
  <li><b>Mesh smoothing</b> сглаживает уже готовую сетку (Laplacian) — убирает “рывки”, но может замыливать мелкие детали.</li>
  <li>Если появляются "лишние перемычки/линии" между отдельными дорожками: включите <b>Ignore travel jumps</b> и/или уменьшите <b>max jump</b> (если доступно), увеличьте <b>Remove noise</b>.</li>
  <li>Если есть мелкие "островки" сетки вокруг: увеличьте <b>Remove mesh islands</b>.</li>
</ul>

<p><b>Как получить гладкий STEP в nTop (рекомендуется для решёток/заполнений):</b><br/>
Utilities → <b>Import Mesh</b> → затем <b>Implicit Body from Mesh</b> → (по ситуации: Smooth/Close/Repair) → Convert to CAD/Solid → Export STEP.<br/>
Если включён <b>nTop bundle</b>, рядом с STL будет файл <code>ntop_recipe.txt</code> с подсказками по стартовым параметрам.</p>

<h3>Быстрый гайд по параметрам (что крутить)</h3>
<table border="1" cellpadding="6" cellspacing="0">
  <tr><th>Параметр</th><th>Эффект</th><th>Плюсы</th><th>Минусы</th><th>Стартовые значения</th></tr>
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
    <td>Убирает “рывки”, делает поверхность приятнее для nTop.</td>
    <td>Слишком много → усадка/замыливание деталей.</td>
    <td>10–30 (начать с 15)</td>
  </tr>
</table>

<h3>3) Вкладка “ANSYS / CAE” (ANSYS 2025 R2)</h3>
<ul>
  <li>Выход: <code>ansys_layers.json</code>, <code>ansys_layers.csv</code>, <code>ansys_mechanical_import_layers.py</code>.</li>
  <li>Идея: назначить ортотропию по слоям (X вдоль печати, Z — build direction).</li>
  <li><b>Пресеты</b> на вкладке ANSYS меняют параметры генерируемого Mechanical-скрипта (группировка слоёв, порог confidence, создавать ли NS/CS).</li>
</ul>
<ol>
  <li>Откройте ANSYS Mechanical, импортируйте геометрию, сделайте <b>Mesh</b>.</li>
  <li>Mechanical → Automation → Scripting → <b>Run Script…</b></li>
  <li>Выберите <code>ansys_mechanical_import_layers.py</code> из папки результата.</li>
  <li>После выполнения появятся Named Selections <code>L_0000</code>, <code>L_0001</code>… и Coordinate Systems <code>CS_L_0000</code>… (если API доступен в вашей конфигурации).</li>
</ol>

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

        header = QtWidgets.QLabel(
            "Цель: получить `*_s2s_preview_structure.stl`, который открывается в nTop и "
            "дальше конвертируется в solid/STEP средствами nTop.\n"
            "Поддержки/подложка (`Type=0`) игнорируются; используется только траектория модели (`Type=1`).\n"
            "Подсказки: наведите курсор на параметр (или Shift+F1 → клик)."
            "Пайплайн: Import Mesh → Implicit Body from Mesh → (Smooth/Close/Repair) → Convert to CAD/Solid → Export STEP."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        io_group = QtWidgets.QGroupBox("Шаг 1 — Входные файлы")
        io_form = QtWidgets.QFormLayout(io_group)
        layout.addWidget(io_group)

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
            r"Папка результата (*_s2s_preview_structure.stl/ply, ntop_recipe.txt, metadata.json, ...)"
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
        layout.addWidget(tabs, 1)

        # --- Tab: nTop / Geometry preview ---
        geometry_tab = QtWidgets.QWidget()
        tabs.addTab(geometry_tab, "nTop / Геометрия")
        geo_layout = QtWidgets.QVBoxLayout(geometry_tab)

        geo_intro = QtWidgets.QLabel(
            "Режим nTop: восстановление внутренней структуры и экспорт `*_s2s_preview_structure.stl`.\n"
            "Дальше: nTop → mesh → implicit → solidify/repair → экспорт STEP."
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

        self.export_ntop = QtWidgets.QCheckBox("Экспортировать набор для nTop (PLY + ntop_points.csv + ntop_recipe.txt)")
        self.export_ntop.setChecked(True)
        self.export_ntop.setToolTip(
            "Доп. файлы для nTop: PLY (mesh), ntop_points.csv (point cloud из вокселей) и краткий рецепт импорта/параметров."
        )
        self.export_ntop.setWhatsThis(self.export_ntop.toolTip())
        geo_form.addRow("Набор для nTop:", self.export_ntop)

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

        self.smooth = QtWidgets.QSpinBox()
        self.smooth.setRange(0, 200)
        self.smooth.setValue(0)
        _set_help(
            self.smooth,
            title="Сглаживание сетки (итерации)",
            body="Сглаживание уже готовой сетки (Laplacian).",
            pros="Убирает ‘рывки’ и делает поверхность приятнее для имплицитизации в nTop.",
            cons="Может вызывать усадку/замыливание деталей при больших значениях.",
            tip="10–30 обычно достаточно. Если форма начинает ‘плыть’ — уменьшите.",
        )
        geo_form.addRow("Сглаживание сетки (итерации):", self.smooth)

        geo_layout.addStretch(1)

        # --- Tab: ANSYS / CAE ---
        ansys_tab = QtWidgets.QWidget()
        tabs.addTab(ansys_tab, "ANSYS / CAE")
        ansys_layout = QtWidgets.QVBoxLayout(ansys_tab)

        ansys_intro = QtWidgets.QLabel(
            "Режим ANSYS: экспорт ориентации печати по слоям для назначения ортотропии в Mechanical.\n"
            "Выход: ansys_layers.json/csv + ansys_mechanical_import_layers.py."
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
            "ANSYS 2025 R2:\n"
            "1) Импортируйте геометрию, сгенерируйте mesh.\n"
            "2) Mechanical → Automation → Scripting → Run Script…\n"
            "3) Запустите ansys_mechanical_import_layers.py из папки результата.\n"
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

        # --- Run + Outputs ---
        self.run_btn = QtWidgets.QPushButton("Запуск")
        self.run_btn.setMinimumHeight(36)
        layout.addWidget(self.run_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        outputs_group = QtWidgets.QGroupBox("Результаты")
        out_layout = QtWidgets.QVBoxLayout(outputs_group)
        layout.addWidget(outputs_group)

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

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, 1)

        self.thread: QtCore.QThread | None = None
        self.worker: Worker | None = None
        self._last_outputs: list[str] = []

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

        # If user edits any parameter manually -> switch preset to Custom.
        for w in (
            self.export_geometry,
            self.export_ntop,
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
            "   Insight → Toolpaths → Simulation data export.\n\n"
            "2) Выберите Slicer job folder (ssys_*):\n"
            "   Папка задания, где лежат toolpathParams/supportParams/sliceParams.\n"
            "   Нужно для Auto-определения ширины дорожки.\n\n"
            "3) Выберите Placed STL:\n"
            "   STL после размещения на столе (placed).\n\n"
            "4) Выберите папку результата и нажмите Запуск.\n\n"
            "Выход:\n"
            " - *_s2s_preview_structure.stl (открывается в nTop; имя включает деталь и параметры)\n"
            " - *_s2s_preview_structure.ply, ntop_recipe.txt (опционально: bundle для nTop)\n"
            " - ntop_points.csv (опционально: point cloud из вокселей для nTop)\n"
            " - metadata.json (параметры/матрица/статистика)\n\n"
            "В nTop: импортируйте STL → implicit/solidify/repair → экспорт STEP."
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
        self.export_ntop.setEnabled(enabled)
        if not enabled:
            self.export_ntop.setChecked(False)

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
        do_ntop = bool(self.export_ntop.isChecked()) and do_geo
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
            seg = (header.segment_filter_length_mm if header else None) or 0.508
            max_jump = 3.0 * float(seg)
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
            smooth_iterations=int(self.smooth.value()),
            export_cae_layers=do_cae,
            export_geometry_preview=do_geo,
            export_ntop_bundle=do_ntop,
            ansys_min_confidence=float(self.ansys_min_conf.value()),
            ansys_group_size_layers=int(self.ansys_group_layers.value()),
            ansys_create_named_selections=bool(self.ansys_create_ns.isChecked()),
            ansys_create_coordinate_systems=bool(self.ansys_create_cs.isChecked()),
        )

        self.progress.setValue(0)
        self.log_box.clear()
        self._append_log("Starting…")
        self.open_out_btn.setEnabled(False)

        self.run_btn.setEnabled(False)
        self.thread = QtCore.QThread()
        self.worker = Worker(cfg)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._done)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _done(self, ok: bool, message: str, outputs: object) -> None:
        self.run_btn.setEnabled(True)
        self._append_log(message)
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Error", message)
            return
        if isinstance(outputs, list):
            self._set_outputs([str(x) for x in outputs])
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


def main() -> None:
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
