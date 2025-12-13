from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from PySide6 import QtCore, QtWidgets

from slice2solid.core.insight_simulation import (
    invert_rowvec_matrix,
    read_simulation_export,
    transform_points_rowvec,
)
from slice2solid.core.insight_params import estimate_bead_width_mm, infer_stl_path_from_job, load_job_params
from slice2solid.core.cae_orientation import compute_layer_orientations
from slice2solid.core.voxelize import mesh_from_voxels_configured, voxelize_toolpath


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
                raise ValueError("No outputs selected: enable Geometry preview and/or ANSYS export.")

            if self.cfg.export_geometry_preview:
                self.log.emit("Loading placed STL…")
                mesh = trimesh.load_mesh(self.cfg.placed_stl, force="mesh")
                bounds_min = mesh.bounds[0]
                bounds_max = mesh.bounds[1]
            else:
                bounds_min = None
                bounds_max = None

            self.log.emit("Preparing coordinate transform (CMB -> placed STL)…")
            stl_to_cmb = sim_header.stl_to_cmb
            cmb_to_stl = invert_rowvec_matrix(stl_to_cmb)

            self.log.emit("Streaming toolpath points (Type=1)…")
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

                self.log.emit("Voxelizing…")
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

                self.log.emit("Extracting mesh (marching cubes)…")
                preview_mesh = mesh_from_voxels_configured(
                    vox,
                    volume_smooth_sigma_vox=self.cfg.volume_smooth_sigma_vox,
                    min_component_faces=self.cfg.min_mesh_component_faces,
                )
                if self.cfg.smooth_iterations > 0:
                    self.log.emit(f"Smoothing mesh ({self.cfg.smooth_iterations} iterations)…")
                    trimesh.smoothing.filter_laplacian(preview_mesh, iterations=int(self.cfg.smooth_iterations))
                self.progress.emit(85)

            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_stl = out_dir / "preview_structure.stl"
            out_json = out_dir / "metadata.json"
            out_layers_json = out_dir / "ansys_layers.json"
            out_layers_csv = out_dir / "ansys_layers.csv"
            out_ansys_script = out_dir / "ansys_mechanical_import_layers.py"

            outputs: list[str] = []

            if self.cfg.export_geometry_preview and preview_mesh is not None:
                self.log.emit(f"Writing {out_stl}…")
                preview_mesh.export(out_stl)
                outputs.append(str(out_stl))

            if self.cfg.export_cae_layers:
                self.log.emit(f"Writing {out_layers_json}…")
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
                self.log.emit(f"Writing {out_layers_csv}…")
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

                self.log.emit(f"Writing {out_ansys_script}…")
                out_ansys_script.write_text(_ANSYS_MECHANICAL_SCRIPT_TEMPLATE, encoding="utf-8")
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
                base = f"{out_stl.name}, {out_json.name}{extra}"
            else:
                base = f"{out_json.name}{extra}"
            self.finished.emit(True, f"Done. Outputs: {base}", outputs)
        except Exception as e:
            self.finished.emit(False, f"Error: {e}", [])


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
import json
import math
import os

HERE = os.path.dirname(__file__)
LAYERS_JSON = os.path.join(HERE, "ansys_layers.json")


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

    # Create outputs per layer.
    created_ns = 0
    created_cs = 0

    for l in layers:
        lid = int(l["layer_id"])
        z_min = float(l["z_min"])
        z_max = float(l["z_max"])
        d = l.get("dir_xyz", None)
        conf = float(l.get("confidence", 0.0))

        # Skip very low-confidence layers to avoid nonsense orientation.
        if d is None or conf < 0.2:
            continue

        # Select elements whose centroid Z falls into this layer band.
        layer_eids = [eid for eid in ids if (elem_z[eid] >= z_min and elem_z[eid] < z_max)]
        if not layer_eids:
            continue

        name = "L_{:04d}".format(lid)
        _create_named_selection_by_ids(model, name, layer_eids)
        created_ns += 1

        # Place coordinate system near the mean centroid of this layer selection.
        cx = sum(elem_xyz[eid][0] for eid in layer_eids) / float(len(layer_eids))
        cy = sum(elem_xyz[eid][1] for eid in layer_eids) / float(len(layer_eids))
        cz = sum(elem_xyz[eid][2] for eid in layer_eids) / float(len(layer_eids))
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


_HELP_HTML = """
<h2>slice2solid — Help</h2>

<h3>1) Что выбрать в Insight</h3>
<ol>
  <li><b>Simulation export (*.txt)</b>: Insight → Toolpaths → Simulation data export.</li>
  <li><b>Slicer job folder (ssys_*)</b>: папка задания, где лежат <code>toolpathParams.*</code>, <code>sliceParams.*</code> (нужно для auto-ширины дорожки).</li>
  <li><b>Placed STL (*.stl)</b>: STL после ориентации/размещения на столе в Insight.</li>
  <li><b>Output folder</b>: папка результата (сюда пишутся файлы).</li>
</ol>

<h3>2) Вкладка “nTop / Geometry”</h3>
<ul>
  <li>Выход: <code>preview_structure.stl</code> + <code>metadata.json</code>.</li>
  <li>Если поверхность “из пирамидок”: уменьшите <b>Voxel size</b> или включите <b>Volume smoothing</b>, либо сгладьте уже в nTop (mesh → implicit).</li>
  <li>Если появляются “лишние линии” внутри: включите <b>Ignore travel jumps</b> и/или увеличьте <b>Remove noise</b>.</li>
</ul>

<h3>3) Вкладка “ANSYS / CAE” (ANSYS 2025 R2)</h3>
<ul>
  <li>Выход: <code>ansys_layers.json</code>, <code>ansys_layers.csv</code>, <code>ansys_mechanical_import_layers.py</code>.</li>
  <li>Идея: назначить ортотропию по слоям (X вдоль печати, Z — build direction).</li>
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
        self.resize(920, 640)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        header = QtWidgets.QLabel(
            "Цель: получить `preview_structure.stl`, который открывается в nTop и "
            "дальше конвертируется в solid/STEP средствами nTop.\n"
            "Поддержки/подложка (`Type=0`) игнорируются; используется только траектория модели (`Type=1`)."
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
        self.sim_btn = QtWidgets.QPushButton("Browse…")
        sim_row = QtWidgets.QHBoxLayout()
        sim_row.addWidget(self.sim_edit, 1)
        sim_row.addWidget(self.sim_btn)
        io_form.addRow("Simulation export (*.txt):", sim_row)

        self.job_edit = QtWidgets.QLineEdit()
        self.job_edit.setPlaceholderText(r"Например: ...\ssys_part-table")
        self.job_edit.setToolTip(
            "Папка задания Stratasys Insight (ssys_*).\n"
            "Нужна для чтения `toolpathParams.*` и автоматического определения ширины дорожки."
        )
        self.job_btn = QtWidgets.QPushButton("Browse…")
        job_row = QtWidgets.QHBoxLayout()
        job_row.addWidget(self.job_edit, 1)
        job_row.addWidget(self.job_btn)
        io_form.addRow("Slicer job folder (ssys_*):", job_row)

        self.stl_edit = QtWidgets.QLineEdit()
        self.stl_edit.setPlaceholderText(r"Например: ...\part-table.stl (placed STL)")
        self.stl_edit.setToolTip(
            "Placed STL — STL после ориентации/размещения на столе в Insight.\n"
            "Используется как эталонная геометрия/габариты для вокселизации и проверки координат."
        )
        self.stl_btn = QtWidgets.QPushButton("Browse…")
        stl_row = QtWidgets.QHBoxLayout()
        stl_row.addWidget(self.stl_edit, 1)
        stl_row.addWidget(self.stl_btn)
        io_form.addRow("Placed STL (*.stl):", stl_row)

        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText(r"Папка, куда будут записаны preview_structure.stl и metadata.json")
        self.out_btn = QtWidgets.QPushButton("Browse…")
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(self.out_btn)
        io_form.addRow("Output folder:", out_row)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 1)

        # --- Tab: nTop / Geometry preview ---
        geometry_tab = QtWidgets.QWidget()
        tabs.addTab(geometry_tab, "nTop / Geometry")
        geo_layout = QtWidgets.QVBoxLayout(geometry_tab)

        geo_intro = QtWidgets.QLabel(
            "Режим nTop: восстановление внутренней структуры и экспорт `preview_structure.stl`.\n"
            "Дальше: nTop → mesh → implicit → solidify/repair → экспорт STEP."
        )
        geo_intro.setWordWrap(True)
        geo_layout.addWidget(geo_intro)

        geo_group = QtWidgets.QGroupBox("Параметры геометрии")
        geo_form = QtWidgets.QFormLayout(geo_group)
        geo_layout.addWidget(geo_group)

        self.export_geometry = QtWidgets.QCheckBox("Generate preview_structure.stl")
        self.export_geometry.setChecked(True)
        self.export_geometry.setToolTip("Отключите, если нужен только экспорт для ANSYS (карта слоёв).")
        geo_form.addRow("Geometry output:", self.export_geometry)

        self.voxel_size = QtWidgets.QDoubleSpinBox()
        self.voxel_size.setRange(0.05, 5.0)
        self.voxel_size.setSingleStep(0.05)
        self.voxel_size.setValue(0.25)
        self.voxel_size.setDecimals(3)
        self.voxel_size.setToolTip("Размер вокселя (мм). Меньше = точнее, но тяжелее.")
        geo_form.addRow("Voxel size (mm):", self.voxel_size)

        self.auto_radius = QtWidgets.QCheckBox("Auto (from slicer params)")
        self.auto_radius.setChecked(True)
        self.max_radius = QtWidgets.QDoubleSpinBox()
        self.max_radius.setRange(0.1, 10.0)
        self.max_radius.setSingleStep(0.1)
        self.max_radius.setValue(1.5)
        self.max_radius.setDecimals(2)
        self.max_radius.setEnabled(False)
        self.radius_hint = QtWidgets.QLabel("Auto: unknown (select ssys_* folder)")
        radius_row = QtWidgets.QHBoxLayout()
        radius_row.addWidget(self.auto_radius)
        radius_row.addWidget(self.max_radius)
        radius_row.addWidget(self.radius_hint, 1)
        geo_form.addRow("Bead radius limit:", radius_row)

        self.estimate = QtWidgets.QLabel("Оценка: —")
        self.estimate.setWordWrap(True)
        geo_form.addRow("Grid estimate:", self.estimate)

        self.jump_filter = QtWidgets.QCheckBox("Ignore travel jumps between toolpaths (recommended)")
        self.jump_filter.setChecked(True)
        geo_form.addRow("Toolpath filter:", self.jump_filter)

        self.min_island = QtWidgets.QSpinBox()
        self.min_island.setRange(0, 10000)
        self.min_island.setValue(150)
        geo_form.addRow("Remove noise (min voxels):", self.min_island)

        self.min_mesh_faces = QtWidgets.QSpinBox()
        self.min_mesh_faces.setRange(0, 50_000_000)
        self.min_mesh_faces.setValue(2000)
        geo_form.addRow("Remove mesh islands (min faces):", self.min_mesh_faces)

        self.vol_sigma = QtWidgets.QDoubleSpinBox()
        self.vol_sigma.setRange(0.0, 5.0)
        self.vol_sigma.setSingleStep(0.1)
        self.vol_sigma.setValue(0.0)
        self.vol_sigma.setDecimals(2)
        geo_form.addRow("Volume smoothing (sigma, vox):", self.vol_sigma)

        self.smooth = QtWidgets.QSpinBox()
        self.smooth.setRange(0, 200)
        self.smooth.setValue(0)
        geo_form.addRow("Mesh smoothing (iterations):", self.smooth)

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

        self.export_cae = QtWidgets.QCheckBox("Export layer orientation map (ANSYS)")
        self.export_cae.setChecked(True)
        ansys_form.addRow("CAE output:", self.export_cae)

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
        tabs.addTab(help_tab, "Help")
        help_layout = QtWidgets.QVBoxLayout(help_tab)
        self.help_view = QtWidgets.QTextBrowser()
        self.help_view.setOpenExternalLinks(True)
        self.help_view.setHtml(_HELP_HTML)
        help_layout.addWidget(self.help_view, 1)

        # --- Run + Outputs ---
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.setMinimumHeight(36)
        layout.addWidget(self.run_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        outputs_group = QtWidgets.QGroupBox("Outputs")
        out_layout = QtWidgets.QVBoxLayout(outputs_group)
        layout.addWidget(outputs_group)

        self.outputs_list = QtWidgets.QListWidget()
        out_layout.addWidget(self.outputs_list, 1)

        out_btn_row = QtWidgets.QHBoxLayout()
        out_layout.addLayout(out_btn_row)
        self.open_selected_btn = QtWidgets.QPushButton("Open selected")
        self.copy_selected_btn = QtWidgets.QPushButton("Copy path")
        self.open_out_btn = QtWidgets.QPushButton("Open output folder")
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
        self.open_out_btn.clicked.connect(self._open_output_folder)
        self.outputs_list.itemSelectionChanged.connect(self._update_output_buttons)
        self.outputs_list.itemDoubleClicked.connect(self._open_selected_output)
        self.open_selected_btn.clicked.connect(self._open_selected_output)
        self.copy_selected_btn.clicked.connect(self._copy_selected_output_path)

    def _pick_sim(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select simulation export", "", "Text (*.txt)")
        if path:
            self.sim_edit.setText(path)
            self._auto_fill_from_sim(path)

    def _pick_job(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select ssys_* folder")
        if path:
            self.job_edit.setText(path)
            self._auto_fill_from_job(path)

    def _pick_stl(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select placed STL", "", "STL (*.stl *.STL)")
        if path:
            self.stl_edit.setText(path)

    def _pick_out(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
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
            "4) Выберите Output folder и нажмите Run.\n\n"
            "Выход:\n"
            " - preview_structure.stl (открывается в nTop)\n"
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

    def _recompute_auto_radius(self) -> None:
        if not self.auto_radius.isChecked():
            return

        job_dir = self.job_edit.text().strip()
        sim_path = self.sim_edit.text().strip()
        if not job_dir:
            self.radius_hint.setText("Auto: unknown (select ssys_* folder)")
            return
        if not Path(job_dir).exists():
            self.radius_hint.setText("Auto: folder not found")
            return

        try:
            header = None
            if sim_path and Path(sim_path).exists():
                header, _it = read_simulation_export(sim_path)
            params = load_job_params(job_dir)
            bead_w = estimate_bead_width_mm(params, sim_slice_height_mm=header.slice_height_mm if header else None)
            if bead_w is None:
                self.radius_hint.setText("Auto: could not read bead width")
                return
            r = bead_w / 2.0
            self.radius_hint.setText(f"Auto: width≈{bead_w:.3f} mm → radius≈{r:.3f} mm")
        except Exception as e:
            self.radius_hint.setText(f"Auto: error ({e})")

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
        do_cae = bool(self.export_cae.isChecked())
        if not sim or not out or (do_geo and not stl):
            QtWidgets.QMessageBox.warning(
                self,
                "Missing input",
                "Please select simulation export and output folder. Placed STL is required for geometry export.",
            )
            return
        if not do_geo and not do_cae:
            QtWidgets.QMessageBox.warning(self, "Nothing to do", "Enable Geometry output and/or CAE output.")
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


def main() -> None:
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
