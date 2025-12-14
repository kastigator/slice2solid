from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import trimesh
from PIL import Image


@dataclass(frozen=True)
class LayerRow:
    layer_id: int
    z_min_mm: float
    z_max_mm: float
    angle_deg: float | None
    dx: float | None
    dy: float | None
    dz: float | None
    confidence: float
    segments_used: int
    total_weight: float


def _parse_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    return float(s)


def load_layers_csv(path: Path) -> list[LayerRow]:
    rows: list[LayerRow] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                LayerRow(
                    layer_id=int(row["layer_id"]),
                    z_min_mm=float(row["z_min_mm"]),
                    z_max_mm=float(row["z_max_mm"]),
                    angle_deg=_parse_float(row.get("angle_deg", "")),
                    dx=_parse_float(row.get("dx", "")),
                    dy=_parse_float(row.get("dy", "")),
                    dz=_parse_float(row.get("dz", "")),
                    confidence=float(row["confidence"]),
                    segments_used=int(row["segments_used"]),
                    total_weight=float(row["total_weight"]),
                )
            )
    return rows


def pick_representative_layer(layers: list[LayerRow]) -> LayerRow:
    candidates = [l for l in layers if l.dx is not None and l.dy is not None and l.confidence > 0]
    if not candidates:
        raise ValueError("No layers with orientation available (dx/dy missing).")
    # Prefer high confidence and sufficient segments.
    candidates.sort(key=lambda l: (l.confidence, l.segments_used, l.total_weight), reverse=True)
    return candidates[0]


def pick_layer_by_id(layers: list[LayerRow], layer_id: int) -> LayerRow:
    for l in layers:
        if l.layer_id == int(layer_id):
            if l.dx is None or l.dy is None:
                raise ValueError(f"Layer {layer_id} has no orientation (dx/dy missing).")
            return l
    raise ValueError(f"Layer {layer_id} not found in ansys_layers.csv.")


def _iter_sim_export_rows(path: Path):
    # The simulation export is a text file; we use slice2solid parser if available.
    from slice2solid.core.insight_simulation import read_simulation_export

    _header, rows_iter = read_simulation_export(path)
    yield from rows_iter


def extract_layer_segments(
    simulation_export: Path,
    *,
    layer: LayerRow,
    slice_height_mm: float,
    z0_mm: float,
    type_filter: int = 1,
    max_jump_mm: float = 20.0,
    min_xy_segment_mm: float = 0.10,
    max_segments: int = 25000,
) -> np.ndarray:
    """
    Returns segments as (M,2,2) array in XY for a specific layer.
    """
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    prev = None
    for pt in _iter_sim_export_rows(simulation_export):
        if pt.type != type_filter:
            prev = None
            continue
        curr = (float(pt.x), float(pt.y), float(pt.z))
        if prev is None:
            prev = curr
            continue

        ax_, ay_, az_ = prev
        bx_, by_, bz_ = curr
        dx = bx_ - ax_
        dy = by_ - ay_
        dz = bz_ - az_
        seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        if seg_len <= 1e-12:
            prev = curr
            continue
        if seg_len > float(max_jump_mm):
            prev = curr
            continue

        len_xy = math.hypot(dx, dy)
        if len_xy < float(min_xy_segment_mm):
            prev = curr
            continue

        z_mid = 0.5 * (az_ + bz_)
        layer_id = int(round((z_mid - float(z0_mm)) / float(slice_height_mm)))
        if layer_id == layer.layer_id:
            segments.append(((ax_, ay_), (bx_, by_)))
            if len(segments) >= int(max_segments):
                break

        prev = curr

    if not segments:
        raise ValueError(f"No segments extracted for layer_id={layer.layer_id}.")

    return np.array(segments, dtype=np.float64)


def _crop_nonwhite(path: Path, *, threshold: int = 250, pad_px: int = 10) -> np.ndarray:
    """
    Crop mostly-white margins to make preview images readable.
    Works well for Insight preview bitmaps; for dark screenshots it will usually keep full frame.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
    mask = gray < threshold
    if not np.any(mask):
        return arr
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - pad_px)
    x0 = max(0, x0 - pad_px)
    y1 = min(arr.shape[0] - 1, y1 + pad_px)
    x1 = min(arr.shape[1] - 1, x1 + pad_px)
    return arr[y0 : y1 + 1, x0 : x1 + 1]


def _add_subfigure_label(ax, label: str) -> None:
    ax.text(
        0.0,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
    )


def _add_annotation(ax, text: str, x: float, y: float, *, color: str = "#C00000") -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=10.5,
        color=color,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def generate_figure(
    *,
    base_dir: Path,
    out_dir: Path,
    simulation_export_name: str | None = None,
) -> tuple[Path, Path]:
    """
    Generates "Figure B" as:
      (a) XY toolpaths of one representative layer + arrow for dominant direction
      (b) angle/confidence vs height (summary)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_svg = out_dir / "fig_B_directionality.svg"
    out_png = out_dir / "fig_B_directionality.png"

    out_slice2solid = base_dir / "slice2solid_out"
    layers_csv = out_slice2solid / "ansys_layers.csv"
    layers_json = out_slice2solid / "ansys_layers.json"
    if not layers_csv.exists():
        raise FileNotFoundError(f"Missing {layers_csv}")
    if not layers_json.exists():
        raise FileNotFoundError(f"Missing {layers_json}")

    # Load z0/slice height from json for consistency with tool output.
    import json

    cfg = json.loads(layers_json.read_text(encoding="utf-8"))
    slice_height_mm = float(cfg["slice_height_mm"])
    z0_mm = float(cfg["z0_mm"])

    layers = load_layers_csv(layers_csv)
    # Default: keep compatibility with your note "слой 24".
    layer = pick_layer_by_id(layers, layer_id=24)

    if simulation_export_name:
        simulation_export = base_dir / simulation_export_name
    else:
        # Pick the largest *.txt in base dir as the simulation export.
        txts = sorted(base_dir.glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)
        if not txts:
            raise FileNotFoundError(f"No *.txt simulation export found in {base_dir}")
        simulation_export = txts[0]

    seg_xy = extract_layer_segments(
        simulation_export,
        layer=layer,
        slice_height_mm=slice_height_mm,
        z0_mm=z0_mm,
    )

    # If a placed STL exists, we can overlay the section contour for the same layer Z.
    stl_candidates = [
        base_dir / "part-table.stl",
        base_dir / "part-table.STL",
        base_dir / "part.STL",
        base_dir / "part.stl",
    ]
    placed_stl = next((p for p in stl_candidates if p.exists()), None)

    # Downsample segments for rendering if needed.
    if seg_xy.shape[0] > 12000:
        idx = np.linspace(0, seg_xy.shape[0] - 1, 12000).astype(int)
        seg_xy = seg_xy[idx]

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    # Optional Insight screenshot(s) (provided by user).
    gui_candidates = [
        base_dir / "docs" / "figure B" / "слой 24.png",
        REPO_ROOT / "docs" / "figure B" / "слой 24.png",
    ]
    gui_shot = next((p for p in gui_candidates if p.exists()), None)
    has_insight_layer_shot = gui_shot is not None

    # Optional Insight preview bitmap from job folder (often exists).
    insight_bmp_candidates = [
        base_dir / "ssys_part-table" / "part-table.bmp",
        base_dir / "ssys_part" / "part.bmp",
    ]
    insight_bmp = next((p for p in insight_bmp_candidates if p.exists()), None)
    has_insight_bmp = insight_bmp is not None

    # Layout:
    # If we have Insight images, show them on the first row (source context), and slice2solid plots below.
    if has_insight_bmp or has_insight_layer_shot:
        fig = plt.figure(figsize=(13.6, 8.4), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[0.82, 1.0])
        ax_src0 = fig.add_subplot(gs[0, 0])
        ax_src1 = fig.add_subplot(gs[0, 1])
        ax0 = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[1, 1])
        ax_src0.axis("off")
        ax_src1.axis("off")
    else:
        fig = plt.figure(figsize=(13.5, 6.0), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])

    lc = LineCollection(seg_xy, linewidths=0.35, colors="#1f4e79", alpha=0.85)
    ax0.add_collection(lc)

    # Limits
    xy = seg_xy.reshape(-1, 2)
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    pad = max(x_max - x_min, y_max - y_min) * 0.08
    ax0.set_xlim(x_min - pad, x_max + pad)
    ax0.set_ylim(y_min - pad, y_max + pad)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("X, мм")
    ax0.set_ylabel("Y, мм")
    ax0.set_title("")
    ax0.grid(True, linewidth=0.4, alpha=0.25)

    # Overlay the placed STL cross-section contour for the same Z, if possible.
    z_sel = (layer.z_min_mm + layer.z_max_mm) * 0.5
    if placed_stl is not None:
        try:
            mesh = trimesh.load_mesh(placed_stl, process=False)
            section = mesh.section(plane_origin=[0.0, 0.0, float(z_sel)], plane_normal=[0.0, 0.0, 1.0])
            if section is not None:
                # Use 3D polylines to keep coordinate system consistent with toolpaths.
                for curve3d in section.discrete:
                    if curve3d is None or len(curve3d) < 2:
                        continue
                    curve3d = np.asarray(curve3d, dtype=float)
                    ax0.plot(curve3d[:, 0], curve3d[:, 1], color="#1A1A1A", linewidth=1.6, alpha=0.9)
        except Exception:
            # If sectioning fails, keep the toolpaths plot without contour.
            pass

    # Direction arrow (dominant)
    cx, cy = xy.mean(axis=0)
    v = np.array([float(layer.dx), float(layer.dy)], dtype=float)
    v /= max(1e-12, np.linalg.norm(v))
    arrow_len = 0.22 * max(x_max - x_min, y_max - y_min)
    p1 = (cx - v[0] * arrow_len * 0.5, cy - v[1] * arrow_len * 0.5)
    p2 = (cx + v[0] * arrow_len * 0.5, cy + v[1] * arrow_len * 0.5)
    ax0.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.0,
            color="#C00000",
        )
    )
    _add_annotation(ax0, "Оценённое доминирующее направление", p2[0], p2[1])
    _add_subfigure_label(ax0, "в)")

    # Summary plot: angle and confidence vs height
    zs = np.array([(l.z_min_mm + l.z_max_mm) * 0.5 for l in layers], dtype=float)
    angles = np.array([l.angle_deg if l.angle_deg is not None else np.nan for l in layers], dtype=float)
    confs = np.array([l.confidence for l in layers], dtype=float)

    ax1.plot(angles, zs, color="#1f4e79", linewidth=1.2, label="Угол укладки, град.")
    ax1.set_xlabel("Угол в плоскости XY, град.")
    ax1.set_ylabel("Z, мм")
    ax1.grid(True, linewidth=0.4, alpha=0.25)
    ax1.set_title("")

    ax1b = ax1.twiny()
    ax1b.plot(confs, zs, color="#2F5597", linewidth=1.0, alpha=0.8, label="Достоверность")
    ax1b.set_xlabel("Достоверность оценки (0…1)", labelpad=8)

    # Mark selected layer
    ax1.axhline(z_sel, color="#C00000", linewidth=1.3, alpha=0.9)
    # Put a short label near the left side so the meaning of the red line is explicit.
    x_left = float(np.nanmin(angles)) if np.isfinite(np.nanmin(angles)) else -90.0
    _add_annotation(ax1, f"Выбранный слой {layer.layer_id}", x_left, z_sel)
    _add_subfigure_label(ax1, "г)")

    # Render Insight source images if available.
    if has_insight_bmp or has_insight_layer_shot:
        if has_insight_bmp:
            img0 = _crop_nonwhite(insight_bmp)
            ax_src0.imshow(img0)
        else:
            ax_src0.text(
                0.5,
                0.5,
                "Нет изображения предпросмотра из Insight",
                ha="center",
                va="center",
                fontsize=10.5,
            )

        if has_insight_layer_shot:
            img1 = np.asarray(Image.open(gui_shot).convert("RGB"))
            ax_src1.imshow(img1)
        else:
            ax_src1.text(
                0.5,
                0.5,
                "Нет скриншота траекторий слоя из Insight",
                ha="center",
                va="center",
                fontsize=10.5,
            )

        _add_subfigure_label(ax_src0, "а)")
        _add_subfigure_label(ax_src1, "б)")

    # Main figure caption should be in Word; avoid long title inside the image to prevent overlaps.
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)

    return out_svg, out_png


def main() -> int:
    base = Path(r"C:\Users\alexa\Desktop\тест проект")
    svg, png = generate_figure(base_dir=base, out_dir=Path("docs") / "figures")
    print(f"OK: wrote {svg}")
    print(f"OK: wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
