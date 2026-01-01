from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from slice2solid.core.insight_simulation import ToolpathPoint


@dataclasses.dataclass(frozen=True)
class LayerOrientation:
    layer_id: int
    z_min: float
    z_max: float
    z_center: float
    # Unit vector in placed STL coordinates. In MVP we estimate in XY plane.
    dir_xyz: tuple[float, float, float] | None
    # Mean in-plane angle in degrees ([-90..90) equivalence via 180° symmetry).
    angle_deg: float | None
    # 0..1, higher means directions are more consistent within the layer.
    confidence: float
    # Diagnostics
    segments_used: int
    total_weight: float


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    out = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return out


def compute_layer_orientations(
    xyz: np.ndarray,
    *,
    slice_height_mm: float,
    z0_mm: float | None = None,
    layer_axis_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_jump_mm: float | None = None,
    min_inplane_segment_mm: float = 1e-3,
    weights: np.ndarray | None = None,
) -> list[LayerOrientation]:
    """
    Compute a dominant toolpath direction per layer (in-plane, i.e. perpendicular to layer axis).

    Uses XY-projected segment directions and 180°-symmetric averaging via doubled angles:
        mean(theta) = 0.5 * atan2(sum(w*sin(2θ)), sum(w*cos(2θ)))

    Args:
        xyz: (N,3) toolpath points in placed STL coordinates (mm), ordered as in export.
        slice_height_mm: slice height (mm).
        z0_mm: Z origin for layer indexing. Defaults to min(Z).
        max_jump_mm: if provided, segments longer than this are treated as travel jumps and ignored.
        min_inplane_segment_mm: ignore near-axis / tiny in-plane segments when computing in-plane direction.
        weights: optional (N,) weights per point to apply to segment (uses average of endpoints).
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")
    if slice_height_mm <= 0:
        raise ValueError("slice_height_mm must be > 0")
    if xyz.shape[0] < 2:
        return []

    ax = np.array(layer_axis_xyz, dtype=float)
    n = float(np.linalg.norm(ax))
    if not np.isfinite(n) or n <= 1e-12:
        raise ValueError("layer_axis_xyz must be non-zero")
    ax = ax / n

    # In-plane orthonormal basis (u, v) where:
    # - u is the projection of global X onto the layer plane (or global Y if X is near-parallel to axis)
    # - v completes a right-handed system: v = ax x u
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    u = ref - ax * float(np.dot(ax, ref))
    un = float(np.linalg.norm(u))
    if un <= 1e-9 or not np.isfinite(un):
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
        u = ref - ax * float(np.dot(ax, ref))
        un = float(np.linalg.norm(u))
    if un <= 1e-12 or not np.isfinite(un):
        raise ValueError("layer_axis_xyz is degenerate for basis construction")
    u = u / un
    v = np.cross(ax, u)
    vn = float(np.linalg.norm(v))
    if vn <= 1e-12 or not np.isfinite(vn):
        raise ValueError("layer_axis_xyz is degenerate for basis construction")
    v = v / vn

    coords = xyz @ ax
    z0 = float(np.min(coords)) if z0_mm is None else float(z0_mm)
    max_jump = float(max_jump_mm) if max_jump_mm is not None else None

    # per layer accumulators
    sum_c = defaultdict(float)
    sum_s = defaultdict(float)
    sum_w = defaultdict(float)
    segs = defaultdict(int)

    for i in range(1, xyz.shape[0]):
        a = xyz[i - 1]
        b = xyz[i]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-12:
            continue
        if max_jump is not None and seg_len > max_jump:
            continue

        # Project segment onto layer plane.
        seg_plane = seg - ax * float(np.dot(seg, ax))
        sx = float(np.dot(seg_plane, u))
        sy = float(np.dot(seg_plane, v))
        len_inplane = math.hypot(sx, sy)
        if len_inplane < float(min_inplane_segment_mm):
            continue

        # layer index by segment midpoint along the layer axis
        z_mid = float(0.5 * ((a @ ax) + (b @ ax)))
        layer_id = int(round((z_mid - z0) / float(slice_height_mm)))

        theta = math.atan2(sy, sx)
        c2 = math.cos(2.0 * theta)
        s2 = math.sin(2.0 * theta)

        w = len_inplane
        if weights is not None:
            w_pt = float(0.5 * (weights[i - 1] + weights[i]))
            if w_pt > 0:
                w *= w_pt

        sum_c[layer_id] += w * c2
        sum_s[layer_id] += w * s2
        sum_w[layer_id] += w
        segs[layer_id] += 1

    if not sum_w:
        return []

    out: list[LayerOrientation] = []
    for layer_id in sorted(sum_w.keys()):
        w = float(sum_w[layer_id])
        c = float(sum_c[layer_id])
        s = float(sum_s[layer_id])
        conf = math.sqrt(c * c + s * s) / w if w > 0 else 0.0

        if w <= 0 or conf <= 1e-9:
            angle = None
            dir_xyz = None
        else:
            theta = 0.5 * math.atan2(s, c)
            theta = _wrap_pi(theta)
            angle = math.degrees(theta)
            d = u * math.cos(theta) + v * math.sin(theta)
            dn = float(np.linalg.norm(d))
            if dn <= 1e-12 or not np.isfinite(dn):
                angle = None
                dir_xyz = None
            else:
                d = d / dn
                dir_xyz = (float(d[0]), float(d[1]), float(d[2]))

        z_min = z0 + layer_id * float(slice_height_mm)
        z_max = z0 + (layer_id + 1) * float(slice_height_mm)
        z_center = 0.5 * (z_min + z_max)

        out.append(
            LayerOrientation(
                layer_id=int(layer_id),
                z_min=float(z_min),
                z_max=float(z_max),
                z_center=float(z_center),
                dir_xyz=dir_xyz,
                angle_deg=angle,
                confidence=float(conf),
                segments_used=int(segs.get(layer_id, 0)),
                total_weight=float(w),
            )
        )

    return out


def compute_layer_orientations_toolpath(
    points: Iterable[ToolpathPoint],
    *,
    slice_height_mm: float,
    z0_mm: float,
    layer_axis_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_jump_mm: float | None = None,
    min_inplane_segment_mm: float = 1e-3,
    type_filter: int = 1,
    weight_by_bead_area: bool = True,
    ignore_zero_factor: bool = True,
    ignore_zero_bead_area: bool = True,
) -> list[LayerOrientation]:
    """
    Streaming variant of `compute_layer_orientations` that consumes ToolpathPoint items.

    This avoids storing large point arrays in memory; intended for GUI/CLI runs on huge exports.

    Args:
        points: toolpath points in placed STL coordinates (mm), ordered as in export.
        slice_height_mm: slice height (mm).
        z0_mm: Z origin for layer indexing (typically min Z of Type=1 points).
        max_jump_mm: if provided, segments longer than this are treated as travel jumps and ignored.
        min_inplane_segment_mm: ignore near-axis / tiny in-plane segments when computing in-plane direction.
        type_filter: toolpath point Type to use (default: 1 = model).
        weight_by_bead_area: if True, weights segments by avg bead_area of endpoints.
    """
    if slice_height_mm <= 0:
        raise ValueError("slice_height_mm must be > 0")

    ax = np.array(layer_axis_xyz, dtype=float)
    n = float(np.linalg.norm(ax))
    if not np.isfinite(n) or n <= 1e-12:
        raise ValueError("layer_axis_xyz must be non-zero")
    ax = ax / n
    z0 = float(z0_mm)
    max_jump = float(max_jump_mm) if max_jump_mm is not None else None

    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    u = ref - ax * float(np.dot(ax, ref))
    un = float(np.linalg.norm(u))
    if un <= 1e-9 or not np.isfinite(un):
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
        u = ref - ax * float(np.dot(ax, ref))
        un = float(np.linalg.norm(u))
    if un <= 1e-12 or not np.isfinite(un):
        raise ValueError("layer_axis_xyz is degenerate for basis construction")
    u = u / un
    v = np.cross(ax, u)
    vn = float(np.linalg.norm(v))
    if vn <= 1e-12 or not np.isfinite(vn):
        raise ValueError("layer_axis_xyz is degenerate for basis construction")
    v = v / vn

    sum_c = defaultdict(float)
    sum_s = defaultdict(float)
    sum_w = defaultdict(float)
    segs = defaultdict(int)

    prev: ToolpathPoint | None = None
    for pt in points:
        if pt.type != int(type_filter):
            prev = None
            continue
        if ignore_zero_factor and float(pt.factor) <= 1e-12:
            prev = None
            continue
        if ignore_zero_bead_area and float(pt.bead_area) <= 1e-12:
            prev = None
            continue
        if prev is None:
            prev = pt
            continue

        prev_x, prev_y, prev_z = float(prev.x), float(prev.y), float(prev.z)
        curr_x, curr_y, curr_z = float(pt.x), float(pt.y), float(pt.z)
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        dz = curr_z - prev_z
        seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        if seg_len <= 1e-12:
            prev = pt
            continue
        if max_jump is not None and seg_len > max_jump:
            prev = pt
            continue

        # Project segment onto layer plane (perpendicular to axis) and measure its in-plane length.
        seg_dot = float(ax[0] * dx + ax[1] * dy + ax[2] * dz)
        px = float(dx - seg_dot * float(ax[0]))
        py = float(dy - seg_dot * float(ax[1]))
        pz = float(dz - seg_dot * float(ax[2]))
        sx = float(u[0] * px + u[1] * py + u[2] * pz)
        sy = float(v[0] * px + v[1] * py + v[2] * pz)
        len_inplane = math.hypot(sx, sy)
        if len_inplane < float(min_inplane_segment_mm):
            prev = pt
            continue

        z_prev = float(ax[0] * prev_x + ax[1] * prev_y + ax[2] * prev_z)
        z_curr = float(ax[0] * curr_x + ax[1] * curr_y + ax[2] * curr_z)
        z_mid = 0.5 * (z_prev + z_curr)
        layer_id = int(round((z_mid - z0) / float(slice_height_mm)))

        theta = math.atan2(sy, sx)
        c2 = math.cos(2.0 * theta)
        s2 = math.sin(2.0 * theta)

        w = float(len_inplane)
        if weight_by_bead_area:
            w_area = 0.5 * (float(prev.bead_area) + float(pt.bead_area))
            if w_area > 0:
                w *= w_area

        sum_c[layer_id] += w * c2
        sum_s[layer_id] += w * s2
        sum_w[layer_id] += w
        segs[layer_id] += 1

        prev = pt

    if not sum_w:
        return []

    out: list[LayerOrientation] = []
    for layer_id in sorted(sum_w.keys()):
        w = float(sum_w[layer_id])
        c = float(sum_c[layer_id])
        s = float(sum_s[layer_id])
        conf = math.sqrt(c * c + s * s) / w if w > 0 else 0.0

        if w <= 0 or conf <= 1e-9:
            angle = None
            dir_xyz = None
        else:
            theta = 0.5 * math.atan2(s, c)
            theta = _wrap_pi(theta)
            angle = math.degrees(theta)
            d = u * math.cos(theta) + v * math.sin(theta)
            dn = float(np.linalg.norm(d))
            if dn <= 1e-12 or not np.isfinite(dn):
                angle = None
                dir_xyz = None
            else:
                d = d / dn
                dir_xyz = (float(d[0]), float(d[1]), float(d[2]))

        z_min = z0 + layer_id * float(slice_height_mm)
        z_max = z0 + (layer_id + 1) * float(slice_height_mm)
        z_center = 0.5 * (z_min + z_max)

        out.append(
            LayerOrientation(
                layer_id=int(layer_id),
                z_min=float(z_min),
                z_max=float(z_max),
                z_center=float(z_center),
                dir_xyz=dir_xyz,
                angle_deg=angle,
                confidence=float(conf),
                segments_used=int(segs.get(layer_id, 0)),
                total_weight=float(w),
            )
        )

    return out
