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
    max_jump_mm: float | None = None,
    min_xy_segment_mm: float = 1e-3,
    weights: np.ndarray | None = None,
) -> list[LayerOrientation]:
    """
    Compute a dominant toolpath direction per layer.

    Uses XY-projected segment directions and 180°-symmetric averaging via doubled angles:
        mean(theta) = 0.5 * atan2(sum(w*sin(2θ)), sum(w*cos(2θ)))

    Args:
        xyz: (N,3) toolpath points in placed STL coordinates (mm), ordered as in export.
        slice_height_mm: slice height (mm).
        z0_mm: Z origin for layer indexing. Defaults to min(Z).
        max_jump_mm: if provided, segments longer than this are treated as travel jumps and ignored.
        min_xy_segment_mm: ignore near-vertical / tiny XY segments when computing in-plane direction.
        weights: optional (N,) weights per point to apply to segment (uses average of endpoints).
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")
    if slice_height_mm <= 0:
        raise ValueError("slice_height_mm must be > 0")
    if xyz.shape[0] < 2:
        return []

    z0 = float(np.min(xyz[:, 2])) if z0_mm is None else float(z0_mm)
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

        vx, vy = float(seg[0]), float(seg[1])
        len_xy = math.hypot(vx, vy)
        if len_xy < float(min_xy_segment_mm):
            continue

        # layer index by segment midpoint Z
        z_mid = float(0.5 * (a[2] + b[2]))
        layer_id = int(round((z_mid - z0) / float(slice_height_mm)))

        theta = math.atan2(vy, vx)
        c2 = math.cos(2.0 * theta)
        s2 = math.sin(2.0 * theta)

        w = len_xy
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
            dir_xyz = (math.cos(theta), math.sin(theta), 0.0)

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
    max_jump_mm: float | None = None,
    min_xy_segment_mm: float = 1e-3,
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
        min_xy_segment_mm: ignore near-vertical / tiny XY segments when computing in-plane direction.
        type_filter: toolpath point Type to use (default: 1 = model).
        weight_by_bead_area: if True, weights segments by avg bead_area of endpoints.
    """
    if slice_height_mm <= 0:
        raise ValueError("slice_height_mm must be > 0")

    z0 = float(z0_mm)
    max_jump = float(max_jump_mm) if max_jump_mm is not None else None

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

        ax, ay, az = float(prev.x), float(prev.y), float(prev.z)
        bx, by, bz = float(pt.x), float(pt.y), float(pt.z)
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        if seg_len <= 1e-12:
            prev = pt
            continue
        if max_jump is not None and seg_len > max_jump:
            prev = pt
            continue

        len_xy = math.hypot(dx, dy)
        if len_xy < float(min_xy_segment_mm):
            prev = pt
            continue

        z_mid = 0.5 * (az + bz)
        layer_id = int(round((z_mid - z0) / float(slice_height_mm)))

        theta = math.atan2(dy, dx)
        c2 = math.cos(2.0 * theta)
        s2 = math.sin(2.0 * theta)

        w = float(len_xy)
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
            dir_xyz = (math.cos(theta), math.sin(theta), 0.0)

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
