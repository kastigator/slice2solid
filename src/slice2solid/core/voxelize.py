from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from slice2solid.core.insight_simulation import ToolpathPoint

try:
    from numpy.core._exceptions import _ArrayMemoryError as _NumpyArrayMemoryError
except Exception:  # pragma: no cover
    _NumpyArrayMemoryError = MemoryError


@dataclasses.dataclass(frozen=True)
class VoxelizationResult:
    voxel_size: float
    origin: np.ndarray  # (3,)
    shape: tuple[int, int, int]  # (nx, ny, nz)
    occupied: np.ndarray  # bool (nx, ny, nz)


def _sphere_offsets(radius_vox: int) -> np.ndarray:
    r = radius_vox
    xs, ys, zs = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    mask = (xs * xs + ys * ys + zs * zs) <= (r * r)
    coords = np.vstack([xs[mask], ys[mask], zs[mask]]).T
    return coords.astype(int)


def voxelize_toolpath(
    points: Iterable[ToolpathPoint],
    voxel_size: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    slice_height: float,
    type_filter: int = 1,
    max_radius_mm: float | None = None,
    max_jump_mm: float | None = None,
    min_component_voxels: int = 0,
) -> VoxelizationResult:
    """
    Builds a voxel occupancy grid by sweeping spheres along toolpath segments.

    - Uses Type filtering (default: model only, Type=1).
    - Converts bead cross-sectional area to an effective radius r via r = sqrt(area/pi).
      This is a crude approximation but stable for MVP visualization.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    if slice_height <= 0:
        raise ValueError("slice_height must be > 0")

    origin = bounds_min.astype(float)
    size = (bounds_max - bounds_min).astype(float)
    shape = tuple(int(math.ceil(s / voxel_size)) + 1 for s in size)
    occupied = np.zeros(shape, dtype=bool)

    # Precompute sphere offset stencils for radii up to max.
    r_max = float(max_radius_mm) if max_radius_mm is not None else 2.0  # safe default
    max_r_vox = max(1, int(math.ceil(r_max / voxel_size)))
    stencils = {rv: _sphere_offsets(rv) for rv in range(1, max_r_vox + 1)}

    prev_xyz = None
    prev_area = None
    prev_type = None

    for pt in points:
        if pt.type != type_filter:
            prev_xyz = None
            prev_area = None
            prev_type = None
            continue

        curr_xyz = np.array([pt.x, pt.y, pt.z], dtype=float)

        if prev_xyz is None:
            prev_xyz = curr_xyz
            prev_area = pt.bead_area
            prev_type = pt.type
            continue

        a = prev_xyz
        b = curr_xyz
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-9:
            prev_xyz = curr_xyz
            prev_area = pt.bead_area
            prev_type = pt.type
            continue
        if max_jump_mm is not None and seg_len > float(max_jump_mm):
            # Likely a travel/jump between separate toolpath chains; do not fill material along it.
            prev_xyz = curr_xyz
            prev_area = pt.bead_area
            prev_type = pt.type
            continue

        # radius estimate from area (mm^2)
        area = float(pt.bead_area if prev_area is None else 0.5 * (pt.bead_area + prev_area))
        r_mm = math.sqrt(max(area, 0.0) / math.pi)
        if max_radius_mm is not None:
            r_mm = min(r_mm, float(max_radius_mm))
        r_vox = max(1, int(math.ceil(r_mm / voxel_size)))
        stencil = stencils.get(r_vox)
        if stencil is None:
            stencil = _sphere_offsets(r_vox)

        # sample along segment
        step = voxel_size * 0.5
        n_steps = max(1, int(math.ceil(seg_len / step)))
        ts = np.linspace(0.0, 1.0, n_steps + 1, dtype=float)
        samples = a[None, :] + ts[:, None] * seg[None, :]

        # convert samples to voxel indices
        idx = np.floor((samples - origin[None, :]) / voxel_size).astype(int)

        for i0, j0, k0 in idx:
            coords = stencil + np.array([i0, j0, k0], dtype=int)
            # clip
            inside = (
                (coords[:, 0] >= 0)
                & (coords[:, 0] < shape[0])
                & (coords[:, 1] >= 0)
                & (coords[:, 1] < shape[1])
                & (coords[:, 2] >= 0)
                & (coords[:, 2] < shape[2])
            )
            coords = coords[inside]
            occupied[coords[:, 0], coords[:, 1], coords[:, 2]] = True

        prev_xyz = curr_xyz
        prev_area = pt.bead_area
        prev_type = pt.type

    return VoxelizationResult(
        voxel_size=voxel_size,
        origin=origin,
        shape=shape,
        occupied=_filter_small_components(occupied, min_component_voxels=min_component_voxels),
    )


def mesh_from_voxels(result: VoxelizationResult) -> trimesh.Trimesh:
    """
    Converts voxel occupancy into a surface mesh using marching cubes.
    """
    return mesh_from_voxels_configured(result)


def mesh_from_voxels_configured(
    result: VoxelizationResult,
    *,
    volume_smooth_sigma_vox: float = 0.0,
    min_component_faces: int = 0,
    max_voxels_for_meshing: int = 60_000_000,
    downsample_factor: int | None = None,
) -> trimesh.Trimesh:
    """
    Converts voxel occupancy into a surface mesh using marching cubes.

    Args:
        volume_smooth_sigma_vox: Optional Gaussian smoothing of the occupancy volume (in voxels).
            This reduces the typical “stair-step/pyramid” look from a purely binary grid.
        min_component_faces: Removes tiny disconnected mesh islands after meshing.
        max_voxels_for_meshing: Upper bound for voxel count used for meshing. If exceeded, the
            occupancy grid is cropped and downsampled before marching cubes to avoid OOM.
        downsample_factor: Optional explicit downsampling factor (>= 1) applied before meshing.
    """
    occ = result.occupied
    if occ.size == 0 or not bool(np.any(occ)):
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)

    # Crop to occupied extent (+1 voxel margin) to reduce memory before meshing.
    nx, ny, nz = occ.shape
    xs = np.flatnonzero(np.any(occ, axis=(1, 2)))
    ys = np.flatnonzero(np.any(occ, axis=(0, 2)))
    zs = np.flatnonzero(np.any(occ, axis=(0, 1)))
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)

    x0 = max(int(xs[0]) - 1, 0)
    x1 = min(int(xs[-1]) + 2, nx)
    y0 = max(int(ys[0]) - 1, 0)
    y1 = min(int(ys[-1]) + 2, ny)
    z0 = max(int(zs[0]) - 1, 0)
    z1 = min(int(zs[-1]) + 2, nz)

    occ = occ[x0:x1, y0:y1, z0:z1]
    origin = result.origin + np.array([x0, y0, z0], dtype=float) * float(result.voxel_size)
    voxel_size = float(result.voxel_size)

    factor = int(downsample_factor) if downsample_factor is not None else 1
    if factor < 1:
        raise ValueError("downsample_factor must be >= 1")
    if downsample_factor is None and int(max_voxels_for_meshing) > 0:
        voxels = int(occ.size)
        if voxels > int(max_voxels_for_meshing):
            factor = int(math.ceil((voxels / float(max_voxels_for_meshing)) ** (1.0 / 3.0)))
            factor = max(2, factor)

    mesh: trimesh.Trimesh | None = None
    used_factor = 1
    last_err: BaseException | None = None
    for extra in (1, 2, 4):
        f = factor * extra
        if f > 64:
            break
        occ_ds = occ if f == 1 else occ[::f, ::f, ::f]
        if min(occ_ds.shape) < 2:
            continue
        try:
            vol = occ_ds.astype(np.float32, copy=True)
            if volume_smooth_sigma_vox and float(volume_smooth_sigma_vox) > 0:
                vol = ndimage.gaussian_filter(vol, sigma=float(volume_smooth_sigma_vox), mode="nearest")
            # skimage expects (z, y, x) ordering; our grid is (x, y, z)
            vol_zyx = np.transpose(vol, (2, 1, 0))
            eff_voxel_size = voxel_size * float(f)
            verts, faces, _normals, _values = measure.marching_cubes(
                vol_zyx, level=0.5, spacing=(eff_voxel_size,) * 3
            )
            # verts are in (z,y,x) space, convert to (x,y,z)
            verts_xyz = np.stack([verts[:, 2], verts[:, 1], verts[:, 0]], axis=1)
            verts_xyz += origin[None, :]
            mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)
            used_factor = int(f)
            break
        except (MemoryError, _NumpyArrayMemoryError) as e:
            last_err = e
            continue

    if mesh is None:
        raise MemoryError("Out of memory during meshing. Increase voxel size or reduce grid size.") from last_err

    mesh.metadata = dict(mesh.metadata or {})
    mesh.metadata.update(
        {
            "meshing_downsample_factor": int(used_factor),
            "meshing_voxel_size_mm": float(voxel_size * float(used_factor)),
            "meshing_cropped_shape": tuple(int(x) for x in occ.shape),
            "meshing_shape": tuple(int(x) for x in (occ if used_factor == 1 else occ[::used_factor, ::used_factor, ::used_factor]).shape),
        }
    )

    if min_component_faces and int(min_component_faces) > 0:
        comps = mesh.split(only_watertight=False)
        comps = [c for c in comps if len(c.faces) >= int(min_component_faces)]
        if comps:
            mesh = trimesh.util.concatenate(comps)
        else:
            mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)

    return mesh


def _filter_small_components(grid: np.ndarray, *, min_component_voxels: int) -> np.ndarray:
    if min_component_voxels <= 1:
        return grid
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, num = ndimage.label(grid, structure=structure)
    if num <= 1:
        return grid
    counts = np.bincount(labeled.ravel())
    # label 0 is background
    remove = np.where(counts < int(min_component_voxels))[0]
    remove = remove[remove != 0]
    if remove.size == 0:
        return grid
    mask_remove = np.isin(labeled, remove)
    out = grid.copy()
    out[mask_remove] = False
    return out
