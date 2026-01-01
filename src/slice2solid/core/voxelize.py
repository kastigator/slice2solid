from __future__ import annotations

import dataclasses
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path

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


def _estimate_nbytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    try:
        return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)
    except Exception:
        return 0


def _default_memmap_threshold_bytes() -> int:
    # Quality-first mode: prefer stability, but avoid disk I/O for small runs.
    # Can be overridden via env S2S_MEMMAP_THRESHOLD_MB.
    try:
        mb = int(os.environ.get("S2S_MEMMAP_THRESHOLD_MB", "256"))
        mb = max(0, mb)
    except Exception:
        mb = 256
    return int(mb) * 1024 * 1024


def _resolve_scratch_dir(scratch_dir: str | Path | None) -> Path:
    if scratch_dir is None:
        base = os.environ.get("S2S_SCRATCH_DIR", "")
        scratch_dir = base.strip() if isinstance(base, str) else ""
    p = Path(scratch_dir) if scratch_dir else Path(os.environ.get("TEMP", "")) / "slice2solid_scratch"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _create_memmap(
    *,
    scratch_dir: str | Path | None,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.memmap:
    p = _resolve_scratch_dir(scratch_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{name}_{stamp}_{os.getpid()}_{abs(hash((shape, str(dtype)))) & 0xFFFF_FFFF}.dat"
    path = p / fname
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape, order="C")


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
    ignore_zero_factor: bool = True,
    ignore_zero_bead_area: bool = True,
    *,
    scratch_dir: str | Path | None = None,
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

    # Quality-first stability: for big volumes, put occupancy on disk to avoid OOM.
    occ_bytes = _estimate_nbytes(shape, np.dtype(bool))
    if occ_bytes >= _default_memmap_threshold_bytes():
        occupied = _create_memmap(scratch_dir=scratch_dir, name="occupied_bool", shape=shape, dtype=np.dtype(bool))
        occupied[:] = False
    else:
        occupied = np.zeros(shape, dtype=bool)

    # Sphere offset stencils (voxel-space) for different radii.
    #
    # IMPORTANT: do not precompute all radii up to max upfront.
    # For small voxel_size and moderate radii this can explode in both time and memory
    # (sum of sphere volumes across radii grows ~O(r^4)).
    r_max = float(max_radius_mm) if max_radius_mm is not None else 2.0  # safe default
    max_r_vox = max(1, int(math.ceil(r_max / voxel_size)))
    stencils: dict[int, np.ndarray] = {}
    stencil_lru: list[int] = []
    max_cached_stencils = 12

    def _get_stencil(rv: int) -> np.ndarray:
        rv = max(1, min(int(rv), int(max_r_vox)))
        hit = stencils.get(rv)
        if hit is not None:
            try:
                stencil_lru.remove(rv)
            except Exception:
                pass
            stencil_lru.append(rv)
            return hit
        s = _sphere_offsets(rv)
        stencils[rv] = s
        stencil_lru.append(rv)
        if len(stencil_lru) > int(max_cached_stencils):
            old = stencil_lru.pop(0)
            try:
                del stencils[old]
            except Exception:
                pass
        return s

    prev_xyz = None
    prev_area = None
    prev_type = None

    for pt in points:
        if pt.type != type_filter:
            prev_xyz = None
            prev_area = None
            prev_type = None
            continue

        if ignore_zero_factor and float(pt.factor) <= 1e-12:
            prev_xyz = None
            prev_area = None
            prev_type = None
            continue

        if ignore_zero_bead_area and float(pt.bead_area) <= 1e-12:
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
        stencil = _get_stencil(r_vox)

        # sample along segment
        # Adaptive sampling:
        # - for thin beads (r_mm < voxel_size) we keep dense sampling (0.5 voxel)
        # - for thick beads we can step proportionally to radius to reduce work, while keeping overlap
        step = max(voxel_size * 0.5, float(r_mm) * 0.5)
        n_steps = max(1, int(math.ceil(seg_len / step)))
        ts = np.linspace(0.0, 1.0, n_steps + 1, dtype=float)
        samples = a[None, :] + ts[:, None] * seg[None, :]

        # convert samples to voxel indices
        idx = np.floor((samples - origin[None, :]) / voxel_size).astype(int)
        # Many samples map to the same voxel; uniquify to avoid redundant stamping.
        if idx.shape[0] > 64:
            try:
                idx = np.unique(idx, axis=0)
            except Exception:
                pass

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
    scratch_dir: str | Path | None = None,
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
            sigma = float(volume_smooth_sigma_vox) if volume_smooth_sigma_vox else 0.0
            if sigma > 0:
                # High-RAM path: use disk-backed arrays when volume is large to avoid OOM.
                # We keep quality: same sigma and same marching cubes resolution.
                vol_shape = tuple(int(x) for x in occ_ds.shape)
                vol_bytes = _estimate_nbytes(vol_shape, np.dtype(np.float32))
                use_memmap = vol_bytes >= _default_memmap_threshold_bytes()

                if use_memmap:
                    vol = _create_memmap(scratch_dir=scratch_dir, name="vol_f32", shape=vol_shape, dtype=np.dtype(np.float32))
                    vol[:] = occ_ds  # dtype cast happens during assignment without a full temp array
                    out = _create_memmap(
                        scratch_dir=scratch_dir, name="vol_f32_blur", shape=vol_shape, dtype=np.dtype(np.float32)
                    )
                    ndimage.gaussian_filter(vol, sigma=sigma, mode="nearest", output=out)
                    # Build a contiguous (z,y,x) volume for skimage.
                    zyx_shape = (vol_shape[2], vol_shape[1], vol_shape[0])
                    vol_zyx = _create_memmap(
                        scratch_dir=scratch_dir, name="vol_zyx_f32", shape=zyx_shape, dtype=np.dtype(np.float32)
                    )
                    vol_zyx[:] = np.transpose(out, (2, 1, 0))
                else:
                    vol = occ_ds.astype(np.float32, copy=True)
                    vol = ndimage.gaussian_filter(vol, sigma=sigma, mode="nearest")
                    vol_zyx = np.transpose(vol, (2, 1, 0))
            else:
                # Avoid 4x RAM blow-up when no smoothing is requested.
                if occ_ds.dtype == bool and int(getattr(occ_ds.dtype, "itemsize", 1)) == 1:
                    vol_u8 = occ_ds.view(np.uint8)
                else:
                    vol_u8 = occ_ds.astype(np.uint8, copy=False)
                vol_zyx = np.transpose(vol_u8, (2, 1, 0))
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
