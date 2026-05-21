"""Vertical-column density slicer — discriminates walls from furniture.

The legacy ``slicer.slice_to_raster`` paints every point in a horizontal slab
as a white pixel.  At any practical slab thickness, that means every dense
*horizontal* feature (filing cabinets, monitors, cubicle dividers, bookshelf
sides) lights up identically to a wall — they all have plenty of points at
1.4–1.8 m.

This module fixes that with a different observation: a **wall** has points at
every height from skirting board to ceiling — its vertical column is **tall
and continuous**.  A piece of furniture has points only across a narrow
height band — its vertical column is **short**.  By scoring every XY cell
on the *fraction of vertical bins that are occupied*, walls become bright
and furniture stays dim.

The output is a single 8-bit grayscale image that downstream stages
(preprocess + line detect) already understand.  No format change required.

Algorithm
---------
1. Collect all points in the wall band ``floor + z_lo → floor + z_hi``.
2. Bin into a 3-D voxel grid: XY at ``resolution_m_per_px``, Z at
   ``z_bin_m`` (default 0.10 m).
3. For each (X, Y) cell, count how many Z bins are occupied (`bin_count`).
   The total possible is ``(z_hi - z_lo) / z_bin_m``.
4. ``density[y, x] = bin_count / total_z_bins`` ∈ [0, 1].
5. Scale to uint8 (clip to 98th percentile to suppress reflectors).
6. Optional threshold-to-binary for callers that need a mask.

Properties vs. legacy slicer
----------------------------
- A wall that runs from floor to ceiling lights up with density ≈ 1.0.
- A 1.2-m filing cabinet over a 1.9-m band → density ≈ 0.55 — still bright
  but visibly dimmer than wall pixels.
- A desk top over a 0.05-m band → density ≈ 0.03 — essentially invisible.
- A monitor screen with a thin vertical extent at one Z → density ≈ 0.1 —
  noise floor.

A simple threshold at 0.55–0.65 yields a wall mask that excludes most
furniture without needing ML.

This is the SAME PNG format the legacy slicer emits.  Downstream code does
not need to know which slicer was used.  See ``SliceResult`` returned by
:func:`slice_to_raster_density` — it's structurally identical to
``slicer.SliceResult`` for drop-in compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from .slicer import MAX_RASTER_DIM_PX, RasterAffine, SliceResult


@dataclass
class DensitySliceParams:
    """Parameters for :func:`slice_to_raster_density`.

    The defaults assume metres + typical interior ceiling height (~2.7 m).
    """
    # Vertical band relative to floor_z.  Default 0.30–2.20 m covers
    # everything from above the skirting to the typical ceiling-suspended
    # fixture line.  Walls always extend through this whole range; furniture
    # never does.
    z_lo_m: float = 0.30
    z_hi_m: float = 2.20
    z_bin_m: float = 0.10            # 10 cm bins; 19 bins over a 1.9 m range
    resolution_m_per_px: float = 0.01
    bbox_padding_m: float = 0.50
    # When converting the float density [0, 1] to uint8, clip to the
    # n-th percentile to avoid one super-bright reflector point dominating
    # the dynamic range.
    intensity_clip_percentile: float = 98.0
    # Final binary threshold on density (0–1).  Walls reliably exceed 0.55;
    # tall furniture (filing cabinets, half-walls) lands 0.45–0.60.  0.55 is
    # the empirical sweet spot.
    binary_threshold: float = 0.55


def slice_to_raster_density(
    pcd: o3d.geometry.PointCloud,
    floor_z: float,
    axis_idx: int = 2,
    params: DensitySliceParams | None = None,
) -> SliceResult:
    """Build a vertical-column density raster from a point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Source cloud.  Best results when pre-filtered to vertical surfaces
        (see :func:`normals.filter_to_vertical_surfaces`) — that drops the
        floor and ceiling points which would otherwise contribute a constant
        offset to the density score, but the slicer still works without
        pre-filtering.
    floor_z : float
        Floor elevation in the cloud's frame.  Used as the origin for the
        vertical band.
    axis_idx : int
        Vertical-axis index.  2 = Z-up, 1 = Y-up.
    params : DensitySliceParams, optional

    Returns
    -------
    SliceResult
        - ``image``: uint8 (H, W) — binary wall mask (255 where
          density >= binary_threshold, 0 elsewhere).  Same shape downstream
          stages expect from the legacy slicer.
        - ``affine``: world↔pixel mapping.
        - ``elevation_m``: nominal slice centre = midpoint of the band.
        - ``elevations_used``: list — all included band bin centres.

    Raises
    ------
    ValueError
        If the band is empty or the raster would exceed the size cap.
    """
    if axis_idx not in (1, 2):
        raise ValueError(f"axis_idx must be 1 or 2; got {axis_idx}")
    if params is None:
        params = DensitySliceParams()
    if params.z_hi_m <= params.z_lo_m:
        raise ValueError(
            f"z_hi_m ({params.z_hi_m}) must exceed z_lo_m ({params.z_lo_m})"
        )

    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError("density slicer: input cloud is empty")

    plane_cols = (0, 1) if axis_idx == 2 else (0, 2)
    z = pts[:, axis_idx]
    z_lo_abs = floor_z + params.z_lo_m
    z_hi_abs = floor_z + params.z_hi_m
    band_mask = (z >= z_lo_abs) & (z <= z_hi_abs)
    n_in_band = int(band_mask.sum())
    if n_in_band == 0:
        raise ValueError(
            f"density slicer: no points in band [{z_lo_abs:.2f}, {z_hi_abs:.2f}] m "
            f"(point z range: [{float(z.min()):.2f}, {float(z.max()):.2f}] m)"
        )

    band_pts_xy = pts[np.ix_(band_mask, plane_cols)]
    band_pts_z = z[band_mask]

    # Raster bounding box from the band's XY extent + padding.
    x_min = float(band_pts_xy[:, 0].min()) - params.bbox_padding_m
    y_min = float(band_pts_xy[:, 1].min()) - params.bbox_padding_m
    x_max = float(band_pts_xy[:, 0].max()) + params.bbox_padding_m
    y_max = float(band_pts_xy[:, 1].max()) + params.bbox_padding_m

    res = params.resolution_m_per_px
    width_px = int(np.ceil((x_max - x_min) / res))
    height_px = int(np.ceil((y_max - y_min) / res))
    if width_px > MAX_RASTER_DIM_PX or height_px > MAX_RASTER_DIM_PX:
        raise ValueError(
            f"density raster {width_px}×{height_px} px exceeds cap "
            f"{MAX_RASTER_DIM_PX}; raise resolution_m_per_px"
        )
    if width_px < 4 or height_px < 4:
        raise ValueError(
            f"density raster {width_px}×{height_px} px too small; "
            f"check band elevations + resolution"
        )

    # Per-point voxel indices.
    cols = np.clip(
        np.floor((band_pts_xy[:, 0] - x_min) / res).astype(np.int32),
        0, width_px - 1,
    )
    rows = np.clip(
        np.floor((band_pts_xy[:, 1] - y_min) / res).astype(np.int32),
        0, height_px - 1,
    )
    z_bins = np.clip(
        np.floor((band_pts_z - z_lo_abs) / params.z_bin_m).astype(np.int32),
        0, int(np.ceil((params.z_hi_m - params.z_lo_m) / params.z_bin_m)) - 1,
    )
    n_z_bins = int(np.ceil((params.z_hi_m - params.z_lo_m) / params.z_bin_m))

    # The fast path: collapse (row, col, z) into a single 64-bit integer key,
    # take np.unique to dedup (col, row, z) tuples, then count per (col, row).
    flat_keys = (
        rows.astype(np.int64) * np.int64(width_px) * np.int64(n_z_bins)
        + cols.astype(np.int64) * np.int64(n_z_bins)
        + z_bins.astype(np.int64)
    )
    unique_keys = np.unique(flat_keys)
    # Decompose unique_keys back into (row, col, z_bin) — only the (row, col)
    # part matters for counting.
    u_rows = (unique_keys // (np.int64(width_px) * np.int64(n_z_bins))).astype(np.int32)
    u_cols = (
        (unique_keys // np.int64(n_z_bins)) % np.int64(width_px)
    ).astype(np.int32)

    # Count occupied Z bins per (row, col).
    density_count = np.zeros((height_px, width_px), dtype=np.int32)
    np.add.at(density_count, (u_rows, u_cols), 1)

    # Convert to [0, 1] density (fraction of vertical bins occupied).
    density = density_count.astype(np.float32) / max(1, n_z_bins)

    # Binary mask = pixels with density above the wall threshold.
    binary_mask = (density >= params.binary_threshold).astype(np.uint8) * 255

    # Persist the float density as a sidecar uint8 too — useful for the
    # operator to see what we saw (and for the editor as an alternate
    # backdrop).  We don't return this — the pipeline reads it from disk
    # via ``slice_cleaned.png``/``slice.png`` paths.
    affine = RasterAffine(
        origin_x=x_min,
        origin_y=y_min,
        resolution_m_per_px=res,
        width_px=width_px,
        height_px=height_px,
    )

    centre_z = (z_lo_abs + z_hi_abs) / 2.0
    elev_bins = [
        floor_z + params.z_lo_m + (i + 0.5) * params.z_bin_m
        for i in range(n_z_bins)
    ]

    return SliceResult(
        image=binary_mask,
        affine=affine,
        elevation_m=float(centre_z),
        slab_thickness_m=float(params.z_hi_m - params.z_lo_m),
        axis_idx=axis_idx,
        n_points_in_slab=n_in_band,
        elevations_used=elev_bins,
    )


def density_image_uint8(
    pcd: o3d.geometry.PointCloud,
    floor_z: float,
    axis_idx: int = 2,
    params: DensitySliceParams | None = None,
) -> tuple[np.ndarray, RasterAffine]:
    """Return the raw density image as uint8 (for visualisation) + affine.

    This is the "show the operator what we saw" view — pre-threshold density
    mapped to a percentile-clipped 8-bit gradient.  Use
    :func:`slice_to_raster_density` for the binary mask that the detector
    actually runs on.
    """
    # Reuse the heavy work by replicating the band → density step here.
    # Slight code duplication is preferable to making the public function
    # return both a binary and a float image (the SliceResult contract
    # would have to change).
    if axis_idx not in (1, 2):
        raise ValueError(f"axis_idx must be 1 or 2; got {axis_idx}")
    if params is None:
        params = DensitySliceParams()

    pts = np.asarray(pcd.points, dtype=np.float64)
    plane_cols = (0, 1) if axis_idx == 2 else (0, 2)
    z = pts[:, axis_idx]
    z_lo_abs = floor_z + params.z_lo_m
    z_hi_abs = floor_z + params.z_hi_m
    mask = (z >= z_lo_abs) & (z <= z_hi_abs)
    pts2d = pts[np.ix_(mask, plane_cols)]
    pts_z = z[mask]
    if len(pts2d) == 0:
        raise ValueError("density_image_uint8: empty band")

    x_min = float(pts2d[:, 0].min()) - params.bbox_padding_m
    y_min = float(pts2d[:, 1].min()) - params.bbox_padding_m
    x_max = float(pts2d[:, 0].max()) + params.bbox_padding_m
    y_max = float(pts2d[:, 1].max()) + params.bbox_padding_m
    res = params.resolution_m_per_px
    width_px = int(np.ceil((x_max - x_min) / res))
    height_px = int(np.ceil((y_max - y_min) / res))

    cols = np.clip(
        np.floor((pts2d[:, 0] - x_min) / res).astype(np.int32),
        0, width_px - 1,
    )
    rows = np.clip(
        np.floor((pts2d[:, 1] - y_min) / res).astype(np.int32),
        0, height_px - 1,
    )
    n_z_bins = int(np.ceil((params.z_hi_m - params.z_lo_m) / params.z_bin_m))
    z_bins = np.clip(
        np.floor((pts_z - z_lo_abs) / params.z_bin_m).astype(np.int32),
        0, n_z_bins - 1,
    )
    flat = (
        rows.astype(np.int64) * np.int64(width_px) * np.int64(n_z_bins)
        + cols.astype(np.int64) * np.int64(n_z_bins)
        + z_bins.astype(np.int64)
    )
    uniq = np.unique(flat)
    u_rows = (uniq // (np.int64(width_px) * np.int64(n_z_bins))).astype(np.int32)
    u_cols = ((uniq // np.int64(n_z_bins)) % np.int64(width_px)).astype(np.int32)

    counts = np.zeros((height_px, width_px), dtype=np.int32)
    np.add.at(counts, (u_rows, u_cols), 1)

    # Density to uint8, percentile-clipped.
    density = counts.astype(np.float32) / max(1, n_z_bins)
    clip_at = float(np.percentile(density[density > 0], params.intensity_clip_percentile)) \
        if (density > 0).any() else 1.0
    clip_at = max(clip_at, 0.05)
    img = np.clip(density / clip_at * 255.0, 0, 255).astype(np.uint8)

    affine = RasterAffine(
        origin_x=x_min,
        origin_y=y_min,
        resolution_m_per_px=res,
        width_px=width_px,
        height_px=height_px,
    )
    return img, affine
