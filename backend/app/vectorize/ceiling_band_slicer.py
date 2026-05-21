"""Ceiling-adjacent wall slice — the cleanest possible wall raster.

What and why
------------
Cloud2BIM 2025 (Zbirovský & Nežerka, *Automation in Construction*) made a
deceptively simple observation that turns out to dominate every chest-height
slicing strategy: *slice the point cloud near the top of the room*.

At 1.9–2.3 m above the floor:

  - Walls are still present (they reach the ceiling).
  - Almost all furniture is below (file cabinets ≤ 1.8 m, cubicle stops at
    1.5 m, kitchen cabinets cap at 1.95 m, the tallest standing desk maxes
    out around 1.25 m raised).
  - Doors are closed and topped out at 2.03 m (the standard interior door
    height) — meaning above 2.03 m, the open *or* closed door slab is gone
    and only the wall around it remains.  This eliminates the entire
    "re-slice above the door" branch we have in v3.
  - Ducts, beams, and HVAC are typically above 2.4 m.  Our ceiling band
    stops at 2.3 m, so it stays in the wall band.

The ceiling-band slice is best used IN COMBINATION with the density slicer:

  - Density slicer (0.3–2.2 m vertical column score) is the *recall* layer —
    it sees every wall that has any continuous vertical extent.  But it
    also lights up tall furniture (5-shelf bookcase running floor → ceiling
    looks identical to a wall to the density score alone).
  - Ceiling band is the *precision* layer — it only fires on things that
    actually reach to ceiling height.  Bookcases stop ~1.95 m, so they don't.

The pipeline runs both and ANDs the resulting binary masks.  The intersection
is the cleanest wall mask achievable with no semantic ML at all.

Why this is its own module
--------------------------
``slicer.slice_to_raster`` already does single-elevation slicing.  We could
just call it with the right elevation and slab thickness.  But:

1. The defaults differ (0.4 m thick band, not 0.2 m centred slab).
2. The output is meant to be ANDed with the density mask — we want a small
   wrapper that handles raster-canvas alignment between the two slicers
   (they may pick slightly different XY extents).
3. Having a named module makes the SSE progress event readable
   ("ceiling_band: ..." instead of "slice: at 2.10 m").

The wrapper resamples the slice onto the density slicer's affine so the AND
operation is a single ``np.logical_and``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from .slicer import MAX_RASTER_DIM_PX, RasterAffine, SliceResult


@dataclass
class CeilingBandParams:
    """Parameters for :func:`slice_ceiling_band`.

    Defaults assume metres + typical commercial / residential ceiling height
    (≥ 2.4 m clearance).  For low-ceiling industrial or older residential
    spaces, lower ``z_hi_m`` to stay clear of beams.
    """
    z_lo_m: float = 1.90           # m above floor
    z_hi_m: float = 2.30           # m above floor
    resolution_m_per_px: float = 0.01
    bbox_padding_m: float = 0.50
    # Morphological CLOSE to bridge scanner gaps before the AND with the
    # density mask.  3 px ≈ 3 cm at 1 cm/px — well below wall thickness.
    close_kernel_px: int = 3


def slice_ceiling_band(
    pcd: o3d.geometry.PointCloud,
    floor_z: float,
    axis_idx: int = 2,
    params: CeilingBandParams | None = None,
    target_affine: RasterAffine | None = None,
) -> SliceResult:
    """Project points from the ceiling-adjacent band to a binary raster.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Source cloud (post-downsample; we don't re-downsample here).
    floor_z : float
        Detected floor elevation in the cloud's frame.
    axis_idx : int
        Vertical-axis index.  2 = Z-up, 1 = Y-up.
    params : CeilingBandParams, optional
        Override defaults.
    target_affine : RasterAffine, optional
        If supplied, the output raster is resampled onto this affine so it
        can be ANDed directly with another raster (e.g. the density slicer
        output).  If omitted, the slicer picks its own canvas from the band
        points' XY extent.

    Returns
    -------
    SliceResult

    Raises
    ------
    ValueError
        If the band is empty or the canvas would exceed the size cap.
    """
    if axis_idx not in (1, 2):
        raise ValueError(f"axis_idx must be 1 (Y-up) or 2 (Z-up); got {axis_idx}")
    if params is None:
        params = CeilingBandParams()
    if params.z_hi_m <= params.z_lo_m:
        raise ValueError(
            f"z_hi_m ({params.z_hi_m}) must exceed z_lo_m ({params.z_lo_m})"
        )

    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError("ceiling_band: point cloud is empty")

    plane_cols = (0, 1) if axis_idx == 2 else (0, 2)
    z = pts[:, axis_idx]
    z_lo_abs = floor_z + params.z_lo_m
    z_hi_abs = floor_z + params.z_hi_m
    mask = (z >= z_lo_abs) & (z <= z_hi_abs)
    n_in_band = int(mask.sum())
    if n_in_band == 0:
        raise ValueError(
            f"ceiling_band: no points in [{z_lo_abs:.2f}, {z_hi_abs:.2f}] m "
            f"(cloud z range: [{float(z.min()):.2f}, {float(z.max()):.2f}] m). "
            f"Try lowering z_lo_m / z_hi_m if the ceiling is < 2.4 m."
        )

    band_xy = pts[np.ix_(mask, plane_cols)]
    res = params.resolution_m_per_px

    # Pick canvas: either honour the supplied affine or carve our own from the
    # band's extent.  Honouring the supplied affine is what lets the caller AND
    # us with the density-slicer raster directly — no resampling needed.
    if target_affine is not None:
        affine = target_affine
        x_min = affine.origin_x
        y_min = affine.origin_y
        width_px = affine.width_px
        height_px = affine.height_px
        # If the target affine uses a different resolution, fall back to its.
        res = affine.resolution_m_per_px
    else:
        x_min = float(band_xy[:, 0].min()) - params.bbox_padding_m
        y_min = float(band_xy[:, 1].min()) - params.bbox_padding_m
        x_max = float(band_xy[:, 0].max()) + params.bbox_padding_m
        y_max = float(band_xy[:, 1].max()) + params.bbox_padding_m
        width_px = int(np.ceil((x_max - x_min) / res))
        height_px = int(np.ceil((y_max - y_min) / res))
        if width_px > MAX_RASTER_DIM_PX or height_px > MAX_RASTER_DIM_PX:
            raise ValueError(
                f"ceiling raster {width_px}×{height_px} exceeds cap "
                f"{MAX_RASTER_DIM_PX}; raise resolution_m_per_px"
            )
        if width_px < 4 or height_px < 4:
            raise ValueError(
                f"ceiling raster {width_px}×{height_px} too small; "
                f"check band elevations or scan coverage"
            )
        affine = RasterAffine(
            origin_x=x_min,
            origin_y=y_min,
            resolution_m_per_px=res,
            width_px=width_px,
            height_px=height_px,
        )

    cols = np.floor((band_xy[:, 0] - x_min) / res).astype(np.int32)
    rows = np.floor((band_xy[:, 1] - y_min) / res).astype(np.int32)
    in_canvas = (cols >= 0) & (cols < width_px) & (rows >= 0) & (rows < height_px)
    cols = cols[in_canvas]
    rows = rows[in_canvas]

    image = np.zeros((height_px, width_px), dtype=np.uint8)
    image[rows, cols] = 255

    # CLOSE bridges sub-cm scanner gaps so the AND with the density mask
    # doesn't punch through walls where one or two pixels are missing in
    # the ceiling band (very common near light fixtures / smoke detectors).
    if params.close_kernel_px > 1:
        k = params.close_kernel_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    centre_z = (z_lo_abs + z_hi_abs) / 2.0
    return SliceResult(
        image=image,
        affine=affine,
        elevation_m=float(centre_z),
        slab_thickness_m=float(params.z_hi_m - params.z_lo_m),
        axis_idx=axis_idx,
        n_points_in_slab=n_in_band,
        elevations_used=[float(z_lo_abs), float(z_hi_abs)],
    )


def gate_with_ceiling(
    density_mask: np.ndarray,
    ceiling_mask: np.ndarray,
    require_overlap_px: int = 50_000,
) -> np.ndarray:
    """Logical-AND the density wall mask with the ceiling-band mask.

    Returns the AND if there's enough overlap (at least ``require_overlap_px``
    foreground pixels in the intersection), otherwise falls back to the
    density mask alone.  The fallback protects against pathological cases:

      - Scan was taken with the scanner tripod higher than the ceiling
        (band has no points).
      - Operator mis-specified ceiling_band_lo_m / hi_m for a low-ceiling
        space.
      - Scan covers only a partial floor where the ceiling is missing.

    If we silently emitted an empty AND in those cases, downstream wall
    detection would find nothing and the operator would see an empty DXF.
    Better to log "ceiling band sparse, falling back to density alone"
    in the SSE progress and let them re-tune.

    Parameters
    ----------
    density_mask : (H, W) uint8
        Binary mask from ``density_slicer.slice_to_raster_density``.
        Values 0 or 255.
    ceiling_mask : (H, W) uint8
        Binary mask from :func:`slice_ceiling_band`.  Same shape + affine.
    require_overlap_px : int
        Minimum number of foreground pixels the AND must contain to be
        accepted.  Default 50 000 ≈ 5 m² of wall pixels at 1 cm/px.

    Returns
    -------
    (H, W) uint8 — the gated wall mask.
    """
    if density_mask.shape != ceiling_mask.shape:
        raise ValueError(
            f"shape mismatch: density {density_mask.shape} vs ceiling "
            f"{ceiling_mask.shape}.  Pass target_affine to slice_ceiling_band "
            f"to align canvases."
        )
    anded = (density_mask > 0) & (ceiling_mask > 0)
    overlap_px = int(anded.sum())
    if overlap_px < require_overlap_px:
        # Sparse band — fall back to density.  Caller logs this case.
        return density_mask
    return (anded.astype(np.uint8) * 255)


def ceiling_band_summary(result: SliceResult) -> str:
    """One-line human-readable summary for SSE progress."""
    return (
        f"ceiling band ({result.elevation_m - result.slab_thickness_m / 2:.2f}–"
        f"{result.elevation_m + result.slab_thickness_m / 2:.2f} m): "
        f"{result.n_points_in_slab:,} points in "
        f"{result.affine.width_px}×{result.affine.height_px} px"
    )
