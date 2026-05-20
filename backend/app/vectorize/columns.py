"""Classical column detection from the cleaned raster.

How it works
------------
After the wall detector has converged on the room's line-shaped features,
the *remaining* foreground pixels that are NOT explained by any wall line
fall into a few archetypes:

  - **Columns** — small, isolated, roughly-square blobs (typical structural
    columns: 30–60 cm square; commercial spaces sometimes 25–80 cm).
  - **Furniture & equipment** — large irregular blobs that survived the
    morphology pass (workbenches, file cabinets, server racks, etc.).
  - **Stippling / scanner noise** — sub-cell speckle the OPEN didn't catch.

Column detection picks the first archetype by:

  1. Subtracting wall coverage from the raster (any pixel within
     :attr:`ColumnParams.wall_clear_m` of a kept wall is removed — those
     pixels "belong to" the wall, not to columns).
  2. Running connected-components analysis on what's left.
  3. Filtering by size, aspect ratio, and area density:
       - bounding-box diagonal in :attr:`min_size_m` .. :attr:`max_size_m`
       - aspect ratio :attr:`max_aspect_ratio` or tighter (no long rectangles)
       - fill ratio (component pixels / bounding-box pixels) ≥
         :attr:`min_fill_ratio` (columns are dense; furniture often hollow)
  4. Emitting one closed-polyline footprint per surviving component, oriented
     to the component's minimum-area bounding rectangle so the rectangle's
     long axis aligns with the column's principal direction.

The output is a list of footprints; each is a *closed* 4-vertex polyline
in world coordinates.  Persisted as segments-of-4 to fit the existing
``segments.json`` shape: 4 segments per column, forming the rectangle.

Future extensions
-----------------
- Round-column detection: fit ellipses to the components and emit circle
  approximations on a separate sub-class.
- Cluster nearby columns into "grids" and emit a column grid annotation.
- Cross-reference floor-plan symbol libraries to label structural vs
  decorative columns.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .slicer import RasterAffine


@dataclass
class ColumnParams:
    """Tunable thresholds for the column detector.

    Defaults target typical commercial / office structural columns.  Wider
    ranges are appropriate for industrial spaces; tighter for residential.
    """
    min_size_m: float = 0.20
    """Lower bound on the column's bounding-box diagonal.  Below 20 cm we're
    almost certainly looking at scanner noise or a small fixture."""

    max_size_m: float = 1.20
    """Upper bound on the column's bounding-box diagonal.  Above 1.2 m we're
    almost certainly looking at a wall stub, workbench, or other furniture."""

    max_aspect_ratio: float = 2.5
    """Maximum ratio of long-side to short-side of the bounding rectangle.
    Real structural columns are usually 1:1 to 2:1; ≤ 2.5:1 keeps detection
    permissive enough for stretched columns at building edges."""

    min_fill_ratio: float = 0.55
    """Component pixels / bounding-box pixels.  Solid columns hit ≈ 0.78
    (circle-in-square) or ≈ 1.0 (rectangular).  Furniture / equipment is
    usually hollow (chair cluster, file cabinet outline) and falls below."""

    wall_clear_m: float = 0.12
    """Dilate kept walls by this radius before subtracting from the raster.
    Sized to cover both faces of a typical 10–15 cm wall plus a slop margin
    so column candidates attached to walls don't get partially erased."""

    rectangle_buffer_m: float = 0.02
    """Outward pad on the emitted bounding rectangle so the footprint reads
    cleanly in CAD (no zero-thickness intersections with adjacent walls)."""


@dataclass
class DetectedColumn:
    """One detected column footprint as a 4-vertex rectangle."""
    corners: np.ndarray   # (4, 2) world coords, CCW
    centre_m: tuple[float, float]
    size_m: tuple[float, float]   # (long_side, short_side)
    fill_ratio: float


def detect_columns(
    cleaned_raster: np.ndarray,
    walls_world: np.ndarray,
    affine: RasterAffine,
    params: ColumnParams | None = None,
) -> list[DetectedColumn]:
    """Find column-shaped blobs in the raster after subtracting wall coverage.

    Parameters
    ----------
    cleaned_raster : np.ndarray
        ``(H, W)`` uint8 image, internal raster convention (row 0 = low world Y).
        Same input the wall detector saw — pre-cleaned, pre-OR-fused.
    walls_world : np.ndarray
        ``(N, 2, 2)`` array of kept wall segments in world coords.
    affine : RasterAffine
        Pixel ↔ world conversion for ``cleaned_raster``.
    params : ColumnParams, optional

    Returns
    -------
    list[DetectedColumn]
        One per surviving blob.  Empty when the raster has no isolated
        column-shaped foreground (the common case for finished residential
        spaces; columns mostly show up in commercial / industrial scans).
    """
    if params is None:
        params = ColumnParams()
    h, w = cleaned_raster.shape
    res = affine.resolution_m_per_px

    # 1. Subtract wall coverage.
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    if len(walls_world) > 0:
        for seg in walls_world:
            x1 = int(round((seg[0, 0] - affine.origin_x) / res))
            y1 = int(round((seg[0, 1] - affine.origin_y) / res))
            x2 = int(round((seg[1, 0] - affine.origin_x) / res))
            y2 = int(round((seg[1, 1] - affine.origin_y) / res))
            cv2.line(wall_mask, (x1, y1), (x2, y2), color=255, thickness=1)
        clear_px = max(1, int(round(params.wall_clear_m / res)))
        # Dilate the wall mask outward so wall-adjacent pixels are removed.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * clear_px + 1, 2 * clear_px + 1),
        )
        wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)

    fg = (cleaned_raster > 0) & (wall_mask == 0)
    residual = (fg.astype(np.uint8)) * 255

    # 2. Connected components on the residual.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    if n_labels <= 1:
        return []

    out: list[DetectedColumn] = []
    min_diag_px = params.min_size_m / res
    max_diag_px = params.max_size_m / res

    for label in range(1, n_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        area_px = stats[label, cv2.CC_STAT_AREA]
        diag_px = float(np.hypot(bw, bh))
        if diag_px < min_diag_px or diag_px > max_diag_px:
            continue

        # 3a. Aspect ratio on axis-aligned bbox first (cheap filter).
        long_side_px = max(bw, bh)
        short_side_px = max(1, min(bw, bh))
        if long_side_px / short_side_px > params.max_aspect_ratio:
            continue

        # 3b. Fill ratio.
        fill = area_px / (bw * bh)
        if fill < params.min_fill_ratio:
            continue

        # 4. Refine to minimum-area rectangle for a tighter footprint.
        mask = (labels[y:y + bh, x:x + bw] == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        contour = max(cnts, key=cv2.contourArea)
        # Shift contour back to full-image coords for cv2.minAreaRect.
        contour = contour + np.array([[x, y]], dtype=np.int32)
        rect = cv2.minAreaRect(contour)   # ((cx, cy), (w, h), angle_deg)
        (cx_px, cy_px), (rw_px, rh_px), _ = rect
        long_px = max(rw_px, rh_px)
        short_px = max(1.0, min(rw_px, rh_px))
        if long_px / short_px > params.max_aspect_ratio:
            continue

        # Re-check size on the min-area rectangle.
        diag_min = float(np.hypot(rw_px, rh_px))
        if diag_min < min_diag_px or diag_min > max_diag_px:
            continue

        # 5. Pad and convert to world coords.
        pad_px = max(1, int(round(params.rectangle_buffer_m / res)))
        padded_rect = (
            (cx_px, cy_px),
            (rw_px + 2 * pad_px, rh_px + 2 * pad_px),
            rect[2],
        )
        box_px = cv2.boxPoints(padded_rect)   # (4, 2) float32
        corners = np.empty((4, 2), dtype=np.float64)
        for i in range(4):
            wx, wy = _px_to_world(box_px[i, 0], box_px[i, 1], affine)
            corners[i] = (wx, wy)
        cx_world, cy_world = _px_to_world(cx_px, cy_px, affine)

        out.append(DetectedColumn(
            corners=corners,
            centre_m=(float(cx_world), float(cy_world)),
            size_m=(float(long_px * res), float(short_px * res)),
            fill_ratio=float(fill),
        ))

    return out


def columns_to_segments(columns: list[DetectedColumn]) -> np.ndarray:
    """Convert detected columns to a flat ``(K*4, 2, 2)`` segment array.

    Each column becomes its 4 bounding-rectangle edges (CCW).  Stored this
    way for compatibility with the existing segment pipeline + DXF writer.
    """
    if not columns:
        return np.zeros((0, 2, 2), dtype=np.float64)
    segs: list[np.ndarray] = []
    for col in columns:
        c = col.corners
        for i in range(4):
            j = (i + 1) % 4
            segs.append(np.array([c[i], c[j]]))
    return np.stack(segs, axis=0)


def render_columns_overlay(
    base_image: np.ndarray,
    walls_px: np.ndarray,
    columns: list[DetectedColumn],
    affine: RasterAffine,
) -> np.ndarray:
    """Render walls (red) + column footprints (magenta) over the raster.

    Used by the pipeline for ``overlay_columns.png`` — quick visual confirm
    that columns landed on real structural posts rather than furniture.
    """
    if base_image.ndim == 2:
        bgr = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = base_image.copy()
    for x1, y1, x2, y2 in walls_px:
        cv2.line(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 220), 2)
    res = affine.resolution_m_per_px
    for col in columns:
        pts_px = np.empty((4, 2), dtype=np.int32)
        for i in range(4):
            pts_px[i, 0] = int(round((col.corners[i, 0] - affine.origin_x) / res))
            pts_px[i, 1] = int(round((col.corners[i, 1] - affine.origin_y) / res))
        cv2.polylines(bgr, [pts_px], isClosed=True, color=(255, 0, 220), thickness=2)
    return bgr


def _px_to_world(col: float, row: float, affine: RasterAffine) -> tuple[float, float]:
    """Inverse of the world→pixel conversion used elsewhere in the pipeline."""
    return (
        affine.origin_x + col * affine.resolution_m_per_px,
        affine.origin_y + row * affine.resolution_m_per_px,
    )
