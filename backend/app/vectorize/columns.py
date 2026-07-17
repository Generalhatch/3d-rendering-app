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

    # v4 shape classifier (Cloud2BIM-inspired).
    # A circle inscribed in a square has fill_ratio = π/4 ≈ 0.785; a true
    # square has fill_ratio ≈ 1.0.  We classify a column as ROUND if its
    # min-area-rect aspect is near 1 (≤ ``round_max_aspect``) AND its fill
    # ratio is in the circle window (``round_fill_lo`` to ``round_fill_hi``).
    # The DXF writer then emits a CIRCLE entity instead of an LWPOLYLINE,
    # producing the cleaner symbol used in commercial CAD drawings.
    round_classifier_enable: bool = True
    round_max_aspect: float = 1.20
    round_fill_lo: float = 0.65
    round_fill_hi: float = 0.92

    # Stricter base criteria (v4): the v3 defaults let counters /
    # workbenches through as "columns" because their fill_ratio came in
    # just above 0.55 and their aspect just under 2.5.  Real columns
    # rarely exceed 80 cm on a side; raise the bar.
    # (Override via the public API for industrial scans.)


@dataclass
class DetectedColumn:
    """One detected column footprint.

    ``corners`` is the bounding rectangle (always 4 vertices) so the DXF
    writer can fall back to a polyline regardless of shape.  When
    ``is_round`` is True, the DXF writer additionally emits a CIRCLE
    entity centred on ``centre_m`` with radius = mean(size_m) / 2.
    """
    corners: np.ndarray   # (4, 2) world coords, CCW
    centre_m: tuple[float, float]
    size_m: tuple[float, float]   # (long_side, short_side)
    fill_ratio: float
    is_round: bool = False
    eccentricity: float = 0.0     # 0 = circle / square, → 1 = elongated


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
            # Pixel-centre convention: idx = (world - origin) / res - 0.5.
            x1 = int(round((seg[0, 0] - affine.origin_x) / res - 0.5))
            y1 = int(round((seg[0, 1] - affine.origin_y) / res - 0.5))
            x2 = int(round((seg[1, 0] - affine.origin_x) / res - 0.5))
            y2 = int(round((seg[1, 1] - affine.origin_y) / res - 0.5))
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

        # 4b. Shape classifier — round vs rectangular.
        # Method: aspect of min-area rect tells us whether we even *could*
        # be round; fill ratio inside the same rect tells us how much of
        # the rect is filled.  A circle inscribed in a 1:1 rect fills
        # exactly π/4 ≈ 0.785; a true square fills ~1.0; a noisy column
        # blob is somewhere in between.
        rect_fill = float(area_px) / max(1.0, rw_px * rh_px)
        rect_aspect = float(long_px) / float(short_px)
        is_round = False
        if params.round_classifier_enable:
            if (rect_aspect <= params.round_max_aspect and
                    params.round_fill_lo <= rect_fill <= params.round_fill_hi):
                is_round = True

        # Eccentricity from second-moment covariance — diagnostic info,
        # not used for filtering (the rect-fill check above is the
        # primary signal).  0 = perfectly circular / square; → 1 as the
        # shape elongates.
        eccentricity = _component_eccentricity(mask)

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
            is_round=bool(is_round),
            eccentricity=float(eccentricity),
        ))

    return out


def _component_eccentricity(mask: np.ndarray) -> float:
    """Compute the eccentricity of a binary blob via second moments.

    Returns sqrt(1 - λ2/λ1) where λ1 ≥ λ2 are eigenvalues of the central
    covariance matrix.  0.0 → perfectly isotropic (circle / square);
    values closer to 1.0 → elongated shapes.

    Used as a diagnostic (logged in DetectedColumn) rather than a hard
    filter — the rect_fill + rect_aspect tests in detect_columns are
    typically sufficient, but eccentricity helps explain edge cases.
    """
    pts = np.column_stack(np.where(mask > 0))  # (N, 2): rows, cols
    if len(pts) < 4:
        return 0.0
    pts = pts.astype(np.float64) - pts.mean(axis=0)
    cov = (pts.T @ pts) / len(pts)
    try:
        evals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 0.0
    evals = np.sort(evals)[::-1]
    if evals[0] <= 0:
        return 0.0
    ratio = max(0.0, evals[1] / evals[0])
    return float(np.sqrt(1.0 - ratio))


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
            pts_px[i, 0] = int(round((col.corners[i, 0] - affine.origin_x) / res - 0.5))
            pts_px[i, 1] = int(round((col.corners[i, 1] - affine.origin_y) / res - 0.5))
        cv2.polylines(bgr, [pts_px], isClosed=True, color=(255, 0, 220), thickness=2)
    return bgr


def _px_to_world(col: float, row: float, affine: RasterAffine) -> tuple[float, float]:
    """Pixel-centre convention, matching ``RasterAffine.pixel_to_world``."""
    return (
        affine.origin_x + (col + 0.5) * affine.resolution_m_per_px,
        affine.origin_y + (row + 0.5) * affine.resolution_m_per_px,
    )
