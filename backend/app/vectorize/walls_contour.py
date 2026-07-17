"""Contour-based wall extraction (Cloud2BIM 2025 strategy).

What this replaces
------------------
The v3 pipeline ran a generic line detector (Hough or FLD) on the cleaned
binary wall mask and got back a *segment soup* — hundreds of short, possibly
parallel, possibly overlapping line segments.  The regularizer then dropped
short ones, merged near-collinear ones, and Manhattan-snapped what was left.
Net result on the demo scan: 688 raw segments → 97 final segments, with
~75 % of the wall pixels left orphaned.

The Cloud2BIM 2025 paper (Zbirovský & Nežerka, *Automation in Construction*)
showed a simpler approach is *strictly* better on this kind of input:

  1. Compute the binary wall mask (already done by density slicer + ceiling
     band gate).
  2. Find connected-component contours with ``cv2.findContours`` — these
     are the wall *surfaces*, expressed as ordered polylines.
  3. Simplify each polyline with Douglas-Peucker (``cv2.approxPolyDP``).
  4. Split each polyline into straight wall segments at "high curvature"
     vertices (corners) using the simplified polygon's vertex sequence.

This is strictly better than Hough/FLD because:

  - **No fragmentation.**  A 5-metre wall is one polyline, not 25 short
    Hough lines.  Wall pairing becomes deterministic (face A and face B
    of the same wall come from the *same* contour, one pass per side).
  - **No orphan drops.**  Single-faced walls (scanner only saw one side)
    still survive — they're a contour, not a missing pair.
  - **Topology is structural.**  Walls share endpoints by construction
    when their contours touch (T-junctions, L-corners).  The downstream
    junction-snap step now only has to handle scanner noise, not detector
    artefacts.
  - **Computation is fast.**  ``findContours`` is O(perimeter pixels);
    Douglas-Peucker is O(N log N) per polyline.  On a 5000² raster the
    whole pass takes ~200 ms.

Output schema
-------------
Each "wall surface" is a :class:`WallContour` — an ordered polyline in
world coordinates, plus metadata (parent contour, hole vs outer, length).
The pairing step in ``walls.py`` then matches inside/outside contour
edges to form thickness-aware walls.

Caveats / tuning notes
----------------------
- ``contour_simplify_eps_m`` (Douglas-Peucker tolerance) is the single
  most important knob.  2 cm is the Cloud2BIM default; matches typical
  wall-corner ambiguity.  Too small → jagged polylines; too large →
  adjacent rooms merge through doorways.
- Holes (interior contour rings) ARE wall surfaces too — they bound
  interior rooms.  We keep them and tag them as ``is_hole = True``.
- We do NOT Manhattan-snap here.  That happens in a separate pass on the
  segment list (existing ``regularize.manhattan_snap``) so curved walls
  can opt out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .slicer import RasterAffine


# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class WallContourParams:
    """Tunable thresholds for contour-based wall extraction."""

    # Douglas-Peucker epsilon in METRES.  2 cm matches the typical corner
    # ambiguity of wall corners in a 1 cm/px raster.  Cloud2BIM default.
    simplify_eps_m: float = 0.02

    # Minimum contour perimeter in METRES.  Below 0.3 m we're almost
    # certainly looking at scanner noise or a tiny piece of furniture
    # speckle that escaped the morphology pass.  Keeps the noise floor
    # off the wall list before any downstream stage sees it.
    min_perimeter_m: float = 0.30

    # Maximum number of vertices on a single polyline.  A real wall surface
    # is < 200 vertices even on the most decorative buildings; cap at 2000
    # to defend against pathological inputs (e.g. a star-shaped noise blob)
    # without artificially limiting reasonable walls.
    max_vertices: int = 2000

    # When splitting a closed polygon into wall segments, do not emit any
    # segment whose length is below this floor.  Removes the "1-pixel
    # spurs" that Douglas-Peucker sometimes produces when two corners
    # are simplified to nearly-coincident vertices.
    min_segment_length_m: float = 0.08

    # Internal CLOSE before findContours.  Must be large enough to heal
    # dashed wall strokes in the density∩ceiling mask (this scan had ~842
    # connected components at 3 px CLOSE; BricsCAD's POINTCLOUDPROJECTSECTION
    # "Gap" tolerance does the same job).  ~9–11 px ≈ 9–11 cm at 1 cm/px —
    # half a typical interior wall — without swallowing standard door gaps.
    close_kernel_px: int = 9

    # BricsCAD OPTIMIZE-equivalent: after shattering contours into edges,
    # fuse near-collinear near-touching runs into long wall axes.  Cloud2BIM
    # § 2.7 does this explicitly; without it every Douglas-Peucker edge
    # becomes its own CAD entity (median wall ~0.8 m on real scans).
    merge_collinear: bool = True
    merge_parallel_tol_deg: float = 6.0
    merge_perp_distance_m: float = 0.12
    merge_endpoint_gap_m: float = 1.00


@dataclass
class WallContour:
    """One wall surface polyline.

    Carries the world-coord polyline plus enough provenance for the pairing
    stage to know which surfaces came from the same parent contour (i.e.
    are the two faces of the same wall) and which surfaces bound a room
    from the inside (holes).
    """
    polyline_xy: np.ndarray            # (M, 2) world coords, NOT closed
    segments: np.ndarray               # (M-1, 2, 2) — successive pairs
    contour_id: int                    # parent findContours ID
    hierarchy_parent: int              # parent in cv2 hierarchy (-1 for outer)
    is_hole: bool                      # True for interior contours
    perimeter_m: float
    n_vertices: int

    @property
    def n_segments(self) -> int:
        return int(self.segments.shape[0])


@dataclass
class WallContourResult:
    """Output of :func:`extract_wall_contours`."""
    walls: list[WallContour]
    flat_segments: np.ndarray          # (sum N_i, 2, 2) — all segments concat'd
    n_contours_raw: int = 0
    n_contours_kept: int = 0
    n_segments_total: int = 0


# ── Public API ────────────────────────────────────────────────────────────────

def extract_wall_contours(
    wall_mask: np.ndarray,
    affine: RasterAffine,
    params: WallContourParams | None = None,
) -> WallContourResult:
    """Find and simplify wall contours from a binary mask.

    Parameters
    ----------
    wall_mask : (H, W) uint8
        Binary wall mask in *internal* convention (row 0 = low world Y).
        Values 0 or 255.  Typically the AND of the density slicer + ceiling
        band gate from ``ceiling_band_slicer.gate_with_ceiling``.
    affine : RasterAffine
        World ↔ pixel mapping that produced the mask.
    params : WallContourParams, optional

    Returns
    -------
    WallContourResult

    Implementation notes
    --------------------
    ``cv2.findContours(RETR_CCOMP, CHAIN_APPROX_NONE)`` returns external
    contours plus a single level of holes (CCOMP).  This gives us:

      - Outer building outline (the wall mask's outer boundary).
      - Every interior wall (holes within the outer contour).

    We could use ``RETR_EXTERNAL`` and skip holes entirely, but the goal
    is to get the *inside* of each room as a separate polyline so the
    pairing step can match outside ↔ inside of the same wall trivially.
    """
    if params is None:
        params = WallContourParams()
    if wall_mask.dtype != np.uint8:
        raise ValueError(
            f"wall_mask must be uint8; got {wall_mask.dtype}"
        )
    if wall_mask.ndim != 2:
        raise ValueError(
            f"wall_mask must be 2-D; got shape {wall_mask.shape}"
        )

    # CLOSE to bridge sub-cm scanner gaps that would otherwise fragment a
    # single wall into two contours.  We do this here rather than relying
    # on the slicer's preprocess so the contour extractor is self-sufficient
    # and easy to A/B test against the legacy detector path.
    if params.close_kernel_px > 1:
        k = params.close_kernel_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_closed = cv2.morphologyEx(
            wall_mask, cv2.MORPH_CLOSE, kernel, iterations=1,
        )
    else:
        mask_closed = wall_mask

    contours, hierarchy = cv2.findContours(
        mask_closed,
        mode=cv2.RETR_CCOMP,
        method=cv2.CHAIN_APPROX_NONE,
    )
    n_contours_raw = len(contours)
    if n_contours_raw == 0:
        return WallContourResult(
            walls=[],
            flat_segments=np.zeros((0, 2, 2), dtype=np.float64),
        )

    res = affine.resolution_m_per_px
    eps_px = max(1.0, params.simplify_eps_m / res)
    min_perim_px = max(2, int(round(params.min_perimeter_m / res)))
    min_seg_px = max(1.0, params.min_segment_length_m / res)

    # hierarchy shape is (1, N, 4); each row is (next, prev, first_child, parent).
    hier_flat = hierarchy[0] if hierarchy is not None else np.zeros((n_contours_raw, 4), dtype=np.int32) - 1

    walls: list[WallContour] = []
    seg_chunks: list[np.ndarray] = []

    for cid, contour in enumerate(contours):
        # contour: (M, 1, 2) int32 — pixel (col, row).
        perim_px = cv2.arcLength(contour, closed=True)
        if perim_px < min_perim_px:
            continue

        # Douglas-Peucker simplify.  closed=True because findContours returns
        # closed contours (last vertex implicitly connects to first).
        simplified = cv2.approxPolyDP(
            contour, epsilon=eps_px, closed=True,
        )
        if simplified.shape[0] < 2:
            continue
        if simplified.shape[0] > params.max_vertices:
            # Pathological — extreme star-shape; skip.
            continue

        # (M, 2) int32 pixel coords.
        pix = simplified.reshape(-1, 2).astype(np.float64)

        # Convert to world coords.  Pixel→world: (origin + (col+0.5)*res,
        # origin + (row+0.5)*res).  We use the centre of the pixel
        # consistently to match RasterAffine.pixel_to_world.
        world_x = affine.origin_x + (pix[:, 0] + 0.5) * res
        world_y = affine.origin_y + (pix[:, 1] + 0.5) * res
        world = np.stack([world_x, world_y], axis=1)

        # Build successive-pair segments along the closed polygon.  We CLOSE
        # the polygon explicitly (append first vertex at end) so the last
        # segment connects the polygon's last vertex back to its first.
        closed_world = np.vstack([world, world[0:1]])
        starts = closed_world[:-1]
        ends = closed_world[1:]
        seg_lengths = np.linalg.norm(ends - starts, axis=1)

        # Drop tiny segments (Douglas-Peucker collapse artefacts).
        keep_mask = seg_lengths >= (min_seg_px * res)
        if not keep_mask.any():
            continue
        starts = starts[keep_mask]
        ends = ends[keep_mask]
        wall_segments = np.stack([starts, ends], axis=1).astype(np.float64)

        hier = hier_flat[cid]
        parent = int(hier[3]) if hier_flat.size else -1
        is_hole = parent != -1  # CCOMP convention: any contour with a parent is a hole

        # Polyline in world coords for the wall record.  We KEEP it open
        # (one vertex per real polygon corner) so consumers don't double-
        # count the closing vertex.
        polyline_world = world

        walls.append(WallContour(
            polyline_xy=polyline_world,
            segments=wall_segments,
            contour_id=cid,
            hierarchy_parent=parent,
            is_hole=is_hole,
            perimeter_m=float(perim_px * res),
            n_vertices=int(polyline_world.shape[0]),
        ))
        seg_chunks.append(wall_segments)

    if not seg_chunks:
        flat = np.zeros((0, 2, 2), dtype=np.float64)
    else:
        flat = np.concatenate(seg_chunks, axis=0)

    # BricsCAD OPTIMIZE / Cloud2BIM post-contour collinear join: turn the
    # edge soup into long wall axes before pairing/topology see them.
    n_before_merge = int(flat.shape[0])
    if params.merge_collinear and n_before_merge > 1:
        from . import regularize as regularize_mod
        flat = regularize_mod.merge_collinear_segments(
            flat,
            parallel_tol_deg=params.merge_parallel_tol_deg,
            perp_distance_m=params.merge_perp_distance_m,
            endpoint_gap_m=params.merge_endpoint_gap_m,
        )
        # Drop residual dust after merging (shorter than min segment).
        if len(flat) > 0:
            lens = np.linalg.norm(flat[:, 1] - flat[:, 0], axis=1)
            flat = flat[lens >= params.min_segment_length_m]

    return WallContourResult(
        walls=walls,
        flat_segments=flat,
        n_contours_raw=n_contours_raw,
        n_contours_kept=len(walls),
        n_segments_total=int(flat.shape[0]),
    )


# ── Reporting helper ────────────────────────────────────────────────────────

def contour_summary(result: WallContourResult) -> str:
    """One-line human-readable summary for SSE progress."""
    n_holes = sum(1 for w in result.walls if w.is_hole)
    n_outer = len(result.walls) - n_holes
    med = 0.0
    if result.n_segments_total > 0 and len(result.flat_segments) > 0:
        lens = np.linalg.norm(
            result.flat_segments[:, 1] - result.flat_segments[:, 0], axis=1,
        )
        med = float(np.median(lens))
    return (
        f"contours: {result.n_contours_raw} raw → {result.n_contours_kept} kept "
        f"({n_outer} outer + {n_holes} holes) → {result.n_segments_total} axes "
        f"(median {med:.2f} m)"
    )
