"""Arc fitting + polyline regularization for the sheet renderer (Phase 3.5).

The reference sheets preserve diagonal and curved walls faithfully; a
curved facade extracted as many short chords should be DRAWN as an arc,
not a polygonal staircase.  This module provides:

- :func:`fit_circle` — algebraic (Kåsa) least-squares circle through
  points; exact on noise-free synthetic arcs.
- :func:`fit_arcs_to_polyline` — greedy chord-chain detection: maximal
  runs of consecutive vertices that lie on one circle (within tolerance)
  become arc spans, everything else stays polyline.
- :func:`ring_path_d` — an SVG path ``d`` string for a closed ring, with
  detected curved runs emitted as ``A`` (arc) segments.
- :func:`regularize_ring` — render-layer envelope regularization:
  Douglas-Peucker simplification + snapping near-axis edges exactly
  parallel to the dominant wall axes.  Genuinely diagonal edges (beyond
  the angle tolerance) are preserved untouched.

Everything here is presentation-layer: measurement polygons never pass
through this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

#: Minimum consecutive vertices for an arc fit (3 points define a circle;
#: requiring 4+ avoids "fitting" every corner).
MIN_ARC_POINTS = 4

#: Arcs flatter than this are just lines (radius sanity cap, metres/mm —
#: unit follows the input).
MAX_ARC_RADIUS_FACTOR = 50.0

#: Max angle one chord may subtend at the fitted centre.  Guards against
#: "fitting" polygon corners: a rectangle's 4 corners lie exactly on its
#: circumcircle but subtend ~90° each — real curve chords subtend far
#: less.  35° keeps octagons (45° steps) safely un-arced too.
MAX_CHORD_ANGLE_RAD = math.radians(35.0)


def fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle (Kåsa algebraic fit) through ``(N, 2)`` points.

    Returns ``(cx, cy, r)``.  Exact for points sampled from a true circle.
    Raises ``ValueError`` for degenerate (collinear / too few) input.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("need at least 3 points to fit a circle")
    x, y = pts[:, 0], pts[:, 1]
    a_mat = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(pts))])
    b_vec = x * x + y * y
    sol, residuals, rank, _sv = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    if rank < 3:
        raise ValueError("degenerate (collinear) points — no unique circle")
    cx, cy, c = sol
    r_sq = c + cx * cx + cy * cy
    if r_sq <= 0.0:
        raise ValueError("degenerate circle fit")
    return float(cx), float(cy), float(math.sqrt(r_sq))


@dataclass
class ArcSpan:
    """A run of consecutive polyline vertices lying on one circular arc."""
    start_idx: int           # first vertex index of the run
    end_idx: int             # last vertex index of the run (inclusive)
    cx: float
    cy: float
    r: float
    ccw: bool                # arc direction from start to end


def _arc_residual(pts: np.ndarray, cx: float, cy: float, r: float) -> float:
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    return float(np.max(np.abs(d - r)))


def _arc_direction(pts: np.ndarray, cx: float, cy: float) -> bool:
    """True when the vertex sequence winds CCW about the fitted centre."""
    ang = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    return bool(ang[-1] > ang[0])


def fit_arcs_to_polyline(
    points: np.ndarray,
    tol: float,
    min_points: int = MIN_ARC_POINTS,
) -> list[ArcSpan]:
    """Find maximal runs of consecutive vertices lying on one circle.

    Greedy: grow a candidate run while the circle fit through all its
    vertices stays within ``tol`` (max radial deviation) and the vertex
    angles progress monotonically (a real arc, not a zig-zag).  Runs
    shorter than ``min_points`` vertices are ignored.  Returns
    non-overlapping spans in vertex order.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    spans: list[ArcSpan] = []
    i = 0
    while i <= n - min_points:
        best: Optional[ArcSpan] = None
        j = i + min_points - 1
        while j < n:
            run = pts[i:j + 1]
            try:
                cx, cy, r = fit_circle(run)
            except ValueError:
                break
            if r > MAX_ARC_RADIUS_FACTOR * _run_extent(run):
                break
            if _arc_residual(run, cx, cy, r) > tol:
                break
            if not _monotone_angles(run, cx, cy):
                break
            if _max_angle_step(run, cx, cy) > MAX_CHORD_ANGLE_RAD:
                break
            best = ArcSpan(
                start_idx=i, end_idx=j, cx=cx, cy=cy, r=r,
                ccw=_arc_direction(run, cx, cy),
            )
            j += 1
        if best is not None:
            spans.append(best)
            i = best.end_idx           # spans may share their boundary vertex
        else:
            i += 1
    return spans


def _run_extent(pts: np.ndarray) -> float:
    return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-9))


def _monotone_angles(pts: np.ndarray, cx: float, cy: float) -> bool:
    ang = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    diffs = np.diff(ang)
    return bool(np.all(diffs > 1e-12) or np.all(diffs < -1e-12))


def _max_angle_step(pts: np.ndarray, cx: float, cy: float) -> float:
    ang = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    return float(np.max(np.abs(np.diff(ang))))


# ── SVG path emission ─────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def ring_path_d(ring: np.ndarray, arc_tol_mm: float = 0.5) -> str:
    """SVG path ``d`` for a closed ring, curved runs emitted as arcs.

    ``ring``: (N, 2) PAPER-mm coordinates, closing vertex not duplicated.
    Straight stretches become ``L`` commands; detected arc runs become
    ``A`` commands (one per run — SVG arcs are exact circles).  Note the
    input is paper space (y down), so a world-CCW run appears CW and the
    sweep flag follows the detected paper-space direction directly.
    """
    pts = np.asarray(ring, dtype=float)
    if len(pts) < 2:
        return ""
    spans = fit_arcs_to_polyline(pts, arc_tol_mm) if len(pts) >= MIN_ARC_POINTS else []
    by_start = {s.start_idx: s for s in spans}

    d_parts = [f"M {_fmt(pts[0, 0])} {_fmt(pts[0, 1])}"]
    i = 1
    k = 0
    while k < len(pts) - 1:
        span = by_start.get(k)
        if span is not None:
            end = pts[span.end_idx]
            # Paper coords are y-down: SVG sweep=1 (clockwise on screen)
            # corresponds to increasing atan2 angle in these coordinates —
            # which is what _arc_direction measured.
            sweep = "1" if span.ccw else "0"
            # Quarter-ish spans: large-arc flag from the swept angle.
            ang = np.unwrap(np.arctan2(
                pts[span.start_idx:span.end_idx + 1, 1] - span.cy,
                pts[span.start_idx:span.end_idx + 1, 0] - span.cx,
            ))
            large = "1" if abs(ang[-1] - ang[0]) > math.pi else "0"
            d_parts.append(
                f"A {_fmt(span.r)} {_fmt(span.r)} 0 {large} {sweep} "
                f"{_fmt(end[0])} {_fmt(end[1])}"
            )
            k = span.end_idx
        else:
            nxt = pts[k + 1]
            d_parts.append(f"L {_fmt(nxt[0])} {_fmt(nxt[1])}")
            k += 1
    d_parts.append("Z")
    return " ".join(d_parts)


# ── Envelope regularization (render-layer only) ───────────────────────────────

def ring_dominant_axes(ring: np.ndarray) -> tuple[float, float]:
    """The ring's two dominant edge axes (rad), via the length-weighted 4θ
    circular mean (same fold as the wall-axis detector).  Returns
    ``(axis, axis + 90°)`` with the primary folded into [−45°, +45°)."""
    pts = np.asarray(ring, dtype=float)
    s4 = c4 = 0.0
    for k in range(len(pts)):
        d = pts[(k + 1) % len(pts)] - pts[k]
        length = float(np.hypot(d[0], d[1]))
        if length < 1e-12:
            continue
        theta = math.atan2(d[1], d[0])
        s4 += length * math.sin(4.0 * theta)
        c4 += length * math.cos(4.0 * theta)
    if abs(s4) < 1e-12 and abs(c4) < 1e-12:
        return (0.0, math.pi / 2.0)
    axis = math.atan2(s4, c4) / 4.0
    quarter = math.pi / 2.0
    axis = (axis + quarter / 2.0) % quarter - quarter / 2.0
    return (axis, axis + quarter)

def snap_ring_edges_to_axes(
    ring: np.ndarray,
    axes_rad: Sequence[float],
    angle_tol_deg: float = 8.0,
) -> np.ndarray:
    """Rotate near-axis ring edges exactly onto the given axes.

    Each edge within ``angle_tol_deg`` of an axis (mod 180°) is rotated
    about its midpoint to the exact axis direction; vertices are then
    rebuilt as intersections of consecutive edge lines, which makes
    snapped runs exactly parallel/collinear.  Edges beyond the tolerance
    (genuine diagonals, curve chords) keep their original direction, so
    the snap can never flatten a diagonal facade.
    """
    pts = np.asarray(ring, dtype=float)
    n = len(pts)
    if n < 3:
        return pts.copy()

    tol = math.radians(angle_tol_deg)
    mids: list[np.ndarray] = []
    dirs: list[np.ndarray] = []
    for k in range(n):
        p, q = pts[k], pts[(k + 1) % n]
        d = q - p
        length = float(np.hypot(d[0], d[1]))
        if length < 1e-12:
            # Degenerate edge: keep as-is (its vertices merge on rebuild).
            mids.append((p + q) / 2.0)
            dirs.append(np.array([1.0, 0.0]))
            continue
        theta = math.atan2(d[1], d[0])
        snapped = None
        for axis in axes_rad:
            diff = (theta - axis + math.pi / 2.0) % math.pi - math.pi / 2.0
            if abs(diff) <= tol:
                snapped = theta - diff
                break
        theta_out = snapped if snapped is not None else theta
        mids.append((p + q) / 2.0)
        dirs.append(np.array([math.cos(theta_out), math.sin(theta_out)]))

    out = np.zeros_like(pts)
    for k in range(n):
        p1, d1 = mids[(k - 1) % n], dirs[(k - 1) % n]
        p2, d2 = mids[k], dirs[k]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-9:
            out[k] = pts[k]              # parallel neighbours: keep vertex
        else:
            diff = p2 - p1
            t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
            out[k] = p1 + t * d1
    return out


def regularize_ring(
    ring: np.ndarray,
    axes_rad: Sequence[float] = (0.0, math.pi / 2.0),
    dp_tol: float = 0.05,
    angle_tol_deg: float = 8.0,
) -> np.ndarray:
    """Render-layer ring clean-up: Douglas-Peucker + axis snapping.

    1. Simplify with Shapely (Douglas-Peucker, topology-preserving) at
       ``dp_tol`` — removes alpha-shape jitter vertices.
    2. Snap near-axis edges exactly onto ``axes_rad``
       (:func:`snap_ring_edges_to_axes`); diagonals beyond
       ``angle_tol_deg`` are untouched.

    NEVER feed the result to measurement — this is drawing geometry only.
    """
    pts = np.asarray(ring, dtype=float)
    if len(pts) < 4:
        return pts.copy()
    poly = ShapelyPolygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty:
            return pts.copy()
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    simplified = poly.simplify(dp_tol, preserve_topology=True)
    simple_pts = np.asarray(simplified.exterior.coords[:-1], dtype=float)
    if len(simple_pts) < 3:
        simple_pts = pts
    return snap_ring_edges_to_axes(simple_pts, axes_rad, angle_tol_deg)
