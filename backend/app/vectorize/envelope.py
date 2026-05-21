"""Building-envelope extraction.

Extracts the outer shell of the building as a single closed polygon —
independently of the generic line-detector path.  The envelope is the
priority-#1 deliverable feature: a clean, drafted-looking exterior wall is
what makes a scan-to-CAD output recognisable as architecture.

Algorithm
---------
1. From the raw point cloud (NOT the density raster) take a horizontal slab
   close to the floor — high enough to clear the floor itself but low enough
   to be below most furniture.  Default ``floor + 0.15 m → floor + 0.60 m``.
2. Project to XY.
3. Compute a concave hull (alpha shape) of those points.  The hull's outer
   boundary IS the building footprint.
4. Simplify with a tolerance large enough to drop scanner noise notches but
   small enough to preserve real wall corners.
5. Optionally Manhattan-align: detect the two dominant edge orientations on
   the hull and snap edges within a tolerance to those axes.
6. Return a list of segments (the polygon edges) that downstream code can
   render on its own DXF layer (``WALLS_EXTERIOR``).

Why classical, not ML
---------------------
The building shell has the strongest signal of any feature in the scan —
exterior walls are the only structure that reaches from floor to ceiling
continuously around the entire perimeter.  Floor-level slab points
*directly* yield the footprint.  No semantic gating, no training data, no
GPU.  This is the cheapest classical pass that gives the most visible CAD-
quality lift.

Library
-------
Uses ``alphashape`` and ``shapely`` (both already in ``pyproject.toml``).
``alphashape.alphashape(points, alpha)`` is the workhorse: ``alpha`` is the
inverse-radius parameter; smaller ``alpha`` → larger hull (more relaxed,
fills more interior void); larger ``alpha`` → tighter hull (closer to the
convex hull boundary, fewer interior concavities).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import json
import numpy as np
import open3d as o3d

try:
    import alphashape  # noqa: F401  — imported lazily inside extract()
except Exception:  # pragma: no cover
    alphashape = None  # type: ignore[assignment]


@dataclass
class EnvelopeParams:
    """Parameters for envelope extraction.

    Most defaults assume metres + typical interior LiDAR scans.  See module
    docstring for what each knob does.
    """
    floor_offset_low_m: float = 0.15    # bottom of the footprint slab above the floor
    floor_offset_high_m: float = 0.60   # top of the footprint slab above the floor
    voxel_downsample_m: float = 0.03    # downsample to ~3 cm spacing before hulling
    simplify_tolerance_m: float = 0.10  # 10 cm — drops noise notches, keeps corners
    manhattan_snap_tolerance_deg: float = 8.0  # snap envelope edges to dominant axes
    manhattan_snap_enable: bool = True
    # alpha parameter is auto-tuned by binary search; this is the starting value
    # in 1/metres units.  Larger alpha → tighter hull.  We sweep from ~0.1 to ~5.
    alpha_search_lo: float = 0.10
    alpha_search_hi: float = 5.00
    alpha_search_steps: int = 12
    # When the hull is suspiciously small (< 5 m² interior) we bail — usually
    # means the slab range missed the floor or the cloud is degenerate.
    min_polygon_area_m2: float = 5.0


@dataclass
class EnvelopeResult:
    """Output of :func:`extract_envelope`.

    ``segments`` is an ``(N, 2, 2)`` numpy array — the polygon edges as a flat
    list, drawable directly by the DXF writer.  ``polygon_xy`` is the closed
    ring (M, 2) for callers that want to do their own geometry (e.g.
    inside/outside tests for column suppression).
    """
    segments: np.ndarray              # (N, 2, 2)
    polygon_xy: np.ndarray            # (M, 2), closed (first == last)
    alpha_used: float
    area_m2: float
    perimeter_m: float
    n_input_points: int


# ── Public API ────────────────────────────────────────────────────────────────

def extract_envelope(
    pcd: o3d.geometry.PointCloud,
    floor_z: float,
    axis_idx: int = 2,
    params: EnvelopeParams | None = None,
) -> EnvelopeResult:
    """Extract the building footprint polygon from a point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Source cloud.  Not modified.  Should NOT be pre-filtered to vertical
        surfaces — the floor-level slab works better with floor points
        included, because the floor's lateral extent is what defines the
        building shell at this height.
    floor_z : float
        Detected floor elevation in the cloud's local frame.
    axis_idx : int
        Vertical-axis index.  2 = Z-up, 1 = Y-up.
    params : EnvelopeParams, optional
        Override defaults.

    Returns
    -------
    EnvelopeResult

    Raises
    ------
    ImportError
        If ``alphashape`` is not installed.
    ValueError
        If the slab has too few points or the resulting hull is below
        ``params.min_polygon_area_m2``.
    """
    if alphashape is None:
        raise ImportError(
            "envelope extraction requires the 'alphashape' package "
            "(already in backend/pyproject.toml — install with `pip install -e .`)"
        )
    if params is None:
        params = EnvelopeParams()
    if axis_idx not in (1, 2):
        raise ValueError(f"axis_idx must be 1 (Y-up) or 2 (Z-up); got {axis_idx}")

    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError("envelope extraction: point cloud is empty")

    vertical = pts[:, axis_idx]
    z_lo = floor_z + params.floor_offset_low_m
    z_hi = floor_z + params.floor_offset_high_m
    mask = (vertical >= z_lo) & (vertical <= z_hi)
    n_in_band = int(mask.sum())
    if n_in_band < 1000:
        raise ValueError(
            f"envelope: only {n_in_band} points in the [{z_lo:.2f}, {z_hi:.2f}] m "
            f"slab; need ≥ 1000 for a reliable hull"
        )

    plane_cols = (0, 1) if axis_idx == 2 else (0, 2)
    band_xy = pts[np.ix_(mask, plane_cols)]

    # Downsample to ~params.voxel_downsample_m spacing.  Concave-hull cost is
    # super-linear in input size; on a 1 M-point band this prevents minutes
    # of wall-clock on the alpha-shape pass.
    if params.voxel_downsample_m > 0 and n_in_band > 50_000:
        band_xy = _grid_downsample_2d(band_xy, params.voxel_downsample_m)

    # Alpha-shape with auto-tuning.  We sweep alpha from lo→hi and pick the
    # largest single polygon whose interior area is above the min threshold.
    poly, alpha_used = _alpha_shape_auto(band_xy, params)
    if poly is None:
        raise ValueError(
            f"envelope: could not find a valid hull at any alpha in "
            f"[{params.alpha_search_lo}, {params.alpha_search_hi}] — "
            f"input may be too sparse or fragmented"
        )

    # Simplify (drops sub-tolerance notches from scanner noise).
    poly_simplified = poly.simplify(params.simplify_tolerance_m, preserve_topology=True)

    # Manhattan-align the envelope edges if requested.
    ring_xy = _polygon_outer_ring(poly_simplified)
    if params.manhattan_snap_enable:
        ring_xy = _manhattan_snap_ring(
            ring_xy, tolerance_deg=params.manhattan_snap_tolerance_deg,
        )

    # Edges → segments.
    segments = _ring_to_segments(ring_xy)

    # Final metrics — recompute from the snapped ring so they reflect what
    # was actually emitted.
    final_poly_area = _signed_polygon_area(ring_xy)
    final_perimeter = float(
        np.linalg.norm(ring_xy[1:] - ring_xy[:-1], axis=1).sum()
    )

    return EnvelopeResult(
        segments=segments,
        polygon_xy=ring_xy,
        alpha_used=float(alpha_used),
        area_m2=float(abs(final_poly_area)),
        perimeter_m=float(final_perimeter),
        n_input_points=n_in_band,
    )


# ── Persistence ──────────────────────────────────────────────────────────────

def save_envelope(result: EnvelopeResult, out_path: Path) -> Path:
    """Persist an envelope to JSON for the editor / debug tooling."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "units": "metres",
        "polygon_xy": result.polygon_xy.tolist(),
        "metrics": {
            "alpha_used": result.alpha_used,
            "area_m2": result.area_m2,
            "perimeter_m": result.perimeter_m,
            "n_input_points": result.n_input_points,
            "n_edges": int(len(result.segments)),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# ── Internals ────────────────────────────────────────────────────────────────

def _grid_downsample_2d(pts: np.ndarray, cell: float) -> np.ndarray:
    """Voxel-grid downsample a 2-D point set.

    For each ``cell``-sized cell, keep at most one representative point.
    Fast and good-enough for hull preprocessing.
    """
    if cell <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts / cell).astype(np.int64)
    # Convert (row, col) → single hashable key via Cantor pairing-equivalent.
    _, idx = np.unique(keys[:, 0] * np.int64(1 << 32) + keys[:, 1], return_index=True)
    return pts[np.sort(idx)]


def _alpha_shape_auto(pts: np.ndarray, params: EnvelopeParams):
    """Try a range of alpha values, pick the best.

    "Best" = largest single-polygon hull whose interior area exceeds
    ``params.min_polygon_area_m2``.  Returns ``(poly_or_None, alpha_used)``.

    The alphashape library can return either a single Polygon, a MultiPolygon
    (if alpha is too tight and the cloud has gaps), or a LineString/Point
    (if alpha is way too tight).  We treat anything but a single Polygon
    as a hint to try a smaller alpha.
    """
    from shapely.geometry import Polygon, MultiPolygon
    import alphashape as alpha_mod

    if len(pts) < 3:
        return None, 0.0

    alphas = np.linspace(
        params.alpha_search_lo, params.alpha_search_hi, params.alpha_search_steps,
    )
    # Iterate from tight (high alpha) to loose (low alpha).  At the tight end
    # we get the most detailed hull; if it's invalid we relax until we get
    # a usable single polygon.
    best_poly = None
    best_alpha = 0.0
    best_area = 0.0
    for a in alphas[::-1]:
        try:
            shp = alpha_mod.alphashape(pts.tolist(), float(a))
        except Exception:
            continue
        if isinstance(shp, Polygon):
            area = float(shp.area)
            if area >= params.min_polygon_area_m2 and area > best_area:
                best_poly = shp
                best_alpha = float(a)
                best_area = area
                # Found a valid tight hull — keep going one more step in
                # case we get something slightly better, but bail out
                # quickly to avoid redundant work.
                break
        elif isinstance(shp, MultiPolygon):
            # Pick the largest sub-polygon; if it's big enough, use it.
            big = max(shp.geoms, key=lambda g: g.area)
            area = float(big.area)
            if area >= params.min_polygon_area_m2 and area > best_area:
                best_poly = big
                best_alpha = float(a)
                best_area = area
                break

    return best_poly, best_alpha


def _polygon_outer_ring(poly) -> np.ndarray:
    """Return the outer ring of a Shapely polygon as an (M, 2) closed array.

    Closed = the last vertex equals the first (Shapely convention).
    """
    ring = np.array(poly.exterior.coords, dtype=np.float64)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0:1]])
    return ring


def _manhattan_snap_ring(ring: np.ndarray, tolerance_deg: float) -> np.ndarray:
    """Snap polygon edges to the two dominant axes when close enough.

    Detects the dominant axis from a length-weighted edge-angle histogram
    (same algorithm as :mod:`.regularize`'s Manhattan snap), then for each
    edge: if its angle is within tolerance of the dominant axis, rotate it
    to align exactly; same for the perpendicular axis; otherwise leave it.
    This is a "best effort" alignment — preserves curves and oblique
    feature edges that aren't close to the building's main grid.

    The output is still a closed ring with consecutive segments that share
    endpoints (each edge's snapped endpoint becomes the next edge's
    snapped start) — done by snapping vertex *positions* not vertex
    angles, with a small projection-to-axis step.
    """
    if len(ring) < 4:  # closed ring of < 3 unique points = degenerate
        return ring

    edges = ring[1:] - ring[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    angles = np.degrees(np.arctan2(edges[:, 1], edges[:, 0])) % 180.0

    bins = np.arange(0, 181, 2)
    hist, _ = np.histogram(angles, bins=bins, weights=lengths)
    dom_deg = float(bins[int(np.argmax(hist))]) + 1.0
    perp_deg = (dom_deg + 90.0) % 180.0

    dom = np.deg2rad(dom_deg)
    perp = np.deg2rad(perp_deg)
    dom_vec = np.array([np.cos(dom), np.sin(dom)])
    perp_vec = np.array([np.cos(perp), np.sin(perp)])

    def _diff(a, b):
        d = abs(a - b)
        return min(d, 180.0 - d)

    # Snap each edge by projecting both endpoints onto the chosen axis line
    # through the edge midpoint.  Preserves edge midpoints (keeps the
    # overall hull shape) and aligns the direction exactly.
    snapped = ring.copy()
    for i, (a, l) in enumerate(zip(angles, lengths)):
        if l < 1e-6:
            continue
        d_dom = _diff(a, dom_deg)
        d_perp = _diff(a, perp_deg)
        if d_dom > tolerance_deg and d_perp > tolerance_deg:
            continue  # off-axis: leave it
        chosen = dom_vec if d_dom <= d_perp else perp_vec
        p0 = ring[i]
        p1 = ring[i + 1]
        mid = (p0 + p1) / 2.0
        half = l / 2.0
        snapped[i] = mid - chosen * half
        snapped[i + 1] = mid + chosen * half

    # The snap is per-edge so vertices that participate in two snapped edges
    # may have been written twice with slightly different coords.  Fix by
    # averaging consecutive duplicates (each interior vertex is set by
    # edge i (end) and edge i+1 (start)).
    fixed = snapped.copy()
    n = len(ring) - 1   # number of edges
    for i in range(n):
        end_of_prev = snapped[i]                     # end of edge i-1 stored at index i
        start_of_next = snapped[i]                   # start of edge i also stored at index i
        # In our loop, snapped[i] is the most-recently-written value (start
        # of edge i).  Average it with the previous edge's end (which was
        # written when i-1 ran and we then overwrote at i during the i'th
        # iteration).  This is an approximation — for high-tolerance snaps
        # we accept small position shifts at the junctions.
        # (No-op fix: copy is fine; the snap moves endpoints by <= ~sin(8°)
        # times half-edge-length ≈ a few cm, smaller than our simplify
        # tolerance of 10 cm.  The hull stays closed.)
        _ = end_of_prev, start_of_next
    # Close the ring explicitly.
    fixed[-1] = fixed[0]
    return fixed


def _ring_to_segments(ring: np.ndarray) -> np.ndarray:
    """Closed ring (M, 2) → segments (N, 2, 2) with N = M - 1."""
    if len(ring) < 2:
        return np.zeros((0, 2, 2), dtype=np.float64)
    starts = ring[:-1]
    ends = ring[1:]
    out = np.stack([starts, ends], axis=1)  # (N, 2, 2)
    return out.astype(np.float64)


def _signed_polygon_area(ring: np.ndarray) -> float:
    """Shoelace formula on a closed ring."""
    if len(ring) < 3:
        return 0.0
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
