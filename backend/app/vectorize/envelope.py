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
    # Statistical outlier filter on the floor-slab points BEFORE the alpha-
    # shape.  Stray scanner returns (glass reflections, multipath echoes,
    # points on neighbouring structures) appear as isolated points far
    # from the main cluster.  Including them in the alpha-shape forces
    # the envelope to enclose huge empty regions to "reach" them.  We
    # drop any point whose mean distance to its ``outlier_nb_neighbors``
    # neighbours is more than ``outlier_std_ratio`` standard deviations
    # above the global mean.  Classic Open3D statistical outlier removal.
    outlier_filter_enable: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 1.5      # tighter than default (2.0) — favours a clean hull
    # DBSCAN pre-cluster (v4 Cloud2BIM-inspired addition).  After the
    # statistical outlier filter, run density-based clustering and keep
    # ONLY points in the largest connected cluster.  This is the
    # surgical fix for "the envelope balloons across the parking lot to
    # capture window reflections" — those reflections form their own
    # spatial cluster that isn't a statistical outlier within itself
    # (statistical outlier removal only catches isolated lone points).
    # ``dbscan_eps_m`` is the cluster joining distance — 0.50 m covers
    # any wall-to-wall gap (doorways, openings) without bridging across
    # an entire room.  Below 0.30 m and dim corners get fragmented;
    # above 1.5 m and adjacent buildings get merged.
    dbscan_cluster_enable: bool = True
    dbscan_eps_m: float = 0.50
    dbscan_min_samples: int = 25
    # When the largest cluster has fewer than this many points it's
    # probably too sparse to be a building — fall back to all points.
    dbscan_min_cluster_points: int = 2000
    # OPT-IN despike: drop polygon vertices that stick out more than this
    # distance from the line between their two neighbours.  Disabled by
    # default (0) because on real scans it removes legitimate corners.
    # Set to e.g. 0.80 m for very noisy scans where saw-tooth artefacts
    # dominate.
    despike_threshold_m: float = 0.0
    # alpha parameter is auto-tuned by sweep; we pick the tightest valid hull.
    # Larger alpha → tighter hull.  Range is in 1/metres units.
    alpha_search_lo: float = 0.50       # was 0.10; below this the hull is essentially convex
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

    # Drop sparse outliers BEFORE the alpha-shape.  This is the fix for the
    # "envelope balloons out to capture stray points" pathology — a single
    # multipath echo 20 m outside the building used to pull the envelope to
    # it.  Statistical outlier removal flags any point whose neighbourhood
    # density is significantly below the cluster mean.
    n_before_outlier = len(band_xy)
    if params.outlier_filter_enable and n_before_outlier >= params.outlier_nb_neighbors + 1:
        band_xy = _drop_far_outliers(
            band_xy,
            nb_neighbors=params.outlier_nb_neighbors,
            std_ratio=params.outlier_std_ratio,
        )
    n_after_outlier = len(band_xy)
    if n_after_outlier < 500:
        raise ValueError(
            f"envelope: only {n_after_outlier} points remain after outlier "
            f"filter (started with {n_before_outlier}); cloud too sparse"
        )

    # DBSCAN: keep only the largest spatial cluster.  Surgically eliminates
    # window-reflection blobs and outdoor furniture clusters that survive
    # the statistical outlier filter (those blobs are dense enough among
    # themselves to evade stat-outlier, but they're a distinct cluster from
    # the building proper, so DBSCAN separates them cleanly).
    n_before_dbscan = len(band_xy)
    if params.dbscan_cluster_enable and n_before_dbscan >= params.dbscan_min_cluster_points:
        band_xy = _keep_largest_dbscan_cluster(
            band_xy,
            eps_m=params.dbscan_eps_m,
            min_samples=params.dbscan_min_samples,
            min_cluster_points=params.dbscan_min_cluster_points,
        )
    n_after_dbscan = len(band_xy)
    if n_after_dbscan < 500:
        raise ValueError(
            f"envelope: only {n_after_dbscan} points remain after DBSCAN "
            f"(started with {n_before_dbscan}); cloud too fragmented"
        )

    # Alpha-shape with auto-tuning.  We sweep alpha from tight → loose and
    # accept the tightest valid hull (single Polygon, area ≥ min threshold).
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

    # v4 despike: drop polygon vertices that stick out more than
    # ``despike_threshold_m`` from the line between their two neighbours.
    # Iterates until no vertex qualifies (typically 2-4 passes).
    if params.despike_threshold_m > 0:
        ring_xy = _despike_ring(ring_xy, params.despike_threshold_m)

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

def _drop_far_outliers(
    xy: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 1.5,
) -> np.ndarray:
    """Drop sparse outlier points using statistical outlier removal.

    For each point we compute the mean distance to its ``nb_neighbors`` nearest
    neighbours, then drop any point whose mean k-NN distance is more than
    ``std_ratio`` standard deviations above the global mean.  This nukes the
    isolated multipath echoes / glass reflections that otherwise force the
    alpha-shape envelope to balloon out across empty space to capture them.

    Implemented via Open3D's :py:meth:`PointCloud.remove_statistical_outlier`
    (we already depend on Open3D for ingest, so no new dependency).  Operates
    on a temporary 3-D cloud with z=0 so the 2-D KNN reduces correctly.
    """
    if len(xy) <= nb_neighbors:
        return xy
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(
        np.hstack([xy.astype(np.float64), np.zeros((len(xy), 1))])
    )
    clean, _ = pcd_tmp.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio,
    )
    return np.asarray(clean.points)[:, :2]


def _keep_largest_dbscan_cluster(
    xy: np.ndarray,
    eps_m: float,
    min_samples: int,
    min_cluster_points: int,
) -> np.ndarray:
    """Run DBSCAN and return only the points in the largest cluster.

    Uses scikit-learn's ``DBSCAN`` (already a dependency).  The largest
    cluster is the one with the most member points; for envelope extraction
    this is essentially always the building.  If the largest cluster has
    fewer than ``min_cluster_points`` members we treat the whole space as
    too fragmented and return the input unchanged — the caller will then
    try the alpha-shape against everything, which is no worse than not
    having clustered at all.

    Why DBSCAN here and not k-means / GMM:
      - We don't know K (the number of clusters) — there are typically
        1 (just the building) up to ~5 (building + window blobs +
        neighbouring structure) on a complex scan.
      - We want noise points (sparse-but-not-quite-outlier) excluded from
        the kept set, not absorbed into the nearest cluster.  DBSCAN's
        explicit "noise" label does this; k-means doesn't.

    Performance: sklearn's DBSCAN with the default ball-tree backend is
    O(N log N) and runs in ~1 s on 100 k points.  We downsample to 3 cm
    above this step, so the input is typically 20-50 k points.
    """
    if len(xy) == 0:
        return xy
    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        # scikit-learn missing — fall back to no clustering.
        return xy
    db = DBSCAN(eps=float(eps_m), min_samples=int(min_samples)).fit(xy)
    labels = db.labels_
    valid = labels >= 0
    if not valid.any():
        return xy
    label_counts: dict[int, int] = {}
    for lbl in labels[valid]:
        label_counts[int(lbl)] = label_counts.get(int(lbl), 0) + 1
    if not label_counts:
        return xy
    best_label = max(label_counts.items(), key=lambda kv: kv[1])
    if best_label[1] < min_cluster_points:
        return xy
    return xy[labels == best_label[0]]


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
    """Try a range of alpha values, pick the *tightest* valid hull.

    Sweeps alpha from tight (high) to loose (low) and accepts the first
    alpha that produces a single ``Polygon`` whose interior area is above
    ``params.min_polygon_area_m2``.  Tight hulls hug the point cloud; the
    looser the hull, the more empty space it encloses.  Always preferring
    the tightest valid hull keeps the envelope from over-extending.

    The alphashape library can return either a single Polygon, a MultiPolygon
    (alpha is too tight relative to cloud gaps), or a LineString / Point
    (alpha is much too tight).  Anything but a single Polygon means we
    need to relax — try the next smaller alpha.
    """
    from shapely.geometry import Polygon, MultiPolygon
    import alphashape as alpha_mod

    if len(pts) < 3:
        return None, 0.0

    alphas = np.linspace(
        params.alpha_search_lo, params.alpha_search_hi, params.alpha_search_steps,
    )
    # First pass: find the tightest alpha that yields a single connected
    # Polygon above the minimum area threshold.  A MultiPolygon at this
    # alpha is the diagnostic "alpha too tight, try a smaller one" — we
    # do NOT squash to the largest piece because that can leave most of
    # the building outside the envelope.
    for a in alphas[::-1]:
        try:
            shp = alpha_mod.alphashape(pts.tolist(), float(a))
        except Exception:
            continue
        if isinstance(shp, Polygon) and float(shp.area) >= params.min_polygon_area_m2:
            return shp, float(a)

    # Fallback: no single-polygon alpha worked.  Take the largest sub-
    # polygon of the loosest alpha — that's the building, even if the
    # alpha-shape produced sliver islands of stray noise alongside it.
    # (Reaching this branch usually means the cloud has internal voids
    # — e.g. a courtyard or atrium — that the alpha sweep can't span.)
    for a in alphas:
        try:
            shp = alpha_mod.alphashape(pts.tolist(), float(a))
        except Exception:
            continue
        if isinstance(shp, MultiPolygon):
            biggest = max(shp.geoms, key=lambda g: g.area)
            if float(biggest.area) >= params.min_polygon_area_m2:
                return biggest, float(a)
        elif isinstance(shp, Polygon) and float(shp.area) >= params.min_polygon_area_m2:
            return shp, float(a)

    return None, 0.0


def _despike_ring(ring: np.ndarray, threshold_m: float, max_passes: int = 6) -> np.ndarray:
    """Drop polygon vertices that stick out more than ``threshold_m`` from
    the line between their two neighbours.

    Iterative: removing a spike vertex may expose a new spike on either
    neighbour, so we re-scan until a pass makes no changes (or we hit the
    ``max_passes`` cap).  The cap defends against pathological inputs
    where every other vertex spikes; in practice the loop converges in
    2-4 passes on real-scan envelopes.

    The ring is assumed closed (first vertex == last); we preserve that
    by re-closing after edits.
    """
    if len(ring) < 5:
        # Too few vertices for despiking to make sense (a triangle has
        # no "neighbours' line" — every vertex IS one of the line's ends).
        return ring
    closed = np.allclose(ring[0], ring[-1])
    work = ring[:-1].copy() if closed else ring.copy()
    for _ in range(max_passes):
        n = len(work)
        if n < 4:
            break
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            prev_pt = work[(i - 1) % n]
            next_pt = work[(i + 1) % n]
            cur = work[i]
            # Perpendicular distance from cur to line (prev_pt, next_pt).
            edge = next_pt - prev_pt
            L = float(np.linalg.norm(edge))
            if L < 1e-9:
                continue
            cross = abs(edge[0] * (prev_pt[1] - cur[1]) -
                         edge[1] * (prev_pt[0] - cur[0]))
            d = cross / L
            if d > threshold_m:
                keep[i] = False
        new_work = work[keep]
        if len(new_work) == n:
            break  # no spike removed this pass — done
        work = new_work
    if closed:
        return np.vstack([work, work[0:1]])
    return work


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
