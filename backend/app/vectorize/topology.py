"""Junction snap + axis-aligned extension + room inference.

v4 (Cloud2BIM-style) rewrite — see ACCURACY_TO_CAD_QUALITY_PLAN.md § 0.6.

What changed vs v3
------------------
1. **Snap tolerance is adaptive to wall thickness**, not a hard-coded 0.15 m.
   When walls are 25 cm thick (exterior block), the v3 default would *fuse*
   the two faces of a single wall into one junction.  We now derive the
   default from wall geometry: ``snap_tol = max(0.08, 0.5 * thickness_median)``.

2. **Axis-aligned extension** (Cloud2BIM § 2.8).  After endpoint snap, each
   wall is allowed to extend along its OWN axis (not the host wall's axis)
   up to ``snapping_distance_m`` if doing so causes it to meet another wall.
   This closes T-junctions where the perpendicular wall ends short of its
   host *and* the host doesn't quite reach.  The v3 behaviour (project
   endpoint to nearest host segment) failed when the host itself was short.

3. **Room finding uses planar-face traversal (angular half-edge walk)**
   (v5).  The v4 use of ``nx.minimum_cycle_basis`` was wrong on two counts:
   its node lists are documented as NOT being in cycle order (feeding them
   to the shoelace formula produced garbage areas), and it returns a cycle
   *basis*, not faces — rooms with dangling partition spurs came back with
   doubled areas.  The half-edge walk enumerates every face exactly once,
   in ring order, at each vertex turning to the next edge clockwise of the
   arrival edge.  Rings are validated with Shapely.

4. **Outer face is dropped by orientation**: with the clockwise turn rule,
   interior faces trace counter-clockwise (positive signed area) and each
   component's unbounded face traces clockwise — an exact test, replacing
   the v4 bounding-box heuristic.

The contract (TopologyResult) is unchanged so callers don't need to migrate.

Why this is one module and not two
----------------------------------
Snap, extend, and room inference all share the junction graph — splitting
them would mean rebuilding it three times.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Polygon area/perimeter math is consolidated on the canonical geometry
# model (Phase 1) — one shoelace implementation for the whole codebase.
from ..geometry.model import ring_perimeter as _polygon_perimeter  # isort: skip
from ..geometry.model import ring_signed_area as _signed_area  # isort: skip


# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class TopologyParams:
    """Tunables for junction snap + room inference.  Defaults safe for 1 cm raster."""

    # Endpoints within this distance are merged into a single junction.
    # If 0, defaults to ``max(0.08, 0.5 * thickness_median_m)`` when a
    # thickness is available (see ``build_topology``).  Set explicitly to
    # override.
    snap_tol_m: float = 0.0

    # Cloud2BIM § 2.8 "snapping distance" — how far a wall is allowed to
    # extend along its OWN axis to find a perpendicular host.  Combines
    # what v3 called ``extend_tol_m`` and what would have been a separate
    # "auto-close gaps in the same axis" step.  0.85 m matches real-scan
    # undershoot on Phase-3 floors; keep below a typical door leaf (~0.9 m).
    snapping_distance_m: float = 0.85

    # Drop faces smaller than this.  Closets/bathrooms can be ~2 m²; below
    # that it's almost certainly a topology artefact from a junction collapse.
    min_room_area_m2: float = 1.5

    # Drop the outer face (= the building envelope projected through the
    # wall graph).  Identified by largest bounding-box area / by being the
    # only cycle whose interior contains all other cycles.
    drop_outer_face: bool = True

    # Optional median wall thickness, used to auto-tune ``snap_tol_m`` if it
    # was left at 0.  Pass through from the wall-pairing stage.
    wall_thickness_median_m: float = 0.0

    # Phase 2 continuity: close residual degree-1 gaps after axis-extend.
    # When False, skip gap-close + envelope projection (tests / debug).
    close_gaps: bool = True

    # Max gap length for collinear degree-1 pairing.  0 → 0.55 m (below a
    # typical door leaf so we don't invent walls across openings).
    gap_close_tol_m: float = 0.0

    # Max distance to project a dangling end onto the building envelope.
    # 0 → reuse ``snapping_distance_m``.
    envelope_project_tol_m: float = 0.0

    # Max along-envelope arc length to bridge two free ends that already
    # sit on the hull.  0 → 1.50 m.  Closes exterior gaps without spanning
    # whole facades.
    envelope_bridge_tol_m: float = 0.0

    # Max extension / trim per leg when closing an L-corner (axis hit).
    # 0 → ``max(gap_close_tol, snapping_distance_m)`` so Phase-3 join can
    # reach ~0.6–1.0 m undershoots like BricsCAD OPTIMIZE extend/trim.
    corner_close_tol_m: float = 0.0

    # Drop short degree-1 overhang stubs left after crossing splits /
    # overshoot joins.  0 → 0.50 m (below a typical door leaf).
    trim_stub_tol_m: float = 0.0

    # Phase 3: snap near-axis wall segments to exact H/V (0°/90°) before
    # join so L/T hits land on clean intersections.  0 disables.
    manhattan_join_tol_deg: float = 8.0

    # Minimum isoperimetric quotient 4πA/P² for an accepted room ring.
    # Rectangles ≈ 0.5–0.8; long spikes fall well below ~0.12.  Set to 0.15
    # so lightly irregular scan rooms survive while orange needles do not.
    min_room_isoperimetric: float = 0.15

    # When True and ``envelope_xy`` is passed to ``build_topology``, append
    # the envelope ring as virtual wall edges so exterior gaps can close
    # rooms.  Editable ``walls`` layers are unaffected (pipeline writes
    # pre-topology centerlines).
    inject_envelope: bool = True


@dataclass
class Junction:
    x: float
    y: float
    degree: int = 0


@dataclass
class WallEdge:
    i: int                          # junction index A
    j: int                          # junction index B
    source_segment_idx: int         # original segment index


@dataclass
class RoomFace:
    vertex_ids: list[int]           # junction indices in cycle order
    polygon: np.ndarray             # (K, 2) ordered points
    area_m2: float
    perimeter_m: float


@dataclass
class TopologyResult:
    junctions: list[Junction]
    edges: list[WallEdge]
    snapped_segments: np.ndarray    # (N, 2, 2) — post-snap centerlines
    rooms: list[RoomFace] = field(default_factory=list)
    n_endpoints_merged: int = 0
    n_extended: int = 0
    # Phase 2 continuity counters (0 when close_gaps is off / no work done).
    n_gaps_closed: int = 0
    n_envelope_projections: int = 0
    n_dangling_before: int = 0
    n_dangling_after: int = 0
    n_envelope_edges_injected: int = 0
    # Cloud2BIM-style finished wall axes for editor/DXF export.  Excludes
    # virtual envelope-injected edges so the walls layer stays editable.
    finished_wall_segments: Optional[np.ndarray] = None


# ── Union-find (for endpoint snap) ───────────────────────────────────────────

def _uf_make(n: int) -> list[int]:
    return list(range(n))


def _uf_root(parent: list[int], a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _uf_union(parent: list[int], a: int, b: int) -> None:
    ra = _uf_root(parent, a)
    rb = _uf_root(parent, b)
    if ra != rb:
        parent[rb] = ra


# ── Step 1: endpoint snap ────────────────────────────────────────────────────

def _snap_endpoints(
    segments: np.ndarray,
    snap_tol_m: float,
) -> tuple[np.ndarray, list[Junction], int]:
    """Cluster endpoints within ``snap_tol_m`` into shared junctions.

    Spatial-grid bucketing keeps this O(N) for typical wall counts (< 10 000
    endpoints across the whole building).

    Returns (snapped_segments, junctions, n_endpoints_merged).
    """
    n = len(segments)
    if n == 0:
        return segments.copy(), [], 0

    pts = segments.reshape(2 * n, 2)
    cell = max(snap_tol_m, 1e-6)
    bucket: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(pts):
        key = (int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell)))
        bucket.setdefault(key, []).append(i)

    parent = _uf_make(2 * n)
    tol_sq = snap_tol_m * snap_tol_m
    for (cx, cy), ids in bucket.items():
        neighbour_ids: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k = (cx + dx, cy + dy)
                if k in bucket:
                    neighbour_ids.extend(bucket[k])
        for a in ids:
            for b in neighbour_ids:
                if a >= b:
                    continue
                d = pts[a] - pts[b]
                if d[0] * d[0] + d[1] * d[1] <= tol_sq:
                    _uf_union(parent, a, b)

    clusters: dict[int, list[int]] = {}
    for i in range(2 * n):
        r = _uf_root(parent, i)
        clusters.setdefault(r, []).append(i)

    junctions: list[Junction] = []
    junction_of: list[int] = [0] * (2 * n)
    for jid, (_root, members) in enumerate(clusters.items()):
        mean = pts[members].mean(axis=0)
        junctions.append(Junction(x=float(mean[0]), y=float(mean[1])))
        for m in members:
            junction_of[m] = jid

    snapped = np.zeros_like(segments)
    for i in range(n):
        a_id = junction_of[2 * i]
        b_id = junction_of[2 * i + 1]
        snapped[i, 0] = (junctions[a_id].x, junctions[a_id].y)
        snapped[i, 1] = (junctions[b_id].x, junctions[b_id].y)

    n_merged = 2 * n - len(junctions)
    return snapped, junctions, n_merged


# ── Step 0: split segments at mutual interior intersections ──────────────────

def _segments_interior_intersect(
    a1: np.ndarray, a2: np.ndarray,
    b1: np.ndarray, b2: np.ndarray,
    eps_t: float = 0.01,
) -> tuple[Optional[np.ndarray], float, float]:
    """Test if two segments cross strictly in their interiors.

    Returns (point, t_on_A, u_on_B).  ``eps_t`` keeps the test "interior" —
    we don't report endpoint touches (those are handled by the endpoint
    snap and the host-split passes).
    """
    d1 = a2 - a1
    d2 = b2 - b1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None, 0.0, 0.0
    diff = b1 - a1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / denom
    if t <= eps_t or t >= 1.0 - eps_t:
        return None, t, u
    if u <= eps_t or u >= 1.0 - eps_t:
        return None, t, u
    return a1 + t * d1, t, u


def _split_at_crossings(segments: np.ndarray) -> np.ndarray:
    """Split every pair of segments that cross in their interiors.

    Common in real floor plans for two interior walls forming a cross
    (corridor intersection, four-way office partition).  Without this
    step, those two walls remain as a 4-degree X in the wall graph with
    no junction at the crossing — and the planar cycle basis would not
    enumerate the 4 quadrant rooms.

    Implementation: O(N²) pair test, but on typical buildings (< 500
    walls) this is < 50 ms.  Each split adds at most 4 new segments
    (each of the two crossing walls splits into 2 halves).
    """
    if len(segments) < 2:
        return segments
    # We need to split iteratively until no new crossings appear (a triple
    # crossing requires two splits).  Bound iterations to defend against
    # degenerate inputs.
    work = [s.copy() for s in segments]
    max_iter = 8
    for _ in range(max_iter):
        new_segs: list[np.ndarray] = []
        split_any = False
        # Build a "splits per segment" map first so we don't double-emit.
        n = len(work)
        cuts: list[list[float]] = [[0.0, 1.0] for _ in range(n)]
        for i in range(n):
            a1, a2 = work[i][0], work[i][1]
            for j in range(i + 1, n):
                b1, b2 = work[j][0], work[j][1]
                pt, t, u = _segments_interior_intersect(a1, a2, b1, b2)
                if pt is None:
                    continue
                cuts[i].append(t)
                cuts[j].append(u)
                split_any = True
        # Materialise.
        for k in range(n):
            ts = sorted(set(cuts[k]))
            a, b = work[k][0], work[k][1]
            for li in range(len(ts) - 1):
                t0, t1 = ts[li], ts[li + 1]
                if t1 - t0 < 1e-6:
                    continue
                p0 = a + t0 * (b - a)
                p1 = a + t1 * (b - a)
                new_segs.append(np.array([p0, p1], dtype=segments.dtype))
        work = new_segs
        if not split_any:
            break
    return np.asarray(work, dtype=segments.dtype).reshape(-1, 2, 2)


# ── Step 2a: split host at dangling endpoints already on host ────────────────

def _project_point_to_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Project ``p`` onto segment [a, b].  Returns (foot, t, dist).

    ``t`` is the parameter along [a, b] (0 = a, 1 = b, clamped to [0, 1]).
    """
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 < 1e-12:
        return a.copy(), 0.0, float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / L2)
    t_clamped = max(0.0, min(1.0, t))
    foot = a + t_clamped * ab
    dist = float(np.linalg.norm(p - foot))
    return foot, t_clamped, dist


def _split_hosts_at_dangling(
    snapped: np.ndarray,
    junctions: list[Junction],
    snap_tol: float,
) -> tuple[np.ndarray, list[Junction], int]:
    """Split host walls whose interior is touched by a degree-1 endpoint.

    Use case: an internal wall's endpoint lies *on* an outer wall (within
    ``snap_tol``), but the outer wall has no junction at that point.  We
    split the outer wall so the graph gains a degree-3 junction there —
    necessary for the cycle basis to enumerate both rooms on either side
    of the internal wall.

    This is the v3 logic, restored as a complementary step to the
    Cloud2BIM-style own-axis extension (which handles "endpoint is close
    but not touching", a different case).

    Returns (segs, junctions, n_splits).
    """
    n_segs = len(snapped)
    if n_segs == 0:
        return snapped, junctions, 0

    segs = snapped.copy()

    def _jidx(pt: np.ndarray) -> int:
        for k, j in enumerate(junctions):
            if abs(j.x - pt[0]) < 1e-9 and abs(j.y - pt[1]) < 1e-9:
                return k
        return -1

    # Compute degree from current segs.
    deg = [0] * len(junctions)
    for s in segs:
        ai = _jidx(s[0])
        bi = _jidx(s[1])
        if ai >= 0:
            deg[ai] += 1
        if bi >= 0:
            deg[bi] += 1

    # We iterate by endpoint; mutation of segs (appending split halves)
    # invalidates the loop bound at the start of the function, so we cap
    # iteration count to the original segs count to avoid endlessly
    # splitting a freshly-split host.
    n_splits = 0
    eps_t = 0.02  # don't split within 2% of either end → snap to endpoint
    for k in range(n_segs):
        for end in (0, 1):
            ep = segs[k, end].copy()
            j_idx = _jidx(ep)
            if j_idx == -1 or deg[j_idx] != 1:
                continue

            best_host = -1
            best_foot = None
            best_dist = snap_tol + 1e-9
            best_t = 0.0
            for h in range(len(segs)):
                if h == k:
                    continue
                a = segs[h, 0]
                b = segs[h, 1]
                foot, t, dist = _project_point_to_segment(ep, a, b)
                if dist >= best_dist:
                    continue
                if t < eps_t or t > 1.0 - eps_t:
                    # Endpoint is close to the host's own endpoint, not interior.
                    # Skip — endpoint snap should have handled it (or will, via
                    # axis-extension snapping to endpoint).
                    continue
                best_dist = dist
                best_host = h
                best_foot = foot
                best_t = t

            if best_host == -1 or best_foot is None:
                continue

            # Snap our endpoint exactly onto the host (the projection may
            # have moved it by `best_dist` < snap_tol).
            new_pt = best_foot
            # Materialise / find junction at new_pt.
            new_jidx = -1
            for k_j, j in enumerate(junctions):
                if abs(j.x - new_pt[0]) < 1e-9 and abs(j.y - new_pt[1]) < 1e-9:
                    new_jidx = k_j
                    break
            if new_jidx == -1:
                junctions.append(Junction(x=float(new_pt[0]), y=float(new_pt[1])))
                deg.append(0)
                new_jidx = len(junctions) - 1

            deg[j_idx] -= 1
            deg[new_jidx] += 1
            segs[k, end] = new_pt

            # Prefer TRIM of a nearby degree-1 host free end over a T-split
            # that would leave a short overhang stub (Phase 3 corner finish).
            a = segs[best_host, 0].copy()
            b = segs[best_host, 1].copy()
            d_to_a = float(np.linalg.norm(new_pt - a))
            d_to_b = float(np.linalg.norm(new_pt - b))
            ja = _jidx(a)
            jb = _jidx(b)
            trimmed = False
            if ja >= 0 and deg[ja] == 1 and d_to_a <= snap_tol and d_to_a <= d_to_b:
                deg[ja] -= 1
                segs[best_host, 0] = new_pt
                deg[new_jidx] += 1
                trimmed = True
            elif jb >= 0 and deg[jb] == 1 and d_to_b <= snap_tol:
                deg[jb] -= 1
                segs[best_host, 1] = new_pt
                deg[new_jidx] += 1
                trimmed = True

            if not trimmed:
                # Split host into (a, new_pt) + (new_pt, b).
                segs[best_host, 0] = a
                segs[best_host, 1] = new_pt
                new_piece = np.array([[new_pt[0], new_pt[1]],
                                      [b[0], b[1]]], dtype=segs.dtype)
                segs = np.vstack([segs, new_piece[None, :, :]])
                deg[new_jidx] += 2  # +2 because we added two halves both incident
            n_splits += 1

    return segs, junctions, n_splits


# ── Step 2a′: Manhattan axis snap before join (Phase 3) ──────────────────────

def _manhattan_snap_segments(
    segments: np.ndarray,
    tol_deg: float,
) -> np.ndarray:
    """Snap near-H/V segments to exact 0°/90° (world axes), preserve centre.

    Rectilinear floors only — walls already within ``tol_deg`` of cardinal
    axes are forced onto those axes so subsequent L/T hits land on clean
    intersections instead of near-misses from contour chord noise.
    """
    segs = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    if len(segs) == 0 or tol_deg <= 0:
        return segs.copy() if len(segs) else segs

    out = segs.copy()
    for i, seg in enumerate(out):
        d = seg[1] - seg[0]
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        ang = abs(float(np.degrees(np.arctan2(d[1], d[0])))) % 180.0
        # Distance to nearest cardinal (0 or 90).
        to_h = min(ang, 180.0 - ang)          # horizontal
        to_v = abs(ang - 90.0)                # vertical
        if to_h <= tol_deg and to_h <= to_v:
            snap_rad = 0.0
        elif to_v <= tol_deg:
            snap_rad = np.pi / 2.0
        else:
            continue
        centre = (seg[0] + seg[1]) / 2.0
        half = L / 2.0
        dx = float(np.cos(snap_rad)) * half
        dy = float(np.sin(snap_rad)) * half
        out[i, 0] = [centre[0] - dx, centre[1] - dy]
        out[i, 1] = [centre[0] + dx, centre[1] + dy]
    return out


# ── Step 2b: axis-aligned wall extension (Cloud2BIM § 2.8) ───────────────────

def _extend_along_own_axis(
    snapped: np.ndarray,
    junctions: list[Junction],
    snapping_distance_m: float,
    *,
    max_passes: int = 3,
) -> tuple[np.ndarray, list[Junction], int]:
    """Extend each wall along its OWN axis to close gaps with other walls.

    For each segment endpoint that is currently the only endpoint at its
    junction (degree 1), extend the segment along its axis by up to
    ``snapping_distance_m`` and check whether the extended ray intersects
    another segment.  If yes, snap to the intersection.

    This is the Cloud2BIM 2025 "snapping distance" rule: walls grow into
    each other along the direction of travel, not by perpendicular
    projection.  Effect: rooms close because half-walls reach their
    perpendicular partners; T-junctions form even when the host wall is
    itself short.

    Phase 3: also accepts infinite-line hits just past a host endpoint
    (host undershoot) by extending the host to the hit, and re-runs until
    a pass finds nothing (capped) so cascading L/T closes land.

    Returns (new_segments, new_junctions, n_extended).
    """
    n_segs = len(snapped)
    if n_segs == 0 or snapping_distance_m <= 0:
        return snapped, junctions, 0

    segs = snapped.copy()
    tol = float(snapping_distance_m)
    # How far past a host endpoint we still treat as "extend the host"
    # rather than a clean miss (BricsCAD-style trim/extend partner reach).
    host_end_pad = max(0.25, min(tol, 1.0))
    n_extended_total = 0

    for _pass in range(max(1, int(max_passes))):
        # Recompute degree each pass — prior hits change connectivity.
        deg = _junction_degrees(segs, junctions)

        def _jidx(pt: np.ndarray) -> int:
            for k, j in enumerate(junctions):
                if abs(j.x - pt[0]) < 1e-9 and abs(j.y - pt[1]) < 1e-9:
                    return k
            return -1

        n_this = 0
        n_at_start = len(segs)
        for k in range(n_at_start):
            for end in (0, 1):
                ep = segs[k, end].copy()
                other = segs[k, 1 - end].copy()
                j_idx = _jidx(ep)
                if j_idx == -1 or deg[j_idx] != 1:
                    continue

                d = ep - other
                L = float(np.linalg.norm(d))
                if L < 1e-9:
                    continue
                d /= L

                best_hit = None
                best_t = float("inf")
                best_host = -1
                best_u = 0.0
                for h in range(len(segs)):
                    if h == k:
                        continue
                    a = segs[h, 0]
                    b = segs[h, 1]
                    host_L = float(np.linalg.norm(b - a))
                    if host_L < 1e-9:
                        continue
                    # Finite segment first (true T onto host body).
                    hit, t_ray, u_host = _ray_segment_intersect(ep, d, a, b)
                    if (
                        hit is not None
                        and 1e-6 < t_ray <= tol
                        and 0.0 <= u_host <= 1.0
                        and t_ray < best_t
                    ):
                        best_t = t_ray
                        best_hit = hit
                        best_host = h
                        best_u = u_host
                        continue
                    # Infinite host line — catch near-endpoint undershoots
                    # (both walls need to grow to the corner).
                    hit2, t2, u2 = _ray_line_intersect(ep, d, a, b)
                    if hit2 is None or t2 <= 1e-6 or t2 > tol:
                        continue
                    pad_u = host_end_pad / host_L
                    if u2 < -pad_u or u2 > 1.0 + pad_u:
                        continue
                    if t2 < best_t:
                        best_t = t2
                        best_hit = hit2
                        best_host = h
                        best_u = u2

                if best_hit is None or best_host == -1:
                    continue

                a = segs[best_host, 0].copy()
                b = segs[best_host, 1].copy()
                d_to_a = float(np.linalg.norm(best_hit - a))
                d_to_b = float(np.linalg.norm(best_hit - b))
                snap_thresh = max(tol * 0.5, 0.05)

                host_extend = False
                host_end_to_move = 0
                if best_u < 0.0:
                    # Hit past endpoint a — extend host a → hit.
                    new_pt = best_hit
                    host_split = False
                    host_extend = True
                    host_end_to_move = 0
                elif best_u > 1.0:
                    new_pt = best_hit
                    host_split = False
                    host_extend = True
                    host_end_to_move = 1
                elif d_to_a < snap_thresh:
                    new_pt = a.copy()
                    host_split = False
                elif d_to_b < snap_thresh:
                    new_pt = b.copy()
                    host_split = False
                else:
                    # Prefer TRIM of a nearby degree-1 host free end over a
                    # T-split that leaves a short stub (BricsCAD extend/trim).
                    ja = _jidx(a)
                    jb = _jidx(b)
                    trim_end = -1
                    if ja >= 0 and deg[ja] == 1 and d_to_a <= tol:
                        trim_end = 0
                    elif jb >= 0 and deg[jb] == 1 and d_to_b <= tol:
                        trim_end = 1
                    if trim_end >= 0:
                        new_pt = best_hit
                        host_split = False
                        host_extend = True  # move free end to hit (shorten)
                        host_end_to_move = trim_end
                    else:
                        new_pt = best_hit
                        host_split = True

                new_jidx = -1
                for j_k, j in enumerate(junctions):
                    if abs(j.x - new_pt[0]) < 1e-9 and abs(j.y - new_pt[1]) < 1e-9:
                        new_jidx = j_k
                        break
                if new_jidx == -1:
                    junctions.append(Junction(x=float(new_pt[0]), y=float(new_pt[1])))
                    deg.append(0)
                    new_jidx = len(junctions) - 1

                deg[j_idx] -= 1
                deg[new_jidx] += 1
                segs[k, end] = new_pt

                if host_extend:
                    # Move the undershot host endpoint to the corner hit so
                    # both walls share the junction (true L, not T-on-stub).
                    old_host_ep = segs[best_host, host_end_to_move].copy()
                    old_j = _jidx(old_host_ep)
                    if old_j >= 0 and deg[old_j] == 1:
                        deg[old_j] -= 1
                    segs[best_host, host_end_to_move] = new_pt
                    deg[new_jidx] += 1
                elif host_split:
                    # MUST copy endpoints before mutate (far-half regression).
                    segs[best_host, 0] = a
                    segs[best_host, 1] = new_pt
                    new_piece = np.array(
                        [[new_pt[0], new_pt[1]], [b[0], b[1]]],
                        dtype=segs.dtype,
                    )
                    segs = np.vstack([segs, new_piece[None, :, :]])
                    deg[new_jidx] += 1  # degree-3 at split

                n_this += 1

        n_extended_total += n_this
        if n_this == 0:
            break

    return segs, junctions, n_extended_total


def _ray_segment_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[Optional[np.ndarray], float, float]:
    """Intersect a 2D ray (origin + t*direction, t ≥ 0) with segment [a, b].

    Returns (point, t_along_ray, u_along_segment) or (None, _, _) if no
    intersection.

    Uses the standard 2D ray-segment formula:
        origin + t * d = a + u * (b - a)
    Solve the 2×2 linear system for (t, u).
    """
    s = b - a
    denom = direction[0] * s[1] - direction[1] * s[0]
    if abs(denom) < 1e-12:
        return None, 0.0, 0.0
    diff = a - origin
    t = (diff[0] * s[1] - diff[1] * s[0]) / denom
    u = (diff[0] * direction[1] - diff[1] * direction[0]) / denom
    if t < 0 or u < 0 or u > 1:
        return None, t, u
    return origin + t * direction, t, u


def _ray_line_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[Optional[np.ndarray], float, float]:
    """Intersect ray with the infinite line through [a, b] (u unrestricted).

    Same solver as ``_ray_segment_intersect`` but does not clamp ``u`` to
    [0, 1].  Used to find L-corner hits just past a short host endpoint.
    """
    s = b - a
    denom = direction[0] * s[1] - direction[1] * s[0]
    if abs(denom) < 1e-12:
        return None, 0.0, 0.0
    diff = a - origin
    t = (diff[0] * s[1] - diff[1] * s[0]) / denom
    u = (diff[0] * direction[1] - diff[1] * direction[0]) / denom
    if t < 0:
        return None, t, u
    return origin + t * direction, t, u


# ── Step 2c: Phase 2 continuity (gap-close + envelope project) ───────────────

def _junction_degrees(segs: np.ndarray, junctions: list[Junction]) -> list[int]:
    """Recount junction degrees from current segment endpoints (exact match)."""
    deg = [0] * len(junctions)

    def _jidx(pt: np.ndarray) -> int:
        for k, j in enumerate(junctions):
            if abs(j.x - pt[0]) < 1e-9 and abs(j.y - pt[1]) < 1e-9:
                return k
        return -1

    for s in segs:
        ai = _jidx(s[0])
        bi = _jidx(s[1])
        if ai >= 0:
            deg[ai] += 1
        if bi >= 0:
            deg[bi] += 1
    return deg


def _count_degree1(deg: list[int]) -> int:
    return sum(1 for d in deg if d == 1)


def _dangling_incident(
    segs: np.ndarray,
    junctions: list[Junction],
    deg: list[int],
) -> list[tuple[int, int, int, np.ndarray]]:
    """Return (junc_idx, seg_idx, end, outward_unit) for each degree-1 end."""
    out: list[tuple[int, int, int, np.ndarray]] = []

    def _jidx(pt: np.ndarray) -> int:
        for k, j in enumerate(junctions):
            if abs(j.x - pt[0]) < 1e-9 and abs(j.y - pt[1]) < 1e-9:
                return k
        return -1

    for k in range(len(segs)):
        for end in (0, 1):
            ep = segs[k, end]
            j_idx = _jidx(ep)
            if j_idx < 0 or deg[j_idx] != 1:
                continue
            other = segs[k, 1 - end]
            d = ep - other
            L = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            out.append((j_idx, k, end, d / L))
    return out


def _snap_dangling_endpoints(
    snapped: np.ndarray,
    junctions: list[Junction],
    snap_tol_m: float,
) -> tuple[np.ndarray, list[Junction], int]:
    """Merge nearby degree-1 endpoints even when wall axes are not collinear.

    Classic endpoint snap already ran at a thickness-derived tol; free ends
    that sit 10–30 cm apart laterally (BricsCAD-style corner near-miss) are
    still open.  Pulling those coincidences into one junction closes L/T
    corners without inventing a bridge segment.
    """
    if len(snapped) == 0 or snap_tol_m <= 0:
        return snapped, junctions, 0

    segs = snapped.copy()
    deg = _junction_degrees(segs, junctions)
    dangling = _dangling_incident(segs, junctions, deg)
    if len(dangling) < 2:
        return segs, junctions, 0

    cell = max(snap_tol_m, 1e-6)
    bucket: dict[tuple[int, int], list[int]] = {}
    for i, (j_idx, _k, _e, _d) in enumerate(dangling):
        j = junctions[j_idx]
        key = (int(np.floor(j.x / cell)), int(np.floor(j.y / cell)))
        bucket.setdefault(key, []).append(i)

    used: set[int] = set()
    n_merged = 0
    tol_sq = snap_tol_m * snap_tol_m

    for i, (ji, ki, ei, _di) in enumerate(dangling):
        if i in used or deg[ji] != 1:
            continue
        pi = np.array([junctions[ji].x, junctions[ji].y], dtype=float)
        cx, cy = int(np.floor(pi[0] / cell)), int(np.floor(pi[1] / cell))
        best_j = -1
        best_dist_sq = tol_sq
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in bucket.get((cx + dx, cy + dy), []):
                    if j <= i or j in used:
                        continue
                    jj, kj, ej, _dj = dangling[j]
                    if jj == ji or deg[jj] != 1:
                        continue
                    pj = np.array([junctions[jj].x, junctions[jj].y], dtype=float)
                    dist_sq = float(np.sum((pj - pi) ** 2))
                    if dist_sq < 1e-12 or dist_sq > best_dist_sq:
                        continue
                    best_dist_sq = dist_sq
                    best_j = j
        if best_j < 0:
            continue

        jj, kj, ej, _dj = dangling[best_j]
        pj = np.array([junctions[jj].x, junctions[jj].y], dtype=float)
        mid = 0.5 * (pi + pj)

        # Materialise mid junction.
        new_jidx = -1
        for jk, junc in enumerate(junctions):
            if abs(junc.x - mid[0]) < 1e-6 and abs(junc.y - mid[1]) < 1e-6:
                new_jidx = jk
                break
        if new_jidx == -1:
            junctions.append(Junction(x=float(mid[0]), y=float(mid[1])))
            deg.append(0)
            new_jidx = len(junctions) - 1

        for j_old, k_seg, end in ((ji, ki, ei), (jj, kj, ej)):
            if deg[j_old] == 1:
                deg[j_old] -= 1
            deg[new_jidx] += 1
            segs[k_seg, end] = mid
            junctions[j_old].x = float(mid[0])
            junctions[j_old].y = float(mid[1])

        used.add(i)
        used.add(best_j)
        n_merged += 1

    return segs, junctions, n_merged


def _close_degree1_gaps(
    snapped: np.ndarray,
    junctions: list[Junction],
    gap_tol_m: float,
) -> tuple[np.ndarray, list[Junction], int]:
    """Bridge pairs of degree-1 ends that face each other across a short gap.

    Prefer roughly collinear opposing free ends (dot of outward dirs ≈ −1).
    Inserts a new centerline segment between the two endpoints so the graph
    gains an edge without inventing geometry far from observed walls.

    Doorways that are wider than ``gap_tol_m`` are left open on purpose.
    """
    if len(snapped) == 0 or gap_tol_m <= 0:
        return snapped, junctions, 0

    segs = snapped.copy()
    deg = _junction_degrees(segs, junctions)
    dangling = _dangling_incident(segs, junctions, deg)
    if len(dangling) < 2:
        return segs, junctions, 0

    # Spatial grid on dangling junction positions.
    cell = max(gap_tol_m, 1e-6)
    bucket: dict[tuple[int, int], list[int]] = {}
    for i, (j_idx, _k, _end, _d) in enumerate(dangling):
        j = junctions[j_idx]
        key = (int(np.floor(j.x / cell)), int(np.floor(j.y / cell)))
        bucket.setdefault(key, []).append(i)

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    tol_sq = gap_tol_m * gap_tol_m
    # Cosine threshold: outward dirs should oppose (collinear gap) or be
    # nearly opposite.  −0.7 ≈ ~135° — allows slight L-corner misalignment.
    min_oppose = -0.70

    for i, (ji, _ki, _ei, di) in enumerate(dangling):
        if i in used:
            continue
        pi = np.array([junctions[ji].x, junctions[ji].y], dtype=float)
        cx, cy = int(np.floor(pi[0] / cell)), int(np.floor(pi[1] / cell))
        best_j = -1
        best_dist_sq = tol_sq
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in bucket.get((cx + dx, cy + dy), []):
                    if j <= i or j in used:
                        continue
                    jj, _kj, _ej, dj = dangling[j]
                    if jj == ji:
                        continue
                    # Outward directions should roughly oppose.
                    if float(np.dot(di, dj)) > min_oppose:
                        continue
                    pj = np.array([junctions[jj].x, junctions[jj].y], dtype=float)
                    delta = pj - pi
                    dist_sq = float(delta[0] * delta[0] + delta[1] * delta[1])
                    if dist_sq < 1e-12 or dist_sq > best_dist_sq:
                        continue
                    # Gap chord should align with either wall axis — but
                    # near-miss corners can be slightly skewed (0.70 floor).
                    gap_dir = delta / np.sqrt(dist_sq)
                    align_i = abs(float(np.dot(gap_dir, di)))
                    align_j = abs(float(np.dot(gap_dir, dj)))
                    if max(align_i, align_j) < 0.70:
                        continue
                    best_dist_sq = dist_sq
                    best_j = j
        if best_j >= 0:
            used.add(i)
            used.add(best_j)
            pairs.append((i, best_j))

    if not pairs:
        return segs, junctions, 0

    n_closed = 0
    new_rows: list[np.ndarray] = []
    for i, j in pairs:
        ji, _ki, _ei, _di = dangling[i]
        jj, _kj, _ej, _dj = dangling[j]
        a = np.array([junctions[ji].x, junctions[ji].y], dtype=float)
        b = np.array([junctions[jj].x, junctions[jj].y], dtype=float)
        if float(np.linalg.norm(b - a)) < 1e-9:
            continue
        new_rows.append(np.array([a, b], dtype=float))
        # Both ends gain one incident edge → degree 1 → 2.
        deg[ji] += 1
        deg[jj] += 1
        n_closed += 1

    if new_rows:
        segs = np.vstack([segs, np.asarray(new_rows, dtype=float)])
    return segs, junctions, n_closed


def _close_l_corners(
    snapped: np.ndarray,
    junctions: list[Junction],
    corner_tol_m: float,
) -> tuple[np.ndarray, list[Junction], int]:
    """Close near-miss L-corners by EXTEND or TRIM to the axis intersection.

    Two degree-1 ends whose outward directions are roughly perpendicular and
    whose axis-lines intersect within ``corner_tol_m`` of each endpoint are
    snapped to that intersection (forming a shared corner junction).

    Phase 3 (BricsCAD OPTIMIZE): ``t`` / ``s`` may be negative (undershoot →
    EXTEND) or positive beyond the free end along +outward (overshoot → TRIM
    by moving the endpoint back to the hit).  Magnitude is still capped by
    ``corner_tol_m``.
    """
    if len(snapped) == 0 or corner_tol_m <= 0:
        return snapped, junctions, 0

    segs = snapped.copy()
    deg = _junction_degrees(segs, junctions)
    dangling = _dangling_incident(segs, junctions, deg)
    if len(dangling) < 2:
        return segs, junctions, 0

    used: set[int] = set()
    n_closed = 0
    # |dot| near 0 → perpendicular; allow up to ~25° off square.
    max_abs_dot = 0.42

    for i, (ji, ki, ei, di) in enumerate(dangling):
        if i in used or deg[ji] != 1:
            continue
        pi = np.array([junctions[ji].x, junctions[ji].y], dtype=float)
        best_j = -1
        best_hit = None
        best_cost = float("inf")
        for j, (jj, kj, ej, dj) in enumerate(dangling):
            if j <= i or j in used or deg[jj] != 1:
                continue
            if abs(float(np.dot(di, dj))) > max_abs_dot:
                continue
            pj = np.array([junctions[jj].x, junctions[jj].y], dtype=float)
            # Solve pi + t*di = pj + s*dj for (t, s).
            # t>0 = extend beyond free end; t<0 = trim (overshoot along +di).
            denom = di[0] * dj[1] - di[1] * dj[0]
            if abs(denom) < 1e-12:
                continue
            diff = pj - pi
            t = (diff[0] * dj[1] - diff[1] * dj[0]) / denom
            s = (di[0] * diff[1] - di[1] * diff[0]) / denom
            if abs(t) > corner_tol_m or abs(s) > corner_tol_m:
                continue
            # Reject pure backspace that would collapse a wall to nothing.
            # The free end moves by |t| along the wall; keep ≥ 5 cm length.
            seg_i_len = float(np.linalg.norm(segs[ki, 1] - segs[ki, 0]))
            seg_j_len = float(np.linalg.norm(segs[kj, 1] - segs[kj, 0]))
            if t < 0 and seg_i_len + t < 0.05:
                continue
            if s < 0 and seg_j_len + s < 0.05:
                continue
            hit = pi + t * di
            cost = abs(t) + abs(s)
            if cost < best_cost:
                best_cost = cost
                best_j = j
                best_hit = hit

        if best_j < 0 or best_hit is None:
            continue

        jj, kj, ej, _dj = dangling[best_j]
        new_pt = best_hit

        # Materialise / find junction at the corner.
        new_jidx = -1
        for jk, junc in enumerate(junctions):
            if abs(junc.x - new_pt[0]) < 1e-6 and abs(junc.y - new_pt[1]) < 1e-6:
                new_jidx = jk
                break
        if new_jidx == -1:
            junctions.append(Junction(x=float(new_pt[0]), y=float(new_pt[1])))
            deg.append(0)
            new_jidx = len(junctions) - 1

        # Move both free ends to the corner (extend or trim).
        for j_old, k_seg, end in ((ji, ki, ei), (jj, kj, ej)):
            if deg[j_old] == 1:
                deg[j_old] -= 1
            deg[new_jidx] += 1
            segs[k_seg, end] = new_pt
            junctions[j_old].x = float(new_pt[0])
            junctions[j_old].y = float(new_pt[1])

        used.add(i)
        used.add(best_j)
        n_closed += 1

    return segs, junctions, n_closed


def _trim_dangling_stubs(
    snapped: np.ndarray,
    junctions: list[Junction],
    stub_tol_m: float,
) -> tuple[np.ndarray, list[Junction], int]:
    """Drop short degree-1 overhangs left by crossing splits / overshoots.

    A stub is a segment with at least one degree-1 end whose length is
    ≤ ``stub_tol_m``.  Removing them cleans T-junction leftovers so rooms
    don't inherit needle faces.  Longer open walls (doors / true gaps) are
    kept.
    """
    if len(snapped) == 0 or stub_tol_m <= 0:
        return snapped, junctions, 0

    segs = snapped.copy()
    deg = _junction_degrees(segs, junctions)

    def _jidx(pt: np.ndarray) -> int:
        for k, j in enumerate(junctions):
            if abs(j.x - pt[0]) < 1e-9 and abs(j.y - pt[1]) < 1e-9:
                return k
        return -1

    keep: list[np.ndarray] = []
    n_trimmed = 0
    for s in segs:
        L = float(np.linalg.norm(s[1] - s[0]))
        ai = _jidx(s[0])
        bi = _jidx(s[1])
        dang = (ai >= 0 and deg[ai] == 1) or (bi >= 0 and deg[bi] == 1)
        if dang and L <= stub_tol_m:
            n_trimmed += 1
            continue
        keep.append(s)

    if n_trimmed == 0:
        return segs, junctions, 0
    if not keep:
        return np.zeros((0, 2, 2), dtype=float), junctions, n_trimmed
    return np.asarray(keep, dtype=float), junctions, n_trimmed


def _bridge_envelope_gaps(
    snapped: np.ndarray,
    junctions: list[Junction],
    envelope_xy: np.ndarray,
    bridge_tol_m: float,
    on_hull_tol_m: float = 0.08,
) -> tuple[np.ndarray, list[Junction], int]:
    """Insert short chords between dangling ends that already sit on the hull.

    When two free ends lie on the envelope within ``bridge_tol_m`` along the
    exterior (chord length), add a wall segment between them.  This closes
    exterior gaps that projection alone cannot (projection moves points but
    does not add edges).
    """
    if (
        len(snapped) == 0
        or bridge_tol_m <= 0
        or envelope_xy is None
        or len(envelope_xy) < 3
    ):
        return snapped, junctions, 0

    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        return snapped, junctions, 0

    ring = np.asarray(envelope_xy, dtype=float).reshape(-1, 2)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    try:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return snapped, junctions, 0
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        exterior = poly.exterior
    except Exception:
        return snapped, junctions, 0

    segs = snapped.copy()
    deg = _junction_degrees(segs, junctions)
    dangling = _dangling_incident(segs, junctions, deg)
    if len(dangling) < 2:
        return segs, junctions, 0

    # Keep only ends already near the hull.
    on_hull: list[tuple[int, int, int, np.ndarray, float]] = []
    for j_idx, k, end, outward in dangling:
        pt = Point(float(junctions[j_idx].x), float(junctions[j_idx].y))
        dist = float(exterior.distance(pt))
        if dist <= on_hull_tol_m:
            s_param = float(exterior.project(pt))
            on_hull.append((j_idx, k, end, outward, s_param))

    if len(on_hull) < 2:
        return segs, junctions, 0

    peri = float(exterior.length)
    used: set[int] = set()
    new_rows: list[np.ndarray] = []
    n_bridged = 0

    for i, (ji, _ki, _ei, _di, si) in enumerate(on_hull):
        if i in used:
            continue
        best_j = -1
        best_arc = bridge_tol_m
        for j, (jj, _kj, _ej, _dj, sj) in enumerate(on_hull):
            if j <= i or j in used or jj == ji:
                continue
            # Shortest arc along the closed ring.
            raw = abs(sj - si)
            arc = min(raw, peri - raw) if peri > 0 else raw
            if arc < 1e-3 or arc > best_arc:
                continue
            # Chord length must also be short (reject wrapping across building).
            pi = np.array([junctions[ji].x, junctions[ji].y], dtype=float)
            pj = np.array([junctions[jj].x, junctions[jj].y], dtype=float)
            chord = float(np.linalg.norm(pj - pi))
            if chord > bridge_tol_m or chord < 1e-3:
                continue
            best_arc = arc
            best_j = j
        if best_j < 0:
            continue
        jj = on_hull[best_j][0]
        a = np.array([junctions[ji].x, junctions[ji].y], dtype=float)
        b = np.array([junctions[jj].x, junctions[jj].y], dtype=float)
        new_rows.append(np.array([a, b], dtype=float))
        deg[ji] += 1
        deg[jj] += 1
        used.add(i)
        used.add(best_j)
        n_bridged += 1

    if new_rows:
        segs = np.vstack([segs, np.asarray(new_rows, dtype=float)])
    return segs, junctions, n_bridged


def _project_dangling_to_envelope(
    snapped: np.ndarray,
    junctions: list[Junction],
    envelope_xy: np.ndarray,
    project_tol_m: float,
) -> tuple[np.ndarray, list[Junction], int]:
    """Move near-envelope degree-1 ends onto the building hull.

    Prefer extending along the wall's own axis until it hits the envelope
    ring; fall back to nearest-point projection when the axis miss is still
    within ``project_tol_m``.  Does not inject envelope edges into the wall
    graph — only relocates existing free ends.
    """
    if (
        len(snapped) == 0
        or project_tol_m <= 0
        or envelope_xy is None
        or len(envelope_xy) < 3
    ):
        return snapped, junctions, 0

    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        return snapped, junctions, 0

    ring = np.asarray(envelope_xy, dtype=float).reshape(-1, 2)
    # Ensure closed ring for Polygon.
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    try:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
            return snapped, junctions, 0
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        exterior = poly.exterior
    except Exception:
        return snapped, junctions, 0

    # Envelope edges as segments for ray hits.
    env_segs = [
        (np.asarray(exterior.coords[i], dtype=float),
         np.asarray(exterior.coords[i + 1], dtype=float))
        for i in range(len(exterior.coords) - 1)
    ]

    segs = snapped.copy()
    deg = _junction_degrees(segs, junctions)
    dangling = _dangling_incident(segs, junctions, deg)
    if not dangling:
        return segs, junctions, 0

    n_proj = 0
    tol = float(project_tol_m)

    for j_idx, k, end, outward in dangling:
        # Re-check degree — a prior projection in this loop may have changed it.
        if deg[j_idx] != 1:
            continue
        ep = segs[k, end].copy()
        # Skip if already essentially on the hull.
        dist0 = float(exterior.distance(Point(float(ep[0]), float(ep[1]))))
        if dist0 <= 1e-3:
            continue
        if dist0 > tol * 2.0:
            # Far from envelope — not an exterior stub.
            continue

        # 1) Axis-extend toward hull.
        best_hit = None
        best_t = float("inf")
        for a, b in env_segs:
            hit, t_ray, _u = _ray_segment_intersect(ep, outward, a, b)
            if hit is None:
                continue
            if t_ray <= 1e-6 or t_ray > tol:
                continue
            if t_ray < best_t:
                best_t = t_ray
                best_hit = hit

        new_pt = None
        if best_hit is not None:
            new_pt = best_hit
        else:
            # 2) Nearest-point projection if still within tol.
            nearest = exterior.interpolate(exterior.project(
                Point(float(ep[0]), float(ep[1]))
            ))
            cand = np.array([nearest.x, nearest.y], dtype=float)
            if float(np.linalg.norm(cand - ep)) <= tol:
                # Prefer projection when wall is roughly normal/parallel to
                # the local hull tangent (avoid sliding along the facade).
                new_pt = cand

        if new_pt is None:
            continue
        if float(np.linalg.norm(new_pt - ep)) < 1e-9:
            continue

        # Relocate endpoint + junction (keep identity; move coordinates).
        junctions[j_idx].x = float(new_pt[0])
        junctions[j_idx].y = float(new_pt[1])
        segs[k, end] = new_pt
        n_proj += 1

    return segs, junctions, n_proj


def _apply_continuity(
    snapped: np.ndarray,
    junctions: list[Junction],
    params: TopologyParams,
    envelope_xy: Optional[np.ndarray],
) -> tuple[np.ndarray, list[Junction], int, int, int, int]:
    """Run gap-close + envelope project; return counters.

    Returns
    -------
    segs, junctions, n_gaps_closed, n_envelope_projections,
    n_dangling_before, n_dangling_after
    """
    deg0 = _junction_degrees(snapped, junctions)
    n_before = _count_degree1(deg0)

    if not params.close_gaps:
        return snapped, junctions, 0, 0, n_before, n_before

    # Default gap tol sits under a typical door leaf (~0.7–0.9 m) so we
    # bridge scanner misses without inventing walls across openings.
    gap_tol = (
        float(params.gap_close_tol_m)
        if params.gap_close_tol_m > 0
        else 0.55
    )
    env_tol = (
        float(params.envelope_project_tol_m)
        if params.envelope_project_tol_m > 0
        else float(params.snapping_distance_m)
    )
    # Phase 3: corner reach matches BricsCAD-style ~0.6–1.0 m join when
    # the operator leaves snap/gap at defaults.
    corner_tol = (
        float(params.corner_close_tol_m)
        if params.corner_close_tol_m > 0
        else max(gap_tol, float(params.snapping_distance_m))
    )
    bridge_tol = (
        float(params.envelope_bridge_tol_m)
        if params.envelope_bridge_tol_m > 0
        else 1.50
    )
    stub_tol = (
        float(params.trim_stub_tol_m)
        if params.trim_stub_tol_m > 0
        else 0.50
    )

    env = envelope_xy if envelope_xy is not None else np.zeros((0, 2))

    segs, junctions, n_gaps = _close_degree1_gaps(snapped, junctions, gap_tol)
    segs, junctions, n_corners = _close_l_corners(segs, junctions, corner_tol)
    # Lateral near-miss: free ends within ~half snap distance share a corner
    # even when axes aren't perpendicular/collinear (Phase 3).
    lateral_tol = max(0.30, min(0.55, float(params.snapping_distance_m) * 0.55))
    segs, junctions, n_lateral = _snap_dangling_endpoints(segs, junctions, lateral_tol)
    # Second extend pass after L / lateral close.
    segs, junctions, n_ext2 = _extend_along_own_axis(
        segs, junctions, float(params.snapping_distance_m),
    )
    segs, junctions, n_env = _project_dangling_to_envelope(segs, junctions, env, env_tol)
    segs, junctions, n_bridge = _bridge_envelope_gaps(
        segs, junctions, env, bridge_tol,
    )
    segs, junctions, n_stubs = _trim_dangling_stubs(segs, junctions, stub_tol)

    n_gaps_total = n_gaps + n_corners + n_bridge + n_stubs + n_lateral
    _ = n_ext2

    # Re-snap so newly coincident ends (gap bridges / L-corners / hull feet)
    # collapse into shared junctions before the graph build.
    if n_gaps_total > 0 or n_env > 0 or n_ext2 > 0:
        re_tol = max(1e-4, min(0.05, gap_tol * 0.25))
        segs, junctions, _n_merged = _snap_endpoints(segs, re_tol)

    deg1 = _junction_degrees(segs, junctions)
    n_after = _count_degree1(deg1)
    return segs, junctions, n_gaps_total, n_env, n_before, n_after


# ── Step 3: graph build + room enumeration via planar-face traversal ─────────

def _build_edges(
    segs: np.ndarray,
    juncs: list[Junction],
) -> list[WallEdge]:
    """Map each segment to its two junctions; recompute junction degree."""
    edges: list[WallEdge] = []
    if not juncs:
        return edges
    juncs_arr = np.array([[j.x, j.y] for j in juncs])
    for k, s in enumerate(segs):
        a = int(np.argmin(np.sum((juncs_arr - s[0]) ** 2, axis=1)))
        b = int(np.argmin(np.sum((juncs_arr - s[1]) ** 2, axis=1)))
        if a == b:
            continue
        edges.append(WallEdge(i=a, j=b, source_segment_idx=k))
    for j in juncs:
        j.degree = 0
    for e in edges:
        juncs[e.i].degree += 1
        juncs[e.j].degree += 1
    return edges


def _planar_face_walks(
    juncs_arr: np.ndarray,
    edges: list[WallEdge],
) -> list[list[int]]:
    """Enumerate every face of the planar wall graph via angular half-edge walk.

    Standard combinatorial-map face traversal:

    1. Each undirected edge becomes two directed half-edges.
    2. At every vertex, outgoing half-edges are sorted by angle (CCW).
    3. The successor of half-edge ``u→v`` is ``v→w`` where ``w`` is the
       neighbour of ``v`` immediately CLOCKWISE of the reverse edge ``v→u``.
       Following successors until returning to the starting half-edge traces
       exactly one face boundary; every half-edge belongs to exactly one face.

    With this turn rule, bounded (interior) faces come out counter-clockwise
    (positive signed area) and each component's unbounded outer face comes
    out clockwise (negative signed area) — so orientation alone identifies
    outer faces exactly, with no bounding-box heuristics.

    Returns each face as an ordered closed vertex walk (first vertex NOT
    repeated at the end).  Bridge (dangling) edges appear twice in their
    face's walk — callers prune those backtrack spikes before building rings.
    """
    # Simple-graph adjacency (dedupe parallel edges and self-loops — both can
    # be produced by _build_edges' nearest-junction mapping).
    nbrs: dict[int, list[int]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for e in edges:
        if e.i == e.j:
            continue
        key = (min(e.i, e.j), max(e.i, e.j))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        nbrs.setdefault(e.i, []).append(e.j)
        nbrs.setdefault(e.j, []).append(e.i)

    # Sort each vertex's neighbours by outgoing angle (CCW order).
    for u, vs in nbrs.items():
        vs.sort(key=lambda v: np.arctan2(
            juncs_arr[v, 1] - juncs_arr[u, 1],
            juncs_arr[v, 0] - juncs_arr[u, 0],
        ))

    # Precompute neighbour → index maps for O(1) successor lookup.
    nbr_index: dict[int, dict[int, int]] = {
        u: {v: k for k, v in enumerate(vs)} for u, vs in nbrs.items()
    }

    faces: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for u in nbrs:
        for v in nbrs[u]:
            if (u, v) in visited:
                continue
            walk: list[int] = []
            cu, cv = u, v
            # Each half-edge is consumed exactly once, so this loop is
            # bounded by the total half-edge count.
            while (cu, cv) not in visited:
                visited.add((cu, cv))
                walk.append(cu)
                lst = nbrs[cv]
                idx = nbr_index[cv][cu]
                # Next outgoing edge clockwise of the reverse edge v→u.
                w = lst[(idx - 1) % len(lst)]
                cu, cv = cv, w
            faces.append(walk)
    return faces


def _prune_backtracks(walk: list[int]) -> list[int]:
    """Remove spur excursions (``… a b a …``) from a closed face walk.

    A dangling wall (bridge edge) is traversed out-and-back by the face walk
    around the room that contains it.  The shoelace contributions cancel, but
    the ring would be non-simple — so we iteratively delete every spike tip
    until the walk is backtrack-free.  Treats the walk as cyclic.
    """
    w = list(walk)
    changed = True
    while changed and len(w) >= 3:
        changed = False
        n = len(w)
        for k in range(n):
            if w[(k - 1) % n] == w[(k + 1) % n]:
                # w[k] is a spike tip: drop it and the duplicated return vertex.
                for d in sorted((k, (k + 1) % n), reverse=True):
                    del w[d]
                changed = True
                break
    # Collapse any consecutive duplicates left over (degenerate input).
    out: list[int] = []
    for v in w:
        if not out or out[-1] != v:
            out.append(v)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _enumerate_rooms(
    juncs: list[Junction],
    edges: list[WallEdge],
    params: TopologyParams,
) -> list[RoomFace]:
    """Find rooms as the bounded faces of the planar wall graph.

    v5 rewrite: the previous implementation fed ``nx.minimum_cycle_basis``
    node lists straight into the shoelace formula.  That function's contract
    explicitly does NOT guarantee ring order ("nodes are not necessarily
    returned in the order in which they appear in the cycle") and its cycles
    are a *basis*, not faces — rooms containing dangling partition spurs
    came back with doubled areas.  Planar-face traversal (angular half-edge
    walk) enumerates every face exactly once, in ring order, and lets
    orientation identify the outer face exactly.  Rings are validated with
    Shapely before being reported.
    """
    if not edges:
        return []
    from shapely.geometry import Polygon as _ShapelyPolygon

    juncs_arr = np.array([[j.x, j.y] for j in juncs])

    candidate_rooms: list[tuple[list[int], np.ndarray, float, float]] = []
    for walk in _planar_face_walks(juncs_arr, edges):
        ring = _prune_backtracks(walk)
        if len(ring) < 3:
            continue  # pure tree component (no enclosed area)
        poly = juncs_arr[ring]
        signed = _signed_area(poly)
        if params.drop_outer_face and signed < 0:
            # Clockwise ⇒ the unbounded outer face of a component.
            continue
        area = abs(signed)
        # Validate the ring with Shapely.  A face walk through a cut vertex
        # (two rooms touching at a single corner point) produces a non-simple
        # ring; buffer(0) splits/repairs it when possible.
        shp = _ShapelyPolygon(poly)
        was_invalid = not shp.is_valid
        if was_invalid:
            repaired = shp.buffer(0)
            if repaired.is_empty:
                continue
            if repaired.geom_type == "MultiPolygon":
                repaired = max(repaired.geoms, key=lambda g: g.area)
            if repaired.geom_type != "Polygon":
                continue
            # Drop face walks whose repair massively changes area — those
            # are self-crossing orange artefacts, not cut-vertex rooms.
            if abs(float(repaired.area) - abs(signed)) > max(0.5, 0.25 * abs(signed)):
                continue
            ext = np.asarray(repaired.exterior.coords)[:-1]
            poly = ext
            ring = [int(np.argmin(np.sum((juncs_arr - p) ** 2, axis=1))) for p in ext]
            area = float(repaired.area)
            shp = _ShapelyPolygon(poly)
        perim = _polygon_perimeter(poly)
        # Reject spiky / needle faces (the orange "spike" artefact): a valid
        # room has a reasonable isoperimetric quotient 4πA/P².  Circles ≈ 1;
        # rectangles ≈ 0.5–0.8; long spikes fall well below the threshold.
        if perim > 1e-6:
            iq = float(4.0 * np.pi * area / (perim * perim))
            if iq < float(params.min_room_isoperimetric):
                continue
        if not shp.is_valid:
            continue
        candidate_rooms.append((ring, poly, area, perim))

    if not candidate_rooms:
        return []

    # Cap the maximum room area at 50% of the global bbox area when it dwarfs
    # every other candidate.  A bounded face that spans nearly the whole
    # building alongside many small faces almost always means a dividing wall
    # has a gap and several real rooms fused into one wrapper face.  (An open
    # floor plan with one genuine big space still passes: the second-biggest
    # test keeps it.)
    outer_indices: set[int] = set()
    all_pts = np.concatenate([r[1] for r in candidate_rooms], axis=0)
    global_area = float(
        (all_pts[:, 0].max() - all_pts[:, 0].min()) *
        (all_pts[:, 1].max() - all_pts[:, 1].min())
    )
    sorted_by_area = sorted(enumerate(candidate_rooms), key=lambda x: -x[1][2])
    if len(sorted_by_area) >= 2:
        biggest_area = sorted_by_area[0][1][2]
        second_biggest_area = sorted_by_area[1][1][2]
        # Only treat the big face as a wrapper when the second face is a
        # plausible room in its own right.  If it's a sub-min-area sliver
        # (a corner artefact from a spur wall), the "building" is really
        # one genuine big room and dropping it would leave zero rooms.
        if (second_biggest_area >= params.min_room_area_m2
                and biggest_area > 0.50 * global_area
                and biggest_area > 3.0 * second_biggest_area):
            outer_indices.add(sorted_by_area[0][0])

    rooms: list[RoomFace] = []
    for idx, (vlist, poly, area, perim) in enumerate(candidate_rooms):
        if idx in outer_indices:
            continue
        if area < params.min_room_area_m2:
            continue
        rooms.append(RoomFace(
            vertex_ids=list(vlist),
            polygon=poly,
            area_m2=float(area),
            perimeter_m=float(perim),
        ))

    rooms.sort(key=lambda r: -r.area_m2)
    return rooms


# ── Public API ────────────────────────────────────────────────────────────────

def _envelope_ring_segments(
    envelope_xy: np.ndarray,
    min_edge_m: float = 0.05,
) -> np.ndarray:
    """Convert an envelope ring into (N, 2, 2) edge segments for topology."""
    ring = np.asarray(envelope_xy, dtype=float).reshape(-1, 2)
    if len(ring) < 3:
        return np.zeros((0, 2, 2), dtype=float)
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    edges = np.array(
        [[ring[i], ring[i + 1]] for i in range(len(ring) - 1)],
        dtype=float,
    )
    if len(edges) == 0:
        return edges
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    return edges[lengths >= min_edge_m]


def _novel_envelope_edges(
    env_segs: np.ndarray,
    finished: np.ndarray,
    cover_tol_m: float = 0.20,
) -> np.ndarray:
    """Keep only envelope edges not already covered by a finished wall.

    Injecting the full hull on top of existing exterior walls creates
    parallel duplicate edges that collapse planar-face walks to zero area.
    An edge is "covered" when its midpoint lies within ``cover_tol_m`` of
    some finished segment (point-to-segment distance).
    """
    if len(env_segs) == 0:
        return env_segs
    if len(finished) == 0:
        return env_segs

    kept: list[np.ndarray] = []
    for e in env_segs:
        mid = 0.5 * (e[0] + e[1])
        covered = False
        for w in finished:
            a, b = w[0], w[1]
            ab = b - a
            L2 = float(np.dot(ab, ab))
            if L2 < 1e-12:
                dist = float(np.linalg.norm(mid - a))
            else:
                t = float(np.dot(mid - a, ab) / L2)
                t = max(0.0, min(1.0, t))
                proj = a + t * ab
                dist = float(np.linalg.norm(mid - proj))
            if dist <= cover_tol_m:
                covered = True
                break
        if not covered:
            kept.append(e)
    if not kept:
        return np.zeros((0, 2, 2), dtype=float)
    return np.asarray(kept, dtype=float)


def _resolve_snap_tol(params: TopologyParams) -> float:
    """Auto-tune snap tolerance from wall thickness if caller didn't override."""
    if params.snap_tol_m > 0:
        return float(params.snap_tol_m)
    # snap_tol must stay below wall thickness (else faces fuse).  Real
    # scanners undershoot corners by 5–15 cm; 0.9 × thickness (~11 cm at
    # 12 cm walls) closes T/L junctions without fusing faces.  Cap 0.18 m.
    thick = params.wall_thickness_median_m
    if thick > 0:
        return min(0.18, max(0.10, 0.9 * thick))
    return 0.15


def _finish_wall_axes(
    centerlines: np.ndarray,
    params: TopologyParams,
    envelope_xy: Optional[np.ndarray],
    *,
    apply_continuity: bool,
) -> tuple[np.ndarray, list[Junction], int, int, int, int, int, int]:
    """Cloud2BIM § 2.8: split crossings, snap, T-split, axis-extend, continuity.

    Returns
    -------
    snapped, junctions, n_merged, n_extended, n_gaps, n_env,
    n_dangling_before, n_dangling_after_pre_graph
    """
    snap_tol = _resolve_snap_tol(params)
    segs = np.asarray(centerlines, dtype=float).reshape(-1, 2, 2)
    # Phase 3: cardinal snap before join so L/T hits land cleanly.
    if params.manhattan_join_tol_deg > 0:
        segs = _manhattan_snap_segments(segs, float(params.manhattan_join_tol_deg))
    segs = _split_at_crossings(segs)
    snapped, junctions, n_merged = _snap_endpoints(segs, snap_tol)
    snapped, junctions, n_split = _split_hosts_at_dangling(
        snapped, junctions, snap_tol=max(snap_tol, params.snapping_distance_m),
    )
    snapped, junctions, n_extended = _extend_along_own_axis(
        snapped, junctions, params.snapping_distance_m,
    )
    n_extended += n_split

    n_gaps = n_env = 0
    deg0 = _junction_degrees(snapped, junctions)
    n_dang_before = _count_degree1(deg0)
    n_dang_after = n_dang_before
    if apply_continuity:
        snapped, junctions, n_gaps, n_env, n_dang_before, n_dang_after = (
            _apply_continuity(snapped, junctions, params, envelope_xy)
        )
    return (
        snapped, junctions, n_merged, n_extended,
        n_gaps, n_env, n_dang_before, n_dang_after,
    )


def build_topology(
    centerlines: np.ndarray,
    params: TopologyParams = TopologyParams(),
    envelope_xy: Optional[np.ndarray] = None,
) -> TopologyResult:
    """Snap, axis-extend, close residual gaps, and find rooms.

    Cloud2BIM-style two-pass (paper § 2.8 + identify_zones):

    1. **Finish wall axes** on observed centerlines only — this is what the
       editor / DXF must show (continuous CAD walls, not detector fragments).
    2. **Close rooms** by optionally injecting the envelope ring into a
       second graph pass.  Virtual envelope edges never enter
       ``finished_wall_segments``.

    Parameters
    ----------
    centerlines : (N, 2, 2) array of wall centerlines (post-pairing).
    params : :class:`TopologyParams`.
    envelope_xy : optional (M, 2) building-envelope ring (world metres).
    """
    if centerlines is None or len(centerlines) == 0:
        return TopologyResult(
            junctions=[], edges=[], snapped_segments=np.zeros((0, 2, 2)),
            finished_wall_segments=np.zeros((0, 2, 2)),
        )

    observed = np.asarray(centerlines, dtype=float).reshape(-1, 2, 2)

    # Pass 1 — finish observed walls for export (no envelope injection).
    (
        finished, fin_juncs, n_merged, n_extended,
        n_gaps, n_env, n_dang_before, _n_dang_mid,
    ) = _finish_wall_axes(
        observed, params, envelope_xy, apply_continuity=True,
    )
    # Drop zero-length / near-degenerate pieces from export.
    if len(finished) > 0:
        lens = np.linalg.norm(finished[:, 1] - finished[:, 0], axis=1)
        finished_export = finished[lens >= 1e-4]
    else:
        finished_export = finished

    # Pass 2 — room closing may inject the envelope as virtual edges.
    n_env_injected = 0
    if (
        params.inject_envelope
        and envelope_xy is not None
        and len(envelope_xy) >= 3
    ):
        env_segs = _envelope_ring_segments(envelope_xy)
        env_segs = _novel_envelope_edges(env_segs, finished_export)
        if len(env_segs) > 0:
            n_env_injected = int(len(env_segs))
            # Light join only: split crossings + snap + T-extend so free
            # wall ends meet the hull.  Do NOT re-run gap-close (that would
            # invent chords across the envelope).
            room_input = np.vstack([finished_export, env_segs])
            snap_tol = _resolve_snap_tol(params)
            room_input = _split_at_crossings(room_input)
            snapped, junctions, n_merged2 = _snap_endpoints(room_input, snap_tol)
            n_merged += n_merged2
            snapped, junctions, n_split2 = _split_hosts_at_dangling(
                snapped, junctions,
                snap_tol=max(snap_tol, params.snapping_distance_m),
            )
            snapped, junctions, n_ext2 = _extend_along_own_axis(
                snapped, junctions, params.snapping_distance_m,
            )
            n_extended += n_split2 + n_ext2
        else:
            snapped, junctions = finished, fin_juncs
    else:
        snapped, junctions = finished, fin_juncs

    edges = _build_edges(snapped, junctions)
    rooms = _enumerate_rooms(junctions, edges, params)

    # CAD dangling count is on the finished export graph (what the editor /
    # DXF show), not the room pass which may include virtual envelope edges.
    fin_deg_juncs = [Junction(x=j.x, y=j.y) for j in fin_juncs]
    _build_edges(finished_export, fin_deg_juncs)
    n_dangling_finished = sum(1 for j in fin_deg_juncs if int(j.degree) == 1)

    return TopologyResult(
        junctions=junctions,
        edges=edges,
        snapped_segments=snapped,
        rooms=rooms,
        n_endpoints_merged=n_merged,
        n_extended=n_extended,
        n_gaps_closed=n_gaps,
        n_envelope_projections=n_env,
        n_dangling_before=n_dang_before,
        n_dangling_after=n_dangling_finished,
        n_envelope_edges_injected=n_env_injected,
        finished_wall_segments=finished_export,
    )


def rooms_to_segments(rooms: list[RoomFace]) -> np.ndarray:
    """Flatten rooms into a (M, 2, 2) array of polyline edges (for DXF)."""
    out: list[list[list[float]]] = []
    for r in rooms:
        n = len(r.polygon)
        for k in range(n):
            a = r.polygon[k]
            b = r.polygon[(k + 1) % n]
            out.append([[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]])
    if not out:
        return np.zeros((0, 2, 2), dtype=float)
    return np.asarray(out, dtype=float)


def dangling_endpoints(result: TopologyResult) -> list[dict[str, float | int]]:
    """Degree-1 junctions on finished walls — open ends the operator should close.

    Uses ``finished_wall_segments`` (editor / DXF graph), not the room pass
    which may include virtual envelope edges.  Returned as JSON-serialisable
    dicts for the editor highlight layer and ``RoomsNotClosedError`` 422.
    """
    segs = result.finished_wall_segments
    if segs is not None and len(segs) > 0:
        snapped, juncs, _ = _snap_endpoints(
            np.asarray(segs, dtype=float).reshape(-1, 2, 2), 1e-4,
        )
        _build_edges(snapped, juncs)
        source = juncs
    else:
        source = result.junctions

    out: list[dict[str, float | int]] = []
    for j in source:
        if int(j.degree) == 1:
            out.append({
                "x": round(float(j.x), 4),
                "y": round(float(j.y), 4),
                "degree": 1,
            })
    return out


def write_dangling_endpoints(result_dir: Path, endpoints: list[dict[str, float | int]]) -> Path:
    """Persist dangling ends next to segments so the editor can load them."""
    path = Path(result_dir) / "dangling_endpoints.json"
    path.write_text(json.dumps({
        "version": 1,
        "units": "metres",
        "endpoints": endpoints,
    }, indent=2))
    return path


def load_dangling_endpoints(result_dir: Path) -> list[dict[str, float | int]]:
    """Load previously persisted dangling ends; empty list if missing."""
    path = Path(result_dir) / "dangling_endpoints.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        eps = data.get("endpoints") if isinstance(data, dict) else data
        if not isinstance(eps, list):
            return []
        out: list[dict[str, float | int]] = []
        for e in eps:
            if not isinstance(e, dict):
                continue
            out.append({
                "x": float(e["x"]),
                "y": float(e["y"]),
                "degree": int(e.get("degree", 1)),
            })
        return out
    except Exception:
        return []


def topology_summary(result: TopologyResult) -> str:
    """One-line human-readable summary for SSE progress."""
    cont = ""
    if result.n_gaps_closed or result.n_envelope_projections:
        cont = (
            f", {result.n_gaps_closed} gaps closed, "
            f"{result.n_envelope_projections} envelope snaps"
        )
    inj = ""
    if result.n_envelope_edges_injected:
        inj = f", {result.n_envelope_edges_injected} envelope edges"
    dang = ""
    if result.n_dangling_before or result.n_dangling_after:
        dang = (
            f" · dangling {result.n_dangling_before}→{result.n_dangling_after}"
        )
    return (
        f"topology: {len(result.junctions)} junctions "
        f"({result.n_endpoints_merged} endpoints merged, "
        f"{result.n_extended} axis-extensions{cont}{inj}) · "
        f"{len(result.rooms)} rooms inferred{dang}"
    )
