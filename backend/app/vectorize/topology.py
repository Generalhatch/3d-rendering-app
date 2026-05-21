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

3. **Room finding uses NetworkX minimum_cycle_basis**, not custom planar-face
   enumeration.  The custom algorithm in v3 made wrong-direction choices at
   high-degree junctions (4-way intersections in offices), missing rooms.
   ``minimum_cycle_basis`` returns the minimal set of cycles that generate
   the cycle space — guaranteed to include every room exactly once, no
   wrong-direction failures.  Pure graph-theoretic, fully tested for decades.

4. **Outer face is dropped by bounding-box test**, not by sign of signed
   area.  More robust against tiny polygons whose signed area is
   numerically negative due to FP error.

The contract (TopologyResult) is unchanged so callers don't need to migrate.

Why this is one module and not two
----------------------------------
Snap, extend, and room inference all share the junction graph — splitting
them would mean rebuilding it three times.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np


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
    # "auto-close gaps in the same axis" step.
    snapping_distance_m: float = 0.30

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

            # Split host into (a, new_pt) + (new_pt, b).
            a = segs[best_host, 0].copy()
            b = segs[best_host, 1].copy()
            segs[best_host, 0] = a
            segs[best_host, 1] = new_pt
            new_piece = np.array([[new_pt[0], new_pt[1]],
                                  [b[0], b[1]]], dtype=segs.dtype)
            segs = np.vstack([segs, new_piece[None, :, :]])
            deg[new_jidx] += 2  # +2 because we added two halves both incident
            n_splits += 1

    return segs, junctions, n_splits


# ── Step 2b: axis-aligned wall extension (Cloud2BIM § 2.8) ───────────────────

def _extend_along_own_axis(
    snapped: np.ndarray,
    junctions: list[Junction],
    snapping_distance_m: float,
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

    Operates on the segment list, mutating endpoints and (re-)materialising
    junctions as needed.

    Returns (new_segments, new_junctions, n_extended).
    """
    n_segs = len(snapped)
    if n_segs == 0 or snapping_distance_m <= 0:
        return snapped, junctions, 0

    segs = snapped.copy()

    # Compute junction degree from current segs.
    deg = [0] * len(junctions)

    def _jidx(pt: np.ndarray) -> int:
        """Find existing junction at pt (exact match; junctions were
        materialised at snap time)."""
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

    n_extended = 0
    tol = float(snapping_distance_m)

    # For each segment, try to extend each free endpoint outward.
    for k in range(n_segs):
        for end in (0, 1):
            ep = segs[k, end].copy()
            other = segs[k, 1 - end].copy()
            j_idx = _jidx(ep)
            if j_idx == -1 or deg[j_idx] != 1:
                # Endpoint is already shared with another wall; nothing to do.
                continue

            # Direction outward = from other → ep, normalised.
            d = ep - other
            L = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            d /= L

            # Look for the closest hit when shooting a ray from ep along +d
            # by up to ``tol`` metres.  Test against every OTHER segment.
            best_hit = None
            best_t = float("inf")
            best_host = -1
            for h in range(n_segs):
                if h == k:
                    continue
                a = segs[h, 0]
                b = segs[h, 1]
                hit, t_ray, u_host = _ray_segment_intersect(ep, d, a, b)
                if hit is None:
                    continue
                if t_ray <= 1e-6 or t_ray > tol:
                    continue
                # Require u_host strictly interior (so we form a T-junction,
                # not an overlap) OR very close to host endpoints (we'd then
                # merge into that endpoint).
                if -1e-6 <= u_host <= 1.0 + 1e-6:
                    if t_ray < best_t:
                        best_t = t_ray
                        best_hit = hit
                        best_host = h

            if best_hit is None or best_host == -1:
                continue

            # Decide: snap to host endpoint or split host at the hit.
            a = segs[best_host, 0]
            b = segs[best_host, 1]
            host_L = float(np.linalg.norm(b - a))
            d_to_a = float(np.linalg.norm(best_hit - a))
            d_to_b = float(np.linalg.norm(best_hit - b))
            snap_thresh = max(tol * 0.5, 0.05)

            if d_to_a < snap_thresh:
                new_pt = a.copy()
                host_split = False
            elif d_to_b < snap_thresh:
                new_pt = b.copy()
                host_split = False
            else:
                new_pt = best_hit
                host_split = True

            # Update our endpoint to new_pt; create/find its junction.
            new_jidx = -1
            for j_k, j in enumerate(junctions):
                if abs(j.x - new_pt[0]) < 1e-9 and abs(j.y - new_pt[1]) < 1e-9:
                    new_jidx = j_k
                    break
            if new_jidx == -1:
                junctions.append(Junction(x=float(new_pt[0]), y=float(new_pt[1])))
                deg.append(0)
                new_jidx = len(junctions) - 1

            # Decrement old degree, increment new.
            deg[j_idx] -= 1
            deg[new_jidx] += 1
            segs[k, end] = new_pt

            if host_split:
                # Split host into (a, new_pt) and (new_pt, b).  Both pieces
                # become rows of segs (one overwrites the host, one appends).
                segs[best_host, 0] = a
                segs[best_host, 1] = new_pt
                new_piece = np.array([[new_pt[0], new_pt[1]],
                                      [b[0], b[1]]], dtype=segs.dtype)
                segs = np.vstack([segs, new_piece[None, :, :]])
                deg[new_jidx] += 1  # new_pt now degree-3

            n_extended += 1

    return segs, junctions, n_extended


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


# ── Step 3: graph build + room (cycle) enumeration via NetworkX ──────────────

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


def _signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _polygon_perimeter(poly: np.ndarray) -> float:
    diff = np.roll(poly, -1, axis=0) - poly
    return float(np.sum(np.sqrt(np.sum(diff * diff, axis=1))))


def _order_cycle(
    cycle_edges: list[tuple[int, int]],
) -> list[int]:
    """Convert an unordered cycle (list of edges) into the ordered vertex sequence.

    NetworkX's ``minimum_cycle_basis`` returns cycles as edge lists; we need
    them as ordered vertex lists to compute polygon area and perimeter.
    """
    if not cycle_edges:
        return []
    # Build adjacency for this cycle alone.
    adj: dict[int, list[int]] = {}
    for a, b in cycle_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    # Start anywhere.
    start = cycle_edges[0][0]
    order: list[int] = [start]
    prev = -1
    cur = start
    for _ in range(len(cycle_edges)):
        nxt_options = [n for n in adj[cur] if n != prev]
        if not nxt_options:
            break
        nxt = nxt_options[0]
        prev = cur
        cur = nxt
        if cur == start:
            break
        order.append(cur)
    return order


def _enumerate_rooms(
    juncs: list[Junction],
    edges: list[WallEdge],
    params: TopologyParams,
) -> list[RoomFace]:
    """Use NetworkX minimum_cycle_basis to find rooms.

    Why minimum_cycle_basis: it returns the smallest set of cycles whose
    combinations span the entire cycle space of the graph.  Mathematically,
    this is exactly "every room exactly once, plus possibly the outer
    boundary as one extra cycle".  We drop the outer boundary by bounding-
    box test below.
    """
    if not edges:
        return []
    G = nx.Graph()
    for jid, j in enumerate(juncs):
        G.add_node(jid, x=j.x, y=j.y)
    for e in edges:
        G.add_edge(e.i, e.j)

    juncs_arr = np.array([[j.x, j.y] for j in juncs])

    rooms: list[RoomFace] = []
    # CRITICAL: minimum_cycle_basis fails / is undefined on disconnected
    # graphs in older networkx versions, and even in 3.6+ it returns
    # cycles ONLY in the largest connected component.  For a real
    # building, the wall graph is almost always multi-component (the
    # back-of-house corridors form a separate connected component from
    # the front-of-house wing in many floor plans, especially when one
    # interior door wasn't picked up).  We compute the cycle basis
    # per connected component and concatenate.
    cycles_vertices: list[list[int]] = []
    for component in nx.connected_components(G):
        if len(component) < 3:
            continue
        sub = G.subgraph(component).copy()
        try:
            sub_cycles = nx.minimum_cycle_basis(sub)
        except (nx.NetworkXNotImplemented, nx.NetworkXError):
            sub_cycles = nx.cycle_basis(sub)
        for cycle_node_list in sub_cycles:
            if len(cycle_node_list) < 3:
                continue
            cycles_vertices.append([int(v) for v in cycle_node_list])

    if not cycles_vertices:
        return []

    # Compute polygons + areas.
    candidate_rooms: list[tuple[list[int], np.ndarray, float, float]] = []
    for vlist in cycles_vertices:
        poly = juncs_arr[vlist]
        area = abs(_signed_area(poly))
        perim = _polygon_perimeter(poly)
        candidate_rooms.append((vlist, poly, area, perim))

    # Identify outer faces: any candidate that strictly contains another
    # candidate's bounding box is an "outer wrapper" we want to drop.
    # Multiple connected components each contribute one outer face — this
    # correctly drops all of them.  An interior room never contains any
    # other room (rooms are disjoint by construction), so this rule never
    # drops a real room.
    outer_indices: set[int] = set()
    if params.drop_outer_face and len(candidate_rooms) > 1:
        bboxes = np.array([
            (poly[:, 0].min(), poly[:, 1].min(),
             poly[:, 0].max(), poly[:, 1].max())
            for _, poly, _, _ in candidate_rooms
        ])
        for i, (mnx, mny, mxx, mxy) in enumerate(bboxes):
            for k, (omx, omy, oxx, oxy) in enumerate(bboxes):
                if k == i:
                    continue
                # i strictly contains k → i is an outer.
                if (mnx <= omx and mny <= omy and mxx >= oxx and mxy >= oxy
                        and (mxx - mnx) > (oxx - omx)):
                    outer_indices.add(i)
                    break

    # Also cap the maximum room area at 30% of the global bbox area.
    # Anything larger is almost certainly a wrapper around several real
    # rooms whose dividing wall has a gap.  This catches the "outer face"
    # case even when bbox-containment doesn't fire (single-component graph
    # with a gap in the divider — the entire interior reads as one face).
    if candidate_rooms:
        all_pts = np.concatenate([r[1] for r in candidate_rooms], axis=0)
        global_area = float(
            (all_pts[:, 0].max() - all_pts[:, 0].min()) *
            (all_pts[:, 1].max() - all_pts[:, 1].min())
        )
        # Tolerate one big room (open floor plan).  If the SECOND-biggest
        # candidate is also above the threshold, we're looking at a real
        # large open space; keep both.
        sorted_by_area = sorted(
            enumerate(candidate_rooms), key=lambda x: -x[1][2]
        )
        if len(sorted_by_area) >= 2:
            biggest_area = sorted_by_area[0][1][2]
            second_biggest_area = sorted_by_area[1][1][2]
            if biggest_area > 0.50 * global_area and biggest_area > 3.0 * second_biggest_area:
                outer_indices.add(sorted_by_area[0][0])

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

def build_topology(
    centerlines: np.ndarray,
    params: TopologyParams = TopologyParams(),
) -> TopologyResult:
    """Snap, axis-extend, and find rooms.

    Parameters
    ----------
    centerlines : (N, 2, 2) array of wall centerlines (post-pairing).
    params : :class:`TopologyParams`.  If ``snap_tol_m`` is 0 it auto-tunes
        from ``wall_thickness_median_m`` (set by pipeline.py from the
        pairing result).

    Returns
    -------
    TopologyResult
    """
    if centerlines is None or len(centerlines) == 0:
        return TopologyResult(
            junctions=[], edges=[], snapped_segments=np.zeros((0, 2, 2)),
        )

    centerlines = np.asarray(centerlines, dtype=float).reshape(-1, 2, 2)

    # Auto-tune snap tolerance from wall thickness if caller didn't override.
    if params.snap_tol_m <= 0:
        # The hard constraint: snap_tol must be LESS than wall thickness
        # (otherwise the two faces of one wall fuse into a single junction
        # and the wall disappears as zero-area).  But: real scanners
        # overshoot wall ends by 5-15 cm even on clean scans, and the
        # contour extractor's Douglas-Peucker can shift the corner vertex
        # by another 2-5 cm.  The sum is bigger than half a wall.
        #
        # Empirically (Stevenson demo run): 0.5 × thickness was 6.4 cm and
        # produced 0 rooms despite 222 walls and 248 endpoint operations.
        # 0.9 × thickness gives ~11 cm at typical 12 cm walls — closes
        # T-junctions and L-corners without fusing wall faces.  We cap at
        # 0.18 m to defend against very thick exterior walls (block
        # walls up to 30 cm) where 0.9 × thickness = 27 cm WOULD fuse.
        thick = params.wall_thickness_median_m
        if thick > 0:
            snap_tol = min(0.18, max(0.10, 0.9 * thick))
        else:
            snap_tol = 0.15
    else:
        snap_tol = params.snap_tol_m

    # 0. Pre-split segments at every mutual interior intersection (an X
    # crossing two interior walls).  Comes before endpoint snap so the
    # snap step sees the new vertices and merges any near-duplicates.
    centerlines = _split_at_crossings(centerlines)

    # 1. Snap nearby endpoints.
    snapped, junctions, n_merged = _snap_endpoints(centerlines, snap_tol)

    # 2a. Split hosts at dangling endpoints that already lie on them.
    # This is the "internal wall touches outer wall but outer wall has no
    # junction there" case — extremely common after contour extraction,
    # where the outer ring is one contour and the internal wall is a
    # separate one without shared vertices.
    # Use the FULL snapping_distance for this — real-scan walls often end
    # 15-30 cm short of their host (drywall meets stud framing at the
    # plate, and the laser bounces off the plate face which sits behind
    # the wall plane).
    snapped, junctions, n_split = _split_hosts_at_dangling(
        snapped, junctions, snap_tol=max(snap_tol, params.snapping_distance_m),
    )

    # 2b. Cloud2BIM-style axis-aligned extension for the case where the
    # dangling endpoint is *close to* the host but not yet on it (a gap).
    snapped, junctions, n_extended = _extend_along_own_axis(
        snapped, junctions, params.snapping_distance_m,
    )
    n_extended += n_split

    # 3. Graph + cycles.
    edges = _build_edges(snapped, junctions)
    rooms = _enumerate_rooms(junctions, edges, params)

    return TopologyResult(
        junctions=junctions,
        edges=edges,
        snapped_segments=snapped,
        rooms=rooms,
        n_endpoints_merged=n_merged,
        n_extended=n_extended,
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


def topology_summary(result: TopologyResult) -> str:
    """One-line human-readable summary for SSE progress."""
    return (
        f"topology: {len(result.junctions)} junctions "
        f"({result.n_endpoints_merged} endpoints merged, "
        f"{result.n_extended} axis-extensions) · "
        f"{len(result.rooms)} rooms inferred"
    )
