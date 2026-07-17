"""Adapters: pipeline-specific outputs → canonical :class:`FloorGeometry`.

Two producers exist today:

- The **vectorize pipeline** (``app.vectorize``): wall centerlines with
  thickness from :func:`app.vectorize.walls.pair_walls` plus rooms from
  :func:`app.vectorize.topology.build_topology`.  Room rings are drawn on
  wall CENTERLINES.
- The **alignment pipeline** (``app.pipeline``): rooms as interior-face
  occupancy polygons (watershed / DXF extraction) with labels and
  categories but no explicit walls.  Walls are synthesized here.

Both emit the same FloorGeometry so the measurement engine has exactly one
input shape.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from .model import (
    BoundaryBasis,
    ColumnFeature,
    FloorGeometry,
    Opening,
    Room,
    Wall,
    WallClass,
    WallFaceRef,
    ring_signed_area,
)

# Re-exported for backwards compatibility; the definition lives in
# app.geometry.classify so both pipelines share one common-area notion.
from .classify import COMMON_CATEGORIES, classify_room_polygon  # noqa: E402


# ── Vectorize pipeline → FloorGeometry ───────────────────────────────────────

def floor_geometry_from_vectorize(
    topology_result: Any,                    # app.vectorize.topology.TopologyResult
    wall_pairing: Any = None,                # app.vectorize.walls.WallPairingResult
    envelope_xy: Optional[np.ndarray] = None,
    openings: Optional[Sequence[Any]] = None,   # DetectedOpening
    columns: Optional[Sequence[Any]] = None,    # DetectedColumn
    floor_id: str = "floor",
    edge_match_tol_m: float = 0.40,
    edge_match_angle_deg: float = 15.0,
) -> FloorGeometry:
    """Build FloorGeometry from TopologyResult + WallPairingResult.

    - Walls come from the pairing stage (centerline + faces + thickness);
      when no pairing is available they are synthesized from the topology's
      snapped centerlines at the fallback thickness.
    - Each room-ring edge is attributed to the nearest parallel wall
      (perpendicular distance ≤ ``edge_match_tol_m``, angle within
      ``edge_match_angle_deg`` — tolerances sized for the topology stage's
      endpoint snapping, which moves segments by up to ~snap_tol).
    - Wall classification:
        rooms on BOTH sides           → demising
        else near envelope / on the union boundary of all rooms → exterior
        else                          → partition
    """
    # 1. Wall entities.
    walls: list[Wall] = []
    if wall_pairing is not None and getattr(wall_pairing, "walls", None):
        for k, vw in enumerate(wall_pairing.walls):
            walls.append(Wall(
                id=f"vw-{k:04d}",
                centerline=np.asarray(vw.centerline, dtype=float),
                thickness_m=float(vw.thickness_m),
                face_a=np.asarray(vw.face_a, dtype=float),
                face_b=np.asarray(vw.face_b, dtype=float),
                is_paired=bool(vw.is_paired),
                source="vectorize.pair_walls",
            ))
    else:
        fallback_t = 0.10
        for k, seg in enumerate(np.asarray(topology_result.snapped_segments)):
            walls.append(Wall(
                id=f"vw-{k:04d}",
                centerline=np.asarray(seg, dtype=float),
                thickness_m=fallback_t,
                source="vectorize.topology",
            ))

    wall_dirs: list[np.ndarray] = []
    wall_lens: list[float] = []
    for w in walls:
        d = w.centerline[1] - w.centerline[0]
        length = float(np.linalg.norm(d))
        wall_dirs.append(d / length if length > 1e-12 else np.array([1.0, 0.0]))
        wall_lens.append(length)

    # 2. Rooms: attribute each ring edge to the nearest parallel wall,
    #    recording which side of that wall the room interior is on.
    rooms: list[Room] = []
    #   wall index → set of (room_id, side_sign)
    wall_room_sides: dict[int, set[tuple[str, int]]] = {}

    cos_tol = np.cos(np.radians(edge_match_angle_deg))
    for r_idx, face in enumerate(topology_result.rooms):
        ring = np.asarray(face.polygon, dtype=float)
        if ring_signed_area(ring) < 0:
            ring = ring[::-1].copy()
        room_id = f"room-{r_idx + 1:03d}"
        refs: list[WallFaceRef] = []
        k_edges = len(ring)
        for k in range(k_edges):
            p = ring[k]
            q = ring[(k + 1) % k_edges]
            mid = (p + q) / 2.0
            d = q - p
            elen = float(np.linalg.norm(d))
            if elen < 1e-12:
                refs.append(WallFaceRef(None))
                continue
            d = d / elen

            best_w = -1
            best_dist = edge_match_tol_m
            for wi, w in enumerate(walls):
                if abs(float(np.dot(d, wall_dirs[wi]))) < cos_tol:
                    continue
                a = w.centerline[0]
                u = wall_dirs[wi]
                rel = mid - a
                t_along = float(np.dot(rel, u))
                if t_along < -edge_match_tol_m or t_along > wall_lens[wi] + edge_match_tol_m:
                    continue
                perp = abs(float(rel[0] * u[1] - rel[1] * u[0]))
                if perp < best_dist:
                    best_dist = perp
                    best_w = wi

            if best_w == -1:
                refs.append(WallFaceRef(None))
                continue

            # Which side of the wall is the room interior on?  Probe a point
            # just inside the room (inward normal of a CCW ring edge).
            w = walls[best_w]
            u = wall_dirs[best_w]
            inward = np.array([-d[1], d[0]])
            probe = mid + inward * 0.05
            side = float(
                (probe[0] - w.centerline[0][0]) * u[1]
                - (probe[1] - w.centerline[0][1]) * u[0]
            )
            side_sign = 1 if side >= 0 else -1
            wall_room_sides.setdefault(best_w, set()).add((room_id, side_sign))
            refs.append(WallFaceRef(walls[best_w].id, side="a" if side_sign > 0 else "b"))

        # Shape-metric classification (Phase 2): corridors and restrooms
        # become common areas so the measurement engine can apportion them.
        clf_label, clf_category, _clf_conf, clf_common = classify_room_polygon(ring)
        rooms.append(Room(
            id=room_id,
            boundary=ring,
            wall_refs=refs,
            boundary_basis=BoundaryBasis.CENTERLINE,
            label=f"Room {r_idx + 1} ({clf_label})",
            category=clf_category,
            is_common=clf_common,
        ))

    # 3. Classify walls.
    envelope_ring = None
    if envelope_xy is not None and len(envelope_xy) >= 3:
        envelope_ring = ShapelyPolygon(np.asarray(envelope_xy, dtype=float)).exterior
    union_boundary = None
    room_polys = [ShapelyPolygon(r.boundary) for r in rooms if len(r.boundary) >= 3]
    if room_polys:
        union = unary_union([p.buffer(0) for p in room_polys])
        union_boundary = union.boundary

    for wi, w in enumerate(walls):
        sides = wall_room_sides.get(wi, set())
        distinct_sides = {s for (_rid, s) in sides}
        distinct_rooms = {rid for (rid, _s) in sides}
        mid = Point(w.centerline.mean(axis=0))
        if len(distinct_sides) == 2 and len(distinct_rooms) >= 2:
            w.wall_class = WallClass.DEMISING
            continue
        near_tol = w.thickness_m / 2.0 + 0.30
        if envelope_ring is not None and envelope_ring.distance(mid) <= near_tol:
            w.wall_class = WallClass.EXTERIOR
        elif union_boundary is not None and union_boundary.distance(mid) <= near_tol:
            w.wall_class = WallClass.EXTERIOR
        else:
            w.wall_class = WallClass.PARTITION

    # 4. Openings / columns.
    geo_openings: list[Opening] = []
    for k, op in enumerate(openings or []):
        geo_openings.append(Opening(
            id=f"op-{k:04d}",
            segment=np.asarray(op.seg, dtype=float),
            width_m=float(op.width_m),
            kind="door",
        ))
    geo_columns: list[ColumnFeature] = []
    for k, col in enumerate(columns or []):
        radius = None
        if getattr(col, "is_round", False):
            radius = float(0.25 * (col.size_m[0] + col.size_m[1]))
        geo_columns.append(ColumnFeature(
            id=f"col-{k:04d}",
            centre=(float(col.centre_m[0]), float(col.centre_m[1])),
            polygon=np.asarray(col.corners, dtype=float),
            radius_m=radius,
            is_round=bool(getattr(col, "is_round", False)),
        ))

    return FloorGeometry(
        floor_id=floor_id,
        walls=walls,
        rooms=rooms,
        openings=geo_openings,
        columns=geo_columns,
        envelope=(
            np.asarray(envelope_xy, dtype=float) if envelope_xy is not None else None
        ),
        source="vectorize",
    )


# ── Alignment pipeline → FloorGeometry ───────────────────────────────────────

def _room_field(room: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict room or a dataclass room."""
    if isinstance(room, dict):
        return room.get(key, default)
    return getattr(room, key, default)


def floor_geometry_from_alignment_rooms(
    rooms: Sequence[Any],
    wall_thickness_m: float = 0.10,
    max_demising_gap_m: float = 0.45,
    pair_angle_tol_deg: float = 10.0,
    min_overlap_fraction: float = 0.30,
    floor_id: str = "floor",
    envelope_xy: Optional[np.ndarray] = None,
) -> FloorGeometry:
    """Build FloorGeometry from alignment-pipeline rooms.

    Accepts :class:`app.pipeline.rooms.Room` dataclasses or the plain dicts
    produced by ``scanplan.generate_rooms_from_scan`` (fields: ``id``,
    ``label``, ``category``, ``polygon_2d``).

    These rooms are occupancy polygons — their boundaries are the interior
    finish faces of the enclosing walls (``boundary_basis=interior_face``).
    Walls are synthesized per edge:

    - Two roughly-parallel edges from DIFFERENT rooms, facing each other
      across a gap ≤ ``max_demising_gap_m`` with ≥ ``min_overlap_fraction``
      longitudinal overlap, become ONE shared demising wall whose thickness
      is the measured gap.
    - Every unpaired edge becomes an exterior wall at ``wall_thickness_m``.
      (Alignment rooms tile the scanned floor, so an edge not facing
      another room faces the outside of the scanned region.)

    Common-area flags come from the room category
    (:data:`COMMON_CATEGORIES`).
    """
    # Normalize rings: drop closing duplicate, force CCW.
    rings: list[np.ndarray] = []
    metas: list[dict] = []
    for room in rooms:
        pts = np.asarray(_room_field(room, "polygon_2d"), dtype=float)
        if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            continue
        if ring_signed_area(pts) < 0:
            pts = pts[::-1].copy()
        rings.append(pts)
        category = str(_room_field(room, "category", "unknown") or "unknown")
        # An explicit is_common flag (operator edit or pipeline decision)
        # wins over the category-derived default.
        explicit_common = _room_field(room, "is_common", None)
        metas.append({
            "id": str(_room_field(room, "id")),
            "label": str(_room_field(room, "label", "") or ""),
            "category": category,
            "is_common": (
                bool(explicit_common) if explicit_common is not None
                else category in COMMON_CATEGORIES
            ),
        })

    # Edge inventory: (room_idx, edge_idx, p, q, mid, dir, len).
    edges: list[dict] = []
    for ri, ring in enumerate(rings):
        k_edges = len(ring)
        for k in range(k_edges):
            p = ring[k]
            q = ring[(k + 1) % k_edges]
            d = q - p
            elen = float(np.linalg.norm(d))
            if elen < 1e-12:
                continue
            edges.append({
                "room": ri, "edge": k, "p": p, "q": q,
                "mid": (p + q) / 2.0, "dir": d / elen, "len": elen,
            })

    # Pair facing edges across rooms (greedy by smallest gap).
    cos_tol = np.cos(np.radians(pair_angle_tol_deg))
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            a, b = edges[i], edges[j]
            if a["room"] == b["room"]:
                continue
            if abs(float(np.dot(a["dir"], b["dir"]))) < cos_tol:
                continue
            u = a["dir"]
            rel = b["mid"] - a["mid"]
            gap = abs(float(rel[0] * u[1] - rel[1] * u[0]))
            if gap > max_demising_gap_m or gap < 1e-9:
                continue
            # Longitudinal overlap along a's direction.
            ta0 = float(np.dot(a["p"] - a["mid"], u))
            ta1 = float(np.dot(a["q"] - a["mid"], u))
            tb0 = float(np.dot(b["p"] - a["mid"], u))
            tb1 = float(np.dot(b["q"] - a["mid"], u))
            lo_a, hi_a = sorted((ta0, ta1))
            lo_b, hi_b = sorted((tb0, tb1))
            overlap = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
            shorter = min(a["len"], b["len"])
            if shorter < 1e-9 or overlap / shorter < min_overlap_fraction:
                continue
            candidates.append((gap, i, j))

    candidates.sort(key=lambda c: c[0])
    partner: dict[int, int] = {}
    for _gap, i, j in candidates:
        if i in partner or j in partner:
            continue
        partner[i] = j
        partner[j] = i

    # Materialize walls + per-room refs.
    walls: list[Wall] = []
    refs_per_room: dict[int, dict[int, WallFaceRef]] = {ri: {} for ri in range(len(rings))}
    wall_of_edge: dict[int, str] = {}

    def _new_wall_id() -> str:
        return f"aw-{len(walls):04d}"

    for ei, e in enumerate(edges):
        if ei in wall_of_edge:
            continue
        pj = partner.get(ei)
        if pj is not None:
            a, b = edges[ei], edges[pj]
            u = a["dir"]
            rel = b["mid"] - a["mid"]
            gap = abs(float(rel[0] * u[1] - rel[1] * u[0]))
            # Centerline midway between the two facing interior faces,
            # spanning the union of their longitudinal extents.
            offset = (b["mid"] - a["mid"]) - float(np.dot(rel, u)) * u
            base = a["mid"] + offset / 2.0
            ts = [
                float(np.dot(pt - base, u))
                for pt in (a["p"], a["q"], b["p"], b["q"])
            ]
            cl = np.array([base + min(ts) * u, base + max(ts) * u])
            wid = _new_wall_id()
            walls.append(Wall(
                id=wid,
                centerline=cl,
                thickness_m=float(gap),
                wall_class=WallClass.DEMISING,
                source="alignment.synthesized",
            ))
            wall_of_edge[ei] = wid
            wall_of_edge[pj] = wid
            refs_per_room[a["room"]][a["edge"]] = WallFaceRef(wid, side="a")
            refs_per_room[b["room"]][b["edge"]] = WallFaceRef(wid, side="b")
        else:
            # Unpaired: exterior wall.  The room ring is the interior face,
            # so the centerline sits half a thickness outward.
            u = e["dir"]
            outward = np.array([u[1], -u[0]])  # right of a CCW edge = outward
            cl = np.array([e["p"], e["q"]]) + outward * (wall_thickness_m / 2.0)
            wid = _new_wall_id()
            walls.append(Wall(
                id=wid,
                centerline=cl,
                thickness_m=float(wall_thickness_m),
                wall_class=WallClass.EXTERIOR,
                source="alignment.synthesized",
            ))
            wall_of_edge[ei] = wid
            refs_per_room[e["room"]][e["edge"]] = WallFaceRef(wid, side="a")

    geo_rooms: list[Room] = []
    for ri, ring in enumerate(rings):
        k_edges = len(ring)
        refs = [
            refs_per_room[ri].get(k, WallFaceRef(None))
            for k in range(k_edges)
        ]
        geo_rooms.append(Room(
            id=metas[ri]["id"],
            boundary=ring,
            wall_refs=refs,
            boundary_basis=BoundaryBasis.INTERIOR_FACE,
            label=metas[ri]["label"],
            category=metas[ri]["category"],
            is_common=metas[ri]["is_common"],
        ))

    envelope = None
    if envelope_xy is not None:
        env = np.asarray(envelope_xy, dtype=float)
        if len(env) >= 3:
            if np.allclose(env[0], env[-1]):
                env = env[:-1]
            if len(env) >= 3:
                envelope = env

    return FloorGeometry(
        floor_id=floor_id,
        walls=walls,
        rooms=geo_rooms,
        envelope=envelope,
        source="alignment",
    )
