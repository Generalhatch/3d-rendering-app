"""Canonical floor geometry model (Phase 1 of the Scan-to-BOMA overhaul).

One standard-agnostic representation of a measured floor that every
extractor emits and every consumer (measurement engine, sheet renderer,
editor) reads:

- **Walls** carry a centerline, BOTH faces, a thickness, and a class
  (exterior / demising / partition).  Capturing both faces makes the model
  IPMS-compatible: IPMS 1 measures to the external face, IPMS 2 to the
  internal dominant face — both are recoverable from this record without
  re-running any extractor.
- **Rooms** are ordered rings of wall-face references: edge ``k`` of
  ``Room.boundary`` runs ``boundary[k] → boundary[(k+1) % K]`` and is
  attributed to ``wall_refs[k]``.  The ring itself is stored on a declared
  ``boundary_basis`` (wall centerline for the vectorize pipeline, interior
  finish face for the alignment pipeline) so measurement rules know what
  offset is still owed.
- **Openings, columns, penetrations, envelope, common-area flags** round
  out what a measurement standard needs.

Consolidated area computations
------------------------------
This module is the single home for polygon area math (`ring_signed_area`,
`ring_perimeter`, `polygon_area`, `offset_ring`,
`FloorGeometry.measured_room_polygon`).  The vectorize topology module and
the alignment room extractor import these helpers instead of carrying
their own shoelace copies.

The one calculation path for measured areas
-------------------------------------------
`FloorGeometry.measured_room_polygon(room_id, rules)` is THE way any
measurement standard turns a room into a number.  A
:class:`BoundaryRules` says, per wall class, which line to measure to
(interior face / centerline / outside face / dominant portion); the model
offsets each ring edge by the signed distance still owed given the room's
``boundary_basis``, intersects adjacent offset lines to rebuild the ring,
and returns a Shapely polygon.  Rulesets differ only in the rules they
pass and in how they apportion the results — never in polygon math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

# ── Enums ─────────────────────────────────────────────────────────────────────

class WallClass(str, Enum):
    """Architectural role of a wall — drives measurement boundary rules."""
    EXTERIOR = "exterior"       # building shell
    DEMISING = "demising"       # separates two occupancies (or suite/common)
    PARTITION = "partition"     # interior wall within one occupancy


class BoundaryBasis(str, Enum):
    """What line a Room.boundary ring is drawn on."""
    CENTERLINE = "centerline"           # vectorize pipeline (wall-graph faces)
    INTERIOR_FACE = "interior_face"     # alignment pipeline (occupancy hulls)
    OUTSIDE_FACE = "outside_face"


class BoundaryTarget(str, Enum):
    """What line a measurement standard measures to, per wall class."""
    INTERIOR_FACE = "interior_face"
    CENTERLINE = "centerline"
    OUTSIDE_FACE = "outside_face"
    # BOMA "dominant portion": the inside face of whatever occupies >50% of
    # the wall's vertical section (glass line when glazing dominates, the
    # interior finish otherwise).  Distance comes from
    # ``Wall.dominant_inward_offset_m``; defaults to thickness/2 (finish).
    DOMINANT_PORTION = "dominant_portion"


# ── Consolidated polygon math ─────────────────────────────────────────────────

def ring_signed_area(ring: np.ndarray) -> float:
    """Shoelace signed area of an ordered ring (CCW positive).

    The single canonical implementation — extractors import this instead of
    keeping local copies.
    """
    ring = np.asarray(ring, dtype=float)
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ring_perimeter(ring: np.ndarray) -> float:
    """Perimeter of an ordered closed ring (closing edge implied)."""
    ring = np.asarray(ring, dtype=float)
    diff = np.roll(ring, -1, axis=0) - ring
    return float(np.sum(np.sqrt(np.sum(diff * diff, axis=1))))


def polygon_area(ring: np.ndarray | Sequence[Sequence[float]]) -> float:
    """Area of a ring, validated/repaired through Shapely.

    Repairs self-touching rings with ``buffer(0)`` (largest part kept) so a
    slightly-degenerate extractor ring still yields an honest area instead
    of shoelace garbage.
    """
    pts = np.asarray(ring, dtype=float)
    if len(pts) < 3:
        return 0.0
    shp = ShapelyPolygon(pts)
    if not shp.is_valid:
        shp = shp.buffer(0)
        if shp.is_empty:
            return 0.0
        if shp.geom_type == "MultiPolygon":
            shp = max(shp.geoms, key=lambda g: g.area)
    return float(shp.area)


def offset_ring(ring: np.ndarray, offsets_outward: Sequence[float]) -> np.ndarray:
    """Offset each edge of a CCW ring by a per-edge signed distance.

    ``offsets_outward[k]`` moves edge ``k`` (``ring[k] → ring[(k+1)%K]``)
    along its outward normal: positive expands the polygon, negative
    shrinks it.  New vertices are the intersections of consecutive offset
    edge lines (the standard variable-offset construction); consecutive
    parallel edges fall back to the directly-offset shared vertex.

    The ring must be counter-clockwise; use :func:`ring_signed_area` to
    check and reverse first if needed.
    """
    ring = np.asarray(ring, dtype=float)
    k_edges = len(ring)
    if k_edges < 3:
        return ring.copy()
    if len(offsets_outward) != k_edges:
        raise ValueError(
            f"offsets length {len(offsets_outward)} != ring edges {k_edges}"
        )

    # Offset line for each edge: (point, unit direction).
    points: list[np.ndarray] = []
    dirs: list[np.ndarray] = []
    for k in range(k_edges):
        p = ring[k]
        q = ring[(k + 1) % k_edges]
        d = q - p
        length = float(np.linalg.norm(d))
        if length < 1e-12:
            raise ValueError(f"degenerate zero-length edge at index {k}")
        d = d / length
        n_out = np.array([d[1], -d[0]])  # outward normal for a CCW ring
        points.append(p + float(offsets_outward[k]) * n_out)
        dirs.append(d)

    out = np.zeros_like(ring)
    for k in range(k_edges):
        p1, d1 = points[(k - 1) % k_edges], dirs[(k - 1) % k_edges]
        p2, d2 = points[k], dirs[k]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-9:
            # Parallel (collinear) neighbours: the shared vertex just slides
            # along the common normal.
            out[k] = p2
        else:
            diff = p2 - p1
            t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
            out[k] = p1 + t * d1
    return out


# ── Entities ──────────────────────────────────────────────────────────────────

@dataclass
class Wall:
    """A thickness-aware wall with both faces and a measurement class."""
    id: str
    centerline: np.ndarray               # (2, 2)
    thickness_m: float
    wall_class: WallClass = WallClass.PARTITION
    face_a: Optional[np.ndarray] = None  # (2, 2); derived from centerline if absent
    face_b: Optional[np.ndarray] = None
    # BOMA dominant portion: distance from the CENTERLINE toward the room
    # interior at which the dominant portion sits.  None → thickness/2
    # (the interior finish face).  A mostly-glass exterior wall has a
    # smaller value (the glass line sits outboard of the finish face).
    dominant_inward_offset_m: Optional[float] = None
    is_paired: bool = True               # both faces observed by the scanner
    source: str = ""                     # provenance tag (extractor name)

    def __post_init__(self) -> None:
        self.centerline = np.asarray(self.centerline, dtype=float).reshape(2, 2)
        if self.face_a is None or self.face_b is None:
            d = self.centerline[1] - self.centerline[0]
            length = float(np.linalg.norm(d))
            if length > 1e-12:
                d = d / length
                perp = np.array([-d[1], d[0]])
                half = self.thickness_m / 2.0
                self.face_a = self.centerline + perp * half
                self.face_b = self.centerline - perp * half

    def dominant_offset_from_centerline(self) -> float:
        """Inward distance from centerline to the BOMA dominant portion."""
        if self.dominant_inward_offset_m is not None:
            return float(self.dominant_inward_offset_m)
        return self.thickness_m / 2.0


@dataclass
class WallFaceRef:
    """One room-ring edge's attribution to a wall.

    ``side`` records which face of the wall the room sees ("a" / "b") when
    the adapter can determine it; "unknown" otherwise.  Measurement only
    needs the wall's thickness + class, so "unknown" is fully usable.
    """
    wall_id: Optional[str]
    side: str = "unknown"


@dataclass
class Room:
    """A room as an ordered ring of wall-face references."""
    id: str
    boundary: np.ndarray                 # (K, 2) ordered ring, no closing dup
    wall_refs: list[WallFaceRef]         # len K; edge k = boundary[k] → boundary[k+1]
    boundary_basis: BoundaryBasis = BoundaryBasis.CENTERLINE
    label: str = ""
    category: str = "unknown"            # office | bathroom | hallway | common | unknown
    suite_id: Optional[str] = None       # rooms sharing a suite_id sum into one suite
    is_common: bool = False              # floor common area (corridor, lobby, restroom)

    def __post_init__(self) -> None:
        self.boundary = np.asarray(self.boundary, dtype=float).reshape(-1, 2)
        if len(self.wall_refs) != len(self.boundary):
            raise ValueError(
                f"room {self.id}: {len(self.wall_refs)} wall refs for "
                f"{len(self.boundary)} boundary edges"
            )


@dataclass
class Opening:
    """A door / gap in a wall."""
    id: str
    segment: np.ndarray                  # (2, 2) world coords along the wall
    width_m: float
    kind: str = "door"                   # door | window | gap
    wall_id: Optional[str] = None


@dataclass
class ColumnFeature:
    """A structural column footprint."""
    id: str
    centre: tuple[float, float]
    polygon: Optional[np.ndarray] = None   # (4, 2) rectangle corners
    radius_m: Optional[float] = None       # round columns
    is_round: bool = False


@dataclass
class Penetration:
    """A major vertical penetration (shaft / stair / elevator / atrium).

    Standards disagree on these: BOMA excludes them from rentable, REBNY
    famously does not.  The model just records them.
    """
    id: str
    polygon: np.ndarray                  # (K, 2) ring
    kind: str = "shaft"                  # shaft | stair | elevator | atrium

    def area_m2(self) -> float:
        return polygon_area(self.polygon)


# ── Measurement boundary rules ────────────────────────────────────────────────

@dataclass(frozen=True)
class BoundaryRules:
    """Per-wall-class boundary targets — the entire geometric difference
    between measurement standards."""
    exterior: BoundaryTarget
    demising: BoundaryTarget
    partition: BoundaryTarget
    # Thickness assumed for ring edges with no wall attribution.
    default_thickness_m: float = 0.10

    def target_for(self, wall_class: WallClass) -> BoundaryTarget:
        return {
            WallClass.EXTERIOR: self.exterior,
            WallClass.DEMISING: self.demising,
            WallClass.PARTITION: self.partition,
        }[wall_class]


_BASIS_OFFSET_FROM_CENTERLINE = {
    # Signed offset (positive = outward from the room) of each basis line
    # relative to the wall centerline, in units of half-thickness.
    BoundaryBasis.CENTERLINE: 0.0,
    BoundaryBasis.INTERIOR_FACE: -1.0,
    BoundaryBasis.OUTSIDE_FACE: +1.0,
}


# ── FloorGeometry ─────────────────────────────────────────────────────────────

@dataclass
class FloorGeometry:
    """The canonical measured floor. All extractors emit this; the
    measurement engine and renderers consume it.  Units: metres."""
    floor_id: str
    walls: list[Wall] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    columns: list[ColumnFeature] = field(default_factory=list)
    penetrations: list[Penetration] = field(default_factory=list)
    envelope: Optional[np.ndarray] = None    # (M, 2) outside-face ring
    units: str = "m"
    source: str = ""                          # which pipeline produced this
    building_name: str = ""
    floor_name: str = ""

    def __post_init__(self) -> None:
        self._walls_by_id = {w.id: w for w in self.walls}
        self._rooms_by_id = {r.id: r for r in self.rooms}

    # ── Lookup ────────────────────────────────────────────────────────────
    def wall_by_id(self, wall_id: Optional[str]) -> Optional[Wall]:
        if wall_id is None:
            return None
        return self._walls_by_id.get(wall_id)

    def room_by_id(self, room_id: str) -> Room:
        return self._rooms_by_id[room_id]

    # ── Consolidated areas ────────────────────────────────────────────────
    def room_area_m2(self, room_id: str) -> float:
        """Room area as drawn (on its own boundary basis, no rule offsets)."""
        return polygon_area(self.room_by_id(room_id).boundary)

    def envelope_area_m2(self) -> float:
        if self.envelope is None:
            return 0.0
        return polygon_area(self.envelope)

    def penetration_area_m2(self) -> float:
        return float(sum(p.area_m2() for p in self.penetrations))

    # ── The one measured-boundary calculation path ────────────────────────
    def measured_room_polygon(
        self,
        room_id: str,
        rules: BoundaryRules,
    ) -> ShapelyPolygon:
        """Room polygon measured per ``rules`` — THE calculation path every
        measurement standard uses.

        Each ring edge is offset by the signed distance between the line
        the ring is drawn on (``room.boundary_basis``) and the line the
        standard measures to (``rules.target_for(wall_class)``), then the
        ring is rebuilt from consecutive offset-line intersections.
        """
        room = self.room_by_id(room_id)
        ring = np.asarray(room.boundary, dtype=float)
        refs = list(room.wall_refs)
        k_edges = len(ring)

        # Normalize to CCW so edge outward normals are well-defined.  When
        # reversing, edge i of the reversed ring is original edge
        # (K - 2 - i) mod K traversed backwards.
        if ring_signed_area(ring) < 0:
            ring = ring[::-1].copy()
            refs = [refs[(k_edges - 2 - i) % k_edges] for i in range(k_edges)]

        basis_half = _BASIS_OFFSET_FROM_CENTERLINE[room.boundary_basis]
        offsets: list[float] = []
        for ref in refs:
            wall = self.wall_by_id(ref.wall_id)
            if wall is None:
                thickness = rules.default_thickness_m
                target = rules.target_for(WallClass.PARTITION)
                dominant_inward = thickness / 2.0
            else:
                thickness = wall.thickness_m
                target = rules.target_for(wall.wall_class)
                dominant_inward = wall.dominant_offset_from_centerline()

            half = thickness / 2.0
            # Signed offset (positive outward) of the TARGET line from the
            # wall centerline:
            if target == BoundaryTarget.CENTERLINE:
                target_off = 0.0
            elif target == BoundaryTarget.INTERIOR_FACE:
                target_off = -half
            elif target == BoundaryTarget.OUTSIDE_FACE:
                target_off = +half
            elif target == BoundaryTarget.DOMINANT_PORTION:
                target_off = -dominant_inward
            else:  # pragma: no cover — enum is closed
                raise ValueError(f"unknown boundary target {target}")

            basis_off = basis_half * half
            offsets.append(target_off - basis_off)

        measured = offset_ring(ring, offsets)
        shp = ShapelyPolygon(measured)
        if not shp.is_valid:
            shp = shp.buffer(0)
            if shp.geom_type == "MultiPolygon":
                shp = max(shp.geoms, key=lambda g: g.area)
        return shp

    # ── Serialization ─────────────────────────────────────────────────────
    @classmethod
    def from_json_dict(cls, data: dict) -> "FloorGeometry":
        """Inverse of :meth:`to_json_dict` — reload a persisted floor.

        Used by the sheet renderer to re-render from ``floor_geometry.json``
        (e.g. after the operator edits suite labels) without re-running any
        pipeline.
        """
        def arr(a):
            return None if a is None else np.asarray(a, dtype=float)

        walls = [
            Wall(
                id=w["id"],
                centerline=arr(w["centerline"]),
                thickness_m=float(w["thickness_m"]),
                wall_class=WallClass(w.get("wall_class", "partition")),
                face_a=arr(w.get("face_a")),
                face_b=arr(w.get("face_b")),
                dominant_inward_offset_m=w.get("dominant_inward_offset_m"),
                is_paired=bool(w.get("is_paired", True)),
                source=w.get("source", ""),
            )
            for w in data.get("walls", [])
        ]
        rooms = [
            Room(
                id=r["id"],
                boundary=arr(r["boundary"]),
                wall_refs=[
                    WallFaceRef(ref.get("wall_id"), ref.get("side", "unknown"))
                    for ref in r["wall_refs"]
                ],
                boundary_basis=BoundaryBasis(r.get("boundary_basis", "centerline")),
                label=r.get("label", ""),
                category=r.get("category", "unknown"),
                suite_id=r.get("suite_id"),
                is_common=bool(r.get("is_common", False)),
            )
            for r in data.get("rooms", [])
        ]
        openings = [
            Opening(
                id=o["id"],
                segment=arr(o["segment"]),
                width_m=float(o["width_m"]),
                kind=o.get("kind", "door"),
                wall_id=o.get("wall_id"),
            )
            for o in data.get("openings", [])
        ]
        columns = [
            ColumnFeature(
                id=c["id"],
                centre=(float(c["centre"][0]), float(c["centre"][1])),
                polygon=arr(c.get("polygon")),
                radius_m=c.get("radius_m"),
                is_round=bool(c.get("is_round", False)),
            )
            for c in data.get("columns", [])
        ]
        penetrations = [
            Penetration(
                id=p["id"],
                polygon=arr(p["polygon"]),
                kind=p.get("kind", "shaft"),
            )
            for p in data.get("penetrations", [])
        ]
        return cls(
            floor_id=data["floor_id"],
            walls=walls,
            rooms=rooms,
            openings=openings,
            columns=columns,
            penetrations=penetrations,
            envelope=arr(data.get("envelope")),
            units=data.get("units", "m"),
            source=data.get("source", ""),
            building_name=data.get("building_name", ""),
            floor_name=data.get("floor_name", ""),
        )

    def to_json_dict(self) -> dict:
        """JSON-safe dict (numpy arrays → nested lists)."""
        def arr(a: Optional[np.ndarray]):
            return None if a is None else np.asarray(a, dtype=float).tolist()

        return {
            "version": 1,
            "units": self.units,
            "floor_id": self.floor_id,
            "source": self.source,
            "building_name": self.building_name,
            "floor_name": self.floor_name,
            "envelope": arr(self.envelope),
            "walls": [
                {
                    "id": w.id,
                    "centerline": arr(w.centerline),
                    "thickness_m": w.thickness_m,
                    "wall_class": w.wall_class.value,
                    "face_a": arr(w.face_a),
                    "face_b": arr(w.face_b),
                    "dominant_inward_offset_m": w.dominant_inward_offset_m,
                    "is_paired": w.is_paired,
                    "source": w.source,
                }
                for w in self.walls
            ],
            "rooms": [
                {
                    "id": r.id,
                    "label": r.label,
                    "category": r.category,
                    "suite_id": r.suite_id,
                    "is_common": r.is_common,
                    "boundary_basis": r.boundary_basis.value,
                    "boundary": arr(r.boundary),
                    "wall_refs": [
                        {"wall_id": ref.wall_id, "side": ref.side}
                        for ref in r.wall_refs
                    ],
                    "area_m2": self.room_area_m2(r.id),
                }
                for r in self.rooms
            ],
            "openings": [
                {
                    "id": o.id,
                    "segment": arr(o.segment),
                    "width_m": o.width_m,
                    "kind": o.kind,
                    "wall_id": o.wall_id,
                }
                for o in self.openings
            ],
            "columns": [
                {
                    "id": c.id,
                    "centre": list(c.centre),
                    "polygon": arr(c.polygon),
                    "radius_m": c.radius_m,
                    "is_round": c.is_round,
                }
                for c in self.columns
            ],
            "penetrations": [
                {
                    "id": p.id,
                    "polygon": arr(p.polygon),
                    "kind": p.kind,
                    "area_m2": p.area_m2(),
                }
                for p in self.penetrations
            ],
        }
