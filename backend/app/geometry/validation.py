"""Floor-closure validation (Phase 2 of the Scan-to-BOMA overhaul).

Sanity checks that a measured :class:`~app.geometry.model.FloorGeometry`
actually closes — i.e. the deliverable can be trusted, or the specific
problems are surfaced as STRUCTURED data (never just a log line):

1. **Envelope-vs-rooms area check** — the union of rooms + wall footprints
   + penetrations must cover the envelope to within a tolerance.  A big
   shortfall means part of the floorplate was never assigned to any room.
2. **Closed exterior perimeter** — the exterior wall centerlines must chain
   into closed loop(s); unmatched endpoints are reported with coordinates.
3. **Unscanned interior gaps** — any interior region of the envelope not
   covered by a room / wall / penetration, larger than ``min_gap_area_m2``
   (default 2 m²), is flagged as an ``unscanned`` polygon in the output.

The result serializes with :meth:`FloorValidation.to_json_dict` and is
persisted alongside the floor geometry by both pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from .model import FloorGeometry, WallClass

# Interior regions smaller than this are treated as extraction slack (wall
# join slivers, corner notches), not as missing scan coverage.
DEFAULT_MIN_GAP_AREA_M2 = 2.0

# Coverage tolerance: the room+wall union must explain at least this
# fraction of the envelope area for the floor to pass the area check.
DEFAULT_AREA_TOLERANCE_FRACTION = 0.05

# Two exterior-wall endpoints within this distance count as connected.
DEFAULT_PERIMETER_SNAP_TOL_M = 0.10


@dataclass
class UnscannedGap:
    """One interior region of the envelope not covered by any room, wall,
    or penetration — i.e. floor area the scan assembly never explained."""
    id: str
    polygon: np.ndarray            # (K, 2) ring
    area_m2: float
    centroid: tuple[float, float]

    def to_json_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": "unscanned",
            "polygon": np.asarray(self.polygon, dtype=float).tolist(),
            "area_m2": round(float(self.area_m2), 4),
            "centroid": [float(self.centroid[0]), float(self.centroid[1])],
        }


@dataclass
class FloorValidation:
    """Structured result of :func:`validate_floor_closure`."""
    # Area accounting (all m², on the geometry's own boundary bases).
    envelope_area_m2: float | None      # None when no envelope was extracted
    rooms_total_m2: float               # every room, suites + common
    suites_total_m2: float              # non-common rooms
    common_total_m2: float              # is_common rooms
    # Envelope coverage: covered-fraction of the envelope (rooms + wall
    # footprints + penetrations).  None when no envelope exists.
    coverage_ratio: float | None
    area_check_passed: bool | None      # None = not checkable (no envelope)
    # Exterior perimeter closure.
    perimeter_closed: bool | None       # None = no exterior walls to check
    perimeter_open_endpoints: list[tuple[float, float]] = field(default_factory=list)
    # Interior coverage gaps > min_gap_area_m2.
    unscanned_gaps: list[UnscannedGap] = field(default_factory=list)
    # Structured warnings — one dict per problem, machine-readable ``code``.
    warnings: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every checkable validation passed and nothing is
        flagged unscanned."""
        return (
            self.area_check_passed is not False
            and self.perimeter_closed is not False
            and not self.unscanned_gaps
        )

    def to_json_dict(self) -> dict:
        return {
            "version": 1,
            "passed": self.passed,
            "envelope_area_m2": (
                None if self.envelope_area_m2 is None
                else round(float(self.envelope_area_m2), 4)
            ),
            "rooms_total_m2": round(float(self.rooms_total_m2), 4),
            "suites_total_m2": round(float(self.suites_total_m2), 4),
            "common_total_m2": round(float(self.common_total_m2), 4),
            "coverage_ratio": (
                None if self.coverage_ratio is None
                else round(float(self.coverage_ratio), 6)
            ),
            "area_check_passed": self.area_check_passed,
            "perimeter_closed": self.perimeter_closed,
            "perimeter_open_endpoints": [
                [float(x), float(y)] for (x, y) in self.perimeter_open_endpoints
            ],
            "unscanned_gaps": [g.to_json_dict() for g in self.unscanned_gaps],
            "warnings": list(self.warnings),
        }


def _valid_polygon(ring: np.ndarray | None) -> ShapelyPolygon | None:
    if ring is None or len(ring) < 3:
        return None
    poly = ShapelyPolygon(np.asarray(ring, dtype=float))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def _wall_footprint(centerline: np.ndarray, thickness_m: float):
    """Rectangle occupied by a wall: its centerline buffered by half the
    thickness with flat caps."""
    line = LineString(np.asarray(centerline, dtype=float))
    if line.length < 1e-9:
        return None
    return line.buffer(max(thickness_m, 1e-6) / 2.0, cap_style="flat")


def _check_exterior_perimeter(
    floor: FloorGeometry,
    snap_tol_m: float,
) -> tuple[bool | None, list[tuple[float, float]]]:
    """Chain exterior wall centerlines by endpoint proximity.

    Closed ⇔ every endpoint pairs with exactly one other exterior-wall
    endpoint within ``snap_tol_m``.  Endpoints with no partner (or with an
    odd number of incident walls) are returned as open-perimeter locations.
    """
    exterior = [w for w in floor.walls if w.wall_class == WallClass.EXTERIOR]
    if len(exterior) < 3:
        return None, []

    endpoints: list[np.ndarray] = []
    for w in exterior:
        endpoints.append(np.asarray(w.centerline[0], dtype=float))
        endpoints.append(np.asarray(w.centerline[1], dtype=float))

    # Cluster endpoints within snap tolerance (greedy union).
    n = len(endpoints)
    cluster_of = [-1] * n
    clusters: list[np.ndarray] = []
    for i, p in enumerate(endpoints):
        for ci, c in enumerate(clusters):
            if float(np.linalg.norm(p - c)) <= snap_tol_m:
                cluster_of[i] = ci
                break
        else:
            cluster_of[i] = len(clusters)
            clusters.append(p)

    counts = np.bincount(np.asarray(cluster_of), minlength=len(clusters))
    open_points = [
        (float(clusters[ci][0]), float(clusters[ci][1]))
        for ci in range(len(clusters))
        if counts[ci] % 2 == 1        # a closed loop touches each junction an even number of times
    ]
    return (len(open_points) == 0), open_points


def validate_floor_closure(
    floor: FloorGeometry,
    min_gap_area_m2: float = DEFAULT_MIN_GAP_AREA_M2,
    area_tolerance_fraction: float = DEFAULT_AREA_TOLERANCE_FRACTION,
    perimeter_snap_tol_m: float = DEFAULT_PERIMETER_SNAP_TOL_M,
) -> FloorValidation:
    """Run the Phase-2 floor-closure checks on a FloorGeometry.

    Returns a :class:`FloorValidation` with structured warnings — callers
    persist ``result.to_json_dict()`` next to the floor geometry so the UI
    and the measurement report can show exactly what is and isn't covered.
    """
    warnings: list[dict] = []

    # ── Area accounting ────────────────────────────────────────────────────
    rooms_total = float(sum(floor.room_area_m2(r.id) for r in floor.rooms))
    common_total = float(sum(
        floor.room_area_m2(r.id) for r in floor.rooms if r.is_common
    ))
    suites_total = rooms_total - common_total

    envelope_poly = _valid_polygon(floor.envelope)
    envelope_area = float(envelope_poly.area) if envelope_poly is not None else None

    # ── Coverage union: rooms + wall footprints + penetrations ────────────
    cover_parts = []
    for room in floor.rooms:
        poly = _valid_polygon(room.boundary)
        if poly is not None:
            cover_parts.append(poly)
    for wall in floor.walls:
        fp = _wall_footprint(wall.centerline, wall.thickness_m)
        if fp is not None:
            cover_parts.append(fp)
    for pen in floor.penetrations:
        poly = _valid_polygon(pen.polygon)
        if poly is not None:
            cover_parts.append(poly)
    coverage = unary_union(cover_parts) if cover_parts else None

    coverage_ratio: float | None = None
    area_check_passed: bool | None = None
    unscanned_gaps: list[UnscannedGap] = []

    if envelope_poly is None:
        warnings.append({
            "code": "no_envelope",
            "message": "no building envelope was extracted — envelope-vs-rooms "
                       "area check and unscanned-gap detection are not possible",
        })
    elif coverage is None:
        area_check_passed = False
        coverage_ratio = 0.0
        warnings.append({
            "code": "no_rooms",
            "message": "envelope exists but no rooms/walls were extracted — "
                       "the entire floorplate is unexplained",
        })
    else:
        covered = envelope_poly.intersection(coverage)
        coverage_ratio = float(covered.area / envelope_poly.area)
        area_check_passed = coverage_ratio >= (1.0 - area_tolerance_fraction)
        if not area_check_passed:
            warnings.append({
                "code": "envelope_area_mismatch",
                "message": (
                    f"rooms + walls cover only {coverage_ratio:.1%} of the "
                    f"envelope ({envelope_poly.area:.2f} m²) — at least "
                    f"{(1.0 - coverage_ratio) * envelope_poly.area:.2f} m² of "
                    f"floorplate is unassigned"
                ),
                "coverage_ratio": round(coverage_ratio, 6),
                "envelope_area_m2": round(float(envelope_poly.area), 4),
                "rooms_total_m2": round(rooms_total, 4),
            })

        # Interior gaps: envelope minus coverage, filtered by area.
        gap_geom = envelope_poly.difference(coverage)
        gap_polys = (
            list(gap_geom.geoms) if isinstance(gap_geom, MultiPolygon)
            else ([gap_geom] if not gap_geom.is_empty else [])
        )
        k = 0
        for gp in gap_polys:
            if gp.is_empty or gp.area < min_gap_area_m2:
                continue
            rp = gp.representative_point()
            unscanned_gaps.append(UnscannedGap(
                id=f"gap-{k:03d}",
                polygon=np.asarray(gp.exterior.coords[:-1], dtype=float),
                area_m2=float(gp.area),
                centroid=(float(rp.x), float(rp.y)),
            ))
            k += 1
        if unscanned_gaps:
            total_gap = sum(g.area_m2 for g in unscanned_gaps)
            warnings.append({
                "code": "unscanned_gaps",
                "message": (
                    f"{len(unscanned_gaps)} interior region(s) totalling "
                    f"{total_gap:.2f} m² have no room/wall coverage — "
                    f"likely unscanned"
                ),
                "gap_ids": [g.id for g in unscanned_gaps],
                "total_gap_area_m2": round(total_gap, 4),
            })

    # ── Exterior perimeter closure ─────────────────────────────────────────
    perimeter_closed, open_endpoints = _check_exterior_perimeter(
        floor, perimeter_snap_tol_m,
    )
    if perimeter_closed is None:
        if envelope_poly is not None:
            # No classified exterior walls, but an envelope hull exists —
            # fall back to its validity as the closure signal.
            perimeter_closed = bool(
                envelope_poly.is_valid and envelope_poly.area > 0
            )
        else:
            warnings.append({
                "code": "no_exterior_walls",
                "message": "fewer than 3 exterior walls and no envelope — "
                           "perimeter closure cannot be checked",
            })
    elif perimeter_closed is False:
        warnings.append({
            "code": "open_perimeter",
            "message": (
                f"exterior perimeter does not close — {len(open_endpoints)} "
                f"open endpoint(s)"
            ),
            "open_endpoints": [[float(x), float(y)] for (x, y) in open_endpoints],
        })

    return FloorValidation(
        envelope_area_m2=envelope_area,
        rooms_total_m2=rooms_total,
        suites_total_m2=suites_total,
        common_total_m2=common_total,
        coverage_ratio=coverage_ratio,
        area_check_passed=area_check_passed,
        perimeter_closed=perimeter_closed,
        perimeter_open_endpoints=open_endpoints,
        unscanned_gaps=unscanned_gaps,
        warnings=warnings,
    )
