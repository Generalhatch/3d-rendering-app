"""Tests for the canonical FloorGeometry model + pipeline adapters.

Style follows the Phase 0 tests: synthetic known-geometry fixtures with
exact-area assertions (hand-computed on paper first — see
tests/reference_floors.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.geometry.model import (
    BoundaryBasis,
    BoundaryRules,
    BoundaryTarget,
    FloorGeometry,
    Room,
    Wall,
    WallClass,
    WallFaceRef,
    offset_ring,
    polygon_area,
    ring_perimeter,
    ring_signed_area,
)
from tests.reference_floors import (
    DOMINANT_A_USABLE,
    DOMINANT_B_USABLE,
    TWO_SUITE_A_RENTABLE_REBNY,
    TWO_SUITE_A_USABLE_BOMA,
    TWO_SUITE_B_USABLE_BOMA,
    two_suite_floor,
)

# ── Consolidated area helpers ────────────────────────────────────────────────

class TestAreaHelpers:
    SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

    def test_ring_signed_area_ccw_positive(self):
        assert ring_signed_area(self.SQUARE) == pytest.approx(100.0, abs=1e-12)

    def test_ring_signed_area_cw_negative(self):
        assert ring_signed_area(self.SQUARE[::-1]) == pytest.approx(-100.0, abs=1e-12)

    def test_ring_perimeter(self):
        assert ring_perimeter(self.SQUARE) == pytest.approx(40.0, abs=1e-12)

    def test_polygon_area_repairs_and_measures(self):
        assert polygon_area(self.SQUARE) == pytest.approx(100.0, abs=1e-12)

    def test_topology_uses_model_helpers(self):
        # The scattered shoelace implementations must be consolidated: the
        # vectorize topology module now imports the model's helpers.
        from app.vectorize import topology
        assert topology._signed_area is ring_signed_area
        assert topology._polygon_perimeter is ring_perimeter


# ── Variable-offset ring (the one calculation path for measured boundaries) ──

class TestOffsetRing:
    SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

    def test_uniform_inward_offset(self):
        # Shrink every edge inward by 0.15 → 9.7 x 9.7 = 94.09 exactly.
        out = offset_ring(self.SQUARE, [-0.15] * 4)
        assert polygon_area(out) == pytest.approx(9.7 * 9.7, abs=1e-9)

    def test_uniform_outward_offset(self):
        out = offset_ring(self.SQUARE, [0.15] * 4)
        assert polygon_area(out) == pytest.approx(10.3 * 10.3, abs=1e-9)

    def test_mixed_offsets(self):
        # bottom -0.15, right 0, top -0.15, left -0.15
        # → x: 0.15..10, y: 0.15..9.85 → 9.85 x 9.7 = 95.545
        out = offset_ring(self.SQUARE, [-0.15, 0.0, -0.15, -0.15])
        assert polygon_area(out) == pytest.approx(9.85 * 9.7, abs=1e-9)

    def test_l_shape_offset(self):
        # L-shape (10x10 minus 4x4 notch), uniform inward 0.1.  The reflex
        # corner at (6,6) grows the notch: interior area =
        # 9.8*9.8 - (4+0.1+0.1)*(4+0.1+0.1) ... worked by coordinates:
        # x:0.1..9.9 minus notch x:5.9..9.9 / y:5.9..9.9
        # = 9.8*9.8 - 4.0*4.0 = 96.04 - 16.0 = 80.04
        ring = np.array(
            [[0, 0], [10, 0], [10, 6], [6, 6], [6, 10], [0, 10]], dtype=float
        )
        out = offset_ring(ring, [-0.1] * 6)
        assert polygon_area(out) == pytest.approx(80.04, abs=1e-9)


# ── Measured room polygons on the reference floor ────────────────────────────

BOMA_RULES = BoundaryRules(
    exterior=BoundaryTarget.DOMINANT_PORTION,
    demising=BoundaryTarget.CENTERLINE,
    partition=BoundaryTarget.CENTERLINE,
)
REBNY_RULES = BoundaryRules(
    exterior=BoundaryTarget.OUTSIDE_FACE,
    demising=BoundaryTarget.CENTERLINE,
    partition=BoundaryTarget.CENTERLINE,
)


class TestMeasuredRoomPolygon:
    def test_boma_interior_face(self):
        floor = two_suite_floor()
        a = floor.measured_room_polygon("suite_a", BOMA_RULES)
        b = floor.measured_room_polygon("suite_b", BOMA_RULES)
        assert a.area == pytest.approx(TWO_SUITE_A_USABLE_BOMA, abs=1e-9)
        assert b.area == pytest.approx(TWO_SUITE_B_USABLE_BOMA, abs=1e-9)

    def test_rebny_outside_face(self):
        floor = two_suite_floor()
        a = floor.measured_room_polygon("suite_a", REBNY_RULES)
        assert a.area == pytest.approx(TWO_SUITE_A_RENTABLE_REBNY, abs=1e-9)

    def test_dominant_portion_glass_line(self):
        # Exterior walls >50% glass: dominant portion 0.05 m inward of
        # centerline (instead of the 0.15 m interior finish face).
        floor = two_suite_floor(dominant_inward_offset_m=0.05)
        a = floor.measured_room_polygon("suite_a", BOMA_RULES)
        b = floor.measured_room_polygon("suite_b", BOMA_RULES)
        assert a.area == pytest.approx(DOMINANT_A_USABLE, abs=1e-9)
        assert b.area == pytest.approx(DOMINANT_B_USABLE, abs=1e-9)

    def test_cw_ring_is_normalized(self):
        # Same floor but suite_a's ring supplied clockwise: measured area
        # must be identical (the model normalizes orientation internally).
        floor = two_suite_floor()
        room = floor.room_by_id("suite_a")
        k = len(room.boundary)
        room.boundary = room.boundary[::-1].copy()
        room.wall_refs = [room.wall_refs[(k - 2 - i) % k] for i in range(k)]
        a = floor.measured_room_polygon("suite_a", BOMA_RULES)
        assert a.area == pytest.approx(TWO_SUITE_A_USABLE_BOMA, abs=1e-9)

    def test_interior_face_basis_needs_no_offset(self):
        # A room whose boundary is ALREADY the interior finish face
        # (alignment-pipeline rooms) must come back unchanged under
        # interior-face rules.
        wall = Wall(
            id="w", centerline=np.zeros((2, 2)), thickness_m=0.2,
            wall_class=WallClass.EXTERIOR,
        )
        ring = np.array([[0, 0], [6, 0], [6, 4], [0, 4]], dtype=float)
        room = Room(
            id="r", boundary=ring,
            wall_refs=[WallFaceRef("w")] * 4,
            boundary_basis=BoundaryBasis.INTERIOR_FACE,
        )
        floor = FloorGeometry(floor_id="f", walls=[wall], rooms=[room])
        rules = BoundaryRules(
            exterior=BoundaryTarget.INTERIOR_FACE,
            demising=BoundaryTarget.CENTERLINE,
            partition=BoundaryTarget.CENTERLINE,
        )
        assert floor.measured_room_polygon("r", rules).area == pytest.approx(24.0, abs=1e-9)
        # And outside-face rules push out by the full thickness:
        # (6+0.4) x (4+0.4) = 28.16
        rules_out = BoundaryRules(
            exterior=BoundaryTarget.OUTSIDE_FACE,
            demising=BoundaryTarget.CENTERLINE,
            partition=BoundaryTarget.CENTERLINE,
        )
        assert floor.measured_room_polygon("r", rules_out).area == pytest.approx(
            6.4 * 4.4, abs=1e-9
        )


class TestFloorGeometryAreas:
    def test_room_area_as_drawn(self):
        floor = two_suite_floor()
        assert floor.room_area_m2("suite_a") == pytest.approx(80.0, abs=1e-9)
        assert floor.room_area_m2("suite_b") == pytest.approx(120.0, abs=1e-9)

    def test_envelope_area(self):
        floor = two_suite_floor()
        assert floor.envelope_area_m2() == pytest.approx(20.3 * 10.3, abs=1e-9)


# ── Vectorize adapter (TopologyResult + WallPairingResult → FloorGeometry) ───

class TestVectorizeAdapter:
    def _build(self):
        from app.geometry.adapters import floor_geometry_from_vectorize
        from app.vectorize.topology import TopologyParams, build_topology
        from app.vectorize.walls import Wall as VWall
        from app.vectorize.walls import WallPairingResult

        centerlines = np.array([
            [[0, 0], [20, 0]],
            [[20, 0], [20, 10]],
            [[20, 10], [0, 10]],
            [[0, 10], [0, 0]],
            [[8, 0], [8, 10]],
        ], dtype=float)
        topo = build_topology(
            centerlines,
            params=TopologyParams(snap_tol_m=0.01, snapping_distance_m=0.05),
        )

        def vwall(p, q, t):
            cl = np.array([p, q], dtype=float)
            d = (cl[1] - cl[0]) / np.linalg.norm(cl[1] - cl[0])
            perp = np.array([-d[1], d[0]])
            return VWall(
                centerline=cl, thickness_m=t,
                face_a=cl + perp * t / 2, face_b=cl - perp * t / 2,
                is_paired=True,
            )

        walls = [
            vwall((0, 0), (20, 0), 0.30),
            vwall((20, 0), (20, 10), 0.30),
            vwall((20, 10), (0, 10), 0.30),
            vwall((0, 10), (0, 0), 0.30),
            vwall((8, 0), (8, 10), 0.20),
        ]
        pairing = WallPairingResult(
            walls=walls,
            face_segments=np.zeros((0, 2, 2)),
            centerline_segments=centerlines,
            median_thickness_m=0.30,
            n_paired=5,
            n_unpaired=0,
        )
        envelope = np.array(
            [[-0.15, -0.15], [20.15, -0.15], [20.15, 10.15], [-0.15, 10.15]],
            dtype=float,
        )
        return floor_geometry_from_vectorize(topo, pairing, envelope_xy=envelope)

    def test_rooms_carried_over_with_exact_areas(self):
        floor = self._build()
        areas = sorted(floor.room_area_m2(r.id) for r in floor.rooms)
        assert areas == pytest.approx([80.0, 120.0], abs=1e-6)

    def test_every_room_edge_has_a_wall_ref(self):
        floor = self._build()
        for room in floor.rooms:
            assert len(room.wall_refs) == len(room.boundary)
            for ref in room.wall_refs:
                assert ref.wall_id is not None

    def test_wall_classification(self):
        floor = self._build()
        # The divider at x=8 has rooms on BOTH sides → demising.
        # The four shell walls border rooms on one side only and lie on the
        # envelope → exterior.
        classes = {}
        for room in floor.rooms:
            for ref in room.wall_refs:
                w = floor.wall_by_id(ref.wall_id)
                mid = w.centerline.mean(axis=0)
                classes[(round(mid[0], 1), round(mid[1], 1))] = w.wall_class
        assert classes[(8.0, 5.0)] == WallClass.DEMISING
        assert classes[(10.0, 0.0)] == WallClass.EXTERIOR
        assert classes[(10.0, 10.0)] == WallClass.EXTERIOR
        assert classes[(0.0, 5.0)] == WallClass.EXTERIOR
        assert classes[(20.0, 5.0)] == WallClass.EXTERIOR

    def test_boundary_basis_is_centerline(self):
        floor = self._build()
        for room in floor.rooms:
            assert room.boundary_basis == BoundaryBasis.CENTERLINE

    def test_measured_area_matches_hand_computation(self):
        # End-to-end: topology output → FloorGeometry → BOMA usable.
        floor = self._build()
        usable = sorted(
            floor.measured_room_polygon(r.id, BOMA_RULES).area for r in floor.rooms
        )
        assert usable == pytest.approx(
            [TWO_SUITE_A_USABLE_BOMA, TWO_SUITE_B_USABLE_BOMA], abs=1e-6
        )


# ── Alignment adapter (pipeline rooms → FloorGeometry) ───────────────────────

class TestAlignmentAdapter:
    ROOMS = [
        {
            "id": "room-001",
            "label": "Office 1",
            "category": "office",
            "polygon_2d": [(0.0, 0.0), (7.9, 0.0), (7.9, 9.7), (0.0, 9.7), (0.0, 0.0)],
            "centroid": (3.95, 4.85),
            "area_m2": 76.63,
        },
        {
            "id": "room-002",
            "label": "Corridor",
            "category": "hallway",
            "polygon_2d": [(8.1, 0.0), (12.0, 0.0), (12.0, 9.7), (8.1, 9.7), (8.1, 0.0)],
            "centroid": (10.05, 4.85),
            "area_m2": 37.83,
        },
    ]

    def _build(self):
        from app.geometry.adapters import floor_geometry_from_alignment_rooms
        return floor_geometry_from_alignment_rooms(self.ROOMS)

    def test_rooms_and_basis(self):
        floor = self._build()
        assert len(floor.rooms) == 2
        for room in floor.rooms:
            assert room.boundary_basis == BoundaryBasis.INTERIOR_FACE
        assert floor.room_area_m2("room-001") == pytest.approx(7.9 * 9.7, abs=1e-6)

    def test_common_area_flag_from_category(self):
        floor = self._build()
        assert floor.room_by_id("room-002").is_common is True
        assert floor.room_by_id("room-001").is_common is False

    def test_shared_edge_becomes_one_demising_wall(self):
        floor = self._build()
        r1 = floor.room_by_id("room-001")
        r2 = floor.room_by_id("room-002")
        # Room 1's right edge and room 2's left edge are 0.2 m apart →
        # they must share a single demising wall of thickness ≈ 0.2.
        w1 = {floor.wall_by_id(ref.wall_id).id for ref in r1.wall_refs}
        w2 = {floor.wall_by_id(ref.wall_id).id for ref in r2.wall_refs}
        shared = w1 & w2
        assert len(shared) == 1
        wall = floor.wall_by_id(shared.pop())
        assert wall.wall_class == WallClass.DEMISING
        assert wall.thickness_m == pytest.approx(0.2, abs=1e-6)

    def test_unshared_edges_are_exterior(self):
        floor = self._build()
        r1 = floor.room_by_id("room-001")
        classes = [
            floor.wall_by_id(ref.wall_id).wall_class for ref in r1.wall_refs
        ]
        assert classes.count(WallClass.DEMISING) == 1
        assert classes.count(WallClass.EXTERIOR) == 3

    def test_accepts_dataclass_rooms(self):
        from app.geometry.adapters import floor_geometry_from_alignment_rooms
        from app.pipeline.rooms import Room as PipelineRoom
        rooms = [
            PipelineRoom(
                id=d["id"], label=d["label"], category=d["category"],
                polygon_2d=list(d["polygon_2d"]), centroid=d["centroid"],
                area_m2=d["area_m2"],
            )
            for d in self.ROOMS
        ]
        floor = floor_geometry_from_alignment_rooms(rooms)
        assert len(floor.rooms) == 2
        assert floor.room_area_m2("room-001") == pytest.approx(7.9 * 9.7, abs=1e-6)


# ── JSON round-trip (Phase 3: sheet re-render from floor_geometry.json) ──────

class TestFromJsonDict:
    def _full_floor(self) -> FloorGeometry:
        from app.geometry.model import ColumnFeature, Opening
        from tests.reference_floors import corridor_floor
        floor = corridor_floor(with_shaft=True)
        floor.building_name = "Round Trip Tower"
        floor.floor_name = "Level 12"
        floor.source = "vectorize"
        floor.openings.append(Opening(
            id="d1", segment=np.array([[4.0, 0.0], [5.0, 0.0]]),
            width_m=1.0, kind="door", wall_id="w_bottom",
        ))
        floor.columns.append(ColumnFeature(
            id="c1", centre=(6.0, 5.0),
            polygon=np.array([[5.8, 4.8], [6.2, 4.8], [6.2, 5.2], [5.8, 5.2]]),
        ))
        floor.columns.append(ColumnFeature(
            id="c2", centre=(10.0, 5.0), radius_m=0.25, is_round=True,
        ))
        return floor

    def test_roundtrip_preserves_geometry(self):
        import json

        original = self._full_floor()
        # Through actual JSON text, not just the dict — catches list/tuple
        # and float32 issues.
        reloaded = FloorGeometry.from_json_dict(
            json.loads(json.dumps(original.to_json_dict()))
        )

        assert reloaded.floor_id == original.floor_id
        assert reloaded.building_name == "Round Trip Tower"
        assert reloaded.floor_name == "Level 12"
        assert reloaded.source == "vectorize"
        assert [w.id for w in reloaded.walls] == [w.id for w in original.walls]
        assert [w.wall_class for w in reloaded.walls] == [
            w.wall_class for w in original.walls
        ]
        for room in original.rooms:
            assert reloaded.room_area_m2(room.id) == pytest.approx(
                original.room_area_m2(room.id), abs=1e-9
            )
            assert reloaded.room_by_id(room.id).is_common == room.is_common
            assert reloaded.room_by_id(room.id).boundary_basis == room.boundary_basis
        assert reloaded.envelope_area_m2() == pytest.approx(
            original.envelope_area_m2(), abs=1e-9
        )
        assert reloaded.penetration_area_m2() == pytest.approx(4.0, abs=1e-9)
        assert reloaded.openings[0].wall_id == "w_bottom"
        assert reloaded.columns[1].is_round is True
        assert reloaded.columns[1].radius_m == pytest.approx(0.25)

    def test_roundtrip_measured_polygon_identical(self):
        import json
        original = self._full_floor()
        reloaded = FloorGeometry.from_json_dict(
            json.loads(json.dumps(original.to_json_dict()))
        )
        rules = BoundaryRules(
            exterior=BoundaryTarget.DOMINANT_PORTION,
            demising=BoundaryTarget.CENTERLINE,
            partition=BoundaryTarget.CENTERLINE,
        )
        for room in original.rooms:
            assert reloaded.measured_room_polygon(room.id, rules).area == \
                pytest.approx(
                    original.measured_room_polygon(room.id, rules).area, abs=1e-9
                )
