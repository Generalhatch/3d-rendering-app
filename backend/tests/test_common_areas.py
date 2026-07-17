"""Phase 2: common areas as a first-class category.

Covers:
  - app.geometry.classify: shared shape classifier + polygon metrics
    (exact values on hand-computed rectangles)
  - conservative shape→common mapping (corridor/restroom yes, big convex
    "lobby" no — a 120 m² open plan is more likely a tenant suite)
  - vectorize adapter: topology rooms get category + is_common
  - alignment adapter: explicit is_common overrides category default
  - scanplan room dicts carry is_common
  - end-to-end: pipeline-style room dicts → FloorGeometry → BOMA
    apportionment actually uses the flag (load factor > 1)
"""
from __future__ import annotations

import numpy as np
import pytest

from app.geometry.classify import (
    COMMON_CATEGORIES,
    SHAPE_COMMON_CATEGORIES,
    classify_room_polygon,
    classify_room_shape,
    polygon_shape_metrics,
)


def _rect(w: float, h: float, x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    return np.array(
        [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], dtype=float
    )


# ── polygon_shape_metrics: exact values on rectangles ────────────────────────

class TestPolygonShapeMetrics:
    def test_rectangle_metrics_exact(self):
        area, ecc, sol = polygon_shape_metrics(_rect(16.0, 2.0))
        assert area == pytest.approx(32.0, abs=1e-9)
        assert ecc == pytest.approx(1.0 - 2.0 / 16.0, abs=1e-9)   # 0.875
        assert sol == pytest.approx(1.0, abs=1e-9)

    def test_square_has_zero_eccentricity(self):
        _, ecc, sol = polygon_shape_metrics(_rect(5.0, 5.0))
        assert ecc == pytest.approx(0.0, abs=1e-9)
        assert sol == pytest.approx(1.0, abs=1e-9)

    def test_rotation_invariant(self):
        ring = _rect(16.0, 2.0)
        theta = np.radians(37.0)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        area, ecc, sol = polygon_shape_metrics(ring @ R.T)
        assert area == pytest.approx(32.0, abs=1e-6)
        assert ecc == pytest.approx(0.875, abs=1e-6)
        assert sol == pytest.approx(1.0, abs=1e-6)

    def test_l_shape_solidity_below_one(self):
        # 10×10 square minus 5×5 corner → area 75, hull area 87.5.
        ring = np.array([
            [0, 0], [10, 0], [10, 5], [5, 5], [5, 10], [0, 10],
        ], dtype=float)
        area, _ecc, sol = polygon_shape_metrics(ring)
        assert area == pytest.approx(75.0, abs=1e-9)
        assert sol == pytest.approx(75.0 / 87.5, abs=1e-9)


# ── classify_room_polygon: category + conservative is_common ────────────────

class TestClassifyRoomPolygon:
    def test_corridor_is_common(self):
        # 16×2 m: ecc 0.875 > 0.85, area 32 < 40 → hallway.
        label, category, conf, is_common = classify_room_polygon(_rect(16.0, 2.0))
        assert label == "Hallway / Corridor"
        assert category == "hallway"
        assert is_common is True
        # conf = 0.55 + ((0.875-0.85)/0.12)*0.38 + (1 - 32/45)*0.07
        assert conf == pytest.approx(
            0.55 + (0.025 / 0.12) * 0.38 + (1.0 - 32.0 / 45.0) * 0.07, abs=1e-9
        )

    def test_small_room_is_bathroom_and_common(self):
        # 2×2.5 m = 5 m² < 7 → bathroom.
        label, category, conf, is_common = classify_room_polygon(_rect(2.0, 2.5))
        assert label == "Bathroom / Storage"
        assert category == "bathroom"
        assert is_common is True
        assert conf == pytest.approx(0.55 + (7.0 - 5.0) / 14.0, abs=1e-9)

    def test_office_not_common(self):
        # 4×5 m = 20 m², regular shape → office.
        label, category, _conf, is_common = classify_room_polygon(_rect(4.0, 5.0))
        assert label == "Office / Meeting"
        assert category == "office"
        assert is_common is False

    def test_large_convex_room_not_flagged_common_from_shape(self):
        # 12×10 = 120 m² convex → classifier says "Open Plan / Lobby"
        # (category "common"), but shape alone cannot prove it's a lobby
        # rather than a tenant open-plan suite → is_common stays False.
        label, category, _conf, is_common = classify_room_polygon(_rect(12.0, 10.0))
        assert label == "Open Plan / Lobby"
        assert category == "common"
        assert is_common is False

    def test_category_sets(self):
        assert SHAPE_COMMON_CATEGORIES == {"hallway", "bathroom"}
        assert SHAPE_COMMON_CATEGORIES < COMMON_CATEGORIES
        assert "common" in COMMON_CATEGORIES


# ── scanplan delegates to the shared classifier ──────────────────────────────

class TestScanplanClassifierParity:
    CASES = [
        (32.0, 0.875, 1.0),
        (5.0, 0.2, 1.0),
        (20.0, 0.2, 1.0),
        (120.0, 0.17, 0.95),
        (25.0, 0.3, 0.5),
        (60.0, 0.4, 0.9),
    ]

    def test_classify_room_identical_to_shared(self):
        from app.pipeline.scanplan import _classify_room
        for area, ecc, sol in self.CASES:
            assert _classify_room(area, ecc, sol) == classify_room_shape(area, ecc, sol)


class TestScanplanRoomDictsCarryIsCommon:
    def _plan(self) -> "object":
        from app.pipeline.scanplan import SyntheticPlan
        # 16×10 shell with a divider at y=8 → 16×8 room + 16×2 corridor.
        segs = np.array([
            [[0, 0], [16, 0]],
            [[16, 0], [16, 10]],
            [[16, 10], [0, 10]],
            [[0, 10], [0, 0]],
            [[0, 8], [16, 8]],
        ], dtype=float)
        return SyntheticPlan(segments=segs, wall_count=5, bounds=(0.0, 0.0, 16.0, 10.0))

    def test_polygonize_rooms_have_is_common(self):
        from app.pipeline.scanplan import _rooms_from_polygonize
        rooms = _rooms_from_polygonize(self._plan())
        assert len(rooms) == 2
        by_area = sorted(rooms, key=lambda r: r["area_m2"])
        corridor, big = by_area
        assert corridor["area_m2"] == pytest.approx(32.0, abs=1e-6)
        assert corridor["category"] == "hallway"
        assert corridor["is_common"] is True
        assert big["area_m2"] == pytest.approx(128.0, abs=1e-6)
        # Big convex room: "common" category from shape, but NOT flagged
        # is_common (conservative shape-only mapping).
        assert big["is_common"] is False

    def test_single_room_fallback_has_is_common(self):
        from app.pipeline.scanplan import _building_as_single_room
        rooms = _building_as_single_room(self._plan())
        assert rooms[0]["is_common"] is False


# ── vectorize adapter: topology rooms classified + flagged ───────────────────

class TestVectorizeAdapterCommonAreas:
    def _build(self):
        from app.geometry.adapters import floor_geometry_from_vectorize
        from app.vectorize.topology import TopologyParams, build_topology

        # 16×10 shell, corridor wall at y=8, divider at x=8 below it →
        # two 8×8 rooms and one 16×2 corridor.
        centerlines = np.array([
            [[0, 0], [16, 0]],
            [[16, 0], [16, 10]],
            [[16, 10], [0, 10]],
            [[0, 10], [0, 0]],
            [[0, 8], [16, 8]],
            [[8, 0], [8, 8]],
        ], dtype=float)
        topo = build_topology(
            centerlines,
            params=TopologyParams(snap_tol_m=0.01, snapping_distance_m=0.05),
        )
        return floor_geometry_from_vectorize(topo, None)

    def test_corridor_room_flagged_common(self):
        floor = self._build()
        assert len(floor.rooms) == 3
        corridor = min(floor.rooms, key=lambda r: floor.room_area_m2(r.id))
        assert floor.room_area_m2(corridor.id) == pytest.approx(32.0, abs=1e-6)
        assert corridor.category == "hallway"
        assert corridor.is_common is True
        assert "Hallway / Corridor" in corridor.label

    def test_office_rooms_not_flagged_common(self):
        floor = self._build()
        offices = sorted(floor.rooms, key=lambda r: floor.room_area_m2(r.id))[1:]
        assert len(offices) == 2
        for room in offices:
            assert floor.room_area_m2(room.id) == pytest.approx(64.0, abs=1e-6)
            assert room.category == "office"
            assert room.is_common is False


# ── alignment adapter: explicit is_common override ───────────────────────────

class TestAlignmentAdapterIsCommonOverride:
    def _rooms(self, **first_extra):
        first = {
            "id": "room-001",
            "label": "Office 1",
            "category": "office",
            "polygon_2d": [(0.0, 0.0), (8.0, 0.0), (8.0, 10.0), (0.0, 10.0)],
            "centroid": (4.0, 5.0),
            "area_m2": 80.0,
        }
        first.update(first_extra)
        return [
            first,
            {
                "id": "room-002",
                "label": "Corridor",
                "category": "hallway",
                "polygon_2d": [(8.2, 0.0), (12.0, 0.0), (12.0, 10.0), (8.2, 10.0)],
                "centroid": (10.1, 5.0),
                "area_m2": 38.0,
            },
        ]

    def test_category_default(self):
        from app.geometry.adapters import floor_geometry_from_alignment_rooms
        floor = floor_geometry_from_alignment_rooms(self._rooms())
        assert floor.room_by_id("room-001").is_common is False
        assert floor.room_by_id("room-002").is_common is True

    def test_explicit_flag_wins_over_category(self):
        from app.geometry.adapters import floor_geometry_from_alignment_rooms
        # Operator marked the "office" as common (e.g. amenity lounge).
        floor = floor_geometry_from_alignment_rooms(self._rooms(is_common=True))
        assert floor.room_by_id("room-001").is_common is True

    def test_explicit_false_wins_over_common_category(self):
        from app.geometry.adapters import floor_geometry_from_alignment_rooms
        rooms = self._rooms()
        rooms[1]["is_common"] = False   # operator: this "corridor" is in-suite
        floor = floor_geometry_from_alignment_rooms(rooms)
        assert floor.room_by_id("room-002").is_common is False


# ── end-to-end: is_common feeds BOMA apportionment ───────────────────────────

class TestCommonAreaFeedsApportionment:
    ROOMS = [
        {
            "id": "room-001",
            "label": "Office 1",
            "category": "office",
            "polygon_2d": [(0.0, 0.0), (8.0, 0.0), (8.0, 10.0), (0.0, 10.0)],
            "centroid": (4.0, 5.0),
            "area_m2": 80.0,
        },
        {
            "id": "room-002",
            "label": "Corridor",
            "category": "hallway",
            "polygon_2d": [(8.2, 0.0), (12.0, 0.0), (12.0, 10.0), (8.2, 10.0)],
            "centroid": (10.1, 5.0),
            "area_m2": 38.0,
        },
    ]

    def test_flagged_corridor_inflates_rentable(self):
        from app.geometry.adapters import floor_geometry_from_alignment_rooms
        from app.measurement import get_ruleset
        floor = floor_geometry_from_alignment_rooms(self.ROOMS)
        result = get_ruleset("boma_2024_a").measure(floor)
        # One occupant suite; the corridor is apportioned onto it.
        assert len(result.suites) == 1
        assert result.floor_rentable.value_m2 > result.floor_usable.value_m2
        assert result.load_factor > 1.0

    def test_unflagged_corridor_is_just_another_suite(self):
        from app.geometry.adapters import floor_geometry_from_alignment_rooms
        from app.measurement import get_ruleset
        rooms = [dict(r) for r in self.ROOMS]
        rooms[1]["category"] = "office"   # not common any more
        floor = floor_geometry_from_alignment_rooms(rooms)
        result = get_ruleset("boma_2024_a").measure(floor)
        assert len(result.suites) == 2
        assert result.load_factor == pytest.approx(1.0, abs=1e-9)
