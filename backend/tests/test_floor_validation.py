"""Tests for floor-closure validation (Phase 2).

Fixtures reuse the hand-computed reference floors from
``tests/reference_floors.py`` — every expected number below is derivable on
paper from that module's docstring geometry:

- Envelope: 20.3 x 10.3 = 209.09 m² (outside face of 0.30 m exterior walls
  whose centerlines trace the 20 x 10 shell).
- Rooms are drawn on wall centerlines, so rooms + wall footprints tile the
  envelope completely except the four 0.15 x 0.15 m corner notches left by
  the flat-capped wall footprints (4 x 0.0225 = 0.09 m²).
- Removing suite_b from the corridor floor leaves an uncovered rectangle
  x ∈ [12.1, 19.85] × y ∈ [0.15, 9.85] = 7.75 x 9.7 = 75.175 m².
"""
from __future__ import annotations

import numpy as np
import pytest

from app.geometry.model import (
    BoundaryBasis,
    FloorGeometry,
    Room,
    Wall,
    WallClass,
    WallFaceRef,
)
from app.geometry.validation import (
    FloorValidation,
    validate_floor_closure,
)
from tests.reference_floors import (
    EXTERIOR_T,
    GROSS_AREA,
    _ring,
    _wall,
    corridor_floor,
    two_suite_floor,
)


def _corridor_missing_suite_b() -> FloorGeometry:
    """The corridor reference floor with suite_b deleted — a 75.175 m²
    region of the envelope has no room covering it."""
    floor = corridor_floor()
    rooms = [r for r in floor.rooms if r.id != "suite_b"]
    return FloorGeometry(
        floor_id=floor.floor_id,
        walls=floor.walls,
        rooms=rooms,
        penetrations=floor.penetrations,
        envelope=floor.envelope,
    )


# ── Closed reference floors pass every check ─────────────────────────────────

class TestClosedFloorPasses:
    def test_two_suite_floor_passes(self):
        v = validate_floor_closure(two_suite_floor())
        assert v.passed is True
        assert v.area_check_passed is True
        assert v.perimeter_closed is True
        assert v.unscanned_gaps == []
        assert v.warnings == []

    def test_two_suite_envelope_area_exact(self):
        v = validate_floor_closure(two_suite_floor())
        assert v.envelope_area_m2 == pytest.approx(GROSS_AREA, abs=1e-6)

    def test_two_suite_room_totals_exact(self):
        # Rooms are drawn on centerlines: suite_a 8x10 + suite_b 12x10.
        v = validate_floor_closure(two_suite_floor())
        assert v.rooms_total_m2 == pytest.approx(200.0, abs=1e-6)
        assert v.suites_total_m2 == pytest.approx(200.0, abs=1e-6)
        assert v.common_total_m2 == pytest.approx(0.0, abs=1e-9)

    def test_coverage_ratio_near_one(self):
        # Only the four 0.15 x 0.15 corner notches are uncovered:
        # ratio = (209.09 - 0.09) / 209.09.
        v = validate_floor_closure(two_suite_floor())
        assert v.coverage_ratio == pytest.approx(
            (GROSS_AREA - 4 * 0.15 * 0.15) / GROSS_AREA, abs=1e-6,
        )

    def test_corridor_floor_passes_and_splits_common(self):
        v = validate_floor_closure(corridor_floor())
        assert v.passed is True
        # suite_a 80 + suite_b 80, corridor 4x10 = 40 (centerline rings).
        assert v.suites_total_m2 == pytest.approx(160.0, abs=1e-6)
        assert v.common_total_m2 == pytest.approx(40.0, abs=1e-6)
        assert v.rooms_total_m2 == pytest.approx(200.0, abs=1e-6)

    def test_corridor_floor_with_shaft_passes(self):
        # The elevator shaft is a penetration — covered floorplate, not an
        # unscanned gap.
        v = validate_floor_closure(corridor_floor(with_shaft=True))
        assert v.passed is True
        assert v.unscanned_gaps == []


# ── Unscanned-gap detection ──────────────────────────────────────────────────

class TestUnscannedGaps:
    def test_missing_suite_flagged_as_unscanned(self):
        v = validate_floor_closure(_corridor_missing_suite_b())
        assert v.passed is False
        assert len(v.unscanned_gaps) == 1
        gap = v.unscanned_gaps[0]
        # x ∈ [12.1, 19.85] × y ∈ [0.15, 9.85] = 7.75 × 9.7 m.
        assert gap.area_m2 == pytest.approx(75.175, abs=1e-6)
        assert 12.1 < gap.centroid[0] < 19.85
        assert 0.15 < gap.centroid[1] < 9.85

    def test_missing_suite_fails_area_check(self):
        v = validate_floor_closure(_corridor_missing_suite_b())
        assert v.area_check_passed is False
        # covered = 209.09 - 75.175 - 4 corner notches (0.09).
        expected_ratio = (GROSS_AREA - 75.175 - 0.09) / GROSS_AREA
        assert v.coverage_ratio == pytest.approx(expected_ratio, abs=1e-6)

    def test_gap_is_structured_output_not_a_log_line(self):
        v = validate_floor_closure(_corridor_missing_suite_b())
        d = v.to_json_dict()
        assert d["passed"] is False
        assert len(d["unscanned_gaps"]) == 1
        gap = d["unscanned_gaps"][0]
        assert gap["kind"] == "unscanned"
        assert gap["area_m2"] == pytest.approx(75.175, abs=1e-4)
        assert len(gap["polygon"]) >= 3
        codes = {w["code"] for w in d["warnings"]}
        assert "unscanned_gaps" in codes
        assert "envelope_area_mismatch" in codes

    def test_small_gaps_below_threshold_ignored(self):
        # With the threshold raised above the missing-suite area the gap
        # disappears from the list (but the area check still fails).
        v = validate_floor_closure(
            _corridor_missing_suite_b(), min_gap_area_m2=80.0,
        )
        assert v.unscanned_gaps == []
        assert v.area_check_passed is False

    def test_default_threshold_is_two_m2(self):
        from app.geometry.validation import DEFAULT_MIN_GAP_AREA_M2
        assert DEFAULT_MIN_GAP_AREA_M2 == 2.0


# ── Exterior perimeter closure ───────────────────────────────────────────────

class TestPerimeterClosure:
    def test_reference_floor_perimeter_closed(self):
        v = validate_floor_closure(two_suite_floor())
        assert v.perimeter_closed is True
        assert v.perimeter_open_endpoints == []

    def test_missing_exterior_wall_detected(self):
        """Delete the right exterior wall: the perimeter no longer closes
        and the two orphaned corners are reported."""
        floor = two_suite_floor()
        walls = [w for w in floor.walls if w.id != "w_right"]
        rooms = []
        for r in floor.rooms:
            refs = [
                WallFaceRef(None) if ref.wall_id == "w_right" else ref
                for ref in r.wall_refs
            ]
            rooms.append(Room(
                id=r.id, boundary=r.boundary, wall_refs=refs,
                boundary_basis=r.boundary_basis, label=r.label,
            ))
        broken = FloorGeometry(
            floor_id="broken", walls=walls, rooms=rooms, envelope=floor.envelope,
        )
        v = validate_floor_closure(broken)
        assert v.perimeter_closed is False
        assert len(v.perimeter_open_endpoints) == 2
        opened = {tuple(np.round(p, 6)) for p in v.perimeter_open_endpoints}
        assert opened == {(20.0, 0.0), (20.0, 10.0)}
        codes = {w["code"] for w in v.warnings}
        assert "open_perimeter" in codes

    def test_endpoints_within_snap_tolerance_still_close(self):
        """A 5 cm construction gap at one corner is within the default
        10 cm snap tolerance — the perimeter still counts as closed."""
        walls = [
            _wall("e1", (0, 0), (10, 0), EXTERIOR_T, WallClass.EXTERIOR),
            _wall("e2", (10, 0.05), (10, 10), EXTERIOR_T, WallClass.EXTERIOR),
            _wall("e3", (10, 10), (0, 10), EXTERIOR_T, WallClass.EXTERIOR),
            _wall("e4", (0, 10), (0, 0), EXTERIOR_T, WallClass.EXTERIOR),
        ]
        room = Room(
            id="r1", boundary=_ring(0, 0, 10, 10),
            wall_refs=[WallFaceRef("e1"), WallFaceRef("e2"),
                       WallFaceRef("e3"), WallFaceRef("e4")],
            boundary_basis=BoundaryBasis.CENTERLINE,
        )
        floor = FloorGeometry(
            floor_id="snap", walls=walls, rooms=[room],
            envelope=_ring(-0.15, -0.15, 10.15, 10.15),
        )
        v = validate_floor_closure(floor)
        assert v.perimeter_closed is True


# ── Degenerate inputs stay honest ────────────────────────────────────────────

class TestDegenerateInputs:
    def test_no_envelope_reports_not_checkable(self):
        floor = two_suite_floor()
        no_env = FloorGeometry(
            floor_id="no-env", walls=floor.walls, rooms=floor.rooms,
            envelope=None,
        )
        v = validate_floor_closure(no_env)
        assert v.envelope_area_m2 is None
        assert v.coverage_ratio is None
        assert v.area_check_passed is None   # honest: unknown, not "passed"
        assert v.unscanned_gaps == []
        codes = {w["code"] for w in v.warnings}
        assert "no_envelope" in codes
        # Perimeter is still checkable from the exterior walls.
        assert v.perimeter_closed is True

    def test_envelope_but_no_rooms_fails(self):
        floor = FloorGeometry(
            floor_id="empty", walls=[], rooms=[],
            envelope=_ring(-0.15, -0.15, 20.15, 10.15),
        )
        v = validate_floor_closure(floor)
        assert v.passed is False
        assert v.coverage_ratio == 0.0
        assert v.area_check_passed is False
        codes = {w["code"] for w in v.warnings}
        assert "no_rooms" in codes

    def test_json_round_trip_is_json_safe(self):
        import json
        v = validate_floor_closure(_corridor_missing_suite_b())
        text = json.dumps(v.to_json_dict())
        parsed = json.loads(text)
        assert parsed["unscanned_gaps"][0]["area_m2"] == pytest.approx(75.175)

    def test_passed_property_semantics(self):
        # None (not checkable) does not fail the floor; False does.
        v = FloorValidation(
            envelope_area_m2=None, rooms_total_m2=10.0,
            suites_total_m2=10.0, common_total_m2=0.0,
            coverage_ratio=None, area_check_passed=None,
            perimeter_closed=None,
        )
        assert v.passed is True
        v.perimeter_closed = False
        assert v.passed is False
