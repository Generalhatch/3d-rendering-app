"""Measurement-engine tests against hand-computed reference floors.

Every expected value here was worked out on paper BEFORE the engine was
written — see tests/reference_floors.py for the arithmetic.  Assertions are
exact (1e-6): the offset construction on rectilinear fixtures has no
legitimate source of error.
"""
from __future__ import annotations

import pytest

from app.measurement import get_ruleset, list_rulesets
from app.measurement.base import FloorMeasurement, Ruleset
from app.measurement.rulesets import (
    BomaOffice2024MethodA,
    BomaOffice2024MethodB,
    GrossArea,
    Rebny,
)
from tests.reference_floors import (
    CORRIDOR_COMMON,
    CORRIDOR_COMMON_REBNY,
    CORRIDOR_FLOOR_RENTABLE,
    CORRIDOR_FLOOR_USABLE,
    CORRIDOR_SUITE_RENTABLE,
    CORRIDOR_SUITE_RENTABLE_REBNY,
    CORRIDOR_SUITE_USABLE,
    DOMINANT_A_USABLE,
    DOMINANT_B_USABLE,
    GROSS_AREA,
    SHAFT_COMMON,
    SHAFT_FLOOR_RENTABLE,
    SHAFT_SUITE_RENTABLE,
    TWO_SUITE_A_RENTABLE_REBNY,
    TWO_SUITE_A_USABLE_BOMA,
    TWO_SUITE_B_RENTABLE_REBNY,
    TWO_SUITE_B_USABLE_BOMA,
    TWO_SUITE_FLOOR_RENTABLE_REBNY,
    TWO_SUITE_FLOOR_USABLE_BOMA,
    corridor_floor,
    two_suite_floor,
)


def _suite(m: FloorMeasurement, suite_id: str):
    return next(s for s in m.suites if s.suite_id == suite_id)


# ── Registry / interface ──────────────────────────────────────────────────────

class TestRegistry:
    def test_all_rulesets_registered(self):
        names = set(list_rulesets())
        assert {"boma_2024_a", "boma_2024_b", "rebny", "gross"} <= names

    def test_get_ruleset_returns_ruleset(self):
        for name in list_rulesets():
            assert isinstance(get_ruleset(name), Ruleset)

    def test_unknown_ruleset_raises(self):
        with pytest.raises(KeyError):
            get_ruleset("ipms-9000")


# ── BOMA Office 2024 ─────────────────────────────────────────────────────────

class TestBomaTwoSuite:
    """Two suites sharing one demising wall, no common area."""

    def test_usable_areas_exact(self):
        m = BomaOffice2024MethodA().measure(two_suite_floor())
        assert _suite(m, "suite_a").usable.value_m2 == pytest.approx(
            TWO_SUITE_A_USABLE_BOMA, abs=1e-6)
        assert _suite(m, "suite_b").usable.value_m2 == pytest.approx(
            TWO_SUITE_B_USABLE_BOMA, abs=1e-6)

    def test_no_common_means_rentable_equals_usable(self):
        m = BomaOffice2024MethodA().measure(two_suite_floor())
        for s in m.suites:
            assert s.rentable.value_m2 == pytest.approx(s.usable.value_m2, abs=1e-9)
        assert m.load_factor == pytest.approx(1.0, abs=1e-9)

    def test_floor_totals(self):
        m = BomaOffice2024MethodA().measure(two_suite_floor())
        assert m.floor_usable.value_m2 == pytest.approx(
            TWO_SUITE_FLOOR_USABLE_BOMA, abs=1e-6)
        assert m.floor_rentable.value_m2 == pytest.approx(
            TWO_SUITE_FLOOR_USABLE_BOMA, abs=1e-6)

    def test_every_number_stamped(self):
        m = BomaOffice2024MethodA().measure(two_suite_floor())
        stamped = [m.floor_usable, m.floor_rentable] + [
            v for s in m.suites for v in (s.usable, s.rentable)
        ]
        for v in stamped:
            assert v.standard == "BOMA Office 2024"
            assert v.method == "Method A"

    def test_exterior_wall_dominance(self):
        # >50% glass exterior: dominant portion 0.05 m inward of the
        # centerline instead of the 0.15 m finish face → bigger usable.
        m = BomaOffice2024MethodA().measure(
            two_suite_floor(dominant_inward_offset_m=0.05))
        assert _suite(m, "suite_a").usable.value_m2 == pytest.approx(
            DOMINANT_A_USABLE, abs=1e-6)
        assert _suite(m, "suite_b").usable.value_m2 == pytest.approx(
            DOMINANT_B_USABLE, abs=1e-6)


class TestBomaCorridorApportionment:
    """Common-area corridor apportioned pro-rata across two equal suites."""

    def test_usable_and_common(self):
        m = BomaOffice2024MethodA().measure(corridor_floor())
        assert _suite(m, "suite_a").usable.value_m2 == pytest.approx(
            CORRIDOR_SUITE_USABLE, abs=1e-6)
        assert _suite(m, "suite_b").usable.value_m2 == pytest.approx(
            CORRIDOR_SUITE_USABLE, abs=1e-6)
        assert m.common_area.value_m2 == pytest.approx(CORRIDOR_COMMON, abs=1e-6)

    def test_rentable_apportioned(self):
        m = BomaOffice2024MethodA().measure(corridor_floor())
        assert _suite(m, "suite_a").rentable.value_m2 == pytest.approx(
            CORRIDOR_SUITE_RENTABLE, abs=1e-6)
        assert _suite(m, "suite_b").rentable.value_m2 == pytest.approx(
            CORRIDOR_SUITE_RENTABLE, abs=1e-6)
        assert m.floor_usable.value_m2 == pytest.approx(CORRIDOR_FLOOR_USABLE, abs=1e-6)
        assert m.floor_rentable.value_m2 == pytest.approx(
            CORRIDOR_FLOOR_RENTABLE, abs=1e-6)

    def test_load_factor(self):
        m = BomaOffice2024MethodA().measure(corridor_floor())
        assert m.load_factor == pytest.approx(
            CORRIDOR_FLOOR_RENTABLE / CORRIDOR_FLOOR_USABLE, abs=1e-9)

    def test_rentable_sums_to_floor_rentable(self):
        m = BomaOffice2024MethodA().measure(corridor_floor())
        total = sum(s.rentable.value_m2 for s in m.suites)
        assert total == pytest.approx(m.floor_rentable.value_m2, abs=1e-9)

    def test_penetration_excluded_from_rentable(self):
        # 2x2 m elevator shaft inside the corridor: BOMA excludes major
        # vertical penetrations.
        m = BomaOffice2024MethodA().measure(corridor_floor(with_shaft=True))
        assert m.common_area.value_m2 == pytest.approx(SHAFT_COMMON, abs=1e-6)
        assert _suite(m, "suite_a").rentable.value_m2 == pytest.approx(
            SHAFT_SUITE_RENTABLE, abs=1e-6)
        assert m.floor_rentable.value_m2 == pytest.approx(
            SHAFT_FLOOR_RENTABLE, abs=1e-6)


class TestBomaMethodB:
    def test_single_load_factor_applied(self):
        # Building-wide single load factor of 10%.
        m = BomaOffice2024MethodB(building_load_factor=1.10).measure(two_suite_floor())
        assert _suite(m, "suite_a").rentable.value_m2 == pytest.approx(
            TWO_SUITE_A_USABLE_BOMA * 1.10, abs=1e-6)
        assert m.load_factor == pytest.approx(1.10, abs=1e-9)
        assert m.floor_rentable.value_m2 == pytest.approx(
            TWO_SUITE_FLOOR_USABLE_BOMA * 1.10, abs=1e-6)

    def test_method_stamp(self):
        m = BomaOffice2024MethodB(building_load_factor=1.10).measure(two_suite_floor())
        assert m.method == "Method B"
        assert _suite(m, "suite_a").rentable.method == "Method B"

    def test_defaults_to_floor_load_factor(self):
        # Without a supplied building factor, Method B falls back to the
        # floor-derived factor (single floor == whole building here).
        m = BomaOffice2024MethodB().measure(corridor_floor())
        assert m.load_factor == pytest.approx(
            CORRIDOR_FLOOR_RENTABLE / CORRIDOR_FLOOR_USABLE, abs=1e-9)
        assert _suite(m, "suite_a").rentable.value_m2 == pytest.approx(
            CORRIDOR_SUITE_RENTABLE, abs=1e-6)


# ── REBNY ─────────────────────────────────────────────────────────────────────

class TestRebny:
    def test_outside_face_measurement(self):
        m = Rebny().measure(two_suite_floor())
        assert _suite(m, "suite_a").rentable.value_m2 == pytest.approx(
            TWO_SUITE_A_RENTABLE_REBNY, abs=1e-6)
        assert _suite(m, "suite_b").rentable.value_m2 == pytest.approx(
            TWO_SUITE_B_RENTABLE_REBNY, abs=1e-6)

    def test_floor_rentable_equals_envelope(self):
        # Measuring every suite to the OUTSIDE face and splitting demising
        # walls at their centerline must tile the envelope exactly.
        m = Rebny().measure(two_suite_floor())
        assert m.floor_rentable.value_m2 == pytest.approx(
            TWO_SUITE_FLOOR_RENTABLE_REBNY, abs=1e-6)

    def test_common_apportioned_into_rentable(self):
        m = Rebny().measure(corridor_floor())
        assert m.common_area.value_m2 == pytest.approx(CORRIDOR_COMMON_REBNY, abs=1e-6)
        assert _suite(m, "suite_a").rentable.value_m2 == pytest.approx(
            CORRIDOR_SUITE_RENTABLE_REBNY, abs=1e-6)

    def test_penetrations_included(self):
        # REBNY does NOT deduct shafts — floor rentable is unchanged.
        m = Rebny().measure(corridor_floor(with_shaft=True))
        assert m.floor_rentable.value_m2 == pytest.approx(
            TWO_SUITE_FLOOR_RENTABLE_REBNY, abs=1e-6)

    def test_declared_loss_factor(self):
        m = Rebny(loss_factor=0.27).measure(two_suite_floor())
        assert _suite(m, "suite_a").usable.value_m2 == pytest.approx(
            TWO_SUITE_A_RENTABLE_REBNY * 0.73, abs=1e-6)
        assert m.load_factor == pytest.approx(1.0 / 0.73, abs=1e-9)

    def test_stamps(self):
        m = Rebny().measure(two_suite_floor())
        assert m.standard == "REBNY"
        for s in m.suites:
            assert s.rentable.standard == "REBNY"


# ── Gross ─────────────────────────────────────────────────────────────────────

class TestGross:
    def test_floor_gross_is_envelope_area(self):
        m = GrossArea().measure(two_suite_floor())
        assert m.floor_gross.value_m2 == pytest.approx(GROSS_AREA, abs=1e-6)

    def test_stamps(self):
        m = GrossArea().measure(two_suite_floor())
        assert m.floor_gross.standard == "Gross Area"

    def test_without_envelope_falls_back_to_room_union(self):
        floor = two_suite_floor()
        floor.envelope = None
        m = GrossArea().measure(floor)
        # Union of the two rooms measured to the outside face is exactly
        # the envelope rectangle.
        assert m.floor_gross.value_m2 == pytest.approx(GROSS_AREA, abs=1e-6)


# ── Cross-standard sanity ─────────────────────────────────────────────────────

class TestCrossStandard:
    def test_boma_usable_below_rebny_below_gross(self):
        floor = two_suite_floor()
        boma = BomaOffice2024MethodA().measure(floor)
        rebny = Rebny().measure(floor)
        gross = GrossArea().measure(floor)
        assert (
            boma.floor_usable.value_m2
            < rebny.floor_rentable.value_m2
            <= gross.floor_gross.value_m2 + 1e-9
        )

    def test_suite_ids_grouped_by_suite_id(self):
        # Two rooms with the same suite_id must merge into one suite whose
        # usable is the sum of both rooms (partition between them measured
        # to centerline from both sides — the wall is fully included).
        floor = corridor_floor()
        floor.room_by_id("suite_a").suite_id = "tenant-1"
        floor.room_by_id("suite_b").suite_id = "tenant-1"
        m = BomaOffice2024MethodA().measure(floor)
        assert len(m.suites) == 1
        assert m.suites[0].suite_id == "tenant-1"
        assert m.suites[0].usable.value_m2 == pytest.approx(
            2 * CORRIDOR_SUITE_USABLE, abs=1e-6)
