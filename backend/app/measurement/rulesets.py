"""Concrete measurement rulesets: BOMA Office 2024 (A + B), REBNY, Gross.

All rulesets share the single calculation path in
:mod:`app.measurement.base` and differ only in boundary rules, penetration
handling, and apportionment:

======================  ==================  ============  ===========  ============
Ruleset                 Exterior boundary   Demising      Partition    Penetrations
======================  ==================  ============  ===========  ============
BOMA Office 2024 (A/B)  dominant portion    centerline    centerline   deducted
REBNY                   OUTSIDE face        centerline    centerline   included
Gross Area              OUTSIDE face        centerline    centerline   included
======================  ==================  ============  ===========  ============

IPMS note: the FloorGeometry model stores both wall faces + thickness, so
IPMS 1 (external face) and IPMS 2 (internal dominant face) are future
rulesets over the same geometry — no extractor changes needed.
"""
from __future__ import annotations

from typing import Optional

from shapely.ops import unary_union

from ..geometry.model import BoundaryRules, BoundaryTarget, FloorGeometry
from .base import (
    AreaValue,
    FloorMeasurement,
    Ruleset,
    SuiteMeasurement,
    group_suites,
    measure_rooms,
    suite_label,
)

BOMA_RULES = BoundaryRules(
    exterior=BoundaryTarget.DOMINANT_PORTION,
    demising=BoundaryTarget.CENTERLINE,
    partition=BoundaryTarget.CENTERLINE,
)

OUTSIDE_FACE_RULES = BoundaryRules(
    exterior=BoundaryTarget.OUTSIDE_FACE,
    demising=BoundaryTarget.CENTERLINE,
    partition=BoundaryTarget.CENTERLINE,
)


# ── BOMA Office 2024 ─────────────────────────────────────────────────────────

class _BomaOffice2024(Ruleset):
    """Shared BOMA computation; Methods A and B differ only in where the
    load factor comes from."""

    standard = "BOMA Office 2024"
    method = ""  # set by subclasses

    def __init__(self, building_load_factor: Optional[float] = None) -> None:
        # Method B: single building-wide load factor.  None → derive from
        # this floor (equivalent to Method A on a single-floor building).
        self.building_load_factor = building_load_factor

    def _area(self, value: float, component: str) -> AreaValue:
        return AreaValue(value, self.standard, self.method, component)

    def measure(self, floor: FloorGeometry) -> FloorMeasurement:
        polys = measure_rooms(floor, BOMA_RULES, deduct_penetrations=True)
        suites = group_suites(floor)

        usable_by_suite = {
            sid: float(sum(polys[r.id].area for r in rooms))
            for sid, rooms in suites.items()
        }
        floor_usable = float(sum(usable_by_suite.values()))
        common_total = float(sum(
            polys[r.id].area for r in floor.rooms if r.is_common
        ))

        notes: list[str] = []
        if floor.penetrations:
            notes.append(
                f"{len(floor.penetrations)} major vertical penetration(s) "
                f"deducted ({floor.penetration_area_m2():.2f} m² gross)"
            )

        floor_rentable_pool = floor_usable + common_total
        derived_factor = (
            floor_rentable_pool / floor_usable if floor_usable > 0 else 1.0
        )
        if self.building_load_factor is not None:
            load_factor = float(self.building_load_factor)
            notes.append("single building-wide load factor supplied by operator")
        else:
            load_factor = derived_factor

        suite_measurements: list[SuiteMeasurement] = []
        for sid, rooms in suites.items():
            usable = usable_by_suite[sid]
            rentable = usable * load_factor
            suite_measurements.append(SuiteMeasurement(
                suite_id=sid,
                label=suite_label(rooms, sid),
                room_ids=[r.id for r in rooms],
                usable=self._area(usable, "usable"),
                rentable=self._area(rentable, "rentable"),
            ))

        floor_rentable = float(sum(s.rentable.value_m2 for s in suite_measurements))
        return FloorMeasurement(
            standard=self.standard,
            method=self.method,
            floor_id=floor.floor_id,
            suites=suite_measurements,
            common_area=self._area(common_total, "common"),
            floor_usable=self._area(floor_usable, "usable"),
            floor_rentable=self._area(floor_rentable, "rentable"),
            load_factor=load_factor,
            notes=notes,
        )


class BomaOffice2024MethodA(_BomaOffice2024):
    """BOMA Office 2024, Method A — load factor derived per floor and
    applied pro-rata on usable area."""
    method = "Method A"

    def __init__(self) -> None:
        super().__init__(building_load_factor=None)


class BomaOffice2024MethodB(_BomaOffice2024):
    """BOMA Office 2024, Method B — a single building-wide load factor
    applied to every suite on every floor."""
    method = "Method B"


# ── REBNY ─────────────────────────────────────────────────────────────────────

class Rebny(Ruleset):
    """REBNY measurement: everything to the OUTSIDE face of exterior walls,
    penetrations included, common area folded into rentable, loss factor
    reported per floor (declared by the landlord or derived)."""

    standard = "REBNY"
    method = ""

    def __init__(self, loss_factor: Optional[float] = None) -> None:
        # Declared per-floor loss factor (0..1).  None → derive from the
        # common share: loss = 1 - usable / rentable.
        self.loss_factor = loss_factor

    def _area(self, value: float, component: str) -> AreaValue:
        return AreaValue(value, self.standard, self.method, component)

    def measure(self, floor: FloorGeometry) -> FloorMeasurement:
        polys = measure_rooms(floor, OUTSIDE_FACE_RULES, deduct_penetrations=False)
        suites = group_suites(floor)

        measured_by_suite = {
            sid: float(sum(polys[r.id].area for r in rooms))
            for sid, rooms in suites.items()
        }
        suite_total = float(sum(measured_by_suite.values()))
        common_total = float(sum(
            polys[r.id].area for r in floor.rooms if r.is_common
        ))
        floor_rentable = suite_total + common_total

        notes: list[str] = []
        if floor.penetrations:
            notes.append(
                "penetrations are NOT deducted under REBNY "
                f"({floor.penetration_area_m2():.2f} m² retained in rentable)"
            )

        suite_measurements: list[SuiteMeasurement] = []
        floor_usable = 0.0
        for sid, rooms in suites.items():
            measured = measured_by_suite[sid]
            share = measured / suite_total if suite_total > 0 else 0.0
            rentable = measured + common_total * share
            if self.loss_factor is not None:
                usable = rentable * (1.0 - self.loss_factor)
            else:
                usable = measured
            floor_usable += usable
            suite_measurements.append(SuiteMeasurement(
                suite_id=sid,
                label=suite_label(rooms, sid),
                room_ids=[r.id for r in rooms],
                usable=self._area(usable, "usable"),
                rentable=self._area(rentable, "rentable"),
            ))

        if self.loss_factor is not None:
            load_factor = 1.0 / (1.0 - self.loss_factor)
            notes.append(f"declared loss factor {self.loss_factor:.2%}")
        else:
            load_factor = (
                floor_rentable / floor_usable if floor_usable > 0 else 1.0
            )
            derived_loss = 1.0 - (floor_usable / floor_rentable) if floor_rentable else 0.0
            notes.append(f"derived loss factor {derived_loss:.2%}")

        return FloorMeasurement(
            standard=self.standard,
            method=self.method,
            floor_id=floor.floor_id,
            suites=suite_measurements,
            common_area=self._area(common_total, "common"),
            floor_usable=self._area(floor_usable, "usable"),
            floor_rentable=self._area(floor_rentable, "rentable"),
            load_factor=load_factor,
            notes=notes,
        )


# ── Gross Area ────────────────────────────────────────────────────────────────

class GrossArea(Ruleset):
    """Simple gross floor area: the envelope (outside face of exterior
    walls).  Falls back to the union of rooms measured to the outside face
    when no envelope polygon was extracted."""

    standard = "Gross Area"
    method = ""

    def _area(self, value: float, component: str) -> AreaValue:
        return AreaValue(value, self.standard, self.method, component)

    def measure(self, floor: FloorGeometry) -> FloorMeasurement:
        polys = measure_rooms(floor, OUTSIDE_FACE_RULES, deduct_penetrations=False)
        suites = group_suites(floor)

        notes: list[str] = []
        if floor.envelope is not None:
            floor_gross = floor.envelope_area_m2()
        else:
            union = unary_union(list(polys.values())) if polys else None
            floor_gross = float(union.area) if union is not None else 0.0
            notes.append("no envelope polygon — gross derived from room union")

        suite_measurements: list[SuiteMeasurement] = []
        for sid, rooms in suites.items():
            measured = float(sum(polys[r.id].area for r in rooms))
            suite_measurements.append(SuiteMeasurement(
                suite_id=sid,
                label=suite_label(rooms, sid),
                room_ids=[r.id for r in rooms],
                usable=self._area(measured, "gross"),
                rentable=self._area(measured, "gross"),
                gross=self._area(measured, "gross"),
            ))

        common_total = float(sum(
            polys[r.id].area for r in floor.rooms if r.is_common
        ))
        return FloorMeasurement(
            standard=self.standard,
            method=self.method,
            floor_id=floor.floor_id,
            suites=suite_measurements,
            common_area=self._area(common_total, "common"),
            floor_usable=self._area(floor_gross, "gross"),
            floor_rentable=self._area(floor_gross, "gross"),
            floor_gross=self._area(floor_gross, "gross"),
            load_factor=1.0,
            notes=notes,
        )
