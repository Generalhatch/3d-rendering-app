"""Measurement engine core: the Ruleset interface + shared calculation path.

Design rule: there is exactly ONE way any standard turns geometry into a
number — :func:`measure_rooms` below, which delegates to
:meth:`app.geometry.model.FloorGeometry.measured_room_polygon`.  Rulesets
differ only in the :class:`~app.geometry.model.BoundaryRules` they pass,
whether penetrations are deducted, and how common area is apportioned.
Every emitted number is an :class:`AreaValue` stamped with its standard
and method.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from ..geometry.model import BoundaryRules, FloorGeometry, Room

SQFT_PER_M2 = 10.7639104167097223


# ── Value objects ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AreaValue:
    """One measured area, stamped with its provenance.

    No bare floats leave the measurement engine — a number without its
    standard + method attached is not a deliverable.
    """
    value_m2: float
    standard: str            # e.g. "BOMA Office 2024"
    method: str              # e.g. "Method A" ("" when the standard has none)
    component: str           # usable | rentable | gross | common

    @property
    def value_sqft(self) -> float:
        return self.value_m2 * SQFT_PER_M2

    def to_json_dict(self) -> dict:
        return {
            "value_m2": round(self.value_m2, 4),
            "value_sqft": round(self.value_sqft, 2),
            "standard": self.standard,
            "method": self.method,
            "component": self.component,
        }


@dataclass
class SuiteMeasurement:
    """Per-suite results.  A suite is one or more rooms sharing a suite_id."""
    suite_id: str
    label: str
    room_ids: list[str]
    usable: AreaValue
    rentable: AreaValue
    gross: Optional[AreaValue] = None

    def to_json_dict(self) -> dict:
        out = {
            "suite_id": self.suite_id,
            "label": self.label,
            "room_ids": list(self.room_ids),
            "usable": self.usable.to_json_dict(),
            "rentable": self.rentable.to_json_dict(),
        }
        if self.gross is not None:
            out["gross"] = self.gross.to_json_dict()
        return out


@dataclass
class FloorMeasurement:
    """Full measurement of one floor under one standard + method."""
    standard: str
    method: str
    floor_id: str
    suites: list[SuiteMeasurement]
    common_area: AreaValue
    floor_usable: AreaValue
    floor_rentable: AreaValue
    floor_gross: Optional[AreaValue] = None
    load_factor: float = 1.0          # rentable / usable
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        out = {
            "standard": self.standard,
            "method": self.method,
            "floor_id": self.floor_id,
            "load_factor": round(self.load_factor, 6),
            "common_area": self.common_area.to_json_dict(),
            "floor_usable": self.floor_usable.to_json_dict(),
            "floor_rentable": self.floor_rentable.to_json_dict(),
            "suites": [s.to_json_dict() for s in self.suites],
            "notes": list(self.notes),
        }
        if self.floor_gross is not None:
            out["floor_gross"] = self.floor_gross.to_json_dict()
        return out


# ── Ruleset interface ─────────────────────────────────────────────────────────

class Ruleset(ABC):
    """A measurement standard implementation."""

    #: Human-readable standard name stamped on every number.
    standard: str = ""
    #: Method within the standard ("" when not applicable).
    method: str = ""

    @abstractmethod
    def measure(self, floor: FloorGeometry) -> FloorMeasurement:
        """Measure the floor.  Must route all polygon math through
        :func:`measure_rooms` (the one calculation path)."""


# ── The one calculation path ──────────────────────────────────────────────────

def measure_rooms(
    floor: FloorGeometry,
    rules: BoundaryRules,
    deduct_penetrations: bool,
) -> dict[str, ShapelyPolygon]:
    """Measured polygon for every room under ``rules``.

    When ``deduct_penetrations`` is True, major vertical penetrations
    (shafts / stairs / elevators) are geometrically subtracted from any
    room polygon they overlap.
    """
    pen_union = None
    if deduct_penetrations and floor.penetrations:
        pen_union = unary_union([
            ShapelyPolygon(p.polygon) for p in floor.penetrations
        ])

    out: dict[str, ShapelyPolygon] = {}
    for room in floor.rooms:
        poly = floor.measured_room_polygon(room.id, rules)
        if pen_union is not None and poly.intersects(pen_union):
            poly = poly.difference(pen_union)
        out[room.id] = poly
    return out


def group_suites(floor: FloorGeometry) -> dict[str, list[Room]]:
    """Group occupant (non-common) rooms into suites by suite_id.

    Rooms without a suite_id are their own suite.  Returns
    ``{suite_id: [rooms]}`` in stable sorted order.
    """
    suites: dict[str, list[Room]] = {}
    for room in floor.rooms:
        if room.is_common:
            continue
        sid = room.suite_id or room.id
        suites.setdefault(sid, []).append(room)
    return dict(sorted(suites.items()))


def suite_label(rooms: list[Room], suite_id: str) -> str:
    labels = [r.label for r in rooms if r.label]
    return labels[0] if len(labels) == 1 else (", ".join(labels) or suite_id)
