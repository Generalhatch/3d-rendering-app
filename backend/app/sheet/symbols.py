"""Architectural symbols for the sheet renderer (Phase 3.5).

- **Door swings** from :class:`Opening` records (kind ``"door"``): leaf
  line + quarter-arc swing.  Double doors (two leaves meeting mid-opening)
  when the opening is wider than :data:`DOUBLE_DOOR_MIN_WIDTH_M`.  The
  swing-direction default comes from a room-aware heuristic
  (:func:`default_swing`) and is overridable per opening via
  ``sheet_overrides.json`` (``door_swings``), so the operator can flip a
  door the scanner couldn't observe.
- **Stair symbols** from :class:`Penetration` records (kinds in
  :data:`STAIR_KINDS`): tread lines perpendicular to the run's long axis,
  plus a direction arrow up the middle.
- **Elevator symbols** (kinds in :data:`ELEVATOR_KINDS`): the classic
  X-box — the shaft rectangle with both diagonals.

All geometry is computed in WORLD metres and mapped to paper through the
render transform, so tests can assert exact coordinates.  Presentation
only — nothing here feeds measurement.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from ..geometry.model import FloorGeometry, Opening, Penetration

#: Openings at least this wide are drawn as DOUBLE doors (two leaves).
DOUBLE_DOOR_MIN_WIDTH_M = 1.4

#: Penetration kinds drawn with the stair symbol / the elevator X-box.
STAIR_KINDS = ("stair", "stairs", "stairwell", "stairway")
ELEVATOR_KINDS = ("elevator", "elevator_shaft", "lift")

#: Stair tread spacing along the run (world metres).
STAIR_TREAD_SPACING_M = 0.28

_SYMBOL_COLOR = "#767676"
_SYMBOL_STROKE_MM = 0.15


@dataclass
class DoorLeaf:
    """One door leaf + its quarter-arc swing, in world coordinates."""
    hinge: np.ndarray        # (2,) hinge point (on the wall line)
    latch: np.ndarray        # (2,) closed-position tip (on the wall line)
    tip_open: np.ndarray     # (2,) open-position tip (leaf line end)
    radius_m: float

    @property
    def sweep_ccw(self) -> bool:
        """True when the arc from ``tip_open`` to ``latch`` runs CCW in
        world coordinates (y-up)."""
        a = self.tip_open - self.hinge
        b = self.latch - self.hinge
        return float(a[0] * b[1] - a[1] * b[0]) > 0.0


def _opening_frame(op: Opening) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(start, unit axis, unit left-normal, width) of an opening segment."""
    seg = np.asarray(op.segment, dtype=float).reshape(2, 2)
    d = seg[1] - seg[0]
    width = float(np.linalg.norm(d))
    if width < 1e-9:
        raise ValueError(f"opening {op.id}: degenerate segment")
    u = d / width
    n_left = np.array([-u[1], u[0]])     # +90° CCW of the segment direction
    return seg[0], u, n_left, width


def default_swing(op: Opening, floor: FloorGeometry) -> dict:
    """Heuristic swing for a door the operator hasn't overridden.

    Doors swing INTO the room being entered — typically the smaller room
    off a corridor.  Probe a point half a metre to each side of the
    opening midpoint: sides that land inside a room are candidates; among
    candidates prefer non-common rooms, then the smaller room.  Falls back
    to the left side when neither probe hits a room.  Hinge defaults to
    the segment start.
    """
    start, u, n_left, width = _opening_frame(op)
    mid = start + u * (width / 2.0)
    wall = floor.wall_by_id(op.wall_id)
    reach = (wall.thickness_m if wall is not None else 0.2) / 2.0 + 0.5

    candidates: list[tuple[bool, float, str]] = []   # (is_common, area, side)
    for side, n in (("left", n_left), ("right", -n_left)):
        probe = ShapelyPoint(mid + n * reach)
        for room in floor.rooms:
            poly = ShapelyPolygon(np.asarray(room.boundary, dtype=float))
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.contains(probe):
                candidates.append(
                    (room.is_common, floor.room_area_m2(room.id), side)
                )
                break

    if candidates:
        candidates.sort()                # non-common first, then smallest area
        side = candidates[0][2]
    else:
        side = "left"
    return {"hinge": "start", "side": side}


def door_leaves(
    op: Opening,
    floor: FloorGeometry,
    override: Optional[dict] = None,
) -> list[DoorLeaf]:
    """The leaf/arc geometry for one door opening (1 leaf, or 2 if double).

    ``override``: ``{"hinge": "start"|"end", "side": "left"|"right"}``
    ("left" = the +90° CCW side of the opening segment's direction).
    """
    swing = dict(default_swing(op, floor))
    if override:
        swing.update({k: v for k, v in override.items() if v})

    start, u, n_left, width = _opening_frame(op)
    end = start + u * width
    n = n_left if swing["side"] == "left" else -n_left

    if width > DOUBLE_DOOR_MIN_WIDTH_M:
        half = width / 2.0
        mid = start + u * half
        return [
            DoorLeaf(hinge=start, latch=mid, tip_open=start + n * half,
                     radius_m=half),
            DoorLeaf(hinge=end, latch=mid, tip_open=end + n * half,
                     radius_m=half),
        ]

    if swing.get("hinge") == "end":
        hinge, latch = end, start
    else:
        hinge, latch = start, end
    return [DoorLeaf(hinge=hinge, latch=latch, tip_open=hinge + n * width,
                     radius_m=width)]


# ── Stairs / elevators ────────────────────────────────────────────────────────

def _rect_corners(polygon: np.ndarray) -> np.ndarray:
    """(4, 2) rectangle corners for a penetration footprint.

    Quads are used as-is (avoids shapely oriented-envelope warnings on
    exact squares); other polygons fall back to the minimum rotated
    rectangle.
    """
    pts = np.asarray(polygon, dtype=float)
    if len(pts) == 4:
        return pts
    shp = ShapelyPolygon(pts)
    if not shp.is_valid:
        shp = shp.buffer(0)
    return np.asarray(
        shp.minimum_rotated_rectangle.exterior.coords[:-1], dtype=float,
    )


def _min_rect_frame(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """(centre, long-axis unit, short-axis unit, long len, short len) of the
    polygon's bounding rectangle frame."""
    corners = _rect_corners(polygon)
    e0 = corners[1] - corners[0]
    e1 = corners[2] - corners[1]
    l0, l1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    if l0 >= l1:
        long_u, long_len, short_u, short_len = e0 / l0, l0, e1 / l1, l1
    else:
        long_u, long_len, short_u, short_len = e1 / l1, l1, e0 / l0, l0
    centre = corners.mean(axis=0)
    return centre, long_u, short_u, long_len, short_len


def stair_treads(pen: Penetration) -> list[np.ndarray]:
    """Tread lines: segments across the short axis, spaced along the long
    axis at :data:`STAIR_TREAD_SPACING_M` (world metres)."""
    centre, long_u, short_u, long_len, short_len = _min_rect_frame(pen.polygon)
    n_treads = int(long_len / STAIR_TREAD_SPACING_M)
    if n_treads < 1:
        return []
    treads: list[np.ndarray] = []
    for k in range(1, n_treads + 1):
        t = -long_len / 2.0 + k * STAIR_TREAD_SPACING_M
        if t >= long_len / 2.0 - 1e-9:
            break
        c = centre + long_u * t
        treads.append(np.array([
            c - short_u * (short_len / 2.0),
            c + short_u * (short_len / 2.0),
        ]))
    return treads


def stair_direction_arrow(pen: Penetration) -> np.ndarray:
    """(2, 2) arrow shaft up the middle of the run (long axis)."""
    centre, long_u, _short_u, long_len, _short_len = _min_rect_frame(pen.polygon)
    half = max(long_len / 2.0 - STAIR_TREAD_SPACING_M, long_len / 4.0)
    return np.array([centre - long_u * half, centre + long_u * half])


def elevator_diagonals(pen: Penetration) -> list[np.ndarray]:
    """The two diagonals of the shaft's bounding rectangle (X-box)."""
    corners = _rect_corners(pen.polygon)
    return [
        np.array([corners[0], corners[2]]),
        np.array([corners[1], corners[3]]),
    ]


# ── SVG rendering ─────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def render_symbols(
    root: ET.Element,
    floor: FloorGeometry,
    to_mm: Callable[[float, float], tuple[float, float]],
    mm_per_m: float,
    door_swings: dict[str, dict] | None = None,
) -> None:
    """Append the ``#symbols`` group: door swings + stair/elevator symbols."""
    door_swings = door_swings or {}
    g = ET.SubElement(root, "g")
    g.set("id", "symbols")

    # ── Door swings ────────────────────────────────────────────────────────
    for op in floor.openings:
        if op.kind != "door":
            continue
        g_d = ET.SubElement(g, "g")
        g_d.set("class", "door-swing")
        g_d.set("data-opening-id", op.id)
        for leaf in door_leaves(op, floor, door_swings.get(op.id)):
            hx, hy = to_mm(float(leaf.hinge[0]), float(leaf.hinge[1]))
            tx, ty = to_mm(float(leaf.tip_open[0]), float(leaf.tip_open[1]))
            lx, ly = to_mm(float(leaf.latch[0]), float(leaf.latch[1]))
            r_mm = leaf.radius_m * mm_per_m
            # Leaf line: hinge → open tip.
            leaf_el = ET.SubElement(g_d, "line")
            leaf_el.set("x1", _fmt(hx))
            leaf_el.set("y1", _fmt(hy))
            leaf_el.set("x2", _fmt(tx))
            leaf_el.set("y2", _fmt(ty))
            leaf_el.set("class", "door-leaf")
            leaf_el.set("stroke", _SYMBOL_COLOR)
            leaf_el.set("stroke-width", _fmt(_SYMBOL_STROKE_MM))
            # Quarter-arc swing: open tip → latch point.  World CCW appears
            # CW on paper (y flips), and SVG sweep=1 is the CW direction on
            # screen — so sweep follows world-CCW directly.
            sweep = "1" if leaf.sweep_ccw else "0"
            arc = ET.SubElement(g_d, "path")
            arc.set("d", (
                f"M {_fmt(tx)} {_fmt(ty)} "
                f"A {_fmt(r_mm)} {_fmt(r_mm)} 0 0 {sweep} {_fmt(lx)} {_fmt(ly)}"
            ))
            arc.set("class", "door-arc")
            arc.set("fill", "none")
            arc.set("stroke", _SYMBOL_COLOR)
            arc.set("stroke-width", _fmt(_SYMBOL_STROKE_MM))

    # ── Stairs + elevators ────────────────────────────────────────────────
    for pen in floor.penetrations:
        kind = pen.kind.lower()
        if kind in STAIR_KINDS:
            g_s = ET.SubElement(g, "g")
            g_s.set("class", "stair-symbol")
            g_s.set("data-penetration-id", pen.id)
            for tread in stair_treads(pen):
                x1, y1 = to_mm(float(tread[0, 0]), float(tread[0, 1]))
                x2, y2 = to_mm(float(tread[1, 0]), float(tread[1, 1]))
                el = ET.SubElement(g_s, "line")
                el.set("x1", _fmt(x1))
                el.set("y1", _fmt(y1))
                el.set("x2", _fmt(x2))
                el.set("y2", _fmt(y2))
                el.set("class", "stair-tread")
                el.set("stroke", _SYMBOL_COLOR)
                el.set("stroke-width", _fmt(_SYMBOL_STROKE_MM))
            shaft = stair_direction_arrow(pen)
            x1, y1 = to_mm(float(shaft[0, 0]), float(shaft[0, 1]))
            x2, y2 = to_mm(float(shaft[1, 0]), float(shaft[1, 1]))
            el = ET.SubElement(g_s, "line")
            el.set("x1", _fmt(x1))
            el.set("y1", _fmt(y1))
            el.set("x2", _fmt(x2))
            el.set("y2", _fmt(y2))
            el.set("class", "stair-arrow")
            el.set("stroke", _SYMBOL_COLOR)
            el.set("stroke-width", _fmt(_SYMBOL_STROKE_MM))
            el.set("marker-end", "url(#arrowhead)")
        elif kind in ELEVATOR_KINDS:
            g_e = ET.SubElement(g, "g")
            g_e.set("class", "elevator-symbol")
            g_e.set("data-penetration-id", pen.id)
            for diag in elevator_diagonals(pen):
                x1, y1 = to_mm(float(diag[0, 0]), float(diag[0, 1]))
                x2, y2 = to_mm(float(diag[1, 0]), float(diag[1, 1]))
                el = ET.SubElement(g_e, "line")
                el.set("x1", _fmt(x1))
                el.set("y1", _fmt(y1))
                el.set("x2", _fmt(x2))
                el.set("y2", _fmt(y2))
                el.set("class", "elevator-diagonal")
                el.set("stroke", _SYMBOL_COLOR)
                el.set("stroke-width", _fmt(_SYMBOL_STROKE_MM))

    # Arrowhead marker (once) if any stair used it.
    if g.find(".//*[@class='stair-arrow']") is not None:
        defs = ET.SubElement(root, "defs")
        marker = ET.SubElement(defs, "marker")
        marker.set("id", "arrowhead")
        marker.set("markerWidth", "6")
        marker.set("markerHeight", "6")
        marker.set("refX", "5")
        marker.set("refY", "3")
        marker.set("orient", "auto")
        tri = ET.SubElement(marker, "path")
        tri.set("d", "M 0 0 L 6 3 L 0 6 z")
        tri.set("fill", _SYMBOL_COLOR)
