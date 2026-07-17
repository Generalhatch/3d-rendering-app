"""Suite label placement for the sheet renderer.

Labels are anchored at each room polygon's Shapely
``representative_point()`` (guaranteed inside the polygon, unlike the
centroid of an L-shaped room), then pushed apart in SHEET space until no
two label boxes overlap.  Collision avoidance runs in sheet millimetres —
whether two labels collide depends on the drawing scale, not on world
distances.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from ..geometry.model import FloorGeometry, Room

# Approximate glyph width as a fraction of font size for Helvetica-like
# sans fonts — good enough for collision boxes.
_CHAR_W_FACTOR = 0.62
_SUITE_PREFIX_RE = re.compile(r"^suite\s+", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"^\d+$")


def format_suite_label(text: str, suite_id: str | None = None) -> str:
    """Display convention for suite labels on the sheet.

    Operator overrides that are bare digits (``"320"``) render as
    ``"Suite 320"``.  Existing ``"Suite …"`` prefixes are preserved.
    When no label is set, ``suite_id`` is used (never a raw room UUID).
    """
    raw = (text or "").strip()
    if not raw:
        if suite_id is not None and str(suite_id).strip():
            return format_suite_label(str(suite_id).strip())
        return ""
    if _SUITE_PREFIX_RE.match(raw):
        return raw
    if _BARE_NUMBER_RE.match(raw):
        return f"Suite {raw}"
    return raw


@dataclass
class PlacedLabel:
    """One suite label, positioned in sheet millimetres."""
    room_id: str
    text: str                # primary line (suite number / room label)
    sub_text: str            # secondary line (area)
    x_mm: float              # box centre
    y_mm: float
    w_mm: float
    h_mm: float
    is_common: bool = False

    def bounds(self) -> tuple[float, float, float, float]:
        hw, hh = self.w_mm / 2.0, self.h_mm / 2.0
        return (self.x_mm - hw, self.y_mm - hh, self.x_mm + hw, self.y_mm + hh)


def _boxes_overlap(a: PlacedLabel, b: PlacedLabel) -> bool:
    ax0, ay0, ax1, ay1 = a.bounds()
    bx0, by0, bx1, by1 = b.bounds()
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def resolve_label_collisions(
    labels: list[PlacedLabel],
    max_iter: int = 100,
    pad_mm: float = 0.5,
) -> list[PlacedLabel]:
    """Push overlapping label boxes apart (in place) until disjoint.

    Deterministic iterative separation: every overlapping pair is pushed
    apart along the line between the two box centres (or vertically when
    the centres coincide) by half the overlap each, plus padding.  Bails
    after ``max_iter`` sweeps — labels may still overlap on pathological
    inputs, but they never fly off to infinity.
    """
    for _ in range(max_iter):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                if not _boxes_overlap(a, b):
                    continue
                moved = True
                dx = b.x_mm - a.x_mm
                dy = b.y_mm - a.y_mm
                dist = float(np.hypot(dx, dy))
                if dist < 1e-9:
                    # Coincident centres: split vertically.
                    ux, uy = 0.0, 1.0
                else:
                    ux, uy = dx / dist, dy / dist
                # Required centre separation along the push direction so the
                # boxes just clear each other (conservative: use the larger
                # of the x/y requirements projected on the push direction).
                need_x = (a.w_mm + b.w_mm) / 2.0 + pad_mm
                need_y = (a.h_mm + b.h_mm) / 2.0 + pad_mm
                # Solve for the smallest s ≥ dist with |s·ux| ≥ need_x OR
                # |s·uy| ≥ need_y (separated on either axis is enough).
                candidates = []
                if abs(ux) > 1e-9:
                    candidates.append(need_x / abs(ux))
                if abs(uy) > 1e-9:
                    candidates.append(need_y / abs(uy))
                target = min(candidates)
                push = (target - dist) / 2.0
                a.x_mm -= ux * push
                a.y_mm -= uy * push
                b.x_mm += ux * push
                b.y_mm += uy * push
        if not moved:
            break
    return labels


def label_box_size_mm(
    text: str, sub_text: str, font_mm: float, sub_font_mm: float,
) -> tuple[float, float]:
    """Bounding box (w, h) of a two-line label at the given font sizes."""
    w = max(
        len(text) * font_mm * _CHAR_W_FACTOR,
        len(sub_text) * sub_font_mm * _CHAR_W_FACTOR,
    )
    h = font_mm + (sub_font_mm * 1.4 if sub_text else 0.0)
    return (w, h)


def _label_text_for(room: Room) -> str:
    if room.label:
        return format_suite_label(room.label, room.suite_id)
    if room.suite_id:
        return format_suite_label(str(room.suite_id))
    return room.label or room.id


def place_suite_labels(
    floor: FloorGeometry,
    world_to_mm: Callable[[float, float], tuple[float, float]],
    font_mm: float = 4.0,
    sub_font_mm: float = 2.6,
    label_overrides: dict[str, str] | None = None,
    include_area: bool = True,
) -> list[PlacedLabel]:
    """One label per room, anchored at ``representative_point()``.

    Rooms sharing a ``suite_id`` get ONE label, placed on the largest room
    of the suite (the suite is one occupancy — two labels would misread).
    ``label_overrides`` maps room id → operator-edited label text (the
    "suite numbering editable in the UI" hook).  ``include_area=False``
    drops the area sub-line (the stevenson_minimal sheet style shows only
    the big gray suite name).
    """
    label_overrides = label_overrides or {}

    # Suite grouping: keep only the largest room of each suite_id group.
    keep: list[Room] = []
    by_suite: dict[str, Room] = {}
    for room in floor.rooms:
        if room.suite_id is None:
            keep.append(room)
            continue
        cur = by_suite.get(room.suite_id)
        if cur is None or floor.room_area_m2(room.id) > floor.room_area_m2(cur.id):
            by_suite[room.suite_id] = room
    keep.extend(by_suite.values())

    labels: list[PlacedLabel] = []
    for room in keep:
        poly = ShapelyPolygon(np.asarray(room.boundary, dtype=float))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        pt = poly.representative_point()
        x_mm, y_mm = world_to_mm(float(pt.x), float(pt.y))

        if room.id in label_overrides:
            text = format_suite_label(label_overrides[room.id], room.suite_id)
        else:
            text = _label_text_for(room)
        area = floor.room_area_m2(room.id)
        sub = f"{area:,.1f} m²" if include_area else ""
        w, h = label_box_size_mm(text, sub, font_mm, sub_font_mm)
        labels.append(PlacedLabel(
            room_id=room.id,
            text=text,
            sub_text=sub,
            x_mm=x_mm,
            y_mm=y_mm,
            w_mm=w,
            h_mm=h,
            is_common=room.is_common,
        ))

    return resolve_label_collisions(labels)
