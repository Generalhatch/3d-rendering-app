"""Server-side SVG floor plan sheet renderer.

Takes the canonical :class:`app.geometry.model.FloorGeometry` (so both the
alignment and vectorize pipelines can render sheets) and produces a complete
drawing sheet in the style of the reference PDFs (Stevenson `DEMO41` /
`1ROCK3`):

- page frame + title block (building name / address / floor / scale / date),
  or — in the ``stevenson_minimal`` style — a bottom-left footer line and a
  light tick-and-letter grid on all four page edges
- graphic scale bar + north arrow
- column grid with letter/number axis bubbles at the margins (full style)
- walls as thin DOUBLE lines from the measured faces with mitred joins
  (``wall_style="double"``, the drafting-quality default) or as single
  centerlines on a line-weight hierarchy (``wall_style="centerline"``,
  the legacy mode); door/gap openings are rendered OPEN in both
- heavy suite-perimeter outlines (line weight encodes tenancy, matching
  the reference sheets)
- door swing symbols, stair/elevator symbols, dashed linetypes
- structural columns, major penetrations
- suite labels at ``representative_point()`` with collision avoidance

Coordinate system
-----------------
The SVG user unit is one millimetre of paper.  The plan is first rotated
by ``−rotation_rad`` so its dominant wall axes run parallel to the page
edges (the north arrow carries the true orientation), then world metres
map to paper via ``X = offset_x + s·(u − min_u)`` and
``Y = offset_y + s·(max_v − v)`` (world Y up, paper Y down), where
``(u, v)`` are the rotated world coordinates and
``s = 1000 / scale_denominator`` mm/m.  The transform is exposed on the
returned :class:`SheetRender` so tests and callers can compute exact paper
coordinates.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional, Sequence

import numpy as np

from ..geometry.model import FloorGeometry, Opening, Wall, WallClass
from .drafting import build_wall_outline, dominant_wall_axis_rad
from .grid import ColumnGrid, grid_line_segments_world
from .labels import place_suite_labels

# ── Sheet configuration ───────────────────────────────────────────────────────

#: Paper-space stroke widths (mm) per wall class — the line-weight hierarchy
#: used by the legacy ``wall_style="centerline"`` mode.
WALL_STROKE_MM: dict[WallClass, float] = {
    WallClass.EXTERIOR: 0.70,
    WallClass.DEMISING: 0.40,
    WallClass.PARTITION: 0.18,
}

#: Double-line (drafting) wall rendering: thin light lines — on the
#: reference sheets line weight encodes TENANCY (suite outlines), not
#: construction, so all wall faces share one light stroke.
DOUBLE_WALL_STROKE_MM = 0.15
DOUBLE_WALL_COLOR = "#767676"

#: Heavy suite-perimeter outline (the dominant line weight on the sheet).
SUITE_OUTLINE_STROKE_MM = 0.80
SUITE_OUTLINE_COLOR = "black"

#: Per-feature-class dashed linetypes (SVG stroke-dasharray values, paper
#: mm).  Applied by ``Penetration.kind`` (overhangs / canopies are drawn
#: dashed on the reference sheets); extendable per feature class.
LINETYPE_DASH_MM: dict[str, str] = {
    "overhang": "2.4 1.2",
    "canopy": "2.4 1.2",
    "soffit": "2.4 1.2",
}

#: Preferred architectural scales, tried smallest-denominator first.
STANDARD_SCALES: tuple[int, ...] = (50, 100, 200, 250, 500, 1000)

#: Drafting sans stack — web-safe for in-browser SVG; svglib maps these
#: to Helvetica/Arial in PDF output.
DRAFTING_FONT_STACK = (
    "Inter, 'Helvetica Neue', Helvetica, Arial, Liberation Sans, sans-serif"
)

#: Opening kinds that cut a gap in the wall line (windows do NOT cut).
GAP_KINDS = ("door", "gap")


@dataclass(frozen=True)
class SheetParams:
    """Page geometry.  Defaults: ISO A1 landscape, full title-block style.

    ``style`` selects the sheet furniture:

    - ``"full"`` — page frame, A1 title-block strip, scale bar, grid-bubble
      gutter on the top/left edges (the Phase 3 sheet).
    - ``"stevenson_minimal"`` — no frame or title block; light
      tick-and-letter grid on all four page edges and a footer line
      (building — floor, address) at the bottom left, matching the
      Stevenson reference sheets.
    """
    page_w_mm: float = 841.0
    page_h_mm: float = 594.0
    margin_mm: float = 12.0
    title_block_h_mm: float = 40.0
    grid_gutter_mm: float = 16.0     # reserved for axis bubbles / edge ticks
    inner_pad_mm: float = 4.0
    style: str = "stevenson_minimal"   # "full" | "stevenson_minimal"
    footer_h_mm: float = 14.0        # stevenson_minimal footer strip height
    rotate_to_axes: bool = True      # rotate plan to its dominant wall axes

    @property
    def draw_x0(self) -> float:
        return self.margin_mm + self.grid_gutter_mm

    @property
    def draw_y0(self) -> float:
        return self.margin_mm + self.grid_gutter_mm

    @property
    def draw_x1(self) -> float:
        if self.style == "stevenson_minimal":
            return self.page_w_mm - self.margin_mm - self.grid_gutter_mm
        return self.page_w_mm - self.margin_mm - self.inner_pad_mm

    @property
    def draw_y1(self) -> float:
        if self.style == "stevenson_minimal":
            return (
                self.page_h_mm - self.margin_mm
                - self.grid_gutter_mm - self.footer_h_mm
            )
        return (
            self.page_h_mm - self.margin_mm
            - self.title_block_h_mm - self.inner_pad_mm
        )


@dataclass
class SheetMetadata:
    """Title block / footer content — from job metadata, operator-overridable."""
    building_name: str = ""
    address: str = ""
    floor_name: str = ""
    date_str: str = ""               # defaults to today when empty
    north_angle_deg: float = 0.0     # 0 = north is up in WORLD coordinates
    sheet_title: str = "FLOOR PLAN"


@dataclass
class SheetRender:
    """A rendered sheet + the exact transform used to draw it.

    ``rotation_rad`` is the dominant-axis page rotation: world points are
    rotated by ``−rotation_rad`` before scaling to paper.
    ``bounds_world`` is the bounding box in the ROTATED frame (the frame
    the paper offsets were computed in).
    """
    svg: str
    scale_denominator: int
    mm_per_m: float
    bounds_world: tuple[float, float, float, float]
    offset_x_mm: float
    offset_y_mm: float
    params: SheetParams = field(default_factory=SheetParams)
    rotation_rad: float = 0.0

    def rotate_world(self, x: float, y: float) -> tuple[float, float]:
        """World → rotated (page-aligned) frame."""
        c = math.cos(self.rotation_rad)
        s = math.sin(self.rotation_rad)
        return (x * c + y * s, -x * s + y * c)

    def world_to_mm(self, x: float, y: float) -> tuple[float, float]:
        u, v = self.rotate_world(x, y)
        min_u, _min_v, _max_u, max_v = self.bounds_world
        return (
            self.offset_x_mm + self.mm_per_m * (u - min_u),
            self.offset_y_mm + self.mm_per_m * (max_v - v),
        )


# ── Geometry helpers ──────────────────────────────────────────────────────────

def floor_bounds(
    floor: FloorGeometry, rotation_rad: float = 0.0,
) -> tuple[float, float, float, float]:
    """Bbox of everything drawable (envelope, walls, rooms, columns), in
    the frame rotated by ``−rotation_rad`` (page-aligned frame)."""
    pts: list[np.ndarray] = []
    if floor.envelope is not None and len(floor.envelope):
        pts.append(np.asarray(floor.envelope, dtype=float))
    for w in floor.walls:
        pts.append(np.asarray(w.centerline, dtype=float))
    for r in floor.rooms:
        pts.append(np.asarray(r.boundary, dtype=float))
    for c in floor.columns:
        pts.append(np.asarray([c.centre], dtype=float))
    if not pts:
        return (0.0, 0.0, 1.0, 1.0)
    all_pts = np.vstack(pts)
    if rotation_rad != 0.0:
        c, s = math.cos(rotation_rad), math.sin(rotation_rad)
        all_pts = np.column_stack([
            all_pts[:, 0] * c + all_pts[:, 1] * s,
            -all_pts[:, 0] * s + all_pts[:, 1] * c,
        ])
    return (
        float(all_pts[:, 0].min()), float(all_pts[:, 1].min()),
        float(all_pts[:, 0].max()), float(all_pts[:, 1].max()),
    )


def choose_scale(
    width_m: float,
    height_m: float,
    avail_w_mm: float,
    avail_h_mm: float,
    standard_scales: Sequence[int] = STANDARD_SCALES,
) -> int:
    """Smallest standard denominator at which the plan fits the paper.

    Falls back to the exact-fit denominator rounded UP to the next multiple
    of 100 when the plan is too large for every standard scale (never
    silently clips the drawing).
    """
    for d in standard_scales:
        if width_m * 1000.0 / d <= avail_w_mm and height_m * 1000.0 / d <= avail_h_mm:
            return d
    needed = max(width_m * 1000.0 / avail_w_mm, height_m * 1000.0 / avail_h_mm)
    return int(math.ceil(needed / 100.0) * 100)


def split_wall_at_openings(
    wall: Wall,
    openings: Sequence[Opening],
) -> list[np.ndarray]:
    """Cut door/gap openings out of a wall centerline.

    Returns the KEPT sub-segments (each ``(2, 2)``) — the parts of the wall
    that are actually drawn.  Openings are projected onto the wall axis;
    overlapping gaps are merged.  Windows are not cut (glazing continues
    the wall line).
    """
    p = np.asarray(wall.centerline[0], dtype=float)
    q = np.asarray(wall.centerline[1], dtype=float)
    d = q - p
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        return []
    u = d / length

    gaps: list[tuple[float, float]] = []
    for op in openings:
        if op.wall_id != wall.id or op.kind not in GAP_KINDS:
            continue
        seg = np.asarray(op.segment, dtype=float).reshape(2, 2)
        t0 = float(np.dot(seg[0] - p, u))
        t1 = float(np.dot(seg[1] - p, u))
        lo, hi = sorted((t0, t1))
        lo, hi = max(0.0, lo), min(length, hi)
        if hi - lo > 1e-9:
            gaps.append((lo, hi))
    if not gaps:
        return [np.array([p, q])]

    gaps.sort()
    merged: list[list[float]] = [list(gaps[0])]
    for lo, hi in gaps[1:]:
        if lo <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    kept: list[np.ndarray] = []
    cursor = 0.0
    for lo, hi in merged:
        if lo - cursor > 1e-9:
            kept.append(np.array([p + cursor * u, p + lo * u]))
        cursor = hi
    if length - cursor > 1e-9:
        kept.append(np.array([p + cursor * u, p + length * u]))
    return kept


# ── SVG assembly ──────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _line(parent: ET.Element, x1: float, y1: float, x2: float, y2: float,
          **attrs: str) -> ET.Element:
    el = ET.SubElement(parent, "line")
    el.set("x1", _fmt(x1))
    el.set("y1", _fmt(y1))
    el.set("x2", _fmt(x2))
    el.set("y2", _fmt(y2))
    for k, v in attrs.items():
        el.set(k.replace("_", "-"), v)
    return el


def _text(parent: ET.Element, x: float, y: float, content: str,
          size_mm: float, **attrs: str) -> ET.Element:
    el = ET.SubElement(parent, "text")
    el.set("x", _fmt(x))
    el.set("y", _fmt(y))
    el.set("font-size", _fmt(size_mm))
    el.set("font-family", DRAFTING_FONT_STACK)
    for k, v in attrs.items():
        el.set(k.replace("_", "-"), v)
    el.text = content
    return el


def ring_points_str(
    ring: np.ndarray, to_mm: Callable[[float, float], tuple[float, float]],
) -> str:
    return " ".join(
        f"{_fmt(px)},{_fmt(py)}"
        for px, py in (to_mm(float(x), float(y)) for x, y in ring)
    )


def render_sheet(
    floor: FloorGeometry,
    meta: SheetMetadata | None = None,
    grid: ColumnGrid | None = None,
    params: SheetParams | None = None,
    label_overrides: dict[str, str] | None = None,
    wall_style: str = "double",
    door_swings: dict[str, dict] | None = None,
) -> SheetRender:
    """Render one FloorGeometry as a complete SVG drawing sheet.

    ``wall_style``: ``"double"`` (drafting-quality double lines from the
    measured wall faces, the default) or ``"centerline"`` (legacy single
    lines on the class line-weight hierarchy).
    ``door_swings``: per-opening operator overrides
    (``{opening_id: {"hinge": "start"|"end", "side": "left"|"right"}}``).
    """
    meta = meta or SheetMetadata()
    params = params or SheetParams()

    # ── Page rotation to the dominant wall axes ────────────────────────────
    rotation = (
        dominant_wall_axis_rad(floor.walls)
        if params.rotate_to_axes else 0.0
    )

    min_u, min_v, max_u, max_v = floor_bounds(floor, rotation)
    width_m = max(max_u - min_u, 1e-6)
    height_m = max(max_v - min_v, 1e-6)
    avail_w = params.draw_x1 - params.draw_x0
    avail_h = params.draw_y1 - params.draw_y0
    denom = choose_scale(width_m, height_m, avail_w, avail_h)
    s = 1000.0 / denom   # mm per metre

    offset_x = params.draw_x0 + (avail_w - s * width_m) / 2.0
    offset_y = params.draw_y0 + (avail_h - s * height_m) / 2.0

    render = SheetRender(
        svg="",
        scale_denominator=denom,
        mm_per_m=s,
        bounds_world=(min_u, min_v, max_u, max_v),
        offset_x_mm=offset_x,
        offset_y_mm=offset_y,
        params=params,
        rotation_rad=rotation,
    )
    to_mm = render.world_to_mm

    root = ET.Element("svg")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", f"{_fmt(params.page_w_mm)}mm")
    root.set("height", f"{_fmt(params.page_h_mm)}mm")
    root.set("viewBox", f"0 0 {_fmt(params.page_w_mm)} {_fmt(params.page_h_mm)}")

    # White page background.
    bg = ET.SubElement(root, "rect")
    bg.set("x", "0")
    bg.set("y", "0")
    bg.set("width", _fmt(params.page_w_mm))
    bg.set("height", _fmt(params.page_h_mm))
    bg.set("fill", "white")

    # ── Page frame (full style only) ───────────────────────────────────────
    if params.style != "stevenson_minimal":
        frame = ET.SubElement(root, "rect")
        frame.set("id", "page-frame")
        frame.set("x", _fmt(params.margin_mm))
        frame.set("y", _fmt(params.margin_mm))
        frame.set("width", _fmt(params.page_w_mm - 2 * params.margin_mm))
        frame.set("height", _fmt(params.page_h_mm - 2 * params.margin_mm))
        frame.set("fill", "none")
        frame.set("stroke", "black")
        frame.set("stroke-width", "0.5")

    # ── Column grid (under everything else) ───────────────────────────────
    _render_grid(root, grid, render, floor)

    # ── Penetrations (shafts / stairs / elevators / overhangs) ────────────
    g_pen = ET.SubElement(root, "g")
    g_pen.set("id", "penetrations")
    for pen in floor.penetrations:
        ring = np.asarray(pen.polygon, dtype=float)
        poly = ET.SubElement(g_pen, "polygon")
        poly.set("points", ring_points_str(ring, to_mm))
        poly.set("class", f"penetration penetration-{pen.kind}")
        dash = LINETYPE_DASH_MM.get(pen.kind)
        if dash is not None:
            poly.set("fill", "none")
            poly.set("stroke", "#444444")
            poly.set("stroke-width", "0.2")
            poly.set("stroke-dasharray", dash)
        else:
            poly.set("fill", "#e8e8e8")
            poly.set("stroke", "#444444")
            poly.set("stroke-width", "0.25")

    # ── Envelope (only when no walls cover the shell — e.g. a floor that
    # produced rooms without wall geometry).  Regularized at the render
    # layer only: DP-simplified, straight runs snapped to the dominant
    # axes, curved facades emitted as arcs.  Measurement still reads the
    # RAW envelope from FloorGeometry.
    g_env = ET.SubElement(root, "g")
    g_env.set("id", "envelope")
    if not floor.walls and floor.envelope is not None and len(floor.envelope) >= 3:
        from .curves import regularize_ring, ring_dominant_axes, ring_path_d
        env = np.asarray(floor.envelope, dtype=float)
        reg = regularize_ring(env, axes_rad=ring_dominant_axes(env))
        ring_mm = np.array([to_mm(float(x), float(y)) for x, y in reg])
        path = ET.SubElement(g_env, "path")
        path.set("d", ring_path_d(ring_mm, arc_tol_mm=0.05 * s))
        path.set("class", "envelope-outline")
        path.set("fill", "none")
        path.set("stroke", DOUBLE_WALL_COLOR)
        path.set("stroke-width", _fmt(DOUBLE_WALL_STROKE_MM))

    # ── Walls ─────────────────────────────────────────────────────────────
    g_walls = ET.SubElement(root, "g")
    g_walls.set("id", "walls")
    if wall_style == "double":
        _render_walls_double(g_walls, floor, to_mm, s)
    else:
        _render_walls_centerline(g_walls, floor, to_mm)

    # ── Window openings (drawn light, walls not cut) ──────────────────────
    g_open = ET.SubElement(root, "g")
    g_open.set("id", "openings")
    for op in floor.openings:
        if op.kind != "window":
            continue
        seg = np.asarray(op.segment, dtype=float).reshape(2, 2)
        x1, y1 = to_mm(float(seg[0, 0]), float(seg[0, 1]))
        x2, y2 = to_mm(float(seg[1, 0]), float(seg[1, 1]))
        _line(
            g_open, x1, y1, x2, y2,
            stroke="#3a6ea5", stroke_width="0.15",
        ).set("class", "opening opening-window")

    # ── Symbols (door swings, stairs, elevators) ──────────────────────────
    from .symbols import render_symbols
    render_symbols(root, floor, to_mm, s, door_swings=door_swings)

    # ── Suite-perimeter outlines (heavy — the tenancy line weight) ────────
    _render_suite_outlines(root, floor, to_mm)

    # ── Columns ───────────────────────────────────────────────────────────
    g_cols = ET.SubElement(root, "g")
    g_cols.set("id", "columns")
    for col in floor.columns:
        if col.is_round and col.radius_m:
            cx, cy = to_mm(float(col.centre[0]), float(col.centre[1]))
            circ = ET.SubElement(g_cols, "circle")
            circ.set("cx", _fmt(cx))
            circ.set("cy", _fmt(cy))
            circ.set("r", _fmt(float(col.radius_m) * s))
            circ.set("class", "column column-round")
            circ.set("fill", "#333333")
        elif col.polygon is not None and len(col.polygon):
            ring = np.asarray(col.polygon, dtype=float)
            poly = ET.SubElement(g_cols, "polygon")
            poly.set("points", ring_points_str(ring, to_mm))
            poly.set("class", "column column-rect")
            poly.set("fill", "#333333")
        else:
            cx, cy = to_mm(float(col.centre[0]), float(col.centre[1]))
            circ = ET.SubElement(g_cols, "circle")
            circ.set("cx", _fmt(cx))
            circ.set("cy", _fmt(cy))
            circ.set("r", _fmt(0.2 * s))
            circ.set("class", "column column-unknown")
            circ.set("fill", "#333333")

    # ── Suite labels ──────────────────────────────────────────────────────
    minimal = params.style == "stevenson_minimal"
    g_labels = ET.SubElement(root, "g")
    g_labels.set("id", "suite-labels")
    label_font = 7.0 if minimal else 4.0
    sub_font = 2.6
    placed = place_suite_labels(
        floor, to_mm,
        font_mm=label_font, sub_font_mm=sub_font,
        label_overrides=label_overrides,
        include_area=not minimal,
    )
    for lab in placed:
        g = ET.SubElement(g_labels, "g")
        g.set("class", "suite-label" + (" suite-label-common" if lab.is_common else ""))
        g.set("data-room-id", lab.room_id)
        if minimal:
            _text(
                g, lab.x_mm, lab.y_mm, lab.text, label_font,
                text_anchor="middle", fill="#9a9a9a",
            ).set("class", "suite-label-name")
        else:
            _text(
                g, lab.x_mm, lab.y_mm, lab.text, label_font,
                text_anchor="middle", font_weight="bold", fill="black",
            ).set("class", "suite-label-name")
            _text(
                g, lab.x_mm, lab.y_mm + 3.6, lab.sub_text, sub_font,
                text_anchor="middle", fill="#333333",
            ).set("class", "suite-label-area")

    # ── North arrow (carries the true orientation after page rotation) ────
    if params.style == "stevenson_minimal":
        _render_north_arrow_minimal(root, meta, params, rotation)
    else:
        _render_north_arrow(root, meta, params, rotation)

    # ── Title block / footer + scale bar ──────────────────────────────────
    if params.style == "stevenson_minimal":
        _render_footer(root, floor, meta, params, denom)
    else:
        _render_title_block(root, floor, meta, params, denom)
        _render_scale_bar(root, params, s)

    render.svg = ET.tostring(root, encoding="unicode")
    return render


# ── Wall sub-renderers ────────────────────────────────────────────────────────

def _render_walls_double(
    g_walls: ET.Element,
    floor: FloorGeometry,
    to_mm: Callable[[float, float], tuple[float, float]],
    mm_per_m: float,
) -> None:
    """Drafting-quality walls: thin double lines from the measured faces.

    The wall network is unioned (mitred joins, hardened junctions) and each
    boundary ring is stroked as one closed path.  Curved runs are emitted
    as SVG arcs (see :mod:`app.sheet.curves`).
    """
    from .curves import ring_path_d
    outline = build_wall_outline(floor)
    for ring in outline.rings:
        ring_mm = np.array([to_mm(float(x), float(y)) for x, y in ring])
        path = ET.SubElement(g_walls, "path")
        path.set("d", ring_path_d(ring_mm, arc_tol_mm=0.05 * mm_per_m))
        path.set("class", "wall-outline")
        path.set("fill", "none")
        path.set("stroke", DOUBLE_WALL_COLOR)
        path.set("stroke-width", _fmt(DOUBLE_WALL_STROKE_MM))
        path.set("stroke-linejoin", "miter")


def _render_walls_centerline(
    g_walls: ET.Element,
    floor: FloorGeometry,
    to_mm: Callable[[float, float], tuple[float, float]],
) -> None:
    """Legacy walls: single centerlines on the class line-weight hierarchy."""
    for wall in floor.walls:
        stroke = WALL_STROKE_MM[wall.wall_class]
        for seg in split_wall_at_openings(wall, floor.openings):
            x1, y1 = to_mm(float(seg[0, 0]), float(seg[0, 1]))
            x2, y2 = to_mm(float(seg[1, 0]), float(seg[1, 1]))
            _line(
                g_walls, x1, y1, x2, y2,
                stroke="black",
                stroke_width=_fmt(stroke),
                stroke_linecap="square",
            ).set("class", f"wall wall-{wall.wall_class.value}")


# ── Suite outlines ────────────────────────────────────────────────────────────

def suite_outline_rings(floor: FloorGeometry) -> list[tuple[str, np.ndarray]]:
    """(suite_key, ring) pairs: the union boundary of each suite's rooms.

    Rooms sharing a ``suite_id`` union into one perimeter; rooms without a
    suite_id are their own perimeter.  Common areas (corridors, lobbies)
    carry no heavy outline — on the reference sheets the heavy line weight
    encodes tenancy.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    groups: dict[str, list] = {}
    for room in floor.rooms:
        if room.is_common:
            continue
        key = room.suite_id if room.suite_id is not None else room.id
        poly = ShapelyPolygon(np.asarray(room.boundary, dtype=float))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        groups.setdefault(str(key), []).append(poly)

    out: list[tuple[str, np.ndarray]] = []
    for key in sorted(groups):
        # simplify(0) drops the collinear vertices the union leaves where a
        # dissolved shared edge met the perimeter.
        merged = unary_union(groups[key]).simplify(0)
        polys = (
            list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
        )
        for poly in polys:
            out.append((key, np.asarray(poly.exterior.coords[:-1], dtype=float)))
            for hole in poly.interiors:
                out.append((key, np.asarray(hole.coords[:-1], dtype=float)))
    return out


def _render_suite_outlines(
    root: ET.Element,
    floor: FloorGeometry,
    to_mm: Callable[[float, float], tuple[float, float]],
) -> None:
    g = ET.SubElement(root, "g")
    g.set("id", "suite-outlines")
    for key, ring in suite_outline_rings(floor):
        poly = ET.SubElement(g, "polygon")
        poly.set("points", ring_points_str(ring, to_mm))
        poly.set("class", "suite-outline")
        poly.set("data-suite-id", key)
        poly.set("fill", "none")
        poly.set("stroke", SUITE_OUTLINE_COLOR)
        poly.set("stroke-width", _fmt(SUITE_OUTLINE_STROKE_MM))
        poly.set("stroke-linejoin", "miter")


# ── Sub-renderers ─────────────────────────────────────────────────────────────

def _render_grid(
    root: ET.Element, grid: ColumnGrid | None, render: SheetRender,
    floor: FloorGeometry,
) -> None:
    g_grid = ET.SubElement(root, "g")
    g_grid.set("id", "grid")
    if grid is None:
        return
    # grid_line_segments_world sizes lines from a WORLD-frame bbox; derive
    # one by inverse-rotating the page-frame bbox corners.
    bounds_world = floor_bounds(floor, 0.0)
    s = render.mm_per_m
    ext_m = 10.0 / s
    segs = grid_line_segments_world(grid, bounds_world, extension_m=ext_m)
    if render.params.style == "stevenson_minimal":
        _render_grid_ticks(g_grid, segs, render)
    else:
        _render_grid_bubbles(g_grid, segs, render)


def _render_grid_bubbles(
    g_grid: ET.Element, segs: list[dict], render: SheetRender,
) -> None:
    """Full style: dashed lines across the plan + axis bubbles (Phase 3)."""
    bubble_r = 3.2
    for seg in segs:
        x1, y1 = render.world_to_mm(*seg["p0"])
        x2, y2 = render.world_to_mm(*seg["p1"])
        line = ET.SubElement(g_grid, "line")
        line.set("x1", _fmt(x1))
        line.set("y1", _fmt(y1))
        line.set("x2", _fmt(x2))
        line.set("y2", _fmt(y2))
        line.set("class", f"grid-line grid-line-{seg['family']}")
        line.set("stroke", "#888888")
        line.set("stroke-width", "0.13")
        line.set("stroke-dasharray", "6 1.5 1.5 1.5")
        # Bubble just past the p1 end.
        d = np.array([x2 - x1, y2 - y1], dtype=float)
        n = float(np.linalg.norm(d))
        u = d / n if n > 1e-9 else np.array([0.0, -1.0])
        bx, by = x2 + u[0] * bubble_r, y2 + u[1] * bubble_r
        g_b = ET.SubElement(g_grid, "g")
        g_b.set("class", "grid-bubble")
        circ = ET.SubElement(g_b, "circle")
        circ.set("cx", _fmt(bx))
        circ.set("cy", _fmt(by))
        circ.set("r", _fmt(bubble_r))
        circ.set("fill", "white")
        circ.set("stroke", "black")
        circ.set("stroke-width", "0.25")
        _text(
            g_b, bx, by + 1.2, seg["label"], 3.4,
            text_anchor="middle", font_weight="bold", fill="black",
        ).set("class", "grid-bubble-label")


def _render_grid_ticks(
    g_grid: ET.Element, segs: list[dict], render: SheetRender,
) -> None:
    """Stevenson-minimal style: light ticks + labels on all four page edges.

    No lines across the plan and no bubbles: each grid line is extended to
    the page-edge gutters and marked with a short tick and its letter or
    number at BOTH crossings, in light gray — matching the reference sheet.
    """
    p = render.params
    x_lo, x_hi = p.margin_mm, p.page_w_mm - p.margin_mm
    y_lo, y_hi = p.margin_mm, p.page_h_mm - p.margin_mm - p.footer_h_mm
    tick_mm = 3.0
    color = "#aaaaaa"

    for seg in segs:
        x1, y1 = render.world_to_mm(*seg["p0"])
        x2, y2 = render.world_to_mm(*seg["p1"])
        d = np.array([x2 - x1, y2 - y1], dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue
        u = d / n

        # Intersect the infinite paper-space line with the 4 gutter edges.
        crossings: list[tuple[float, float, str]] = []
        if abs(u[1]) > 1e-9:   # crosses horizontal edges
            for edge_y, side in ((y_lo, "top"), (y_hi, "bottom")):
                t = (edge_y - y1) / u[1]
                ex = x1 + t * u[0]
                if x_lo - 1e-6 <= ex <= x_hi + 1e-6:
                    crossings.append((ex, edge_y, side))
        if abs(u[0]) > 1e-9:   # crosses vertical edges
            for edge_x, side in ((x_lo, "left"), (x_hi, "right")):
                t = (edge_x - x1) / u[0]
                ey = y1 + t * u[1]
                if y_lo - 1e-6 <= ey <= y_hi + 1e-6:
                    crossings.append((edge_x, ey, side))

        for ex, ey, side in crossings:
            g_t = ET.SubElement(g_grid, "g")
            g_t.set("class", f"grid-tick grid-tick-{seg['family']}")
            g_t.set("data-label", seg["label"])
            if side in ("top", "bottom"):
                sign = -1.0 if side == "top" else 1.0
                _line(g_t, ex, ey, ex, ey + sign * tick_mm,
                      stroke=color, stroke_width="0.2")
                ty = ey + sign * (tick_mm + 3.2) + (0.0 if side == "top" else 1.6)
                _text(g_t, ex, ty, seg["label"], 4.0,
                      text_anchor="middle", fill=color,
                      ).set("class", "grid-tick-label")
            else:
                sign = -1.0 if side == "left" else 1.0
                _line(g_t, ex, ey, ex + sign * tick_mm, ey,
                      stroke=color, stroke_width="0.2")
                tx = ex + sign * (tick_mm + 2.4)
                _text(g_t, tx, ey + 1.4, seg["label"], 4.0,
                      text_anchor="middle", fill=color,
                      ).set("class", "grid-tick-label")


def _render_north_arrow_minimal(
    root: ET.Element, meta: SheetMetadata, params: SheetParams,
    page_rotation_rad: float = 0.0,
) -> None:
    """Tiny north indicator for stevenson_minimal — unobtrusive deliverable."""
    cx = params.page_w_mm - params.margin_mm - 8.0
    cy = params.page_h_mm - params.margin_mm - params.footer_h_mm - 8.0
    angle = meta.north_angle_deg + math.degrees(page_rotation_rad)
    g = ET.SubElement(root, "g")
    g.set("id", "north-arrow")
    g.set("opacity", "0.45")
    g.set("transform", f"rotate({_fmt(angle)} {_fmt(cx)} {_fmt(cy)})")
    tri = ET.SubElement(g, "polygon")
    tri.set("points", (
        f"{_fmt(cx)},{_fmt(cy - 4)} "
        f"{_fmt(cx - 1.8)},{_fmt(cy + 2)} "
        f"{_fmt(cx + 1.8)},{_fmt(cy + 2)}"
    ))
    tri.set("fill", "#666666")
    _text(g, cx, cy + 5.0, "N", 2.2, text_anchor="middle", fill="#666666")


def _render_north_arrow(
    root: ET.Element, meta: SheetMetadata, params: SheetParams,
    page_rotation_rad: float = 0.0,
) -> None:
    cx = params.page_w_mm - params.margin_mm - 18.0
    cy = params.margin_mm + 18.0
    # The plan was rotated on the page; the arrow compensates so it always
    # points at TRUE north.
    angle = meta.north_angle_deg + math.degrees(page_rotation_rad)
    g = ET.SubElement(root, "g")
    g.set("id", "north-arrow")
    g.set("transform", f"rotate({_fmt(angle)} {_fmt(cx)} {_fmt(cy)})")
    circ = ET.SubElement(g, "circle")
    circ.set("cx", _fmt(cx))
    circ.set("cy", _fmt(cy))
    circ.set("r", "10")
    circ.set("fill", "white")
    circ.set("stroke", "black")
    circ.set("stroke-width", "0.3")
    tri = ET.SubElement(g, "polygon")
    tri.set("points", (
        f"{_fmt(cx)},{_fmt(cy - 8)} "
        f"{_fmt(cx - 3)},{_fmt(cy + 4)} "
        f"{_fmt(cx + 3)},{_fmt(cy + 4)}"
    ))
    tri.set("fill", "black")
    _text(g, cx, cy + 8.5, "N", 3.2, text_anchor="middle", font_weight="bold")


def _render_scale_bar(root: ET.Element, params: SheetParams, s: float) -> None:
    """Graphic scale bar: 4 alternating divisions, sited in the title strip."""
    total_candidates = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    bar_m = total_candidates[0]
    for cand in total_candidates:
        if cand * s <= 100.0:
            bar_m = cand
    bar_mm = bar_m * s
    n_div = 4
    x0 = params.margin_mm + 6.0
    y0 = params.page_h_mm - params.margin_mm - params.title_block_h_mm + 26.0
    h = 2.4

    g = ET.SubElement(root, "g")
    g.set("id", "scale-bar")
    for i in range(n_div):
        seg = ET.SubElement(g, "rect")
        seg.set("x", _fmt(x0 + i * bar_mm / n_div))
        seg.set("y", _fmt(y0))
        seg.set("width", _fmt(bar_mm / n_div))
        seg.set("height", _fmt(h))
        seg.set("fill", "black" if i % 2 == 0 else "white")
        seg.set("stroke", "black")
        seg.set("stroke-width", "0.2")
        seg.set("class", "scale-bar-div")
    _text(g, x0, y0 - 1.2, "0", 2.4, text_anchor="middle")
    _text(
        g, x0 + bar_mm, y0 - 1.2,
        f"{bar_m:g} m", 2.4, text_anchor="middle",
    ).set("class", "scale-bar-max")


def _render_footer(
    root: ET.Element,
    floor: FloorGeometry,
    meta: SheetMetadata,
    params: SheetParams,
    denom: int,
) -> None:
    """Stevenson-minimal footer: building — floor (left), centred address,
    scale + date (right).  No generator watermark on deliverables."""
    building = meta.building_name or floor.building_name or "—"
    floor_name = meta.floor_name or floor.floor_name or "—"
    address = meta.address or ""
    date_str = meta.date_str or date.today().isoformat()

    x0 = params.margin_mm + 2.0
    x_c = params.page_w_mm / 2.0
    x_r = params.page_w_mm - params.margin_mm - 2.0
    y1 = params.page_h_mm - params.margin_mm

    g = ET.SubElement(root, "g")
    g.set("id", "footer")
    _text(g, x0, y1 - 4.5, f"{building} - {floor_name}", 4.2,
          font_weight="bold").set("id", "footer-building")
    if address:
        _text(g, x_c, y1 - 2.0, address, 3.6,
              text_anchor="middle", fill="#333333").set("id", "footer-address")
    _text(g, x_r, y1 - 4.5, f"SCALE 1:{denom}", 2.8,
          text_anchor="end", fill="#555555").set("id", "footer-scale")
    _text(g, x_r, y1 - 1.6, date_str, 2.8,
          text_anchor="end", fill="#555555").set("id", "footer-date")


def _render_title_block(
    root: ET.Element,
    floor: FloorGeometry,
    meta: SheetMetadata,
    params: SheetParams,
    denom: int,
) -> None:
    x0 = params.margin_mm
    x1 = params.page_w_mm - params.margin_mm
    y1 = params.page_h_mm - params.margin_mm
    y0 = y1 - params.title_block_h_mm

    g = ET.SubElement(root, "g")
    g.set("id", "title-block")

    box = ET.SubElement(g, "rect")
    box.set("x", _fmt(x0))
    box.set("y", _fmt(y0))
    box.set("width", _fmt(x1 - x0))
    box.set("height", _fmt(y1 - y0))
    box.set("fill", "none")
    box.set("stroke", "black")
    box.set("stroke-width", "0.5")

    # Vertical field dividers.
    div_xs = (x0 + 220.0, x0 + 420.0, x0 + 560.0, x0 + 680.0)
    for dx in div_xs:
        _line(g, dx, y0, dx, y1, stroke="black", stroke_width="0.3")

    building = meta.building_name or floor.building_name or "—"
    floor_name = meta.floor_name or floor.floor_name or "—"
    address = meta.address or "—"
    date_str = meta.date_str or date.today().isoformat()

    _text(g, x0 + 6, y0 + 12, building, 6.0, font_weight="bold").set("id", "tb-building")
    _text(g, x0 + 6, y0 + 20, address, 3.2, fill="#333333").set("id", "tb-address")
    _text(g, div_xs[0] + 6, y0 + 12, meta.sheet_title, 5.0,
          font_weight="bold").set("id", "tb-title")
    _text(g, div_xs[0] + 6, y0 + 20, floor_name, 3.6).set("id", "tb-floor")
    _text(g, div_xs[1] + 6, y0 + 12, f"SCALE 1:{denom}", 3.6).set("id", "tb-scale")
    _text(g, div_xs[1] + 6, y0 + 20, f"DATE {date_str}", 3.2).set("id", "tb-date")
    _text(g, div_xs[2] + 6, y0 + 12, f"JOB {floor.floor_id[:18]}", 3.0,
          fill="#333333").set("id", "tb-job")
    _text(g, div_xs[2] + 6, y0 + 20,
          f"SOURCE {floor.source or 'scan'}", 3.0, fill="#333333").set("id", "tb-source")
    _text(g, div_xs[3] + 6, y0 + 12, "GENERATED BY", 2.6,
          fill="#666666").set("id", "tb-generator-caption")
    _text(g, div_xs[3] + 6, y0 + 20, "AlignAI Scan-to-CAD", 3.4,
          font_weight="bold").set("id", "tb-generator")
