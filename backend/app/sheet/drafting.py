"""Drafting-quality wall geometry for the sheet renderer (Phase 3.5).

The reference sheets (Stevenson ``DEMO41-Fullfloor``) draw walls as thin,
light DOUBLE lines — the two measured faces of each wall — with clean
mitred corner joins, not as single centerlines with a line-weight
hierarchy.  This module produces that outline geometry:

1. Each wall becomes a rectangle spanning its two measured faces
   (:attr:`Wall.face_a` / :attr:`Wall.face_b` — wall pairing produces
   them; ``Wall.__post_init__`` derives them from centerline + thickness
   when the extractor didn't).
2. **Junction closure hardening** (render-stage ONLY — measurement
   geometry is never touched): a wall end is extended by up to
   ``join_ext_m`` *only when the extension strip actually reaches another
   wall*, so small detection gaps close and corners join, while
   free-standing wall ends stay exactly at their measured length.
3. All rectangles are unioned (Shapely), which produces mitred joins at
   corners and T-junctions for free.
4. Door / gap openings are subtracted across the full wall thickness, so
   openings read as true gaps with capped jambs on both sides.
5. The union boundary is returned as closed rings (exteriors + holes) for
   the renderer to stroke as thin double lines.

Everything here is presentation: no measurement number can change because
none of this feeds :meth:`FloorGeometry.measured_room_polygon`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from ..geometry.model import FloorGeometry, Opening, Wall

#: Opening kinds that cut a real gap through the wall (windows do not).
GAP_KINDS = ("door", "gap")

#: Default junction-closure reach: how far a wall end may be extended to
#: meet a neighbouring wall.  Detection gaps are typically a few cm; this
#: also covers the half-thickness of the crossing wall at a corner.
DEFAULT_JOIN_EXT_M = 0.30


def _wall_frame(wall: Wall) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """(start point, unit axis, length, unit normal) of a wall centerline."""
    p = np.asarray(wall.centerline[0], dtype=float)
    q = np.asarray(wall.centerline[1], dtype=float)
    d = q - p
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        raise ValueError(f"wall {wall.id}: degenerate zero-length centerline")
    u = d / length
    n = np.array([-u[1], u[0]])
    return p, u, length, n


def wall_rect(wall: Wall, ext_start_m: float = 0.0, ext_end_m: float = 0.0) -> ShapelyPolygon:
    """The wall's face-to-face rectangle, optionally extended along its axis.

    The rectangle spans the two measured faces (``face_a``/``face_b``);
    when both faces exist they are used directly, so a wall whose faces
    are asymmetric about the centerline still renders where it was
    measured.
    """
    if wall.face_a is not None and wall.face_b is not None and ext_start_m == 0.0 and ext_end_m == 0.0:
        a = np.asarray(wall.face_a, dtype=float).reshape(2, 2)
        b = np.asarray(wall.face_b, dtype=float).reshape(2, 2)
        return ShapelyPolygon([a[0], a[1], b[1], b[0]])

    p, u, length, n = _wall_frame(wall)
    half = wall.thickness_m / 2.0
    # Prefer the measured faces for the half-width when present.
    if wall.face_a is not None and wall.face_b is not None:
        a = np.asarray(wall.face_a, dtype=float).reshape(2, 2)
        b = np.asarray(wall.face_b, dtype=float).reshape(2, 2)
        half = float(abs(np.dot(a[0] - b[0], n))) / 2.0 or half
    p0 = p - u * ext_start_m
    p1 = p + u * (length + ext_end_m)
    return ShapelyPolygon([
        p0 + n * half, p1 + n * half, p1 - n * half, p0 - n * half,
    ])


def _end_strip(wall: Wall, end: int, reach_m: float) -> ShapelyPolygon:
    """The would-be extension strip beyond one wall end (0=start, 1=end):
    the wall's own width, reaching ``reach_m`` along its axis."""
    p, u, length, n = _wall_frame(wall)
    half = wall.thickness_m / 2.0
    if end == 0:
        anchor, out_dir = p, -u
    else:
        anchor, out_dir = p + u * length, u
    return ShapelyPolygon([
        anchor + n * half,
        anchor + out_dir * reach_m + n * half,
        anchor + out_dir * reach_m - n * half,
        anchor - n * half,
    ])


def _end_join_pieces(
    wall: Wall,
    end: int,
    reach_m: float,
    others: list[tuple[ShapelyPolygon, ShapelyPolygon]],
) -> list[ShapelyPolygon]:
    """Junction-closure material for one wall end (0=start, 1=end).

    ``others``: (rect, slab) per neighbouring wall, where ``slab`` is the
    rect dilated by ``reach_m`` along its own axis.  For every neighbour
    whose actual rect intersects the extension strip, the strip is clipped
    to that neighbour's slab — so the extension always terminates exactly
    at the neighbour's far face.  That produces true mitres at any joint
    angle (an oblique neighbour trims the extension along its face instead
    of letting a square-cut corner poke through), closes small detection
    gaps, and never bridges a gap wider than ``reach_m`` (the strip must
    touch the neighbour's REAL rect, not just its slab).  Free ends get no
    pieces: measured wall length is preserved.
    """
    strip = _end_strip(wall, end, reach_m)
    pieces: list[ShapelyPolygon] = []
    for rect, slab in others:
        if not strip.intersects(rect):
            continue
        piece = strip.intersection(slab)
        if not piece.is_empty and piece.area > 1e-12:
            pieces.append(piece)
    return pieces


@dataclass
class WallOutline:
    """The drafting outline of the whole wall network.

    ``rings``: list of closed rings (first/last vertex NOT duplicated),
    outer boundaries and hole boundaries alike — the renderer strokes each
    as a thin double line.  ``extended_ends``: (wall_id, "start"|"end")
    pairs whose junction was closed by extension (for tests/audit).
    """
    rings: list[np.ndarray]
    extended_ends: list[tuple[str, str]]


def build_wall_outline(
    floor: FloorGeometry,
    join_ext_m: float = DEFAULT_JOIN_EXT_M,
) -> WallOutline:
    """Union of wall face-rectangles with junction closure and open doors.

    Render-stage geometry only.  Returns the boundary rings of
    ``union(wall rects, junction-closing extensions) − union(door rects)``.
    """
    walls = [w for w in floor.walls
             if float(np.linalg.norm(
                 np.asarray(w.centerline[1]) - np.asarray(w.centerline[0]))) > 1e-9]
    if not walls:
        return WallOutline(rings=[], extended_ends=[])

    base_rects = {w.id: wall_rect(w) for w in walls}
    # "Slab" per wall: its rect dilated join_ext_m along its own axis —
    # the region a neighbour's extension may fill (through to the far face).
    slabs = {w.id: wall_rect(w, join_ext_m, join_ext_m) for w in walls}

    # Junction closure: fill each wall end's extension strip only where it
    # reaches ANOTHER wall, clipped to that wall's slab.  (Decided against
    # the base rects — order-independent.)
    extended_ends: list[tuple[str, str]] = []
    pieces = [base_rects[w.id] for w in walls]
    for w in walls:
        others = [
            (base_rects[o.id], slabs[o.id]) for o in walls if o.id != w.id
        ]
        if not others:
            continue
        for end, name in ((0, "start"), (1, "end")):
            joins = _end_join_pieces(w, end, join_ext_m, others)
            if joins:
                pieces.extend(joins)
                extended_ends.append((w.id, name))

    solid = unary_union(pieces)

    # Cut door/gap openings through the full wall thickness (+ a hair so
    # the subtraction is numerically clean), leaving capped jambs.
    cutters = []
    for op in floor.openings:
        if op.kind not in GAP_KINDS:
            continue
        rect = _opening_rect(op, floor)
        if rect is not None:
            cutters.append(rect)
    if cutters:
        solid = solid.difference(unary_union(cutters))

    return WallOutline(rings=_polygon_rings(solid), extended_ends=extended_ends)


def _opening_rect(op: Opening, floor: FloorGeometry) -> ShapelyPolygon | None:
    """Rectangle covering the opening across its wall's full thickness."""
    seg = np.asarray(op.segment, dtype=float).reshape(2, 2)
    d = seg[1] - seg[0]
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return None
    u = d / length
    n = np.array([-u[1], u[0]])
    wall = floor.wall_by_id(op.wall_id)
    thickness = wall.thickness_m if wall is not None else 0.30
    half = thickness / 2.0 + 0.01
    return ShapelyPolygon([
        seg[0] + n * half, seg[1] + n * half,
        seg[1] - n * half, seg[0] - n * half,
    ])


def _polygon_rings(geom) -> list[np.ndarray]:
    """All boundary rings (exterior + interiors) of a (Multi)Polygon.

    Rings are returned without the closing duplicate vertex.
    """
    if geom.is_empty:
        return []
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    rings: list[np.ndarray] = []
    for poly in polys:
        rings.append(np.asarray(poly.exterior.coords[:-1], dtype=float))
        for hole in poly.interiors:
            rings.append(np.asarray(hole.coords[:-1], dtype=float))
    return rings


# ── Dominant wall axis (page rotation) ────────────────────────────────────────

def dominant_wall_axis_rad(walls: list[Wall]) -> float:
    """Length-weighted dominant wall axis, folded into [−45°, +45°).

    Uses the standard 4θ circular mean (a rectangular grid is invariant
    under 90° rotation, so angles are compared modulo 90°).  Rotating the
    plan by the NEGATIVE of this angle puts its dominant axes parallel to
    the page edges.  Returns 0.0 when there are no usable walls.
    """
    s4 = c4 = 0.0
    for w in walls:
        d = np.asarray(w.centerline[1], dtype=float) - np.asarray(w.centerline[0], dtype=float)
        length = float(np.hypot(d[0], d[1]))
        if length < 1e-9:
            continue
        theta = float(np.arctan2(d[1], d[0]))
        s4 += length * np.sin(4.0 * theta)
        c4 += length * np.cos(4.0 * theta)
    if abs(s4) < 1e-12 and abs(c4) < 1e-12:
        return 0.0
    axis = float(np.arctan2(s4, c4)) / 4.0            # in (−45°, 45°]
    # Fold to [−45°, +45°) for a minimal page rotation.
    quarter = np.pi / 2.0
    axis = (axis + quarter / 2.0) % quarter - quarter / 2.0
    return axis
