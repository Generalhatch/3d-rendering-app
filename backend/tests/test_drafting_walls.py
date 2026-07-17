"""Phase 3.5 (drafting-walls): double-line walls from measured faces,
render-stage junction closure, and page rotation to dominant wall axes.

All numbers hand-computed:

Two-suite reference floor (see tests/reference_floors.py): exterior walls
t=0.30 (faces at centerline ± 0.15), demising t=0.20 at x=8.  With mitred
junction closure the union of wall rectangles is the closed shell band +
demising bar, so its boundary is exactly:

  exterior ring:  (-0.15, -0.15) … (20.15, 10.15)
  hole (suite_a): ( 0.15,  0.15) … ( 7.90,  9.85)
  hole (suite_b): ( 8.10,  0.15) … (19.85,  9.85)

L-junction closure example: wall A (0,0)→(5,0), wall B (5,0.05)→(5,4),
both t=0.20.  B's start extends 0.15 to A's far face (y=−0.10); A's end
extends 0.10 to B's far face (x=5.10).  Union area:
  A: 5.10 × 0.20 = 1.02;  B: (4 − (−0.10)) × 0.20 = 0.82
  overlap: 0.20 × 0.20 = 0.04  →  union = 1.80 m², one closed ring.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from app.geometry.model import FloorGeometry, Opening, Room, Wall, WallClass, WallFaceRef
from app.sheet.drafting import (
    DEFAULT_JOIN_EXT_M,
    build_wall_outline,
    dominant_wall_axis_rad,
    wall_rect,
)
from app.sheet.render import SheetMetadata, render_sheet
from tests.reference_floors import two_suite_floor


def _ring_bbox(ring: np.ndarray) -> tuple[float, float, float, float]:
    return (
        round(float(ring[:, 0].min()), 6), round(float(ring[:, 1].min()), 6),
        round(float(ring[:, 0].max()), 6), round(float(ring[:, 1].max()), 6),
    )


def _wall(wid, p, q, t=0.20, cls=WallClass.PARTITION):
    return Wall(id=wid, centerline=np.array([p, q], dtype=float),
                thickness_m=t, wall_class=cls)


# ── wall_rect ─────────────────────────────────────────────────────────────────

class TestWallRect:
    def test_rect_spans_measured_faces(self):
        w = _wall("w", (0, 0), (10, 0), t=0.30)
        rect = wall_rect(w)
        assert rect.bounds == pytest.approx((0.0, -0.15, 10.0, 0.15), abs=1e-12)
        assert rect.area == pytest.approx(3.0, abs=1e-12)

    def test_rect_extension(self):
        w = _wall("w", (0, 0), (10, 0), t=0.30)
        rect = wall_rect(w, ext_start_m=0.2, ext_end_m=0.5)
        assert rect.bounds == pytest.approx((-0.2, -0.15, 10.5, 0.15), abs=1e-12)


# ── Junction closure ──────────────────────────────────────────────────────────

class TestJunctionClosure:
    def test_l_junction_gap_closed_exactly(self):
        floor = FloorGeometry(floor_id="l-gap", walls=[
            _wall("a", (0, 0), (5, 0)),
            _wall("b", (5, 0.05), (5, 4)),
        ])
        outline = build_wall_outline(floor)
        assert len(outline.rings) == 1          # one closed L — gap healed
        poly = ShapelyPolygon(outline.rings[0])
        assert poly.area == pytest.approx(1.80, abs=1e-9)
        # Both closing extensions recorded.
        assert ("b", "start") in outline.extended_ends
        assert ("a", "end") in outline.extended_ends

    def test_free_standing_end_not_extended(self):
        floor = FloorGeometry(floor_id="solo", walls=[
            _wall("only", (0, 0), (10, 0), t=0.30),
        ])
        outline = build_wall_outline(floor)
        assert len(outline.rings) == 1
        assert outline.extended_ends == []
        assert _ring_bbox(outline.rings[0]) == (0.0, -0.15, 10.0, 0.15)

    def test_gap_beyond_reach_stays_open(self):
        floor = FloorGeometry(floor_id="far-gap", walls=[
            _wall("a", (0, 0), (5, 0)),
            _wall("b", (5.0 + DEFAULT_JOIN_EXT_M + 0.2, 0), (9, 0)),
        ])
        outline = build_wall_outline(floor)
        assert len(outline.rings) == 2          # honest gap: NOT bridged
        assert outline.extended_ends == []

    def test_two_suite_outline_exact_rings(self):
        outline = build_wall_outline(two_suite_floor())
        bboxes = sorted(_ring_bbox(r) for r in outline.rings)
        assert bboxes == [
            (-0.15, -0.15, 20.15, 10.15),       # shell outer face
            (0.15, 0.15, 7.9, 9.85),            # suite_a inner faces
            (8.1, 0.15, 19.85, 9.85),           # suite_b inner faces
        ]

    def test_door_cut_through_full_thickness(self):
        floor = FloorGeometry(
            floor_id="door",
            walls=[_wall("w", (0, 0), (10, 0), t=0.30)],
            openings=[Opening(
                id="d", segment=np.array([[4.0, 0.0], [5.0, 0.0]]),
                width_m=1.0, kind="door", wall_id="w",
            )],
        )
        outline = build_wall_outline(floor)
        bboxes = sorted(_ring_bbox(r) for r in outline.rings)
        assert bboxes == [
            (0.0, -0.15, 4.0, 0.15),
            (5.0, -0.15, 10.0, 0.15),
        ]

    def test_window_does_not_cut(self):
        floor = FloorGeometry(
            floor_id="win",
            walls=[_wall("w", (0, 0), (10, 0), t=0.30)],
            openings=[Opening(
                id="win", segment=np.array([[4.0, 0.0], [6.0, 0.0]]),
                width_m=2.0, kind="window", wall_id="w",
            )],
        )
        outline = build_wall_outline(floor)
        assert len(outline.rings) == 1
        assert _ring_bbox(outline.rings[0]) == (0.0, -0.15, 10.0, 0.15)


# ── Double-line rendering in the SVG ──────────────────────────────────────────

class TestDoubleLineSvg:
    def test_walls_group_contains_outline_paths(self):
        render = render_sheet(two_suite_floor())    # wall_style="double" default
        root = ET.fromstring(render.svg)
        walls_g = next(el for el in root.iter() if el.get("id") == "walls")
        paths = [e for e in walls_g if e.tag.endswith("path")]
        assert len(paths) == 3                      # shell + two room holes
        for p in paths:
            assert p.get("class") == "wall-outline"
            assert p.get("fill") == "none"
            assert float(p.get("stroke-width")) == pytest.approx(0.15)
            assert p.get("d").startswith("M ")
            assert p.get("d").endswith(" Z")

    def test_outline_corner_exact_paper_coordinates(self):
        render = render_sheet(two_suite_floor())
        root = ET.fromstring(render.svg)
        walls_g = next(el for el in root.iter() if el.get("id") == "walls")
        # The shell ring must pass through the outer corner (-0.15, -0.15)
        # stevenson_minimal default: X = 217.5, Y = 187 + 20·10.3 = 393.
        corner = render.world_to_mm(-0.15, -0.15)
        assert corner == pytest.approx((217.5, 393.0), abs=1e-9)
        all_d = " ".join(p.get("d") for p in walls_g)
        assert "217.5 393" in all_d


# ── Dominant-axis page rotation ───────────────────────────────────────────────

def _rotated_two_suite(deg: float) -> FloorGeometry:
    """The two-suite reference floor rigidly rotated about the origin."""
    th = math.radians(deg)
    r_mat = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])

    def rot(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=float) @ r_mat.T

    base = two_suite_floor()
    walls = [
        Wall(id=w.id, centerline=rot(w.centerline), thickness_m=w.thickness_m,
             wall_class=w.wall_class)
        for w in base.walls
    ]
    rooms = [
        Room(id=r.id, boundary=rot(r.boundary),
             wall_refs=[WallFaceRef(ref.wall_id) for ref in r.wall_refs],
             boundary_basis=r.boundary_basis, label=r.label)
        for r in base.rooms
    ]
    return FloorGeometry(
        floor_id=base.floor_id, walls=walls, rooms=rooms,
        envelope=rot(base.envelope),
    )


class TestPageRotation:
    def test_dominant_axis_of_axis_aligned_floor_is_zero(self):
        assert dominant_wall_axis_rad(two_suite_floor().walls) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_dominant_axis_of_rotated_floor(self):
        floor = _rotated_two_suite(30.0)
        assert dominant_wall_axis_rad(floor.walls) == pytest.approx(
            math.radians(30.0), abs=1e-9
        )

    def test_dominant_axis_folds_to_minimal_rotation(self):
        # 60° ≡ −30° modulo the 90° grid symmetry: minimal page rotation.
        floor = _rotated_two_suite(60.0)
        assert dominant_wall_axis_rad(floor.walls) == pytest.approx(
            math.radians(-30.0), abs=1e-9
        )

    def test_rotated_floor_renders_identically_to_unrotated(self):
        base_render = render_sheet(two_suite_floor())
        rot_render = render_sheet(_rotated_two_suite(30.0))
        assert rot_render.rotation_rad == pytest.approx(math.radians(30.0), abs=1e-9)
        # Same paper layout: bounds in the rotated frame match the
        # unrotated floor, so scale + offsets are identical.
        assert rot_render.scale_denominator == base_render.scale_denominator
        assert rot_render.offset_x_mm == pytest.approx(base_render.offset_x_mm, abs=1e-6)
        assert rot_render.offset_y_mm == pytest.approx(base_render.offset_y_mm, abs=1e-6)
        # A rotated world point lands exactly where the unrotated one did.
        th = math.radians(30.0)
        p = (8.0, 10.0)
        p_rot = (
            p[0] * math.cos(th) - p[1] * math.sin(th),
            p[0] * math.sin(th) + p[1] * math.cos(th),
        )
        assert rot_render.world_to_mm(*p_rot) == pytest.approx(
            base_render.world_to_mm(*p), abs=1e-6
        )

    def test_north_arrow_carries_true_orientation(self):
        render = render_sheet(
            _rotated_two_suite(30.0), meta=SheetMetadata(north_angle_deg=10.0),
        )
        root = ET.fromstring(render.svg)
        arrow = next(el for el in root.iter() if el.get("id") == "north-arrow")
        angle = float(arrow.get("transform").split("(")[1].split(" ")[0])
        assert angle == pytest.approx(40.0, abs=1e-6)   # 10 true + 30 page

    def test_rotation_disabled_via_params(self):
        from app.sheet.render import SheetParams
        render = render_sheet(
            _rotated_two_suite(30.0),
            params=SheetParams(rotate_to_axes=False),
        )
        assert render.rotation_rad == 0.0
