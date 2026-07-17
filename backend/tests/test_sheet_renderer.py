"""Phase 3: SVG sheet renderer (app.sheet.render / labels / pdf).

All assertions are on parsed SVG structure and exact coordinates — never on
pixel output.  Layout numbers below are hand-computed from the documented
A1-landscape sheet parameters:

  page 841×594 mm, margin 12, grid gutter 16, inner pad 4, title block 40
  drawing area: x ∈ [28, 825] (797 mm), y ∈ [28, 538] (510 mm)

Reference floor (``two_suite_floor``): envelope (−0.15, −0.15)…(20.15, 10.15)
→ 20.3 × 10.3 m.  At 1:50 that is 406 × 206 mm — fits → scale 1:50,
s = 20 mm/m, offsets:

  offset_x = 28 + (797 − 406) / 2 = 223.5
  offset_y = 28 + (510 − 206) / 2 = 180.0
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from app.geometry.model import (
    ColumnFeature,
    FloorGeometry,
    Opening,
    Wall,
    WallClass,
)
from app.sheet.grid import manual_grid
from app.sheet.labels import PlacedLabel, resolve_label_collisions
from app.sheet.render import (
    STANDARD_SCALES,
    WALL_STROKE_MM,
    SheetMetadata,
    SheetParams,
    choose_scale,
    floor_bounds,
    render_sheet,
    split_wall_at_openings,
)
from tests.reference_floors import two_suite_floor

# Hand-computed layout constants (see module docstring).
EXPECTED_SCALE_DENOM = 50
EXPECTED_MM_PER_M = 20.0
EXPECTED_OFFSET_X = 223.5
EXPECTED_OFFSET_Y = 180.0


def _render_two_suite(**kwargs):
    floor = two_suite_floor()
    meta = kwargs.pop("meta", SheetMetadata(
        building_name="Demo Tower",
        address="41 Demo Street, Springfield",
        floor_name="Level 3",
        date_str="2026-07-05",
    ))
    params = kwargs.pop("params", SheetParams(style="full"))
    render = render_sheet(floor, meta=meta, params=params, **kwargs)
    return render, ET.fromstring(render.svg)


def _group(root: ET.Element, gid: str) -> ET.Element:
    for el in root.iter():
        if el.get("id") == gid:
            return el
    raise AssertionError(f"element #{gid} not found in SVG")


def _texts(el: ET.Element) -> list[str]:
    return [t.text for t in el.iter() if t.tag.endswith("text")]


# ── Transform + scale selection ───────────────────────────────────────────────

class TestScaleAndTransform:
    def test_standard_scale_chosen(self):
        render, _ = _render_two_suite()
        assert render.scale_denominator == EXPECTED_SCALE_DENOM
        assert render.mm_per_m == pytest.approx(EXPECTED_MM_PER_M, abs=1e-9)

    def test_offsets_hand_computed(self):
        render, _ = _render_two_suite()
        assert render.offset_x_mm == pytest.approx(EXPECTED_OFFSET_X, abs=1e-9)
        assert render.offset_y_mm == pytest.approx(EXPECTED_OFFSET_Y, abs=1e-9)

    def test_world_to_mm_formula(self):
        render, _ = _render_two_suite()
        # World (8, 0): X = 223.5 + 20·(8 − (−0.15)) = 386.5
        #               Y = 180 + 20·(10.15 − 0)     = 383.0
        assert render.world_to_mm(8.0, 0.0) == pytest.approx((386.5, 383.0), abs=1e-9)
        assert render.world_to_mm(8.0, 10.0) == pytest.approx((386.5, 183.0), abs=1e-9)

    def test_choose_scale_fallback_beyond_standards(self):
        # A 1 km building fits no standard scale on this paper: needs
        # 1_000_000/797 ≈ 1254.7 → rounded up to 1300.
        assert choose_scale(1000.0, 10.0, 797.0, 510.0) == 1300

    def test_choose_scale_prefers_largest_drawing(self):
        assert choose_scale(20.3, 10.3, 797.0, 510.0) == 50
        assert choose_scale(60.0, 30.0, 797.0, 510.0) == 100
        assert STANDARD_SCALES[0] == 50

    def test_floor_bounds_from_envelope(self):
        assert floor_bounds(two_suite_floor()) == pytest.approx(
            (-0.15, -0.15, 20.15, 10.15), abs=1e-9
        )


# ── Page furniture ────────────────────────────────────────────────────────────

class TestPageFurniture:
    def test_page_frame(self):
        _, root = _render_two_suite()
        frame = _group(root, "page-frame")
        assert frame.tag.endswith("rect")
        assert float(frame.get("x")) == 12.0
        assert float(frame.get("y")) == 12.0
        assert float(frame.get("width")) == 841.0 - 24.0
        assert float(frame.get("height")) == 594.0 - 24.0

    def test_title_block_fields(self):
        _, root = _render_two_suite()
        assert _group(root, "tb-building").text == "Demo Tower"
        assert _group(root, "tb-address").text == "41 Demo Street, Springfield"
        assert _group(root, "tb-floor").text == "Level 3"
        assert _group(root, "tb-scale").text == "SCALE 1:50"
        assert _group(root, "tb-date").text == "DATE 2026-07-05"

    def test_title_block_falls_back_to_floor_geometry_names(self):
        floor = two_suite_floor()
        floor.building_name = "From Geometry"
        floor.floor_name = "Floor 9"
        render = render_sheet(
            floor, meta=SheetMetadata(), params=SheetParams(style="full"),
        )
        root = ET.fromstring(render.svg)
        assert _group(root, "tb-building").text == "From Geometry"
        assert _group(root, "tb-floor").text == "Floor 9"

    def test_scale_bar_divisions_sum_to_bar_length(self):
        # s = 20 mm/m → longest nice bar ≤ 100 mm is 5 m = 100 mm; 4 divisions.
        _, root = _render_two_suite()
        bar = _group(root, "scale-bar")
        rects = [e for e in bar.iter() if e.tag.endswith("rect")]
        assert len(rects) == 4
        widths = [float(r.get("width")) for r in rects]
        assert widths == pytest.approx([25.0] * 4, abs=1e-9)
        labels = _texts(bar)
        assert "0" in labels and "5 m" in labels

    def test_north_arrow_rotation(self):
        render, root = _render_two_suite(
            meta=SheetMetadata(north_angle_deg=37.5),
        )
        arrow = _group(root, "north-arrow")
        assert arrow.get("transform").startswith("rotate(37.5 ")
        assert "N" in _texts(arrow)


# ── Walls (legacy centerline mode): line-weight hierarchy + door gaps ─────────
# The drafting-quality double-line mode (Phase 3.5 default) is covered in
# test_drafting_walls.py; these tests pin the preserved legacy mode.

class TestWalls:
    def test_line_weight_hierarchy(self):
        floor = two_suite_floor()
        floor.walls.append(Wall(
            id="w_partition",
            centerline=np.array([[2.0, 2.0], [2.0, 8.0]]),
            thickness_m=0.10,
            wall_class=WallClass.PARTITION,
        ))
        floor = FloorGeometry(
            floor_id=floor.floor_id, walls=floor.walls, rooms=floor.rooms,
            envelope=floor.envelope,
        )
        render = render_sheet(floor, wall_style="centerline")
        root = ET.fromstring(render.svg)
        walls_g = _group(root, "walls")
        widths = {}
        for line in walls_g:
            cls = line.get("class")
            widths[cls] = float(line.get("stroke-width"))
        assert widths["wall wall-exterior"] == pytest.approx(0.70)
        assert widths["wall wall-demising"] == pytest.approx(0.40)
        assert widths["wall wall-partition"] == pytest.approx(0.18)
        assert (
            widths["wall wall-exterior"]
            > widths["wall wall-demising"]
            > widths["wall wall-partition"]
        )
        assert WALL_STROKE_MM[WallClass.EXTERIOR] == 0.70

    def test_demising_wall_exact_coordinates(self):
        _, root = _render_two_suite(wall_style="centerline")
        walls_g = _group(root, "walls")
        demising = [
            line for line in walls_g
            if line.get("class") == "wall wall-demising"
        ]
        assert len(demising) == 1
        line = demising[0]
        # Centerline (8,0)→(8,10) at the hand-computed transform.
        assert float(line.get("x1")) == pytest.approx(386.5, abs=1e-6)
        assert float(line.get("y1")) == pytest.approx(383.0, abs=1e-6)
        assert float(line.get("x2")) == pytest.approx(386.5, abs=1e-6)
        assert float(line.get("y2")) == pytest.approx(183.0, abs=1e-6)

    def test_split_wall_at_openings_exact(self):
        wall = Wall(
            id="w",
            centerline=np.array([[0.0, 0.0], [20.0, 0.0]]),
            thickness_m=0.3,
            wall_class=WallClass.EXTERIOR,
        )
        door = Opening(
            id="d1", segment=np.array([[4.0, 0.0], [5.0, 0.0]]),
            width_m=1.0, kind="door", wall_id="w",
        )
        kept = split_wall_at_openings(wall, [door])
        assert len(kept) == 2
        assert kept[0] == pytest.approx(np.array([[0, 0], [4, 0]]), abs=1e-9)
        assert kept[1] == pytest.approx(np.array([[5, 0], [20, 0]]), abs=1e-9)

    def test_split_merges_overlapping_gaps(self):
        wall = Wall(
            id="w", centerline=np.array([[0.0, 0.0], [10.0, 0.0]]),
            thickness_m=0.1,
        )
        doors = [
            Opening(id="d1", segment=np.array([[2.0, 0.0], [3.0, 0.0]]),
                    width_m=1.0, kind="door", wall_id="w"),
            Opening(id="d2", segment=np.array([[2.5, 0.0], [4.0, 0.0]]),
                    width_m=1.5, kind="gap", wall_id="w"),
        ]
        kept = split_wall_at_openings(wall, doors)
        assert len(kept) == 2
        assert kept[0] == pytest.approx(np.array([[0, 0], [2, 0]]), abs=1e-9)
        assert kept[1] == pytest.approx(np.array([[4, 0], [10, 0]]), abs=1e-9)

    def test_window_does_not_cut_wall(self):
        wall = Wall(
            id="w", centerline=np.array([[0.0, 0.0], [10.0, 0.0]]),
            thickness_m=0.1,
        )
        window = Opening(
            id="win", segment=np.array([[4.0, 0.0], [6.0, 0.0]]),
            width_m=2.0, kind="window", wall_id="w",
        )
        kept = split_wall_at_openings(wall, [window])
        assert len(kept) == 1
        assert kept[0] == pytest.approx(np.array([[0, 0], [10, 0]]), abs=1e-9)

    def test_door_gap_rendered_open_in_svg(self):
        floor = two_suite_floor()
        floor.openings.append(Opening(
            id="d1", segment=np.array([[4.0, 0.0], [5.0, 0.0]]),
            width_m=1.0, kind="door", wall_id="w_bottom",
        ))
        render = render_sheet(
            floor, wall_style="centerline", params=SheetParams(style="full"),
        )
        root = ET.fromstring(render.svg)
        walls_g = _group(root, "walls")
        # 5 walls, one with a door → 6 line elements.
        lines = [e for e in walls_g if e.tag.endswith("line")]
        assert len(lines) == 6
        bottom = [
            line for line in lines
            if abs(float(line.get("y1")) - 383.0) < 1e-6
            and abs(float(line.get("y2")) - 383.0) < 1e-6
        ]
        assert len(bottom) == 2
        xs = sorted(
            tuple(sorted((float(line.get("x1")), float(line.get("x2")))))
            for line in bottom
        )
        # X(x) = 223.5 + 20·(x + 0.15): the bottom wall (0→20) splits at the
        # door (4→5) into [226.5, 306.5] and [326.5, 626.5] mm.
        assert xs[0] == pytest.approx((226.5, 306.5), abs=1e-6)
        assert xs[1] == pytest.approx((326.5, 626.5), abs=1e-6)


# ── Columns + penetrations ────────────────────────────────────────────────────

class TestColumnsAndPenetrations:
    def test_rect_column_polygon(self):
        floor = two_suite_floor()
        floor.columns.append(ColumnFeature(
            id="c1", centre=(6.0, 5.0),
            polygon=np.array([[5.8, 4.8], [6.2, 4.8], [6.2, 5.2], [5.8, 5.2]]),
        ))
        render = render_sheet(floor)
        root = ET.fromstring(render.svg)
        cols_g = _group(root, "columns")
        polys = [e for e in cols_g if e.tag.endswith("polygon")]
        assert len(polys) == 1
        first_pt = polys[0].get("points").split(" ")[0]
        x, y = (float(v) for v in first_pt.split(","))
        assert (x, y) == pytest.approx(render.world_to_mm(5.8, 4.8), abs=1e-6)

    def test_round_column_circle(self):
        floor = two_suite_floor()
        floor.columns.append(ColumnFeature(
            id="c2", centre=(10.0, 5.0), radius_m=0.25, is_round=True,
        ))
        render = render_sheet(floor)
        root = ET.fromstring(render.svg)
        cols_g = _group(root, "columns")
        circles = [e for e in cols_g if e.tag.endswith("circle")]
        assert len(circles) == 1
        assert float(circles[0].get("r")) == pytest.approx(0.25 * 20.0, abs=1e-9)

    def test_penetration_polygon_rendered(self):
        from tests.reference_floors import corridor_floor
        render = render_sheet(corridor_floor(with_shaft=True))
        root = ET.fromstring(render.svg)
        pen_g = _group(root, "penetrations")
        polys = [e for e in pen_g if e.tag.endswith("polygon")]
        assert len(polys) == 1
        assert "penetration-elevator_shaft" in polys[0].get("class")


# ── Column grid on the sheet ──────────────────────────────────────────────────

class TestGridOnSheet:
    def test_grid_lines_and_bubbles(self):
        grid = manual_grid([0.0, 10.0, 20.0], [0.0, 10.0])
        _, root = _render_two_suite(grid=grid)
        grid_g = _group(root, "grid")
        lines = [e for e in grid_g if e.tag.endswith("line")]
        assert len(lines) == 5
        bubbles = [e for e in grid_g if e.get("class") == "grid-bubble"]
        labels = set()
        for b in bubbles:
            labels.update(t for t in _texts(b) if t)
        assert labels == {"1", "2", "3", "A", "B"}

    def test_no_grid_group_empty(self):
        _, root = _render_two_suite(grid=None)
        grid_g = _group(root, "grid")
        assert len(list(grid_g)) == 0

    def test_grid_lines_dashed(self):
        grid = manual_grid([0.0, 10.0], [0.0])
        _, root = _render_two_suite(grid=grid)
        grid_g = _group(root, "grid")
        for line in (e for e in grid_g if e.tag.endswith("line")):
            assert line.get("stroke-dasharray") is not None


# ── Suite labels ──────────────────────────────────────────────────────────────

class TestSuiteLabels:
    def test_labels_at_representative_points(self):
        render, root = _render_two_suite()
        labels_g = _group(root, "suite-labels")
        by_room = {g.get("data-room-id"): g for g in labels_g}
        assert set(by_room) == {"suite_a", "suite_b"}
        name_a = [t for t in by_room["suite_a"].iter() if t.tag.endswith("text")][0]
        assert name_a.text == "Suite A"
        # Rectangle rooms: representative_point == centre (4, 5) / (14, 5).
        assert float(name_a.get("x")) == pytest.approx(
            render.world_to_mm(4.0, 5.0)[0], abs=1e-6
        )
        assert float(name_a.get("y")) == pytest.approx(
            render.world_to_mm(4.0, 5.0)[1], abs=1e-6
        )

    def test_label_area_line(self):
        _, root = _render_two_suite()
        labels_g = _group(root, "suite-labels")
        area_texts = [
            t.text for t in labels_g.iter()
            if t.tag.endswith("text") and t.get("class") == "suite-label-area"
        ]
        assert "80.0 m²" in area_texts    # suite_a: 8 × 10 as drawn
        assert "120.0 m²" in area_texts   # suite_b: 12 × 10 as drawn

    def test_label_override_applied(self):
        _, root = _render_two_suite(label_overrides={"suite_a": "Suite 301"})
        labels_g = _group(root, "suite-labels")
        names = [
            t.text for t in labels_g.iter()
            if t.tag.endswith("text") and t.get("class") == "suite-label-name"
        ]
        assert "Suite 301" in names
        assert "Suite A" not in names

    def test_suite_grouping_single_label(self):
        floor = two_suite_floor()
        floor.rooms[0].suite_id = "S-1"
        floor.rooms[1].suite_id = "S-1"
        render = render_sheet(floor)
        root = ET.fromstring(render.svg)
        labels_g = _group(root, "suite-labels")
        groups = list(labels_g)
        assert len(groups) == 1
        # Placed on the LARGER room of the suite (suite_b, 120 m²).
        assert groups[0].get("data-room-id") == "suite_b"

    def test_collision_resolution_separates_boxes(self):
        a = PlacedLabel("a", "Suite 100", "80.0 m²", 100.0, 100.0, 24.0, 8.0)
        b = PlacedLabel("b", "Suite 200", "80.0 m²", 100.0, 100.0, 24.0, 8.0)
        resolve_label_collisions([a, b])
        ax0, ay0, ax1, ay1 = a.bounds()
        bx0, by0, bx1, by1 = b.bounds()
        overlap = ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
        assert not overlap
        # Coincident centres split vertically, x untouched.
        assert a.x_mm == pytest.approx(100.0, abs=1e-9)
        assert b.x_mm == pytest.approx(100.0, abs=1e-9)
        assert a.y_mm < b.y_mm

    def test_non_overlapping_labels_not_moved(self):
        a = PlacedLabel("a", "X", "", 0.0, 0.0, 10.0, 5.0)
        b = PlacedLabel("b", "Y", "", 100.0, 0.0, 10.0, 5.0)
        resolve_label_collisions([a, b])
        assert (a.x_mm, a.y_mm) == (0.0, 0.0)
        assert (b.x_mm, b.y_mm) == (100.0, 0.0)

    def test_common_room_label_class(self):
        from tests.reference_floors import corridor_floor
        render = render_sheet(corridor_floor())
        root = ET.fromstring(render.svg)
        labels_g = _group(root, "suite-labels")
        corridor = [g for g in labels_g if g.get("data-room-id") == "corridor"]
        assert len(corridor) == 1
        assert "suite-label-common" in corridor[0].get("class")


# ── PDF conversion ────────────────────────────────────────────────────────────

class TestPdfConversion:
    def test_svg_to_pdf_writes_valid_pdf(self, tmp_path):
        from app.sheet.pdf import svg_to_pdf
        render, _ = _render_two_suite()
        out = svg_to_pdf(render.svg, tmp_path / "sheet.pdf")
        data = out.read_bytes()
        assert data[:5] == b"%PDF-"
        assert len(data) > 1000
