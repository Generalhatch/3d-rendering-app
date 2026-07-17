"""Phase 3.5 (drafting-suites): heavy suite-perimeter outlines + the
"stevenson_minimal" sheet style (four-edge tick grid, footer instead of
the A1 title block, large light-gray suite labels).

Layout numbers for the stevenson_minimal style, hand-computed:

  page 841×594, margin 12, gutter 16, footer 14
  drawing area: x ∈ [28, 813] (785 mm), y ∈ [28, 552] (524 mm)

Two-suite floor is 20.3 × 10.3 m → at 1:50 that is 406 × 206 mm — fits →
scale 1:50, s = 20 mm/m:

  offset_x = 28 + (785 − 406) / 2 = 217.5
  offset_y = 28 + (524 − 206) / 2 = 187.0
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from app.geometry.model import FloorGeometry, Room, WallFaceRef
from app.sheet.grid import manual_grid
from app.sheet.render import (
    SUITE_OUTLINE_STROKE_MM,
    SheetMetadata,
    SheetParams,
    render_sheet,
    suite_outline_rings,
)
from tests.reference_floors import corridor_floor, two_suite_floor


def _group(root: ET.Element, gid: str) -> ET.Element:
    for el in root.iter():
        if el.get("id") == gid:
            return el
    raise AssertionError(f"element #{gid} not found in SVG")


def _no_group(root: ET.Element, gid: str) -> bool:
    return all(el.get("id") != gid for el in root.iter())


STEVENSON = SheetParams(style="stevenson_minimal")


# ── Suite-perimeter outlines ──────────────────────────────────────────────────

class TestSuiteOutlines:
    def test_one_ring_per_suite(self):
        rings = suite_outline_rings(two_suite_floor())
        keys = sorted(k for k, _ in rings)
        assert keys == ["suite_a", "suite_b"]

    def test_rooms_sharing_suite_id_union_into_one_ring(self):
        floor = two_suite_floor()
        floor.rooms[0].suite_id = "S-1"
        floor.rooms[1].suite_id = "S-1"
        rings = suite_outline_rings(floor)
        assert len(rings) == 1
        key, ring = rings[0]
        assert key == "S-1"
        # Union of the two abutting rooms = the full 20×10 rectangle.
        xs, ys = ring[:, 0], ring[:, 1]
        assert (xs.min(), ys.min(), xs.max(), ys.max()) == pytest.approx(
            (0.0, 0.0, 20.0, 10.0), abs=1e-9
        )
        assert len(ring) == 4     # the shared demising edge dissolved

    def test_common_rooms_get_no_heavy_outline(self):
        rings = suite_outline_rings(corridor_floor())
        keys = sorted(k for k, _ in rings)
        assert keys == ["suite_a", "suite_b"]     # no "corridor"

    def test_svg_outline_heavy_stroke_exact_coordinates(self):
        render = render_sheet(two_suite_floor())
        root = ET.fromstring(render.svg)
        g = _group(root, "suite-outlines")
        polys = [e for e in g if e.tag.endswith("polygon")]
        assert len(polys) == 2
        by_suite = {p.get("data-suite-id"): p for p in polys}
        assert set(by_suite) == {"suite_a", "suite_b"}
        for p in polys:
            assert float(p.get("stroke-width")) == pytest.approx(
                SUITE_OUTLINE_STROKE_MM
            )
            assert p.get("fill") == "none"
        # suite_a corner (0,0) → stevenson_minimal transform (default style):
        # X = 217.5 + 20·0.15 = 220.5, Y = 187 + 20·10.15 = 390.
        pts = by_suite["suite_a"].get("points").split(" ")
        coords = [tuple(float(v) for v in pt.split(",")) for pt in pts]
        assert render.world_to_mm(0.0, 0.0) == pytest.approx((220.5, 390.0), abs=1e-9)

    def test_suite_outline_heavier_than_wall_lines(self):
        from app.sheet.render import DOUBLE_WALL_STROKE_MM
        assert SUITE_OUTLINE_STROKE_MM > 4 * DOUBLE_WALL_STROKE_MM


# ── Stevenson-minimal page furniture ──────────────────────────────────────────

class TestStevensonMinimalStyle:
    def test_transform_hand_computed(self):
        render = render_sheet(two_suite_floor(), params=STEVENSON)
        assert render.scale_denominator == 50
        assert render.offset_x_mm == pytest.approx(217.5, abs=1e-9)
        assert render.offset_y_mm == pytest.approx(187.0, abs=1e-9)

    def test_no_frame_no_title_block_no_scale_bar(self):
        render = render_sheet(two_suite_floor(), params=STEVENSON)
        root = ET.fromstring(render.svg)
        assert _no_group(root, "page-frame")
        assert _no_group(root, "title-block")
        assert _no_group(root, "scale-bar")

    def test_footer_content(self):
        render = render_sheet(
            two_suite_floor(),
            meta=SheetMetadata(
                building_name="321 Kinzua Road",
                floor_name="Floor 1",
                address="321 Kinzua Road, Buffalo, NY 14205",
                date_str="2026-07-06",
            ),
            params=STEVENSON,
        )
        root = ET.fromstring(render.svg)
        assert _group(root, "footer-building").text == "321 Kinzua Road - Floor 1"
        addr = _group(root, "footer-address")
        assert addr.text == "321 Kinzua Road, Buffalo, NY 14205"
        assert addr.get("text-anchor") == "middle"
        assert _group(root, "footer-scale").text == "SCALE 1:50"
        assert _group(root, "footer-date").text == "2026-07-06"

    def test_full_style_unchanged(self):
        from app.sheet.render import SheetParams
        render = render_sheet(two_suite_floor(), params=SheetParams(style="full"))
        root = ET.fromstring(render.svg)
        assert _group(root, "page-frame") is not None
        assert _group(root, "title-block") is not None
        assert _no_group(root, "footer")


class TestStevensonTickGrid:
    def test_ticks_on_all_four_edges_no_bubbles_no_lines(self):
        grid = manual_grid([0.0, 10.0, 20.0], [0.0, 10.0])
        render = render_sheet(two_suite_floor(), grid=grid, params=STEVENSON)
        root = ET.fromstring(render.svg)
        g = _group(root, "grid")
        ticks = [e for e in g if (e.get("class") or "").startswith("grid-tick")]
        # 3 u-lines cross top+bottom, 2 v-lines cross left+right → 10 ticks.
        assert len(ticks) == 10
        assert all(not (e.get("class") or "").startswith("grid-bubble") for e in g)
        assert all(not (e.get("class") or "").startswith("grid-line") for e in g)
        # Every grid label appears exactly twice (once per opposite edge).
        labels = [t.get("data-label") for t in ticks]
        assert sorted(labels) == ["1", "1", "2", "2", "3", "3", "A", "A", "B", "B"]

    def test_tick_x_positions_exact(self):
        grid = manual_grid([0.0, 20.0], [])
        render = render_sheet(two_suite_floor(), grid=grid, params=STEVENSON)
        root = ET.fromstring(render.svg)
        g = _group(root, "grid")
        ticks = [e for e in g if (e.get("class") or "").startswith("grid-tick")]
        xs = sorted({
            float(line.get("x1"))
            for t in ticks for line in t if line.tag.endswith("line")
        })
        # u=0 → X = 217.5 + 20·0.15 = 220.5;  u=20 → X = 220.5 + 400 = 620.5.
        assert xs == pytest.approx([220.5, 620.5], abs=1e-6)

    def test_ticks_light_gray(self):
        grid = manual_grid([0.0], [0.0])
        render = render_sheet(two_suite_floor(), grid=grid, params=STEVENSON)
        root = ET.fromstring(render.svg)
        for t in _group(root, "grid"):
            for line in (e for e in t if e.tag.endswith("line")):
                assert line.get("stroke") == "#aaaaaa"

    def test_full_style_keeps_bubbles(self):
        from app.sheet.render import SheetParams
        grid = manual_grid([0.0, 10.0], [0.0])
        render = render_sheet(two_suite_floor(), grid=grid, params=SheetParams(style="full"))
        root = ET.fromstring(render.svg)
        g = _group(root, "grid")
        bubbles = [e for e in g if e.get("class") == "grid-bubble"]
        assert len(bubbles) == 3


class TestStevensonLabels:
    def test_labels_large_gray_no_area(self):
        render = render_sheet(two_suite_floor(), params=STEVENSON)
        root = ET.fromstring(render.svg)
        labels_g = _group(root, "suite-labels")
        names = [
            t for t in labels_g.iter()
            if t.tag.endswith("text") and t.get("class") == "suite-label-name"
        ]
        assert len(names) == 2
        for t in names:
            assert float(t.get("font-size")) == pytest.approx(7.0)
            assert t.get("fill") == "#9a9a9a"
            assert t.get("font-weight") is None
        areas = [
            t for t in labels_g.iter()
            if t.tag.endswith("text") and t.get("class") == "suite-label-area"
        ]
        assert areas == []

    def test_full_style_labels_keep_area_line(self):
        from app.sheet.render import SheetParams
        render = render_sheet(two_suite_floor(), params=SheetParams(style="full"))
        root = ET.fromstring(render.svg)
        labels_g = _group(root, "suite-labels")
        areas = [
            t.text for t in labels_g.iter()
            if t.tag.endswith("text") and t.get("class") == "suite-label-area"
        ]
        assert "80.0 m²" in areas and "120.0 m²" in areas


# ── Style through the service/overrides layer ─────────────────────────────────

class TestStyleOverride:
    def test_default_style_is_stevenson_minimal(self, tmp_path):
        import json
        from app.sheet.service import render_job_sheet, resolve_params
        (tmp_path / "floor_geometry.json").write_text(
            json.dumps(two_suite_floor().to_json_dict())
        )
        assert resolve_params({}).style == "stevenson_minimal"
        render_job_sheet(tmp_path, write_pdf=False)
        svg = (tmp_path / "sheet.svg").read_text()
        assert 'id="title-block"' not in svg
        assert 'id="footer"' in svg

    def test_style_override_switches_sheet(self, tmp_path):
        import json
        from app.sheet.service import render_job_sheet, save_overrides
        (tmp_path / "floor_geometry.json").write_text(
            json.dumps(two_suite_floor().to_json_dict())
        )
        render_job_sheet(tmp_path, write_pdf=False)
        svg = (tmp_path / "sheet.svg").read_text()
        assert 'id="title-block"' not in svg

        save_overrides(tmp_path, {"style": "full"})
        render_job_sheet(tmp_path, write_pdf=False)
        svg = (tmp_path / "sheet.svg").read_text()
        assert 'id="title-block"' in svg

        # Clearing the style returns to the default stevenson_minimal sheet.
        save_overrides(tmp_path, {"style": None})
        render_job_sheet(tmp_path, write_pdf=False)
        svg = (tmp_path / "sheet.svg").read_text()
        assert 'id="title-block"' not in svg
        assert 'id="footer"' in svg

    def test_invalid_style_falls_back_to_full(self):
        from app.sheet.service import resolve_params
        assert resolve_params({"style": "bogus"}).style == "full"
        assert resolve_params({}).style == "stevenson_minimal"


class TestSuiteLabelConvention:
    def test_bare_digits_get_suite_prefix(self):
        from app.sheet.labels import format_suite_label
        assert format_suite_label("320") == "Suite 320"
        assert format_suite_label("Suite 320") == "Suite 320"
        assert format_suite_label("", suite_id="105") == "Suite 105"

    def test_override_bare_digits_on_sheet(self, tmp_path):
        import json
        from app.sheet.service import render_job_sheet, save_overrides
        floor = two_suite_floor()
        (tmp_path / "floor_geometry.json").write_text(
            json.dumps(floor.to_json_dict())
        )
        save_overrides(tmp_path, {"labels": {"suite_a": "301"}})
        render_job_sheet(tmp_path, write_pdf=False)
        assert "Suite 301" in (tmp_path / "sheet.svg").read_text()


class TestBboxGridFallback:
    def test_no_columns_still_gets_edge_ticks(self, tmp_path):
        import json
        from app.sheet.service import render_job_sheet
        (tmp_path / "floor_geometry.json").write_text(
            json.dumps(two_suite_floor().to_json_dict())
        )
        render_job_sheet(tmp_path, write_pdf=False)
        root = ET.fromstring((tmp_path / "sheet.svg").read_text())
        g = _group(root, "grid")
        ticks = [e for e in g if (e.get("class") or "").startswith("grid-tick")]
        assert len(ticks) >= 4
