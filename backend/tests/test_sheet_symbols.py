"""Phase 3.5 (drafting-symbols): door leaf + quarter-arc swings, stair and
elevator symbols from Penetration kinds, dashed linetypes.

Door geometry hand-computed on the two-suite reference floor transform
(s = 20 mm/m, X(x) = 223.5 + 20·(x + 0.15), Y(y) = 180 + 20·(10.15 − y)):

Single 1.0 m door on the bottom wall segment (4,0)→(5,0), hinge=start,
side=left (+90° CCW of +x is +y, into suite_a):
  hinge (4,0) → paper (306.5, 383);  latch (5,0) → (326.5, 383)
  open tip (4,1) → (306.5, 363);  radius 1 m → 20 mm.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from app.geometry.model import FloorGeometry, Opening, Penetration
from app.sheet.render import LINETYPE_DASH_MM, SheetParams, render_sheet
from app.sheet.symbols import (
    DOUBLE_DOOR_MIN_WIDTH_M,
    STAIR_TREAD_SPACING_M,
    default_swing,
    door_leaves,
    elevator_diagonals,
    stair_treads,
)
from tests.reference_floors import corridor_floor, two_suite_floor


def _render_floor(floor, **kwargs):
    """Coordinate-exact symbol tests use the full title-block layout."""
    kwargs.setdefault("params", SheetParams(style="full"))
    return render_sheet(floor, **kwargs)


def _group(root: ET.Element, gid: str) -> ET.Element:
    for el in root.iter():
        if el.get("id") == gid:
            return el
    raise AssertionError(f"element #{gid} not found in SVG")


def _door(width: float = 1.0, x0: float = 4.0) -> Opening:
    return Opening(
        id="d1",
        segment=np.array([[x0, 0.0], [x0 + width, 0.0]]),
        width_m=width, kind="door", wall_id="w_bottom",
    )


# ── Door leaf geometry (world space) ──────────────────────────────────────────

class TestDoorLeaves:
    def test_single_leaf_exact_geometry(self):
        floor = two_suite_floor()
        leaves = door_leaves(_door(), floor, override={"hinge": "start", "side": "left"})
        assert len(leaves) == 1
        leaf = leaves[0]
        assert leaf.hinge == pytest.approx((4.0, 0.0), abs=1e-12)
        assert leaf.latch == pytest.approx((5.0, 0.0), abs=1e-12)
        assert leaf.tip_open == pytest.approx((4.0, 1.0), abs=1e-12)
        assert leaf.radius_m == pytest.approx(1.0, abs=1e-12)

    def test_hinge_end_flips_pivot(self):
        floor = two_suite_floor()
        leaf = door_leaves(_door(), floor, override={"hinge": "end", "side": "left"})[0]
        assert leaf.hinge == pytest.approx((5.0, 0.0), abs=1e-12)
        assert leaf.latch == pytest.approx((4.0, 0.0), abs=1e-12)
        assert leaf.tip_open == pytest.approx((5.0, 1.0), abs=1e-12)

    def test_side_right_swings_other_way(self):
        floor = two_suite_floor()
        leaf = door_leaves(_door(), floor, override={"hinge": "start", "side": "right"})[0]
        assert leaf.tip_open == pytest.approx((4.0, -1.0), abs=1e-12)

    def test_double_door_above_width_threshold(self):
        floor = two_suite_floor()
        assert DOUBLE_DOOR_MIN_WIDTH_M == pytest.approx(1.4)
        leaves = door_leaves(
            _door(width=1.6), floor, override={"side": "left"},
        )
        assert len(leaves) == 2
        left, right = leaves
        assert left.hinge == pytest.approx((4.0, 0.0), abs=1e-12)
        assert left.latch == pytest.approx((4.8, 0.0), abs=1e-12)
        assert left.tip_open == pytest.approx((4.0, 0.8), abs=1e-12)
        assert right.hinge == pytest.approx((5.6, 0.0), abs=1e-12)
        assert right.latch == pytest.approx((4.8, 0.0), abs=1e-12)
        assert right.tip_open == pytest.approx((5.6, 0.8), abs=1e-12)

    def test_default_swing_prefers_room_side(self):
        # Door on the bottom exterior wall: suite_a is above (left of the
        # +x segment direction), outside is below → heuristic picks left.
        floor = two_suite_floor()
        assert default_swing(_door(), floor) == {"hinge": "start", "side": "left"}

    def test_default_swing_prefers_smaller_room(self):
        # Door on the demising wall segment (8,4)→(8,5): suite_a (80 m²)
        # is on the RIGHT of the +y direction... left normal of (0,1) is
        # (−1,0) → suite_a; right is suite_b (120 m²).  Smaller wins → left.
        floor = two_suite_floor()
        door = Opening(
            id="d2", segment=np.array([[8.0, 4.0], [8.0, 5.0]]),
            width_m=1.0, kind="door", wall_id="w_demising",
        )
        assert default_swing(door, floor)["side"] == "left"


# ── Door swings in the SVG ────────────────────────────────────────────────────

class TestDoorSwingSvg:
    def _render(self, door_swings=None):
        floor = two_suite_floor()
        floor.openings.append(_door())
        render = _render_floor(floor, door_swings=door_swings)
        return render, ET.fromstring(render.svg)

    def test_leaf_line_and_arc_exact_paper_coords(self):
        _, root = self._render(
            door_swings={"d1": {"hinge": "start", "side": "left"}},
        )
        g = _group(root, "symbols")
        swings = [e for e in g if e.get("class") == "door-swing"]
        assert len(swings) == 1
        assert swings[0].get("data-opening-id") == "d1"
        leaf = next(e for e in swings[0] if e.get("class") == "door-leaf")
        assert float(leaf.get("x1")) == pytest.approx(306.5, abs=1e-6)
        assert float(leaf.get("y1")) == pytest.approx(383.0, abs=1e-6)
        assert float(leaf.get("x2")) == pytest.approx(306.5, abs=1e-6)
        assert float(leaf.get("y2")) == pytest.approx(363.0, abs=1e-6)
        arc = next(e for e in swings[0] if e.get("class") == "door-arc")
        d = arc.get("d")
        # M open-tip, A r r 0 0 sweep latch — radius 1 m = 20 mm.
        assert d.startswith("M 306.5 363 A 20 20 0 0 ")
        assert d.endswith(" 326.5 383")

    def test_operator_can_flip_swing(self):
        _, root = self._render(
            door_swings={"d1": {"hinge": "start", "side": "right"}},
        )
        g = _group(root, "symbols")
        leaf = next(
            e for sw in g if sw.get("class") == "door-swing"
            for e in sw if e.get("class") == "door-leaf"
        )
        # tip (4,−1) → Y = 180 + 20·(10.15 + 1) = 403.
        assert float(leaf.get("y2")) == pytest.approx(403.0, abs=1e-6)

    def test_double_door_renders_two_leaves_two_arcs(self):
        floor = two_suite_floor()
        floor.openings.append(_door(width=1.6))
        render = _render_floor(floor)
        root = ET.fromstring(render.svg)
        swing = next(
            e for e in _group(root, "symbols") if e.get("class") == "door-swing"
        )
        leaves = [e for e in swing if e.get("class") == "door-leaf"]
        arcs = [e for e in swing if e.get("class") == "door-arc"]
        assert len(leaves) == 2 and len(arcs) == 2

    def test_windows_get_no_swing(self):
        floor = two_suite_floor()
        floor.openings.append(Opening(
            id="win", segment=np.array([[4.0, 0.0], [6.0, 0.0]]),
            width_m=2.0, kind="window", wall_id="w_bottom",
        ))
        render = _render_floor(floor)
        root = ET.fromstring(render.svg)
        assert [e for e in _group(root, "symbols")
                if e.get("class") == "door-swing"] == []


# ── Stairs + elevators ────────────────────────────────────────────────────────

def _stair_pen(kind: str = "stair") -> Penetration:
    # 3.0 × 1.2 m run, axis along +x.
    return Penetration(
        id="p-stair",
        polygon=np.array([[2.0, 2.0], [5.0, 2.0], [5.0, 3.2], [2.0, 3.2]]),
        kind=kind,
    )


class TestStairElevatorSymbols:
    def test_stair_tread_count_and_exact_positions(self):
        treads = stair_treads(_stair_pen())
        # 3.0 m long axis / 0.28 m spacing → treads at x = 2.28, 2.56, …
        # strictly inside → 10 treads (k = 1..10; k=10 → 4.8 < 5.0).
        assert STAIR_TREAD_SPACING_M == pytest.approx(0.28)
        assert len(treads) == 10
        first = treads[0]
        xs = sorted((first[0, 0], first[1, 0]))
        ys = sorted((first[0, 1], first[1, 1]))
        assert xs == pytest.approx([2.28, 2.28], abs=1e-9)
        assert ys == pytest.approx([2.0, 3.2], abs=1e-9)

    def test_elevator_diagonals_exact(self):
        pen = Penetration(
            id="p-elev",
            polygon=np.array([[9.0, 4.0], [11.0, 4.0], [11.0, 6.0], [9.0, 6.0]]),
            kind="elevator_shaft",
        )
        diags = elevator_diagonals(pen)
        assert len(diags) == 2
        sets = {tuple(sorted(map(tuple, d))) for d in diags}
        assert ((9.0, 4.0), (11.0, 6.0)) in sets
        assert ((9.0, 6.0), (11.0, 4.0)) in sets

    def test_stair_symbol_in_svg(self):
        floor = two_suite_floor()
        floor.penetrations.append(_stair_pen())
        render = _render_floor(floor)
        root = ET.fromstring(render.svg)
        stair = next(
            e for e in _group(root, "symbols")
            if e.get("class") == "stair-symbol"
        )
        treads = [e for e in stair if e.get("class") == "stair-tread"]
        arrows = [e for e in stair if e.get("class") == "stair-arrow"]
        assert len(treads) == 10
        assert len(arrows) == 1
        assert arrows[0].get("marker-end") == "url(#arrowhead)"

    def test_elevator_symbol_in_svg(self):
        render = _render_floor(corridor_floor(with_shaft=True))
        root = ET.fromstring(render.svg)
        elev = next(
            e for e in _group(root, "symbols")
            if e.get("class") == "elevator-symbol"
        )
        diags = [e for e in elev if e.get("class") == "elevator-diagonal"]
        assert len(diags) == 2

    def test_plain_shaft_gets_no_symbol(self):
        floor = two_suite_floor()
        floor.penetrations.append(Penetration(
            id="p-shaft",
            polygon=np.array([[9.0, 4.0], [11.0, 4.0], [11.0, 6.0], [9.0, 6.0]]),
            kind="shaft",
        ))
        render = _render_floor(floor)
        root = ET.fromstring(render.svg)
        assert list(_group(root, "symbols")) == []


# ── Dashed linetypes ──────────────────────────────────────────────────────────

class TestDashedLinetypes:
    def test_overhang_rendered_dashed_unfilled(self):
        floor = two_suite_floor()
        floor.penetrations.append(Penetration(
            id="p-oh",
            polygon=np.array([[0.0, 10.5], [20.0, 10.5], [20.0, 12.0], [0.0, 12.0]]),
            kind="overhang",
        ))
        render = _render_floor(floor)
        root = ET.fromstring(render.svg)
        pen_g = _group(root, "penetrations")
        oh = next(
            e for e in pen_g if "penetration-overhang" in (e.get("class") or "")
        )
        assert oh.get("stroke-dasharray") == LINETYPE_DASH_MM["overhang"]
        assert oh.get("fill") == "none"

    def test_shaft_stays_solid_filled(self):
        render = _render_floor(corridor_floor(with_shaft=True))
        root = ET.fromstring(render.svg)
        pen_g = _group(root, "penetrations")
        shaft = next(
            e for e in pen_g
            if "penetration-elevator_shaft" in (e.get("class") or "")
        )
        assert shaft.get("stroke-dasharray") is None
        assert shaft.get("fill") == "#e8e8e8"

    def test_linetype_table_covers_reference_classes(self):
        for kind in ("overhang", "canopy"):
            assert kind in LINETYPE_DASH_MM


# ── Door-swing overrides through the service layer ────────────────────────────

class TestDoorSwingOverridePersistence:
    def test_merge_and_clear(self, tmp_path):
        from app.sheet.service import load_overrides, save_overrides
        save_overrides(tmp_path, {"door_swings": {
            "d1": {"hinge": "start", "side": "left"},
        }})
        save_overrides(tmp_path, {"door_swings": {
            "d2": {"hinge": "end", "side": "right"},
        }})
        assert load_overrides(tmp_path)["door_swings"] == {
            "d1": {"hinge": "start", "side": "left"},
            "d2": {"hinge": "end", "side": "right"},
        }
        save_overrides(tmp_path, {"door_swings": {"d1": {}}})
        assert load_overrides(tmp_path)["door_swings"] == {
            "d2": {"hinge": "end", "side": "right"},
        }

    def test_override_flips_rendered_sheet(self, tmp_path):
        import json
        from app.sheet.service import render_job_sheet, save_overrides
        floor = two_suite_floor()
        floor.openings.append(_door())
        (tmp_path / "floor_geometry.json").write_text(
            json.dumps(floor.to_json_dict())
        )
        render_job_sheet(tmp_path, write_pdf=False)
        svg1 = (tmp_path / "sheet.svg").read_text()

        save_overrides(tmp_path, {"door_swings": {
            "d1": {"hinge": "end", "side": "right"},
        }})
        render_job_sheet(tmp_path, write_pdf=False)
        svg2 = (tmp_path / "sheet.svg").read_text()
        assert svg1 != svg2
        # Flipped leaf pivots at the far jamb: hinge (5,0) → x = 320.5
        # (stevenson_minimal default layout).
        root = ET.fromstring(svg2)
        leaf = next(
            e for sw in _group(root, "symbols")
            if sw.get("class") == "door-swing"
            for e in sw if e.get("class") == "door-leaf"
        )
        assert float(leaf.get("x1")) == pytest.approx(320.5, abs=1e-6)
