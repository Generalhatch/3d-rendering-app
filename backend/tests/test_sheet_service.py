"""Phase 3: job-level sheet orchestration (app.sheet.service).

Covers the floor_geometry.json → sheet.svg/sheet.pdf path both pipelines
and the HTTP routes share, plus operator overrides (suite labels, title
block metadata, manual grid entry) and the manual-grid-beats-fitted rule.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from app.geometry.model import ColumnFeature
from app.sheet.service import (
    load_overrides,
    render_job_sheet,
    resolve_grid,
    resolve_metadata,
    resolve_params,
    save_overrides,
)
from tests.reference_floors import two_suite_floor


def _group(root: ET.Element, gid: str) -> ET.Element:
    for el in root.iter():
        if el.get("id") == gid:
            return el
    raise AssertionError(f"element #{gid} not found in SVG")


def _write_geometry(tmp_path, floor) -> None:
    (tmp_path / "floor_geometry.json").write_text(
        json.dumps(floor.to_json_dict())
    )


class TestRenderJobSheet:
    def test_renders_svg_and_pdf_from_persisted_geometry(self, tmp_path):
        _write_geometry(tmp_path, two_suite_floor())
        render = render_job_sheet(tmp_path)
        svg_path = tmp_path / "sheet.svg"
        pdf_path = tmp_path / "sheet.pdf"
        assert svg_path.exists()
        assert pdf_path.read_bytes()[:5] == b"%PDF-"
        assert render.scale_denominator == 50
        root = ET.fromstring(svg_path.read_text())
        assert _group(root, "walls") is not None

    def test_missing_geometry_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            render_job_sheet(tmp_path)

    def test_label_override_changes_rendered_sheet(self, tmp_path):
        _write_geometry(tmp_path, two_suite_floor())
        render_job_sheet(tmp_path, write_pdf=False)
        assert "Suite 301" not in (tmp_path / "sheet.svg").read_text()

        save_overrides(tmp_path, {"labels": {"suite_a": "Suite 301"}})
        render_job_sheet(tmp_path, write_pdf=False)
        svg = (tmp_path / "sheet.svg").read_text()
        assert "Suite 301" in svg
        assert "Suite A" not in svg

    def test_meta_override_fills_title_block(self, tmp_path):
        _write_geometry(tmp_path, two_suite_floor())
        save_overrides(tmp_path, {"style": "full", "meta": {
            "building_name": "One Rock Plaza",
            "address": "1 Rock Ave",
            "floor_name": "Floor 3",
        }})
        render_job_sheet(tmp_path, write_pdf=False)
        root = ET.fromstring((tmp_path / "sheet.svg").read_text())
        assert _group(root, "tb-building").text == "One Rock Plaza"
        assert _group(root, "tb-address").text == "1 Rock Ave"
        assert _group(root, "tb-floor").text == "Floor 3"

    def test_manual_grid_override_renders_bubbles(self, tmp_path):
        _write_geometry(tmp_path, two_suite_floor())   # no columns → bbox fallback
        render_job_sheet(tmp_path, write_pdf=False)
        root = ET.fromstring((tmp_path / "sheet.svg").read_text())
        g = _group(root, "grid")
        ticks = [e for e in g if (e.get("class") or "").startswith("grid-tick")]
        assert len(ticks) >= 4

        save_overrides(tmp_path, {"style": "full", "manual_grid": {
            "rotation_deg": 0.0,
            "u_offsets_m": [0.0, 10.0, 20.0],
            "v_offsets_m": [0.0, 10.0],
        }})
        render_job_sheet(tmp_path, write_pdf=False)
        root = ET.fromstring((tmp_path / "sheet.svg").read_text())
        lines = [e for e in _group(root, "grid") if e.tag.endswith("line")]
        assert len(lines) == 5


class TestOverridesPersistence:
    def test_labels_merge_and_clear(self, tmp_path):
        save_overrides(tmp_path, {"labels": {"a": "Suite 1", "b": "Suite 2"}})
        save_overrides(tmp_path, {"labels": {"b": "Suite 2B", "c": "Suite 3"}})
        assert load_overrides(tmp_path)["labels"] == {
            "a": "Suite 1", "b": "Suite 2B", "c": "Suite 3",
        }
        # Empty string clears an override.
        save_overrides(tmp_path, {"labels": {"a": ""}})
        assert load_overrides(tmp_path)["labels"] == {
            "b": "Suite 2B", "c": "Suite 3",
        }

    def test_manual_grid_replace_and_remove(self, tmp_path):
        save_overrides(tmp_path, {"manual_grid": {"u_offsets_m": [0.0]}})
        assert load_overrides(tmp_path)["manual_grid"] == {"u_offsets_m": [0.0]}
        save_overrides(tmp_path, {"manual_grid": None})
        assert "manual_grid" not in load_overrides(tmp_path)


class TestResolveGridAndMeta:
    def test_manual_grid_beats_fitted(self):
        floor = two_suite_floor()
        for i, (x, y) in enumerate(
            [(0, 0), (6, 0), (12, 0), (0, 5), (6, 5), (12, 5)]
        ):
            floor.columns.append(
                ColumnFeature(id=f"c{i}", centre=(float(x), float(y)))
            )
        fitted = resolve_grid(floor, {})
        assert fitted is not None and fitted.source == "fitted"

        manual = resolve_grid(floor, {"manual_grid": {
            "rotation_deg": 0.0, "u_offsets_m": [0.0, 5.0], "v_offsets_m": [],
        }})
        assert manual.source == "manual"
        assert [g.offset_m for g in manual.u_lines] == [0.0, 5.0]

    def test_no_columns_uses_bbox_fallback(self):
        grid = resolve_grid(two_suite_floor(), {})
        assert grid is not None
        assert grid.source == "manual"
        assert len(grid.u_lines) >= 2
        assert len(grid.v_lines) >= 2

    def test_initial_overrides_seeded_on_first_render(self, tmp_path):
        floor = two_suite_floor()
        floor.building_name = "Test Tower"
        _write_geometry(tmp_path, floor)
        render_job_sheet(tmp_path, write_pdf=False)
        overrides = load_overrides(tmp_path)
        assert overrides["style"] == "stevenson_minimal"
        assert overrides["meta"]["building_name"] == "Test Tower"

    def test_resolve_params_default(self):
        assert resolve_params({}).style == "stevenson_minimal"

    def test_meta_falls_back_to_floor_fields(self):
        floor = two_suite_floor()
        floor.building_name = "Geometry Name"
        floor.floor_name = "L7"
        meta = resolve_metadata(floor, {})
        assert meta.building_name == "Geometry Name"
        assert meta.floor_name == "L7"
        meta2 = resolve_metadata(floor, {"meta": {"building_name": "Override"}})
        assert meta2.building_name == "Override"
        assert meta2.floor_name == "L7"
