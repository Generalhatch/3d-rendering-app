"""Measurement report generator tests — JSON structure + rendered PDF."""
from __future__ import annotations

import json

import pytest

from app.measurement import get_ruleset
from app.measurement.report import build_report, write_json_report, write_pdf_report
from tests.reference_floors import (
    CORRIDOR_SUITE_RENTABLE,
    CORRIDOR_SUITE_USABLE,
    corridor_floor,
)

RULESET_NAMES = ["boma_2024_a", "rebny", "gross"]


@pytest.fixture()
def report():
    floor = corridor_floor()
    measurements = [get_ruleset(n).measure(floor) for n in RULESET_NAMES]
    return build_report(floor, measurements)


class TestReportJson:
    def test_report_shape(self, report):
        assert report["version"] == 1
        assert report["floor_id"] == "corridor-reference"
        assert len(report["measurements"]) == 3
        standards = {m["standard"] for m in report["measurements"]}
        assert standards == {"BOMA Office 2024", "REBNY", "Gross Area"}

    def test_every_number_stamped(self, report):
        for m in report["measurements"]:
            for key in ("floor_usable", "floor_rentable", "common_area"):
                assert m[key]["standard"] == m["standard"]
                assert m[key]["method"] == m["method"]
            for suite in m["suites"]:
                for comp in ("usable", "rentable"):
                    assert suite[comp]["standard"] == m["standard"]
                    assert suite[comp]["method"] == m["method"]

    def test_hand_computed_values_survive_serialization(self, report):
        boma = next(
            m for m in report["measurements"]
            if m["standard"] == "BOMA Office 2024"
        )
        suite_a = next(s for s in boma["suites"] if s["suite_id"] == "suite_a")
        assert suite_a["usable"]["value_m2"] == pytest.approx(
            CORRIDOR_SUITE_USABLE, abs=1e-3)
        assert suite_a["rentable"]["value_m2"] == pytest.approx(
            CORRIDOR_SUITE_RENTABLE, abs=1e-3)

    def test_sqft_conversion(self, report):
        boma = next(
            m for m in report["measurements"]
            if m["standard"] == "BOMA Office 2024"
        )
        fu = boma["floor_usable"]
        assert fu["value_sqft"] == pytest.approx(fu["value_m2"] * 10.7639, rel=1e-4)

    def test_write_json_roundtrip(self, report, tmp_path):
        path = tmp_path / "measurement_report.json"
        write_json_report(report, path)
        loaded = json.loads(path.read_text())
        assert loaded == report


class TestReportPdf:
    def test_pdf_written(self, report, tmp_path):
        path = tmp_path / "measurement_report.pdf"
        write_pdf_report(report, path)
        data = path.read_bytes()
        assert data[:5] == b"%PDF-"
        assert len(data) > 1024

    def test_pdf_handles_no_common_area(self, tmp_path):
        from tests.reference_floors import two_suite_floor
        floor = two_suite_floor()
        measurements = [get_ruleset(n).measure(floor) for n in RULESET_NAMES]
        report = build_report(floor, measurements)
        path = tmp_path / "r.pdf"
        write_pdf_report(report, path)
        assert path.read_bytes()[:5] == b"%PDF-"
