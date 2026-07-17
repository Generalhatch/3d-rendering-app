"""Tests for Generate BOMA from editor wall segments.

Covers the operator workflow: when auto topology finds 0 rooms, closing
walls and calling ``generate_measurement_from_segments`` must write
floor_geometry.json + measurement_report.json.  An open U-shape must raise
RoomsNotClosedError instead of inventing Gross area.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.models.vectorize_job import EditableSegment
from app.vectorize.measure_from_segments import (
    RoomsNotClosedError,
    generate_measurement_from_segments,
)


def _rect_segments(w: float = 6.0, h: float = 4.0) -> list[EditableSegment]:
    """Closed rectangle — one room."""
    corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    segs: list[EditableSegment] = []
    for i, (x1, y1) in enumerate(corners):
        x2, y2 = corners[(i + 1) % 4]
        segs.append(EditableSegment(
            id=f"w{i}", layer="walls", x1=x1, y1=y1, x2=x2, y2=y2,
        ))
    return segs


def _u_shape_segments() -> list[EditableSegment]:
    """Open U — three walls, no enclosed room."""
    return [
        EditableSegment(id="a", layer="walls", x1=0, y1=0, x2=0, y2=4),
        EditableSegment(id="b", layer="walls", x1=0, y1=0, x2=6, y2=0),
        EditableSegment(id="c", layer="walls", x1=6, y1=0, x2=6, y2=4),
    ]


def _write_segments(result_dir: Path, segs: list[EditableSegment]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "units": "metres",
        "segments": [s.model_dump() for s in segs],
    }
    (result_dir / "segments.json").write_text(json.dumps(payload))


class TestGenerateMeasurementFromSegments:
    def test_closed_rectangle_writes_boma(self, tmp_path: Path):
        segs = _rect_segments()
        _write_segments(tmp_path, segs)
        # Minimal DXF so sheet dimension step is optional.
        (tmp_path / "vectorized.dxf").write_text("0\nSECTION\n0\nENDSEC\n0\nEOF\n")

        result = generate_measurement_from_segments(
            tmp_path, "job-rect", segments=segs, render_sheet=False,
        )
        assert result.rooms_detected >= 1
        assert result.has_floor_geometry
        assert result.has_measurement_report
        assert (tmp_path / "floor_geometry.json").exists()
        assert (tmp_path / "measurement_report.json").exists()
        assert (tmp_path / "measurement_report.pdf").exists()

        report = json.loads((tmp_path / "measurement_report.json").read_text())
        standards = {m["standard"] for m in report["measurements"]}
        assert "BOMA Office 2024" in standards or any(
            "boma" in m["standard"].lower() for m in report["measurements"]
        )
        geo = json.loads((tmp_path / "floor_geometry.json").read_text())
        assert len(geo["rooms"]) >= 1
        assert geo.get("source") == "vectorize.editor"

    def test_open_u_raises_rooms_not_closed(self, tmp_path: Path):
        segs = _u_shape_segments()
        _write_segments(tmp_path, segs)
        with pytest.raises(RoomsNotClosedError) as ei:
            generate_measurement_from_segments(
                tmp_path, "job-u", segments=segs, render_sheet=False,
            )
        assert ei.value.n_walls == 3
        assert len(ei.value.dangling_endpoints) >= 2
        assert (tmp_path / "dangling_endpoints.json").exists()
        dangling_payload = json.loads(
            (tmp_path / "dangling_endpoints.json").read_text()
        )
        assert len(dangling_payload["endpoints"]) >= 2
        assert not (tmp_path / "floor_geometry.json").exists()
        assert not (tmp_path / "measurement_report.json").exists()

    def test_reads_segments_json_when_not_passed(self, tmp_path: Path):
        _write_segments(tmp_path, _rect_segments())
        result = generate_measurement_from_segments(
            tmp_path, "job-disk", render_sheet=False,
        )
        assert result.rooms_detected >= 1

    def test_too_few_walls_raises(self, tmp_path: Path):
        segs = [
            EditableSegment(id="a", layer="walls", x1=0, y1=0, x2=1, y2=0),
            EditableSegment(id="b", layer="walls", x1=1, y1=0, x2=1, y2=1),
        ]
        with pytest.raises(RoomsNotClosedError):
            generate_measurement_from_segments(
                tmp_path, "job-few", segments=segs, render_sheet=False,
            )

    def test_closing_gap_then_succeeds(self, tmp_path: Path):
        """U-shape fails; adding the fourth wall succeeds — the editor workflow."""
        open_segs = _u_shape_segments()
        _write_segments(tmp_path, open_segs)
        with pytest.raises(RoomsNotClosedError):
            generate_measurement_from_segments(
                tmp_path, "job-gap", segments=open_segs, render_sheet=False,
            )

        closed = open_segs + [
            EditableSegment(id="d", layer="walls", x1=0, y1=4, x2=6, y2=4),
        ]
        result = generate_measurement_from_segments(
            tmp_path, "job-gap", segments=closed, render_sheet=False,
        )
        assert result.rooms_detected >= 1
        assert (tmp_path / "measurement_report.json").exists()

    def test_near_closed_gap_auto_bridges(self, tmp_path: Path):
        """15 cm break in a rectangle — Phase 2 continuity closes it for BOMA."""
        segs = [
            EditableSegment(id="a", layer="walls", x1=0, y1=0, x2=4.925, y2=0),
            EditableSegment(id="b", layer="walls", x1=5.075, y1=0, x2=10, y2=0),
            EditableSegment(id="c", layer="walls", x1=10, y1=0, x2=10, y2=8),
            EditableSegment(id="d", layer="walls", x1=10, y1=8, x2=0, y2=8),
            EditableSegment(id="e", layer="walls", x1=0, y1=8, x2=0, y2=0),
        ]
        _write_segments(tmp_path, segs)
        result = generate_measurement_from_segments(
            tmp_path, "job-bridge", segments=segs, render_sheet=False,
            snapping_distance_m=0.30,
        )
        assert result.rooms_detected >= 1
        assert (tmp_path / "floor_geometry.json").exists()
        # Dangling cleared when rooms close.
        dangling = json.loads(
            (tmp_path / "dangling_endpoints.json").read_text()
        )
        assert dangling["endpoints"] == []

    def test_wide_open_still_422(self, tmp_path: Path):
        """Wide U still fails — continuity must not invent a closing wall."""
        segs = _u_shape_segments()
        _write_segments(tmp_path, segs)
        with pytest.raises(RoomsNotClosedError) as ei:
            generate_measurement_from_segments(
                tmp_path, "job-wide", segments=segs, render_sheet=False,
                snapping_distance_m=0.30,
            )
        assert len(ei.value.dangling_endpoints) >= 2
        assert not (tmp_path / "measurement_report.json").exists()
