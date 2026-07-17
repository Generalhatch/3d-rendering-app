"""End-to-end Phase 4 wiring: run_vectorize on a synthetic tilted room.

One full pipeline run over a known-geometry fixture (10×10 m room with
0.1 m-thick walls, 2.5 m ceiling, tilted 2° about X) proving that:

- gravity re-leveling fires and is reported in metrics,
- the ceiling clearance is measured and recorded,
- ``result.json`` carries the structured ``warnings`` list and the
  ``room_confidence`` list, and both survive the on_complete payload.

This is the only test that runs the whole vectorize pipeline — keep the
fixture small (~250 k points at 4 cm spacing, 2 cm/px raster).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from app.models.vectorize_job import VectorizeParams
from app.vectorize.pipeline import run_vectorize


def _rot_x(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(r), -np.sin(r)],
        [0.0, np.sin(r), np.cos(r)],
    ])


def _grid(xs, ys, z) -> np.ndarray:
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def _wall(x0, y0, x1, y1, zs, step) -> np.ndarray:
    """Points on a vertical plane from (x0,y0) to (x1,y1).

    Sampled 4× denser along the wall length than vertically: the density
    slicer needs every 2 cm pixel column to see points in most of its
    10 cm vertical bins, which a uniform 4 cm grid cannot guarantee once
    scanner noise scatters points across pixel boundaries.
    """
    length = float(np.hypot(x1 - x0, y1 - y0))
    step = step / 4.0
    ts = np.arange(0.0, length + step / 2, step) / max(length, 1e-9)
    lx = x0 + ts * (x1 - x0)
    ly = y0 + ts * (y1 - y0)
    out = []
    for z in zs:
        out.append(np.column_stack([lx, ly, np.full(len(lx), z)]))
    return np.vstack(out)


@pytest.fixture(scope="module")
def tilted_room_scan(tmp_path_factory) -> Path:
    """PLY of a 10×10 m room, 0.1 m wall thickness, tilted 2° about X."""
    step = 0.04
    xs = np.arange(0.0, 10.0 + step / 2, step)
    zs = np.arange(0.0, 2.5 + step / 2, step)

    parts = [
        _grid(xs, xs, 0.0),      # floor
        _grid(xs, xs, 2.5),      # ceiling
    ]
    # Both faces of each perimeter wall (outer shell + 0.1 m inner face).
    for y_out, y_in in ((0.0, 0.1), (10.0, 9.9)):
        parts.append(_wall(0.0, y_out, 10.0, y_out, zs, step))
        parts.append(_wall(0.0, y_in, 10.0, y_in, zs, step))
    for x_out, x_in in ((0.0, 0.1), (10.0, 9.9)):
        parts.append(_wall(x_out, 0.0, x_out, 10.0, zs, step))
        parts.append(_wall(x_in, 0.0, x_in, 10.0, zs, step))

    pts = np.vstack(parts)
    # Real scanners spread returns ~1-2 cm around each surface; without this
    # the perfectly planar walls rasterize to 1-px lines that the speckle
    # filter (correctly) treats as noise.
    pts = pts + np.random.default_rng(11).normal(0.0, 0.012, pts.shape)
    pts = pts @ _rot_x(2.0).T   # the tilt under test

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    scan_path = tmp_path_factory.mktemp("scan") / "tilted_room.ply"
    o3d.io.write_point_cloud(str(scan_path), pcd)
    return scan_path


@pytest.fixture(scope="module")
def pipeline_run(tilted_room_scan, tmp_path_factory):
    """One shared full pipeline run; tests assert on different outputs."""
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    result_dir = tmp_path_factory.mktemp("results")

    completed: list[dict] = []
    errors: list[str] = []
    params = VectorizeParams(
        resolution_m_per_px=0.02,     # 2 cm/px keeps the run fast
        detect_openings=False,        # no doors in the fixture
        detect_columns=False,
    )
    run_vectorize(
        "phase4-e2e", tilted_room_scan, artifact_dir, result_dir, params,
        on_complete=lambda payload: completed.append(payload),
        on_error=lambda err: errors.append(err),
    )
    return completed, errors, result_dir


class TestPhase4EndToEnd:
    def test_run_completes(self, pipeline_run):
        completed, errors, _ = pipeline_run
        assert errors == [], f"pipeline failed: {errors}"
        assert len(completed) == 1

    def test_gravity_relevel_reported_in_metrics(self, pipeline_run):
        completed, _, _ = pipeline_run
        metrics = completed[0]["metrics"]
        assert metrics["gravity_relevel_applied"] is True
        assert metrics["gravity_tilt_deg"] == pytest.approx(2.0, abs=0.4)

    def test_relevel_transform_persisted(self, pipeline_run):
        completed, _, _ = pipeline_run
        T = np.array(completed[0]["relevel_transform"])
        assert T.shape == (4, 4)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_ceiling_clearance_measured(self, pipeline_run):
        completed, _, _ = pipeline_run
        metrics = completed[0]["metrics"]
        assert metrics["ceiling_clearance_m"] == pytest.approx(2.5, abs=0.1)
        # 2.5 m clearance is tall enough for the default bands.
        assert metrics["slice_band_adjusted"] is False

    def test_result_json_carries_warnings_list(self, pipeline_run):
        completed, _, result_dir = pipeline_run
        payload = json.loads((result_dir / "result.json").read_text())
        assert isinstance(payload["warnings"], list)
        for w in payload["warnings"]:
            assert set(w) == {"code", "stage", "message"}
        # in-memory payload and persisted file agree
        assert payload["warnings"] == completed[0]["warnings"]

    def test_room_confidence_persisted(self, pipeline_run):
        completed, _, result_dir = pipeline_run
        payload = json.loads((result_dir / "result.json").read_text())
        rooms = payload["room_confidence"]
        assert len(rooms) >= 1
        for rc in rooms:
            assert 0.0 <= rc["confidence"] <= 1.0
            assert 0.0 <= rc["boundary_coverage"] <= 1.0
            assert isinstance(rc["flagged"], bool)
            assert len(rc["polygon"]) >= 3
        # A fully scanned single room must NOT be flagged for review.
        biggest = max(rooms, key=lambda r: r["area_m2"])
        assert biggest["boundary_coverage"] > 0.9
        assert biggest["flagged"] is False

    def test_measured_room_area_unaffected(self, pipeline_run):
        """The 9.8×9.8 m interior must come out at ~96 m² despite tilt +
        relevel — Phase 4 must not move measurement numbers."""
        completed, _, _ = pipeline_run
        payload = completed[0]
        rooms = payload["room_confidence"]
        biggest = max(rooms, key=lambda r: r["area_m2"])
        assert biggest["area_m2"] == pytest.approx(96.04, rel=0.05)
