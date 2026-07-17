"""Tests for feet/metres unit detection at ingest.

Guards against the P0 bug where every scan was assumed to be in metres: a
feet-based US scan read as metres inflates every area by 10.76x.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from app.pipeline.ingest import (
    UnitDetection,
    UnitDetectionError,
    detect_units,
    normalize_units_to_metres,
)

_FT_TO_M = 0.3048


def _room_cloud(
    width: float,
    depth: float,
    height: float,
    res: float,
    wall_thickness: float = 0.0,
) -> np.ndarray:
    """Synthetic interior scan: dense floor + ceiling slabs and four walls.

    All dimensions in the raw (unknown) unit; ``res`` is the sampling step
    in the same unit.
    """
    xs = np.arange(0, width + res, res)
    ys = np.arange(0, depth + res, res)
    zs = np.arange(0, height + res, res)

    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    floor = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    ceiling = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, height)], axis=1)

    wx, wz = np.meshgrid(xs, zs, indexing="ij")
    wall_y0 = np.stack([wx.ravel(), np.zeros(wx.size), wz.ravel()], axis=1)
    wall_y1 = np.stack([wx.ravel(), np.full(wx.size, depth), wz.ravel()], axis=1)
    wy, wz2 = np.meshgrid(ys, zs, indexing="ij")
    wall_x0 = np.stack([np.zeros(wy.size), wy.ravel(), wz2.ravel()], axis=1)
    wall_x1 = np.stack([np.full(wy.size, width), wy.ravel(), wz2.ravel()], axis=1)

    parts = [floor, ceiling, wall_y0, wall_y1, wall_x0, wall_x1]
    if wall_thickness > 0:
        # Second face of each wall to give it measurable thickness.
        parts += [
            wall_y0 + [0, wall_thickness, 0],
            wall_y1 - [0, wall_thickness, 0],
            wall_x0 + [wall_thickness, 0, 0],
            wall_x1 - [wall_thickness, 0, 0],
        ]
    return np.vstack(parts)


class TestDetectUnits:
    def test_metric_room_detected_as_metres(self):
        """A 6m x 4m room with a 2.7m ceiling is metres — no conversion."""
        pts = _room_cloud(width=6.0, depth=4.0, height=2.7, res=0.05)
        det = detect_units(pts)
        assert det.unit == "m"
        assert det.scale_to_m == 1.0
        assert det.method == "ceiling_gap"
        assert det.floor_to_ceiling_raw == pytest.approx(2.7, abs=0.15)

    def test_feet_room_detected_as_feet(self):
        """A 20ft x 13ft room with a 9ft ceiling reads as 9.0 raw units —
        implausible as metres, exactly right as feet."""
        pts = _room_cloud(width=20.0, depth=13.0, height=9.0, res=0.15)
        det = detect_units(pts)
        assert det.unit == "ft"
        assert det.scale_to_m == pytest.approx(_FT_TO_M)
        assert det.method == "ceiling_gap"
        assert det.floor_to_ceiling_raw == pytest.approx(9.0, abs=0.3)

    def test_tall_warehouse_metres(self):
        """5.2m ceiling: plausible metres, implausible feet (1.6m)."""
        pts = _room_cloud(width=30.0, depth=20.0, height=5.2, res=0.12)
        det = detect_units(pts)
        assert det.unit == "m"
        assert det.scale_to_m == 1.0

    def test_implausible_gap_fails_loudly(self):
        """Floor-to-ceiling of 6.2 raw units fits neither metres (max 5.5)
        nor feet (1.9m) — and with no measurable wall thickness the loader
        must refuse rather than silently guess."""
        # Slabs only, no walls -> no thickness tie-break available.
        res = 0.05
        xs = np.arange(0, 8 + res, res)
        ys = np.arange(0, 6 + res, res)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        floor = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        ceiling = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 6.2)], axis=1)
        pts = np.vstack([floor, ceiling])
        with pytest.raises(UnitDetectionError):
            detect_units(pts)

    def test_no_vertical_structure_assumes_metres(self):
        """A flat outdoor-style cloud has no ceiling: assume metres and say so."""
        rng = np.random.default_rng(0)
        pts = rng.uniform([0, 0, 0], [50, 50, 0.3], size=(5000, 3))
        det = detect_units(pts)
        assert det.unit == "m"
        assert det.scale_to_m == 1.0
        assert det.method == "assumed_metres"

    def test_huge_cloud_is_subsampled_not_scanned(self, monkeypatch):
        """140 M-point clouds must not run the full histogram — subsample first."""
        from app.pipeline import ingest as ingest_mod

        seen = {"n": None}

        def _spy(pts, axis_idx=2):
            seen["n"] = len(pts)
            return 2.7

        monkeypatch.setattr(ingest_mod, "_find_floor_ceiling_gap", _spy)
        monkeypatch.setattr(ingest_mod, "_UNIT_DETECT_MAX_POINTS", 1_000)
        pts = np.zeros((50_000, 3), dtype=np.float64)
        pts[:, 2] = np.linspace(0, 2.7, 50_000)
        det = detect_units(pts)
        assert seen["n"] == 1_000
        assert det.unit == "m"
        assert det.method == "ceiling_gap"


class TestNormalizeUnits:
    def test_feet_cloud_converted_to_metres(self):
        """After normalization a 9ft ceiling must be ~2.74 m, and a
        10ft x 10ft floor must cover ~9.29 m^2 -- not 100 'm^2'."""
        pts = _room_cloud(width=10.0, depth=10.0, height=9.0, res=0.1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        pcd, det = normalize_units_to_metres(pcd)
        assert det.unit == "ft"

        out = np.asarray(pcd.points)
        assert out[:, 2].max() == pytest.approx(9.0 * _FT_TO_M, abs=1e-6)
        w = out[:, 0].max() - out[:, 0].min()
        d = out[:, 1].max() - out[:, 1].min()
        assert w * d == pytest.approx((10 * _FT_TO_M) ** 2, rel=0.05)

    def test_metric_cloud_unchanged(self):
        pts = _room_cloud(width=6.0, depth=4.0, height=2.7, res=0.05)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.copy())

        pcd, det = normalize_units_to_metres(pcd)
        assert det.scale_to_m == 1.0
        np.testing.assert_allclose(np.asarray(pcd.points), pts)

    def test_empty_cloud_passthrough(self):
        pcd = o3d.geometry.PointCloud()
        pcd, det = normalize_units_to_metres(pcd)
        assert det.method == "assumed_metres"
        assert isinstance(det, UnitDetection)
