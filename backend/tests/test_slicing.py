"""Unit tests for the floor detection and wall band slicing pipeline."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import open3d as o3d
    from app.pipeline.slicing import detect_floor, extract_wall_band
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


@pytest.mark.skipif(not HAS_OPEN3D, reason="open3d not installed")
class TestFloorDetection:
    def _make_scene(self, floor_z: float = 0.0, axis: int = 2) -> "o3d.geometry.PointCloud":
        """Create a synthetic scene: flat floor + some wall points."""
        rng = np.random.default_rng(42)
        # Floor
        floor_x = rng.uniform(-10, 10, 3000)
        floor_y = rng.uniform(-10, 10, 3000)
        floor_zs = np.full(3000, floor_z) + rng.normal(0, 0.005, 3000)
        floor_pts = np.column_stack([floor_x, floor_y, floor_zs])
        if axis == 1:  # Y-up: swap z and y
            floor_pts = floor_pts[:, [0, 2, 1]]

        # Walls
        wall_pts = rng.uniform(-10, 10, (2000, 3))
        wall_pts[:, axis] = rng.uniform(floor_z + 0.5, floor_z + 3.0, 2000)

        pts = np.vstack([floor_pts, wall_pts])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def test_detects_floor_zup(self):
        pcd = self._make_scene(floor_z=0.0, axis=2)
        result = detect_floor(pcd)
        assert abs(result.floor_z_estimate) < 0.15, f"Expected floor at ~0, got {result.floor_z_estimate}"
        assert result.inlier_count > 500

    def test_detects_floor_at_nonzero_height(self):
        pcd = self._make_scene(floor_z=1.5, axis=2)
        result = detect_floor(pcd)
        assert abs(result.floor_z_estimate - 1.5) < 0.3

    def test_wall_band_extracts_correct_range(self):
        pcd = self._make_scene(floor_z=0.0, axis=2)
        floor = detect_floor(pcd)
        band = extract_wall_band(pcd, floor, band_low_m=0.75, band_high_m=1.80)
        pts = np.asarray(band.points)
        heights = pts[:, floor.axis_idx]
        # All band points should be in the expected height range
        assert heights.min() >= floor.floor_z_estimate + 0.70  # small tolerance
        assert heights.max() <= floor.floor_z_estimate + 1.85

    def test_wall_band_not_empty(self):
        pcd = self._make_scene(floor_z=0.0, axis=2)
        floor = detect_floor(pcd)
        band = extract_wall_band(pcd, floor)
        assert len(band.points) > 100
