"""Unit tests for RANSAC wall plane segmentation."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("open3d") is None,
    reason="open3d not installed",
)
class TestWallSegmentation:
    def _make_axis_aligned_box(self, n_per_wall: int = 500) -> "open3d.geometry.PointCloud":
        import open3d as o3d
        rng = np.random.default_rng(7)
        walls = []
        # 4 walls of a 10x8m box, sampled at z=1m (wall band height)
        for z in np.linspace(0.75, 1.80, 15):
            # North wall y=8
            x = rng.uniform(0, 10, n_per_wall)
            walls.append(np.column_stack([x, np.full(n_per_wall, 8.0) + rng.normal(0, 0.01, n_per_wall), np.full(n_per_wall, z)]))
            # South wall y=0
            x = rng.uniform(0, 10, n_per_wall)
            walls.append(np.column_stack([x, np.full(n_per_wall, 0.0) + rng.normal(0, 0.01, n_per_wall), np.full(n_per_wall, z)]))
            # East wall x=10
            y = rng.uniform(0, 8, n_per_wall)
            walls.append(np.column_stack([np.full(n_per_wall, 10.0) + rng.normal(0, 0.01, n_per_wall), y, np.full(n_per_wall, z)]))
            # West wall x=0
            y = rng.uniform(0, 8, n_per_wall)
            walls.append(np.column_stack([np.full(n_per_wall, 0.0) + rng.normal(0, 0.01, n_per_wall), y, np.full(n_per_wall, z)]))
        pts = np.vstack(walls)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def test_detects_expected_planes(self):
        import open3d as o3d
        from app.pipeline.segment import extract_wall_planes
        pcd = self._make_axis_aligned_box()
        planes = extract_wall_planes(pcd, voxel_size=0.05)
        # Should find at least 2 planes (4 is ideal but depends on resolution)
        assert len(planes) >= 2, f"Expected >= 2 wall planes, got {len(planes)}"

    def test_plane_normals_are_horizontal(self):
        import open3d as o3d
        from app.pipeline.segment import extract_wall_planes
        pcd = self._make_axis_aligned_box()
        planes = extract_wall_planes(pcd)
        for p in planes:
            # Normal z-component should be small for a vertical wall
            nz = abs(p.plane_eq[2])
            assert nz < 0.25, f"Wall plane has large z-component {nz:.3f} — not vertical"
