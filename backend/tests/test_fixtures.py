"""Unit tests for fixture (wall protrusion) detection."""
from __future__ import annotations

import numpy as np
import pytest


def _make_wall_plane():
    """Create a synthetic wall plane and points in front of it."""
    from app.pipeline.segment import WallPlane

    # Wall at x=5, normal pointing in +x direction
    plane_eq = np.array([1.0, 0.0, 0.0, -5.0])  # x=5 plane
    rng = np.random.default_rng(99)
    # Inlier points on the wall (x≈5)
    ys = rng.uniform(0, 4, 500)
    zs = rng.uniform(0, 3, 500)
    xs = np.full(500, 5.0) + rng.normal(0, 0.01, 500)
    inliers = np.column_stack([xs, ys, zs])
    return WallPlane(id="wall-001", plane_eq=plane_eq, inlier_points=inliers)


def _make_scan_with_fixture(wall_plane, fixture_x=5.10, center_y=2.0, center_z=1.2, n=200):
    """Generate scan points including a synthetic fixture cluster."""
    rng = np.random.default_rng(77)
    # Background scan (walls, floor)
    bg = rng.uniform([0, 0, 0], [10, 4, 3], (5000, 3))
    # Fixture cluster: 10cm in front of wall at x=5
    fx = rng.normal(fixture_x, 0.02, n)
    fy = rng.normal(center_y, 0.08, n)
    fz = rng.normal(center_z, 0.08, n)
    cluster = np.column_stack([fx, fy, fz])
    return np.vstack([bg, cluster])


try:
    from app.pipeline.fixtures import detect_fixtures
    from app.pipeline.segment import WallPlane
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


@pytest.mark.skipif(not _HAS_DEPS, reason="sklearn not installed")
class TestFixtureDetection:
    def test_detects_known_fixture(self):
        wall = _make_wall_plane()
        scan = _make_scan_with_fixture(wall, fixture_x=5.10)
        fixtures = detect_fixtures(
            scan,
            [wall],
            min_protrusion_m=0.02,
            max_protrusion_m=0.30,
            min_cluster_points=30,
        )
        assert len(fixtures) >= 1, f"Expected at least 1 fixture, got {len(fixtures)}"

    def test_fixture_protrusion_depth_in_range(self):
        wall = _make_wall_plane()
        scan = _make_scan_with_fixture(wall, fixture_x=5.10)
        fixtures = detect_fixtures(scan, [wall], min_cluster_points=30)
        for fx in fixtures:
            assert 0.01 <= fx.protrusion_depth_m <= 0.35, (
                f"Protrusion depth {fx.protrusion_depth_m:.3f}m out of expected range"
            )

    def test_no_false_positives_on_clean_wall(self):
        """A flat wall with no protrusions should produce no fixtures."""
        wall = _make_wall_plane()
        rng = np.random.default_rng(11)
        # Points only on the wall surface itself (no protrusions)
        xs = np.full(2000, 5.0) + rng.normal(0, 0.005, 2000)
        ys = rng.uniform(0, 4, 2000)
        zs = rng.uniform(0, 3, 2000)
        scan = np.column_stack([xs, ys, zs])
        fixtures = detect_fixtures(scan, [wall], min_cluster_points=50)
        assert len(fixtures) == 0, f"Expected 0 fixtures, got {len(fixtures)}"
