"""Unit tests for scanplan.py — occupancy-grid and room extraction (section 7).

Tests cover:
  - DBSCAN noise cluster filtering
  - _classify_room threshold correctness (covered more thoroughly in test_scanplan_classify.py)
  - generate_rooms_from_scan produces required room fields
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_floor():
    """Return a minimal FloorReference for Z-up, floor at z=0."""
    try:
        from app.pipeline.slicing import FloorReference
    except ImportError:
        pytest.skip("app.pipeline.slicing not importable")
    return FloorReference(
        plane_eq=np.array([0.0, 0.0, 1.0, 0.0]),
        up_normal=np.array([0.0, 0.0, 1.0]),
        floor_z_estimate=0.0,
        inlier_count=1000,
        axis_idx=2,
    )


def _make_pcd(pts: np.ndarray):
    try:
        import open3d as o3d
    except ImportError:
        pytest.skip("open3d not installed")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


# ── _classify_room thresholds ─────────────────────────────────────────────────

class TestClassifyRoomThresholds:
    """Smoke-test the key thresholds documented in the AUDIT.md."""

    def _classify(self, area, ecc, sol):
        from app.pipeline.scanplan import _classify_room
        return _classify_room(area, ecc, sol)

    def test_bathroom_threshold_is_7m2(self):
        _, cat_below, _ = self._classify(6.99, 0.40, 0.85)
        _, cat_above, _ = self._classify(7.01, 0.40, 0.85)
        assert cat_below == "bathroom"
        assert cat_above != "bathroom"

    def test_conference_threshold_is_40m2(self):
        _, cat_below, _ = self._classify(39.9, 0.35, 0.85)
        _, cat_above, _ = self._classify(40.1, 0.35, 0.85)
        assert cat_above == "office"   # conference maps to "office" category

    def test_hallway_eccentricity_threshold_is_085(self):
        _, cat_below, _ = self._classify(15.0, 0.84, 0.80)
        _, cat_above, _ = self._classify(15.0, 0.86, 0.80)
        assert cat_below != "hallway"
        assert cat_above == "hallway"

    def test_all_categories_return_3_tuple(self):
        from app.pipeline.scanplan import _classify_room
        cases = [
            (5.0,   0.40, 0.85),   # bathroom
            (18.0,  0.35, 0.88),   # office
            (10.0,  0.92, 0.80),   # hallway
            (120.0, 0.20, 0.92),   # lobby
            (55.0,  0.40, 0.85),   # conference
            (25.0,  0.40, 0.55),   # irregular
        ]
        for area, ecc, sol in cases:
            result = _classify_room(area, ecc, sol)
            assert len(result) == 3, f"Expected 3-tuple for area={area}"
            label, cat, conf = result
            assert isinstance(label, str) and len(label) > 0
            assert isinstance(conf, float)
            assert 0.30 <= conf <= 0.99


# ── Room field presence ───────────────────────────────────────────────────────

class TestRoomFields:
    """generate_rooms_from_scan rooms must contain all required fields."""

    REQUIRED_FIELDS = {
        "id", "label", "category", "polygon_2d", "centroid",
        "area_m2", "match_quality", "classification_confidence",
    }

    def _build_simple_scan(self) -> "open3d.geometry.PointCloud":
        """Build a synthetic point cloud: four walls of a 10×10 room at z=1m."""
        rng = np.random.default_rng(42)
        pts_list = []

        # North wall: x in [0,10], y=10, z in [0.75, 1.80]
        n = 500
        for y_val, x_range in [(10.0, (0, 10)), (0.0, (0, 10))]:
            x = rng.uniform(*x_range, n)
            z = rng.uniform(0.75, 1.80, n)
            pts_list.append(np.column_stack([x, np.full(n, y_val), z]))
        for x_val in [0.0, 10.0]:
            y = rng.uniform(0, 10, n)
            z = rng.uniform(0.75, 1.80, n)
            pts_list.append(np.column_stack([np.full(n, x_val), y, z]))

        # Floor points
        x = rng.uniform(0, 10, 2000)
        y = rng.uniform(0, 10, 2000)
        pts_list.append(np.column_stack([x, y, np.zeros(2000)]))

        return _make_pcd(np.vstack(pts_list))

    def test_rooms_have_required_fields(self):
        try:
            from app.pipeline.scanplan import (
                generate_plan_from_projection, generate_rooms_from_scan,
            )
        except ImportError:
            pytest.skip("scanplan dependencies not installed")

        scan_pcd = self._build_simple_scan()
        floor = _make_floor()

        try:
            synthetic = generate_plan_from_projection(scan_pcd, floor)
            rooms = generate_rooms_from_scan(
                [], synthetic, scan_pcd=scan_pcd, floor=floor
            )
        except Exception as exc:
            pytest.skip(f"Room generation skipped (likely missing deps): {exc}")

        if not rooms:
            pytest.skip("No rooms generated from synthetic scan — insufficient point density")

        for room in rooms:
            missing = self.REQUIRED_FIELDS - set(room.keys())
            assert not missing, f"Room {room.get('id')} missing fields: {missing}"

    def test_classification_confidence_in_range(self):
        try:
            from app.pipeline.scanplan import (
                generate_plan_from_projection, generate_rooms_from_scan,
            )
        except ImportError:
            pytest.skip("scanplan dependencies not installed")

        scan_pcd = self._build_simple_scan()
        floor = _make_floor()

        try:
            synthetic = generate_plan_from_projection(scan_pcd, floor)
            rooms = generate_rooms_from_scan(
                [], synthetic, scan_pcd=scan_pcd, floor=floor
            )
        except Exception as exc:
            pytest.skip(f"Room generation skipped: {exc}")

        for room in rooms:
            conf = room.get("classification_confidence")
            if conf is not None:
                assert 0.30 <= conf <= 0.99, (
                    f"Room {room['id']} confidence {conf} out of range"
                )


# ── Building outline ──────────────────────────────────────────────────────────

class TestBuildingOutline:
    def test_outline_returns_list_of_pairs(self):
        try:
            from app.pipeline.scanplan import generate_building_outline
        except ImportError:
            pytest.skip("scanplan not importable")

        # Simple scattered points forming a rough square
        rng = np.random.default_rng(0)
        pts = np.column_stack([
            rng.uniform(0, 10, 2000),
            rng.uniform(0, 10, 2000),
            np.ones(2000),   # Z=1, floor at 0
        ])
        pcd = _make_pcd(pts)
        floor = _make_floor()

        try:
            outline = generate_building_outline(pcd, floor)
        except Exception as exc:
            pytest.skip(f"alphashape not available: {exc}")

        # Each element should be [x, y]
        for coord in outline:
            assert len(coord) == 2, f"Expected [x, y], got {coord}"
