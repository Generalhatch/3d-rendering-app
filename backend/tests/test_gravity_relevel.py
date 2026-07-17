"""Gravity re-leveling: tilted scans must be rotated so the floor is level.

Synthetic fixture: a 10×10 m room (floor slab + ceiling slab + four
perimeter walls) rigidly rotated by a known tilt.  After
``gravity_relevel`` the floor points must sit at height ≈ 0, the transform
must be rigid, and the safety rails (ignore sub-threshold tilt, refuse
implausible tilt) must hold.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from app.pipeline.relevel import RelevelResult, gravity_relevel
from app.pipeline.slicing import FloorReference, detect_floor


def _room_points(step: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(all_points, floor_mask) for a level 10×10×2.5 m room at z=0."""
    xs = np.arange(0.0, 10.0 + step / 2, step)
    zs = np.arange(0.0, 2.5 + step / 2, step)

    gx, gy = np.meshgrid(xs, xs)
    floor = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    ceiling = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 2.5)])

    walls = []
    wx, wz = np.meshgrid(xs, zs)
    for y0 in (0.0, 10.0):
        walls.append(np.column_stack([wx.ravel(), np.full(wx.size, y0), wz.ravel()]))
    for x0 in (0.0, 10.0):
        walls.append(np.column_stack([np.full(wx.size, x0), wx.ravel(), wz.ravel()]))

    pts = np.vstack([floor, ceiling] + walls)
    floor_mask = np.zeros(len(pts), dtype=bool)
    floor_mask[: len(floor)] = True
    return pts, floor_mask


def _rot_x(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(r), -np.sin(r)],
        [0.0, np.sin(r), np.cos(r)],
    ])


def _cloud(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


class TestGravityRelevel:
    def test_tilted_floor_releveled_to_height_zero(self):
        """3° tilt about X + 0.5 m offset → floor lands at z ≈ 0."""
        pts, floor_mask = _room_points()
        tilted = pts @ _rot_x(3.0).T
        tilted[:, 2] += 0.5
        pcd = _cloud(tilted)

        result = gravity_relevel(pcd, floor=detect_floor(pcd))

        assert result.applied is True
        assert result.reason == "releveled"
        assert result.tilt_deg == pytest.approx(3.0, abs=0.3)
        out = np.asarray(result.pcd.points)
        floor_z = out[floor_mask, 2]
        # RANSAC plane on a clean synthetic slab is near-exact.
        assert float(np.abs(floor_z).max()) < 0.03
        assert float(np.abs(floor_z.mean())) < 0.01

    def test_transform_is_rigid(self):
        """Rotation orthonormal, det=+1, pairwise distances preserved."""
        pts, _ = _room_points()
        tilted = pts @ _rot_x(4.0).T
        pcd = _cloud(tilted.copy())

        result = gravity_relevel(pcd, floor=detect_floor(pcd))

        assert result.applied
        R = result.transform[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-12)

        out = np.asarray(result.pcd.points)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(pts), size=(50, 2))
        d_before = np.linalg.norm(tilted[idx[:, 0]] - tilted[idx[:, 1]], axis=1)
        d_after = np.linalg.norm(out[idx[:, 0]] - out[idx[:, 1]], axis=1)
        np.testing.assert_allclose(d_after, d_before, atol=1e-9)

    def test_level_floor_left_untouched(self):
        """Tilt below 0.5° threshold → no transform at all."""
        pts, _ = _room_points()
        pcd = _cloud(pts.copy())

        result = gravity_relevel(pcd, floor=detect_floor(pcd))

        assert result.applied is False
        assert result.reason == "below_threshold"
        assert result.tilt_deg < 0.5
        np.testing.assert_allclose(result.transform, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(np.asarray(result.pcd.points), pts, atol=1e-12)

    def test_excessive_tilt_refused(self):
        """A 'floor' plane 20° off axis is a mis-detection — refuse to rotate
        the whole building by it."""
        pts, _ = _room_points()
        pcd = _cloud(pts.copy())
        # Hand-built FloorReference with a 20°-tilted plane equation.
        n = _rot_x(20.0) @ np.array([0.0, 0.0, 1.0])
        floor = FloorReference(
            plane_eq=np.array([n[0], n[1], n[2], 0.0]),
            up_normal=n,
            floor_z_estimate=0.0,
            inlier_count=1000,
            axis_idx=2,
        )

        result = gravity_relevel(pcd, floor=floor)

        assert result.applied is False
        assert result.reason == "excessive_tilt"
        assert result.tilt_deg == pytest.approx(20.0, abs=0.01)
        np.testing.assert_allclose(np.asarray(result.pcd.points), pts, atol=1e-12)

    def test_relevel_result_defaults(self):
        """Identity default transform on the dataclass."""
        pcd = _cloud(np.zeros((1, 3)))
        r = RelevelResult(pcd=pcd)
        np.testing.assert_allclose(r.transform, np.eye(4))
        assert r.applied is False
