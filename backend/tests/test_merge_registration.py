"""Tests for multi-scan assembly (Phase 2): registration quality gates,
sequential pairwise ICP with the upload/walking-order prior, and pose-graph
global optimization.

All fixtures are synthetic point clouds with KNOWN rigid transforms, so the
registration error is exactly checkable: for scan ``i`` with ground-truth
world points ``W_i`` and applied transform ``T_i`` (local = T_i @ world), a
correct registration recovers a pose with ``pose_i @ T_i ≈ I`` — i.e. the
placed cloud lands back on ``W_i``.

The chain fixture is deliberately a star-topology killer: scan 2 shares NO
overlap with scan 0, only with scan 1.  The old star registration (every
scan → scan 0) cannot place it; the sequential walking-order prior can.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from app.pipeline import merge as merge_mod
from app.pipeline.merge import (
    RegistrationGate,
    ScanRegistration,
    default_gate_for_voxel,
    merge_scans,
    register_scans_sequential,
)

RNG_SEED = 20260705


# ── Synthetic geometry helpers ───────────────────────────────────────────────

def _wall_points(
    p: tuple[float, float],
    q: tuple[float, float],
    height: float = 2.7,
    spacing: float = 0.08,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a vertical wall plane from p to q as a regular grid + jitter."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    length = float(np.linalg.norm(q - p))
    n_along = max(2, int(length / spacing))
    n_up = max(2, int(height / spacing))
    t = np.linspace(0.0, 1.0, n_along)
    z = np.linspace(0.0, height, n_up)
    tt, zz = np.meshgrid(t, z)
    xy = p[None, :] + tt.reshape(-1, 1) * (q - p)[None, :]
    pts = np.column_stack([xy, zz.reshape(-1)])
    if rng is not None:
        pts = pts + rng.normal(0.0, 0.004, size=pts.shape)
    return pts


def _box_points(
    cx: float, cy: float, size: float,
    height: float = 2.0, spacing: float = 0.08,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A square column footprint — a distinctive feature that breaks the
    rotational symmetry of rectangular rooms (FPFH needs asymmetry)."""
    h = size / 2.0
    corners = [
        (cx - h, cy - h), (cx + h, cy - h),
        (cx + h, cy + h), (cx - h, cy + h),
    ]
    walls = [
        _wall_points(corners[k], corners[(k + 1) % 4], height, spacing, rng)
        for k in range(4)
    ]
    return np.vstack(walls)


def _slab_points(
    x0: float, y0: float, x1: float, y1: float,
    z: float = 0.0,
    spacing: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A horizontal slab (floor at z=0, ceiling at z=height)."""
    xs = np.arange(x0, x1, spacing)
    ys = np.arange(y0, y1, spacing)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([
        xx.reshape(-1), yy.reshape(-1), np.full(xx.size, float(z)),
    ])
    if rng is not None:
        pts = pts + rng.normal(0.0, 0.004, size=pts.shape)
    return pts


def _strip_building_points() -> np.ndarray:
    """A three-room strip building whose rooms are GEOMETRICALLY DISTINCT —
    different depths (6 / 4 / 7 m), a differently-angled diagonal wall and a
    differently-sized column in each room.  One global point set — scans are
    crops of THIS array so overlap regions contain identical surface samples.

    Distinctness matters: with identical rectangular rooms, sliding a scan
    exactly one room over yields MORE raw ICP consensus than the true
    alignment (floors/ceilings match anywhere), which is exactly the failure
    mode honest registration must not fall into.  Floor and ceiling slabs
    are included so unit detection sees a plausible 2.7 m gap.

      room 1: x ∈ [0, 5],  y ∈ [0, 6]
      room 2: x ∈ [5, 12], y ∈ [0, 4]
      room 3: x ∈ [12, 18], y ∈ [0, 7]
    """
    rng = np.random.default_rng(RNG_SEED)
    parts = [
        # bottom wall (shared by all rooms)
        _wall_points((0, 0), (18, 0), rng=rng),
        # room 1 shell (depth 6)
        _wall_points((0, 0), (0, 6), rng=rng),
        _wall_points((0, 6), (5, 6), rng=rng),
        _wall_points((5, 6), (5, 0), rng=rng),          # divider 1
        # room 2 shell (depth 4)
        _wall_points((5, 4), (12, 4), rng=rng),
        _wall_points((12, 4), (12, 0), rng=rng),        # divider 2
        # room 3 shell (depth 7)
        _wall_points((12, 7), (18, 7), rng=rng),
        _wall_points((18, 7), (18, 0), rng=rng),
        _wall_points((12, 4), (12, 7), rng=rng),
        # one uniquely-angled diagonal wall per room
        _wall_points((1.0, 1.0), (3.0, 3.0), rng=rng),
        _wall_points((7.0, 3.5), (9.5, 2.0), rng=rng),
        _wall_points((13.0, 6.0), (16.5, 5.0), rng=rng),
        # one uniquely-sized column per room
        _box_points(2.5, 4.5, 0.6, rng=rng),
        _box_points(10.5, 1.0, 0.4, rng=rng),
        _box_points(15.0, 2.0, 0.9, rng=rng),
        # floor + ceiling slabs per room
        _slab_points(0, 0, 5, 6, z=0.0, rng=rng),
        _slab_points(5, 0, 12, 4, z=0.0, rng=rng),
        _slab_points(12, 0, 18, 7, z=0.0, rng=rng),
        _slab_points(0, 0, 5, 6, z=2.7, rng=rng),
        _slab_points(5, 0, 12, 4, z=2.7, rng=rng),
        _slab_points(12, 0, 18, 7, z=2.7, rng=rng),
    ]
    return np.vstack(parts)


def _ring_corridor_points() -> np.ndarray:
    """A 16 x 16 m ring corridor around an OFF-CENTRE core so every side
    has a different corridor width (bottom 4 / right 5 / top 6 / left 3 m),
    plus a differently-sized column and a uniquely-angled diagonal wall
    near each corner — the four sides are unambiguous."""
    rng = np.random.default_rng(RNG_SEED + 1)
    parts = [
        # outer shell
        _wall_points((0, 0), (16, 0), rng=rng),
        _wall_points((16, 0), (16, 16), rng=rng),
        _wall_points((16, 16), (0, 16), rng=rng),
        _wall_points((0, 16), (0, 0), rng=rng),
        # inner core x ∈ [3, 11], y ∈ [4, 10]
        _wall_points((3, 4), (11, 4), rng=rng),
        _wall_points((11, 4), (11, 10), rng=rng),
        _wall_points((11, 10), (3, 10), rng=rng),
        _wall_points((3, 10), (3, 4), rng=rng),
        # unique corner features: column + diagonal per corner
        _box_points(1.5, 2.0, 0.5, rng=rng),
        _wall_points((0.8, 0.8), (2.4, 1.6), rng=rng),
        _box_points(14.0, 2.0, 0.9, rng=rng),
        _wall_points((13.0, 1.0), (14.0, 3.0), rng=rng),
        _box_points(14.0, 14.0, 0.4, rng=rng),
        _wall_points((12.5, 14.5), (14.5, 12.5), rng=rng),
        _box_points(1.5, 13.5, 0.7, rng=rng),
        _wall_points((1.0, 12.0), (2.5, 14.8), rng=rng),
        _box_points(8.0, 2.0, 0.6, rng=rng),
        # corridor floor + ceiling (ring around the core)
        _slab_points(0, 0, 16, 4, z=0.0, rng=rng),
        _slab_points(0, 10, 16, 16, z=0.0, rng=rng),
        _slab_points(0, 4, 3, 10, z=0.0, rng=rng),
        _slab_points(11, 4, 16, 10, z=0.0, rng=rng),
        _slab_points(0, 0, 16, 4, z=2.7, rng=rng),
        _slab_points(0, 10, 16, 16, z=2.7, rng=rng),
        _slab_points(0, 4, 3, 10, z=2.7, rng=rng),
        _slab_points(11, 4, 16, 10, z=2.7, rng=rng),
    ]
    return np.vstack(parts)


def _rigid(theta_deg: float, tx: float, ty: float, tz: float = 0.0) -> np.ndarray:
    """4x4 rigid transform: rotate ``theta_deg`` about Z, then translate."""
    th = np.radians(theta_deg)
    T = np.eye(4)
    T[0, 0] = np.cos(th)
    T[0, 1] = -np.sin(th)
    T[1, 0] = np.sin(th)
    T[1, 1] = np.cos(th)
    T[:3, 3] = (tx, ty, tz)
    return T


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def _cloud(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pcd


def _crop_x(pts: np.ndarray, x0: float, x1: float) -> np.ndarray:
    mask = (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
    return pts[mask]


def _mean_registration_error(
    pose: np.ndarray, T_applied: np.ndarray, world_pts: np.ndarray,
) -> float:
    """Mean distance between the placed scan and its ground-truth position."""
    placed = _apply(pose, _apply(T_applied, world_pts))
    return float(np.linalg.norm(placed - world_pts, axis=1).mean())


def _noise_blob(n: int = 2000, offset: float = 100.0) -> np.ndarray:
    """Uniform random junk — shares no surface with any building scan."""
    rng = np.random.default_rng(RNG_SEED + 2)
    return rng.uniform(0.0, 3.0, size=(n, 3)) + offset


VOXEL = 0.08


# ── Registration gate (pure unit tests, exact thresholds) ────────────────────

class TestRegistrationGate:
    def test_passes_when_both_metrics_good(self):
        gate = RegistrationGate(min_fitness=0.25, max_inlier_rmse_m=0.05)
        passed, reason = gate.check(fitness=0.80, inlier_rmse_m=0.01)
        assert passed is True
        assert reason == ""

    def test_boundary_values_pass(self):
        gate = RegistrationGate(min_fitness=0.25, max_inlier_rmse_m=0.05)
        passed, _ = gate.check(fitness=0.25, inlier_rmse_m=0.05)
        assert passed is True

    def test_low_fitness_fails(self):
        gate = RegistrationGate(min_fitness=0.25, max_inlier_rmse_m=0.05)
        passed, reason = gate.check(fitness=0.249, inlier_rmse_m=0.01)
        assert passed is False
        assert "fitness" in reason

    def test_high_rmse_fails(self):
        gate = RegistrationGate(min_fitness=0.25, max_inlier_rmse_m=0.05)
        passed, reason = gate.check(fitness=0.90, inlier_rmse_m=0.051)
        assert passed is False
        assert "RMSE" in reason

    def test_both_failures_reported(self):
        gate = RegistrationGate(min_fitness=0.25, max_inlier_rmse_m=0.05)
        passed, reason = gate.check(fitness=0.0, inlier_rmse_m=1.0)
        assert passed is False
        assert "fitness" in reason and "RMSE" in reason

    def test_zero_fitness_never_passes_default_gate(self):
        # A scan with no correspondences at all must never place.
        passed, _ = default_gate_for_voxel(0.03).check(
            fitness=0.0, inlier_rmse_m=0.0,
        )
        assert passed is False

    def test_default_gate_scales_rmse_with_voxel(self):
        # Production merge voxel is 3 cm → RMSE gate 4.8 cm.
        gate = default_gate_for_voxel(0.03)
        assert gate.min_fitness == pytest.approx(0.25)
        assert gate.max_inlier_rmse_m == pytest.approx(0.048)


class TestScanRegistrationSerialization:
    def test_json_dict_shape(self):
        reg = ScanRegistration(
            scan_index=3, scan_name="room4.laz", placed=False,
            fitness=0.12, inlier_rmse_m=0.041,
            transformation=np.eye(4), method="sequential_icp",
            reason="failed registration gate vs scan 3: fitness 0.120 < required 0.250",
        )
        d = reg.to_json_dict()
        assert d["scan_index"] == 3
        assert d["scan_name"] == "room4.laz"
        assert d["placed"] is False
        assert d["fitness"] == pytest.approx(0.12)
        assert d["inlier_rmse_m"] == pytest.approx(0.041)
        assert d["inlier_rmse_mm"] == pytest.approx(41.0)
        assert d["transformation"] == np.eye(4).tolist()
        assert "fitness" in d["reason"]


# ── Sequential registration with the walking-order prior ────────────────────

class TestSequentialRegistration:
    """Three overlapping strip scans.  Scan 2 overlaps scan 1 but NOT scan 0
    — the star topology (every scan → scan 0) cannot place it."""

    def _make_scans(self):
        world = _strip_building_points()
        crops = [
            _crop_x(world, 0.0, 9.0),     # room 1 + into room 2
            _crop_x(world, 4.0, 14.0),    # room 2 + overlap both sides
            _crop_x(world, 10.0, 18.0),   # room 3 — zero overlap with scan 0
        ]
        transforms = [
            np.eye(4),
            _rigid(25.0, -5.0, 2.0, 0.05),
            _rigid(-40.0, 3.0, -4.0, -0.03),
        ]
        clouds = [_cloud(_apply(T, c)) for T, c in zip(transforms, crops)]
        return crops, transforms, clouds

    def test_all_scans_placed(self):
        crops, transforms, clouds = self._make_scans()
        placed_clouds, regs = register_scans_sequential(clouds, VOXEL)
        assert len(regs) == 3
        assert all(r.placed for r in regs), [
            (r.scan_index, r.reason) for r in regs if not r.placed
        ]
        assert len(placed_clouds) == 3

    def test_registration_error_exactly_checkable(self):
        """pose_i @ T_i must be ≈ identity: each placed scan lands back on
        its ground-truth world points to within registration resolution."""
        crops, transforms, clouds = self._make_scans()
        _placed, regs = register_scans_sequential(clouds, VOXEL)
        for reg, T, world_pts in zip(regs, transforms, crops):
            assert reg.placed
            err = _mean_registration_error(reg.transformation, T, world_pts)
            assert err < 0.03, (
                f"scan {reg.scan_index}: mean registration error {err:.4f} m"
            )

    def test_reference_scan_identity(self):
        _crops, _transforms, clouds = self._make_scans()
        _placed, regs = register_scans_sequential(clouds, VOXEL)
        assert regs[0].method == "reference"
        np.testing.assert_allclose(regs[0].transformation, np.eye(4))

    def test_gate_metrics_recorded_for_placed_scans(self):
        _crops, _transforms, clouds = self._make_scans()
        _placed, regs = register_scans_sequential(clouds, VOXEL)
        default_gate = default_gate_for_voxel(VOXEL)
        for reg in regs[1:]:
            assert reg.fitness >= default_gate.min_fitness
            assert reg.inlier_rmse_m <= default_gate.max_inlier_rmse_m


class TestUnplacedScans:
    """A scan that cannot be honestly registered must be flagged unplaced —
    never silently guessed into position."""

    def test_noise_scan_flagged_unplaced(self):
        world = _strip_building_points()
        clouds = [
            _cloud(_crop_x(world, 0.0, 9.0)),
            _cloud(_noise_blob()),
        ]
        placed_clouds, regs = register_scans_sequential(clouds, VOXEL)
        assert regs[0].placed is True
        assert regs[1].placed is False
        assert regs[1].reason != ""
        assert len(placed_clouds) == 1   # noise excluded from the assembly

    def test_registration_continues_past_unplaced_scan(self):
        """After a failure, the next scan registers against the last scan
        that DID place (scan 0 here) — one bad file must not sink the job."""
        world = _strip_building_points()
        good_crop = _crop_x(world, 4.0, 14.0)
        T_good = _rigid(30.0, -2.0, 3.0)
        clouds = [
            _cloud(_crop_x(world, 0.0, 9.0)),
            _cloud(_noise_blob()),
            _cloud(_apply(T_good, good_crop)),
        ]
        placed_clouds, regs = register_scans_sequential(clouds, VOXEL)
        assert [r.placed for r in regs] == [True, False, True]
        assert len(placed_clouds) == 2
        err = _mean_registration_error(regs[2].transformation, T_good, good_crop)
        assert err < 0.03

    def test_unplaced_transform_is_identity(self):
        """An unplaced scan's transform must NOT carry the rejected guess."""
        world = _strip_building_points()
        clouds = [
            _cloud(_crop_x(world, 0.0, 9.0)),
            _cloud(_noise_blob()),
        ]
        _placed, regs = register_scans_sequential(clouds, VOXEL)
        np.testing.assert_allclose(regs[1].transformation, np.eye(4))


# ── Pose-graph global optimization over a loop closure ──────────────────────

class TestPoseGraphLoop:
    """Four scans walking around a square ring corridor: scan 3 overlaps
    both scan 2 (chain) and scan 0 (loop closure).  Pose-graph optimization
    must place every scan back on its ground-truth position."""

    def _make_loop_scans(self):
        world = _ring_corridor_points()
        x, y = world[:, 0], world[:, 1]
        # L-shaped crops: each scan covers two adjacent sides of the ring,
        # so every consecutive pair shares one full side (with its unique
        # columns + diagonals) and scan 3 shares the bottom side with scan
        # 0 — the loop closure.
        crops = [
            world[(y <= 5.5) | (x >= 10.5)],   # bottom + right
            world[(x >= 10.5) | (y >= 10.5)],  # right + top
            world[(y >= 10.5) | (x <= 5.5)],   # top + left
            world[(x <= 5.5) | (y <= 5.5)],    # left + bottom → closes loop
        ]
        transforms = [
            np.eye(4),
            _rigid(15.0, -3.0, 1.5, 0.02),
            _rigid(-30.0, 2.0, -2.5, -0.04),
            _rigid(50.0, -1.0, 4.0, 0.03),
        ]
        clouds = [_cloud(_apply(T, c)) for T, c in zip(transforms, crops)]
        return crops, transforms, clouds

    def test_loop_scans_all_placed_with_pose_graph(self):
        crops, transforms, clouds = self._make_loop_scans()
        placed_clouds, regs = register_scans_sequential(clouds, VOXEL)
        assert all(r.placed for r in regs), [
            (r.scan_index, r.reason) for r in regs if not r.placed
        ]
        # ≥3 scans placed → the pose graph ran and refined the chain poses.
        assert all(r.method == "pose_graph" for r in regs[1:])
        for reg, T, world_pts in zip(regs, transforms, crops):
            err = _mean_registration_error(reg.transformation, T, world_pts)
            assert err < 0.05, (
                f"scan {reg.scan_index}: mean error {err:.4f} m after "
                f"pose-graph optimization"
            )

    def test_pose_graph_can_be_disabled(self):
        _crops, _transforms, clouds = self._make_loop_scans()
        _placed, regs = register_scans_sequential(
            clouds, VOXEL, pose_graph_optimize=False,
        )
        assert all(r.placed for r in regs)
        assert all(r.method == "sequential_icp" for r in regs[1:])


# ── merge_scans integration: unplaced scans surfaced in MergeResult ──────────

class TestMergeScansRegistrationSurface:
    def _write_ply(self, tmp_path, name: str, pts: np.ndarray):
        path = tmp_path / name
        o3d.io.write_point_cloud(str(path), _cloud(pts))
        return path

    def test_merge_flags_unplaced_scan(self, tmp_path, monkeypatch):
        world = _strip_building_points()
        p0 = self._write_ply(tmp_path, "scan0.ply", _crop_x(world, 0.0, 9.0))
        p1 = self._write_ply(tmp_path, "scan1.ply", _noise_blob())
        T2 = _rigid(20.0, -2.0, 1.0)
        p2 = self._write_ply(
            tmp_path, "scan2.ply", _apply(T2, _crop_x(world, 4.0, 14.0)),
        )

        # Force the ICP path (frame heuristics are not under test here).
        monkeypatch.setattr(
            merge_mod, "_scans_share_coordinate_frame", lambda clouds: False,
        )
        result = merge_scans([p0, p1, p2], voxel_size=VOXEL)

        assert result.strategy == "icp_registered"
        assert len(result.registrations) == 3
        assert [r.placed for r in result.registrations] == [True, False, True]
        assert [r.scan_name for r in result.unplaced] == ["scan1.ply"]
        assert result.unplaced[0].reason != ""
        # The noise blob (offset at +100 m) must NOT be in the merged cloud.
        merged_pts = np.asarray(result.merged.points)
        assert merged_pts[:, 0].max() < 50.0

    def test_pre_registered_scans_all_marked_placed(self, tmp_path):
        world = _strip_building_points()
        p0 = self._write_ply(tmp_path, "a.ply", _crop_x(world, 0.0, 9.0))
        p1 = self._write_ply(tmp_path, "b.ply", _crop_x(world, 4.0, 14.0))
        result = merge_scans([p0, p1], voxel_size=VOXEL)
        assert result.strategy == "concatenate"
        assert len(result.registrations) == 2
        assert all(r.placed for r in result.registrations)
        assert all(r.method == "pre_registered" for r in result.registrations)
        assert result.unplaced == []

    def test_single_scan_registration_record(self, tmp_path):
        world = _strip_building_points()
        p0 = self._write_ply(tmp_path, "only.ply", _crop_x(world, 0.0, 9.0))
        result = merge_scans([p0], voxel_size=VOXEL)
        assert result.strategy == "single"
        assert len(result.registrations) == 1
        assert result.registrations[0].placed is True
        assert result.registrations[0].method == "single"
