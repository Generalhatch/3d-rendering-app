"""Accuracy guards for voxel downsample / auto-escalation.

The CAD-quality plan treats 5 mm as the production default and 10 mm as the
coarsest spacing that still preserves wall features ≥ 1 cm.  Auto-escalation
exists so dense multi-scan mosaics don't hang the pipeline — but it must
never silently thin past that accuracy ceiling.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from app.vectorize import ingest_downsample as ds


def _uniform_cloud(n: int, extent: float = 10.0, seed: int = 0) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        rng.uniform(0, extent, size=(n, 3)),
    )
    return pcd


class TestVoxelDownsampleAutoAccuracy:
    def test_accuracy_ceiling_is_10mm(self):
        assert ds.ACCURACY_MAX_VOXEL_M == pytest.approx(0.01)

    def test_stays_at_requested_voxel_when_already_under_target(self):
        """A modest cloud at 5 mm must NOT escalate — keep full fidelity."""
        pcd = _uniform_cloud(5_000)
        result = ds.voxel_downsample_auto(
            pcd, initial_voxel_m=0.005, target_max_points=20_000_000,
        )
        assert result.voxel_m == pytest.approx(0.005)
        assert result.n_after <= result.n_before

    def test_never_exceeds_10mm_even_if_caller_asks_for_50mm(self):
        """A larger max_voxel_m argument must be clamped to the accuracy ceiling."""
        pcd = _uniform_cloud(80_000, extent=2.0, seed=3)
        result = ds.voxel_downsample_auto(
            pcd,
            initial_voxel_m=0.005,
            target_max_points=1_000,  # force escalation
            max_voxel_m=0.05,         # caller tries 50 mm — must be ignored
        )
        assert result.voxel_m <= ds.ACCURACY_MAX_VOXEL_M + 1e-12
        assert result.voxel_m <= 0.01 + 1e-12

    def test_escalates_toward_ceiling_when_still_too_dense(self):
        """Dense cloud under a tiny target should climb 5 → 7.5 → 10 mm, then stop."""
        # Pack points tightly so 5 mm still leaves many voxels occupied.
        pcd = _uniform_cloud(200_000, extent=1.0, seed=7)
        result = ds.voxel_downsample_auto(
            pcd,
            initial_voxel_m=0.005,
            target_max_points=500,  # unreachable at 10 mm on this cloud
        )
        assert result.voxel_m == pytest.approx(ds.ACCURACY_MAX_VOXEL_M)
        # Even at the ceiling we must still have produced a cloud.
        assert result.n_after > 0
        assert result.n_after < result.n_before

    def test_fixed_downsample_preserves_extent(self):
        """Centroid voxel downsample must not shrink the bounding box."""
        pcd = _uniform_cloud(20_000, extent=8.0, seed=11)
        before = np.asarray(pcd.points)
        result = ds.voxel_downsample(pcd, voxel_m=0.01)
        after = np.asarray(result.pcd.points)
        # Extent should stay within one voxel of the original.
        assert after[:, 0].min() == pytest.approx(before[:, 0].min(), abs=0.01)
        assert after[:, 0].max() == pytest.approx(before[:, 0].max(), abs=0.01)
        assert after[:, 1].min() == pytest.approx(before[:, 1].min(), abs=0.01)
        assert after[:, 1].max() == pytest.approx(before[:, 1].max(), abs=0.01)
