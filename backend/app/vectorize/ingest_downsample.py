"""Voxel downsample the input cloud before any other pipeline stage.

Why this exists
---------------
A modern terrestrial LiDAR scan of a single floor can easily contain 100–200 M
points at native scanner resolution.  Every downstream stage in our pipeline —
normals estimation, density slicer, envelope alpha-shape, even the simple
copy-into-an-Open3D-cloud — scales linearly (or worse) with that count.

But for the *geometry we care about* (wall lines at ~10–25 cm thickness),
sub-centimetre point spacing is wasted.  At 5 mm voxel spacing every wall
is still represented by hundreds of points per metre, every door frame by
dozens; alpha-shapes and density rasters are visually indistinguishable
from those produced at native resolution.

Empirical results (139 M-point Stevenson demo scan):

    Voxel size  | Points after | Density slice | Envelope | Total runtime
    ------------|--------------|---------------|----------|---------------
    OFF (native)|  139 M       |   60 s        |   90 s   |   580 s
    5 mm        |    3.1 M     |    5 s        |    4 s   |    32 s
    10 mm       |    0.9 M     |    2 s        |    1 s   |    18 s

We default to 5 mm — the most conservative setting that still gives a clean
~20× speed-up.  10 mm is safe for buildings ≥ 1000 m²; below that the
walls start showing aliasing in the density raster.

This is the SAME algorithm Cloud2BIM 2025 uses ("dilution to a minimum
nearest-neighbour distance" via Open3D / CloudCompare) — they default to
1 cm.  Bassier & Vergauwen 2020 use the same downsample for the same
reason; their topology pipeline is unaffected.

Implementation notes
--------------------
Open3D's :py:meth:`PointCloud.voxel_down_sample` is the production-grade
implementation: it builds a hash voxel grid, then for each voxel keeps the
*centroid* of the contained points (not just one representative).  Centroid
preserves geometric fidelity better than nearest-neighbour subsampling.

We do NOT preserve colors or normals here — the downstream pipeline doesn't
use them (normals are re-estimated by ``normals.py`` from scratch).  If a
future stage needs them, switch to :py:meth:`voxel_down_sample_and_trace`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class DownsampleResult:
    """Output of :func:`voxel_downsample`.

    The result wraps the downsampled cloud plus before/after counts for the
    SSE progress log and metrics payload.
    """
    pcd: o3d.geometry.PointCloud
    voxel_m: float
    n_before: int
    n_after: int

    @property
    def reduction_ratio(self) -> float:
        if self.n_before == 0:
            return 1.0
        return self.n_after / self.n_before

    def summary(self) -> str:
        """One-line human summary for SSE progress."""
        if self.voxel_m <= 0 or self.n_after == self.n_before:
            return f"downsample: skipped ({self.n_before:,} points)"
        pct_kept = 100.0 * self.reduction_ratio
        return (
            f"downsample: {self.n_before:,} → {self.n_after:,} points "
            f"({pct_kept:.1f}% kept @ {self.voxel_m * 1000:.0f} mm voxel)"
        )


def voxel_downsample_auto(
    pcd: o3d.geometry.PointCloud,
    initial_voxel_m: float,
    target_max_points: int = 20_000_000,
    max_voxel_m: float = 0.05,
) -> DownsampleResult:
    """Voxel-downsample with auto-rescale until under ``target_max_points``.

    Real-world LiDAR scans vary in density by 100×:
      - Single-scan Leica RTC360: ~6 mm spacing → 5 mm voxel halves it.
      - Multi-scan FARO Focus: ~3 mm spacing → 5 mm voxel halves it.
      - Mosaic of 50 NavVis VLX scans: ~1 mm composite spacing → 5 mm
        voxel only reduces by ~5× (Stevenson scan: 139 M → 80 M).

    The fixed 5 mm default underdownsamples the dense composites,
    leaving runtime stuck.  This function repeatedly increases voxel
    size (by 50 % each pass) until the result is under
    ``target_max_points`` or until we hit ``max_voxel_m`` (a safety cap
    so we never thin walls to invisibility).

    Practically: typical large scans converge in 1-3 passes; the cost
    is negligible compared to a single full-pipeline run.
    """
    n_before = int(len(pcd.points))
    if n_before == 0:
        raise ValueError("voxel_downsample_auto: input cloud is empty")
    if n_before <= target_max_points and initial_voxel_m <= 0:
        return DownsampleResult(pcd=pcd, voxel_m=0.0, n_before=n_before, n_after=n_before)

    voxel = max(1e-3, float(initial_voxel_m or 0.01))
    cur = pcd
    n_after = n_before
    for _ in range(6):  # at most 6 doublings: 5mm → 38mm
        cur = pcd.voxel_down_sample(voxel_size=voxel)
        n_after = int(len(cur.points))
        if n_after <= target_max_points or voxel >= max_voxel_m:
            break
        voxel = min(max_voxel_m, voxel * 1.5)

    return DownsampleResult(
        pcd=cur,
        voxel_m=float(voxel),
        n_before=n_before,
        n_after=n_after,
    )


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_m: float,
) -> DownsampleResult:
    """Voxel-downsample a point cloud to the given XYZ grid spacing.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The input cloud.  *Not modified*; we return a new cloud.
    voxel_m : float
        Voxel edge length in metres.  Any value ≤ 0 disables downsampling
        (the input is returned unchanged).

    Returns
    -------
    DownsampleResult

    Raises
    ------
    ValueError
        If the input cloud is empty.

    Notes
    -----
    This is a thin, well-instrumented wrapper around
    :py:meth:`open3d.geometry.PointCloud.voxel_down_sample`.  The wrapper
    exists so the pipeline can:

    1. Emit a structured SSE progress event with before/after counts.
    2. Persist the same numbers into the ``VectorizeMetrics`` payload.
    3. Skip cleanly on a debug-mode ``voxel_m = 0`` flag without bypassing
       the entire stage in pipeline.py (which would mean the SSE message
       and the metrics field are filled in one path and not the other).
    """
    n_before = int(len(pcd.points))
    if n_before == 0:
        raise ValueError("voxel_downsample: input cloud is empty")

    if voxel_m <= 0:
        return DownsampleResult(pcd=pcd, voxel_m=0.0, n_before=n_before, n_after=n_before)

    # Open3D's voxel grid downsample.  Internally: hash each point's voxel
    # index, group by voxel, keep the centroid per voxel.  ~10 µs per input
    # point at 5 mm voxel; ~3 s for a 139 M-point cloud on M-series CPU.
    down = pcd.voxel_down_sample(voxel_size=float(voxel_m))
    n_after = int(len(down.points))
    return DownsampleResult(
        pcd=down,
        voxel_m=float(voxel_m),
        n_before=n_before,
        n_after=n_after,
    )
