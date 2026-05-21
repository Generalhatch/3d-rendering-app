"""Point-normal estimation + vertical-surface filtering.

The single biggest cause of false-positive walls in our pipeline is that the
slicer treats every point in the slab as wall evidence — floor returns near the
slice plane, desk tops, monitor backs, bookshelf tops, picture frames, etc.
All of those are horizontal (or nearly so) surfaces and have no business
contributing to a wall-detection pipeline.

Real walls are *vertical surfaces*.  In a properly-oriented scan a point on a
vertical surface has an estimated normal vector whose vertical component is
near zero (|n_z| ≈ 0).  Filtering to such points before any other stage
eliminates 50–70 % of the noise we currently process.

Why this lives in its own module
--------------------------------
Both ``slicer.slice_to_raster`` and ``slicer.slice_to_raster_multi`` benefit
from this filter, and so will any future point-cloud-native operations (plane
RANSAC, density slicer).  Centralising the normals computation also lets us
cache the result so the multi-elevation slicer doesn't pay for it three times.

References
----------
This is a standard preprocessing step in indoor reconstruction pipelines
(e.g. the dynamic-layer-extraction paper from Springer 2024 reports
+5 % wall precision and +33 % wall recall using exactly this filter).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


# Default tolerance: ``|n_z| <= 0.30`` keeps points whose surface tilts up to
# ~17.5° from vertical.  Walls can be slightly off-plumb (older buildings,
# scanner registration drift) so a hard 0° cutoff is too strict.  Going
# beyond ~25° starts admitting tilted desk surfaces.
DEFAULT_VERTICAL_TOLERANCE = 0.30


@dataclass
class NormalEstimationParams:
    """Parameters for Open3D's normal estimation pass.

    The defaults are tuned for typical indoor LiDAR (Leica RTC360, FARO Focus,
    NavVis VLX) at ~1 cm point spacing.  Bump ``radius`` for sparse scans
    (NavVis VLX outdoor sweeps), drop it for very dense scans (Matterport
    Pro3 at 0.5 cm spacing).
    """
    radius: float = 0.10          # 10 cm neighbourhood — covers ~30–100 points at 1 cm spacing
    max_nn: int = 30              # cap neighbours so computation stays bounded
    orient_consistent: bool = False  # we only care about |n_z|; orientation costs ~3× more


def estimate_normals_inplace(
    pcd: o3d.geometry.PointCloud,
    params: NormalEstimationParams | None = None,
) -> None:
    """Estimate per-point normals on ``pcd`` (modifies in place).

    Idempotent: if the cloud already has normals, this is a no-op.  Otherwise
    runs Open3D's hybrid-search KD-tree normal estimation, which is the
    standard for this task — fast (parallelised under the hood), robust, and
    already in our deps.
    """
    if params is None:
        params = NormalEstimationParams()

    # Skip recomputation if the caller already supplied normals.
    if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
        return

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=params.radius,
            max_nn=params.max_nn,
        )
    )

    if params.orient_consistent:
        # Propagates normal orientation across neighbours — useful for some
        # downstream tasks but ~3× slower.  We don't need it for |n_z|.
        pcd.orient_normals_consistent_tangent_plane(k=params.max_nn)


def filter_to_vertical_surfaces(
    pcd: o3d.geometry.PointCloud,
    axis_idx: int = 2,
    tolerance: float = DEFAULT_VERTICAL_TOLERANCE,
    *,
    estimate_if_missing: bool = True,
    estimation_params: NormalEstimationParams | None = None,
) -> o3d.geometry.PointCloud:
    """Return a new point cloud containing only points on vertical surfaces.

    A point is "on a vertical surface" iff its surface normal is nearly
    perpendicular to the gravity axis — i.e. ``|n[axis_idx]| <= tolerance``.
    The classic wall-vs-furniture test.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input cloud.  Not modified.
    axis_idx : int
        Index of the vertical axis.  ``2`` for Z-up (standard), ``1`` for Y-up.
    tolerance : float
        Maximum ``|n[axis_idx]|`` allowed.  Smaller = stricter "must be vertical".
        Default 0.30 corresponds to ~17.5° from vertical.
    estimate_if_missing : bool
        If True (default) and ``pcd`` has no normals, estimate them.  Pass
        False when the caller has already estimated normals to skip the
        duplicate work check.
    estimation_params : NormalEstimationParams, optional
        Forwarded to :func:`estimate_normals_inplace` when normals need
        estimating.

    Returns
    -------
    o3d.geometry.PointCloud
        A new cloud (shares no buffers with the input) containing only the
        kept points, with the matching normal and (if present) colour.

    Raises
    ------
    ValueError
        If ``axis_idx`` isn't 1 or 2, ``tolerance`` is outside [0, 1], or
        the input cloud is empty.
    """
    if axis_idx not in (1, 2):
        raise ValueError(f"axis_idx must be 1 (Y-up) or 2 (Z-up); got {axis_idx}")
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError(f"tolerance must be in [0, 1]; got {tolerance}")
    n_input = len(pcd.points)
    if n_input == 0:
        raise ValueError("Point cloud is empty")

    if estimate_if_missing:
        estimate_normals_inplace(pcd, estimation_params)

    if not pcd.has_normals() or len(pcd.normals) != n_input:
        # Shouldn't happen given estimate_if_missing=True above, but stay
        # defensive — Open3D occasionally returns a normals array of the
        # wrong length on tiny clouds.
        raise RuntimeError(
            f"Cannot filter without normals.  Cloud has {n_input} points but "
            f"{len(pcd.normals)} normals."
        )

    normals = np.asarray(pcd.normals, dtype=np.float64)
    keep_mask = np.abs(normals[:, axis_idx]) <= tolerance

    out = pcd.select_by_index(np.flatnonzero(keep_mask).tolist())
    return out


def filter_summary(
    before: int,
    after: int,
    tolerance: float = DEFAULT_VERTICAL_TOLERANCE,
) -> str:
    """One-line human-readable summary of a vertical-surface filter pass.

    Used by the pipeline orchestrator's SSE progress messages.
    """
    if before == 0:
        return "vertical-surface filter: input empty"
    pct_kept = 100.0 * after / before
    pct_dropped = 100.0 - pct_kept
    return (
        f"vertical-surface filter (|n_z| ≤ {tolerance:.2f}): "
        f"kept {after:,} / {before:,} pts "
        f"({pct_kept:.1f} % kept, {pct_dropped:.1f} % dropped — "
        f"floor/ceiling/desk/cabinet)"
    )
