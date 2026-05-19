"""Multi-scan merging: combine N LAZ/LAS/PLY/E57 scan files into one unified point cloud.

Strategy (in order):
  1. Load all scans.
  2. Detect if they share a common coordinate frame (georeferenced / pre-registered):
     - If their bounding boxes have meaningful overlap → they're already aligned → just concatenate.
     - If bounding boxes don't overlap → attempt pairwise ICP registration before merging.
  3. Voxel-downsample the merged cloud for a clean, manageable output.

Professional LiDAR workflows (Leica, FARO, NavVis, Matterport Pro3, etc.) typically produce
per-room scans that are already registered to a common site coordinate frame. In that case
step 2 is a simple concatenation and the whole merge takes a few seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d

from .ingest import load_point_cloud


@dataclass
class MergeResult:
    merged: o3d.geometry.PointCloud
    num_scans: int
    total_points_before: int
    total_points_after: int
    strategy: str   # "concatenate" | "icp_registered"


def merge_scans(
    scan_paths: list[Path],
    voxel_size: float = 0.03,          # 3cm voxel for merged cloud (dense but manageable)
    overlap_threshold: float = 0.10,   # fraction of bbox overlap to call them "pre-registered"
    progress_cb: Callable[[str, float], None] | None = None,
) -> MergeResult:
    """Load and merge multiple scan files into a single unified point cloud.

    Args:
        scan_paths: List of paths to individual scan files.
        voxel_size: Voxel size for downsampling the merged result.
        overlap_threshold: Min bbox overlap fraction to treat scans as pre-registered.
        progress_cb: Optional callback(message, 0.0–1.0) for progress reporting.
    """
    if not scan_paths:
        raise ValueError("No scan files provided")

    if len(scan_paths) == 1:
        pcd = load_point_cloud(scan_paths[0])
        pts_before = len(pcd.points)
        merged = pcd.voxel_down_sample(voxel_size)
        return MergeResult(
            merged=merged,
            num_scans=1,
            total_points_before=pts_before,
            total_points_after=len(merged.points),
            strategy="single",
        )

    def _emit(msg: str, p: float):
        if progress_cb:
            progress_cb(msg, p)

    # ── Step 1: Load all scans ────────────────────────────────────────────────
    clouds: list[o3d.geometry.PointCloud] = []
    total_before = 0
    for i, path in enumerate(scan_paths):
        _emit(f"Loading scan {i + 1}/{len(scan_paths)}: {path.name}…", i / len(scan_paths) * 0.5)
        pcd = load_point_cloud(path)
        total_before += len(pcd.points)
        clouds.append(pcd)

    # ── Step 2: Decide strategy ───────────────────────────────────────────────
    strategy = "concatenate"
    if _scans_appear_preregistered(clouds, overlap_threshold):
        _emit("Scans share coordinate frame — merging by concatenation…", 0.55)
        strategy = "concatenate"
    else:
        _emit("Scans appear unregistered — running pairwise ICP registration…", 0.55)
        strategy = "icp_registered"
        clouds = _register_pairwise(clouds, voxel_size, _emit)

    # ── Step 3: Concatenate ───────────────────────────────────────────────────
    _emit("Concatenating point clouds…", 0.80)
    merged = clouds[0]
    for pcd in clouds[1:]:
        merged = merged + pcd

    # ── Step 4: Voxel downsample ──────────────────────────────────────────────
    _emit("Downsampling merged cloud…", 0.90)
    merged = merged.voxel_down_sample(voxel_size)

    # Estimate normals (helps ICP in the alignment step)
    merged.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
    )

    _emit(f"Merge complete — {len(merged.points):,} points from {len(scan_paths)} scans", 1.0)

    return MergeResult(
        merged=merged,
        num_scans=len(scan_paths),
        total_points_before=total_before,
        total_points_after=len(merged.points),
        strategy=strategy,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_bbox(pcd: o3d.geometry.PointCloud) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pcd.points)
    return pts.min(axis=0), pts.max(axis=0)


def _bbox_overlap_fraction(
    min1: np.ndarray, max1: np.ndarray,
    min2: np.ndarray, max2: np.ndarray,
) -> float:
    """Fraction of the smaller bbox's volume that overlaps with the larger."""
    overlap_min = np.maximum(min1, min2)
    overlap_max = np.minimum(max1, max2)
    overlap_dims = np.maximum(0, overlap_max - overlap_min)
    overlap_vol = float(np.prod(overlap_dims))
    vol1 = float(np.prod(np.maximum(0, max1 - min1)))
    vol2 = float(np.prod(np.maximum(0, max2 - min2)))
    smaller = min(vol1, vol2)
    return overlap_vol / smaller if smaller > 1e-9 else 0.0


def _scans_appear_preregistered(
    clouds: list[o3d.geometry.PointCloud],
    threshold: float,
) -> bool:
    """Return True if most scan pairs have meaningful bounding-box overlap.

    Pre-registered scans from the same floor will overlap substantially.
    Scans in completely separate coordinate frames won't overlap at all.
    """
    if len(clouds) < 2:
        return True
    bboxes = [_get_bbox(c) for c in clouds]
    overlapping = 0
    pairs = 0
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            pairs += 1
            frac = _bbox_overlap_fraction(bboxes[i][0], bboxes[i][1], bboxes[j][0], bboxes[j][1])
            if frac >= threshold:
                overlapping += 1
    # Treat as pre-registered if >50% of pairs overlap
    return pairs > 0 and (overlapping / pairs) >= 0.5


def _register_pairwise(
    clouds: list[o3d.geometry.PointCloud],
    voxel_size: float,
    emit: Callable[[str, float], None],
) -> list[o3d.geometry.PointCloud]:
    """Register each scan to the first scan (reference) using FPFH + ICP.

    This is a "star" topology: every scan registers to cloud[0].
    Good enough for MVP; global pose-graph optimization is v2.
    """
    registered = [clouds[0]]
    reference = clouds[0].voxel_down_sample(voxel_size * 2)
    reference.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30)
    )

    for i, pcd in enumerate(clouds[1:], start=1):
        emit(f"Registering scan {i + 1}/{len(clouds)} to reference…", 0.55 + i / len(clouds) * 0.20)

        ds = pcd.voxel_down_sample(voxel_size * 2)
        ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30)
        )

        # Global registration via FPFH features + RANSAC
        T_init = _global_registration(reference, ds, voxel_size * 2)

        # ICP refinement
        result = o3d.pipelines.registration.registration_icp(
            ds, reference,
            voxel_size * 3,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        transformed = o3d.geometry.PointCloud(pcd)
        transformed.transform(result.transformation)
        registered.append(transformed)

    return registered


def _global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
) -> np.ndarray:
    """Fast global registration via FPFH features + RANSAC.

    Returns the initial 4x4 transform.
    """
    radius_feature = voxel_size * 5

    def _compute_fpfh(pcd: o3d.geometry.PointCloud):
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )

    src_fpfh = _compute_fpfh(source)
    tgt_fpfh = _compute_fpfh(target)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 0.999),
    )
    return np.array(result.transformation)
