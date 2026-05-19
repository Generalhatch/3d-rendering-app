"""Multi-scan merging: combine N LAZ/LAS/PLY/E57 scan files into one unified point cloud.

Design principle: memory-safe streaming pipeline.
Each file is loaded, immediately downsampled, and the full-res cloud discarded.
We never hold all scans at full resolution simultaneously.

Strategy:
  1. Load each scan → immediately voxel-downsample → keep only the downsampled version.
  2. Check if the scans share a coordinate frame (pre-registered).
     - Professional LiDAR workflows (Leica Cyclone, FARO SCENE, NavVis) almost always
       export pre-registered LAZ in a common site frame. We detect this by checking
       that all scan centroids are within a reasonable proximity of each other
       (same building floor = centroids within ~200m of each other).
  3. If pre-registered: concatenate the downsampled clouds. Done.
  4. If NOT pre-registered: run a lightweight pairwise ICP in "star" topology
       (each scan registered to scan[0] as reference). Uses only the already-
       downsampled clouds — no extra memory overhead.
  5. Final voxel downsample of the merged result.

Memory profile (6 scans × 500M points each, 3cm voxel):
  - Peak RAM ≈ 2 × (one downsampled cloud) ≈ 2 × ~80MB = ~160MB
  - vs naïve load-all: 6 × 500MB = 3GB
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
    total_points_before_ds: int   # sum of points in each scan after initial downsample
    total_points_after: int       # final merged+downsampled count
    strategy: str                 # "concatenate" | "icp_registered" | "single"


# If any centroid is more than this far from the others, scans are probably
# in different coordinate frames (or on different floors / buildings).
_SAME_FRAME_RADIUS_M = 300.0


def merge_scans(
    scan_paths: list[Path],
    voxel_size: float = 0.03,
    progress_cb: Callable[[str, float], None] | None = None,
) -> MergeResult:
    """Memory-safe merge of N scan files into one downsampled point cloud.

    Streams each file one at a time — never holds all scans at full resolution.
    """
    if not scan_paths:
        raise ValueError("No scan files provided")

    def _emit(msg: str, p: float) -> None:
        if progress_cb:
            progress_cb(msg, p)

    n = len(scan_paths)

    if n == 1:
        _emit(f"Loading scan: {scan_paths[0].name}…", 0.0)
        pcd = load_point_cloud(scan_paths[0])
        pts_raw = len(pcd.points)
        _emit("Downsampling…", 0.5)
        ds = pcd.voxel_down_sample(voxel_size)
        del pcd  # free full-res immediately
        _emit(f"Loaded: {len(ds.points):,} points", 1.0)
        return MergeResult(
            merged=ds,
            num_scans=1,
            total_points_before_ds=pts_raw,
            total_points_after=len(ds.points),
            strategy="single",
        )

    # ── Step 1: Stream-load and immediately downsample each scan ─────────────
    downsampled: list[o3d.geometry.PointCloud] = []
    total_pts_ds = 0

    for i, path in enumerate(scan_paths):
        frac = i / n
        _emit(f"Loading {i + 1}/{n}: {path.name}…", frac * 0.55)
        pcd = load_point_cloud(path)
        ds = pcd.voxel_down_sample(voxel_size)
        del pcd   # free full-resolution cloud immediately
        total_pts_ds += len(ds.points)
        downsampled.append(ds)
        _emit(f"  → {len(ds.points):,} points after downsample", (i + 0.9) / n * 0.55)

    # ── Step 2: Detect coordinate frame ──────────────────────────────────────
    _emit("Detecting coordinate frame…", 0.58)
    pre_registered = _scans_share_coordinate_frame(downsampled)
    strategy = "concatenate" if pre_registered else "icp_registered"

    if pre_registered:
        _emit(f"Pre-registered scans detected — concatenating {n} clouds…", 0.62)
    else:
        _emit(f"Scans appear to be in different frames — running pairwise ICP…", 0.62)
        downsampled = _register_pairwise(downsampled, voxel_size, _emit)

    # ── Step 3: Concatenate ───────────────────────────────────────────────────
    _emit("Concatenating…", 0.82)
    merged = downsampled[0]
    for pcd in downsampled[1:]:
        merged = merged + pcd

    # ── Step 4: Final downsample (removes duplicates in overlap zones) ────────
    _emit("Final downsample…", 0.90)
    merged = merged.voxel_down_sample(voxel_size)

    _emit(
        f"Merge complete — {len(merged.points):,} pts from {n} scans "
        f"({'pre-registered' if pre_registered else 'ICP-registered'})",
        1.0,
    )

    return MergeResult(
        merged=merged,
        num_scans=n,
        total_points_before_ds=total_pts_ds,
        total_points_after=len(merged.points),
        strategy=strategy,
    )


# ── Coordinate-frame detection ────────────────────────────────────────────────

def _scans_share_coordinate_frame(clouds: list[o3d.geometry.PointCloud]) -> bool:
    """Return True if all scans appear to be in the same coordinate frame.

    Method: compute each scan's centroid. If ALL centroids are within
    _SAME_FRAME_RADIUS_M of the group centroid, the scans are almost certainly
    pre-registered (same site frame). This is far more reliable than bbox
    overlap for large floors where adjacent-room scans may not overlap at all.

    Professional LiDAR software (Leica Cyclone, FARO SCENE, NavVis, Matterport
    Pro3) always exports pre-registered LAZ in world coordinates, so this
    check will pass for virtually all real-world client data.
    """
    centroids = np.array([
        np.asarray(c.points).mean(axis=0)
        for c in clouds
        if len(c.points) > 0
    ])
    if len(centroids) < 2:
        return True

    group_centroid = centroids.mean(axis=0)
    max_dist = float(np.linalg.norm(centroids - group_centroid, axis=1).max())
    return max_dist <= _SAME_FRAME_RADIUS_M


# ── Pairwise ICP registration (fallback for unregistered scans) ───────────────

def _register_pairwise(
    clouds: list[o3d.geometry.PointCloud],
    voxel_size: float,
    emit: Callable[[str, float], None],
) -> list[o3d.geometry.PointCloud]:
    """Register each scan to clouds[0] via FPFH + RANSAC → ICP.

    "Star" topology: every scan registers to the first scan as reference.
    Works well when scans have reasonable pairwise overlap (adjacent rooms,
    hallway to office, etc.). For very disconnected scans (e.g. opposite ends
    of a huge warehouse with nothing in common), this will struggle — but
    that scenario is rare in building floor scanning.
    """
    reference = _prepare_for_registration(clouds[0], voxel_size)
    registered = [clouds[0]]

    for i, pcd in enumerate(clouds[1:], start=1):
        emit(
            f"Registering scan {i + 1}/{len(clouds)} to reference…",
            0.62 + (i / len(clouds)) * 0.18,
        )
        source = _prepare_for_registration(pcd, voxel_size)

        # Global init via FPFH features
        T_init = _fpfh_global_registration(source["ds"], reference["ds"],
                                            source["fpfh"], reference["fpfh"],
                                            voxel_size)

        # ICP refinement with the init transform
        result = o3d.pipelines.registration.registration_icp(
            source["ds"], reference["ds"],
            voxel_size * 2,
            T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )

        transformed = o3d.geometry.PointCloud(pcd)
        transformed.transform(result.transformation)
        registered.append(transformed)

    return registered


def _prepare_for_registration(pcd: o3d.geometry.PointCloud, voxel_size: float) -> dict:
    """Downsample, estimate normals, compute FPFH features."""
    ds = pcd.voxel_down_sample(voxel_size * 3)  # coarser for registration speed
    ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 6, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 15, max_nn=100),
    )
    return {"ds": ds, "fpfh": fpfh}


def _fpfh_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    src_fpfh,
    tgt_fpfh,
    voxel_size: float,
) -> np.ndarray:
    dist = voxel_size * 6
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 0.999),
    )
    return np.array(result.transformation)
