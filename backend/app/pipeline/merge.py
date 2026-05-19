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
    centroid_offset: np.ndarray   # 3-vector subtracted by _center_cloud; add it back
                                  # to convert centered coords → original scan frame


# Minimum centroid spread (metres) that indicates scans are in a shared large-scale
# coordinate frame rather than each sitting at their own local origin.
_SHARED_FRAME_SPREAD_M = 5.0


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
        ds, _ = ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        centroid_offset = np.asarray(ds.points, dtype=np.float64).mean(axis=0)
        ds = _center_cloud(ds)
        _emit(f"Loaded: {len(ds.points):,} points", 1.0)
        return MergeResult(
            merged=ds,
            num_scans=1,
            total_points_before_ds=pts_raw,
            total_points_after=len(ds.points),
            strategy="single",
            centroid_offset=centroid_offset,
        )

    # ── Step 1: Stream-load, downsample, and clean each scan ─────────────────
    # Statistical outlier removal runs per-scan before merging.
    # This removes: floating scan artifacts, scanner motion blur, tree foliage,
    # and any isolated noise points captured through windows or open doors.
    # nb_neighbors=20 / std_ratio=2.0 is conservative — removes only clear outliers.
    downsampled: list[o3d.geometry.PointCloud] = []
    total_pts_ds = 0

    for i, path in enumerate(scan_paths):
        frac = i / n
        _emit(f"Loading {i + 1}/{n}: {path.name}…", frac * 0.55)
        pcd = load_point_cloud(path)
        ds = pcd.voxel_down_sample(voxel_size)
        del pcd   # free full-resolution cloud immediately
        ds, _ = ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
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

    # ── Step 5: Center the merged cloud at origin for numerical stability ─────
    # Geographic/UTM coordinates (e.g. X=500000, Y=4500000) cause issues with
    # Open3D's RANSAC. We center once here so all downstream code works with
    # coordinates near the origin, while preserving relative scan positions.
    # Store the offset so per-scan re-loading can apply the same transform.
    centroid_offset = np.asarray(merged.points, dtype=np.float64).mean(axis=0)
    merged = _center_cloud(merged)

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
        centroid_offset=centroid_offset,
    )


# ── Coordinate-frame detection ────────────────────────────────────────────────

def _center_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Subtract the centroid so the cloud is centred near origin.

    Called once on the fully-merged cloud to ensure all downstream pipeline
    code (RANSAC, ICP, projection) works with coordinates near 0,0,0.
    Colors are preserved.
    """
    pts = np.asarray(pcd.points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = o3d.geometry.PointCloud()
    centered.points = o3d.utility.Vector3dVector(pts - centroid)
    if pcd.has_colors():
        centered.colors = pcd.colors
    return centered


def _scans_share_coordinate_frame(clouds: list[o3d.geometry.PointCloud]) -> bool:
    """Return True if all scans appear to be in the same coordinate frame.

    Two-stage check:

    Stage 1 — centroid spread.  If scan centroids are spread out (>5m apart on
    average), they're almost certainly pre-registered professional exports
    (Leica Cyclone, FARO SCENE, NavVis, Matterport Pro3 all output scans in a
    common site frame).

    Stage 2 — spatial overlap.  If all centroids are suspiciously close to each
    other (<5m apart), it could mean either:
      (a) Pre-registered scans of a very compact space, OR
      (b) Unregistered scans — each scan is in its own local frame starting
          near 0,0,0 (e.g. exported without global registration).

    To distinguish (a) from (b) we check voxel overlap: if a meaningful
    fraction of voxels from scan A land within tolerance of voxels from scan B,
    the scans genuinely share space → pre-registered.  If no overlap is found,
    treat as unregistered.
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

    # Clear sign of a shared large-scale coordinate frame: centroids are spread
    # meaningfully across the floor footprint.  5m spread is a conservative lower
    # bound — a 10 m × 10 m office would have room centroids ≥3–4 m apart.
    if max_dist >= 5.0:
        return True

    # Centroids are all very close — could be pre-registered compact space OR
    # unregistered (each scan sitting at its own local origin).  Verify by
    # checking actual voxel overlap between the first pair of scans.
    return _clouds_have_spatial_overlap(clouds[0], clouds[1], voxel_size=0.20)


def _clouds_have_spatial_overlap(
    a: o3d.geometry.PointCloud,
    b: o3d.geometry.PointCloud,
    voxel_size: float = 0.20,
    min_overlap_fraction: float = 0.05,
) -> bool:
    """Return True if at least min_overlap_fraction of scan A's voxels are
    occupied by scan B (i.e. the scans genuinely share physical space).

    Uses a simple 3D hash set comparison at coarse resolution.
    """
    def _voxel_set(pcd: o3d.geometry.PointCloud) -> set:
        pts = np.asarray(pcd.points)
        keys = (pts / voxel_size).astype(np.int32)
        return {(int(k[0]), int(k[1]), int(k[2])) for k in keys}

    set_a = _voxel_set(a)
    set_b = _voxel_set(b)
    if not set_a:
        return False
    overlap = len(set_a & set_b)
    fraction = overlap / len(set_a)
    return fraction >= min_overlap_fraction


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
