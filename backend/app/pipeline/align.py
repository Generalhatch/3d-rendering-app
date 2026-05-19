"""Core alignment pipeline.

Pipeline:
  floor detect → wall band slice → voxel downsample → RANSAC wall planes →
  principal axes → rotation + centroid translation → ICP refinement →
  4-way ambiguity resolution → confidence scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .slicing import detect_floor, extract_wall_band, FloorReference
from .segment import extract_wall_planes, extract_wall_planes_with_fallback, WallPlane
from .axes import principal_axes_from_walls, principal_axes_from_lines


@dataclass
class AlignmentResult:
    transformation: np.ndarray          # 4x4 matrix, scan → plan frame
    residual_rmse: float                 # meters (convert to mm for display)
    confidence: float                    # 0.0 – 1.0
    inlier_ratio: float
    rotation_candidate_used: int         # which of 0/90/180/270
    iterations: int
    num_wall_planes: int
    floor: FloorReference = field(repr=False)
    wall_planes: list[WallPlane] = field(repr=False, default_factory=list)


def align_scan_to_plan(
    scan_pcd: o3d.geometry.PointCloud,
    plan_lines_2d: np.ndarray,           # (N, 2, 2): N line segments
    voxel_size: float = 0.05,
    band_low_m: float = 0.75,
    band_high_m: float = 1.80,
) -> AlignmentResult:
    """Deterministic scan-to-plan alignment.

    Same inputs always produce the same output.
    """
    # 1. Floor detection and wall band extraction
    floor = detect_floor(scan_pcd)
    wall_band = extract_wall_band(scan_pcd, floor, band_low_m, band_high_m)

    # 2. Voxel downsample for speed
    scan_ds = wall_band.voxel_down_sample(voxel_size)

    # 3. RANSAC wall plane detection on the clean slice
    wall_planes, _ = extract_wall_planes_with_fallback(
        scan_ds,
        up_axis_idx=floor.axis_idx,
        full_cloud=scan_pcd,
        floor_z=floor.floor_z_estimate,
    )

    if len(wall_planes) < 2:
        raise ValueError(
            f"Only {len(wall_planes)} wall planes detected. "
            "Cannot compute principal axes. "
            "Try adjusting band_low_m/band_high_m, or the scan may lack planar walls."
        )

    # 4. Principal axes: scan vs plan
    scan_axes = principal_axes_from_walls(wall_planes)
    plan_axes = principal_axes_from_lines(plan_lines_2d)

    # 5. Initial 2D rotation to align scan axes → plan axes
    R_init_2d = _rotation_between_axes_2d(scan_axes[0], plan_axes[0])

    # 6. Project scan wall footprint to 2D, find centroid-based translation
    scan_pts_2d = _wall_planes_to_2d(wall_planes)
    plan_pts_2d = _lines_to_2d_points(plan_lines_2d)

    scan_pts_2d_rot = scan_pts_2d @ R_init_2d.T
    t_init = plan_pts_2d.mean(axis=0) - scan_pts_2d_rot.mean(axis=0)

    # 7. Try all 4 rotational ambiguity candidates (90° increments)
    best: AlignmentResult | None = None

    for k in range(4):
        angle = k * np.pi / 2.0
        R_extra = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)],
        ])
        R_test_2d = R_extra @ R_init_2d

        # Build 4x4 transform (rotation in XY, no Z change, translation XY)
        T_4x4 = _build_4x4_from_2d(R_test_2d, t_init)

        # ICP refinement using wall band points vs plan line-sampled points
        plan_cloud = lines_to_pcd(plan_lines_2d, sample_spacing=0.05)
        scan_cloud_transformed = transform_pcd(wall_band, T_4x4)

        result = run_icp(scan_cloud_transformed, plan_cloud, T_4x4)

        if best is None or result.residual_rmse < best.residual_rmse:
            best = AlignmentResult(
                transformation=result.transformation,
                residual_rmse=result.residual_rmse,
                confidence=0.0,
                inlier_ratio=result.inlier_ratio,
                rotation_candidate_used=k,
                iterations=result.iterations,
                num_wall_planes=len(wall_planes),
                floor=floor,
                wall_planes=wall_planes,
            )

    assert best is not None

    # 8. Confidence scoring
    best.confidence = _compute_confidence(best)

    return best


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rotation_between_axes_2d(scan_ax: np.ndarray, plan_ax: np.ndarray) -> np.ndarray:
    """2D rotation matrix that maps scan_ax to plan_ax."""
    angle_scan = float(np.arctan2(scan_ax[1], scan_ax[0]))
    angle_plan = float(np.arctan2(plan_ax[1], plan_ax[0]))
    angle_diff = angle_plan - angle_scan
    c, s = np.cos(angle_diff), np.sin(angle_diff)
    return np.array([[c, -s], [s, c]])


def _wall_planes_to_2d(planes: list[WallPlane]) -> np.ndarray:
    """Stack all inlier points from wall planes, return their XY coords."""
    all_pts = [p.inlier_points[:, :2] for p in planes]
    return np.vstack(all_pts)


def _lines_to_2d_points(lines: np.ndarray) -> np.ndarray:
    """Flatten line segment endpoints to (N*2, 2) point array."""
    return lines.reshape(-1, 2)


def _build_4x4_from_2d(R_2d: np.ndarray, t_2d: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 2D rotation and translation."""
    T = np.eye(4)
    T[:2, :2] = R_2d
    T[0, 3] = t_2d[0]
    T[1, 3] = t_2d[1]
    return T


def lines_to_pcd(lines: np.ndarray, sample_spacing: float = 0.05) -> o3d.geometry.PointCloud:
    """Sample points along plan line segments to make a 2D target point cloud for ICP."""
    points = []
    for seg in lines:
        start = np.array([seg[0][0], seg[0][1], 0.0])
        end   = np.array([seg[1][0], seg[1][1], 0.0])
        length = float(np.linalg.norm(end - start))
        if length < 1e-6:
            points.append(start)
            continue
        n = max(2, int(length / sample_spacing))
        for i in range(n + 1):
            points.append(start + (end - start) * i / n)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    return pcd


def transform_pcd(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    """Apply 4x4 transform to a point cloud."""
    transformed = o3d.geometry.PointCloud(pcd)
    transformed.transform(T)
    return transformed


@dataclass
class _ICPResult:
    transformation: np.ndarray
    residual_rmse: float
    inlier_ratio: float
    iterations: int


def run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_correspondence_dist: float = 0.5,
    max_iter: int = 100,
) -> _ICPResult:
    """Run Open3D ICP and return result."""
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_dist,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )
    n_correspondences = len(result.correspondence_set)
    inlier_ratio = n_correspondences / max(len(source.points), 1)
    return _ICPResult(
        transformation=np.array(result.transformation),
        residual_rmse=float(result.inlier_rmse),
        inlier_ratio=float(inlier_ratio),
        iterations=max_iter,
    )


def _compute_confidence(result: AlignmentResult) -> float:
    """Heuristic confidence in [0, 1].

    Components:
      - rmse_score: 0mm → 1.0, 50mm → 0.0 (linear)
      - inlier_ratio: fraction of source points with a correspondence
      - support_score: more wall planes = more geometric constraints
    """
    rmse_mm = result.residual_rmse * 1000.0
    rmse_score = max(0.0, 1.0 - rmse_mm / 50.0)
    inlier_score = min(1.0, result.inlier_ratio)
    support_score = min(1.0, result.num_wall_planes / 12.0)
    return 0.5 * rmse_score + 0.3 * inlier_score + 0.2 * support_score
