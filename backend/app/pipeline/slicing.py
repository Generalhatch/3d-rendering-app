"""Floor detection and wall-band extraction.

Detects the dominant horizontal plane (floor) and extracts a clean
horizontal slice at chest height — approximately 0.75m–1.8m above the floor —
to feed into RANSAC plane detection for the alignment step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class FloorReference:
    plane_eq: np.ndarray        # (a, b, c, d) for ax+by+cz+d=0
    up_normal: np.ndarray       # unit vector pointing up (away from floor)
    floor_z_estimate: float     # signed distance along up_normal to the floor
    inlier_count: int
    axis_idx: int               # 2 for Z-up, 1 for Y-up


def detect_floor(scan: o3d.geometry.PointCloud) -> FloorReference:
    """Find the dominant horizontal plane (the floor).

    Handles both Z-up (standard LiDAR) and Y-up (some workflows).
    Picks the candidate with the lowest centroid so desks/ledges don't win.
    """
    scan_ds = scan.voxel_down_sample(0.1)
    pts = np.asarray(scan_ds.points)
    if len(pts) < 1000:
        # Very sparse — try the original
        pts = np.asarray(scan.points)
        scan_ds = scan

    candidates: list[tuple[float, np.ndarray, np.ndarray, int, int]] = []
    remaining = scan_ds

    for _ in range(8):
        if len(remaining.points) < 500:
            break
        try:
            plane, inliers = remaining.segment_plane(
                distance_threshold=0.03, ransac_n=3, num_iterations=1000
            )
        except Exception:
            break
        inlier_cloud = remaining.select_by_index(inliers)
        inlier_pts = np.asarray(inlier_cloud.points)
        normal = np.array(plane[:3])
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            break
        normal = normal / norm_len

        for axis_idx in [2, 1]:  # Z-up first, then Y-up
            axis = np.zeros(3)
            axis[axis_idx] = 1.0
            cos_angle = abs(float(np.dot(normal, axis)))
            if cos_angle > 0.96:  # within ~16° of horizontal
                centroid = inlier_pts.mean(axis=0)
                floor_h = float(centroid[axis_idx])
                up = axis if float(np.dot(normal, axis)) > 0 else -axis
                candidates.append((floor_h, np.array(plane), up, len(inliers), axis_idx))
                break

        remaining = remaining.select_by_index(inliers, invert=True)

    if not candidates:
        raise ValueError(
            "No horizontal floor plane detected. "
            "The scan may be too sparse, have no flat floor, or use an unusual coordinate convention."
        )

    # Pick the lowest centroid — desks/ceiling planes are above the floor
    candidates.sort(key=lambda c: c[0])
    floor_z, plane_eq, up, count, axis_idx = candidates[0]

    return FloorReference(
        plane_eq=plane_eq,
        up_normal=up,
        floor_z_estimate=floor_z,
        inlier_count=count,
        axis_idx=axis_idx,
    )


def extract_wall_band(
    scan: o3d.geometry.PointCloud,
    floor: FloorReference,
    band_low_m: float = 0.75,   # ~ankle/knee — clears furniture tops
    band_high_m: float = 1.80,  # ~shoulder height — clears ceiling fixtures
) -> o3d.geometry.PointCloud:
    """Extract the horizontal slice between band_low_m and band_high_m above the floor.

    This slice contains wall surfaces with minimal interference from:
    - floor furniture (desks, chairs, cabinets sit below ~1m)
    - ceiling fixtures (lights, HVAC diffusers sit above ~2.2m)
    - baseboards (below 0.75m)
    """
    pts = np.asarray(scan.points)
    # Height above floor = signed projection onto up_normal, shifted by floor height
    heights = pts @ floor.up_normal
    in_band = (heights >= floor.floor_z_estimate + band_low_m) & (
        heights <= floor.floor_z_estimate + band_high_m
    )
    indices = np.where(in_band)[0]
    if len(indices) < 100:
        raise ValueError(
            f"Wall band extraction yielded only {len(indices)} points. "
            f"Floor detected at z={floor.floor_z_estimate:.2f}m. "
            "Try adjusting band_low_m / band_high_m for this scan."
        )
    return scan.select_by_index(indices.tolist())
