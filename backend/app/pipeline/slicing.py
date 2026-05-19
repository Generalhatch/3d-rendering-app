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


def _collect_floor_candidates(
    scan: o3d.geometry.PointCloud,
) -> tuple[list[FloorReference], np.ndarray]:
    """Run iterative RANSAC to find all horizontal planes in the scan.

    Returns (candidates_sorted_by_wall_score, pts_array).  The first entry
    in the list is the best floor candidate (highest wall-content score).

    Internal helper shared by detect_floor() and detect_all_floors().
    """
    scan_ds = scan.voxel_down_sample(0.1)
    pts = np.asarray(scan_ds.points)
    if len(pts) < 1000:
        pts = np.asarray(scan.points)
        scan_ds = scan

    raw: list[tuple[float, np.ndarray, np.ndarray, int, int]] = []
    remaining = scan_ds

    for _ in range(12):
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
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        normal = normal / norm_len

        for axis_idx in [2, 1]:  # Z-up first, then Y-up
            axis = np.zeros(3)
            axis[axis_idx] = 1.0
            cos_angle = abs(float(np.dot(normal, axis)))
            if cos_angle > 0.96:  # within ~16° of horizontal
                centroid = inlier_pts.mean(axis=0)
                floor_h = float(centroid[axis_idx])
                up = axis if float(np.dot(normal, axis)) > 0 else -axis
                raw.append((floor_h, np.array(plane), up, len(inliers), axis_idx))
                break

        remaining = remaining.select_by_index(inliers, invert=True)

    if not raw:
        # Hard fallback: histogram peak of the vertical axis
        for axis_idx in [2, 1]:
            z_col = pts[:, axis_idx]
            hist, edges = np.histogram(z_col, bins=200)
            floor_h = float(edges[int(np.argmax(hist))])
            axis = np.zeros(3)
            axis[axis_idx] = 1.0
            dummy_plane = np.array([axis[0], axis[1], axis[2], -floor_h])
            raw.append((floor_h, dummy_plane, axis, 1, axis_idx))
            break

    def _wall_score(floor_h: float, ax: int) -> int:
        z = pts[:, ax]
        return int(((z > floor_h + 0.30) & (z < floor_h + 3.00)).sum())

    refs = [
        FloorReference(
            plane_eq=plane_eq,
            up_normal=up,
            floor_z_estimate=floor_h,
            inlier_count=count,
            axis_idx=axis_idx,
        )
        for floor_h, plane_eq, up, count, axis_idx in raw
    ]
    refs.sort(key=lambda r: -_wall_score(r.floor_z_estimate, r.axis_idx))
    return refs, pts


def detect_floor(scan: o3d.geometry.PointCloud) -> FloorReference:
    """Find the dominant horizontal plane (the floor).

    Handles both Z-up (standard LiDAR) and Y-up (some workflows).

    Selection strategy: among all detected horizontal planes, pick the one whose
    height maximises the number of *wall-type* points directly above it (0.3–3 m
    band).  This reliably selects the indoor building floor over outdoor terrain,
    sub-floors, or parking structures — even when those lower surfaces have large
    RANSAC inlier counts.
    """
    candidates, _ = _collect_floor_candidates(scan)
    return candidates[0]


def detect_all_floors(scan: o3d.geometry.PointCloud) -> list[FloorReference]:
    """Return all detected horizontal floor planes sorted by wall-content score.

    The first entry matches what detect_floor() returns.  Subsequent entries
    represent additional levels (upper floors, mezzanines, parking decks, etc.)
    that were found by the iterative RANSAC pass.

    Each ``FloorReference.floor_z_estimate`` gives the elevation of that level
    in the scan's local coordinate frame.

    Useful for multi-floor buildings: call this once, persist the result, then
    let the user select which level to generate rooms for.
    """
    candidates, _ = _collect_floor_candidates(scan)
    return candidates


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
