"""RANSAC-based wall plane segmentation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d


@dataclass
class WallPlane:
    id: str
    plane_eq: np.ndarray        # (a, b, c, d)
    inlier_points: np.ndarray   # (N, 3)


def _ransac_walls(
    pcd: o3d.geometry.PointCloud,
    max_planes: int = 40,
    min_inliers: int = 80,
    distance_threshold: float = 0.05,
    up_axis_idx: int = 2,
    normal_up_max: float = 0.30,
    min_height_range: float = 0.15,
    id_prefix: str = "wall",
) -> list[WallPlane]:
    """Inner RANSAC loop.  Operates on a pre-prepared point cloud."""
    remaining = pcd
    planes: list[WallPlane] = []
    counter = 0

    for _ in range(max_planes):
        if len(remaining.points) < min_inliers:
            break
        try:
            plane_model, inliers = remaining.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000,
            )
        except Exception:
            break

        if len(inliers) < min_inliers:
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        norm_len = float(np.linalg.norm(normal))
        if norm_len < 1e-9:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        normal = normal / norm_len

        if abs(normal[up_axis_idx]) > normal_up_max:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        inlier_cloud = remaining.select_by_index(inliers)
        inlier_pts = np.asarray(inlier_cloud.points)

        up_range = (inlier_pts[:, up_axis_idx].max()
                    - inlier_pts[:, up_axis_idx].min())
        if up_range < min_height_range:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        counter += 1
        planes.append(WallPlane(
            id=f"{id_prefix}-{counter:03d}",
            plane_eq=np.array([a, b, c, d]) / norm_len,
            inlier_points=inlier_pts,
        ))
        remaining = remaining.select_by_index(inliers, invert=True)

    return planes


def extract_wall_planes(
    wall_band: o3d.geometry.PointCloud,
    voxel_size: float = 0.05,
    max_planes: int = 40,
    min_inliers: int = 80,
    distance_threshold: float = 0.05,
    up_axis_idx: int = 2,
    normal_up_max: float = 0.30,
    min_height_range: float = 0.15,
) -> list[WallPlane]:
    """Extract vertical (wall) planes from a wall-band slice.

    Uses ``up_axis_idx`` from the floor-detection result so Y-up scans work
    correctly.
    """
    ds = wall_band.voxel_down_sample(voxel_size)
    return _ransac_walls(
        ds,
        max_planes=max_planes,
        min_inliers=min_inliers,
        distance_threshold=distance_threshold,
        up_axis_idx=up_axis_idx,
        normal_up_max=normal_up_max,
        min_height_range=min_height_range,
    )


def extract_wall_planes_from_full_cloud(
    scan: o3d.geometry.PointCloud,
    up_axis_idx: int = 2,
    voxel_size: float = 0.05,
    normal_up_max: float = 0.35,
    min_inliers: int = 50,
    distance_threshold: float = 0.06,
    max_planes: int = 40,
    min_height_range: float = 0.15,
    floor_z: float | None = None,
) -> list[WallPlane]:
    """Normal-based wall extraction — more robust for open-plan spaces.

    Instead of slicing at a fixed height (which fails when furniture dominates
    that band), this estimates per-point normals on the full downsampled cloud
    and filters to points whose surface is roughly vertical (wall-like).
    Works reliably even when the wall-band contains mostly horizontal surfaces
    (reception desks, counters, open shelving, glass walls, etc.).
    """
    ds = scan.voxel_down_sample(voxel_size)

    ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=30)
    )

    normals = np.asarray(ds.normals)
    pts = np.asarray(ds.points)

    # Points on vertical surfaces have a normal with small up-axis component
    vert_mask = np.abs(normals[:, up_axis_idx]) < normal_up_max

    # Exclude points very close to the floor (baseboards, debris)
    if floor_z is not None:
        vert_mask &= pts[:, up_axis_idx] > floor_z + 0.20

    if vert_mask.sum() < min_inliers:
        return []

    wall_cloud = ds.select_by_index(np.where(vert_mask)[0])

    return _ransac_walls(
        wall_cloud,
        max_planes=max_planes,
        min_inliers=min_inliers,
        distance_threshold=distance_threshold,
        up_axis_idx=up_axis_idx,
        normal_up_max=normal_up_max,
        min_height_range=min_height_range,
        id_prefix="nwall",
    )


def extract_wall_planes_with_fallback(
    wall_band: o3d.geometry.PointCloud,
    up_axis_idx: int = 2,
    full_cloud: o3d.geometry.PointCloud | None = None,
    floor_z: float | None = None,
) -> tuple[list[WallPlane], str]:
    """Try progressively looser wall-band settings, then normal-based extraction.

    The normal-based fallback is essential for open-plan spaces where the
    0.75–1.80m band is dominated by horizontal furniture surfaces rather than
    wall planes.

    Returns ``(planes, description_of_method_used)``.
    """
    attempts = [
        # (voxel, min_inliers, dist_thresh, normal_up_max, min_height)
        (0.05, 80,  0.05, 0.30, 0.15),   # default
        (0.08, 50,  0.08, 0.40, 0.10),   # looser
        (0.10, 30,  0.10, 0.50, 0.08),   # very loose — tilted or low-density scans
    ]
    for voxel, min_inl, dist, nup, mh in attempts:
        planes = extract_wall_planes(
            wall_band,
            voxel_size=voxel,
            min_inliers=min_inl,
            distance_threshold=dist,
            up_axis_idx=up_axis_idx,
            normal_up_max=nup,
            min_height_range=mh,
        )
        if planes:
            return planes, f"wall-band voxel={voxel}m dist={dist}m nup={nup}"

    # Wall-band failed (open-plan / glass walls / furniture-dominated band).
    # Fall back to normal-based extraction on the full cloud.
    if full_cloud is not None:
        for nup, vox in [(0.35, 0.05), (0.45, 0.08), (0.55, 0.10)]:
            planes = extract_wall_planes_from_full_cloud(
                full_cloud,
                up_axis_idx=up_axis_idx,
                voxel_size=vox,
                normal_up_max=nup,
                min_inliers=50,
                distance_threshold=0.06,
                floor_z=floor_z,
            )
            if planes:
                return planes, f"normal-based full-cloud nup={nup} vox={vox}m"

    return [], "all attempts failed"
