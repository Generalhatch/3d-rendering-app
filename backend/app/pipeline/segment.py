"""RANSAC-based wall plane segmentation.

Runs iterative RANSAC on the wall-band slice to extract the dominant
vertical planes. Each pass removes the plane inliers, then tries again
until it can't find more significant planes or hits max_planes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d


@dataclass
class WallPlane:
    id: str
    plane_eq: np.ndarray        # (a, b, c, d) — normal points INTO the room (positive side = room)
    inlier_points: np.ndarray   # (N, 3)


def extract_wall_planes(
    wall_band: o3d.geometry.PointCloud,
    voxel_size: float = 0.05,
    max_planes: int = 30,
    min_inliers: int = 200,
    distance_threshold: float = 0.03,
    normal_z_max: float = 0.20,     # normal z-component must be small → nearly vertical
    min_height_range: float = 0.30, # plane must span at least 30cm vertically
) -> list[WallPlane]:
    """Extract vertical (wall) planes from the wall band.

    Only planes whose normal vector has a small z-component are kept.
    A minimum vertical span filters out horizontal remnants that
    sneak through the band slicing.
    """
    # Downsample for speed
    ds = wall_band.voxel_down_sample(voxel_size)
    remaining = ds
    planes: list[WallPlane] = []
    plane_counter = 0

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
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        normal = normal / norm_len

        # Vertical plane check: normal z-component must be small
        if abs(normal[2]) > normal_z_max:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        inlier_cloud = remaining.select_by_index(inliers)
        inlier_pts = np.asarray(inlier_cloud.points)

        # Height range check — rules out thin horizontal bands
        z_range = inlier_pts[:, 2].max() - inlier_pts[:, 2].min()
        if z_range < min_height_range:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        # Normalize so plane normal always points "positive x,y" direction
        plane_eq_norm = np.array([a / norm_len, b / norm_len, c / norm_len, d / norm_len])

        plane_counter += 1
        planes.append(WallPlane(
            id=f"wall-{plane_counter:03d}",
            plane_eq=plane_eq_norm,
            inlier_points=inlier_pts,
        ))

        remaining = remaining.select_by_index(inliers, invert=True)

    return planes
