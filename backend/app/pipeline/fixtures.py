"""Wall fixture (protrusion) detection.

For each detected wall plane, find clusters of scan points that protrude
2cm–30cm in front of the wall. Each cluster → one Fixture record.

Uses DBSCAN from scikit-learn for density-based clustering.
The "within wall extent" filter prevents a fixture on Wall A from being
attributed to adjacent Wall B (critical at corners).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fixture:
    id: str
    wall_id: str
    centroid: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    protrusion_depth_m: float
    width_m: float
    height_m: float
    point_count: int
    confidence: float


def detect_fixtures(
    full_scan_pts: np.ndarray,          # (N, 3) in PLAN coordinates (post-alignment)
    wall_planes: list,                  # list[WallPlane] from segment.py
    min_protrusion_m: float = 0.02,
    max_protrusion_m: float = 0.30,
    min_cluster_points: int = 50,
    cluster_eps_m: float = 0.05,
) -> list[Fixture]:
    """Detect wall protrusions using DBSCAN on points in front of each wall plane."""
    from sklearn.cluster import DBSCAN

    fixtures: list[Fixture] = []
    fixture_counter = 0

    for wall in wall_planes:
        eq = wall.plane_eq
        a, b, c, d = float(eq[0]), float(eq[1]), float(eq[2]), float(eq[3])
        normal_len = np.sqrt(a**2 + b**2 + c**2)
        if normal_len < 1e-9:
            continue
        normal = np.array([a, b, c]) / normal_len
        d_norm = d / normal_len

        # Signed distance: positive = room side (in front of wall)
        dists = full_scan_pts @ normal + d_norm

        # Points 2cm–30cm in front of the wall
        in_front = (dists >= min_protrusion_m) & (dists <= max_protrusion_m)
        in_front_pts = full_scan_pts[in_front]
        if len(in_front_pts) < min_cluster_points:
            continue

        # Restrict to the wall's XY extent (+ 10cm buffer)
        wx = wall.inlier_points[:, 0]
        wy = wall.inlier_points[:, 1]
        w2d_min = np.array([wx.min() - 0.10, wy.min() - 0.10])
        w2d_max = np.array([wx.max() + 0.10, wy.max() + 0.10])
        within = (
            (in_front_pts[:, 0] >= w2d_min[0]) & (in_front_pts[:, 0] <= w2d_max[0]) &
            (in_front_pts[:, 1] >= w2d_min[1]) & (in_front_pts[:, 1] <= w2d_max[1])
        )
        candidate_pts = in_front_pts[within]
        if len(candidate_pts) < min_cluster_points:
            continue

        clustering = DBSCAN(eps=cluster_eps_m, min_samples=min_cluster_points).fit(candidate_pts)
        labels = clustering.labels_

        for label in set(labels):
            if label == -1:
                continue
            cluster_pts = candidate_pts[labels == label]
            if len(cluster_pts) < min_cluster_points:
                continue

            bbox_min = cluster_pts.min(axis=0)
            bbox_max = cluster_pts.max(axis=0)
            centroid = cluster_pts.mean(axis=0)

            # Protrusion: max distance from wall within this cluster
            c_dists = cluster_pts @ normal + d_norm
            protrusion = float(c_dists.max())

            extent = bbox_max - bbox_min
            width = float(np.linalg.norm(extent[:2]))
            height = float(extent[2])

            # Confidence from point density
            volume = max(float(np.prod(np.maximum(extent, 1e-4))), 1e-9)
            density = len(cluster_pts) / volume
            confidence = float(min(1.0, density / 1000.0))

            # Skip very thin artifacts in all dimensions
            if width < 0.05 and height < 0.05 and protrusion < 0.03:
                continue

            fixture_counter += 1
            fixtures.append(Fixture(
                id=f"fix-{fixture_counter:03d}",
                wall_id=wall.id,
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                bbox_min=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
                bbox_max=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
                protrusion_depth_m=protrusion,
                width_m=width,
                height_m=height,
                point_count=int(len(cluster_pts)),
                confidence=confidence,
            ))

    return fixtures
