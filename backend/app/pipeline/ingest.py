"""Point cloud ingestion: LAS, LAZ, PLY, E57 → Open3D PointCloud.

Also produces a decimated PLY for the browser viewer.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """Load a point cloud from LAS/LAZ, PLY, or E57."""
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        return _load_las(path)
    elif suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {path}")
        return pcd
    elif suffix == ".e57":
        return _load_e57(path)
    else:
        raise ValueError(f"Unsupported scan format: {suffix}. Expected .las, .laz, .ply, .e57")


def _load_las(path: Path) -> o3d.geometry.PointCloud:
    import laspy
    las = laspy.read(str(path))
    pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # Try to load RGB if present
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        try:
            r = np.asarray(las.red, dtype=np.float64) / 65535.0
            g = np.asarray(las.green, dtype=np.float64) / 65535.0
            b = np.asarray(las.blue, dtype=np.float64) / 65535.0
            colors = np.vstack([r, g, b]).T
            pcd.colors = o3d.utility.Vector3dVector(colors)
        except Exception:
            pass
    return pcd


def _load_e57(path: Path) -> o3d.geometry.PointCloud:
    import pye57
    e57 = pye57.E57(str(path))
    data = e57.read_scan(0, intensity=False, colors=False, ignore_missing_fields=True)
    pts = np.vstack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


def write_decimated_ply(
    pcd: o3d.geometry.PointCloud,
    output_path: Path,
    target_points: int = 200_000,
) -> Path:
    """Downsample to at most target_points, write PLY for browser viewer."""
    n = len(pcd.points)
    if n > target_points:
        voxel_size = _estimate_voxel_for_target(pcd, target_points)
        pcd = pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False, compressed=True)
    return output_path


def _estimate_voxel_for_target(pcd: o3d.geometry.PointCloud, target: int) -> float:
    """Binary search for a voxel size that yields ~target points."""
    pts = np.asarray(pcd.points)
    bbox = pts.max(axis=0) - pts.min(axis=0)
    max_dim = float(bbox.max())
    lo, hi = 0.001, max_dim / 2.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        ds = pcd.voxel_down_sample(mid)
        if len(ds.points) > target:
            lo = mid
        else:
            hi = mid
    return hi
