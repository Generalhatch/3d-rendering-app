"""Gravity re-leveling: rotate a scan so the detected floor plane is level.

Tilted scanners (tripod on an uneven slab, handheld SLAM drift) produce
clouds whose floor plane is a few degrees off the vertical axis.  Every
downstream stage assumes horizontal slices — a 3° tilt across a 40 m floor
plate smears walls by 2 m of height across the building, so a "shoulder
height" slab catches ceiling at one end and desks at the other.

``gravity_relevel`` rotates the cloud so the floor plane's normal aligns
exactly with the vertical axis and translates it so the floor plane sits at
height 0.  The applied 4×4 transform is returned so callers can (a) map
coordinates back to the original scan frame and (b) re-apply the transform
if they reload the raw cloud.

Safety rails
------------
- Tilts below ``min_tilt_deg`` are ignored (measurement noise; re-leveling
  a level scan would only churn coordinates).
- Tilts above ``max_tilt_deg`` are refused: a "floor" plane 15°+ off axis
  is almost certainly a mis-detected ramp or wall, and silently rotating
  the whole building by that much corrupts everything downstream.  The
  caller should surface a structured warning instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .slicing import FloorReference, detect_floor


def _identity_4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


@dataclass
class RelevelResult:
    """Outcome of a gravity re-level attempt.

    ``pcd`` is the (possibly transformed-in-place) cloud; ``transform`` is
    the 4×4 that was applied (identity when ``applied`` is False).
    ``reason`` is one of ``"releveled"``, ``"below_threshold"``,
    ``"excessive_tilt"``.
    """
    pcd: o3d.geometry.PointCloud
    transform: np.ndarray = field(default_factory=_identity_4)
    tilt_deg: float = 0.0
    applied: bool = False
    reason: str = "below_threshold"


def _rotation_aligning(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix taking unit vector ``u`` onto unit vector ``v``."""
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    k = np.cross(u, v)
    s = float(np.linalg.norm(k))
    if s < 1e-12:
        # Parallel (tilt ≈ 0) — anti-parallel can't occur for tilts < 90°.
        return np.eye(3, dtype=np.float64)
    k = k / s
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def gravity_relevel(
    pcd: o3d.geometry.PointCloud,
    floor: FloorReference | None = None,
    min_tilt_deg: float = 0.5,
    max_tilt_deg: float = 15.0,
) -> RelevelResult:
    """Rotate ``pcd`` (in place) so the floor plane is level at height 0.

    ``floor`` is the RANSAC floor plane from :func:`detect_floor`; detected
    here when not supplied.  Only tilts in ``[min_tilt_deg, max_tilt_deg]``
    are corrected — see the module docstring for why both bounds exist.
    """
    if floor is None:
        floor = detect_floor(pcd)

    axis = np.zeros(3, dtype=np.float64)
    axis[floor.axis_idx] = 1.0

    # Unit plane normal, oriented upward (same hemisphere as the axis).
    n_raw = np.asarray(floor.plane_eq[:3], dtype=np.float64)
    n_len = float(np.linalg.norm(n_raw))
    if n_len < 1e-12:
        return RelevelResult(pcd=pcd, reason="below_threshold")
    normal = n_raw / n_len
    d = float(floor.plane_eq[3]) / n_len
    if float(np.dot(normal, axis)) < 0:
        normal = -normal
        d = -d

    tilt_deg = float(np.degrees(np.arccos(np.clip(np.dot(normal, axis), -1.0, 1.0))))

    if tilt_deg < min_tilt_deg:
        return RelevelResult(pcd=pcd, tilt_deg=tilt_deg, reason="below_threshold")
    if tilt_deg > max_tilt_deg:
        return RelevelResult(pcd=pcd, tilt_deg=tilt_deg, reason="excessive_tilt")

    # Rotate normal → axis, then translate so the floor plane lands at
    # height 0 along the axis.  Points on the plane satisfy n·x = −d; after
    # x' = R x (with R n = axis) they satisfy axis·x' = −d, so shifting by
    # +d along the axis puts the plane exactly at coordinate 0.
    R = _rotation_aligning(normal, axis)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = d * axis

    pcd.transform(T)
    return RelevelResult(
        pcd=pcd, transform=T, tilt_deg=tilt_deg, applied=True, reason="releveled",
    )
