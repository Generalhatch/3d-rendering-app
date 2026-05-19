"""Principal axis extraction for scan and plan.

Both the scan (from wall normals) and the plan (from DXF line angles) are
reduced to their two dominant horizontal directions, which are then used for
the initial rotation alignment.
"""
from __future__ import annotations

import numpy as np


def principal_axes_from_walls(wall_planes: list) -> np.ndarray:
    """Compute the two dominant horizontal principal axes from wall normals.

    Strategy: collect the horizontal component of each wall normal, build
    an angle histogram in [0, π), find the dominant peak, and return the
    two orthogonal axes corresponding to it.

    Returns shape (2, 2) — two unit vectors in 2D.
    """
    normals_2d = []
    for plane in wall_planes:
        eq = plane.plane_eq
        nx, ny = float(eq[0]), float(eq[1])
        length = np.sqrt(nx**2 + ny**2)
        if length < 1e-9:
            continue
        # Map to [0, π) — opposite-facing normals represent the same wall direction
        angle = np.arctan2(ny, nx) % np.pi
        normals_2d.append(angle)

    if len(normals_2d) < 2:
        # Fallback: assume axis-aligned
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    # Histogram over [0, π)
    bins = 180
    hist, bin_edges = np.histogram(normals_2d, bins=bins, range=(0, np.pi))

    # Find dominant peak
    peak_idx = int(np.argmax(hist))
    dominant_angle = bin_edges[peak_idx] + (bin_edges[1] - bin_edges[0]) / 2.0

    # Primary axis
    ax1 = np.array([np.cos(dominant_angle), np.sin(dominant_angle)])
    ax1 = ax1 / np.linalg.norm(ax1)
    # Secondary axis: perpendicular
    ax2 = np.array([-ax1[1], ax1[0]])

    return np.vstack([ax1, ax2])


def principal_axes_from_lines(plan_lines: np.ndarray) -> np.ndarray:
    """Compute the two dominant horizontal axes from DXF line segments.

    plan_lines: shape (N, 2, 2) — N segments, each (start, end) in 2D.
    Returns shape (2, 2).
    """
    if len(plan_lines) == 0:
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    angles = []
    weights = []
    for seg in plan_lines:
        dx = float(seg[1][0] - seg[0][0])
        dy = float(seg[1][1] - seg[0][1])
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-9:
            continue
        angle = np.arctan2(dy, dx) % np.pi
        angles.append(angle)
        weights.append(length)  # weight by segment length

    if not angles:
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    bins = 360
    hist, bin_edges = np.histogram(angles, bins=bins, range=(0, np.pi), weights=weights)

    peak_idx = int(np.argmax(hist))
    dominant_angle = bin_edges[peak_idx] + (bin_edges[1] - bin_edges[0]) / 2.0

    ax1 = np.array([np.cos(dominant_angle), np.sin(dominant_angle)])
    ax1 = ax1 / np.linalg.norm(ax1)
    ax2 = np.array([-ax1[1], ax1[0]])

    return np.vstack([ax1, ax2])
