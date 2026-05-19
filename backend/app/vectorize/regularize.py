"""Geometric clean-up of raw detector output.

Detectors return many short, slightly-misaligned segments — the wall at
``X = 12.0 m`` might appear as a dozen pieces between ``X = 11.97`` and
``X = 12.04``.  This module turns those into a small set of clean,
human-meaningful CAD entities.

Operations (each is independently toggleable):
  1. drop_short            — discard segments below a minimum length
  2. manhattan_snap        — snap segments to the two dominant building axes
  3. merge_collinear       — fuse near-parallel, near-collinear segments that
                              are close end-to-end into one longer segment

All inputs / outputs are ``(N, 2, 2)`` world-coordinate arrays —
``[[x1, y1], [x2, y2]]`` per segment, in metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegularizeParams:
    drop_short_below_m: float = 0.50
    manhattan_snap: bool = True
    manhattan_tolerance_deg: float = 12.0
    merge_collinear: bool = True
    merge_parallel_tolerance_deg: float = 3.0
    merge_perpendicular_distance_m: float = 0.05    # max perp distance between two segments to merge
    merge_endpoint_gap_m: float = 0.30              # max endpoint-to-endpoint gap along the line


# ── Public entry point ────────────────────────────────────────────────────────

def regularize(
    segments: np.ndarray,
    params: RegularizeParams | None = None,
) -> np.ndarray:
    """Apply the full regularization pipeline.

    Returns an ``(M, 2, 2)`` array where ``M <= N``.
    """
    if params is None:
        params = RegularizeParams()
    if len(segments) == 0:
        return segments

    out = segments.astype(np.float64).copy()

    out = drop_short(out, params.drop_short_below_m)
    if len(out) == 0:
        return out

    if params.manhattan_snap:
        out = snap_to_manhattan(out, params.manhattan_tolerance_deg)
        if len(out) == 0:
            return out

    if params.merge_collinear:
        out = merge_collinear_segments(
            out,
            parallel_tol_deg=params.merge_parallel_tolerance_deg,
            perp_distance_m=params.merge_perpendicular_distance_m,
            endpoint_gap_m=params.merge_endpoint_gap_m,
        )

    return out


# ── Individual stages ─────────────────────────────────────────────────────────

def drop_short(segments: np.ndarray, min_length_m: float) -> np.ndarray:
    """Remove segments shorter than ``min_length_m``."""
    if len(segments) == 0:
        return segments
    deltas = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    return segments[lengths >= min_length_m]


def snap_to_manhattan(
    segments: np.ndarray,
    tolerance_deg: float = 12.0,
) -> np.ndarray:
    """Keep only segments aligned with the two dominant building axes.

    The dominant axis is found via a length-weighted angle histogram; the
    perpendicular axis is its complement.  Segments within ``tolerance_deg``
    of either axis are kept *and* snapped to that axis exactly.

    If filtering would drop all segments (building is highly rotated or
    non-Manhattan), the input is returned unchanged.
    """
    if len(segments) == 0:
        return segments

    deltas = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 180.0  # [0, 180)

    # Length-weighted histogram (long segments vote more than short ones).
    bins = np.arange(0, 181, 2)
    hist, _ = np.histogram(angles, bins=bins, weights=lengths)
    dominant_angle = float(bins[int(np.argmax(hist))]) + 1.0   # bin centre
    perp_angle = (dominant_angle + 90.0) % 180.0

    def _diff(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 180.0 - d)

    kept: list[np.ndarray] = []
    for seg, ang, length in zip(segments, angles, lengths):
        d_dom = _diff(ang, dominant_angle)
        d_perp = _diff(ang, perp_angle)

        if d_dom <= tolerance_deg:
            snap_angle_rad = np.deg2rad(dominant_angle)
        elif d_perp <= tolerance_deg:
            snap_angle_rad = np.deg2rad(perp_angle)
        else:
            continue

        # Snap: rotate the segment so its direction matches the snap angle exactly,
        # preserving its centre and length.
        centre = (seg[0] + seg[1]) / 2.0
        half = length / 2.0
        dx = np.cos(snap_angle_rad) * half
        dy = np.sin(snap_angle_rad) * half
        kept.append(np.array([
            [centre[0] - dx, centre[1] - dy],
            [centre[0] + dx, centre[1] + dy],
        ]))

    if not kept:
        return segments  # filtering wiped everything — bail out, return original
    return np.stack(kept, axis=0)


def merge_collinear_segments(
    segments: np.ndarray,
    parallel_tol_deg: float = 3.0,
    perp_distance_m: float = 0.05,
    endpoint_gap_m: float = 0.30,
) -> np.ndarray:
    """Fuse near-parallel, near-collinear, near-touching segments.

    Algorithm:
      - Build a "compatibility" relation: two segments are compatible if their
        angles differ by ≤ ``parallel_tol_deg``, their perpendicular distance
        is ≤ ``perp_distance_m``, and there is a point on one that lies within
        ``endpoint_gap_m`` of the other (i.e. they're not on disjoint walls of
        the same orientation across the building).
      - Find connected components in this relation (union-find).
      - For each component, project all endpoints onto the component's mean
        direction and emit a single segment spanning min→max projection.

    This is O(N²) in segment count but N is typically < 1000 after Manhattan
    snapping, so it stays comfortable.
    """
    n = len(segments)
    if n <= 1:
        return segments

    deltas = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    # Avoid divide-by-zero for degenerate segments.
    safe_lengths = np.where(lengths > 1e-9, lengths, 1.0)
    dirs = deltas / safe_lengths[:, None]
    angles = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180.0
    centres = (segments[:, 0, :] + segments[:, 1, :]) / 2.0

    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def _angle_diff(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 180.0 - d)

    for i in range(n):
        for j in range(i + 1, n):
            if _angle_diff(angles[i], angles[j]) > parallel_tol_deg:
                continue

            # Perpendicular distance: distance from j's centre to the infinite
            # line through i, measured perpendicular to i's direction.
            di = dirs[i]
            perp = np.array([-di[1], di[0]])
            perp_dist = abs(float(np.dot(centres[j] - centres[i], perp)))
            if perp_dist > perp_distance_m:
                continue

            # Endpoint gap along the (shared) direction.  Project each endpoint
            # of both segments onto direction i, take the min gap between the
            # two segments' projections.
            t_i0 = float(np.dot(segments[i, 0] - centres[i], di))
            t_i1 = float(np.dot(segments[i, 1] - centres[i], di))
            t_j0 = float(np.dot(segments[j, 0] - centres[i], di))
            t_j1 = float(np.dot(segments[j, 1] - centres[i], di))
            i_lo, i_hi = min(t_i0, t_i1), max(t_i0, t_i1)
            j_lo, j_hi = min(t_j0, t_j1), max(t_j0, t_j1)
            # Overlap or gap?
            gap = max(0.0, max(i_lo, j_lo) - min(i_hi, j_hi))
            if gap > endpoint_gap_m:
                continue

            union(i, j)

    # Build components
    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    out: list[np.ndarray] = []
    for members in components.values():
        if len(members) == 1:
            out.append(segments[members[0]])
            continue

        # Mean direction (length-weighted) of the component.
        comp_dirs = dirs[members]
        comp_lens = lengths[members]
        # Force all directions into the same half-plane before averaging.
        ref = comp_dirs[0]
        flipped = comp_dirs * np.sign(comp_dirs @ ref)[:, None]
        mean_dir = (flipped * comp_lens[:, None]).sum(axis=0)
        mean_dir = mean_dir / max(float(np.linalg.norm(mean_dir)), 1e-9)

        # Component centroid (length-weighted)
        comp_centres = centres[members]
        centroid = (comp_centres * comp_lens[:, None]).sum(axis=0) / comp_lens.sum()

        # Project every endpoint onto mean_dir
        endpoints = segments[members].reshape(-1, 2)
        ts = (endpoints - centroid) @ mean_dir
        t_min, t_max = float(ts.min()), float(ts.max())

        merged = np.array([
            centroid + mean_dir * t_min,
            centroid + mean_dir * t_max,
        ])
        out.append(merged)

    return np.stack(out, axis=0)
