"""Wall-thickness pairing: collapse parallel detections into thickness-aware walls.

What this fixes
---------------
After the detector + regularizer, a real wall typically appears as **two
parallel centerlines** — one for each face of the wall — about 10–25 cm
apart (interior partition vs. demising vs. exterior shell).  Drawn as raw
centerlines they look like duplicate line noise.  The deliverable should
emit them as a single ``Wall`` entity with two face polylines and a
thickness — that's what makes a DXF look like architectural CAD instead of
a scribble.

Algorithm
---------
1. Compute angle, length, and centre for every segment.
2. For each ordered pair (i, j) with i.length ≥ j.length:
   - Reject if angle differs by > ``parallel_tol_deg``.
   - Reject if perpendicular distance < ``min_thickness_m`` or
     > ``max_thickness_m``.
   - Reject if overlap along i's direction < ``min_overlap_fraction``.
   - Otherwise: candidate pair, score = overlap_length / max_thickness_dev.
3. Greedy assign: walk candidates by descending score, mark both segments
   as paired if both are still unpaired.
4. For each paired set: compute centerline (midline of the two face
   projections onto the mean direction), thickness, and the two face
   polylines (parallel to the centerline, offset by ±thickness/2).
5. Unpaired segments stay as-is — emitted as single-faced walls with the
   median thickness from the paired set (or 0.10 m default if no pairs).

Output schema
-------------
``pair_walls`` returns a :class:`WallPairingResult` with:
- ``walls``: list of :class:`Wall` (paired + unpaired).
- ``face_segments``: ``(N*2, 2, 2)`` numpy array — both faces of every
  wall, flattened.  Drawn on ``WALLS_FACES`` layer.
- ``centerline_segments``: ``(N, 2, 2)`` — one centerline per wall.
  Drawn on the existing ``WALLS`` layer.

Thresholds tuned for typical interior + commercial scans at 1 cm/px
resolution.  Sane edge defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class WallPairingParams:
    """Tunables for wall pairing.  Defaults handle 95 % of cases."""

    # Allowable thickness range for paired walls.  Interior partition ~10 cm,
    # demising wall ~15 cm, exterior block ~20–25 cm, brick / CMU ~25–35 cm.
    min_thickness_m: float = 0.07
    max_thickness_m: float = 0.40

    # Two segments are "parallel" if their direction angles differ by ≤ this.
    parallel_tol_deg: float = 4.0

    # Minimum overlap of the shorter segment over the longer segment,
    # measured along the longer segment's direction.  60 % is conservative
    # (eliminates near-parallel walls that share only an end).
    min_overlap_fraction: float = 0.60

    # Default thickness for unpaired segments when no statistics are
    # available (residential scan with weak wall returns).  Used only as a
    # fallback — most unpaired-wall thickness will come from the median of
    # paired thicknesses on the same scan.
    fallback_thickness_m: float = 0.10

    # After pairing, drop UNPAIRED walls shorter than this.  These are almost
    # always furniture edges or scanner artefacts that survived regularize but
    # don't pair with anything.  Paired walls are kept regardless of length —
    # if two parallel detections agreed on a wall existing, we trust it.
    drop_unpaired_below_m: float = 0.50


@dataclass
class Wall:
    """One thickness-aware wall.

    ``centerline`` is the midline through the wall body.  ``thickness_m`` is
    the perpendicular distance between the two faces.  ``face_a`` / ``face_b``
    are the two parallel face polylines, both parallel to ``centerline``.
    ``is_paired`` = True when both faces were detected; False for single-
    sided walls (one face scanned only).
    """
    centerline: np.ndarray            # (2, 2) — [[x1, y1], [x2, y2]]
    thickness_m: float
    face_a: np.ndarray                # (2, 2)
    face_b: np.ndarray                # (2, 2)
    is_paired: bool
    # Indices into the original ``segments`` array that contributed.
    # Useful for the editor to map an edit on a face back to the source.
    source_indices: tuple[int, ...] = ()


@dataclass
class WallPairingResult:
    """Output of :func:`pair_walls`."""
    walls: list[Wall]
    face_segments: np.ndarray         # (2*N, 2, 2) — flat list of both faces
    centerline_segments: np.ndarray   # (N, 2, 2)
    median_thickness_m: float
    n_paired: int
    n_unpaired: int
    n_orphans_dropped: int = 0        # short unpaired fragments dropped


# ── Public API ────────────────────────────────────────────────────────────────

def pair_walls(
    segments: np.ndarray,
    params: WallPairingParams | None = None,
) -> WallPairingResult:
    """Find parallel-segment pairs, build thickness-aware Wall entities.

    Parameters
    ----------
    segments : np.ndarray
        ``(N, 2, 2)`` world-coordinate segments, in metres.
    params : WallPairingParams, optional

    Returns
    -------
    WallPairingResult
    """
    if params is None:
        params = WallPairingParams()
    n = len(segments)
    if n == 0:
        return WallPairingResult(
            walls=[],
            face_segments=np.zeros((0, 2, 2), dtype=np.float64),
            centerline_segments=np.zeros((0, 2, 2), dtype=np.float64),
            median_thickness_m=params.fallback_thickness_m,
            n_paired=0,
            n_unpaired=0,
        )

    segs = segments.astype(np.float64)

    # Per-segment geometry.
    deltas = segs[:, 1, :] - segs[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    safe_len = np.where(lengths > 1e-9, lengths, 1.0)
    dirs = deltas / safe_len[:, None]
    angles = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180.0
    centres = (segs[:, 0, :] + segs[:, 1, :]) / 2.0

    # Build candidate pair list with scores.
    pairs: list[tuple[float, int, int, float]] = []  # (score, i, j, thickness)
    for i in range(n):
        for j in range(i + 1, n):
            # Quick rejects in increasing-cost order.
            ang_diff = _angle_diff(angles[i], angles[j])
            if ang_diff > params.parallel_tol_deg:
                continue

            # Perpendicular distance between j's centre and the infinite line through i.
            di = dirs[i]
            perp = np.array([-di[1], di[0]])
            perp_dist = abs(float(np.dot(centres[j] - centres[i], perp)))
            if perp_dist < params.min_thickness_m or perp_dist > params.max_thickness_m:
                continue

            # Overlap along i's direction.  Project all four endpoints onto
            # the line through i's centre with i's direction; compute the
            # overlap of [min_i, max_i] with [min_j, max_j].
            t_i0 = float(np.dot(segs[i, 0] - centres[i], di))
            t_i1 = float(np.dot(segs[i, 1] - centres[i], di))
            t_j0 = float(np.dot(segs[j, 0] - centres[i], di))
            t_j1 = float(np.dot(segs[j, 1] - centres[i], di))
            i_lo, i_hi = sorted((t_i0, t_i1))
            j_lo, j_hi = sorted((t_j0, t_j1))
            overlap = max(0.0, min(i_hi, j_hi) - max(i_lo, j_lo))
            shorter_len = min(lengths[i], lengths[j])
            if shorter_len < 1e-6:
                continue
            overlap_frac = overlap / shorter_len
            if overlap_frac < params.min_overlap_fraction:
                continue

            # Score: prefer high overlap and middle-of-range thickness.
            # A wall around 12 cm is the most common interior partition;
            # weight scores so we pair those first when ambiguous.
            target_thickness = 0.12
            thickness_penalty = abs(perp_dist - target_thickness) / target_thickness
            score = overlap_frac * (1.0 - 0.15 * thickness_penalty)
            pairs.append((float(score), i, j, float(perp_dist)))

    # Greedy assign highest-score pairs first.
    pairs.sort(key=lambda x: -x[0])
    paired_of: list[Optional[int]] = [None] * n
    thicknesses_paired: list[float] = []
    pair_records: list[tuple[int, int, float]] = []
    for score, i, j, thickness in pairs:
        if paired_of[i] is not None or paired_of[j] is not None:
            continue
        paired_of[i] = j
        paired_of[j] = i
        pair_records.append((i, j, thickness))
        thicknesses_paired.append(thickness)

    median_thickness = (
        float(np.median(thicknesses_paired))
        if thicknesses_paired
        else params.fallback_thickness_m
    )

    walls: list[Wall] = []

    # Build paired walls first.
    for i, j, thickness in pair_records:
        wall = _build_paired_wall(segs[i], segs[j], dirs[i], dirs[j], thickness)
        wall.source_indices = (i, j)
        walls.append(wall)

    # Then unpaired ones, using median thickness from the paired set —
    # but drop fragments shorter than the orphan-drop threshold.  These
    # are almost always furniture edges or scanner artefacts that survived
    # regularize without finding a pair.
    n_orphans_dropped = 0
    for idx in range(n):
        if paired_of[idx] is not None:
            continue
        if float(lengths[idx]) < params.drop_unpaired_below_m:
            n_orphans_dropped += 1
            continue
        wall = _build_unpaired_wall(
            segs[idx], dirs[idx], thickness=median_thickness,
        )
        wall.source_indices = (idx,)
        walls.append(wall)

    # Flatten faces + centerlines.
    face_segments = (
        np.stack([np.stack([w.face_a, w.face_b], axis=0) for w in walls], axis=0)
        .reshape(-1, 2, 2)
        if walls else np.zeros((0, 2, 2), dtype=np.float64)
    )
    centerline_segments = (
        np.stack([w.centerline for w in walls], axis=0)
        if walls else np.zeros((0, 2, 2), dtype=np.float64)
    )

    n_unpaired_kept = sum(
        1 for idx, p in enumerate(paired_of)
        if p is None and float(lengths[idx]) >= params.drop_unpaired_below_m
    )
    return WallPairingResult(
        walls=walls,
        face_segments=face_segments,
        centerline_segments=centerline_segments,
        median_thickness_m=median_thickness,
        n_paired=len(pair_records),
        n_unpaired=n_unpaired_kept,
        n_orphans_dropped=n_orphans_dropped,
    )


# ── Internals ────────────────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    """Smallest difference between two angles modulo 180°."""
    d = abs(a - b)
    return min(d, 180.0 - d)


def _build_paired_wall(
    seg_i: np.ndarray,
    seg_j: np.ndarray,
    dir_i: np.ndarray,
    dir_j: np.ndarray,
    thickness: float,
) -> Wall:
    """Construct a Wall from two paired face segments.

    Strategy:
      1. Use length-weighted mean of the two directions as the wall axis.
      2. Project all four endpoints onto the axis through the combined
         centroid → get the wall's longitudinal extent [t_min, t_max].
      3. Centerline = centroid + axis × {t_min, t_max}.
      4. Face polylines = centerline ± offset where offset is the perpendicular
         vector of length thickness/2.
    """
    # Force directions into the same half-plane before averaging (one segment
    # may be reversed relative to the other).
    if np.dot(dir_i, dir_j) < 0:
        dir_j = -dir_j

    len_i = float(np.linalg.norm(seg_i[1] - seg_i[0]))
    len_j = float(np.linalg.norm(seg_j[1] - seg_j[0]))
    total_len = max(len_i + len_j, 1e-9)
    axis = (dir_i * len_i + dir_j * len_j) / total_len
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)

    centroid = (
        (seg_i[0] + seg_i[1]) * len_i + (seg_j[0] + seg_j[1]) * len_j
    ) / (2.0 * total_len)

    # Project all endpoints onto the axis through centroid.
    endpoints = np.vstack([seg_i, seg_j])  # (4, 2)
    ts = (endpoints - centroid) @ axis
    t_min, t_max = float(ts.min()), float(ts.max())

    centerline_a = centroid + axis * t_min
    centerline_b = centroid + axis * t_max
    centerline = np.array([centerline_a, centerline_b])

    # Perpendicular vector (axis rotated 90° CCW), pointing toward face A.
    perp = np.array([-axis[1], axis[0]])
    half_thick = thickness / 2.0

    # Decide which side of the centerline each input segment was on, so
    # face_a matches seg_i's side (less surprising in the editor).
    side_of_i = float(np.dot(seg_i[0] - centroid, perp))
    sign = 1.0 if side_of_i >= 0 else -1.0

    face_a = np.array([
        centerline_a + perp * (sign * half_thick),
        centerline_b + perp * (sign * half_thick),
    ])
    face_b = np.array([
        centerline_a - perp * (sign * half_thick),
        centerline_b - perp * (sign * half_thick),
    ])

    return Wall(
        centerline=centerline,
        thickness_m=float(thickness),
        face_a=face_a,
        face_b=face_b,
        is_paired=True,
    )


def _build_unpaired_wall(
    seg: np.ndarray,
    direction: np.ndarray,
    thickness: float,
) -> Wall:
    """Construct a single-faced wall (only one face was detected).

    We treat the detected segment AS the centerline of the wall — the most
    conservative choice given we have no information about which side of
    the wall the scanner saw.  Faces are emitted on either side at ±t/2.
    Operator can flip the offset side in the editor if it's wrong.
    """
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    perp = np.array([-direction[1], direction[0]])
    half_thick = thickness / 2.0
    face_a = np.array([seg[0] + perp * half_thick, seg[1] + perp * half_thick])
    face_b = np.array([seg[0] - perp * half_thick, seg[1] - perp * half_thick])
    return Wall(
        centerline=seg.copy(),
        thickness_m=float(thickness),
        face_a=face_a,
        face_b=face_b,
        is_paired=False,
    )


def faces_from_centerlines(
    centerlines: np.ndarray,
    thickness_m: float,
) -> np.ndarray:
    """Emit double-line faces for already-finished centerlines.

    Used after topology finishes wall axes (Cloud2BIM § 2.8) so the editor
    / DXF show continuous CAD walls instead of the pre-snap fragments.
    """
    segs = np.asarray(centerlines, dtype=float).reshape(-1, 2, 2)
    if len(segs) == 0:
        return np.zeros((0, 2, 2), dtype=np.float64)
    thick = max(float(thickness_m), 0.05)
    faces: list[np.ndarray] = []
    for seg in segs:
        d = seg[1] - seg[0]
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        direction = d / L
        wall = _build_unpaired_wall(seg, direction, thickness=thick)
        faces.append(wall.face_a)
        faces.append(wall.face_b)
    if not faces:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.stack(faces, axis=0)


def pairing_summary(result: WallPairingResult) -> str:
    """One-line human-readable summary for SSE progress."""
    extra = (
        f" · dropped {result.n_orphans_dropped} short orphans"
        if result.n_orphans_dropped else ""
    )
    return (
        f"walls: {result.n_paired} paired + {result.n_unpaired} single-face = "
        f"{len(result.walls)} total · "
        f"median thickness {result.median_thickness_m * 100:.1f} cm{extra}"
    )
