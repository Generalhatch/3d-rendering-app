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

Two public entry points:
  - :func:`regularize`                — returns just the kept segments
                                        (legacy; used by callers that don't
                                        care about provenance)
  - :func:`regularize_with_provenance`— returns kept *and* rejected segments
                                        with per-segment "why it was
                                        dropped" tags.  The Vectorize editor
                                        consumes the rejected list to render
                                        ghost candidates the operator can
                                        rescue with one click.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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

    # ── Diagonal / curved wall protection (Phase 3.5) ──────────────────────
    # Genuinely non-Manhattan geometry must survive regularization: the
    # reference sheets preserve diagonal and curved facades faithfully.
    keep_diagonal_min_length_m: float = 1.0
    """Off-axis segments at least this long are KEPT unchanged instead of
    being dropped by the Manhattan filter — a 5 m wall at 45° is a real
    diagonal wall, not noise.  Shorter off-axis segments are still dropped
    (unless part of a protected curve chain)."""

    protect_curve_chains: bool = True
    """Detect chains of end-to-end segments whose direction turns
    progressively (chords of a curved wall) and exempt them from Manhattan
    snapping entirely — snapping individual chords near the axes would
    flatten the curve into a staircase."""

    curve_chain_endpoint_tol_m: float = 0.15
    curve_chain_min_segments: int = 3
    curve_chain_min_step_deg: float = 2.0
    curve_chain_max_step_deg: float = 30.0


@dataclass
class RejectedSegment:
    """One segment the pipeline dropped, with enough metadata for the editor
    to (a) render it as a ghost and (b) explain *why* on hover.

    ``dropped_by`` is one of:
      - ``"short"``     — under :attr:`RegularizeParams.drop_short_below_m`
      - ``"manhattan"`` — too far from either dominant axis
      - ``"merged"``    — absorbed by another segment during collinear merge
                           (the kept segment's centreline replaces this one's
                            geometry; rescuing it is sometimes useful if the
                            merge was over-aggressive)
    """
    seg: np.ndarray          # shape (2, 2), float64 world coords
    dropped_by: str          # "short" | "manhattan" | "merged"
    length_m: float
    angle_deg: float         # 0..180


@dataclass
class RegularizeResult:
    kept: np.ndarray                          # (M, 2, 2)
    rejected: list[RejectedSegment] = field(default_factory=list)
    # True when the Manhattan filter would have dropped EVERY segment and
    # bailed out, returning the input unchanged (highly rotated or
    # non-Manhattan building).  Surfaced as a structured pipeline warning —
    # previously this fallback was completely silent.
    manhattan_bailed: bool = False


# ── Public entry point ────────────────────────────────────────────────────────

def regularize(
    segments: np.ndarray,
    params: RegularizeParams | None = None,
) -> np.ndarray:
    """Apply the full regularization pipeline (kept-only, legacy entry).

    Equivalent to ``regularize_with_provenance(...).kept`` — preserved so
    older call-sites don't change shape.
    """
    return regularize_with_provenance(segments, params).kept


def regularize_with_provenance(
    segments: np.ndarray,
    params: RegularizeParams | None = None,
) -> RegularizeResult:
    """Run the full regularization pipeline and report what was dropped.

    The Vectorize editor uses the ``rejected`` list to render ghost candidates
    that the operator can promote back to active with one click.
    """
    if params is None:
        params = RegularizeParams()
    if len(segments) == 0:
        return RegularizeResult(kept=segments, rejected=[])

    rejected: list[RejectedSegment] = []
    manhattan_bailed = False
    out = segments.astype(np.float64).copy()

    # Stage 1: drop short
    out, short_rejected = _drop_short_with_provenance(out, params.drop_short_below_m)
    rejected.extend(short_rejected)
    if len(out) == 0:
        return RegularizeResult(kept=out, rejected=rejected)

    # Stage 2: Manhattan snap (with diagonal / curve-chain protection)
    if params.manhattan_snap:
        protected: set[int] = set()
        if params.protect_curve_chains:
            protected = find_curve_chain_indices(
                out,
                endpoint_tol_m=params.curve_chain_endpoint_tol_m,
                min_segments=params.curve_chain_min_segments,
                min_step_deg=params.curve_chain_min_step_deg,
                max_step_deg=params.curve_chain_max_step_deg,
            )
        out, manhattan_rejected, manhattan_bailed = (
            _snap_to_manhattan_with_provenance(
                out, params.manhattan_tolerance_deg,
                protected_indices=protected,
                keep_diagonal_min_length_m=params.keep_diagonal_min_length_m,
            )
        )
        rejected.extend(manhattan_rejected)
        if len(out) == 0:
            return RegularizeResult(
                kept=out, rejected=rejected, manhattan_bailed=manhattan_bailed,
            )

    # Stage 3: merge collinear
    if params.merge_collinear:
        out, merged_rejected = _merge_collinear_with_provenance(
            out,
            parallel_tol_deg=params.merge_parallel_tolerance_deg,
            perp_distance_m=params.merge_perpendicular_distance_m,
            endpoint_gap_m=params.merge_endpoint_gap_m,
        )
        rejected.extend(merged_rejected)

    return RegularizeResult(
        kept=out, rejected=rejected, manhattan_bailed=manhattan_bailed,
    )


def _segment_length_angle(seg: np.ndarray) -> tuple[float, float]:
    dx = float(seg[1, 0] - seg[0, 0])
    dy = float(seg[1, 1] - seg[0, 1])
    length = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
    return length, angle


# ── Individual stages ─────────────────────────────────────────────────────────

def drop_short(segments: np.ndarray, min_length_m: float) -> np.ndarray:
    """Remove segments shorter than ``min_length_m``."""
    if len(segments) == 0:
        return segments
    deltas = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    return segments[lengths >= min_length_m]


def _drop_short_with_provenance(
    segments: np.ndarray, min_length_m: float,
) -> tuple[np.ndarray, list[RejectedSegment]]:
    if len(segments) == 0:
        return segments, []
    deltas = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    kept_mask = lengths >= min_length_m
    rejected = []
    for i, drop in enumerate(~kept_mask):
        if drop:
            length, angle = _segment_length_angle(segments[i])
            rejected.append(RejectedSegment(
                seg=segments[i].copy(),
                dropped_by="short",
                length_m=length,
                angle_deg=angle,
            ))
    return segments[kept_mask], rejected


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


def find_curve_chain_indices(
    segments: np.ndarray,
    endpoint_tol_m: float = 0.15,
    min_segments: int = 3,
    min_step_deg: float = 2.0,
    max_step_deg: float = 30.0,
) -> set[int]:
    """Indices of segments that are chords of a curved wall.

    A curved wall sliced into short chords appears as a chain of
    end-to-end segments whose direction turns progressively — a small,
    consistent-sign angle step from each chord to the next.  Snapping the
    near-axis chords of such a chain to the Manhattan axes would flatten
    the curve into a staircase, so these chains are exempted.

    Detection: build endpoint-adjacency chains (segments joined end to end
    within ``endpoint_tol_m``, path nodes only — junction segments with
    3+ neighbours break chains), orient directions along each walk, and
    protect every maximal run of ≥ ``min_segments`` segments whose
    consecutive direction deltas all share one sign and lie within
    ``[min_step_deg, max_step_deg]``.  A rectangle (±90° steps) or a long
    straight run (~0° steps) never qualifies.
    """
    n = len(segments)
    if n < min_segments:
        return set()

    ends = segments.reshape(n, 2, 2)
    tol_sq = endpoint_tol_m * endpoint_tol_m

    # adjacency[i] = list of (j, end_of_i, end_of_j) sharing an endpoint.
    adjacency: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            for ei in (0, 1):
                for ej in (0, 1):
                    d = ends[i, ei] - ends[j, ej]
                    if float(d @ d) <= tol_sq:
                        adjacency[i].append((j, ei, ej))
                        adjacency[j].append((i, ej, ei))

    # Path nodes only: a segment meeting 3+ others is a junction, not a
    # curve chord link — chains break there.
    degree = {i: len(adjacency[i]) for i in range(n)}

    protected: set[int] = set()
    visited: set[int] = set()
    for start in range(n):
        if start in visited:
            continue
        if degree[start] > 2:
            visited.add(start)          # junction segment: never a chord
            continue
        if degree[start] == 2:
            continue                    # mid-chain: reached from an end

        # degree 0 or 1 → walk the chain from this end.
        path: list[int] = [start]
        dirs: list[np.ndarray] = []
        visited.add(start)
        cur = start
        entry_end: int | None = None    # which end of `cur` we entered through
        while True:
            nbrs = [
                (j, ei, ej) for j, ei, ej in adjacency[cur]
                if j not in visited and degree[j] <= 2
                and (entry_end is None or ei != entry_end)
            ]
            if not nbrs:
                break
            j, ei, ej = nbrs[0]
            # Direction of `cur` oriented toward the shared endpoint.
            d_cur = ends[cur, ei] - ends[cur, 1 - ei]
            dirs.append(d_cur / max(float(np.hypot(*d_cur)), 1e-12))
            path.append(j)
            visited.add(j)
            cur = j
            entry_end = ej
        if entry_end is not None:
            # Final segment: oriented away from its entry endpoint.
            d_last = ends[cur, 1 - entry_end] - ends[cur, entry_end]
            dirs.append(d_last / max(float(np.hypot(*d_last)), 1e-12))

        if len(path) < min_segments or len(dirs) != len(path):
            continue

        # Signed direction delta from each chord to the next.
        steps: list[float] = []
        for a, b in zip(dirs[:-1], dirs[1:]):
            cross = float(a[0] * b[1] - a[1] * b[0])
            dot = float(a @ b)
            steps.append(float(np.degrees(np.arctan2(cross, dot))))

        qualifies = [min_step_deg <= abs(s) <= max_step_deg for s in steps]

        # Protect every maximal run of qualifying, same-sign steps that
        # covers ≥ min_segments segments (a run of L steps spans L+1
        # segments).
        run_start = 0
        for k in range(len(steps) + 1):
            run_continues = (
                k < len(steps)
                and qualifies[k]
                and (k == run_start or np.sign(steps[k]) == np.sign(steps[k - 1]))
            )
            if run_continues:
                continue
            if k > run_start and (k - run_start) + 1 >= min_segments:
                protected.update(path[run_start:k + 1])
            run_start = k if (k < len(steps) and qualifies[k]) else k + 1

    return protected


def _snap_to_manhattan_with_provenance(
    segments: np.ndarray,
    tolerance_deg: float = 12.0,
    protected_indices: set[int] | None = None,
    keep_diagonal_min_length_m: float = 0.0,
) -> tuple[np.ndarray, list[RejectedSegment], bool]:
    """Provenance-aware ``snap_to_manhattan``.

    Returns ``(snapped_kept, dropped, bailed)`` where ``dropped`` are the
    segments that were too far from either dominant axis and ``bailed`` is
    True when the filter would have wiped every segment and returned the
    input unchanged instead.  Snapped survivors are returned in their
    snapped (axis-aligned) form, matching the legacy behaviour, with two
    Phase 3.5 protections:

    - ``protected_indices`` (curve chords) pass through UNSNAPPED — never
      rotated, never dropped.
    - Off-axis segments of at least ``keep_diagonal_min_length_m`` pass
      through unsnapped instead of being dropped (a long diagonal is a
      real wall, not noise).
    """
    if len(segments) == 0:
        return segments, [], False
    protected_indices = protected_indices or set()

    deltas = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 180.0

    bins = np.arange(0, 181, 2)
    hist, _ = np.histogram(angles, bins=bins, weights=lengths)
    dominant_angle = float(bins[int(np.argmax(hist))]) + 1.0
    perp_angle = (dominant_angle + 90.0) % 180.0

    def _diff(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 180.0 - d)

    kept: list[np.ndarray] = []
    rejected: list[RejectedSegment] = []

    for idx, (seg, ang, length) in enumerate(zip(segments, angles, lengths)):
        if idx in protected_indices:
            kept.append(seg.copy())     # curve chord: pass through untouched
            continue

        d_dom = _diff(ang, dominant_angle)
        d_perp = _diff(ang, perp_angle)

        if d_dom <= tolerance_deg:
            snap_angle_rad = np.deg2rad(dominant_angle)
        elif d_perp <= tolerance_deg:
            snap_angle_rad = np.deg2rad(perp_angle)
        elif (
            keep_diagonal_min_length_m > 0.0
            and length >= keep_diagonal_min_length_m
        ):
            kept.append(seg.copy())     # genuine diagonal wall: keep as-is
            continue
        else:
            rejected.append(RejectedSegment(
                seg=seg.copy(),
                dropped_by="manhattan",
                length_m=float(length),
                angle_deg=float(ang),
            ))
            continue

        centre = (seg[0] + seg[1]) / 2.0
        half = length / 2.0
        dx = np.cos(snap_angle_rad) * half
        dy = np.sin(snap_angle_rad) * half
        kept.append(np.array([
            [centre[0] - dx, centre[1] - dy],
            [centre[0] + dx, centre[1] + dy],
        ]))

    if not kept:
        # Bail out: filter wiped everything → return original input, drop
        # the rejection list (legacy behaviour matched).  The ``bailed``
        # flag lets the pipeline surface this as a structured warning.
        return segments, [], True
    return np.stack(kept, axis=0), rejected, False


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


def _merge_collinear_with_provenance(
    segments: np.ndarray,
    parallel_tol_deg: float = 3.0,
    perp_distance_m: float = 0.05,
    endpoint_gap_m: float = 0.30,
) -> tuple[np.ndarray, list[RejectedSegment]]:
    """Provenance-aware merge.

    Tracks the input segments that got absorbed into a multi-member
    component — those become :class:`RejectedSegment` entries tagged
    ``"merged"``.  Single-member components emit no rejections (no
    information was lost).
    """
    n = len(segments)
    if n <= 1:
        return segments, []

    deltas = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    safe_lengths = np.where(lengths > 1e-9, lengths, 1.0)
    dirs = deltas / safe_lengths[:, None]
    angles = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180.0
    centres = (segments[:, 0, :] + segments[:, 1, :]) / 2.0

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
            di = dirs[i]
            perp = np.array([-di[1], di[0]])
            perp_dist = abs(float(np.dot(centres[j] - centres[i], perp)))
            if perp_dist > perp_distance_m:
                continue
            t_i0 = float(np.dot(segments[i, 0] - centres[i], di))
            t_i1 = float(np.dot(segments[i, 1] - centres[i], di))
            t_j0 = float(np.dot(segments[j, 0] - centres[i], di))
            t_j1 = float(np.dot(segments[j, 1] - centres[i], di))
            i_lo, i_hi = min(t_i0, t_i1), max(t_i0, t_i1)
            j_lo, j_hi = min(t_j0, t_j1), max(t_j0, t_j1)
            gap = max(0.0, max(i_lo, j_lo) - min(i_hi, j_hi))
            if gap > endpoint_gap_m:
                continue
            union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    out: list[np.ndarray] = []
    rejected: list[RejectedSegment] = []
    for members in components.values():
        if len(members) == 1:
            out.append(segments[members[0]])
            continue

        # Multi-member component → merge.  Pick a canonical "kept" segment
        # (the longest) so its index becomes the survivor; everyone else
        # becomes a rejection record.  The kept segment's geometry is then
        # replaced by the merged centreline (same as legacy behaviour).
        member_lens = lengths[members]
        keeper_idx = members[int(np.argmax(member_lens))]
        for m in members:
            if m == keeper_idx:
                continue
            rejected.append(RejectedSegment(
                seg=segments[m].copy(),
                dropped_by="merged",
                length_m=float(lengths[m]),
                angle_deg=float(angles[m]),
            ))

        comp_dirs = dirs[members]
        comp_lens = lengths[members]
        ref = comp_dirs[0]
        flipped = comp_dirs * np.sign(comp_dirs @ ref)[:, None]
        mean_dir = (flipped * comp_lens[:, None]).sum(axis=0)
        mean_dir = mean_dir / max(float(np.linalg.norm(mean_dir)), 1e-9)

        comp_centres = centres[members]
        centroid = (comp_centres * comp_lens[:, None]).sum(axis=0) / comp_lens.sum()
        endpoints = segments[members].reshape(-1, 2)
        ts = (endpoints - centroid) @ mean_dir
        t_min, t_max = float(ts.min()), float(ts.max())

        merged = np.array([
            centroid + mean_dir * t_min,
            centroid + mean_dir * t_max,
        ])
        out.append(merged)

    return np.stack(out, axis=0), rejected


# ── Editor-only utilities ────────────────────────────────────────────────────

def find_near_duplicate_pairs(
    segments: np.ndarray,
    perp_distance_m: float = 0.15,
    parallel_tol_deg: float = 5.0,
    endpoint_gap_m: float = 0.50,
) -> list[tuple[int, int]]:
    """Find pairs of segments that look like duplicates of the same wall.

    Used by the editor's "Find duplicates" tool to suggest merges that the
    pipeline's regularizer was too conservative to make (e.g. two scans of
    the same wall from slightly different angles end up 10 cm apart instead
    of perfectly collinear, so the regularizer's 5 cm threshold doesn't
    trigger).

    Default thresholds are ~3× the regularizer's so the operator only sees
    pairs the regularizer *wouldn't* have merged on its own.  Returns a list
    of ``(i, j)`` index pairs with ``i < j``.  Each pair is reported once.
    """
    n = len(segments)
    if n <= 1:
        return []

    deltas = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    safe_lengths = np.where(lengths > 1e-9, lengths, 1.0)
    dirs = deltas / safe_lengths[:, None]
    angles = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180.0
    centres = (segments[:, 0, :] + segments[:, 1, :]) / 2.0

    def _angle_diff(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 180.0 - d)

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if _angle_diff(angles[i], angles[j]) > parallel_tol_deg:
                continue
            di = dirs[i]
            perp = np.array([-di[1], di[0]])
            perp_dist = abs(float(np.dot(centres[j] - centres[i], perp)))
            if perp_dist > perp_distance_m:
                continue
            t_i0 = float(np.dot(segments[i, 0] - centres[i], di))
            t_i1 = float(np.dot(segments[i, 1] - centres[i], di))
            t_j0 = float(np.dot(segments[j, 0] - centres[i], di))
            t_j1 = float(np.dot(segments[j, 1] - centres[i], di))
            i_lo, i_hi = min(t_i0, t_i1), max(t_i0, t_i1)
            j_lo, j_hi = min(t_j0, t_j1), max(t_j0, t_j1)
            gap = max(0.0, max(i_lo, j_lo) - min(i_hi, j_hi))
            if gap > endpoint_gap_m:
                continue
            pairs.append((i, j))
    return pairs


def merge_segment_group(segments: np.ndarray) -> np.ndarray:
    """Collapse N segments into a single best-fit centreline.

    Used by the editor's "Merge selected" action — operator picks N near-
    collinear walls, this returns the (2,2) centreline that replaces them.
    Length-weighted average direction, projected-extent endpoints.
    """
    if len(segments) == 0:
        raise ValueError("Cannot merge an empty group")
    if len(segments) == 1:
        return segments[0]

    deltas = segments[:, 1, :] - segments[:, 0, :]
    lengths = np.linalg.norm(deltas, axis=1)
    safe = np.where(lengths > 1e-9, lengths, 1.0)
    dirs = deltas / safe[:, None]
    centres = (segments[:, 0, :] + segments[:, 1, :]) / 2.0

    ref = dirs[0]
    flipped = dirs * np.sign(dirs @ ref)[:, None]
    mean_dir = (flipped * lengths[:, None]).sum(axis=0)
    mean_dir = mean_dir / max(float(np.linalg.norm(mean_dir)), 1e-9)

    centroid = (centres * lengths[:, None]).sum(axis=0) / lengths.sum()
    endpoints = segments.reshape(-1, 2)
    ts = (endpoints - centroid) @ mean_dir
    t_min, t_max = float(ts.min()), float(ts.max())
    return np.array([
        centroid + mean_dir * t_min,
        centroid + mean_dir * t_max,
    ])
