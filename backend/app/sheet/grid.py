"""Column grid fitting: best-fit orthogonal grid through detected columns.

Reference floor plan sheets (Stevenson `DEMO41-Fullfloor`, `1ROCK3-Fullfloor`)
carry a structural column grid: two orthogonal families of grid lines with
letter bubbles on one family and number bubbles on the other.  This module
recovers that grid from the columns the vectorize pipeline detected
(:class:`app.geometry.model.ColumnFeature`), or builds one from manual
operator entry when no columns were detected.

Fitting method (RANSAC-style over candidate rotations)
------------------------------------------------------
1. Every pair of column centres proposes a grid rotation: the pair's
   direction angle folded into ``[0°, 90°)`` (an orthogonal grid is
   invariant under 90° rotation).
2. For each candidate rotation, rotate all centres into the grid frame
   ``u = x·cosθ + y·sinθ``, ``v = −x·sinθ + y·cosθ`` and greedily cluster
   the ``u`` and ``v`` coordinates (sorted 1-D clustering, split at gaps
   larger than :attr:`GridFitParams.line_tol_m`).
3. Score = (total number of clusters, sum of squared residuals).  The true
   grid rotation needs the FEWEST lines to explain the columns; residuals
   break ties.  This parsimony objective is what makes the fit exactly
   checkable on synthetic layouts.
4. Grid lines with fewer than :attr:`GridFitParams.min_line_support`
   columns are dropped; columns explained by no surviving line in either
   family are reported as off-grid outliers (never silently snapped).

Labelling convention (matches the reference sheets):
- Lines of constant **u** (vertical on an unrotated plan) are numbered
  ``1, 2, 3…`` in ascending ``u``.
- Lines of constant **v** (horizontal on an unrotated plan) are lettered
  ``A, B, C…`` in DESCENDING ``v`` — bubble "A" sits at the top of the
  sheet.  Letters ``I`` and ``O`` are skipped per drafting convention
  (they read as ``1`` and ``0``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..geometry.model import ColumnFeature

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GridFitParams:
    min_columns: int = 4
    """Below this many detected columns a fit is not attempted (returns None)."""

    line_tol_m: float = 0.20
    """1-D clustering gap: two rotated coordinates within this distance are
    considered on the same grid line.  Real column centre jitter (detection
    noise + formwork tolerance) is a few centimetres; grid spacing is metres."""

    min_line_support: int = 2
    """Grid lines must pass through at least this many columns.  A singleton
    'line' is an off-grid column, not structure."""

    angle_dedup_deg: float = 0.5
    """Candidate rotations closer than this are considered the same."""


@dataclass
class GridLine:
    """One grid line: a line of constant ``u`` or constant ``v``."""
    label: str
    offset_m: float          # the constant coordinate value in the grid frame
    support: int             # number of columns this line passes through


@dataclass
class ColumnGrid:
    """A fitted (or manually entered) orthogonal column grid.

    Grid frame: ``u = x·cosθ + y·sinθ``, ``v = −x·sinθ + y·cosθ`` where
    ``θ = rotation_rad``.  ``u_lines[k]`` is the line ``u = offset_m``
    (runs along the v direction); ``v_lines[k]`` is ``v = offset_m``.
    """
    rotation_rad: float
    u_lines: list[GridLine] = field(default_factory=list)   # numbered
    v_lines: list[GridLine] = field(default_factory=list)   # lettered
    rmse_m: float = 0.0
    n_columns: int = 0
    n_outliers: int = 0
    source: str = "fitted"    # "fitted" | "manual"

    def to_json_dict(self) -> dict:
        return {
            "rotation_rad": float(self.rotation_rad),
            "rotation_deg": float(np.degrees(self.rotation_rad)),
            "u_lines": [
                {"label": g.label, "offset_m": float(g.offset_m), "support": g.support}
                for g in self.u_lines
            ],
            "v_lines": [
                {"label": g.label, "offset_m": float(g.offset_m), "support": g.support}
                for g in self.v_lines
            ],
            "rmse_m": float(self.rmse_m),
            "n_columns": int(self.n_columns),
            "n_outliers": int(self.n_outliers),
            "source": self.source,
        }


# ── Labels ────────────────────────────────────────────────────────────────────

_LETTERS = [c for c in "ABCDEFGHJKLMNPQRSTUVWXYZ"]   # I and O skipped


def letter_label(index: int) -> str:
    """0 → 'A', 1 → 'B', …, skipping I/O; wraps to 'AA', 'AB', … after 24."""
    n = len(_LETTERS)
    if index < n:
        return _LETTERS[index]
    return _LETTERS[index // n - 1] + _LETTERS[index % n]


# ── Fitting ───────────────────────────────────────────────────────────────────

def _cluster_1d(values: np.ndarray, tol: float) -> list[np.ndarray]:
    """Greedy sorted clustering: split wherever the gap exceeds ``tol``.

    Returns index arrays into ``values`` (one per cluster).
    """
    order = np.argsort(values)
    clusters: list[list[int]] = [[int(order[0])]]
    for idx in order[1:]:
        if values[idx] - values[clusters[-1][-1]] > tol:
            clusters.append([int(idx)])
        else:
            clusters[-1].append(int(idx))
    return [np.asarray(c, dtype=int) for c in clusters]


def _rotate_to_grid(centres: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(theta), np.sin(theta)
    u = centres[:, 0] * c + centres[:, 1] * s
    v = -centres[:, 0] * s + centres[:, 1] * c
    return u, v


def _score_rotation(
    centres: np.ndarray, theta: float, tol: float,
) -> tuple[int, float]:
    """(total number of grid lines, sum of squared residuals) — lower is better."""
    u, v = _rotate_to_grid(centres, theta)
    n_lines = 0
    ssr = 0.0
    for values in (u, v):
        for cluster in _cluster_1d(values, tol):
            n_lines += 1
            ssr += float(np.sum((values[cluster] - values[cluster].mean()) ** 2))
    return n_lines, ssr


def fit_column_grid(
    columns: Sequence[ColumnFeature],
    params: GridFitParams | None = None,
) -> Optional[ColumnGrid]:
    """Fit an orthogonal grid through detected column centres.

    Returns ``None`` when there are too few columns to support a fit —
    the sheet renderer then falls back to manual grid entry (or no grid).
    Off-grid columns are counted in ``n_outliers``, never snapped.
    """
    if params is None:
        params = GridFitParams()
    centres = np.array([c.centre for c in columns], dtype=float).reshape(-1, 2)
    if len(centres) < params.min_columns:
        return None

    # 1. Candidate rotations from every pair direction, folded into [0, 90°).
    quarter = np.pi / 2.0
    candidates: list[float] = []
    n = len(centres)
    for i in range(n):
        for j in range(i + 1, n):
            d = centres[j] - centres[i]
            if float(np.hypot(d[0], d[1])) < 1e-9:
                continue
            candidates.append(float(np.arctan2(d[1], d[0])) % quarter)
    if not candidates:
        return None
    candidates.sort()
    dedup_tol = np.radians(params.angle_dedup_deg)
    unique: list[float] = []
    for a in candidates:
        if not unique or a - unique[-1] > dedup_tol:
            unique.append(a)

    # 2/3. Pick the rotation that explains the columns with the fewest lines.
    best_theta = unique[0]
    best_score: tuple[int, float] | None = None
    for theta in unique:
        score = _score_rotation(centres, theta, params.line_tol_m)
        if best_score is None or score < best_score:
            best_score = score
            best_theta = theta

    # 4. Build grid lines at the best rotation; drop unsupported lines.
    u, v = _rotate_to_grid(centres, best_theta)
    u_clusters = _cluster_1d(u, params.line_tol_m)
    v_clusters = _cluster_1d(v, params.line_tol_m)

    on_u = np.zeros(n, dtype=bool)
    on_v = np.zeros(n, dtype=bool)
    sq_res = np.zeros(n, dtype=float)

    u_lines: list[GridLine] = []
    for cluster in u_clusters:
        if len(cluster) < params.min_line_support:
            continue
        mean = float(u[cluster].mean())
        u_lines.append(GridLine(label="", offset_m=mean, support=len(cluster)))
        on_u[cluster] = True
        sq_res[cluster] += (u[cluster] - mean) ** 2
    v_lines: list[GridLine] = []
    for cluster in v_clusters:
        if len(cluster) < params.min_line_support:
            continue
        mean = float(v[cluster].mean())
        v_lines.append(GridLine(label="", offset_m=mean, support=len(cluster)))
        on_v[cluster] = True
        sq_res[cluster] += (v[cluster] - mean) ** 2

    if not u_lines and not v_lines:
        return None

    # Labels: numbers ascend with u; letters descend with v (A at top).
    u_lines.sort(key=lambda g: g.offset_m)
    for k, line in enumerate(u_lines):
        line.label = str(k + 1)
    v_lines.sort(key=lambda g: -g.offset_m)
    for k, line in enumerate(v_lines):
        line.label = letter_label(k)

    on_grid = on_u | on_v
    n_outliers = int(np.count_nonzero(~on_grid))
    rmse = (
        float(np.sqrt(np.mean(sq_res[on_grid]))) if np.any(on_grid) else 0.0
    )

    return ColumnGrid(
        rotation_rad=float(best_theta),
        u_lines=u_lines,
        v_lines=v_lines,
        rmse_m=rmse,
        n_columns=n,
        n_outliers=n_outliers,
        source="fitted",
    )


# ── Manual fallback ───────────────────────────────────────────────────────────

def manual_grid(
    u_offsets_m: Sequence[float],
    v_offsets_m: Sequence[float],
    rotation_deg: float = 0.0,
) -> ColumnGrid:
    """Build a grid from operator-entered line offsets (no columns needed).

    Same labelling convention as the fitter: numbers ascend with ``u``,
    letters descend with ``v``.
    """
    u_sorted = sorted(float(x) for x in u_offsets_m)
    v_sorted = sorted((float(x) for x in v_offsets_m), reverse=True)
    return ColumnGrid(
        rotation_rad=float(np.radians(rotation_deg)),
        u_lines=[
            GridLine(label=str(k + 1), offset_m=off, support=0)
            for k, off in enumerate(u_sorted)
        ],
        v_lines=[
            GridLine(label=letter_label(k), offset_m=off, support=0)
            for k, off in enumerate(v_sorted)
        ],
        rmse_m=0.0,
        n_columns=0,
        n_outliers=0,
        source="manual",
    )


# ── World-space line segments (for the renderer) ─────────────────────────────

def grid_line_segments_world(
    grid: ColumnGrid,
    bounds_world: tuple[float, float, float, float],
    extension_m: float = 1.0,
) -> list[dict]:
    """Grid lines as world-coordinate segments spanning the plan bounds.

    Each returned dict: ``{"label", "family" ("u"|"v"), "p0", "p1"}`` where
    ``p0``/``p1`` are ``(x, y)`` world coordinates.  ``p1`` is the END the
    renderer puts the bubble at: the max-``v`` end for u-lines (top of the
    sheet) and the min-``u`` end for v-lines (left of the sheet).
    """
    min_x, min_y, max_x, max_y = bounds_world
    corners = np.array(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
        dtype=float,
    )
    u_c, v_c = _rotate_to_grid(corners, grid.rotation_rad)
    u_lo, u_hi = float(u_c.min()) - extension_m, float(u_c.max()) + extension_m
    v_lo, v_hi = float(v_c.min()) - extension_m, float(v_c.max()) + extension_m

    c, s = np.cos(grid.rotation_rad), np.sin(grid.rotation_rad)

    def to_world(u: float, v: float) -> tuple[float, float]:
        return (u * c - v * s, u * s + v * c)

    out: list[dict] = []
    for line in grid.u_lines:
        out.append({
            "label": line.label,
            "family": "u",
            "p0": to_world(line.offset_m, v_lo),
            "p1": to_world(line.offset_m, v_hi),   # bubble end: top
        })
    for line in grid.v_lines:
        out.append({
            "label": line.label,
            "family": "v",
            "p0": to_world(u_hi, line.offset_m),
            "p1": to_world(u_lo, line.offset_m),   # bubble end: left
        })
    return out


# ── Bbox-aligned fallback (no detected columns) ───────────────────────────────

def _nice_spacing(extent_m: float, target_lines: int = 5) -> tuple[float, int]:
    """Pick an even grid spacing that yields roughly ``target_lines`` ticks."""
    if extent_m <= 1e-6:
        return 1.0, 2
    raw = extent_m / max(target_lines - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(max(raw, 1e-6)))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        spacing = mult * magnitude
        n = max(2, int(math.floor(extent_m / spacing)) + 1)
        if n <= target_lines + 1:
            return spacing, n
    return raw, target_lines


def bbox_aligned_grid(
    bounds_rotated: tuple[float, float, float, float],
    rotation_rad: float = 0.0,
    target_lines: int = 5,
) -> ColumnGrid:
    """Even-spaced tick grid aligned to the plan bbox.

    Used when column detection finds too few posts — every sheet still gets
    A–H / 1–5 edge ticks (Stevenson style) instead of a grid-less drawing.
    """
    min_u, min_v, max_u, max_v = bounds_rotated
    width_u = max(max_u - min_u, 1e-6)
    height_v = max(max_v - min_v, 1e-6)

    u_spacing, _ = _nice_spacing(width_u, target_lines)
    v_spacing, _ = _nice_spacing(height_v, target_lines)

    u_offsets: list[float] = []
    u = min_u
    while u <= max_u + 1e-6:
        u_offsets.append(float(u))
        u += u_spacing
    if u_offsets[-1] < max_u - u_spacing * 0.2:
        u_offsets.append(float(max_u))

    v_offsets: list[float] = []
    v = max_v
    while v >= min_v - 1e-6:
        v_offsets.append(float(v))
        v -= v_spacing
    if v_offsets and v_offsets[-1] > min_v + v_spacing * 0.2:
        v_offsets.append(float(min_v))

    return manual_grid(
        u_offsets_m=u_offsets,
        v_offsets_m=v_offsets,
        rotation_deg=float(np.degrees(rotation_rad)),
    )
