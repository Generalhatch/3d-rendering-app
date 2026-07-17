"""Classical opening (door / wall-gap) detection.

How it works
------------
For every kept wall segment from the regularizer, we walk along the wall's
direction in fine steps and sample the cleaned raster perpendicular to the
wall.  Pixels in the raster that lie within a thin band centred on the wall
line are "wall support".

A run of consecutive samples with **no wall support** that

  - is between :attr:`OpeningParams.min_width_m` and :attr:`max_width_m`
    long (door-shaped), AND
  - sits ≥ :attr:`min_distance_from_endpoint_m` away from either end of the
    parent wall (so corners / wall terminations aren't mistaken for doors),

becomes an opening.  We emit one segment per gap, oriented along the parent
wall, with its endpoints at the gap's start and end in world coordinates.

This is intentionally conservative — we'd rather miss an opening than emit a
false one.  Operators can always draw missing openings manually in the editor
(``D`` for wall, ``O`` for opening).

Why this works on real scans
----------------------------
Doors that were *open* during the scan leave a clean rectangular gap in the
wall's raster signature.  Doors that were *closed* still show as gaps when
sliced at shoulder height (1.6 m default) because most door slabs end at
~2.0 m — at 1.6 m we see the door slab itself, which is geometrically thinner
than the wall it lives in.  In the cleaned raster (post morphology + speckle
removal) the door slab usually doesn't extend across the wall's thickness,
so the perpendicular search comes up empty.

Future extensions
-----------------
- Distinguish doors from windows by checking the *other* slab elevations
  (multi-elevation slicing already produces them — we just need to retain
   the per-elevation rasters).  Windows show wall support at chest height
  but a gap at shoulder height; doors show a gap at both.
- Detect openings the operator drew manually by parsing
  ``edits.log.jsonl`` for ``op=add`` events tagged ``layer=openings``.
- Promote opening detection from "per wall" to "per axis" so doors at wall
  junctions (an opening that spans two walls meeting at a corner) get
  captured as one segment instead of two.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .slicer import RasterAffine


@dataclass
class OpeningParams:
    """Tunable thresholds for the door / opening detector.

    Defaults match conventional interior door geometry — adjust for industrial
    or commercial spaces (taller, wider openings) via the API.
    """
    min_width_m: float = 0.70
    """Lower bound on door width.  US/ANSI residential interior doors are
    762 mm (30 in) at the narrow end; bathroom doors are sometimes 610 mm."""

    max_width_m: float = 1.20
    """Upper bound on door width.  A 1.2 m gap captures double-leaf doors
    and most commercial entries without misfiring on short wall stubs."""

    sample_step_m: float = 0.02
    """Step size along the wall when sampling for gaps (2 cm = 2× raster
    resolution at the default 1 cm/px → stable signal, fast detection)."""

    perpendicular_band_m: float = 0.08
    """Half-width of the perpendicular search band around the wall line.
    8 cm comfortably brackets typical interior walls (10–15 cm thick) while
    staying tight enough that doors thinner than the wall register as gaps."""

    min_distance_from_endpoint_m: float = 0.30
    """A gap closer than this to either end of the parent wall is treated as
    a wall termination (corner / end of run), not a door.  30 cm trims most
    false positives without losing doors snug against corners."""

    min_support_pixels: int = 2
    """Minimum foreground pixel count inside the perpendicular band to call
    a sample "supported".  2 px filters speckle that survived the OPEN."""

    extend_into_wall_m: float = 0.05
    """Doors snap their endpoints 5 cm into the parent wall on each side so
    the opening visually attaches to the wall in the editor + DXF instead of
    floating slightly off."""


@dataclass
class DetectedOpening:
    """One detected opening, parented to the wall it sits on."""
    seg: np.ndarray              # (2, 2) world coords
    parent_wall_index: int
    width_m: float
    centre_m: tuple[float, float]


def detect_openings(
    cleaned_raster: np.ndarray,
    walls_world: np.ndarray,
    affine: RasterAffine,
    params: OpeningParams | None = None,
) -> list[DetectedOpening]:
    """Detect door-shaped gaps in every kept wall.

    Parameters
    ----------
    cleaned_raster : np.ndarray
        ``(H, W)`` uint8 image — the preprocessed slice in *internal raster
        convention* (row 0 = low world Y).  Pass the array from
        :func:`preprocess.preprocess`, **not** the Y-flipped display version
        that gets written to disk.
    walls_world : np.ndarray
        ``(N, 2, 2)`` array of wall segments in world coordinates (metres).
    affine : RasterAffine
        Pixel ↔ world conversion for ``cleaned_raster``.
    params : OpeningParams, optional
        Thresholds.  Defaults to standard interior-door geometry.

    Returns
    -------
    list[DetectedOpening]
        One per detected door / wall-gap.  Empty if ``walls_world`` is empty
        or every wall is below the minimum width that could even contain a
        door (so any "gap" would actually be wall termination).
    """
    if params is None:
        params = OpeningParams()
    if len(walls_world) == 0:
        return []

    h, w = cleaned_raster.shape
    res = affine.resolution_m_per_px
    perp_band_px = max(1, int(round(params.perpendicular_band_m / res)))

    out: list[DetectedOpening] = []

    for wall_idx, wall in enumerate(walls_world):
        p0 = wall[0]
        p1 = wall[1]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        wall_len = float(np.hypot(dx, dy))
        # Wall must be at least 2× (min door width + 2× endpoint buffer) to
        # have room for any opening — anything shorter is automatically
        # skipped to save work and avoid false positives.
        min_wall_for_openings = (
            params.min_width_m + 2 * params.min_distance_from_endpoint_m
        )
        if wall_len < min_wall_for_openings:
            continue

        # Unit direction + perpendicular (rotate 90°).
        ux = dx / wall_len
        uy = dy / wall_len
        nx = -uy
        ny = ux

        # Walk along the wall in world-step ``sample_step_m``, sample each.
        n_samples = int(np.floor(wall_len / params.sample_step_m))
        supported = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            t = i * params.sample_step_m
            sx = p0[0] + ux * t
            sy = p0[1] + uy * t

            # Build a small rectangle window centred at (sx, sy), oriented
            # perpendicular to the wall, half-width perp_band_px.  Test the
            # raster only inside the rectangle by sampling per-pixel along
            # the perpendicular direction.
            count = 0
            for k in range(-perp_band_px, perp_band_px + 1):
                px_world_x = sx + nx * (k * res)
                px_world_y = sy + ny * (k * res)
                # Pixel-centre convention: idx = (world - origin) / res - 0.5.
                col = int(round((px_world_x - affine.origin_x) / res - 0.5))
                row = int(round((px_world_y - affine.origin_y) / res - 0.5))
                if 0 <= col < w and 0 <= row < h and cleaned_raster[row, col] > 0:
                    count += 1
                    if count >= params.min_support_pixels:
                        break
            supported[i] = count >= params.min_support_pixels

        # Find runs of False (unsupported = wall absent at this sample).
        # Apply the endpoint buffer by trimming the supported array on each
        # side — gaps that fall inside the trimmed window are eligible.
        buffer_samples = int(round(
            params.min_distance_from_endpoint_m / params.sample_step_m
        ))
        if buffer_samples * 2 >= n_samples:
            continue
        eligible = supported[buffer_samples : n_samples - buffer_samples]
        eligible_offset = buffer_samples

        gaps = _find_gap_runs(eligible)
        for gap_start, gap_end in gaps:
            gap_len_samples = gap_end - gap_start + 1
            gap_len_m = gap_len_samples * params.sample_step_m
            if gap_len_m < params.min_width_m or gap_len_m > params.max_width_m:
                continue

            # Convert sample indices back to world coords along the parent
            # wall, then extend each end slightly into the wall body so the
            # opening visually attaches.
            t_start = (eligible_offset + gap_start) * params.sample_step_m
            t_end = (eligible_offset + gap_end + 1) * params.sample_step_m

            extend = params.extend_into_wall_m
            t_start = max(0.0, t_start - extend)
            t_end = min(wall_len, t_end + extend)

            sx1 = p0[0] + ux * t_start
            sy1 = p0[1] + uy * t_start
            sx2 = p0[0] + ux * t_end
            sy2 = p0[1] + uy * t_end

            cx = (sx1 + sx2) / 2.0
            cy = (sy1 + sy2) / 2.0

            out.append(DetectedOpening(
                seg=np.array([[sx1, sy1], [sx2, sy2]]),
                parent_wall_index=wall_idx,
                width_m=float(t_end - t_start),
                centre_m=(float(cx), float(cy)),
            ))

    # Deduplicate openings that two parallel walls (e.g. door frames) both
    # claim — keep the centre nearest to the average of the two midpoints.
    return _deduplicate_openings(out)


def _find_gap_runs(supported: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, end_inclusive)`` index pairs for every run of False."""
    runs: list[tuple[int, int]] = []
    n = len(supported)
    i = 0
    while i < n:
        if supported[i]:
            i += 1
            continue
        j = i
        while j < n and not supported[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _deduplicate_openings(
    openings: list[DetectedOpening],
    centre_tolerance_m: float = 0.30,
) -> list[DetectedOpening]:
    """Drop openings whose centres land near another opening's centre.

    A door at a wall junction is often detected on both walls — we keep just
    the wider of the two (longer-actually-empty), discarding the duplicate.
    """
    if len(openings) <= 1:
        return openings
    keep_flags = [True] * len(openings)
    for i, a in enumerate(openings):
        if not keep_flags[i]:
            continue
        for j in range(i + 1, len(openings)):
            if not keep_flags[j]:
                continue
            b = openings[j]
            d = np.hypot(a.centre_m[0] - b.centre_m[0], a.centre_m[1] - b.centre_m[1])
            if d <= centre_tolerance_m:
                # Keep the wider of the two — usually the more legit detection.
                if a.width_m >= b.width_m:
                    keep_flags[j] = False
                else:
                    keep_flags[i] = False
                    break
    return [o for o, k in zip(openings, keep_flags) if k]


def render_openings_overlay(
    base_image: np.ndarray,
    walls_px: np.ndarray,
    openings_px: np.ndarray,
) -> np.ndarray:
    """Render walls (red) + openings (yellow) over the cleaned raster.

    Used by the pipeline to write ``overlay_openings.png`` — a quick visual
    sanity-check that the door detector found the right gaps.
    """
    if base_image.ndim == 2:
        bgr = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = base_image.copy()
    for x1, y1, x2, y2 in walls_px:
        cv2.line(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 220), 2)
    for x1, y1, x2, y2 in openings_px:
        cv2.line(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 240), 4)
    return bgr
