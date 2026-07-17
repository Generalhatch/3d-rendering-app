"""Adaptive slice-band selection from the scan's height histogram.

The pipeline's default vertical bands assume a typical commercial interior
with ≥ 2.5 m of floor-to-ceiling clearance:

  - shoulder-height slice at floor + 1.60 m,
  - ceiling gate band at floor + 1.90–2.30 m,
  - density-slicer wall band at floor + 0.30–2.20 m.

Low-clearance spaces break those assumptions: in a 2.1 m data-center
aisle or an old residential basement, the default ceiling band straddles
(or sits entirely above) the actual ceiling slab, so the ceiling gate sees
nothing and silently falls back — or worse, the band clips ceiling returns
and the "wall" mask fills with ceiling noise.

This module measures the actual clearance from the height histogram (the
ceiling slab is the densest horizontal band above head height) and scales
the bands down to fit, preserving the defaults whenever the room is tall
enough for them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Keep the band tops this far below the detected ceiling slab so ceiling
# returns / recessed fixtures never leak into the wall mask.
CEILING_MARGIN_M = 0.15

# A slice closer than this to the ceiling would catch door headers and
# soffits; keep the shoulder slice at least this far below the ceiling.
SLICE_HEADROOM_M = 0.60

# Hard floors for the scaled bands — below these there is no usable wall
# band at all and adapting further would be meaningless.
MIN_BAND_HI_M = 1.00
MIN_SLICE_OFFSET_M = 0.80

# Ceiling search window (heights above floor, in metres).  Below 1.2 m is
# furniture; a ceiling further than 8 m up is outside the interior regime
# these bands are for.
_CEILING_SEARCH_LO_M = 1.2
_CEILING_SEARCH_HI_M = 8.0


@dataclass
class SliceBandPlan:
    """The vertical bands one vectorize run should use, in metres above floor.

    ``reason`` is one of:

    - ``"default"``     — clearance is comfortable, defaults kept.
    - ``"low_ceiling"`` — clearance measured below the default band top;
      bands scaled down (``adjusted`` is True).
    - ``"no_ceiling"``  — no ceiling slab found in the histogram (open-top
      scan, missing ceiling coverage); defaults kept but the caller should
      surface a warning because the ceiling gate may misfire.
    """
    clearance_m: float | None
    slice_offset_m: float
    ceiling_band_lo_m: float
    ceiling_band_hi_m: float
    density_band_hi_m: float
    adjusted: bool
    reason: str


def estimate_ceiling_clearance(
    pts: np.ndarray,
    floor_z: float,
    axis_idx: int = 2,
    bin_m: float = 0.05,
) -> float | None:
    """Floor-to-ceiling clearance from the height histogram, or None.

    The ceiling slab is a dense horizontal band: the most-populated height
    bin above head height, provided it stands out from the wall-return
    noise floor.  Returns the bin-centre height above the floor.
    """
    if len(pts) < 1_000:
        return None
    heights = np.asarray(pts)[:, axis_idx] - floor_z
    window = heights[
        (heights >= _CEILING_SEARCH_LO_M) & (heights <= _CEILING_SEARCH_HI_M)
    ]
    if len(window) < 500:
        return None

    n_bins = int(np.ceil((_CEILING_SEARCH_HI_M - _CEILING_SEARCH_LO_M) / bin_m))
    counts, edges = np.histogram(
        window, bins=n_bins, range=(_CEILING_SEARCH_LO_M, _CEILING_SEARCH_HI_M),
    )
    peak = int(np.argmax(counts))
    # Walls spread returns evenly across height bins; a ceiling slab
    # concentrates them.  Require the peak to clearly dominate the typical
    # OCCUPIED bin (bins above the ceiling are empty and would drag a
    # whole-histogram median to ~0, making any uniform wall band look
    # like a peak).
    occupied = counts[counts > 0]
    noise_floor = float(np.median(occupied)) * 3.0
    if counts[peak] <= noise_floor:
        return None
    return float((edges[peak] + edges[peak + 1]) / 2.0)


def choose_slice_band(
    clearance_m: float | None,
    default_slice_offset_m: float = 1.60,
    default_ceiling_lo_m: float = 1.90,
    default_ceiling_hi_m: float = 2.30,
    default_density_hi_m: float = 2.20,
) -> SliceBandPlan:
    """Scale the vertical bands to the measured clearance.

    Pure function so the decision logic is exactly testable.  The default
    band values may be operator overrides — they are treated as the
    ceiling-tall preference and only clamped downward, never raised.
    """
    if clearance_m is None:
        return SliceBandPlan(
            clearance_m=None,
            slice_offset_m=default_slice_offset_m,
            ceiling_band_lo_m=default_ceiling_lo_m,
            ceiling_band_hi_m=default_ceiling_hi_m,
            density_band_hi_m=default_density_hi_m,
            adjusted=False,
            reason="no_ceiling",
        )

    usable_top = clearance_m - CEILING_MARGIN_M
    if usable_top >= default_ceiling_hi_m:
        return SliceBandPlan(
            clearance_m=clearance_m,
            slice_offset_m=default_slice_offset_m,
            ceiling_band_lo_m=default_ceiling_lo_m,
            ceiling_band_hi_m=default_ceiling_hi_m,
            density_band_hi_m=default_density_hi_m,
            adjusted=False,
            reason="default",
        )

    band_width = default_ceiling_hi_m - default_ceiling_lo_m
    hi = max(usable_top, MIN_BAND_HI_M)
    lo = max(hi - band_width, MIN_BAND_HI_M - band_width)
    slice_offset = max(
        min(default_slice_offset_m, clearance_m - SLICE_HEADROOM_M),
        MIN_SLICE_OFFSET_M,
    )
    density_hi = min(default_density_hi_m, usable_top)
    return SliceBandPlan(
        clearance_m=clearance_m,
        slice_offset_m=slice_offset,
        ceiling_band_lo_m=lo,
        ceiling_band_hi_m=hi,
        density_band_hi_m=density_hi,
        adjusted=True,
        reason="low_ceiling",
    )


def band_summary(plan: SliceBandPlan) -> str:
    """One-line human-readable summary for SSE progress."""
    if plan.reason == "no_ceiling":
        return "Slice bands: no ceiling found in height histogram — defaults kept"
    clearance = f"{plan.clearance_m:.2f} m clearance"
    if not plan.adjusted:
        return f"Slice bands: {clearance} — defaults kept"
    return (
        f"Slice bands adapted to {clearance}: slice at floor + "
        f"{plan.slice_offset_m:.2f} m, ceiling gate "
        f"{plan.ceiling_band_lo_m:.2f}–{plan.ceiling_band_hi_m:.2f} m"
    )
