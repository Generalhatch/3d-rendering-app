"""Hand-computed reference floors for geometry + measurement tests.

Every expected number in this module was worked out by hand on paper BEFORE
any implementation code existed.  Do not "fix" these constants to match the
code — if the code disagrees, the code is wrong.

Two-suite floor (``two_suite_floor``)
=====================================

Plan view (wall CENTERLINES, metres)::

    (0,10) ┌──────────┬────────────────┐ (20,10)
           │ suite_a  │    suite_b     │
           │  8 x 10  │    12 x 10     │
    (0,0)  └──────────┴────────────────┘ (20,0)
                     x=8

  - Exterior walls: thickness 0.30 m (faces at centerline ± 0.15).
  - Demising wall at x=8: thickness 0.20 m.
  - Room rings are drawn on wall centerlines (boundary_basis=centerline).
  - Envelope = outside face of exterior walls:
    (-0.15, -0.15) .. (20.15, 10.15)  →  20.3 x 10.3.

BOMA Office 2024 usable (exterior → dominant portion = interior finish,
i.e. inward 0.15; demising → centerline):

  suite_a: (8 - 0.15) x (9.85 - 0.15) = 7.85 x 9.7  = 76.145 m²
  suite_b: (19.85 - 8) x 9.7          = 11.85 x 9.7 = 114.945 m²
  floor usable                                       = 191.090 m²
  No common areas → rentable == usable, load factor = 1.0.

REBNY (exterior → OUTSIDE face, i.e. outward 0.15; demising → centerline):

  suite_a: (8 + 0.15) x (10.3)        = 8.15 x 10.3  = 83.945 m²
  suite_b: (20.15 - 8) x 10.3         = 12.15 x 10.3 = 125.145 m²
  floor rentable                                      = 209.090 m²
  (equals the envelope area — the demising wall is split half/half)

Gross area = envelope = 20.3 x 10.3 = 209.09 m².

Dominant-portion variant (``two_suite_floor(dominant_inward_offset_m=0.05)``):
exterior walls are >50% glass, dominant portion 0.05 m inward of centerline
instead of the 0.15 m finish face:

  suite_a: (8 - 0.05) x (10 - 0.10) = 7.95 x 9.9  = 78.705 m²
  suite_b: (11.95) x 9.9            =              118.305 m²

Corridor floor (``corridor_floor``)
===================================

Same shell, demising walls at x=8 AND x=12 (0.20 m each); the middle strip
is a common-area corridor::

    (0,10) ┌──────────┬──────┬──────────┐ (20,10)
           │ suite_a  │ corr │ suite_b  │
    (0,0)  └──────────┴──────┴──────────┘ (20,0)
                     x=8    x=12

BOMA usable:
  suite_a: (8 - 0.15) x 9.7  = 7.85 x 9.7 = 76.145 m²
  suite_b: (19.85 - 12) x 9.7 = 7.85 x 9.7 = 76.145 m²
  floor usable                             = 152.290 m²
  corridor (common, demising→centerline both sides): 4.0 x 9.7 = 38.800 m²

BOMA Method A rentable (common apportioned pro-rata on usable — the two
suites are equal, so each gets half of 38.8 = 19.4):

  suite_a rentable = 76.145 + 19.4 = 95.545 m²
  suite_b rentable = 76.145 + 19.4 = 95.545 m²
  floor rentable   = 152.29 + 38.8 = 191.090 m²
  R/U ratio        = 191.09 / 152.29

With a 2x2 m elevator shaft penetration at (9,4)..(11,6) (inside the
corridor): BOMA excludes major vertical penetrations →
  corridor common = 38.8 - 4.0 = 34.800 m²
  suite rentable  = 76.145 + 17.4 = 93.545 m²
  floor rentable  = 152.29 + 34.8 = 187.090 m²
REBNY includes the shaft → floor rentable stays 209.09 m².

REBNY on the corridor floor:
  suite_a measured: (8 + 0.15) x 10.3 = 83.945 m²
  suite_b measured: (20.15 - 12) x 10.3 = 83.945 m²
  corridor measured: 4.0 x 10.3 = 41.200 m²
  floor rentable = 209.090 m²
  suite rentable = 83.945 + 41.2/2 = 104.545 m²
"""
from __future__ import annotations

import numpy as np

from app.geometry.model import (
    BoundaryBasis,
    FloorGeometry,
    Penetration,
    Room,
    Wall,
    WallClass,
    WallFaceRef,
)

# ── Hand-computed constants (see module docstring) ───────────────────────────

EXTERIOR_T = 0.30
DEMISING_T = 0.20

TWO_SUITE_A_USABLE_BOMA = 76.145
TWO_SUITE_B_USABLE_BOMA = 114.945
TWO_SUITE_FLOOR_USABLE_BOMA = 191.090

TWO_SUITE_A_RENTABLE_REBNY = 83.945
TWO_SUITE_B_RENTABLE_REBNY = 125.145
TWO_SUITE_FLOOR_RENTABLE_REBNY = 209.090

GROSS_AREA = 209.09  # 20.3 x 10.3

DOMINANT_A_USABLE = 78.705
DOMINANT_B_USABLE = 118.305

CORRIDOR_SUITE_USABLE = 76.145
CORRIDOR_COMMON = 38.800
CORRIDOR_SUITE_RENTABLE = 95.545
CORRIDOR_FLOOR_USABLE = 152.290
CORRIDOR_FLOOR_RENTABLE = 191.090

SHAFT_COMMON = 34.800
SHAFT_SUITE_RENTABLE = 93.545
SHAFT_FLOOR_RENTABLE = 187.090

CORRIDOR_SUITE_RENTABLE_REBNY = 104.545
CORRIDOR_COMMON_REBNY = 41.200


def _ring(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """CCW rectangle ring (no closing duplicate vertex)."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _wall(wid: str, p: tuple, q: tuple, t: float, cls: WallClass,
          dominant_inward_offset_m: float | None = None) -> Wall:
    return Wall(
        id=wid,
        centerline=np.array([p, q], dtype=np.float64),
        thickness_m=t,
        wall_class=cls,
        dominant_inward_offset_m=dominant_inward_offset_m,
    )


def two_suite_floor(dominant_inward_offset_m: float | None = None) -> FloorGeometry:
    """20x10 m shell, one demising wall at x=8 → suite_a (8x10) + suite_b (12x10)."""
    d = dominant_inward_offset_m
    walls = [
        _wall("w_bottom", (0, 0), (20, 0), EXTERIOR_T, WallClass.EXTERIOR, d),
        _wall("w_right", (20, 0), (20, 10), EXTERIOR_T, WallClass.EXTERIOR, d),
        _wall("w_top", (20, 10), (0, 10), EXTERIOR_T, WallClass.EXTERIOR, d),
        _wall("w_left", (0, 10), (0, 0), EXTERIOR_T, WallClass.EXTERIOR, d),
        _wall("w_demising", (8, 0), (8, 10), DEMISING_T, WallClass.DEMISING),
    ]
    rooms = [
        Room(
            id="suite_a",
            boundary=_ring(0, 0, 8, 10),
            wall_refs=[
                WallFaceRef("w_bottom"), WallFaceRef("w_demising"),
                WallFaceRef("w_top"), WallFaceRef("w_left"),
            ],
            boundary_basis=BoundaryBasis.CENTERLINE,
            label="Suite A",
        ),
        Room(
            id="suite_b",
            boundary=_ring(8, 0, 20, 10),
            wall_refs=[
                WallFaceRef("w_bottom"), WallFaceRef("w_right"),
                WallFaceRef("w_top"), WallFaceRef("w_demising"),
            ],
            boundary_basis=BoundaryBasis.CENTERLINE,
            label="Suite B",
        ),
    ]
    return FloorGeometry(
        floor_id="two-suite-reference",
        walls=walls,
        rooms=rooms,
        envelope=_ring(-0.15, -0.15, 20.15, 10.15),
    )


def corridor_floor(with_shaft: bool = False) -> FloorGeometry:
    """20x10 m shell, demising walls at x=8 and x=12, common corridor between."""
    walls = [
        _wall("w_bottom", (0, 0), (20, 0), EXTERIOR_T, WallClass.EXTERIOR),
        _wall("w_right", (20, 0), (20, 10), EXTERIOR_T, WallClass.EXTERIOR),
        _wall("w_top", (20, 10), (0, 10), EXTERIOR_T, WallClass.EXTERIOR),
        _wall("w_left", (0, 10), (0, 0), EXTERIOR_T, WallClass.EXTERIOR),
        _wall("w_dem_1", (8, 0), (8, 10), DEMISING_T, WallClass.DEMISING),
        _wall("w_dem_2", (12, 0), (12, 10), DEMISING_T, WallClass.DEMISING),
    ]
    rooms = [
        Room(
            id="suite_a",
            boundary=_ring(0, 0, 8, 10),
            wall_refs=[
                WallFaceRef("w_bottom"), WallFaceRef("w_dem_1"),
                WallFaceRef("w_top"), WallFaceRef("w_left"),
            ],
            boundary_basis=BoundaryBasis.CENTERLINE,
            label="Suite A",
        ),
        Room(
            id="corridor",
            boundary=_ring(8, 0, 12, 10),
            wall_refs=[
                WallFaceRef("w_bottom"), WallFaceRef("w_dem_2"),
                WallFaceRef("w_top"), WallFaceRef("w_dem_1"),
            ],
            boundary_basis=BoundaryBasis.CENTERLINE,
            label="Corridor",
            category="hallway",
            is_common=True,
        ),
        Room(
            id="suite_b",
            boundary=_ring(12, 0, 20, 10),
            wall_refs=[
                WallFaceRef("w_bottom"), WallFaceRef("w_right"),
                WallFaceRef("w_top"), WallFaceRef("w_dem_2"),
            ],
            boundary_basis=BoundaryBasis.CENTERLINE,
            label="Suite B",
        ),
    ]
    penetrations = []
    if with_shaft:
        penetrations.append(Penetration(
            id="shaft-1",
            polygon=_ring(9, 4, 11, 6),
            kind="elevator_shaft",
        ))
    return FloorGeometry(
        floor_id="corridor-reference",
        walls=walls,
        rooms=rooms,
        penetrations=penetrations,
        envelope=_ring(-0.15, -0.15, 20.15, 10.15),
    )
