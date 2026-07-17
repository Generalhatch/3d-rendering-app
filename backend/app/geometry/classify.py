"""Room classification shared by both pipelines (Phase 2 common areas).

The shape-metric classifier used to live in ``app.pipeline.scanplan`` as
``_classify_room``; it moved here so the vectorize pipeline can classify
its topology rooms with the SAME rules, and so the common-area definition
feeding measurement apportionment (BOMA floor service + amenity areas)
has exactly one home.

Common-area semantics
---------------------
``COMMON_CATEGORIES`` is the full set of categories that count as floor
common area when the category is TRUSTED (matched from a DXF room label
like "Corridor" / "WC" / "Lobby", or set by an operator in the editor).

``SHAPE_COMMON_CATEGORIES`` is the conservative subset used when the
category comes from shape metrics alone: corridors (strong elongation
signal) and restrooms (very small rooms).  "Open Plan / Lobby" is
deliberately excluded — from shape alone a 100 m² convex space is far
more likely a tenant open-plan suite than a building lobby, and
mis-flagging a suite as common corrupts its rent apportionment.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

# Room categories that count as building/floor common area for measurement
# apportionment (BOMA floor service + amenity areas) when the category
# source is trusted (DXF labels, operator edits).
COMMON_CATEGORIES = {"hallway", "common", "bathroom"}

# Conservative subset for shape-only classification (no labels available).
SHAPE_COMMON_CATEGORIES = {"hallway", "bathroom"}


def classify_room_shape(
    area_m2: float, eccentricity: float, solidity: float,
) -> tuple[str, str, float]:
    """Classify a room from shape metrics. Returns (label, category, confidence: 0.0–1.0).

    Uses watershed regionprops values that are already computed for each room:
      eccentricity  — 0=square, 1=very elongated line (>0.85 suggests corridor)
      solidity      — area / convex_hull_area (1=convex, <0.7=highly irregular)
      area_m2       — floor area in square metres

    Confidence reflects how strongly the metrics match the assigned category.
    A score of 0.95 means the room clearly fits; 0.50 means it is a weak match.
    """
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def _final(x: float) -> float:
        # A shape-metric heuristic can never be CERTAIN — cap below 1.0 so
        # perfect-looking metrics still read as "very confident", not "proven".
        return max(0.0, min(0.99, x))

    # ── Hallway / Corridor ────────────────────────────────────────────────────
    # Key signal: high eccentricity (elongated shape).  Area must be reasonable.
    if eccentricity > 0.85 and area_m2 < 40:
        ecc_conf = _clamp01((eccentricity - 0.85) / 0.12)   # 0 at 0.85 → 1 at 0.97
        area_conf = _clamp01(1.0 - area_m2 / 45.0)
        return "Hallway / Corridor", "hallway", _final(0.55 + ecc_conf * 0.38 + area_conf * 0.07)

    # ── Bathroom / Storage ────────────────────────────────────────────────────
    # Key signal: unusually small floor area.
    if area_m2 < 7.0:
        # Score scales from 0.55 (just-under-7 m²) up to 0.92 (≤2 m²)
        conf = _final(0.55 + (7.0 - area_m2) / 14.0)
        return "Bathroom / Storage", "bathroom", conf

    # ── Open Plan / Lobby ─────────────────────────────────────────────────────
    # Key signals: large area AND convex shape.  Both criteria needed.
    if area_m2 > 80.0 and solidity > 0.85:
        area_conf = _clamp01((area_m2 - 80.0) / 120.0)       # 0 at 80 m² → 1 at 200 m²
        sol_conf  = _clamp01((solidity - 0.85) / 0.14)        # 0 at 0.85 → 1 at 0.99
        return "Open Plan / Lobby", "common", _final(0.60 + area_conf * 0.25 + sol_conf * 0.15)

    # ── Conference Room ───────────────────────────────────────────────────────
    # Medium-large area; confidence grows with size above the threshold.
    if area_m2 > 40.0:
        conf = _final(0.55 + (area_m2 - 40.0) / 100.0)       # 0.55 at 40 m² → ~0.90 at 75 m²
        return "Conference Room", "office", conf

    # ── Irregular Space ───────────────────────────────────────────────────────
    # Low solidity indicates a complex, non-convex polygon.
    if solidity < 0.70:
        conf = _final(0.50 + (0.70 - solidity) / 0.60)        # 0.50 at 0.70 → 0.83 at 0.40
        return "Irregular Space", "unknown", conf

    # ── Office / Meeting ─────────────────────────────────────────────────────
    # Default: regular shape, typical office size (8–30 m²).
    area_fit = 1.0 - abs(area_m2 - 18.0) / 25.0              # peak at 18 m²
    sol_fit  = _clamp01((solidity - 0.60) / 0.35)
    conf = _final(0.50 + max(0.0, area_fit) * 0.25 + sol_fit * 0.15)
    return "Office / Meeting", "office", conf


def polygon_shape_metrics(ring: np.ndarray) -> tuple[float, float, float]:
    """(area_m2, eccentricity, solidity) for a polygon ring.

    Eccentricity is approximated from the minimum rotated rectangle's
    aspect ratio (rotation-invariant, matching the regionprops intuition:
    0 = square, → 1 = very elongated).  Solidity = area / convex-hull area.
    """
    poly = ShapelyPolygon(np.asarray(ring, dtype=float))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return 0.0, 0.0, 1.0

    # Shapely's oriented_envelope emits spurious divide-by-zero RuntimeWarnings
    # for axis-aligned inputs; the result is still correct.
    with np.errstate(divide="ignore", invalid="ignore"):
        mrr = poly.minimum_rotated_rectangle
    coords = np.asarray(mrr.exterior.coords)
    e0 = float(np.linalg.norm(coords[1] - coords[0]))
    e1 = float(np.linalg.norm(coords[2] - coords[1]))
    long_side = max(e0, e1)
    short_side = max(min(e0, e1), 1e-9)
    eccentricity = 1.0 - short_side / long_side

    hull_area = poly.convex_hull.area
    solidity = float(poly.area / hull_area) if hull_area > 0 else 1.0
    return float(poly.area), eccentricity, solidity


def classify_room_polygon(
    ring: np.ndarray,
) -> tuple[str, str, float, bool]:
    """Classify a room directly from its polygon ring (x, y in metres).

    Returns ``(label, category, confidence, is_common)``.  ``is_common``
    uses the conservative :data:`SHAPE_COMMON_CATEGORIES` mapping because
    shape metrics alone cannot distinguish a lobby from an open-plan suite.
    """
    area_m2, eccentricity, solidity = polygon_shape_metrics(ring)
    label, category, confidence = classify_room_shape(
        area_m2, eccentricity, solidity,
    )
    return label, category, confidence, category in SHAPE_COMMON_CATEGORIES
