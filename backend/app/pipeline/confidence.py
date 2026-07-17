"""Confidence scoring — already embedded in align.py, exposed here for reuse."""
from __future__ import annotations


def compute_confidence_pct(confidence: float) -> int:
    """Convert 0.0–1.0 float confidence to 0–100 integer."""
    return int(round(min(100, max(0, confidence * 100))))


# Scan-only mode has no ground-truth plan to verify against, so its output can
# never honestly claim certainty.  Cap the score below "high" (see
# confidence_label) so operators always review scan-only geometry.
_SCAN_ONLY_MAX_CONFIDENCE = 0.89


def compute_scan_only_confidence(
    num_wall_planes: int,
    plan_source: str,
    merge_strategy: str,
    rooms: list[dict],
) -> float:
    """Compute an honest 0–1 confidence for scan-only (no-plan) mode.

    Replaces the old hardcoded ``confidence_pct: 100`` — a fabricated number
    for the mode with the LEAST verification.  Multiplicative factors:

    - **structure** — how well the wall detector captured the building.
      Saturates at 8 detected 3D wall planes; the 2D-projection fallback
      (used when < 4 planes were found) is inherently weaker and is scored
      from how few planes forced the fallback.
    - **room closure** — fraction of extracted rooms with a closed, sane
      polygon (≥ 3 vertices, positive area).  No rooms at all is a strong
      signal the geometry is unreliable.
    - **registration** — single scans need no registration; pre-registered
      multi-scan exports are trustworthy; our own ICP fallback is the
      riskiest path (no ground truth to validate the fit).

    Parameters
    ----------
    num_wall_planes : number of 3D wall planes detected.
    plan_source : ``"3d_planes"`` or ``"2d_projection"``.
    merge_strategy : ``"single"``, ``"concatenate"``, or ``"icp_registered"``.
    rooms : room dicts with ``polygon_2d`` and ``area_m2`` keys.
    """
    # Structure factor.
    if plan_source == "3d_planes":
        structure = min(1.0, num_wall_planes / 8.0)
    else:
        # 2D projection fallback: engaged because 3D plane detection failed.
        structure = 0.45 + 0.05 * min(num_wall_planes, 3)

    # Room-closure factor.
    if rooms:
        n_closed = 0
        for room in rooms:
            poly = room.get("polygon_2d") or []
            area = float(room.get("area_m2") or 0.0)
            if len(poly) >= 3 and area > 0.0:
                n_closed += 1
        closure = 0.4 + 0.6 * (n_closed / len(rooms))
    else:
        closure = 0.4

    # Registration factor.
    registration = {
        "single": 1.0,
        "concatenate": 0.95,
        "icp_registered": 0.80,
    }.get(merge_strategy, 0.80)

    score = structure * closure * registration
    return float(min(_SCAN_ONLY_MAX_CONFIDENCE, max(0.0, score)))


def confidence_label(confidence_pct: int) -> str:
    if confidence_pct >= 90:
        return "high"
    elif confidence_pct >= 70:
        return "medium"
    else:
        return "low"
