"""Per-room confidence scoring — how much should the operator trust each room?

The floor-level confidence (Phase 0) tells the operator whether the JOB is
trustworthy; it says nothing about WHICH room needs review.  A floor where
eleven rooms traced perfectly and one was hallucinated across an unscanned
gap scores "pretty good" overall — exactly the case where a per-room flag
earns its keep.

Two measurable, orthogonal signals per room:

- **Boundary coverage** (raster agreement): the fraction of the room's
  polygon boundary that runs along observed wall pixels in the cleaned
  raster.  Sample points every ``sample_step_m`` along each ring edge and
  test the distance to the nearest raster foreground pixel (one distance
  transform for the whole raster, O(1) per sample).  A boundary edge
  through empty raster means the room ring was closed by inference
  (junction snapping / axis extension), not by observation.

- **Snap correction**: how far the topology stage had to move geometry to
  close this room.  Measured as the mean distance from each room-polygon
  vertex to the nearest endpoint of the PRE-topology wall centerlines.
  0 m = the room closed exactly where walls were detected; 0.3 m = every
  corner was manufactured by the snapping tolerance.

Combined score (each factor in [0, 1], hand-computable for tests):

    snap_factor = 1 − 0.5 · min(mean_correction / 0.30 m, 1)
    confidence  = boundary_coverage × snap_factor

so an unobserved boundary dominates (coverage scales the whole score) and
even maximal snap correction alone can only halve it.  Rooms below
``flag_threshold`` (default 0.70) are flagged for operator review.

Presentation-only: these scores never alter room polygons or areas.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .slicer import RasterAffine
from .topology import RoomFace

# A boundary sample counts as "observed" when a raster foreground pixel is
# within this distance — matches the pipeline-level coverage diagnostic
# (COVERAGE_RADIUS_M), wide enough for both faces of a 10–15 cm wall.
BOUNDARY_RADIUS_M = 0.08

# Snap corrections are normalized against the topology stage's own
# snapping_distance default: a room whose corners all moved by the full
# snapping distance was entirely manufactured.
SNAP_NORM_M = 0.30

FLAG_THRESHOLD = 0.70

_SAMPLE_STEP_M = 0.05


@dataclass
class RoomConfidence:
    """Per-room trust score, persisted in result.json."""
    room_index: int                 # index into TopologyResult.rooms
    area_m2: float
    boundary_coverage: float        # [0, 1] fraction of ring near foreground
    snap_correction_m: float        # mean vertex distance to detected endpoints
    confidence: float               # [0, 1] combined score
    flagged: bool                   # True = review recommended
    polygon: list[list[float]]      # ring vertices for editor highlighting

    def to_json_dict(self) -> dict:
        return {
            "room_index": self.room_index,
            "area_m2": self.area_m2,
            "boundary_coverage": self.boundary_coverage,
            "snap_correction_m": self.snap_correction_m,
            "confidence": self.confidence,
            "flagged": self.flagged,
            "polygon": self.polygon,
        }


def _sample_ring(polygon: np.ndarray, step_m: float) -> np.ndarray:
    """Points every ``step_m`` along the closed ring (vertices included)."""
    samples: list[np.ndarray] = []
    n = len(polygon)
    for k in range(n):
        a = polygon[k]
        b = polygon[(k + 1) % n]
        edge_len = float(np.hypot(*(b - a)))
        n_steps = max(1, int(np.ceil(edge_len / step_m)))
        for t in np.arange(n_steps) / n_steps:
            samples.append(a + t * (b - a))
    return np.asarray(samples, dtype=np.float64)


def boundary_coverage(
    polygon: np.ndarray,
    raster: np.ndarray,
    affine: RasterAffine,
    radius_m: float = BOUNDARY_RADIUS_M,
    sample_step_m: float = _SAMPLE_STEP_M,
) -> float:
    """Fraction of the ring boundary within ``radius_m`` of raster foreground."""
    if len(polygon) < 3:
        return 0.0
    fg = raster > 0
    if not fg.any():
        return 0.0

    # Distance (px) from every pixel to the nearest foreground pixel.
    inv = np.where(fg, 0, 255).astype(np.uint8)
    dist_px = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)

    samples = _sample_ring(np.asarray(polygon, dtype=np.float64), sample_step_m)
    res = affine.resolution_m_per_px
    # Pixel-centre convention: idx = (world − origin) / res − 0.5.
    cols = np.rint((samples[:, 0] - affine.origin_x) / res - 0.5).astype(np.int64)
    rows = np.rint((samples[:, 1] - affine.origin_y) / res - 0.5).astype(np.int64)
    h, w = raster.shape
    in_bounds = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)

    radius_px = radius_m / res
    covered = np.zeros(len(samples), dtype=bool)
    covered[in_bounds] = dist_px[rows[in_bounds], cols[in_bounds]] <= radius_px
    return float(covered.sum() / len(samples))


def vertex_snap_correction(
    polygon: np.ndarray,
    detected_segments: np.ndarray,
) -> float:
    """Mean distance from each ring vertex to the nearest detected endpoint.

    ``detected_segments`` are the PRE-topology wall centerlines — the
    geometry the detector actually produced before junction snapping and
    axis extension moved/manufactured vertices.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    if len(polygon) == 0:
        return 0.0
    if detected_segments is None or len(detected_segments) == 0:
        # No detected geometry at all — maximal correction.
        return SNAP_NORM_M
    endpoints = np.asarray(detected_segments, dtype=np.float64).reshape(-1, 2)
    d = polygon[:, None, :] - endpoints[None, :, :]
    dists = np.sqrt((d ** 2).sum(axis=2)).min(axis=1)
    return float(dists.mean())


def combine_confidence(coverage: float, snap_correction_m: float) -> float:
    """The one combination formula (see module docstring)."""
    snap_factor = 1.0 - 0.5 * min(snap_correction_m / SNAP_NORM_M, 1.0)
    return float(max(0.0, min(1.0, coverage * snap_factor)))


def compute_room_confidences(
    rooms: list[RoomFace],
    raster: np.ndarray,
    affine: RasterAffine,
    detected_segments: np.ndarray,
    flag_threshold: float = FLAG_THRESHOLD,
) -> list[RoomConfidence]:
    """Score every room of one topology result."""
    out: list[RoomConfidence] = []
    for idx, room in enumerate(rooms):
        cov = boundary_coverage(room.polygon, raster, affine)
        snap = vertex_snap_correction(room.polygon, detected_segments)
        conf = combine_confidence(cov, snap)
        out.append(RoomConfidence(
            room_index=idx,
            area_m2=float(room.area_m2),
            boundary_coverage=round(cov, 4),
            snap_correction_m=round(snap, 4),
            confidence=round(conf, 4),
            flagged=conf < flag_threshold,
            polygon=[[float(x), float(y)] for x, y in room.polygon],
        ))
    return out
