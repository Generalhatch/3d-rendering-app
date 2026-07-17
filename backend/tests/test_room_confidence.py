"""Per-room confidence scoring (Phase 4).

Synthetic known-geometry fixtures: a 10×10 m room rasterized at 1 cm/px.
Boundary coverage, snap correction, and the combination formula are all
hand-computable:

    snap_factor = 1 − 0.5 · min(correction / 0.30, 1)
    confidence  = boundary_coverage × snap_factor
    flagged     = confidence < 0.70
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.vectorize.room_confidence import (
    FLAG_THRESHOLD,
    boundary_coverage,
    combine_confidence,
    compute_room_confidences,
    vertex_snap_correction,
)
from app.vectorize.slicer import RasterAffine
from app.vectorize.topology import RoomFace

_RES = 0.01  # m per px

SQUARE_10 = np.array([
    [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
])


def _affine(width_px: int = 1100, height_px: int = 1100) -> RasterAffine:
    # Origin at (−0.5, −0.5) so the room sits comfortably inside the canvas.
    return RasterAffine(
        origin_x=-0.5, origin_y=-0.5, resolution_m_per_px=_RES,
        width_px=width_px, height_px=height_px,
    )


def _draw_walls(affine: RasterAffine, edges: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Rasterize wall edges with the pipeline's pixel-centre convention."""
    img = np.zeros((affine.height_px, affine.width_px), dtype=np.uint8)
    for a, b in edges:
        x1 = int(round((a[0] - affine.origin_x) / _RES - 0.5))
        y1 = int(round((a[1] - affine.origin_y) / _RES - 0.5))
        x2 = int(round((b[0] - affine.origin_x) / _RES - 0.5))
        y2 = int(round((b[1] - affine.origin_y) / _RES - 0.5))
        cv2.line(img, (x1, y1), (x2, y2), color=255, thickness=2)
    return img


def _ring_edges(polygon: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(polygon)
    return [(polygon[k], polygon[(k + 1) % n]) for k in range(n)]


class TestBoundaryCoverage:
    def test_fully_observed_room_scores_one(self):
        affine = _affine()
        raster = _draw_walls(affine, _ring_edges(SQUARE_10))
        assert boundary_coverage(SQUARE_10, raster, affine) == pytest.approx(1.0)

    def test_one_missing_wall_scores_three_quarters(self):
        """Erase the y=10 edge: 1 of 4 equal edges unobserved → ≈ 0.75
        (a hair above — corner samples sit within radius of the side walls)."""
        affine = _affine()
        edges = _ring_edges(SQUARE_10)
        raster = _draw_walls(affine, [e for e in edges
                                      if not (e[0][1] == 10.0 and e[1][1] == 10.0)])
        cov = boundary_coverage(SQUARE_10, raster, affine)
        assert 0.72 <= cov <= 0.78

    def test_empty_raster_scores_zero(self):
        affine = _affine()
        raster = np.zeros((affine.height_px, affine.width_px), dtype=np.uint8)
        assert boundary_coverage(SQUARE_10, raster, affine) == 0.0

    def test_degenerate_polygon_scores_zero(self):
        affine = _affine()
        raster = _draw_walls(affine, _ring_edges(SQUARE_10))
        assert boundary_coverage(SQUARE_10[:2], raster, affine) == 0.0


class TestVertexSnapCorrection:
    def test_zero_when_vertices_match_endpoints(self):
        segs = np.array([
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 10.0]],
            [[10.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 0.0]],
        ])
        assert vertex_snap_correction(SQUARE_10, segs) == pytest.approx(0.0)

    def test_uniform_offset_measured_exactly(self):
        """Every detected endpoint 0.10 m inside its room vertex (along x)
        → mean correction exactly 0.10."""
        segs = np.array([
            [[0.10, 0.0], [9.90, 0.0]],
            [[0.10, 10.0], [9.90, 10.0]],
        ])
        # Nearest endpoint to each corner is 0.10 m away along x.
        assert vertex_snap_correction(SQUARE_10, segs) == pytest.approx(0.10)

    def test_no_detected_segments_is_maximal(self):
        assert vertex_snap_correction(
            SQUARE_10, np.zeros((0, 2, 2)),
        ) == pytest.approx(0.30)


class TestCombineConfidence:
    def test_perfect_room(self):
        assert combine_confidence(1.0, 0.0) == pytest.approx(1.0)

    def test_coverage_scales_score(self):
        assert combine_confidence(0.75, 0.0) == pytest.approx(0.75)

    def test_snap_factor_hand_computed(self):
        # snap 0.15 / 0.30 = 0.5 → factor 0.75 → 0.8 × 0.75 = 0.6
        assert combine_confidence(0.8, 0.15) == pytest.approx(0.60)

    def test_snap_penalty_caps_at_half(self):
        # Even absurd corrections can only halve the score.
        assert combine_confidence(1.0, 5.0) == pytest.approx(0.50)


class TestComputeRoomConfidences:
    def _room(self, polygon: np.ndarray) -> RoomFace:
        from app.geometry.model import ring_perimeter, ring_signed_area
        return RoomFace(
            vertex_ids=list(range(len(polygon))),
            polygon=polygon,
            area_m2=abs(ring_signed_area(polygon)),
            perimeter_m=ring_perimeter(polygon),
        )

    def test_perfect_room_full_confidence_not_flagged(self):
        affine = _affine()
        raster = _draw_walls(affine, _ring_edges(SQUARE_10))
        segs = np.array([
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 10.0]],
            [[10.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 0.0]],
        ])

        scores = compute_room_confidences(
            [self._room(SQUARE_10)], raster, affine, segs,
        )

        assert len(scores) == 1
        rc = scores[0]
        assert rc.room_index == 0
        assert rc.area_m2 == pytest.approx(100.0)
        assert rc.boundary_coverage == pytest.approx(1.0)
        assert rc.snap_correction_m == pytest.approx(0.0)
        assert rc.confidence == pytest.approx(1.0)
        assert rc.flagged is False
        assert rc.polygon == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

    def test_half_observed_room_flagged(self):
        """Only 2 of 4 walls observed → coverage ≈ 0.5 → confidence < 0.70."""
        affine = _affine()
        edges = _ring_edges(SQUARE_10)
        raster = _draw_walls(affine, edges[:2])   # y=0 and x=10 walls only
        segs = np.array([
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 10.0]],
            [[10.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 0.0]],
        ])

        scores = compute_room_confidences(
            [self._room(SQUARE_10)], raster, affine, segs,
        )

        rc = scores[0]
        assert 0.45 <= rc.boundary_coverage <= 0.55
        assert rc.confidence < FLAG_THRESHOLD
        assert rc.flagged is True

    def test_mixed_floor_flags_only_the_bad_room(self):
        """Two rooms side by side; only the unobserved one is flagged."""
        good = SQUARE_10
        bad = SQUARE_10 + np.array([12.0, 0.0])
        affine = _affine(width_px=2400, height_px=1100)
        raster = _draw_walls(affine, _ring_edges(good))  # bad room: no walls
        segs = np.array([
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 10.0]],
            [[10.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 0.0]],
        ])

        scores = compute_room_confidences(
            [self._room(good), self._room(bad)], raster, affine, segs,
        )

        assert scores[0].flagged is False
        assert scores[1].flagged is True
        assert scores[1].confidence < scores[0].confidence

    def test_scores_never_alter_geometry(self):
        """Presentation-only guarantee: input rooms are untouched."""
        affine = _affine()
        raster = _draw_walls(affine, _ring_edges(SQUARE_10))
        room = self._room(SQUARE_10.copy())
        before = room.polygon.copy()

        compute_room_confidences([room], raster, affine, np.zeros((0, 2, 2)))

        np.testing.assert_array_equal(room.polygon, before)
        assert room.area_m2 == pytest.approx(100.0)
