"""Contour wall extraction must produce long CAD axes, not edge soup.

BricsCAD POINTCLOUDPROJECTSECTION closes gaps then OPTIMIZE-merges
collinear runs.  Cloud2BIM § 2.7 does the same after Douglas-Peucker.
Without both steps, median wall length stays < 1 m and DXF is a sketch.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.vectorize.slicer import RasterAffine
from app.vectorize.walls_contour import WallContourParams, extract_wall_contours


def _affine(res: float = 0.01, w: int = 400, h: int = 400) -> RasterAffine:
    return RasterAffine(
        origin_x=0.0, origin_y=0.0,
        resolution_m_per_px=res,
        width_px=w, height_px=h,
    )


def _dashed_horizontal_wall(
    h: int = 200, w: int = 500, y: int = 100, thickness: int = 8,
) -> np.ndarray:
    """Binary mask with a 4 m wall drawn as dashed 20 px strokes + 8 px gaps."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x = 20
    while x < w - 20:
        x2 = min(x + 20, w - 20)
        mask[y : y + thickness, x:x2] = 255
        x = x2 + 8  # 8 px gap ≈ 8 cm — healed by close_kernel_px=9
    return mask


class TestContourCadAxes:
    def test_dashed_wall_becomes_few_long_axes(self):
        mask = _dashed_horizontal_wall()
        affine = _affine(w=500, h=200)
        # No merge → many short edges
        broken = extract_wall_contours(
            mask, affine,
            params=WallContourParams(
                close_kernel_px=9,
                merge_collinear=False,
                simplify_eps_m=0.02,
                min_perimeter_m=0.30,
                min_segment_length_m=0.08,
            ),
        )
        # With BricsCAD-style merge → long continuous axis
        merged = extract_wall_contours(
            mask, affine,
            params=WallContourParams(
                close_kernel_px=9,
                merge_collinear=True,
                merge_endpoint_gap_m=1.0,
                merge_perp_distance_m=0.12,
                merge_parallel_tol_deg=6.0,
                simplify_eps_m=0.02,
                min_perimeter_m=0.30,
                min_segment_length_m=0.08,
            ),
        )
        assert merged.n_segments_total < broken.n_segments_total
        lens = np.linalg.norm(
            merged.flat_segments[:, 1] - merged.flat_segments[:, 0], axis=1,
        )
        assert float(lens.max()) >= 3.0  # healed dashes span metres

    def test_close_kernel_heals_gaps(self):
        mask = _dashed_horizontal_wall()
        affine = _affine(w=500, h=200)
        weak = extract_wall_contours(
            mask, affine,
            params=WallContourParams(
                close_kernel_px=1, merge_collinear=False, min_perimeter_m=0.10,
            ),
        )
        strong = extract_wall_contours(
            mask, affine,
            params=WallContourParams(
                close_kernel_px=9, merge_collinear=False, min_perimeter_m=0.10,
            ),
        )
        # Stronger CLOSE → fewer raw contours (dashes fuse into one ribbon)
        assert strong.n_contours_raw <= weak.n_contours_raw
