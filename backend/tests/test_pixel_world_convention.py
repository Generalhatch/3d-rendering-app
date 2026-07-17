"""Pixel <-> world convention consistency across the vectorize stack.

Canonical convention (see RasterAffine): integer pixel index ``i`` has its
CENTRE at ``origin + (i + 0.5) * res``.

Guards against the P0 half-pixel mismatch: ``classical.segments_pixels_to_world``
omitted the +0.5 while ``RasterAffine.pixel_to_world`` and the contour wall
extractor included it — a 5 mm systematic bias at 1 cm resolution that shifted
every wall relative to every other geometry source.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.vectorize.classical import segments_pixels_to_world
from app.vectorize.columns import _px_to_world
from app.vectorize.pipeline import _segments_world_to_pixels
from app.vectorize.slicer import RasterAffine


@pytest.fixture
def affine() -> RasterAffine:
    return RasterAffine(
        origin_x=10.0,
        origin_y=-4.0,
        resolution_m_per_px=0.01,
        width_px=1000,
        height_px=800,
    )


class TestConventionAgreement:
    def test_segments_pixels_to_world_matches_pixel_to_world(self, affine):
        """The segment converter and RasterAffine must map the same pixel to
        the same world point (this failed by exactly half a pixel before)."""
        seg_px = np.array([[100.0, 200.0, 300.0, 400.0]])
        world = segments_pixels_to_world(seg_px, affine)

        expected_start = affine.pixel_to_world(np.array([[100, 200]]))[0]
        expected_end = affine.pixel_to_world(np.array([[300, 400]]))[0]

        np.testing.assert_allclose(world[0, 0], expected_start, atol=1e-12)
        np.testing.assert_allclose(world[0, 1], expected_end, atol=1e-12)

    def test_columns_px_to_world_matches_pixel_to_world(self, affine):
        wx, wy = _px_to_world(123.0, 456.0, affine)
        expected = affine.pixel_to_world(np.array([[123, 456]]))[0]
        assert wx == pytest.approx(expected[0], abs=1e-12)
        assert wy == pytest.approx(expected[1], abs=1e-12)

    def test_world_to_pixel_center_roundtrip(self, affine):
        """pixel -> world (centre) -> pixel must return the original index."""
        pix = np.array([[0, 0], [7, 3], [999, 799]])
        world = affine.pixel_to_world(pix)
        back = affine.world_to_pixel(world)
        np.testing.assert_array_equal(back, pix)


class TestSegmentsRoundTrip:
    def test_px_world_px_roundtrip_exact(self, affine):
        """(N,4) pixel segments -> world -> pixels must be the identity."""
        seg_px = np.array([
            [0.0, 0.0, 10.0, 0.0],
            [5.0, 7.0, 5.0, 300.0],
            [999.0, 799.0, 1.0, 2.0],
        ])
        world = segments_pixels_to_world(seg_px, affine)
        back = _segments_world_to_pixels(world, affine)
        np.testing.assert_array_equal(back, seg_px.astype(np.int32))

    def test_world_px_world_bias_below_half_pixel(self, affine):
        """world -> pixel -> world must not carry a systematic bias.

        The old code truncated in one direction and skipped the +0.5 in the
        other, giving every segment a consistent half-pixel (5 mm) drift.
        """
        rng = np.random.default_rng(42)
        n = 500
        segs_world = np.empty((n, 2, 2))
        segs_world[:, :, 0] = rng.uniform(10.5, 19.5, size=(n, 2))
        segs_world[:, :, 1] = rng.uniform(-3.5, 3.5, size=(n, 2))

        px = _segments_world_to_pixels(segs_world, affine)
        # Convert back through the canonical pixel-centre mapping.
        seg_px_flat = px.astype(np.float64)
        back = segments_pixels_to_world(seg_px_flat, affine)

        err = back - segs_world
        res = affine.resolution_m_per_px
        # Rounding error must stay within half a pixel...
        assert np.abs(err).max() <= res / 2 + 1e-9
        # ...and be unbiased: the MEAN error must be far below half a pixel.
        assert abs(err.mean()) < 0.1 * res
