"""Adaptive slice-band selection from the height histogram.

Low-clearance spaces (data centers, basements) break the default vertical
bands: the 1.90–2.30 m ceiling gate straddles a 2.1 m ceiling.  The band
planner must measure clearance from the histogram and scale bands down —
and must NOT touch them when the room is tall enough for the defaults.
All expected numbers below are hand-computed from the module's constants
(ceiling margin 0.15 m, slice headroom 0.60 m, band width preserved).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.vectorize.adaptive_band import (
    band_summary,
    choose_slice_band,
    estimate_ceiling_clearance,
)


class TestChooseSliceBand:
    def test_no_clearance_keeps_defaults(self):
        plan = choose_slice_band(None)
        assert plan.reason == "no_ceiling"
        assert plan.adjusted is False
        assert plan.slice_offset_m == pytest.approx(1.60)
        assert plan.ceiling_band_lo_m == pytest.approx(1.90)
        assert plan.ceiling_band_hi_m == pytest.approx(2.30)
        assert plan.density_band_hi_m == pytest.approx(2.20)

    def test_tall_room_keeps_defaults(self):
        """2.70 m clearance: usable top 2.55 ≥ default 2.30 → unchanged."""
        plan = choose_slice_band(2.70)
        assert plan.reason == "default"
        assert plan.adjusted is False
        assert plan.ceiling_band_hi_m == pytest.approx(2.30)
        assert plan.slice_offset_m == pytest.approx(1.60)

    def test_boundary_clearance_keeps_defaults(self):
        """2.45 m: usable top exactly 2.30 → still default."""
        plan = choose_slice_band(2.45)
        assert plan.reason == "default"
        assert plan.adjusted is False

    def test_low_ceiling_2_20(self):
        """2.20 m clearance: hi = 2.05, lo = 1.65 (0.40 width kept),
        slice offset min(1.60, 2.20 − 0.60) = 1.60."""
        plan = choose_slice_band(2.20)
        assert plan.reason == "low_ceiling"
        assert plan.adjusted is True
        assert plan.ceiling_band_hi_m == pytest.approx(2.05)
        assert plan.ceiling_band_lo_m == pytest.approx(1.65)
        assert plan.slice_offset_m == pytest.approx(1.60)
        assert plan.density_band_hi_m == pytest.approx(2.05)

    def test_data_center_2_00(self):
        """2.00 m aisle: hi = 1.85, lo = 1.45, slice = 1.40, density 1.85."""
        plan = choose_slice_band(2.00)
        assert plan.adjusted is True
        assert plan.ceiling_band_hi_m == pytest.approx(1.85)
        assert plan.ceiling_band_lo_m == pytest.approx(1.45)
        assert plan.slice_offset_m == pytest.approx(1.40)
        assert plan.density_band_hi_m == pytest.approx(1.85)

    def test_extreme_low_clearance_hits_floors(self):
        """1.20 m crawl space: band top floored at 1.05 (usable) vs hard min
        1.00; slice offset floored at 0.80."""
        plan = choose_slice_band(1.20)
        assert plan.ceiling_band_hi_m == pytest.approx(1.05)
        assert plan.slice_offset_m == pytest.approx(0.80)

    def test_operator_overrides_never_raised(self):
        """A manually-lowered band (1.50–1.80) stays put in a tall room —
        adaptation only clamps downward."""
        plan = choose_slice_band(
            3.00, default_ceiling_lo_m=1.50, default_ceiling_hi_m=1.80,
        )
        assert plan.adjusted is False
        assert plan.ceiling_band_lo_m == pytest.approx(1.50)
        assert plan.ceiling_band_hi_m == pytest.approx(1.80)

    def test_band_width_preserved_when_scaled(self):
        plan = choose_slice_band(2.10)
        width = plan.ceiling_band_hi_m - plan.ceiling_band_lo_m
        assert width == pytest.approx(0.40)


class TestEstimateCeilingClearance:
    def _cloud(self, ceiling_h: float | None, n_wall: int = 20_000) -> np.ndarray:
        """Wall returns uniform over height + optional dense ceiling slab."""
        rng = np.random.default_rng(3)
        top = ceiling_h if ceiling_h is not None else 3.0
        walls = np.column_stack([
            rng.uniform(0, 10, n_wall),
            rng.uniform(0, 10, n_wall),
            rng.uniform(0.0, top, n_wall),
        ])
        parts = [walls]
        if ceiling_h is not None:
            slab = np.column_stack([
                rng.uniform(0, 10, 30_000),
                rng.uniform(0, 10, 30_000),
                rng.normal(ceiling_h, 0.01, 30_000),
            ])
            parts.append(slab)
        return np.vstack(parts)

    def test_finds_ceiling_slab(self):
        pts = self._cloud(ceiling_h=2.20)
        clearance = estimate_ceiling_clearance(pts, floor_z=0.0)
        assert clearance == pytest.approx(2.20, abs=0.05)

    def test_respects_floor_offset(self):
        pts = self._cloud(ceiling_h=2.20)
        pts[:, 2] += 5.0
        clearance = estimate_ceiling_clearance(pts, floor_z=5.0)
        assert clearance == pytest.approx(2.20, abs=0.05)

    def test_no_slab_returns_none(self):
        """Uniform wall returns with no concentrated ceiling → None."""
        pts = self._cloud(ceiling_h=None)
        assert estimate_ceiling_clearance(pts, floor_z=0.0) is None

    def test_too_few_points_returns_none(self):
        pts = np.random.default_rng(0).uniform(0, 3, size=(500, 3))
        assert estimate_ceiling_clearance(pts, floor_z=0.0) is None


class TestBandSummary:
    def test_summaries_render(self):
        assert "no ceiling" in band_summary(choose_slice_band(None))
        assert "defaults kept" in band_summary(choose_slice_band(2.70))
        adapted = band_summary(choose_slice_band(2.00))
        assert "adapted" in adapted and "1.40" in adapted
