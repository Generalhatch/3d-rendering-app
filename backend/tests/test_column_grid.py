"""Phase 3: column grid fitting (app.sheet.grid).

Synthetic column layouts with known grid spacing/rotation so the fit is
exactly checkable:

  - 3×4 orthogonal grid at 0° — exact offsets, labels, zero RMSE
  - the same grid rotated 30° — rotation + offsets recovered
  - an off-grid column reported as an outlier, never snapped
  - jittered centres — fit within tolerance
  - too few columns → None (manual fallback territory)
  - manual grid entry produces the same structure
  - world-space grid line segments for the renderer
"""
from __future__ import annotations

import numpy as np
import pytest

from app.geometry.model import ColumnFeature
from app.sheet.grid import (
    ColumnGrid,
    GridFitParams,
    fit_column_grid,
    grid_line_segments_world,
    letter_label,
    manual_grid,
)

X_OFFSETS = [0.0, 6.0, 12.0]
Y_OFFSETS = [0.0, 5.0, 10.0, 15.0]


def _grid_columns(rotation_deg: float = 0.0) -> list[ColumnFeature]:
    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    cols = []
    for xi in X_OFFSETS:
        for yi in Y_OFFSETS:
            x = xi * c - yi * s
            y = xi * s + yi * c
            cols.append(ColumnFeature(id=f"c-{xi}-{yi}", centre=(x, y)))
    return cols


class TestFitExactGrid:
    def test_axis_aligned_grid_exact(self):
        grid = fit_column_grid(_grid_columns())
        assert grid is not None
        assert grid.source == "fitted"
        assert grid.rotation_rad == pytest.approx(0.0, abs=1e-9)
        assert grid.rmse_m == pytest.approx(0.0, abs=1e-9)
        assert grid.n_columns == 12
        assert grid.n_outliers == 0

        # u (vertical) lines: numbered ascending with u.
        assert [g.label for g in grid.u_lines] == ["1", "2", "3"]
        assert [g.offset_m for g in grid.u_lines] == pytest.approx(X_OFFSETS, abs=1e-9)
        assert [g.support for g in grid.u_lines] == [4, 4, 4]

        # v (horizontal) lines: lettered DESCENDING with v (A at top).
        assert [g.label for g in grid.v_lines] == ["A", "B", "C", "D"]
        assert [g.offset_m for g in grid.v_lines] == pytest.approx(
            sorted(Y_OFFSETS, reverse=True), abs=1e-9
        )
        assert [g.support for g in grid.v_lines] == [3, 3, 3, 3]

    def test_rotated_grid_recovers_rotation_and_offsets(self):
        grid = fit_column_grid(_grid_columns(rotation_deg=30.0))
        assert grid is not None
        assert grid.rotation_rad == pytest.approx(np.radians(30.0), abs=1e-9)
        # In the grid frame the offsets are the original layout coordinates.
        assert [g.offset_m for g in grid.u_lines] == pytest.approx(X_OFFSETS, abs=1e-6)
        assert [g.offset_m for g in grid.v_lines] == pytest.approx(
            sorted(Y_OFFSETS, reverse=True), abs=1e-6
        )
        assert grid.rmse_m == pytest.approx(0.0, abs=1e-9)

    def test_off_grid_column_is_outlier_not_snapped(self):
        cols = _grid_columns()
        cols.append(ColumnFeature(id="rogue", centre=(3.0, 2.4)))
        grid = fit_column_grid(cols)
        assert grid is not None
        assert grid.n_columns == 13
        assert grid.n_outliers == 1
        # Grid lines unchanged — the rogue column did not create an axis.
        assert [g.offset_m for g in grid.u_lines] == pytest.approx(X_OFFSETS, abs=1e-9)
        assert len(grid.v_lines) == 4

    def test_jittered_grid_within_tolerance(self):
        rng = np.random.default_rng(42)
        cols = []
        for col in _grid_columns():
            dx, dy = rng.uniform(-0.03, 0.03, size=2)
            cols.append(ColumnFeature(
                id=col.id, centre=(col.centre[0] + dx, col.centre[1] + dy),
            ))
        grid = fit_column_grid(cols)
        assert grid is not None
        assert abs(np.degrees(grid.rotation_rad) % 90.0) < 0.5 or \
            abs(np.degrees(grid.rotation_rad) % 90.0) > 89.5
        assert len(grid.u_lines) == 3
        assert len(grid.v_lines) == 4
        assert [g.offset_m for g in grid.u_lines] == pytest.approx(X_OFFSETS, abs=0.05)
        assert grid.rmse_m < 0.05
        assert grid.n_outliers == 0


class TestFitEdgeCases:
    def test_too_few_columns_returns_none(self):
        cols = [
            ColumnFeature(id="a", centre=(0.0, 0.0)),
            ColumnFeature(id="b", centre=(6.0, 0.0)),
            ColumnFeature(id="c", centre=(0.0, 5.0)),
        ]
        assert fit_column_grid(cols) is None

    def test_no_columns_returns_none(self):
        assert fit_column_grid([]) is None

    def test_min_columns_param_respected(self):
        cols = _grid_columns()[:6]
        assert fit_column_grid(cols, GridFitParams(min_columns=7)) is None
        assert fit_column_grid(cols, GridFitParams(min_columns=6)) is not None


class TestLetterLabels:
    def test_skips_i_and_o(self):
        # A B C D E F G H J — index 8 is J (I skipped).
        assert letter_label(0) == "A"
        assert letter_label(8) == "J"
        labels = [letter_label(k) for k in range(24)]
        assert "I" not in labels and "O" not in labels

    def test_wraps_to_double_letters(self):
        assert letter_label(24) == "AA"
        assert letter_label(25) == "AB"


class TestManualGrid:
    def test_manual_grid_structure(self):
        grid = manual_grid([12.0, 0.0, 6.0], [10.0, 0.0], rotation_deg=15.0)
        assert isinstance(grid, ColumnGrid)
        assert grid.source == "manual"
        assert grid.rotation_rad == pytest.approx(np.radians(15.0), abs=1e-12)
        # Offsets sorted; numbers ascend with u, letters descend with v.
        assert [(g.label, g.offset_m) for g in grid.u_lines] == [
            ("1", 0.0), ("2", 6.0), ("3", 12.0),
        ]
        assert [(g.label, g.offset_m) for g in grid.v_lines] == [
            ("A", 10.0), ("B", 0.0),
        ]
        assert grid.n_columns == 0


class TestGridLineSegments:
    def test_axis_aligned_segments_exact(self):
        grid = manual_grid([0.0, 6.0, 12.0], [0.0, 5.0, 10.0, 15.0])
        segs = grid_line_segments_world(
            grid, bounds_world=(0.0, 0.0, 12.0, 15.0), extension_m=1.0,
        )
        assert len(segs) == 7
        by_label = {s["label"]: s for s in segs}
        # u-line "1" at u=0: vertical, bubble end (p1) at max v.
        s1 = by_label["1"]
        assert s1["family"] == "u"
        assert s1["p0"] == pytest.approx((0.0, -1.0), abs=1e-9)
        assert s1["p1"] == pytest.approx((0.0, 16.0), abs=1e-9)
        # v-line "A" at v=15: horizontal, bubble end (p1) at min u (left).
        sa = by_label["A"]
        assert sa["family"] == "v"
        assert sa["p0"] == pytest.approx((13.0, 15.0), abs=1e-9)
        assert sa["p1"] == pytest.approx((-1.0, 15.0), abs=1e-9)

    def test_json_dict_roundtrippable(self):
        grid = fit_column_grid(_grid_columns())
        d = grid.to_json_dict()
        assert d["source"] == "fitted"
        assert [line["label"] for line in d["u_lines"]] == ["1", "2", "3"]
        assert d["rotation_deg"] == pytest.approx(0.0, abs=1e-9)
