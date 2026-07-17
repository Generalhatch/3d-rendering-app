"""Phase 3.5 (drafting-curves): arc fitting for curved walls, render-layer
envelope regularization, and Manhattan-snap protection for genuinely
diagonal / curved walls.

Synthetic reference fixtures with exact assertions (hand-computed):

- Circle fixture: points on the circle centre (3, 2), r = 5 — the Kåsa
  fit must recover (3, 2, 5) exactly.
- Quarter-arc polyline: r = 4 about the origin, vertices every 10° from
  0° to 90° (10 points) — one arc span covering all vertices.
- Envelope fixture: a 10×6 rectangle whose top edge is drawn 2° off-axis
  and with a redundant mid-edge vertex; regularization must snap it back
  to exactly horizontal and drop the redundant vertex, while a true 45°
  chamfer must survive untouched.
- Curve-chain fixture for the Manhattan guard: a quarter circle r = 6
  sliced into 9 end-to-end chords (10° steps).  The chords near 0°/90°
  fall inside the 12° snap tolerance and — without the guard — get
  rotated onto the axes, flattening the curve.  With the guard the whole
  chain passes through byte-identical.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from app.geometry.model import FloorGeometry
from app.sheet.curves import (
    fit_arcs_to_polyline,
    fit_circle,
    regularize_ring,
    ring_dominant_axes,
    ring_path_d,
    snap_ring_edges_to_axes,
)
from app.vectorize.regularize import (
    RegularizeParams,
    find_curve_chain_indices,
    regularize_with_provenance,
)


def _arc_points(cx, cy, r, deg0, deg1, step_deg) -> np.ndarray:
    degs = np.arange(deg0, deg1 + step_deg / 2.0, step_deg)
    th = np.radians(degs)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


# ── Circle fitting ────────────────────────────────────────────────────────────

class TestFitCircle:
    def test_exact_on_true_circle(self):
        pts = _arc_points(3.0, 2.0, 5.0, 10, 170, 20)
        cx, cy, r = fit_circle(pts)
        assert (cx, cy, r) == pytest.approx((3.0, 2.0, 5.0), abs=1e-9)

    def test_collinear_points_raise(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        with pytest.raises(ValueError):
            fit_circle(pts)


# ── Arc-run detection ─────────────────────────────────────────────────────────

class TestFitArcsToPolyline:
    def test_quarter_arc_detected_exactly(self):
        pts = _arc_points(0.0, 0.0, 4.0, 0, 90, 10)   # 10 vertices
        spans = fit_arcs_to_polyline(pts, tol=1e-6)
        assert len(spans) == 1
        span = spans[0]
        assert (span.start_idx, span.end_idx) == (0, 9)
        assert (span.cx, span.cy, span.r) == pytest.approx((0.0, 0.0, 4.0), abs=1e-9)
        assert span.ccw is True

    def test_clockwise_arc_direction(self):
        pts = _arc_points(0.0, 0.0, 4.0, 0, 90, 10)[::-1]
        spans = fit_arcs_to_polyline(pts, tol=1e-6)
        assert len(spans) == 1
        assert spans[0].ccw is False

    def test_rectangle_corners_are_not_an_arc(self):
        # All 4 corners of a rectangle lie exactly on its circumcircle —
        # the chord-angle guard must reject the "fit".
        pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]])
        assert fit_arcs_to_polyline(pts, tol=1e-6) == []

    def test_straight_polyline_no_arcs(self):
        pts = np.column_stack([np.linspace(0, 10, 8), np.zeros(8)])
        assert fit_arcs_to_polyline(pts, tol=1e-6) == []

    def test_line_then_arc_mixed(self):
        line = np.column_stack([np.linspace(-6.0, -1.0, 4), np.full(4, -4.0)])
        arc = _arc_points(0.0, 0.0, 4.0, 270, 350, 10)   # starts at (0, −4)
        pts = np.vstack([line, arc[1:]])
        spans = fit_arcs_to_polyline(pts, tol=1e-6)
        assert len(spans) == 1
        span = spans[0]
        assert span.start_idx == 4                        # arc starts after the line
        assert span.end_idx == len(pts) - 1
        assert (span.cx, span.cy, span.r) == pytest.approx((0.0, 0.0, 4.0), abs=1e-8)


# ── SVG path emission ─────────────────────────────────────────────────────────

class TestRingPathD:
    def test_square_ring_all_line_commands(self):
        ring = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0]])
        d = ring_path_d(ring)
        assert d == "M 0 0 L 100 0 L 100 60 L 0 60 Z"

    def test_curved_run_emitted_as_arc(self):
        # Half-stadium: straight bottom edge + semicircular right cap
        # (r = 30 about (50, 30)), vertices every 15°.
        cap = _arc_points(50.0, 30.0, 30.0, -90, 90, 15)
        ring = np.vstack([[[0.0, 0.0]], cap, [[0.0, 60.0]]])
        d = ring_path_d(ring, arc_tol_mm=1e-6)
        assert " A 30 30 0 " in d
        # The arc lands exactly on the cap's top point (50, 60).
        assert "50 60" in d
        # No staircase: at most a couple of L commands remain (the straight
        # closing edges), never the 12 chords.
        assert d.count("L") <= 3

    def test_arc_sweep_flag_matches_direction(self):
        cap = _arc_points(50.0, 30.0, 30.0, -90, 90, 15)
        ring = np.vstack([[[0.0, 0.0]], cap, [[0.0, 60.0]]])
        d_ccw = ring_path_d(ring, arc_tol_mm=1e-6)
        d_cw = ring_path_d(ring[::-1], arc_tol_mm=1e-6)
        assert " A 30 30 0 0 1 " in d_ccw
        assert " A 30 30 0 0 0 " in d_cw


# ── Envelope regularization ───────────────────────────────────────────────────

class TestSnapRingEdgesToAxes:
    def test_near_axis_edge_snapped_exactly(self):
        # Top edge tilted 2°: (0,6) → (10, 6 + 10·tan2°).  After the snap
        # the edge is exactly horizontal at the tilt's midpoint height.
        dy = 10.0 * math.tan(math.radians(2.0))
        ring = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 6.0 + dy], [0.0, 6.0]])
        snapped = snap_ring_edges_to_axes(ring, axes_rad=(0.0, math.pi / 2.0))
        assert snapped[2, 1] == pytest.approx(snapped[3, 1], abs=1e-12)
        assert snapped[2, 1] == pytest.approx(6.0 + dy / 2.0, abs=1e-9)
        # Bottom edge untouched.
        assert snapped[0] == pytest.approx((0.0, 0.0), abs=1e-9)
        assert snapped[1] == pytest.approx((10.0, 0.0), abs=1e-9)

    def test_true_diagonal_preserved(self):
        # 45° chamfer corner — far beyond the 8° tolerance, must not move.
        ring = np.array([
            [0.0, 0.0], [8.0, 0.0], [10.0, 2.0], [10.0, 6.0], [0.0, 6.0],
        ])
        snapped = snap_ring_edges_to_axes(ring, axes_rad=(0.0, math.pi / 2.0))
        assert snapped == pytest.approx(ring, abs=1e-9)


class TestRegularizeRing:
    def test_simplify_plus_snap(self):
        # Rectangle with a redundant mid-edge vertex and a 2° top tilt.
        dy = 10.0 * math.tan(math.radians(2.0))
        ring = np.array([
            [0.0, 0.0], [5.0, 0.0], [10.0, 0.0],      # collinear midpoint
            [10.0, 6.0 + dy], [0.0, 6.0],
        ])
        reg = regularize_ring(ring, axes_rad=(0.0, math.pi / 2.0), dp_tol=0.05)
        assert len(reg) == 4                          # midpoint dropped
        ys_top = sorted(reg[:, 1])[-2:]
        assert ys_top[0] == pytest.approx(ys_top[1], abs=1e-9)   # exactly level

    def test_ring_dominant_axes(self):
        th = math.radians(20.0)
        c, s = math.cos(th), math.sin(th)
        base = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]])
        rot = base @ np.array([[c, s], [-s, c]])
        a0, a1 = ring_dominant_axes(rot)
        assert a0 == pytest.approx(th, abs=1e-9)
        assert a1 == pytest.approx(th + math.pi / 2.0, abs=1e-9)

    def test_area_change_is_negligible(self):
        # Presentation-only guarantee: regularizing a jittered ring may not
        # move area more than the jitter scale (it never feeds measurement,
        # but a drawing that visibly disagrees with the numbers is a bug).
        rng = np.random.default_rng(7)
        base = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]])
        dense = []
        for k in range(4):
            p, q = base[k], base[(k + 1) % 4]
            for t in np.linspace(0.0, 1.0, 12, endpoint=False):
                dense.append(p + t * (q - p))
        dense = np.asarray(dense) + rng.normal(0.0, 0.01, (48, 2))
        reg = regularize_ring(dense, axes_rad=(0.0, math.pi / 2.0), dp_tol=0.05)
        from app.geometry.model import polygon_area
        assert polygon_area(reg) == pytest.approx(60.0, rel=0.01)


class TestEnvelopeInSvg:
    def test_wall_less_floor_draws_regularized_envelope(self):
        floor = FloorGeometry(
            floor_id="env-only",
            envelope=np.array([
                [0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0],
            ]),
        )
        from app.sheet.render import render_sheet
        render = render_sheet(floor)
        root = ET.fromstring(render.svg)
        env_g = next(el for el in root.iter() if el.get("id") == "envelope")
        paths = [e for e in env_g if e.tag.endswith("path")]
        assert len(paths) == 1
        assert paths[0].get("class") == "envelope-outline"

    def test_floor_with_walls_skips_envelope_layer(self):
        from app.sheet.render import render_sheet
        from tests.reference_floors import two_suite_floor
        render = render_sheet(two_suite_floor())
        root = ET.fromstring(render.svg)
        env_g = next(el for el in root.iter() if el.get("id") == "envelope")
        assert list(env_g) == []


# ── Manhattan-snap protection (pipeline geometry — reference fixtures) ────────

def _quarter_circle_chords(r: float = 6.0, step_deg: float = 10.0) -> np.ndarray:
    """Quarter circle as end-to-end chords: 9 segments for 10° steps."""
    pts = _arc_points(0.0, 0.0, r, 0, 90, step_deg)
    return np.stack([pts[:-1], pts[1:]], axis=1)


class TestManhattanCurveProtection:
    def test_chain_detector_finds_all_chords(self):
        segs = _quarter_circle_chords()
        protected = find_curve_chain_indices(segs)
        assert protected == set(range(9))

    def test_rectangle_walls_not_a_chain(self):
        segs = np.array([
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 6.0]],
            [[10.0, 6.0], [0.0, 6.0]],
            [[0.0, 6.0], [0.0, 0.0]],
        ])
        assert find_curve_chain_indices(segs) == set()

    def test_collinear_run_not_a_chain(self):
        segs = np.array([
            [[0.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0]],
            [[4.0, 0.0], [6.0, 0.0]],
            [[6.0, 0.0], [8.0, 0.0]],
        ])
        assert find_curve_chain_indices(segs) == set()

    def test_snap_does_not_flatten_curve_chords(self):
        # Rectangle (establishes the dominant axes) + quarter-circle chords.
        rect = np.array([
            [[0.0, -10.0], [20.0, -10.0]],
            [[20.0, -10.0], [20.0, -2.0]],
            [[20.0, -2.0], [0.0, -2.0]],
            [[0.0, -2.0], [0.0, -10.0]],
        ])
        chords = _quarter_circle_chords()
        segs = np.vstack([rect, chords])
        result = regularize_with_provenance(
            segs,
            RegularizeParams(
                drop_short_below_m=0.0, merge_collinear=False,
            ),
        )
        # Every chord survives byte-identical: neither snapped nor dropped.
        kept = result.kept
        assert len(kept) == len(segs)
        for chord in chords:
            assert any(
                np.allclose(k, chord, atol=1e-12) for k in kept
            ), f"curve chord {chord.tolist()} was altered by Manhattan snap"

    def test_without_protection_chords_are_flattened(self):
        # Regression guard for the guard: turning protection off reproduces
        # the old flattening behaviour (first chord ≈5° gets axis-snapped).
        rect = np.array([
            [[0.0, -10.0], [20.0, -10.0]],
            [[20.0, -10.0], [20.0, -2.0]],
            [[20.0, -2.0], [0.0, -2.0]],
            [[0.0, -2.0], [0.0, -10.0]],
        ])
        chords = _quarter_circle_chords()
        segs = np.vstack([rect, chords])
        result = regularize_with_provenance(
            segs,
            RegularizeParams(
                drop_short_below_m=0.0, merge_collinear=False,
                protect_curve_chains=False, keep_diagonal_min_length_m=0.0,
            ),
        )
        first_chord = chords[0]
        assert not any(
            np.allclose(k, first_chord, atol=1e-12) for k in result.kept
        )

    def test_long_diagonal_wall_kept_not_dropped(self):
        rect = np.array([
            [[0.0, 0.0], [20.0, 0.0]],
            [[20.0, 0.0], [20.0, 10.0]],
            [[20.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 0.0]],
        ])
        diagonal = np.array([[[2.0, 2.0], [7.0, 7.0]]])   # 7.07 m at 45°
        segs = np.vstack([rect, diagonal])
        result = regularize_with_provenance(
            segs,
            RegularizeParams(drop_short_below_m=0.0, merge_collinear=False),
        )
        assert any(
            np.allclose(k, diagonal[0], atol=1e-12) for k in result.kept
        )
        assert all(r.dropped_by != "manhattan" for r in result.rejected)

    def test_short_offaxis_noise_still_dropped(self):
        rect = np.array([
            [[0.0, 0.0], [20.0, 0.0]],
            [[20.0, 0.0], [20.0, 10.0]],
            [[20.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 0.0]],
        ])
        noise = np.array([[[5.0, 5.0], [5.4, 5.4]]])      # 0.57 m at 45°
        segs = np.vstack([rect, noise])
        result = regularize_with_provenance(
            segs,
            RegularizeParams(drop_short_below_m=0.0, merge_collinear=False),
        )
        assert len(result.kept) == 4
        dropped = [r for r in result.rejected if r.dropped_by == "manhattan"]
        assert len(dropped) == 1
        assert dropped[0].length_m == pytest.approx(math.hypot(0.4, 0.4), abs=1e-9)
