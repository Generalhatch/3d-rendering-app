"""Tests for vectorize topology — room enumeration must produce correctly
ordered polygon rings with exact areas on synthetic known geometry.

These tests guard against the P0 bug where ``nx.minimum_cycle_basis`` node
lists were used as if they were in cycle order (they are not — polygon areas
came out as garbage for any room with more than 4 vertices, and even
4-vertex rooms were order-dependent).
"""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from app.vectorize.topology import TopologyParams, build_topology


def _segments(points_pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> np.ndarray:
    return np.asarray(points_pairs, dtype=np.float64)


def _params() -> TopologyParams:
    # Tight snap tolerance: fixture geometry is exact, we don't want the
    # snapping stages to move anything.
    return TopologyParams(snap_tol_m=0.01, snapping_distance_m=0.05)


class TestSquareRoom:
    """A perfect 10x10 room must have area exactly 100 m²."""

    def test_area_exact(self):
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        result = build_topology(segs, params=_params())
        assert len(result.rooms) == 1
        assert result.rooms[0].area_m2 == pytest.approx(100.0, abs=1e-6)

    def test_perimeter_exact(self):
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        result = build_topology(segs, params=_params())
        assert result.rooms[0].perimeter_m == pytest.approx(40.0, abs=1e-6)

    def test_polygon_is_valid_ring(self):
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        result = build_topology(segs, params=_params())
        poly = Polygon(result.rooms[0].polygon)
        assert poly.is_valid
        assert poly.area == pytest.approx(100.0, abs=1e-6)


class TestLShapedRoom:
    """L-shape: 10x10 square minus a 4x4 notch = 84 m².

    Six vertices — this is where unordered cycle-basis node lists produce
    self-intersecting (bowtie) polygons with wrong areas.
    """

    L_SEGS = [
        ((0, 0), (10, 0)),
        ((10, 0), (10, 6)),
        ((10, 6), (6, 6)),
        ((6, 6), (6, 10)),
        ((6, 10), (0, 10)),
        ((0, 10), (0, 0)),
    ]

    def test_area_exact(self):
        result = build_topology(_segments(self.L_SEGS), params=_params())
        assert len(result.rooms) == 1
        assert result.rooms[0].area_m2 == pytest.approx(84.0, abs=1e-6)

    def test_polygon_ring_ordered(self):
        result = build_topology(_segments(self.L_SEGS), params=_params())
        poly = Polygon(result.rooms[0].polygon)
        assert poly.is_valid, "room ring must not self-intersect"
        assert poly.area == pytest.approx(84.0, abs=1e-6)

    def test_perimeter_exact(self):
        result = build_topology(_segments(self.L_SEGS), params=_params())
        # 10 + 6 + 4 + 4 + 6 + 10 = 40
        assert result.rooms[0].perimeter_m == pytest.approx(40.0, abs=1e-6)


class TestFourRoomGrid:
    """20x20 envelope divided by a full-height and full-width wall = four
    10x10 rooms, each exactly 100 m²."""

    GRID_SEGS = [
        # envelope
        ((0, 0), (20, 0)),
        ((20, 0), (20, 20)),
        ((20, 20), (0, 20)),
        ((0, 20), (0, 0)),
        # dividers crossing at (10, 10)
        ((10, 0), (10, 20)),
        ((0, 10), (20, 10)),
    ]

    def test_four_rooms_each_100(self):
        result = build_topology(_segments(self.GRID_SEGS), params=_params())
        assert len(result.rooms) == 4
        for room in result.rooms:
            assert room.area_m2 == pytest.approx(100.0, abs=1e-6)

    def test_total_area_equals_envelope(self):
        result = build_topology(_segments(self.GRID_SEGS), params=_params())
        total = sum(r.area_m2 for r in result.rooms)
        assert total == pytest.approx(400.0, abs=1e-6)

    def test_all_rings_valid(self):
        result = build_topology(_segments(self.GRID_SEGS), params=_params())
        for room in result.rooms:
            assert Polygon(room.polygon).is_valid


class TestOuterFaceDropped:
    def test_outer_face_not_reported_as_room(self):
        """A single square: the unbounded face must not appear as a room."""
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        result = build_topology(segs, params=_params())
        assert len(result.rooms) == 1

    def test_two_disjoint_rooms(self):
        """Two separate squares (disconnected components) → two rooms, no
        outer faces."""
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
            ((20, 0), (28, 0)),
            ((28, 0), (28, 8)),
            ((28, 8), (20, 8)),
            ((20, 8), (20, 0)),
        ])
        result = build_topology(segs, params=_params())
        assert len(result.rooms) == 2
        areas = sorted(r.area_m2 for r in result.rooms)
        assert areas[0] == pytest.approx(64.0, abs=1e-6)
        assert areas[1] == pytest.approx(100.0, abs=1e-6)


class TestExtensionHostSplit:
    """Regression: axis-extension splitting a host wall must keep BOTH halves.

    ``_extend_along_own_axis`` took numpy views of the host endpoints and then
    mutated the host row before building the far half — the far half collapsed
    to a zero-length segment, the wall lost a piece, and the room ring never
    closed (Phase 4 e2e fixture: rooms_detected == 0 on a clean square room).
    """

    def test_extension_split_preserves_host_far_half(self):
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
            # Spur whose free end must AXIS-EXTEND into the right wall
            # (projection parameter t > 0.98 puts it past the host-split
            # pass, exactly like a contour spur near a corner).
            ((5.0, 9.8), (9.85, 9.8)),
        ])
        params = TopologyParams(snap_tol_m=0.01, snapping_distance_m=0.30)
        result = build_topology(segs, params=params)

        # The right wall was split at the extension hit — both halves must
        # survive as non-degenerate segments covering the full 0 → 10 span.
        right = [
            s for s in result.snapped_segments
            if abs(s[0, 0] - 10.0) < 0.02 and abs(s[1, 0] - 10.0) < 0.02
        ]
        total_len = sum(float(np.linalg.norm(s[1] - s[0])) for s in right)
        assert total_len == pytest.approx(10.0, abs=0.05)
        for s in right:
            assert float(np.linalg.norm(s[1] - s[0])) > 1e-6

        # And the room must close: one big face, area ≈ envelope minus the
        # sliver between the spur and the top wall.
        assert len(result.rooms) == 1
        assert result.rooms[0].area_m2 == pytest.approx(99.5, abs=0.5)


class TestWrapperHeuristicWithSliver:
    """Regression: the 'biggest face is a wrapper' cap must not fire when the
    runner-up face is a sub-min-area sliver — that dropped the only genuine
    room and produced zero rooms on a clean single-room floor."""

    def test_single_room_with_corner_sliver_kept(self):
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
            # Corner chamfer → a 0.125 m² triangular face (below
            # min_room_area_m2) next to the 99.875 m² room face.
            ((9.5, 10), (10, 9.5)),
        ])
        result = build_topology(segs, params=_params())
        assert len(result.rooms) == 1
        assert result.rooms[0].area_m2 == pytest.approx(99.875, abs=1e-6)

    def test_wrapper_still_dropped_when_runner_up_is_real_room(self):
        """The cap must still fire in its intended case: a wrapper face
        alongside genuinely enclosed rooms (divider with a gap)."""
        segs = _segments([
            # 30x30 envelope
            ((0, 0), (30, 0)),
            ((30, 0), (30, 30)),
            ((30, 30), (0, 30)),
            ((0, 30), (0, 0)),
            # A fully-enclosed 4x4 room in one corner
            ((0, 4), (4, 4)),
            ((4, 4), (4, 0)),
        ])
        result = build_topology(segs, params=_params())
        # Wrapper (envelope minus corner room, 884 m²) > 50% of bbox and
        # > 3x the 16 m² room → dropped; only the small room remains.
        assert len(result.rooms) == 1
        assert result.rooms[0].area_m2 == pytest.approx(16.0, abs=1e-6)


class TestDanglingWall:
    def test_divider_spurs_do_not_inflate_area(self):
        """Regression for the minimum_cycle_basis ordering bug.

        Envelope 12x8 with a full divider at y=4 and five dangling cubicle
        spurs hanging from the divider into the lower room.  Both rooms are
        exactly 48 m².  The old cycle-basis code reported the lower room as
        96 m² (double) because the cycle node list was not in ring order.
        """
        segs = _segments([
            ((0, 0), (12, 0)),
            ((12, 0), (12, 8)),
            ((12, 8), (0, 8)),
            ((0, 8), (0, 0)),
            ((0, 4), (12, 4)),
            # dangling cubicle partitions off the divider
            ((2, 4), (2, 2)),
            ((4, 4), (4, 2)),
            ((6, 4), (6, 2)),
            ((8, 4), (8, 2)),
            ((10, 4), (10, 2)),
        ])
        params = TopologyParams(snap_tol_m=0.01, snapping_distance_m=0.0)
        result = build_topology(segs, params=params)
        assert len(result.rooms) == 2
        for room in result.rooms:
            assert room.area_m2 == pytest.approx(48.0, abs=1e-6)
            assert Polygon(room.polygon).is_valid

    def test_spur_does_not_break_room(self):
        """A room with a dangling partial partition inside: the room area
        must still be the full envelope area (the spur encloses nothing)."""
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
            # dangling spur from the bottom wall, doesn't reach the top
            ((5, 0), (5, 4)),
        ])
        # Disable snapping_distance so the spur is NOT extended to close.
        params = TopologyParams(snap_tol_m=0.01, snapping_distance_m=0.0)
        result = build_topology(segs, params=params)
        assert len(result.rooms) == 1
        assert result.rooms[0].area_m2 == pytest.approx(100.0, abs=1e-6)


class TestPhase2Continuity:
    """Gap-close + envelope projection (Phase 2)."""

    def test_collinear_gap_closes_room(self):
        """15 cm break in one side of a square must auto-bridge into one room."""
        segs = _segments([
            ((0, 0), (4.925, 0)),
            ((5.075, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.05,  # axis-extend alone cannot bridge 0.15 m
            gap_close_tol_m=0.30,
            close_gaps=True,
        )
        result = build_topology(segs, params=params)
        assert result.n_gaps_closed >= 1
        assert len(result.rooms) == 1
        assert result.rooms[0].area_m2 == pytest.approx(100.0, abs=0.5)

    def test_wide_door_gap_not_closed(self):
        """A 1 m doorway must stay open — continuity must not invent a wall."""
        segs = _segments([
            ((0, 0), (4.0, 0)),
            ((5.0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.05,
            gap_close_tol_m=0.30,
            close_gaps=True,
        )
        result = build_topology(segs, params=params)
        assert result.n_gaps_closed == 0
        assert len(result.rooms) == 0

    def test_envelope_projects_near_hull_ends(self):
        """Free ends ~15 cm inside the right facade move onto the hull."""
        # Inset from the envelope so distance-to-hull > 0; axis points +X.
        segs = _segments([
            ((0.0, 0.15), (9.85, 0.15)),
            ((0.0, 0.15), (0.0, 9.85)),
            ((0.0, 9.85), (9.85, 9.85)),
        ])
        envelope = np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            dtype=float,
        )
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.0,  # disable axis-extend so only envelope acts
            gap_close_tol_m=0.30,
            envelope_project_tol_m=0.30,
            inject_envelope=False,
            close_gaps=True,
        )
        result = build_topology(segs, params=params, envelope_xy=envelope)
        assert result.n_envelope_projections >= 1
        xs = [float(s[1][0]) for s in result.snapped_segments] + [
            float(s[0][0]) for s in result.snapped_segments
        ]
        assert any(abs(x - 10.0) < 1e-3 for x in xs)

    def test_close_gaps_off_leaves_collinear_open(self):
        segs = _segments([
            ((0, 0), (4.925, 0)),
            ((5.075, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.05,
            close_gaps=False,
        )
        result = build_topology(segs, params=params)
        assert result.n_gaps_closed == 0
        assert len(result.rooms) == 0

    def test_l_corner_closes(self):
        """Two walls that miss at a corner by 20 cm form an L after continuity."""
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 9.8)),       # stops 20 cm short of top-right
            ((0, 10), (9.8, 10)),       # stops 20 cm short of top-right
            ((0, 10), (0, 0)),
        ])
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.05,
            gap_close_tol_m=0.30,
            corner_close_tol_m=0.40,
            close_gaps=True,
        )
        result = build_topology(segs, params=params)
        assert result.n_gaps_closed >= 1
        assert len(result.rooms) == 1
        assert result.rooms[0].area_m2 == pytest.approx(100.0, abs=1.0)

    def test_envelope_bridge_closes_hull_gap(self):
        """Two free ends already on the hull get a short chord between them."""
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (5.4, 10)),
            ((4.6, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        envelope = np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            dtype=float,
        )
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.05,
            gap_close_tol_m=0.30,          # 0.8 m gap > collinear tol
            envelope_bridge_tol_m=1.0,
            inject_envelope=False,        # isolate bridge behaviour
            close_gaps=True,
        )
        result = build_topology(segs, params=params, envelope_xy=envelope)
        assert result.n_gaps_closed >= 1
        assert len(result.rooms) == 1

    def test_envelope_injection_closes_open_shell(self):
        """Interior walls + envelope ring → rooms even when exterior is open."""
        # Three walls of a square; missing the right side.  Envelope supplies it.
        segs = _segments([
            ((0, 0), (10, 0)),
            ((0, 0), (0, 10)),
            ((0, 10), (10, 10)),
            # interior divider
            ((0, 5), (8, 5)),
        ])
        envelope = np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            dtype=float,
        )
        params = TopologyParams(
            snap_tol_m=0.05,
            snapping_distance_m=0.30,
            inject_envelope=True,
            close_gaps=True,
            min_room_area_m2=1.0,
        )
        result = build_topology(segs, params=params, envelope_xy=envelope)
        # Only the missing right facade should be injected (other three
        # envelope edges are already covered by finished walls).
        assert result.n_envelope_edges_injected >= 1
        assert len(result.rooms) >= 1
        assert sum(r.area_m2 for r in result.rooms) == pytest.approx(100.0, abs=5.0)
        # Finished walls for export must NOT include the virtual envelope edges.
        assert result.finished_wall_segments is not None
        assert len(result.finished_wall_segments) <= len(result.snapped_segments)

    def test_finished_walls_meet_at_corners(self):
        """Cloud2BIM write-back: short T-gap walls become continuous for export."""
        # Square with a 25 cm gap on the top edge — axis-extend / gap-close
        # must finish it so export walls enclose the room.
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (5.25, 10)),
            ((4.75, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.60,
            gap_close_tol_m=0.55,
            inject_envelope=False,
            close_gaps=True,
        )
        result = build_topology(segs, params=params)
        assert result.finished_wall_segments is not None
        assert len(result.rooms) == 1
        # Every finished endpoint should sit on a degree ≥ 2 junction
        # (no dangling free ends on a closed square).
        assert result.n_dangling_after == 0


class TestCornerFinishPhase3:
    """Phase 3: BricsCAD-style EXTEND / TRIM so L/T corners actually meet."""

    def test_l_corner_overshoot_trimmed(self):
        """Two walls that overshoot past their L intersection are trimmed."""
        # Horizontal runs past the vertical; vertical runs past the horizontal.
        # Ideal corner at (5, 5).  A lone L keeps 2 far-end dangling points.
        segs = _segments([
            ((0, 5), (5.40, 5)),   # overshoot +0.40 m past x=5
            ((5, 0), (5, 5.35)),   # overshoot +0.35 m past y=5
        ])
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.60,
            gap_close_tol_m=0.55,
            corner_close_tol_m=0.60,
            inject_envelope=False,
            close_gaps=True,
            manhattan_join_tol_deg=8.0,
            min_room_area_m2=0.5,
        )
        result = build_topology(segs, params=params)
        assert result.finished_wall_segments is not None
        finished = result.finished_wall_segments
        # No leftover stub beyond the corner — at most 2 segments.
        assert len(finished) == 2
        # Far ends remain open; the corner itself is shared (deg ≥ 2).
        assert result.n_dangling_after == 2
        corner_hits = 0
        for s in finished:
            for end in (0, 1):
                e = s[end]
                if abs(float(e[0]) - 5.0) < 0.05 and abs(float(e[1]) - 5.0) < 0.05:
                    corner_hits += 1
        assert corner_hits == 2

    def test_l_corner_undershoot_extended(self):
        """Degree-1 undershoot (~40 cm) reaches the perpendicular partner."""
        segs = _segments([
            ((0, 5), (4.60, 5)),   # stops 40 cm short of x=5
            ((5, 0), (5, 4.55)),   # stops 45 cm short of y=5
        ])
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.60,
            gap_close_tol_m=0.55,
            corner_close_tol_m=0.60,
            inject_envelope=False,
            close_gaps=True,
            manhattan_join_tol_deg=8.0,
        )
        result = build_topology(segs, params=params)
        assert result.finished_wall_segments is not None
        finished = result.finished_wall_segments
        assert len(finished) == 2
        assert result.n_dangling_after == 2
        corner_hits = 0
        for s in finished:
            for end in (0, 1):
                e = s[end]
                if abs(float(e[0]) - 5.0) < 0.08 and abs(float(e[1]) - 5.0) < 0.08:
                    corner_hits += 1
        assert corner_hits == 2

    def test_manhattan_snap_before_join(self):
        """Slightly skewed segments snap to H/V then form a clean L."""
        segs = _segments([
            ((0.0, 0.05), (5.0, 0.0)),
            ((0.05, 0.0), (0.0, 5.0)),
        ])
        params = TopologyParams(
            snap_tol_m=0.08,
            snapping_distance_m=0.60,
            corner_close_tol_m=0.60,
            inject_envelope=False,
            close_gaps=True,
            manhattan_join_tol_deg=8.0,
        )
        result = build_topology(segs, params=params)
        finished = result.finished_wall_segments
        assert finished is not None
        assert result.n_dangling_after == 2
        # After Manhattan + join, both segments should be nearly axis-aligned.
        for s in finished:
            d = s[1] - s[0]
            ang = abs(float(np.degrees(np.arctan2(d[1], d[0])))) % 180.0
            to_h = min(ang, 180.0 - ang)
            to_v = abs(ang - 90.0)
            assert min(to_h, to_v) < 1.0

    def test_closed_square_no_dangling(self):
        """Four walls with L-gaps close into a watertight square."""
        segs = _segments([
            ((0.0, 0.0), (9.55, 0.0)),   # short of (10,0)
            ((10.0, 0.0), (10.0, 9.60)),
            ((10.0, 10.0), (0.40, 10.0)),
            ((0.0, 10.0), (0.0, 0.35)),
        ])
        params = TopologyParams(
            snap_tol_m=0.05,
            snapping_distance_m=0.60,
            gap_close_tol_m=0.55,
            corner_close_tol_m=0.60,
            inject_envelope=False,
            close_gaps=True,
            manhattan_join_tol_deg=8.0,
            min_room_area_m2=1.0,
        )
        result = build_topology(segs, params=params)
        assert len(result.rooms) == 1
        assert result.rooms[0].area_m2 == pytest.approx(100.0, abs=2.0)
        assert result.n_dangling_after == 0

    def test_spiky_room_rejected_by_isoperimetric(self):
        """Needle faces below min_room_isoperimetric never reach rooms layer."""
        segs = _segments([
            ((0, 0), (10, 0)),
            ((10, 0), (10, 10)),
            ((10, 10), (0, 10)),
            ((0, 10), (0, 0)),
        ])
        params = TopologyParams(
            snap_tol_m=0.01,
            snapping_distance_m=0.05,
            inject_envelope=False,
            min_room_isoperimetric=0.20,
        )
        result = build_topology(segs, params=params)
        assert len(result.rooms) == 1
        a = result.rooms[0].area_m2
        p = result.rooms[0].perimeter_m
        iq = 4.0 * np.pi * a / (p * p)
        assert iq >= 0.20
