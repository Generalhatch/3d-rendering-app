"""Tests for the computed scan-only confidence score.

Guards against the P0 bug where scan-only mode (the mode with the LEAST
verification — no plan to align against) hardcoded ``confidence_pct: 100``.
"""
from __future__ import annotations

import pytest

from app.pipeline.confidence import (
    compute_confidence_pct,
    compute_scan_only_confidence,
)


def _closed_room(area: float = 20.0) -> dict:
    return {
        "polygon_2d": [[0, 0], [5, 0], [5, 4], [0, 4]],
        "area_m2": area,
    }


def _broken_room() -> dict:
    return {"polygon_2d": [[0, 0], [5, 0]], "area_m2": 0.0}


class TestScanOnlyConfidence:
    def test_never_reports_100_percent(self):
        """Even a best-case scan-only run must not claim certainty."""
        conf = compute_scan_only_confidence(
            num_wall_planes=40,
            plan_source="3d_planes",
            merge_strategy="single",
            rooms=[_closed_room() for _ in range(10)],
        )
        assert conf < 1.0
        assert compute_confidence_pct(conf) < 100

    def test_good_run_scores_high(self):
        conf = compute_scan_only_confidence(
            num_wall_planes=12,
            plan_source="3d_planes",
            merge_strategy="single",
            rooms=[_closed_room() for _ in range(6)],
        )
        assert conf >= 0.8

    def test_more_wall_planes_scores_higher(self):
        low = compute_scan_only_confidence(
            num_wall_planes=4, plan_source="3d_planes",
            merge_strategy="single", rooms=[_closed_room()],
        )
        high = compute_scan_only_confidence(
            num_wall_planes=8, plan_source="3d_planes",
            merge_strategy="single", rooms=[_closed_room()],
        )
        assert high > low

    def test_2d_projection_fallback_scores_lower(self):
        planes = compute_scan_only_confidence(
            num_wall_planes=8, plan_source="3d_planes",
            merge_strategy="single", rooms=[_closed_room()],
        )
        projection = compute_scan_only_confidence(
            num_wall_planes=2, plan_source="2d_projection",
            merge_strategy="single", rooms=[_closed_room()],
        )
        assert projection < planes

    def test_icp_registration_scores_lower_than_preregistered(self):
        pre = compute_scan_only_confidence(
            num_wall_planes=8, plan_source="3d_planes",
            merge_strategy="concatenate", rooms=[_closed_room()],
        )
        icp = compute_scan_only_confidence(
            num_wall_planes=8, plan_source="3d_planes",
            merge_strategy="icp_registered", rooms=[_closed_room()],
        )
        assert icp < pre

    def test_broken_rooms_lower_score(self):
        good = compute_scan_only_confidence(
            num_wall_planes=8, plan_source="3d_planes",
            merge_strategy="single",
            rooms=[_closed_room() for _ in range(4)],
        )
        broken = compute_scan_only_confidence(
            num_wall_planes=8, plan_source="3d_planes",
            merge_strategy="single",
            rooms=[_broken_room() for _ in range(4)],
        )
        assert broken < good

    def test_no_rooms_scores_low(self):
        conf = compute_scan_only_confidence(
            num_wall_planes=8, plan_source="3d_planes",
            merge_strategy="single", rooms=[],
        )
        assert conf <= 0.5

    def test_worst_case_scores_very_low(self):
        conf = compute_scan_only_confidence(
            num_wall_planes=0, plan_source="2d_projection",
            merge_strategy="icp_registered", rooms=[],
        )
        assert conf < 0.25
        assert compute_confidence_pct(conf) < 25

    def test_bounded_zero_to_one(self):
        for planes in (0, 1, 50):
            for src in ("3d_planes", "2d_projection"):
                for strat in ("single", "concatenate", "icp_registered", "weird"):
                    conf = compute_scan_only_confidence(
                        num_wall_planes=planes, plan_source=src,
                        merge_strategy=strat, rooms=[_closed_room()],
                    )
                    assert 0.0 <= conf < 1.0


class TestRunnerNoHardcodedConfidence:
    def test_runner_source_has_no_hardcoded_100(self):
        """The scan-only branch must not reintroduce a literal 100% score."""
        from pathlib import Path
        import app.pipeline.runner as runner_mod
        src = Path(runner_mod.__file__).read_text()
        assert '"confidence_pct": 100' not in src
        assert '"confidence": 1.0' not in src
