"""Unit tests for _classify_room() — MJ1 confidence scoring.

These tests verify that the rule-based classifier returns the expected
category AND that the confidence score is within the documented range
(0.30–0.99) and correctly reflects how strongly the metrics match.
"""
from __future__ import annotations

import pytest


def _classify(area: float, ecc: float, sol: float):
    """Helper to import and call _classify_room directly."""
    from app.pipeline.scanplan import _classify_room
    return _classify_room(area, ecc, sol)


# ── Category correctness ──────────────────────────────────────────────────────

class TestCategoryAssignment:
    def test_hallway_classic(self):
        label, cat, conf = _classify(15.0, 0.92, 0.80)
        assert cat == "hallway", f"Expected hallway, got {cat}"
        assert label == "Hallway / Corridor"

    def test_hallway_borderline_ecc(self):
        # Just above the 0.85 eccentricity threshold
        label, cat, conf = _classify(10.0, 0.86, 0.75)
        assert cat == "hallway"

    def test_hallway_not_triggered_for_round_shape(self):
        # Round, compact room (ecc=0.2) should NOT be classified as hallway
        _, cat, _ = _classify(20.0, 0.20, 0.90)
        assert cat != "hallway"

    def test_hallway_not_triggered_for_large_area(self):
        # Very elongated but huge — lobby, not corridor
        _, cat, _ = _classify(120.0, 0.92, 0.80)
        assert cat != "hallway"   # area > 40 m² disqualifies corridor

    def test_bathroom_small(self):
        label, cat, conf = _classify(5.0, 0.40, 0.85)
        assert cat == "bathroom", f"Expected bathroom, got {cat}"
        assert label == "Bathroom / Storage"

    def test_bathroom_tiny(self):
        label, cat, conf = _classify(2.5, 0.30, 0.90)
        assert cat == "bathroom"

    def test_bathroom_boundary(self):
        # Exactly 7.0 m² — just inside bathroom threshold
        label, cat, conf = _classify(6.99, 0.40, 0.85)
        assert cat == "bathroom"

    def test_bathroom_not_triggered_for_normal_size(self):
        _, cat, _ = _classify(15.0, 0.30, 0.90)
        assert cat != "bathroom"

    def test_lobby_large_convex(self):
        label, cat, conf = _classify(120.0, 0.20, 0.92)
        assert cat == "common", f"Expected common, got {cat}"
        assert label == "Open Plan / Lobby"

    def test_lobby_requires_convexity(self):
        # Large but low solidity (irregular shape) — should NOT be lobby
        _, cat, _ = _classify(100.0, 0.20, 0.50)
        assert cat != "common"

    def test_conference_medium_large(self):
        label, cat, conf = _classify(55.0, 0.40, 0.85)
        assert cat == "office", f"Expected office, got {cat}"
        assert label == "Conference Room"

    def test_conference_boundary(self):
        # Just above 40 m² threshold
        label, cat, conf = _classify(41.0, 0.35, 0.82)
        assert cat == "office"
        assert label == "Conference Room"

    def test_irregular_low_solidity(self):
        label, cat, conf = _classify(25.0, 0.40, 0.55)
        assert cat == "unknown", f"Expected unknown, got {cat}"
        assert label == "Irregular Space"

    def test_office_typical(self):
        label, cat, conf = _classify(18.0, 0.35, 0.88)
        assert cat == "office", f"Expected office, got {cat}"
        assert label == "Office / Meeting"

    def test_office_small_regular(self):
        label, cat, conf = _classify(10.0, 0.30, 0.85)
        assert cat == "office"


# ── Confidence score range ────────────────────────────────────────────────────

class TestConfidenceRange:
    """All returned confidences must be in [0.30, 0.99]."""

    @pytest.mark.parametrize("area,ecc,sol", [
        (5.0, 0.40, 0.85),     # bathroom
        (18.0, 0.35, 0.88),    # office
        (10.0, 0.92, 0.80),    # hallway
        (120.0, 0.20, 0.92),   # lobby
        (55.0, 0.40, 0.85),    # conference
        (25.0, 0.40, 0.55),    # irregular
        (7.0, 0.50, 0.75),     # borderline
        (1.0, 0.10, 0.99),     # extreme small
        (300.0, 0.05, 0.99),   # extreme large
    ])
    def test_confidence_in_range(self, area, ecc, sol):
        _, _, conf = _classify(area, ecc, sol)
        assert 0.30 <= conf <= 0.99, f"conf={conf:.3f} out of range for area={area}, ecc={ecc}, sol={sol}"


# ── Confidence should reflect certainty ──────────────────────────────────────

class TestConfidenceOrdering:
    def test_small_room_higher_confidence_than_borderline(self):
        # A 2 m² room is more clearly a bathroom than a 6.8 m² room
        _, _, conf_tiny = _classify(2.0, 0.30, 0.90)
        _, _, conf_borderline = _classify(6.8, 0.30, 0.90)
        assert conf_tiny > conf_borderline, (
            f"Tiny room confidence {conf_tiny:.3f} should exceed borderline {conf_borderline:.3f}"
        )

    def test_clearly_elongated_hallway_vs_borderline(self):
        _, _, conf_clear = _classify(10.0, 0.97, 0.80)
        _, _, conf_border = _classify(10.0, 0.86, 0.80)
        assert conf_clear >= conf_border, (
            f"Clear hallway {conf_clear:.3f} should have >= confidence than borderline {conf_border:.3f}"
        )

    def test_large_lobby_gains_confidence_with_area(self):
        # 200 m² lobby should have higher confidence than 85 m² lobby
        _, _, conf_huge = _classify(200.0, 0.15, 0.95)
        _, _, conf_small = _classify(85.0, 0.15, 0.90)
        assert conf_huge >= conf_small


# ── Return type correctness ───────────────────────────────────────────────────

class TestReturnTypes:
    def test_returns_three_tuple(self):
        result = _classify(15.0, 0.40, 0.85)
        assert len(result) == 3

    def test_label_is_string(self):
        label, _, _ = _classify(15.0, 0.40, 0.85)
        assert isinstance(label, str) and len(label) > 0

    def test_category_valid(self):
        _, cat, _ = _classify(15.0, 0.40, 0.85)
        assert cat in {"hallway", "bathroom", "common", "office", "unknown"}

    def test_confidence_is_float(self):
        _, _, conf = _classify(15.0, 0.40, 0.85)
        assert isinstance(conf, float)
