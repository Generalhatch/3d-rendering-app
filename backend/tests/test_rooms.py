"""Unit tests for room extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.skipif(
    not (FIXTURES / "synthetic_plan.dxf").exists(),
    reason="Synthetic test data not generated. Run: make generate-test-data",
)
def test_room_extraction_finds_rooms():
    try:
        from app.pipeline.rooms import extract_rooms
    except ImportError:
        pytest.skip("ezdxf/shapely not installed")

    rooms = extract_rooms(str(FIXTURES / "synthetic_plan.dxf"))
    assert len(rooms) >= 1, "Expected at least 1 room from synthetic plan"


@pytest.mark.skipif(
    not (FIXTURES / "synthetic_plan.dxf").exists(),
    reason="Synthetic test data not generated. Run: make generate-test-data",
)
def test_room_extraction_has_required_fields():
    try:
        from app.pipeline.rooms import extract_rooms
    except ImportError:
        pytest.skip("ezdxf/shapely not installed")

    rooms = extract_rooms(str(FIXTURES / "synthetic_plan.dxf"))
    for r in rooms:
        assert r.id.startswith("room-")
        assert isinstance(r.label, str) and len(r.label) > 0
        assert r.category in {"office", "bathroom", "hallway", "common", "unknown"}
        assert len(r.polygon_2d) >= 3
        assert r.area_m2 > 0


def test_extract_wall_lines_from_synthetic():
    try:
        from app.pipeline.rooms import extract_wall_lines
    except ImportError:
        pytest.skip("ezdxf not installed")

    if not (FIXTURES / "synthetic_plan.dxf").exists():
        pytest.skip("Synthetic test data not generated")

    lines = extract_wall_lines(str(FIXTURES / "synthetic_plan.dxf"))
    assert lines.shape[1] == 2
    assert lines.shape[2] == 2
    assert len(lines) >= 6  # 7 walls in our synthetic plan
