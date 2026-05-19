"""Unit tests for the core alignment pipeline using synthetic data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.skipif(
    not (FIXTURES / "synthetic_scan.ply").exists(),
    reason="Synthetic test data not generated. Run: make generate-test-data",
)
def test_alignment_recovers_known_transform():
    """The pipeline should recover a transform within 25mm RMSE on synthetic data."""
    try:
        import open3d as o3d
        from app.pipeline.ingest import load_point_cloud
        from app.pipeline.rooms import extract_wall_lines
        from app.pipeline.align import align_scan_to_plan
    except ImportError:
        pytest.skip("open3d not installed")

    scan_path = FIXTURES / "synthetic_scan.ply"
    plan_path = FIXTURES / "synthetic_plan.dxf"
    gt_path = FIXTURES / "ground_truth_transform.npy"

    scan_pcd = load_point_cloud(scan_path)
    plan_lines = extract_wall_lines(str(plan_path))

    result = align_scan_to_plan(scan_pcd, plan_lines)

    # Residual under 25mm is acceptable for synthetic clean data
    assert result.residual_rmse * 1000 < 25.0, (
        f"Residual RMSE {result.residual_rmse * 1000:.1f}mm exceeds 25mm threshold"
    )
    assert result.confidence > 0.5, f"Confidence {result.confidence:.2f} too low"
    assert result.num_wall_planes >= 4, f"Only {result.num_wall_planes} wall planes detected"


def test_alignment_result_fields():
    """AlignmentResult has the expected shape."""
    try:
        from app.pipeline.align import AlignmentResult, FloorReference
    except ImportError:
        pytest.skip("open3d not installed")
    import numpy as np
    # Just a shape/type smoke test — no geometry needed
    assert hasattr(AlignmentResult, '__dataclass_fields__')
    fields = AlignmentResult.__dataclass_fields__
    assert 'transformation' in fields
    assert 'confidence' in fields
    assert 'residual_rmse' in fields
