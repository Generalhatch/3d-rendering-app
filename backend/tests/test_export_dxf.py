"""Tests for write_aligned_dxf — the export must actually apply the
alignment transform to the DXF entities.

Guards against the P0 bug where the function's docstring claimed "all
entities transformed by the alignment matrix" but the matrix was never
applied — downstream CAD users received an unmodified copy of the plan.
"""
from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import numpy as np
import pytest

from app.pipeline.export import write_aligned_dxf


@pytest.fixture
def simple_plan_dxf(tmp_path: Path) -> Path:
    """A plan with one known LINE, LWPOLYLINE, and CIRCLE."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 5), (0, 5)], close=True)
    msp.add_circle(center=(5, 5), radius=2.0)
    path = tmp_path / "plan.dxf"
    doc.saveas(str(path))
    return path


def _translation(tx: float, ty: float, tz: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (tx, ty, tz)
    return T


def _rotation_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    T = np.eye(4)
    T[0, 0] = math.cos(a)
    T[0, 1] = -math.sin(a)
    T[1, 0] = math.sin(a)
    T[1, 1] = math.cos(a)
    return T


class TestWriteAlignedDxf:
    def test_translation_applied_to_line(self, simple_plan_dxf, tmp_path):
        out = tmp_path / "aligned.dxf"
        write_aligned_dxf(simple_plan_dxf, _translation(100.0, -50.0), out)

        doc = ezdxf.readfile(str(out))
        lines = doc.modelspace().query("LINE")
        assert len(lines) == 1
        start = np.array(lines[0].dxf.start)[:2]
        end = np.array(lines[0].dxf.end)[:2]
        np.testing.assert_allclose(start, [100.0, -50.0], atol=1e-9)
        np.testing.assert_allclose(end, [110.0, -50.0], atol=1e-9)

    def test_rotation_applied_to_line(self, simple_plan_dxf, tmp_path):
        out = tmp_path / "aligned.dxf"
        write_aligned_dxf(simple_plan_dxf, _rotation_z(90.0), out)

        doc = ezdxf.readfile(str(out))
        line = doc.modelspace().query("LINE")[0]
        # (10, 0) rotated +90 deg about Z -> (0, 10)
        np.testing.assert_allclose(np.array(line.dxf.end)[:2], [0.0, 10.0], atol=1e-9)

    def test_combined_transform_applied_to_polyline(self, simple_plan_dxf, tmp_path):
        T = _translation(3.0, 4.0) @ _rotation_z(90.0)
        out = tmp_path / "aligned.dxf"
        write_aligned_dxf(simple_plan_dxf, T, out)

        doc = ezdxf.readfile(str(out))
        poly = doc.modelspace().query("LWPOLYLINE")[0]
        pts = np.array([(p[0], p[1]) for p in poly.get_points()])
        # (10, 0) -> rot90 -> (0, 10) -> +T -> (3, 14)
        expected_corner = np.array([3.0, 14.0])
        assert np.min(np.linalg.norm(pts - expected_corner, axis=1)) < 1e-6

    def test_circle_center_transformed(self, simple_plan_dxf, tmp_path):
        out = tmp_path / "aligned.dxf"
        write_aligned_dxf(simple_plan_dxf, _translation(-5.0, -5.0), out)

        doc = ezdxf.readfile(str(out))
        circle = doc.modelspace().query("CIRCLE")[0]
        np.testing.assert_allclose(
            np.array(circle.dxf.center)[:2], [0.0, 0.0], atol=1e-9
        )
        assert circle.dxf.radius == pytest.approx(2.0)

    def test_identity_preserves_geometry(self, simple_plan_dxf, tmp_path):
        out = tmp_path / "aligned.dxf"
        write_aligned_dxf(simple_plan_dxf, np.eye(4), out)

        doc = ezdxf.readfile(str(out))
        line = doc.modelspace().query("LINE")[0]
        np.testing.assert_allclose(np.array(line.dxf.start)[:2], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(np.array(line.dxf.end)[:2], [10.0, 0.0], atol=1e-9)

    def test_meta_note_not_transformed(self, simple_plan_dxf, tmp_path):
        """The traceability note is added AFTER the transform and must stay
        on its meta layer at the origin."""
        out = tmp_path / "aligned.dxf"
        write_aligned_dxf(simple_plan_dxf, _translation(1000.0, 1000.0), out)

        doc = ezdxf.readfile(str(out))
        notes = [e for e in doc.modelspace().query("TEXT")
                 if e.dxf.layer == "ALIGNAI_META"]
        assert len(notes) == 1
        np.testing.assert_allclose(
            np.array(notes[0].dxf.insert)[:2], [0.0, 0.0], atol=1e-9
        )
