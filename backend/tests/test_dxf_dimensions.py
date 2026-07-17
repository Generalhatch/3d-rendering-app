"""Phase 3: DIMENSION entities in the DXF export (app.sheet.dimensions).

The dimensions are measured on the OUTER face of each exterior wall of the
canonical FloorGeometry — for the two-suite reference floor those faces are
hand-computable: horizontal exterior walls are 20 m end-to-end, vertical
ones 10 m; the demising wall must NOT be dimensioned.
"""
from __future__ import annotations

import ezdxf
import numpy as np
import pytest

from app.sheet.dimensions import (
    DIMENSIONS_LAYER,
    add_dimensions_to_dxf,
    add_wall_dimensions,
    outer_face,
)
from tests.reference_floors import two_suite_floor


class TestOuterFace:
    def test_bottom_wall_outer_face_is_below(self):
        floor = two_suite_floor()
        wall = floor.wall_by_id("w_bottom")   # centerline (0,0)→(20,0), t=0.30
        face = outer_face(wall, interior_point=np.array([10.0, 5.0]))
        assert face[:, 1] == pytest.approx([-0.15, -0.15], abs=1e-9)
        assert float(np.linalg.norm(face[1] - face[0])) == pytest.approx(20.0, abs=1e-9)

    def test_right_wall_outer_face_is_outboard(self):
        floor = two_suite_floor()
        wall = floor.wall_by_id("w_right")    # centerline (20,0)→(20,10)
        face = outer_face(wall, interior_point=np.array([10.0, 5.0]))
        assert face[:, 0] == pytest.approx([20.15, 20.15], abs=1e-9)


class TestAddWallDimensions:
    def _doc_with_dims(self):
        doc = ezdxf.new(dxfversion="R2018", setup=True)
        floor = two_suite_floor()
        n = add_wall_dimensions(doc.modelspace(), floor)
        return doc, n

    def test_one_dimension_per_exterior_wall(self):
        doc, n = self._doc_with_dims()
        assert n == 4
        dims = doc.modelspace().query("DIMENSION")
        assert len(dims) == 4

    def test_dimensions_on_dimensions_layer(self):
        doc, _ = self._doc_with_dims()
        assert DIMENSIONS_LAYER in doc.layers
        for dim in doc.modelspace().query("DIMENSION"):
            assert dim.dxf.layer == DIMENSIONS_LAYER

    def test_measurements_match_outer_face_lengths(self):
        doc, _ = self._doc_with_dims()
        measurements = sorted(
            float(d.get_measurement()) for d in doc.modelspace().query("DIMENSION")
        )
        # Two vertical exterior walls (10 m faces), two horizontal (20 m).
        assert measurements == pytest.approx([10.0, 10.0, 20.0, 20.0], abs=1e-9)

    def test_demising_wall_not_dimensioned(self):
        doc, _ = self._doc_with_dims()
        # No dimension measures ~10.0 at the demising x=8 location; count
        # already proves only 4 exterior dims exist, so verify defpoints
        # never sit on the demising centerline x=8.
        for dim in doc.modelspace().query("DIMENSION"):
            for attr in ("defpoint2", "defpoint3"):
                p = dim.dxf.get(attr, None)
                if p is not None:
                    assert abs(float(p[0]) - 8.0) > 0.5 or abs(float(p[1])) > 11.0

    def test_short_walls_skipped(self):
        doc = ezdxf.new(dxfversion="R2018", setup=True)
        floor = two_suite_floor()
        n = add_wall_dimensions(
            doc.modelspace(), floor, min_wall_length_m=15.0,
        )
        # Only the two 20 m horizontal faces survive a 15 m minimum.
        assert n == 2

    def test_dimension_lines_placed_outside(self):
        doc, _ = self._doc_with_dims()
        # Every rendered dimension block's text sits outside the envelope
        # bbox (offset 0.8 m outward from the outer faces).
        for dim in doc.modelspace().query("DIMENSION"):
            mid = dim.dxf.text_midpoint
            inside = (-0.15 < mid[0] < 20.15) and (-0.15 < mid[1] < 10.15)
            assert not inside


class TestAddDimensionsToDxfFile:
    def test_roundtrip_file(self, tmp_path):
        path = tmp_path / "deliverable.dxf"
        doc = ezdxf.new(dxfversion="R2018", setup=True)
        doc.modelspace().add_lwpolyline([(0, 0), (20, 0)])
        doc.saveas(str(path))

        n = add_dimensions_to_dxf(path, two_suite_floor())
        assert n == 4

        reloaded = ezdxf.readfile(str(path))
        dims = reloaded.modelspace().query("DIMENSION")
        assert len(dims) == 4
        measurements = sorted(float(d.get_measurement()) for d in dims)
        assert measurements == pytest.approx([10.0, 10.0, 20.0, 20.0], abs=1e-9)
        # Original geometry untouched.
        assert len(reloaded.modelspace().query("LWPOLYLINE")) == 1
