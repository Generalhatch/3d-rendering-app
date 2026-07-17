"""Structured pipeline warnings (Phase 4).

Every silent fallback must produce a stable-coded warning record.  These
tests cover the collector itself and the one fallback that needed new
provenance plumbing: the Manhattan-filter bail in the regularizer.
"""
from __future__ import annotations

import numpy as np

from app.vectorize import regularize
from app.vectorize.pipeline_warnings import (
    KNOWN_CODES,
    PipelineWarning,
    WarningCollector,
)


class TestWarningCollector:
    def test_add_and_serialize(self):
        c = WarningCollector()
        assert len(c) == 0

        c.add("ceiling_band_sparse", "slice", "gate skipped")
        c.add("manhattan_bail", "regularize", "filter bailed")

        assert len(c) == 2
        assert c.codes() == ["ceiling_band_sparse", "manhattan_bail"]
        assert c.to_json_list() == [
            {"code": "ceiling_band_sparse", "stage": "slice",
             "message": "gate skipped"},
            {"code": "manhattan_bail", "stage": "regularize",
             "message": "filter bailed"},
        ]

    def test_warning_record_shape(self):
        w = PipelineWarning(code="envelope_failed", stage="envelope", message="x")
        assert w.to_json_dict() == {
            "code": "envelope_failed", "stage": "envelope", "message": "x",
        }

    def test_pipeline_codes_are_registered(self):
        """Every code the pipeline emits must be in the reference list."""
        import inspect
        from app.vectorize import pipeline as pipeline_mod
        src = inspect.getsource(pipeline_mod)
        import re
        emitted = set(re.findall(r'warnings\.add\(\s*\n?\s*"([a-z_]+)"', src))
        assert emitted, "pipeline should emit at least one warning code"
        unknown = emitted - KNOWN_CODES
        assert not unknown, f"unregistered warning codes in pipeline: {unknown}"


class TestManhattanBailProvenance:
    def _bail_segments(self) -> np.ndarray:
        """Short segments all at 44° — with a 0.5° tolerance the dominant
        axis lands on the 45° bin centre, so EVERY segment misses both axes
        and the filter would wipe the input → bail path."""
        d = 0.5 * np.array([np.cos(np.deg2rad(44.0)), np.sin(np.deg2rad(44.0))])
        starts = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        return np.stack([starts, starts + d], axis=1)

    def test_bail_flag_set_and_input_preserved(self):
        segs = self._bail_segments()
        params = regularize.RegularizeParams(
            drop_short_below_m=0.10,
            manhattan_snap=True,
            manhattan_tolerance_deg=0.5,
            merge_collinear=False,
            protect_curve_chains=False,
            keep_diagonal_min_length_m=1.0,   # 0.5 m segments stay below this
        )

        result = regularize.regularize_with_provenance(segs, params)

        assert result.manhattan_bailed is True
        # Bail = the input passes through unchanged (legacy behaviour).
        np.testing.assert_allclose(result.kept, segs, atol=1e-12)

    def test_no_bail_on_normal_input(self):
        """A clean Manhattan grid never bails."""
        segs = np.array([
            [[0.0, 0.0], [10.0, 0.0]],
            [[0.0, 0.0], [0.0, 10.0]],
            [[10.0, 0.0], [10.0, 10.0]],
        ])
        result = regularize.regularize_with_provenance(
            segs, regularize.RegularizeParams(drop_short_below_m=0.1),
        )
        assert result.manhattan_bailed is False
        assert len(result.kept) == 3

    def test_no_bail_when_snap_disabled(self):
        segs = self._bail_segments()
        result = regularize.regularize_with_provenance(
            segs,
            regularize.RegularizeParams(
                drop_short_below_m=0.1, manhattan_snap=False,
            ),
        )
        assert result.manhattan_bailed is False
