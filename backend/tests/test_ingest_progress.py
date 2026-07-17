"""Incremental progress emission during chunked LAS loading.

Large-scan UX bug: the ingest stage emitted nothing between "Loading scan…"
(progress 0.05) and "Loaded N points" (0.12) — a 689 MB chunked LAS load is
silent for many minutes and the job looks hung at 5%.  The chunked loader
must invoke ``on_progress(points_read, total_points)`` after every chunk,
and the vectorize pipeline must map that onto the [0.05, 0.12] progress
span with throttling (so the bounded SSE replay buffer isn't flooded).

Phase 4.5 additions (the 680 MB / 22 GB RSS incident):

- xyz preallocated from the header count and filled per chunk — no
  list-of-chunks + vstack double allocation.  Header short/long reads must
  still produce output byte-identical to the unchunked loader.
- ``load_colors=False`` skips color/intensity accumulation entirely
  (geometry-only load for the vectorize pipeline); with colors ON, raw
  values stay uint16 until after masking and the final colors must match
  the unchunked loader exactly.
- ``on_phase(name)`` assembly callbacks fire after the last chunk so the
  progress bar keeps moving through the (previously silent) assembly +
  unit-detection window past 12%.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import pytest

from app.pipeline.ingest import _load_las, _load_las_chunked, load_point_cloud
from app.vectorize import pipeline as pipeline_mod


def _write_las(path: Path, n: int, with_colors: bool = False) -> None:
    rng = np.random.default_rng(7)
    # Point format 2 carries uint16 RGB; format 0 has no color fields.
    header = laspy.LasHeader(point_format=2 if with_colors else 0, version="1.2")
    header.scales = (0.001, 0.001, 0.001)
    las = laspy.LasData(header)
    las.x = rng.uniform(0, 10, n)
    las.y = rng.uniform(0, 10, n)
    las.z = rng.uniform(0, 3, n)
    las.classification = np.zeros(n, dtype=np.uint8)
    if with_colors:
        las.red = rng.integers(0, 65536, n).astype(np.uint16)
        las.green = rng.integers(0, 65536, n).astype(np.uint16)
        las.blue = rng.integers(0, 65536, n).astype(np.uint16)
    las.write(str(path))


class TestChunkedLoaderProgress:
    def test_callback_fires_per_chunk_with_running_totals(self, tmp_path):
        """4000 points at chunk_points=1000 → exactly 4 callbacks with
        cumulative counts 1000, 2000, 3000, 4000 against total 4000."""
        path = tmp_path / "scan.las"
        _write_las(path, 4000)

        calls: list[tuple[int, int]] = []
        _load_las_chunked(
            path, chunk_points=1000,
            on_progress=lambda read, total: calls.append((read, total)),
        )

        assert [read for read, _ in calls] == [1000, 2000, 3000, 4000]
        assert all(total == 4000 for _, total in calls)

    def test_final_callback_reaches_total(self, tmp_path):
        """Ragged final chunk: 2500 points at chunk_points=1000 ends
        exactly at (2500, 2500)."""
        path = tmp_path / "scan.las"
        _write_las(path, 2500)

        calls: list[tuple[int, int]] = []
        _load_las_chunked(
            path, chunk_points=1000,
            on_progress=lambda read, total: calls.append((read, total)),
        )

        assert calls[-1] == (2500, 2500)
        reads = [read for read, _ in calls]
        assert reads == sorted(reads), "points_read must be monotonic"

    def test_progress_does_not_change_loaded_points(self, tmp_path):
        """The callback must be purely observational."""
        path = tmp_path / "scan.las"
        _write_las(path, 3000)

        silent = _load_las_chunked(path, chunk_points=1000)
        observed = _load_las_chunked(
            path, chunk_points=1000, on_progress=lambda *_: None,
        )
        np.testing.assert_allclose(
            np.asarray(silent.points), np.asarray(observed.points), atol=1e-9,
        )

    def test_load_point_cloud_accepts_on_progress(self, tmp_path):
        """The public entry point must accept the callback (small files
        load unchunked and simply never invoke it)."""
        path = tmp_path / "scan.las"
        _write_las(path, 2000)

        calls: list[tuple[int, int]] = []
        pcd = load_point_cloud(
            path, normalize_units=False,
            on_progress=lambda read, total: calls.append((read, total)),
        )
        assert len(pcd.points) == 2000
        assert calls == []  # < 500 MB → unchunked path, no chunk callbacks


_REAL_LASPY_OPEN = laspy.open


@contextmanager
def _lying_header_open(path: str, fake_count: int):
    """Wrap laspy.open so reader.header.point_count reports ``fake_count``
    while the chunk stream still delivers the file's real points."""
    with _REAL_LASPY_OPEN(path) as real:
        wrapper = SimpleNamespace(
            header=SimpleNamespace(point_count=fake_count),
            chunk_iterator=real.chunk_iterator,
        )
        yield wrapper


class TestChunkedPreallocation:
    """Phase 4.5: xyz preallocated from the header count, filled per chunk."""

    def test_chunked_matches_unchunked_with_colors(self, tmp_path):
        """Points AND colors byte-identical to the unchunked loader — raw
        uint16 color accumulation + post-mask normalization must not change
        the output."""
        path = tmp_path / "colored.las"
        _write_las(path, 3000, with_colors=True)

        unchunked = _load_las(path)
        chunked = _load_las_chunked(path, chunk_points=1000)

        np.testing.assert_array_equal(
            np.asarray(chunked.points), np.asarray(unchunked.points),
        )
        assert unchunked.has_colors() and chunked.has_colors()
        np.testing.assert_array_equal(
            np.asarray(chunked.colors), np.asarray(unchunked.colors),
        )

    def test_load_colors_false_skips_colors(self, tmp_path):
        """Geometry-only load: identical points, no colors — in BOTH the
        chunked and unchunked paths."""
        path = tmp_path / "colored.las"
        _write_las(path, 3000, with_colors=True)

        with_colors = _load_las_chunked(path, chunk_points=1000)
        for pcd in (
            _load_las_chunked(path, chunk_points=1000, load_colors=False),
            _load_las(path, load_colors=False),
        ):
            assert not pcd.has_colors()
            np.testing.assert_array_equal(
                np.asarray(pcd.points), np.asarray(with_colors.points),
            )

    def test_header_undercount_grows_buffer(self, tmp_path, monkeypatch):
        """A header that claims FEWER points than the file holds must not
        truncate or corrupt the load."""
        path = tmp_path / "scan.las"
        _write_las(path, 3000)
        truth = _load_las_chunked(path, chunk_points=1000)

        monkeypatch.setattr(
            laspy, "open", lambda p: _lying_header_open(p, fake_count=1000),
        )
        pcd = _load_las_chunked(path, chunk_points=1000)
        np.testing.assert_array_equal(
            np.asarray(pcd.points), np.asarray(truth.points),
        )

    def test_header_overcount_trims_buffer(self, tmp_path, monkeypatch):
        """A header that claims MORE points than the file holds must not
        leave uninitialized rows in the output."""
        path = tmp_path / "scan.las"
        _write_las(path, 2500)
        truth = _load_las_chunked(path, chunk_points=1000)

        monkeypatch.setattr(
            laspy, "open", lambda p: _lying_header_open(p, fake_count=9999),
        )
        pcd = _load_las_chunked(path, chunk_points=1000)
        assert len(pcd.points) == 2500
        np.testing.assert_array_equal(
            np.asarray(pcd.points), np.asarray(truth.points),
        )

    def test_mixed_classification_with_colors_matches_unchunked(self, tmp_path):
        """Colors must survive a REAL (non-trivial) classification mask —
        the uint16 raw buffer is masked before normalization."""
        n = 2000
        rng = np.random.default_rng(3)
        header = laspy.LasHeader(point_format=2, version="1.2")
        header.scales = (0.001, 0.001, 0.001)
        las = laspy.LasData(header)
        las.x = rng.uniform(0, 10, n)
        las.y = rng.uniform(0, 10, n)
        las.z = rng.uniform(0, 3, n)
        # 25% class 6 → file-global "class 6 only" regime drops 75%.
        cls = np.zeros(n, dtype=np.uint8)
        cls[::4] = 6
        las.classification = cls
        las.red = rng.integers(0, 65536, n).astype(np.uint16)
        las.green = rng.integers(0, 65536, n).astype(np.uint16)
        las.blue = rng.integers(0, 65536, n).astype(np.uint16)
        path = tmp_path / "masked.las"
        las.write(str(path))

        unchunked = _load_las(path)
        chunked = _load_las_chunked(path, chunk_points=300)
        assert len(chunked.points) == n // 4
        np.testing.assert_array_equal(
            np.asarray(chunked.points), np.asarray(unchunked.points),
        )
        np.testing.assert_array_equal(
            np.asarray(chunked.colors), np.asarray(unchunked.colors),
        )


class TestAssemblyPhaseCallbacks:
    """Phase 4.5: the silent post-chunk assembly window must report phases."""

    def test_phases_fire_in_order_after_chunks(self, tmp_path):
        path = tmp_path / "colored.las"
        _write_las(path, 3000, with_colors=True)

        events: list[str] = []
        _load_las_chunked(
            path, chunk_points=1000,
            on_progress=lambda *_: events.append("chunk"),
            on_phase=lambda name: events.append(name),
        )
        assert events == ["chunk", "chunk", "chunk", "classify", "assemble", "colors"]

    def test_no_colors_phase_when_geometry_only(self, tmp_path):
        path = tmp_path / "colored.las"
        _write_las(path, 2000, with_colors=True)

        phases: list[str] = []
        _load_las_chunked(
            path, chunk_points=1000, load_colors=False,
            on_phase=phases.append,
        )
        assert phases == ["classify", "assemble"]

    def test_load_point_cloud_fires_unit_detect_phase(self, tmp_path):
        path = tmp_path / "scan.las"
        _write_las(path, 2000)

        phases: list[str] = []
        load_point_cloud(path, normalize_units=True, on_phase=phases.append)
        assert phases[-1] == "unit_detect"

    def test_phase_callback_is_observational(self, tmp_path):
        path = tmp_path / "scan.las"
        _write_las(path, 3000)
        silent = _load_las_chunked(path, chunk_points=1000)
        observed = _load_las_chunked(
            path, chunk_points=1000, on_phase=lambda *_: None,
        )
        np.testing.assert_array_equal(
            np.asarray(silent.points), np.asarray(observed.points),
        )


class TestPipelineIngestEmitter:
    """The vectorize stage-1 wrapper: fraction → SSE progress in [0.05, 0.12]."""

    @pytest.fixture
    def captured(self, monkeypatch):
        events: list[tuple[str, str, str, float]] = []
        monkeypatch.setattr(
            pipeline_mod, "publish",
            lambda job_id, stage, message, progress:
                events.append((job_id, stage, message, progress)),
        )
        return events

    def test_progress_mapped_into_ingest_span(self, captured):
        cb = pipeline_mod._make_ingest_progress_emitter("job-1", "scan.laz")
        total = 10_000_000
        cb(0, total)
        cb(5_000_000, total)
        cb(10_000_000, total)

        assert len(captured) == 3
        stages = {e[1] for e in captured}
        assert stages == {"ingest"}
        progresses = [e[3] for e in captured]
        # Exact interpolation: lo + (hi - lo) * frac with lo=0.05, hi=0.12.
        assert progresses[0] == pytest.approx(0.05)
        assert progresses[1] == pytest.approx(0.085)
        assert progresses[2] == pytest.approx(0.12)
        # All events stay strictly inside the stage-1 window.
        assert all(0.05 <= p <= 0.12 for p in progresses)

    def test_messages_carry_counts_and_percent(self, captured):
        cb = pipeline_mod._make_ingest_progress_emitter("job-1", "scan.laz")
        cb(5_000_000, 10_000_000)
        assert len(captured) == 1
        msg = captured[0][2]
        assert "5,000,000" in msg
        assert "10,000,000" in msg
        assert "50%" in msg
        assert "scan.laz" in msg

    def test_throttled_to_five_percent_steps(self, captured):
        """689 chunks of a huge file must NOT produce 689 events."""
        cb = pipeline_mod._make_ingest_progress_emitter("job-1", "big.laz")
        total = 689
        for chunk in range(1, total + 1):
            cb(chunk, total)
        # 5% steps → at most ~21 events (1/0.05 + first + final).
        assert len(captured) <= 21
        # Final event always fires at 100% regardless of throttling.
        assert captured[-1][3] == pytest.approx(0.12)

    def test_monotonic_progress(self, captured):
        cb = pipeline_mod._make_ingest_progress_emitter("job-1", "scan.laz")
        for read in range(0, 101):
            cb(read, 100)
        progresses = [e[3] for e in captured]
        assert progresses == sorted(progresses)
        assert len(progresses) >= 2  # incremental, not just start/end


class TestPipelinePhaseEmitter:
    """Phase 4.5: assembly phases land in (0.12, 0.135) — past the chunk
    span, before the downsample emit — so the bar moves past 12% honestly."""

    @pytest.fixture
    def captured(self, monkeypatch):
        events: list[tuple[str, str, str, float]] = []
        monkeypatch.setattr(
            pipeline_mod, "publish",
            lambda job_id, stage, message, progress:
                events.append((job_id, stage, message, progress)),
        )
        return events

    def test_phases_map_into_assembly_window(self, captured):
        cb = pipeline_mod._make_ingest_phase_emitter("job-1", "scan.laz")
        for name in ("classify", "assemble", "colors", "unit_detect"):
            cb(name)

        assert len(captured) == 4
        progresses = [e[3] for e in captured]
        # Strictly after the chunk span, strictly before the 0.135 downsample
        # emit, and monotone in the order the loader fires them.
        assert all(
            pipeline_mod.INGEST_PROGRESS_HI < p < 0.135 for p in progresses
        )
        assert progresses == sorted(progresses)
        # "Loaded N points" lands after every phase, still before 0.135.
        assert progresses[-1] < pipeline_mod.INGEST_LOADED_PROGRESS < 0.135

    def test_unknown_phase_ignored(self, captured):
        cb = pipeline_mod._make_ingest_phase_emitter("job-1", "scan.laz")
        cb("some_future_phase")
        assert captured == []

    def test_messages_carry_scan_name(self, captured):
        cb = pipeline_mod._make_ingest_phase_emitter("job-1", "scan.laz")
        cb("classify")
        assert "scan.laz" in captured[0][2]
        assert captured[0][1] == "ingest"
