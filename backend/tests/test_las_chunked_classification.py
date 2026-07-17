"""The chunked LAS loader must apply a FILE-GLOBAL classification mask.

Guards against the P0 bug where ``_build_classification_mask`` was applied
per chunk: a scan whose class-6 ("Building") percentage varies across chunks
got a different filtering policy per chunk — e.g. "class 6 only" for the
first million points and "keep everything" for the next — and disagreed with
what the unchunked loader produces for the identical file.
"""
from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pytest

from app.pipeline.ingest import _load_las, _load_las_chunked


def _write_las(path: Path, classifications: np.ndarray) -> None:
    """Write a minimal LAS file with the given per-point classification."""
    n = len(classifications)
    rng = np.random.default_rng(0)
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = (0.001, 0.001, 0.001)
    las = laspy.LasData(header)
    las.x = rng.uniform(0, 10, n)
    las.y = rng.uniform(0, 10, n)
    las.z = rng.uniform(0, 3, n)
    las.classification = classifications
    las.write(str(path))


@pytest.fixture
def mixed_classification_las(tmp_path: Path) -> tuple[Path, int]:
    """A file whose chunks would make DIFFERENT per-chunk mask decisions.

    - First 1000 points: 50% class 6 / 50% class 1  (chunk-local: >=10%
      class 6 -> "keep class 6 only" -> 500 points kept)
    - Next 1000 points: all class 0                 (chunk-local: <2%
      class 6 -> "keep all non-noise" -> 1000 points kept)

    File-global: 25% class 6 -> "keep class 6 only" -> exactly 500 points.
    The old per-chunk code kept 1500.
    """
    cls = np.concatenate([
        np.repeat([6, 1], 500),
        np.zeros(1000, dtype=np.int64),
    ]).astype(np.uint8)
    path = tmp_path / "mixed.las"
    _write_las(path, cls)
    expected_kept = 500  # global class-6-only regime
    return path, expected_kept


class TestChunkedClassificationGlobal:
    def test_chunked_mask_is_file_global(self, mixed_classification_las):
        path, expected_kept = mixed_classification_las
        pcd = _load_las_chunked(path, chunk_points=1000)
        assert len(pcd.points) == expected_kept

    def test_chunked_matches_unchunked(self, mixed_classification_las):
        """Chunk size must never change WHICH points survive."""
        path, _ = mixed_classification_las
        unchunked = _load_las(path)
        for chunk_points in (250, 1000, 1999, 10_000):
            chunked = _load_las_chunked(path, chunk_points=chunk_points)
            assert len(chunked.points) == len(unchunked.points), (
                f"chunk_points={chunk_points} changed the surviving point count"
            )
            np.testing.assert_allclose(
                np.asarray(chunked.points),
                np.asarray(unchunked.points),
                atol=1e-9,
            )

    def test_interior_scan_keeps_all_non_noise(self, tmp_path):
        """All-unclassified interior scan: every non-noise point survives,
        regardless of chunking."""
        cls = np.zeros(2000, dtype=np.uint8)
        cls[100:110] = 7   # low-noise points must be dropped
        path = tmp_path / "interior.las"
        _write_las(path, cls)

        pcd = _load_las_chunked(path, chunk_points=500)
        assert len(pcd.points) == 1990
