"""Downsample cache: reprocess skips the raw re-ingest (Phase 4.5).

Reprocessing a 680 MB scan re-paid minutes of decompression + the full
assembly memory spike just to rebuild the same ~3 M-point downsampled
cloud.  Stage 1b now persists the downsampled cloud (lossless float64,
zlib-compressed npz) keyed by ``(scan content hash, requested voxel,
auto flag)``; a rerun with the same key loads the cache and never calls
the raw loader.  Cache-hit results must be IDENTICAL to cache-miss
results — same metrics, same segment geometry.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from app.vectorize import downsample_cache


def _random_cloud(n: int = 500, seed: int = 5) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rng.uniform(0, 10, (n, 3)))
    return pcd


@pytest.fixture()
def temp_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.sqlite"))
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestCacheRoundTrip:
    def test_store_then_load_returns_exact_points(self, temp_cache):
        pcd = _random_cloud()
        original = np.asarray(pcd.points).copy()
        downsample_cache.store_cache(
            "hash-a", 0.005, False, pcd,
            effective_voxel_m=0.005, raw_points=9999, n_before=9999,
        )

        hit = downsample_cache.load_cache("hash-a", 0.005, False)
        assert hit is not None
        # LOSSLESS is the contract: float64 exact, not approximately equal.
        np.testing.assert_array_equal(hit.points, original)
        assert hit.meta["raw_points"] == 9999
        assert hit.meta["voxel_m"] == 0.005

    def test_different_voxel_misses(self, temp_cache):
        downsample_cache.store_cache(
            "hash-a", 0.005, False, _random_cloud(),
            effective_voxel_m=0.005, raw_points=1, n_before=1,
        )
        assert downsample_cache.load_cache("hash-a", 0.010, False) is None

    def test_different_hash_misses(self, temp_cache):
        downsample_cache.store_cache(
            "hash-a", 0.005, False, _random_cloud(),
            effective_voxel_m=0.005, raw_points=1, n_before=1,
        )
        assert downsample_cache.load_cache("hash-b", 0.005, False) is None

    def test_different_auto_flag_misses(self, temp_cache):
        """Auto mode can change the effective voxel — the flag is part of
        the key so auto and fixed runs never share an entry."""
        downsample_cache.store_cache(
            "hash-a", 0.005, False, _random_cloud(),
            effective_voxel_m=0.005, raw_points=1, n_before=1,
        )
        assert downsample_cache.load_cache("hash-a", 0.005, True) is None

    def test_corrupt_entry_is_a_miss_not_a_crash(self, temp_cache):
        downsample_cache.store_cache(
            "hash-a", 0.005, False, _random_cloud(),
            effective_voxel_m=0.005, raw_points=1, n_before=1,
        )
        npz_path, _ = downsample_cache.cache_paths("hash-a", 0.005, False)
        npz_path.write_bytes(b"not an npz")
        assert downsample_cache.load_cache("hash-a", 0.005, False) is None


    def test_large_cloud_skips_zlib(self, temp_cache, monkeypatch):
        """Compressing tens of millions of float64 points freezes the UI —
        large clouds must use uncompressed np.savez."""
        monkeypatch.setattr(downsample_cache, "_COMPRESS_MAX_POINTS", 100)
        pcd = _random_cloud(n=250, seed=9)
        downsample_cache.store_cache(
            "hash-big", 0.005, False, pcd,
            effective_voxel_m=0.005, raw_points=250, n_before=250,
        )
        hit = downsample_cache.load_cache("hash-big", 0.005, False)
        assert hit is not None
        np.testing.assert_array_equal(hit.points, np.asarray(pcd.points))
        npz_path, _ = downsample_cache.cache_paths("hash-big", 0.005, False)
        with np.load(npz_path) as data:
            assert "points" in data.files


# ── Pipeline integration ─────────────────────────────────────────────────────

def _write_room_ply(path: Path) -> None:
    """6×6 m room, 0.1 m walls (both faces), 2.5 m ceiling — deterministic
    (seeded noise), small enough for three pipeline runs."""
    step = 0.06

    def _wall(x0, y0, x1, y1, zs):
        length = float(np.hypot(x1 - x0, y1 - y0))
        s = step / 4.0
        ts = np.arange(0.0, length + s / 2, s) / max(length, 1e-9)
        lx = x0 + ts * (x1 - x0)
        ly = y0 + ts * (y1 - y0)
        return np.vstack([
            np.column_stack([lx, ly, np.full(len(lx), z)]) for z in zs
        ])

    xs = np.arange(0.0, 6.0 + step / 2, step)
    zs = np.arange(0.0, 2.5 + step / 2, step)
    gx, gy = np.meshgrid(xs, xs)
    parts = [
        np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]),
        np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 2.5)]),
    ]
    for y_out, y_in in ((0.0, 0.1), (6.0, 5.9)):
        parts.append(_wall(0.0, y_out, 6.0, y_out, zs))
        parts.append(_wall(0.0, y_in, 6.0, y_in, zs))
    for x_out, x_in in ((0.0, 0.1), (6.0, 5.9)):
        parts.append(_wall(x_out, 0.0, x_out, 6.0, zs))
        parts.append(_wall(x_in, 0.0, x_in, 6.0, zs))

    pts = np.vstack(parts)
    pts = pts + np.random.default_rng(23).normal(0.0, 0.012, pts.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(path), pcd)


def _segment_geometry(result_dir: Path) -> list[tuple]:
    payload = json.loads((result_dir / "segments.json").read_text())
    return sorted(
        (s["layer"], s["x1"], s["y1"], s["x2"], s["y2"])
        for s in payload["segments"]
    )


@pytest.fixture(scope="module")
def cache_pipeline_runs(tmp_path_factory):
    """Three full pipeline runs sharing one scan + cache tree:

    1. cold  — raw loader called, cache populated
    2. rerun — same voxel → cache HIT, raw loader NOT called
    3. other — different voxel → cache MISS, raw loader called again
    """
    from app.config import get_settings
    from app.models.vectorize_job import VectorizeParams
    from app.pipeline import ingest as ingest_mod
    from app.vectorize.pipeline import run_vectorize

    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("cache-e2e")
    mp.setenv("DATA_DIR", str(tmp / "data"))
    mp.setenv("UPLOADS_DIR", str(tmp / "uploads"))
    mp.setenv("ARTIFACTS_DIR", str(tmp / "artifacts"))
    mp.setenv("RESULTS_DIR", str(tmp / "results"))
    mp.setenv("DB_PATH", str(tmp / "jobs.sqlite"))
    get_settings.cache_clear()

    scan_path = tmp / "room.ply"
    _write_room_ply(scan_path)

    load_calls = {"n": 0}
    real_load = ingest_mod.load_point_cloud

    def counting_load(*args, **kwargs):
        load_calls["n"] += 1
        return real_load(*args, **kwargs)

    mp.setattr(ingest_mod, "load_point_cloud", counting_load)

    # Explicit elevation → no floor RANSAC / relevel → the pipeline is
    # fully deterministic, so hit-vs-miss identity can be asserted exactly.
    params = VectorizeParams(
        elevation_m=1.6,
        resolution_m_per_px=0.02,
        detect_openings=False,
        detect_columns=False,
    )

    def _run(job_id: str, run_params: VectorizeParams) -> tuple[dict, Path]:
        artifact_dir = tmp / "artifacts" / job_id
        result_dir = tmp / "results" / job_id
        artifact_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True)
        completed: list[dict] = []
        errors: list[str] = []
        run_vectorize(
            job_id, scan_path, artifact_dir, result_dir, run_params,
            on_complete=lambda p: completed.append(p),
            on_error=lambda e: errors.append(e),
        )
        assert errors == [], f"pipeline failed: {errors}"
        return completed[0], result_dir

    calls_trace: dict[str, int] = {}
    payload_cold, dir_cold = _run("cache-cold", params)
    calls_trace["after_cold"] = load_calls["n"]
    payload_hit, dir_hit = _run("cache-hit", params)
    calls_trace["after_hit"] = load_calls["n"]
    other_params = params.model_copy(update={"voxel_downsample_m": 0.01})
    payload_other, _ = _run("cache-other-voxel", other_params)
    calls_trace["after_other"] = load_calls["n"]

    yield {
        "cold": (payload_cold, dir_cold),
        "hit": (payload_hit, dir_hit),
        "other": payload_other,
        "calls": calls_trace,
        "tmp": tmp,
    }
    mp.undo()
    get_settings.cache_clear()


class TestDownsampleCachePipeline:
    def test_cold_run_calls_loader_and_populates_cache(self, cache_pipeline_runs):
        r = cache_pipeline_runs
        assert r["calls"]["after_cold"] == 1
        entries = list((r["tmp"] / "data" / "cache" / "downsample").glob("*.npz"))
        assert len(entries) >= 1

    def test_rerun_same_voxel_skips_raw_ingest(self, cache_pipeline_runs):
        """THE point of the cache: the raw loader is not called on rerun."""
        r = cache_pipeline_runs
        assert r["calls"]["after_hit"] == r["calls"]["after_cold"]

    def test_different_voxel_misses_cache(self, cache_pipeline_runs):
        r = cache_pipeline_runs
        assert r["calls"]["after_other"] == r["calls"]["after_hit"] + 1

    def test_hit_and_miss_metrics_identical(self, cache_pipeline_runs):
        r = cache_pipeline_runs
        m_cold = dict(r["cold"][0]["metrics"])
        m_hit = dict(r["hit"][0]["metrics"])
        m_cold.pop("elapsed_s")
        m_hit.pop("elapsed_s")
        assert m_hit == m_cold

    def test_hit_and_miss_segments_identical(self, cache_pipeline_runs):
        """Exact geometry equality — the cache must not move a single
        coordinate of any output segment."""
        r = cache_pipeline_runs
        assert _segment_geometry(r["hit"][1]) == _segment_geometry(r["cold"][1])
