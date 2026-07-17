"""Size-capped LRU eviction for derived data (Phase 4.5).

The 3-tier storage policy: raw scans are durable (deduped, age-cleaned),
``results/`` deliverables are kept indefinitely, and everything derived —
``artifacts/<job_id>/`` plus the downsample cache — is a disposable cache
bounded by ``storage_cap_gb``.  When over the cap, least-recently-accessed
entries are evicted first, stopping as soon as the total is back under.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


MB = 1024 * 1024


@pytest.fixture()
def temp_storage(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from app.config import get_settings
    get_settings.cache_clear()
    from app import storage
    yield storage
    get_settings.cache_clear()


def _make_artifact_dir(storage, job_id: str, size_bytes: int, age_s: float) -> Path:
    """Job artifact dir with one file of the given size, last accessed
    ``age_s`` seconds ago."""
    d = storage.artifacts_dir(job_id)
    f = d / "overlay.png"
    f.write_bytes(b"\x00" * size_bytes)
    ts = time.time() - age_s
    os.utime(f, (ts, ts))
    return d


def _make_cache_entry(storage, key: str, size_bytes: int, age_s: float) -> list[Path]:
    cache_root = storage.get_settings().data_dir / "cache" / "downsample"
    cache_root.mkdir(parents=True, exist_ok=True)
    npz = cache_root / f"{key}.npz"
    meta = cache_root / f"{key}.json"
    npz.write_bytes(b"\x00" * size_bytes)
    meta.write_text("{}")
    ts = time.time() - age_s
    for p in (npz, meta):
        os.utime(p, (ts, ts))
    return [npz, meta]


class TestEnforceStorageCap:
    def test_under_cap_evicts_nothing(self, temp_storage):
        storage = temp_storage
        d = _make_artifact_dir(storage, "job-1", 1 * MB, age_s=1000)
        evicted = storage.enforce_storage_cap(cap_gb=1.0)
        assert evicted == 0
        assert d.exists()

    def test_evicts_least_recently_accessed_first_and_stops_at_cap(self, temp_storage):
        """Three 1 MB dirs, cap 2 MB → exactly the OLDEST-accessed one goes."""
        storage = temp_storage
        oldest = _make_artifact_dir(storage, "job-old", 1 * MB, age_s=30_000)
        middle = _make_artifact_dir(storage, "job-mid", 1 * MB, age_s=20_000)
        newest = _make_artifact_dir(storage, "job-new", 1 * MB, age_s=10_000)

        evicted = storage.enforce_storage_cap(cap_gb=2 * MB / 1024 ** 3)

        assert evicted == 1
        assert not oldest.exists()
        assert middle.exists()
        assert newest.exists()

    def test_eviction_order_is_access_time_not_name(self, temp_storage):
        storage = temp_storage
        # "job-a" sorts first by name but is the most recently accessed.
        recent = _make_artifact_dir(storage, "job-a", 1 * MB, age_s=100)
        stale = _make_artifact_dir(storage, "job-z", 1 * MB, age_s=90_000)

        evicted = storage.enforce_storage_cap(cap_gb=1.5 * MB / 1024 ** 3)

        assert evicted == 1
        assert recent.exists()
        assert not stale.exists()

    def test_results_never_evicted(self, temp_storage):
        """Deliverables are sacrosanct — they neither count toward the cap
        nor get evicted, no matter how far over the cap we are."""
        storage = temp_storage
        result_file = storage.results_dir("job-1") / "vectorized.dxf"
        result_file.write_bytes(b"\x00" * (10 * MB))
        old = time.time() - 90_000
        os.utime(result_file, (old, old))

        evicted = storage.enforce_storage_cap(cap_gb=1 * MB / 1024 ** 3)

        assert evicted == 0
        assert result_file.exists()

    def test_uploads_never_evicted(self, temp_storage):
        storage = temp_storage
        scan = storage.uploads_dir("job-1") / "scan.las"
        scan.write_bytes(b"\x00" * (10 * MB))
        old = time.time() - 90_000
        os.utime(scan, (old, old))

        assert storage.enforce_storage_cap(cap_gb=1 * MB / 1024 ** 3) == 0
        assert scan.exists()

    def test_downsample_cache_participates(self, temp_storage):
        """Cache pairs (npz + json) are LRU entries too, evicted as a unit."""
        storage = temp_storage
        stale_pair = _make_cache_entry(storage, "hash-a_vox0.005000_auto0", 2 * MB, age_s=90_000)
        fresh_dir = _make_artifact_dir(storage, "job-new", 1 * MB, age_s=100)

        evicted = storage.enforce_storage_cap(cap_gb=1.5 * MB / 1024 ** 3)

        assert evicted == 1
        assert all(not p.exists() for p in stale_pair)
        assert fresh_dir.exists()

    def test_cache_load_refreshes_lru_position(self, temp_storage):
        """A cache HIT must move the entry to the back of the eviction
        queue — that's what makes the policy least-recently-USED."""
        import numpy as np
        from app.vectorize import downsample_cache

        storage = temp_storage
        pcd_points = np.random.default_rng(1).uniform(0, 5, (2000, 3))
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        downsample_cache.store_cache(
            "hash-hot", 0.005, False, pcd,
            effective_voxel_m=0.005, raw_points=1, n_before=1,
        )
        npz_path, meta_path = downsample_cache.cache_paths("hash-hot", 0.005, False)
        old = time.time() - 90_000
        for p in (npz_path, meta_path):
            os.utime(p, (old, old))

        # Competing entry, accessed less recently than "now" but more
        # recently than the hot entry's ORIGINAL (stale) stamp.
        competing = _make_artifact_dir(storage, "job-mid", 1 * MB, age_s=50_000)

        # Touch via a real cache load, then evict down to ~the competing size.
        assert downsample_cache.load_cache("hash-hot", 0.005, False) is not None
        cache_size = npz_path.stat().st_size + meta_path.stat().st_size
        cap_gb = (cache_size + 0.5 * MB) / 1024 ** 3
        evicted = storage.enforce_storage_cap(cap_gb=cap_gb)

        assert evicted >= 1
        assert npz_path.exists(), "freshly-hit cache entry must survive"
        assert not competing.exists()

    def test_cap_zero_disables(self, temp_storage):
        storage = temp_storage
        d = _make_artifact_dir(storage, "job-1", 5 * MB, age_s=90_000)
        assert storage.enforce_storage_cap(cap_gb=0) == 0
        assert d.exists()

    def test_default_cap_from_settings(self, temp_storage, monkeypatch):
        storage = temp_storage
        d = _make_artifact_dir(storage, "job-1", 2 * MB, age_s=90_000)
        monkeypatch.setenv("STORAGE_CAP_GB", str(1 * MB / 1024 ** 3))
        storage.get_settings.cache_clear()

        evicted = storage.enforce_storage_cap()  # cap read from settings

        assert evicted == 1
        assert not d.exists()


class TestMaintenanceWiring:
    def test_maintenance_pass_runs_all_three_steps(self, temp_storage, monkeypatch):
        """The cleanup loop entry point must chain age cleanup → blob GC →
        cap eviction (the sweep order matters: a deleted job dir may free a
        blob, and eviction should see the post-cleanup state)."""
        from app import main as main_mod

        calls: list[str] = []
        monkeypatch.setattr(
            main_mod, "cleanup_old_jobs", lambda *a, **k: calls.append("age") or 0,
        )
        monkeypatch.setattr(
            main_mod, "cleanup_orphaned_blobs", lambda: calls.append("blobs") or 0,
        )
        monkeypatch.setattr(
            main_mod, "enforce_storage_cap", lambda: calls.append("cap") or 0,
        )

        main_mod._run_storage_maintenance(30, True)

        assert calls == ["age", "blobs", "cap"]
