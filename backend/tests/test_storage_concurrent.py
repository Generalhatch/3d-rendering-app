"""Unit tests for SQLite storage — concurrent write safety (section 7).

These tests verify that simultaneous calls to update_job_result() from
multiple threads don't corrupt data or raise "database is locked" errors.
WAL mode + PRAGMA busy_timeout=5000 (added in Phase 1) should make these
tests pass reliably.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path: Path, monkeypatch):
    """Override the database path and upload/artifact dirs to use a temp dir."""
    db_file = tmp_path / "test_jobs.sqlite"
    monkeypatch.setenv("DB_PATH", str(db_file))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))

    # Re-import settings inside the test environment so it picks up the new env vars
    import importlib
    from app import config, storage
    importlib.reload(config)
    importlib.reload(storage)

    storage.init_db()
    return storage


def _make_job(storage_mod) -> str:
    """Create a queued job and return its job_id."""
    job_id = str(uuid.uuid4())
    storage_mod.create_job(job_id, ["scan_00.laz"], "")
    return job_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConcurrentWrites:
    def test_two_threads_write_different_jobs(self, temp_db):
        """Two threads writing to two separate jobs should not interfere."""
        storage = temp_db
        job_a = _make_job(storage)
        job_b = _make_job(storage)

        errors: list[Exception] = []

        def write_job(job_id: str, payload: dict):
            try:
                storage.update_job_result(job_id, payload, 1.0)
            except Exception as exc:
                errors.append(exc)

        payload_a = {"rooms": [{"id": "room-a1"}], "alignment": {"confidence_pct": 80}}
        payload_b = {"rooms": [{"id": "room-b1"}], "alignment": {"confidence_pct": 90}}

        t1 = threading.Thread(target=write_job, args=(job_a, payload_a))
        t2 = threading.Thread(target=write_job, args=(job_b, payload_b))
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)

        assert not errors, f"Concurrent writes raised: {errors}"

        result_a = storage.load_result_json(job_a)
        result_b = storage.load_result_json(job_b)
        assert result_a["rooms"][0]["id"] == "room-a1"
        assert result_b["rooms"][0]["id"] == "room-b1"

    def test_many_threads_write_same_job(self, temp_db):
        """Multiple threads writing to the same job — last write wins, no corruption."""
        storage = temp_db
        job_id = _make_job(storage)

        errors: list[Exception] = []
        n_threads = 5

        def write(thread_idx: int):
            try:
                payload = {"thread": thread_idx, "rooms": [], "alignment": {}}
                storage.update_job_result(job_id, payload, float(thread_idx))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Concurrent same-job writes raised: {errors}"

        # The stored result should be valid JSON (not corrupted)
        result = storage.load_result_json(job_id)
        assert "thread" in result
        assert isinstance(result["thread"], int)
        assert 0 <= result["thread"] < n_threads

    def test_read_during_write(self, temp_db):
        """A read should not raise while another thread is writing."""
        storage = temp_db
        job_id = _make_job(storage)

        # Seed an initial result
        storage.update_job_result(job_id, {"rooms": [], "alignment": {}}, 0.0)

        read_errors: list[Exception] = []

        def reader():
            for _ in range(20):
                try:
                    storage.load_job(job_id)
                except Exception as exc:
                    read_errors.append(exc)
                time.sleep(0.005)

        def writer():
            for i in range(10):
                payload = {"rooms": [{"id": f"room-{i}"}], "alignment": {}}
                storage.update_job_result(job_id, payload, float(i))
                time.sleep(0.008)

        t_read = threading.Thread(target=reader)
        t_write = threading.Thread(target=writer)
        t_read.start(); t_write.start()
        t_read.join(timeout=15); t_write.join(timeout=15)

        assert not read_errors, f"Read-during-write raised: {read_errors}"


class TestMultiFloorStorage:
    """Verify that floor_candidates round-trip through storage correctly."""

    def test_floor_candidates_stored_and_loaded(self, temp_db):
        storage = temp_db
        job_id = _make_job(storage)

        payload: dict[str, Any] = {
            "rooms": [],
            "fixtures": [],
            "alignment": {"confidence_pct": 100, "mode": "scan_only"},
            "floor_z": 0.5,
            "floor_axis_idx": 2,
            "floor_candidates": [
                {"floor_z": 0.5,  "inlier_count": 5000, "wall_score": 12000, "axis_idx": 2},
                {"floor_z": 3.2,  "inlier_count": 3000, "wall_score": 8000,  "axis_idx": 2},
                {"floor_z": -1.1, "inlier_count": 1500, "wall_score": 2000,  "axis_idx": 2},
            ],
            "plan_bounds": [0, 0, 10, 10],
            "plan_segments": [],
            "building_outline": [],
            "overlay_png": "",
            "merged_ply": "",
            "num_scans": 1,
            "merge_strategy": "single",
            "scan_only": True,
        }
        storage.update_job_result(job_id, payload, 5.0)

        job_detail = storage.load_job(job_id)
        assert len(job_detail.floor_candidates) == 3

        # First candidate should be the best (highest wall_score)
        fc0 = job_detail.floor_candidates[0]
        assert fc0.floor_z == pytest.approx(0.5)
        assert fc0.wall_score == 12000
        assert fc0.axis_idx == 2

        fc1 = job_detail.floor_candidates[1]
        assert fc1.floor_z == pytest.approx(3.2)

    def test_empty_floor_candidates_ok(self, temp_db):
        storage = temp_db
        job_id = _make_job(storage)

        payload: dict[str, Any] = {
            "rooms": [], "fixtures": [], "alignment": {}, "floor_z": 0.0,
            "floor_candidates": [], "plan_bounds": [0, 0, 1, 1],
            "plan_segments": [], "building_outline": [], "overlay_png": "",
            "merged_ply": "", "num_scans": 1, "merge_strategy": "single",
            "scan_only": True,
        }
        storage.update_job_result(job_id, payload, 1.0)

        job_detail = storage.load_job(job_id)
        assert job_detail.floor_candidates == []
