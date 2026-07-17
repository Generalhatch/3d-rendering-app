"""Orphaned-job recovery at server startup.

Pipeline runs execute in worker threads inside the server process; a restart
(deploy, crash, uvicorn --reload picking up a file edit) kills them without a
terminal status write.  ``fail_orphaned_jobs()`` runs at startup and must
transition every 'queued'/'processing' job — in BOTH job tables — to 'failed'
with an explanatory error message, while leaving terminal jobs untouched.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest


@pytest.fixture()
def temp_db(tmp_path: Path, monkeypatch):
    """Point storage at a temp SQLite DB (same pattern as test_storage_concurrent)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test_jobs.sqlite"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))

    import importlib
    from app import config, storage
    importlib.reload(config)
    importlib.reload(storage)

    storage.init_db()
    return storage


def _make_vectorize_job(storage, status: str) -> str:
    job_id = str(uuid.uuid4())
    storage.create_vectorize_job(job_id, "scan.laz", {})
    if status != "queued":
        storage.update_vectorize_status(job_id, status)
    return job_id


def _make_alignment_job(storage, status: str) -> str:
    job_id = str(uuid.uuid4())
    storage.create_job(job_id, ["scan_00.laz"], "")
    if status != "queued":
        storage.update_job_status(job_id, status)
    return job_id


class TestFailOrphanedVectorizeJobs:
    def test_processing_job_marked_failed(self, temp_db):
        storage = temp_db
        job_id = _make_vectorize_job(storage, "processing")

        n = storage.fail_orphaned_jobs()

        assert n == 1
        record = storage.load_vectorize_job(job_id)
        assert record["status"] == "failed"
        assert "restart" in record["error_message"].lower()

    def test_queued_job_marked_failed(self, temp_db):
        storage = temp_db
        job_id = _make_vectorize_job(storage, "queued")

        storage.fail_orphaned_jobs()

        assert storage.load_vectorize_job(job_id)["status"] == "failed"

    def test_terminal_jobs_untouched(self, temp_db):
        storage = temp_db
        complete_id = _make_vectorize_job(storage, "queued")
        storage.update_vectorize_result(complete_id, {"metrics": {}})
        failed_id = _make_vectorize_job(storage, "queued")
        storage.update_vectorize_status(failed_id, "failed", error="boom")

        n = storage.fail_orphaned_jobs()

        assert n == 0
        assert storage.load_vectorize_job(complete_id)["status"] == "complete"
        failed = storage.load_vectorize_job(failed_id)
        assert failed["status"] == "failed"
        assert failed["error_message"] == "boom"  # original error preserved

    def test_completed_result_payload_preserved(self, temp_db):
        """The sweep must not clear result_json of completed jobs."""
        storage = temp_db
        job_id = _make_vectorize_job(storage, "queued")
        storage.update_vectorize_result(job_id, {"metrics": {"elapsed_s": 1.0}})

        storage.fail_orphaned_jobs()

        assert storage.load_vectorize_result(job_id) == {
            "metrics": {"elapsed_s": 1.0}
        }


class TestFailOrphanedAlignmentJobs:
    def test_alignment_processing_marked_failed(self, temp_db):
        storage = temp_db
        job_id = _make_alignment_job(storage, "processing")

        n = storage.fail_orphaned_jobs()

        assert n == 1
        detail = storage.load_job(job_id)
        assert detail.status == "failed"
        assert "restart" in (detail.error_message or "").lower()

    def test_aligned_job_untouched(self, temp_db):
        storage = temp_db
        job_id = _make_alignment_job(storage, "queued")
        storage.update_job_result(job_id, {"rooms": [], "alignment": {}}, 1.0)

        storage.fail_orphaned_jobs()

        assert storage.load_job(job_id).status == "aligned"

    def test_both_tables_swept_in_one_call(self, temp_db):
        storage = temp_db
        v_id = _make_vectorize_job(storage, "processing")
        a_id = _make_alignment_job(storage, "processing")

        n = storage.fail_orphaned_jobs()

        assert n == 2
        assert storage.load_vectorize_job(v_id)["status"] == "failed"
        assert storage.load_job(a_id).status == "failed"
