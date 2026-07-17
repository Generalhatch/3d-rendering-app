"""Content-hash dedupe of uploaded scans (Phase 4.5).

Storage audit finding: data/uploads held four raw copies of the SAME
~640 MB scan (2.5 GB of 2.7 GB total app data).  Uploads are now streamed
through sha256 into a content-addressed store — ``uploads/by-hash/
<sha256><ext>`` — and each job's upload dir holds a relative symlink to the
shared blob.  Deleting one job must never break another job sharing the
same bytes: blob garbage collection is reference-scan based.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture()
def temp_storage(tmp_path: Path, monkeypatch):
    """Point settings at temp dirs WITHOUT module reloads.

    ``get_settings`` is lru-cached and every storage/route function calls it
    at call time, so clearing the cache after setting the env vars redirects
    the whole app — including the already-imported route modules — to the
    temp tree.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    from app.config import get_settings
    get_settings.cache_clear()

    from app import storage
    storage.init_db()
    yield storage
    get_settings.cache_clear()


CONTENT_A = b"LASF" + b"\x01" * 500
CONTENT_B = b"LASF" + b"\x02" * 500


class TestStoreUploadDeduped:
    def test_identical_uploads_stored_once(self, temp_storage):
        storage = temp_storage
        path_1, digest_1 = storage.store_upload_deduped(
            io.BytesIO(CONTENT_A), "job-1", "scan.las",
        )
        path_2, digest_2 = storage.store_upload_deduped(
            io.BytesIO(CONTENT_A), "job-2", "scan.las",
        )

        assert digest_1 == digest_2 == hashlib.sha256(CONTENT_A).hexdigest()
        blobs = [p for p in storage.by_hash_dir().iterdir() if p.is_file()]
        assert len(blobs) == 1
        assert blobs[0].name == f"{digest_1}.las"
        # Both job paths are symlinks resolving to the one blob, and both
        # read back the original bytes.
        for p in (path_1, path_2):
            assert p.is_symlink()
            assert p.resolve() == blobs[0].resolve()
            assert p.read_bytes() == CONTENT_A

    def test_different_content_stored_separately(self, temp_storage):
        storage = temp_storage
        _, digest_a = storage.store_upload_deduped(
            io.BytesIO(CONTENT_A), "job-1", "a.las",
        )
        _, digest_b = storage.store_upload_deduped(
            io.BytesIO(CONTENT_B), "job-2", "b.las",
        )
        assert digest_a != digest_b
        blobs = [p for p in storage.by_hash_dir().iterdir() if p.is_file()]
        assert len(blobs) == 2

    def test_no_tmp_files_left_behind(self, temp_storage):
        storage = temp_storage
        storage.store_upload_deduped(io.BytesIO(CONTENT_A), "job-1", "scan.las")
        storage.store_upload_deduped(io.BytesIO(CONTENT_A), "job-2", "scan.las")
        leftovers = [
            p for p in storage.by_hash_dir().iterdir()
            if p.name.startswith(".tmp-")
        ]
        assert leftovers == []

    def test_scan_content_hash_reads_digest_from_blob_name(self, temp_storage):
        storage = temp_storage
        path, digest = storage.store_upload_deduped(
            io.BytesIO(CONTENT_A), "job-1", "scan.las",
        )
        assert storage.scan_content_hash(path) == digest

    def test_scan_content_hash_streams_legacy_files(self, temp_storage, tmp_path):
        """Pre-dedupe uploads (plain files) still hash correctly."""
        storage = temp_storage
        legacy = storage.uploads_dir("legacy-job") / "old.las"
        legacy.write_bytes(CONTENT_B)
        assert storage.scan_content_hash(legacy) == (
            hashlib.sha256(CONTENT_B).hexdigest()
        )


class TestBlobGarbageCollection:
    def test_shared_blob_survives_one_job_cleanup(self, temp_storage):
        """Deleting one job's upload dir must NOT break the other job."""
        storage = temp_storage
        storage.store_upload_deduped(io.BytesIO(CONTENT_A), "job-1", "scan.las")
        path_2, _ = storage.store_upload_deduped(
            io.BytesIO(CONTENT_A), "job-2", "scan.las",
        )

        shutil.rmtree(storage.uploads_dir("job-1"))
        removed = storage.cleanup_orphaned_blobs()

        assert removed == 0
        assert path_2.exists()
        assert path_2.read_bytes() == CONTENT_A

    def test_unreferenced_blob_removed(self, temp_storage):
        storage = temp_storage
        storage.store_upload_deduped(io.BytesIO(CONTENT_A), "job-1", "scan.las")
        storage.store_upload_deduped(io.BytesIO(CONTENT_A), "job-2", "scan.las")

        shutil.rmtree(storage.uploads_dir("job-1"))
        shutil.rmtree(storage.uploads_dir("job-2"))
        removed = storage.cleanup_orphaned_blobs()

        assert removed == 1
        assert [p for p in storage.by_hash_dir().iterdir() if p.is_file()] == []

    def test_fresh_tmp_files_kept_stale_removed(self, temp_storage):
        """A .tmp file could be a live in-flight upload — only reap old ones."""
        storage = temp_storage
        blob_dir = storage.by_hash_dir()
        fresh = blob_dir / ".tmp-fresh"
        fresh.write_bytes(b"in-flight")
        stale = blob_dir / ".tmp-stale"
        stale.write_bytes(b"crashed upload")
        old = os.path.getmtime(stale) - 7200
        os.utime(stale, (old, old))

        storage.cleanup_orphaned_blobs()

        assert fresh.exists()
        assert not stale.exists()

    def test_noop_when_no_by_hash_dir(self, temp_storage):
        """A tree with no by-hash dir (pre-dedupe deploy) returns 0 cleanly."""
        by_hash = temp_storage.get_settings().uploads_dir / "by-hash"
        if by_hash.exists():
            shutil.rmtree(by_hash)
        assert temp_storage.cleanup_orphaned_blobs() == 0


class TestUploadRoutesDedupe:
    """The HTTP upload routes must go through the content-addressed store."""

    @pytest.fixture()
    def client(self, temp_storage, monkeypatch):
        from fastapi.testclient import TestClient
        from app import main as main_mod
        from app.routes import jobs as jobs_routes
        from app.routes import vectorize as vectorize_routes

        # Don't run the actual pipelines — uploads only.
        monkeypatch.setattr(
            vectorize_routes, "_enqueue_run", lambda *a, **k: None,
        )

        async def _noop(*args, **kwargs):
            return None
        monkeypatch.setattr(jobs_routes, "run_pipeline_guarded", _noop)

        # No context manager: lifespan (startup cleanup etc.) intentionally
        # skipped; the fixture already ran init_db against the temp tree.
        return TestClient(main_mod.app)

    def test_vectorize_upload_twice_stores_once(self, temp_storage, client):
        storage = temp_storage
        job_ids = []
        for _ in range(2):
            resp = client.post(
                "/api/vectorize",
                files={"scan": ("scan.las", io.BytesIO(CONTENT_A), "application/octet-stream")},
            )
            assert resp.status_code == 200, resp.text
            job_ids.append(resp.json()["job_id"])

        blobs = [p for p in storage.by_hash_dir().iterdir() if p.is_file()]
        assert len(blobs) == 1
        # Both jobs remain runnable through their own upload path.
        for job_id in job_ids:
            scan_path = storage.uploads_dir(job_id) / "scan.las"
            assert scan_path.exists()
            assert scan_path.read_bytes() == CONTENT_A

    def test_alignment_upload_twice_stores_once(self, temp_storage, client):
        storage = temp_storage
        for _ in range(2):
            resp = client.post(
                "/api/jobs",
                files={"scans": ("scan.las", io.BytesIO(CONTENT_A), "application/octet-stream")},
            )
            assert resp.status_code == 200, resp.text

        blobs = [p for p in storage.by_hash_dir().iterdir() if p.is_file()]
        assert len(blobs) == 1
