"""Filesystem helpers and SQLite job persistence."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO, Optional

from .config import get_settings
from .models.job import FloorCandidate, JobDetail, JobStatus

logger = logging.getLogger(__name__)


def _conn() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id      TEXT PRIMARY KEY,
                status      TEXT NOT NULL DEFAULT 'queued',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                scan_filename  TEXT NOT NULL DEFAULT '',
                scan_filenames TEXT NOT NULL DEFAULT '[]',
                plan_filename  TEXT NOT NULL DEFAULT '',
                error_message  TEXT,
                result_json    TEXT,
                elapsed_s      REAL
            )
        """)
        # Vectorize jobs live in a separate table from alignment jobs — they
        # have different inputs, different statuses, and different result
        # payloads.  Sharing one table would be a leaky abstraction.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectorize_jobs (
                job_id        TEXT PRIMARY KEY,
                status        TEXT NOT NULL DEFAULT 'queued',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                scan_filename TEXT NOT NULL DEFAULT '',
                params_json   TEXT NOT NULL DEFAULT '{}',
                result_json   TEXT,
                error_message TEXT
            )
        """)
        conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(job_id: str, scan_filenames: list[str], plan_filename: str) -> JobDetail:
    now = _now()
    primary = scan_filenames[0] if scan_filenames else ""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, created_at, updated_at, scan_filename, scan_filenames, plan_filename) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, JobStatus.queued, now, now, primary, json.dumps(scan_filenames), plan_filename),
        )
        conn.commit()
    return load_job(job_id)


def update_job_status(job_id: str, status: JobStatus, error: Optional[str] = None) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ?, error_message = ? WHERE job_id = ?",
            (status, _now(), error, job_id),
        )
        conn.commit()


def update_job_result(job_id: str, result_json: dict, elapsed_s: float) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ?, result_json = ?, elapsed_s = ? WHERE job_id = ?",
            (JobStatus.aligned, _now(), json.dumps(result_json), elapsed_s, job_id),
        )
        conn.commit()


def load_job(job_id: str) -> JobDetail:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(f"Job {job_id} not found")
    d = dict(row)
    result = None
    num_rooms = None
    num_fixtures = None
    scan_only = False
    plan_bounds = None
    floor_z = 0.0
    floor_candidates: list[FloorCandidate] = []
    if d.get("result_json"):
        parsed = json.loads(d["result_json"])
        alignment = parsed.get("alignment", {})
        result = alignment if alignment else None
        num_rooms = len(parsed.get("rooms", []))
        num_fixtures = len(parsed.get("fixtures", []))
        scan_only = bool(parsed.get("scan_only", False))
        plan_bounds = parsed.get("plan_bounds")
        floor_z = float(parsed.get("floor_z", 0.0))
        floor_candidates = [
            FloorCandidate(**fc)
            for fc in parsed.get("floor_candidates", [])
        ]
    return JobDetail(
        job_id=d["job_id"],
        status=d["status"],
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        scan_filename=d["scan_filename"],
        scan_filenames=json.loads(d.get("scan_filenames") or "[]"),
        plan_filename=d["plan_filename"],
        error_message=d.get("error_message"),
        result=result,
        scan_only=scan_only,
        num_rooms=num_rooms,
        num_fixtures=num_fixtures,
        elapsed_s=d.get("elapsed_s"),
        plan_bounds=plan_bounds,
        floor_z=floor_z,
        floor_candidates=floor_candidates,
    )


def load_result_json(job_id: str) -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT result_json FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    if row is None or row["result_json"] is None:
        raise KeyError(f"No result for job {job_id}")
    return json.loads(row["result_json"])


# ── File path helpers ─────────────────────────────────────────────────────────

def uploads_dir(job_id: str) -> Path:
    p = get_settings().uploads_dir / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Content-hash deduped uploads (Phase 4.5) ─────────────────────────────────
# Raw scans are the durable tier of the storage policy: one 640 MB scan
# uploaded to four different jobs must be stored ONCE.  The bytes live under
# uploads/by-hash/<sha256><ext>; each job's upload dir holds a relative
# symlink to the shared blob, so every existing consumer (laspy, Open3D,
# FileResponse) keeps working through the job path unchanged.

_BY_HASH_DIRNAME = "by-hash"
_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB read chunks for streaming hash/copy


def by_hash_dir() -> Path:
    p = get_settings().uploads_dir / _BY_HASH_DIRNAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def store_upload_deduped(
    fileobj: BinaryIO, job_id: str, safe_name: str,
) -> tuple[Path, str]:
    """Stream an upload to the content-addressed store and link it into the
    job's upload dir.

    The stream is sha256-hashed WHILE being written (single pass, no second
    read of a multi-GB upload).  If a blob with the same digest already
    exists, the new copy is discarded — dedupe.  Returns
    ``(job_path, sha256_hex)`` where ``job_path`` is the symlink at
    ``uploads/<job_id>/<safe_name>``.
    """
    blob_dir = by_hash_dir()
    suffix = Path(safe_name).suffix.lower()

    hasher = hashlib.sha256()
    tmp_path = blob_dir / f".tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "wb") as out:
            while True:
                chunk = fileobj.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
        digest = hasher.hexdigest()
        blob_path = blob_dir / f"{digest}{suffix}"
        if blob_path.exists():
            tmp_path.unlink()
        else:
            tmp_path.rename(blob_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    link_path = uploads_dir(job_id) / safe_name
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    # Relative target so the whole data dir can be moved without breaking links.
    link_path.symlink_to(Path("..") / _BY_HASH_DIRNAME / blob_path.name)
    return link_path, digest


def scan_content_hash(path: Path) -> str:
    """sha256 of a stored scan's content.

    Deduped uploads resolve to ``by-hash/<sha256><ext>`` — the digest is
    read straight off the filename.  Legacy files (pre-dedupe uploads,
    merged PLYs) are hashed by streaming; still far cheaper than a full
    raw re-ingest.
    """
    resolved = path.resolve()
    stem = resolved.stem
    if resolved.parent.name == _BY_HASH_DIRNAME and len(stem) == 64:
        try:
            int(stem, 16)
            return stem
        except ValueError:
            pass
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def cleanup_orphaned_blobs() -> int:
    """Delete by-hash blobs no job upload dir references any more.

    Job upload dirs hold symlinks into ``uploads/by-hash/``; deleting one
    job's dir must never break another job sharing the same blob, so blob
    removal is REFERENCE-scan based: a blob is deleted only when no symlink
    under any remaining job dir resolves to it.  Stale ``.tmp-*`` files
    (crashed mid-upload) are removed too.  Returns the number of blobs
    deleted.
    """
    settings = get_settings()
    blob_dir = settings.uploads_dir / _BY_HASH_DIRNAME
    if not blob_dir.is_dir():
        return 0

    referenced: set[Path] = set()
    for job_dir in settings.uploads_dir.iterdir():
        if not job_dir.is_dir() or job_dir.name == _BY_HASH_DIRNAME:
            continue
        for f in job_dir.iterdir():
            if f.is_symlink():
                try:
                    referenced.add(f.resolve())
                except OSError:
                    pass

    removed = 0
    now_ts = datetime.now(timezone.utc).timestamp()
    for blob in blob_dir.iterdir():
        if not blob.is_file():
            continue
        if blob.name.startswith(".tmp-"):
            # Only reap tmp files old enough that no live upload can still
            # be writing them (uploads finish in minutes, not hours).
            try:
                if now_ts - blob.stat().st_mtime > 3600:
                    blob.unlink()
                    removed += 1
            except OSError:
                pass
            continue
        if blob.resolve() not in referenced:
            try:
                blob.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("blob cleanup: failed to remove %s: %s", blob, exc)

    if removed:
        logger.info("blob cleanup: removed %d unreferenced blob(s)", removed)
    return removed


def artifacts_dir(job_id: str) -> Path:
    p = get_settings().artifacts_dir / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(job_id: str) -> Path:
    p = get_settings().results_dir / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Vectorize-job CRUD ────────────────────────────────────────────────────────
# Mirrors the alignment-job helpers above but targets the vectorize_jobs table.
# Kept side-by-side so the two surfaces never accidentally share state.

def create_vectorize_job(job_id: str, scan_filename: str, params: dict) -> None:
    now = _now()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO vectorize_jobs "
            "(job_id, status, created_at, updated_at, scan_filename, params_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, "queued", now, now, scan_filename, json.dumps(params)),
        )
        conn.commit()


def update_vectorize_status(
    job_id: str, status: str, error: Optional[str] = None,
) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE vectorize_jobs "
            "SET status = ?, updated_at = ?, error_message = ? WHERE job_id = ?",
            (status, _now(), error, job_id),
        )
        conn.commit()


def update_vectorize_result(job_id: str, result: dict) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE vectorize_jobs "
            "SET status = ?, updated_at = ?, result_json = ? WHERE job_id = ?",
            ("complete", _now(), json.dumps(result), job_id),
        )
        conn.commit()


def update_vectorize_params(job_id: str, params: dict) -> None:
    """Replace the stored params (used by reprocess)."""
    with _conn() as conn:
        conn.execute(
            "UPDATE vectorize_jobs "
            "SET params_json = ?, updated_at = ?, status = ?, "
            "    result_json = NULL, error_message = NULL "
            "WHERE job_id = ?",
            (json.dumps(params), _now(), "queued", job_id),
        )
        conn.commit()


def load_vectorize_job(job_id: str) -> dict:
    """Return a flat dict for the API layer to serialise."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM vectorize_jobs WHERE job_id = ?", (job_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Vectorize job {job_id} not found")

    d = dict(row)
    return {
        "job_id":         d["job_id"],
        "status":         d["status"],
        "created_at":     d["created_at"],
        "updated_at":     d["updated_at"],
        "scan_filename":  d["scan_filename"],
        "params":         json.loads(d.get("params_json") or "{}"),
        "result":         json.loads(d["result_json"]) if d.get("result_json") else None,
        "error_message":  d.get("error_message"),
    }


def load_vectorize_result(job_id: str) -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT result_json FROM vectorize_jobs WHERE job_id = ?", (job_id,),
        ).fetchone()
    if row is None or row["result_json"] is None:
        raise KeyError(f"No result for vectorize job {job_id}")
    return json.loads(row["result_json"])


def list_vectorize_jobs(limit: int = 50) -> list[dict]:
    """Recent vectorize jobs, newest first.  Used by the history panel."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT job_id, status, created_at, updated_at, scan_filename, error_message "
            "FROM vectorize_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Startup recovery ──────────────────────────────────────────────────────────

_ORPHAN_ERROR_MESSAGE = (
    "Interrupted by a server restart before completing — the in-flight "
    "pipeline run was lost (uvicorn --reload restarts kill running jobs). "
    "Re-run the job to process it again."
)


def fail_orphaned_jobs() -> int:
    """Mark jobs stuck in 'queued'/'processing' as failed.

    Pipeline runs execute in in-process worker threads; a server restart
    (deploy, crash, or uvicorn --reload picking up a file edit) kills them
    with no chance to write a terminal status.  Without this sweep those
    jobs show "processing" forever in the UI.  Called once at startup,
    BEFORE any new work is accepted, so every non-terminal job in the DB is
    guaranteed to be an orphan from a previous process.

    Returns the number of jobs transitioned to failed (both tables).
    """
    now = _now()
    with _conn() as conn:
        cur_v = conn.execute(
            "UPDATE vectorize_jobs SET status = 'failed', updated_at = ?, "
            "error_message = ? WHERE status IN ('queued', 'processing')",
            (now, _ORPHAN_ERROR_MESSAGE),
        )
        cur_a = conn.execute(
            "UPDATE jobs SET status = 'failed', updated_at = ?, "
            "error_message = ? WHERE status IN ('queued', 'processing')",
            (now, _ORPHAN_ERROR_MESSAGE),
        )
        conn.commit()
    n = int(cur_v.rowcount) + int(cur_a.rowcount)
    if n:
        logger.info("startup: marked %d orphaned job(s) as failed", n)
    return n


# ── Storage cleanup ───────────────────────────────────────────────────────────

def _entry_size_and_access(files: list[Path]) -> tuple[int, float]:
    """Total byte size and most-recent access (max of atime/mtime — macOS
    and many Linux mounts don't update atime reliably, so mtime is the
    floor) over a list of files."""
    size = 0
    access = 0.0
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        size += st.st_size
        access = max(access, st.st_mtime, st.st_atime)
    return size, access


def enforce_storage_cap(cap_gb: Optional[float] = None) -> int:
    """Size-capped LRU eviction over DISPOSABLE derived data (Phase 4.5).

    Covered tiers: per-job ``artifacts/<job_id>/`` dirs and downsample-cache
    entries (``data/cache/downsample/`` npz+json pairs).  NEVER touched:
    ``results/`` deliverables (result.json / DXF / SVG / PDF — KBs each,
    kept indefinitely) and ``uploads/`` raw scans (durable tier, handled by
    the age-based cleanup + blob GC).

    When the covered total exceeds the cap, least-recently-accessed entries
    are evicted first until the total is back under the cap.  ``cap_gb``
    defaults to ``settings.storage_cap_gb``; a cap <= 0 disables eviction.
    Returns the number of entries (job artifact dirs + cache pairs) evicted.
    """
    settings = get_settings()
    if cap_gb is None:
        cap_gb = settings.storage_cap_gb
    if cap_gb is None or cap_gb <= 0:
        return 0
    cap_bytes = int(cap_gb * 1024 ** 3)

    # (last_access, size_bytes, [paths to delete], label)
    entries: list[tuple[float, int, list[Path], str]] = []

    artifacts_root = settings.artifacts_dir
    if artifacts_root.is_dir():
        for job_dir in artifacts_root.iterdir():
            if not job_dir.is_dir():
                continue
            files = [f for f in job_dir.rglob("*") if f.is_file()]
            size, access = _entry_size_and_access(files)
            if size > 0:
                entries.append((access, size, [job_dir], f"artifacts/{job_dir.name}"))

    cache_root = settings.data_dir / "cache" / "downsample"
    if cache_root.is_dir():
        # An entry is the npz + its json sidecar — evicted as a unit.
        groups: dict[str, list[Path]] = {}
        for f in cache_root.iterdir():
            if f.is_file():
                groups.setdefault(f.name.rsplit(".", 1)[0], []).append(f)
        for key, files in groups.items():
            size, access = _entry_size_and_access(files)
            if size > 0:
                entries.append((access, size, files, f"cache/{key}"))

    total = sum(size for _, size, _, _ in entries)
    if total <= cap_bytes:
        return 0

    entries.sort(key=lambda e: e[0])  # least-recently-accessed first
    evicted = 0
    for _access, size, paths, label in entries:
        if total <= cap_bytes:
            break
        try:
            for p in paths:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
        except OSError as exc:
            logger.warning("storage cap: failed to evict %s: %s", label, exc)
            continue
        total -= size
        evicted += 1
        logger.info("storage cap: evicted %s (%.1f MB)", label, size / 1e6)

    if evicted:
        logger.info(
            "storage cap: evicted %d entrie(s); %.2f GB remain (cap %.2f GB)",
            evicted, total / 1024 ** 3, cap_gb,
        )
    return evicted


def cleanup_old_jobs(max_age_days: int, uploads_only: bool = True) -> int:
    """Delete stored files for jobs older than *max_age_days*.

    When *uploads_only* is True (the default) only the raw scan / DXF uploads
    are removed — the small processed artefacts (decimated PLY, overlay PNG,
    aligned.json, aligned.dxf) are kept so the viewer can still load past jobs.
    When False, the artefacts and results directories are also wiped.

    The SQLite record is never deleted so job history remains intact.

    Returns the number of jobs that had files removed.
    """
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cutoff_iso = cutoff.isoformat()

    with _conn() as conn:
        rows = conn.execute(
            "SELECT job_id, created_at FROM jobs WHERE created_at < ?",
            (cutoff_iso,),
        ).fetchall()

    cleaned = 0
    for row in rows:
        job_id = row["job_id"]
        dirs_to_remove: list[Path] = [settings.uploads_dir / job_id]
        if not uploads_only:
            dirs_to_remove += [
                settings.artifacts_dir / job_id,
                settings.results_dir / job_id,
            ]

        removed_any = False
        for d in dirs_to_remove:
            if d.exists():
                try:
                    shutil.rmtree(d)
                    removed_any = True
                except Exception as exc:
                    logger.warning("cleanup: failed to remove %s: %s", d, exc)

        if removed_any:
            cleaned += 1
            logger.info(
                "cleanup: removed %s for job %s (created %s)",
                "uploads" if uploads_only else "uploads+artifacts",
                job_id,
                row["created_at"],
            )

    if rows:
        logger.info(
            "cleanup: scanned %d old jobs (>%d days), cleaned %d",
            len(rows), max_age_days, cleaned,
        )
    return cleaned
