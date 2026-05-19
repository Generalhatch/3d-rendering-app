"""Filesystem helpers and SQLite job persistence."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

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


def artifacts_dir(job_id: str) -> Path:
    p = get_settings().artifacts_dir / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(job_id: str) -> Path:
    p = get_settings().results_dir / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Storage cleanup ───────────────────────────────────────────────────────────

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
