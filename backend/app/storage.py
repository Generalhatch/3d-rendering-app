"""Filesystem helpers and SQLite job persistence."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import get_settings
from .models.job import JobDetail, JobStatus


def _conn() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
    if d.get("result_json"):
        parsed = json.loads(d["result_json"])
        alignment = parsed.get("alignment", {})
        result = alignment if alignment else None
        num_rooms = len(parsed.get("rooms", []))
        num_fixtures = len(parsed.get("fixtures", []))
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
        num_rooms=num_rooms,
        num_fixtures=num_fixtures,
        elapsed_s=d.get("elapsed_s"),
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
