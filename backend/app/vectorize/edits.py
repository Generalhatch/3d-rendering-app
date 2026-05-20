"""Persistence + DXF re-emission for operator edits (Phase 3).

The pipeline initially writes:
  - ``vectorized.dxf``         the raw classical detection output
  - ``segments.json``          machine-readable form of the above, with IDs

When an operator finishes reviewing in the browser, the frontend POSTs the
final segment state.  We:
  1. Snapshot the current ``segments.json`` to ``segments_v{N-1}.json`` (audit
     trail — every save is recoverable)
  2. Overwrite ``segments.json`` with the new state
  3. Append the operator's edit log to ``edits.log.jsonl`` (append-only, one
     line per save, contains *all* discrete actions from the browser session)
  4. Re-emit ``vectorized.dxf`` from the edited segments
  5. Also keep ``vectorized_v{N-1}.dxf`` snapshots for rollback

The edit log eventually feeds Phase 4 (ML training signal: "model produced
this, operator corrected to that").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..models.vectorize_job import EditableSegment, EditEvent
from . import dxf_writer


@dataclass
class SaveResult:
    """Returned to the route handler — what the API echoes back to the client."""
    edit_version: int
    segments_saved: int
    dxf_path: Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def segments_json_path(result_dir: Path) -> Path:
    return result_dir / "segments.json"


def edits_log_path(result_dir: Path) -> Path:
    return result_dir / "edits.log.jsonl"


def dxf_path(result_dir: Path) -> Path:
    return result_dir / "vectorized.dxf"


def _next_version(result_dir: Path) -> int:
    """Find the next monotonic edit version by scanning existing snapshots."""
    versions = []
    for f in result_dir.glob("segments_v*.json"):
        try:
            stem = f.stem.removeprefix("segments_v")
            versions.append(int(stem))
        except ValueError:
            continue
    return (max(versions) + 1) if versions else 1


# ── Read ──────────────────────────────────────────────────────────────────────

def load_segments(result_dir: Path) -> dict[str, Any]:
    """Read the current ``segments.json``.  Raises FileNotFoundError if absent."""
    path = segments_json_path(result_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No segments.json at {path}.  The job may not have completed "
            f"successfully or was created by an older pipeline version."
        )
    return json.loads(path.read_text())


def load_edit_history(result_dir: Path) -> list[dict[str, Any]]:
    """Return all previously-saved edit events as a flat list (newest last)."""
    path = edits_log_path(result_dir)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


# ── Write ─────────────────────────────────────────────────────────────────────

def save_edits(
    result_dir: Path,
    job_id: str,
    segments: list[EditableSegment],
    log: list[EditEvent],
) -> SaveResult:
    """Persist a save event from the editor.

    Operations
    ----------
    - Snapshots the current ``segments.json`` and ``vectorized.dxf`` under
      versioned names so previous saves remain accessible.
    - Writes the new state to ``segments.json`` (canonical location).
    - Appends the log to ``edits.log.jsonl`` with a save-event header so the
      log can be replayed event-by-event later.
    - Re-emits ``vectorized.dxf`` from the new segments.

    Raises
    ------
    ValueError
        If two segments share the same id (would corrupt edit replay).
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    ids = [s.id for s in segments]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate segment IDs in save payload — refusing to persist.")

    version = _next_version(result_dir)

    # 1. snapshot current segments.json (if present)
    current_segments = segments_json_path(result_dir)
    if current_segments.exists():
        snapshot = result_dir / f"segments_v{version - 1 if version > 1 else 0}.json"
        snapshot.write_text(current_segments.read_text())

    # 2. snapshot current dxf
    current_dxf = dxf_path(result_dir)
    if current_dxf.exists():
        dxf_snap = result_dir / f"vectorized_v{version - 1 if version > 1 else 0}.dxf"
        dxf_snap.write_bytes(current_dxf.read_bytes())

    # 3. write new segments.json
    payload = {
        "version": 1,
        "units": "metres",
        "edit_version": version,
        "segments": [s.model_dump() for s in segments],
    }
    current_segments.write_text(json.dumps(payload, indent=2))

    # 4. append edit log (JSONL — one save-record per line, contains the inner events)
    save_record = {
        "edit_version": version,
        "job_id": job_id,
        "segment_count": len(segments),
        "events": [e.model_dump() for e in log],
    }
    with edits_log_path(result_dir).open("a") as f:
        f.write(json.dumps(save_record) + "\n")

    # 5. re-emit DXF
    if segments:
        arr = np.array(
            [
                [[s.x1, s.y1], [s.x2, s.y2]]
                for s in segments
                if s.layer == "walls"
            ],
            dtype=np.float64,
        )
    else:
        arr = np.zeros((0, 2, 2), dtype=np.float64)

    annotation = (
        f"Stevenson Vectorize | job={job_id} | edit_version={version} | "
        f"walls={len(arr)} (operator-edited)"
    )
    dxf_writer.write_walls_dxf(arr, current_dxf, annotation_text=annotation)

    return SaveResult(
        edit_version=version,
        segments_saved=len(segments),
        dxf_path=current_dxf,
    )
