"""Server-Sent Events progress streaming.

Each job keeps a small in-memory ring buffer of its most recent events so that
late-arriving subscribers (e.g. a browser whose ``EventSource`` only finishes
its handshake *after* the pipeline has already emitted "Loading scan…") still
see early progress instead of staring at 0 % until the next emit.

The buffer is intentionally bounded; once a terminal event ("complete" /
"error") is published, future subscribers replay the whole buffer and then
disconnect, which is what powers the "rehydrate a finished job on refresh"
UX.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from typing import AsyncGenerator, Deque

# In-process event queues keyed by job_id
_queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

# Replay buffer: last N raw JSON payloads per job_id. 64 comfortably covers
# every stage emitted by both the alignment and vectorize pipelines.
_BUFFER_SIZE = 64
_history: dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=_BUFFER_SIZE))


def publish(job_id: str, stage: str, message: str, progress: float) -> None:
    """Push a progress event to all current and future listeners for this job."""
    payload = json.dumps({"stage": stage, "message": message, "progress": progress, "job_id": job_id})
    _history[job_id].append(payload)
    for q in _queues.get(job_id, []):
        q.put_nowait(payload)


def clear_history(job_id: str) -> None:
    """Drop the replay buffer for a job (call when re-processing the same id)."""
    _history.pop(job_id, None)


async def progress_generator(job_id: str) -> AsyncGenerator[str, None]:
    """SSE generator — replays buffered events, then streams live ones."""
    q: asyncio.Queue = asyncio.Queue()
    _queues[job_id].append(q)

    # Replay any events that fired before this subscriber arrived. If the job
    # already terminated, we replay everything and then disconnect cleanly.
    buffered = list(_history.get(job_id, ()))
    terminated = False
    try:
        for payload in buffered:
            yield f"data: {payload}\n\n"
            data = json.loads(payload)
            if data.get("stage") in ("complete", "error"):
                terminated = True
                break

        if terminated:
            return

        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {payload}\n\n"
                data = json.loads(payload)
                if data.get("stage") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                # Send a keepalive comment so the connection doesn't time out
                yield ": keepalive\n\n"
    finally:
        try:
            _queues[job_id].remove(q)
        except ValueError:
            pass
