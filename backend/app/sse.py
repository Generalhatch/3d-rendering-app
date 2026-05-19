"""Server-Sent Events progress streaming."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import AsyncGenerator

# In-process event queues keyed by job_id
_queues: dict[str, list[asyncio.Queue]] = defaultdict(list)


def publish(job_id: str, stage: str, message: str, progress: float) -> None:
    """Push a progress event to all listeners for this job."""
    payload = json.dumps({"stage": stage, "message": message, "progress": progress, "job_id": job_id})
    for q in _queues.get(job_id, []):
        q.put_nowait(payload)


async def progress_generator(job_id: str) -> AsyncGenerator[str, None]:
    """SSE generator — yields 'data: {...}\n\n' lines."""
    q: asyncio.Queue = asyncio.Queue()
    _queues[job_id].append(q)
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {payload}\n\n"
                data = json.loads(payload)
                # Stop after terminal stages
                if data.get("stage") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                # Send a keepalive comment so the connection doesn't time out
                yield ": keepalive\n\n"
    finally:
        _queues[job_id].remove(q)
