"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .storage import (
    cleanup_old_jobs,
    cleanup_orphaned_blobs,
    enforce_storage_cap,
    fail_orphaned_jobs,
    init_db,
)
from .routes.jobs import router as jobs_router
from .routes.export import router as export_router
from .routes.vectorize import router as vectorize_router

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL_S = 24 * 3600  # run once every 24 hours


def _run_storage_maintenance(max_age_days: int, uploads_only: bool) -> int:
    """One full maintenance pass: age-based job cleanup, then dedupe-blob
    GC (a removed job dir may have been the last reference to a shared
    scan), then the size-capped LRU eviction over artifacts + the
    downsample cache."""
    cleaned = cleanup_old_jobs(max_age_days, uploads_only)
    try:
        cleanup_orphaned_blobs()
    except Exception as exc:
        logger.warning("Blob GC failed: %s", exc)
    try:
        enforce_storage_cap()
    except Exception as exc:
        logger.warning("Storage-cap eviction failed: %s", exc)
    return cleaned


async def _cleanup_loop(max_age_days: int, uploads_only: bool) -> None:
    """Background coroutine: run storage maintenance every 24 hours."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_S)
        try:
            cleaned = await asyncio.to_thread(
                _run_storage_maintenance, max_age_days, uploads_only,
            )
            logger.info("Periodic cleanup complete: %d job(s) cleaned", cleaned)
        except Exception as exc:
            logger.warning("Periodic cleanup failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    # Any job still 'queued'/'processing' at startup is an orphan — its
    # worker thread died with the previous process (uvicorn --reload kills
    # in-flight jobs on any file edit).  Fail them so they don't show as
    # stuck-at-N% forever.
    try:
        fail_orphaned_jobs()
    except Exception as exc:
        logger.warning("Orphaned-job sweep failed: %s", exc)

    # Run an initial cleanup pass on startup, then schedule the periodic loop
    if settings.cleanup_enabled:
        try:
            cleaned = _run_storage_maintenance(
                settings.cleanup_max_age_days, settings.cleanup_uploads_only,
            )
            logger.info("Startup cleanup: %d job(s) cleaned", cleaned)
        except Exception as exc:
            logger.warning("Startup cleanup failed: %s", exc)

        task = asyncio.create_task(
            _cleanup_loop(settings.cleanup_max_age_days, settings.cleanup_uploads_only)
        )

    yield

    # Cancel the background task on shutdown
    if settings.cleanup_enabled:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="AlignAI",
    description="LiDAR scan-to-plan alignment API",
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)
app.include_router(export_router)
app.include_router(vectorize_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
