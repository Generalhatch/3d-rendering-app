"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .storage import init_db, cleanup_old_jobs
from .routes.jobs import router as jobs_router
from .routes.export import router as export_router
from .routes.vectorize import router as vectorize_router

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL_S = 24 * 3600  # run once every 24 hours


async def _cleanup_loop(max_age_days: int, uploads_only: bool) -> None:
    """Background coroutine: run storage cleanup every 24 hours."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_S)
        try:
            cleaned = await asyncio.to_thread(cleanup_old_jobs, max_age_days, uploads_only)
            logger.info("Periodic cleanup complete: %d job(s) cleaned", cleaned)
        except Exception as exc:
            logger.warning("Periodic cleanup failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    # Run an initial cleanup pass on startup, then schedule the periodic loop
    if settings.cleanup_enabled:
        try:
            cleaned = cleanup_old_jobs(settings.cleanup_max_age_days, settings.cleanup_uploads_only)
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
