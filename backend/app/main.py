"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .storage import init_db
from .routes.jobs import router as jobs_router
from .routes.export import router as export_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    init_db()
    yield


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


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
