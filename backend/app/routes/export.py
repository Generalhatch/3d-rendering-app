"""Export download routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..storage import results_dir, load_result_json

router = APIRouter(prefix="/api/jobs")


@router.get("/{job_id}/export/json")
async def export_json(job_id: str):
    path = results_dir(job_id) / "aligned.json"
    if not path.exists():
        raise HTTPException(404, "Export not ready — run alignment first")
    return FileResponse(str(path), media_type="application/json", filename="aligned.json")


@router.get("/{job_id}/export/dxf")
async def export_dxf(job_id: str):
    path = results_dir(job_id) / "aligned.dxf"
    if not path.exists():
        raise HTTPException(404, "Export not ready — run alignment first")
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        filename="aligned.dxf",
    )
