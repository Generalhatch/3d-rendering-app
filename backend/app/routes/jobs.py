"""Jobs API routes."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..config import get_settings
from ..models.job import JobCreate, JobDetail, JobStatus
from ..models.room import RoomSchema
from ..models.fixture import FixtureSchema
from ..storage import (
    create_job, load_job, load_result_json, update_job_status,
    uploads_dir, results_dir, artifacts_dir,
)
from ..sse import progress_generator
from ..pipeline.runner import run_pipeline
from ..pipeline.ai_review import review_alignment
from ..pipeline.align import align_scan_to_plan
from ..pipeline.rooms import extract_wall_lines
from ..pipeline.ingest import load_point_cloud

router = APIRouter(prefix="/api/jobs")


@router.post("", response_model=JobCreate)
async def create_job_endpoint(
    bg: BackgroundTasks,
    plan: UploadFile = File(...),
    scan: UploadFile = File(...),
    band_low_m: float = Form(default=0.75),
    band_high_m: float = Form(default=1.80),
):
    """Upload a plan + scan, kick off the alignment pipeline."""
    job_id = str(uuid.uuid4())
    upload_dir = uploads_dir(job_id)

    # Save uploaded files
    plan_path = upload_dir / _safe_filename(plan.filename or "plan.dxf")
    scan_path  = upload_dir / _safe_filename(scan.filename  or "scan.ply")

    plan_content = await plan.read()
    scan_content = await scan.read()

    if len(plan_content) == 0:
        raise HTTPException(400, "Plan file is empty")
    if len(scan_content) == 0:
        raise HTTPException(400, "Scan file is empty")

    plan_path.write_bytes(plan_content)
    scan_path.write_bytes(scan_content)

    create_job(job_id, scan_path.name, plan_path.name)
    bg.add_task(run_pipeline, job_id, scan_path, plan_path, band_low_m, band_high_m)

    return JobCreate(job_id=job_id, status=JobStatus.queued)


@router.get("/{job_id}", response_model=JobDetail)
async def get_job(job_id: str):
    try:
        return load_job(job_id)
    except KeyError:
        raise HTTPException(404, f"Job {job_id} not found")


@router.get("/{job_id}/sse")
async def progress_stream(job_id: str):
    return StreamingResponse(
        progress_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/scan")
async def get_scan_decimated(job_id: str):
    """Return the decimated PLY for the browser viewer."""
    path = artifacts_dir(job_id) / "scan_decimated.ply"
    if not path.exists():
        raise HTTPException(404, "Decimated scan not ready yet")
    return FileResponse(str(path), media_type="application/octet-stream", filename="scan.ply")


@router.get("/{job_id}/plan")
async def get_plan_geojson(job_id: str):
    """Return plan line segments as GeoJSON for the viewer."""
    result = _load_result_or_404(job_id)
    plan_path = _find_plan_file(job_id)
    lines = extract_wall_lines(str(plan_path))
    features = []
    for seg in lines.tolist():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": seg,
            },
            "properties": {},
        })
    return {"type": "FeatureCollection", "features": features}


@router.get("/{job_id}/rooms")
async def get_rooms(job_id: str):
    result = _load_result_or_404(job_id)
    return {"rooms": result.get("rooms", [])}


@router.get("/{job_id}/fixtures")
async def get_fixtures(job_id: str):
    result = _load_result_or_404(job_id)
    return {"fixtures": result.get("fixtures", [])}


@router.post("/{job_id}/ai")
async def trigger_ai_review(job_id: str):
    """Call OpenRouter vision model to review the alignment overlay."""
    result = _load_result_or_404(job_id)
    settings = get_settings()
    overlay_path = result.get("overlay_png", "")
    alignment = result.get("alignment", {})
    confidence = alignment.get("confidence", 0.0)
    residual_mm = alignment.get("residual_rmse_mm", 0.0)

    review = await review_alignment(
        overlay_path,
        confidence,
        residual_mm,
        settings.openrouter_api_key,
        settings.ai_review_model,
    )
    return {"ai_review": review}


@router.post("/{job_id}/manual")
async def apply_manual_transform(job_id: str, body: dict):
    """Apply a user-supplied 4x4 transform and recompute confidence."""
    import numpy as np
    from ..pipeline.confidence import compute_confidence_pct
    from ..pipeline.align import run_icp, lines_to_pcd, transform_pcd
    from ..storage import update_job_result
    import time

    result = _load_result_or_404(job_id)
    plan_path = _find_plan_file(job_id)
    scan_path = _find_scan_file(job_id)

    matrix = body.get("transformation")
    if not matrix or len(matrix) != 4:
        raise HTTPException(400, "Expected 4x4 transformation matrix")

    T = np.array(matrix, dtype=np.float64)

    # Reload scan, apply transform, run ICP to get new residual
    scan_pcd = load_point_cloud(scan_path)
    plan_lines = extract_wall_lines(str(plan_path))

    from ..pipeline.slicing import detect_floor, extract_wall_band
    floor = detect_floor(scan_pcd)
    wall_band = extract_wall_band(scan_pcd, floor)
    plan_cloud = lines_to_pcd(plan_lines)
    transformed_band = transform_pcd(wall_band, T)

    icp_result = run_icp(transformed_band, plan_cloud, T)

    from ..pipeline.align import _compute_confidence
    from dataclasses import dataclass

    @dataclass
    class _Stub:
        residual_rmse: float
        inlier_ratio: float
        num_wall_planes: int
        floor: object
        wall_planes: list

    stub = _Stub(
        residual_rmse=icp_result.residual_rmse,
        inlier_ratio=icp_result.inlier_ratio,
        num_wall_planes=result["alignment"].get("num_wall_planes", 4),
        floor=floor,
        wall_planes=[],
    )
    confidence = _compute_confidence(stub)
    conf_pct = compute_confidence_pct(confidence)

    # Update the stored result with the new transform and scores
    result["alignment"]["transformation"] = icp_result.transformation.tolist()
    result["alignment"]["residual_rmse"] = float(icp_result.residual_rmse)
    result["alignment"]["residual_rmse_mm"] = float(icp_result.residual_rmse * 1000)
    result["alignment"]["confidence"] = float(confidence)
    result["alignment"]["confidence_pct"] = conf_pct
    result["alignment"]["inlier_ratio"] = float(icp_result.inlier_ratio)

    update_job_result(job_id, result, time.time())

    return {
        "transformation": icp_result.transformation.tolist(),
        "confidence": confidence,
        "confidence_pct": conf_pct,
        "residual_rmse_mm": float(icp_result.residual_rmse * 1000),
    }


@router.post("/{job_id}/approve")
async def approve_job(job_id: str):
    try:
        update_job_status(job_id, JobStatus.approved)
        return {"status": "approved"}
    except KeyError:
        raise HTTPException(404, f"Job {job_id} not found")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_result_or_404(job_id: str) -> dict:
    try:
        return load_result_json(job_id)
    except KeyError:
        raise HTTPException(404, f"Results for job {job_id} not ready or not found")


def _find_plan_file(job_id: str) -> Path:
    upload_dir = uploads_dir(job_id)
    for f in upload_dir.iterdir():
        if f.suffix.lower() == ".dxf":
            return f
    raise HTTPException(404, "Plan DXF not found for this job")


def _find_scan_file(job_id: str) -> Path:
    upload_dir = uploads_dir(job_id)
    for f in upload_dir.iterdir():
        if f.suffix.lower() in (".las", ".laz", ".ply", ".e57"):
            return f
    raise HTTPException(404, "Scan file not found for this job")


def _safe_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "._-")
