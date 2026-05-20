"""Jobs API routes."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

MAX_SCAN_MB = 2048   # 2 GB hard limit per individual scan file
MAX_SCANS   = 100    # max scan files per job (≈ full building floor with overlap)

_LAS_MAGIC = b"LASF"
_PLY_MAGIC = b"ply"


def _validate_scan_magic(content: bytes, filename: str) -> None:
    """Reject files whose magic bytes don't match their extension."""
    ext = Path(filename).suffix.lower()
    if ext in (".las", ".laz") and content[:4] != _LAS_MAGIC:
        raise HTTPException(400, f"{filename!r} is not a valid LAS/LAZ file (bad magic bytes)")
    if ext == ".ply" and content[:3] != _PLY_MAGIC:
        raise HTTPException(400, f"{filename!r} is not a valid PLY file (bad magic bytes)")

from ..config import get_settings
from ..models.job import FloorCandidate, JobCreate, JobDetail, JobStatus, ReprocessRequest
from ..models.room import RoomSchema
from ..models.fixture import FixtureSchema
from ..storage import (
    create_job, load_job, load_result_json, update_job_status,
    uploads_dir, results_dir, artifacts_dir,
)
from ..sse import progress_generator
from ..pipeline.runner import run_pipeline, run_pipeline_guarded, reprocess_rooms_guarded
from ..pipeline.ai_review import review_alignment
from ..pipeline.align import align_scan_to_plan
from ..pipeline.rooms import extract_wall_lines
from ..pipeline.ingest import load_point_cloud

router = APIRouter(prefix="/api/jobs")


@router.post("", response_model=JobCreate)
async def create_job_endpoint(
    bg: BackgroundTasks,
    scans: list[UploadFile] = File(...),
    plan: UploadFile | None = File(default=None),
    band_low_m: float = Form(default=0.75),
    band_high_m: float = Form(default=1.80),
):
    """Upload scan(s) and an optional DXF plan, kick off the alignment pipeline.

    - With plan: full scan-to-plan alignment.
    - Without plan: scan-only mode — synthetic floor plan generated from the scan.
    """
    if not scans:
        raise HTTPException(400, "At least one scan file is required")
    if len(scans) > MAX_SCANS:
        raise HTTPException(
            400,
            f"Too many scan files ({len(scans)}). Maximum is {MAX_SCANS} per job.",
        )

    job_id = str(uuid.uuid4())
    upload_dir = uploads_dir(job_id)

    # Save plan (optional)
    plan_path: Path | None = None
    plan_filename = ""
    if plan is not None and plan.filename:
        plan_content = await plan.read()
        if len(plan_content) > 0:
            plan_path = upload_dir / _safe_filename(plan.filename)
            plan_path.write_bytes(plan_content)
            plan_filename = plan_path.name

    # Save all scan files
    scan_paths: list[Path] = []
    scan_filenames: list[str] = []
    for i, scan_file in enumerate(scans):
        # Size check using seek/tell — no content loaded into RAM yet.
        scan_file.file.seek(0, 2)
        size_bytes = scan_file.file.tell()
        scan_file.file.seek(0)
        if size_bytes == 0:
            raise HTTPException(400, f"Scan file {scan_file.filename!r} is empty")
        if size_bytes > MAX_SCAN_MB * 1024 * 1024:
            raise HTTPException(
                413,
                f"Scan file {scan_file.filename!r} is too large "
                f"({size_bytes / 1e9:.1f} GB, max {MAX_SCAN_MB // 1024} GB)",
            )

        safe_name = f"{i:02d}_{_safe_filename(scan_file.filename or f'scan_{i}.laz')}"
        path = upload_dir / safe_name

        # Stream directly from the spooled upload buffer to disk — avoids
        # creating a second in-memory copy of the entire file.
        with open(path, "wb") as f_out:
            shutil.copyfileobj(scan_file.file, f_out)

        # Magic-byte validation reads only the first 4 bytes from the saved file.
        with open(path, "rb") as f_head:
            magic = f_head.read(4)
        _validate_scan_magic(magic, scan_file.filename or "")

        scan_paths.append(path)
        scan_filenames.append(safe_name)

    create_job(job_id, scan_filenames, plan_filename)
    bg.add_task(run_pipeline_guarded, job_id, scan_paths, plan_path, band_low_m, band_high_m)

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
    """Return plan line segments as GeoJSON for the viewer.

    In scan-only mode, returns the synthetic wall segments generated from the scan.
    In aligned mode, returns the original DXF line segments.
    """
    result = _load_result_or_404(job_id)

    if result.get("scan_only"):
        # Use stored plan_segments if available; fall back to bounding-box rect
        stored_segs = result.get("plan_segments", [])
        if stored_segs:
            features = [_seg_feature(seg) for seg in stored_segs]
        else:
            bounds = result.get("plan_bounds", [0, 0, 100, 100])
            minx, miny, maxx, maxy = bounds
            features = [
                _seg_feature([[minx, miny], [maxx, miny]]),
                _seg_feature([[maxx, miny], [maxx, maxy]]),
                _seg_feature([[maxx, maxy], [minx, maxy]]),
                _seg_feature([[minx, maxy], [minx, miny]]),
            ]
        return {"type": "FeatureCollection", "features": features}

    plan_path = _find_plan_file(job_id)
    lines = extract_wall_lines(str(plan_path))
    features = [
        _seg_feature(seg)
        for seg in lines.tolist()
    ]
    return {"type": "FeatureCollection", "features": features}


def _seg_feature(coords: list) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {},
    }


@router.get("/{job_id}/rooms")
async def get_rooms(job_id: str):
    result = _load_result_or_404(job_id)
    return {"rooms": result.get("rooms", [])}


@router.get("/{job_id}/fixtures")
async def get_fixtures(job_id: str):
    result = _load_result_or_404(job_id)
    return {"fixtures": result.get("fixtures", [])}


@router.get("/{job_id}/outline")
async def get_building_outline(job_id: str):
    """Return the building exterior alphashape hull as a GeoJSON Polygon.

    The polygon is in the scan's local coordinate frame (same as rooms/plan).
    Returns 404 if the outline was not computed for this job (e.g. alphashape
    failed or the job predates this feature).
    """
    result = _load_result_or_404(job_id)
    coords = result.get("building_outline", [])
    if not coords:
        raise HTTPException(404, "Building outline not available for this job")
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {},
            }
        ],
    }


@router.get("/{job_id}/overlay")
async def get_overlay_png(job_id: str):
    """Return the scan-vs-plan 2D overlay image (PNG) for visual inspection."""
    result = _load_result_or_404(job_id)
    overlay_path = result.get("overlay_png", "")
    if not overlay_path or not Path(overlay_path).exists():
        raise HTTPException(404, "Overlay image not available for this job")
    return FileResponse(str(overlay_path), media_type="image/png", filename="overlay.png")


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

    matrix = body.get("transformation")
    if not matrix or len(matrix) != 4:
        raise HTTPException(400, "Expected 4x4 transformation matrix")

    T = np.array(matrix, dtype=np.float64)

    # Prefer the merged+decimated PLY that was written during the pipeline run.
    # For multi-scan jobs this contains the full merged cloud, not just the first
    # raw scan file that _find_scan_file() would return.
    merged_ply = Path(result.get("merged_ply", ""))
    if merged_ply.exists():
        scan_pcd = load_point_cloud(merged_ply)
    else:
        scan_path = _find_scan_file(job_id)
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


@router.get("/{job_id}/floors", response_model=list[FloorCandidate])
async def get_floors(job_id: str):
    """Return all horizontal floor planes detected in this scan (MJ2).

    The list is sorted by wall-content score (descending) — the first entry
    is the floor that was used for the original job run.  Subsequent entries
    are other levels such as upper floors, mezzanines, or parking decks.

    Each entry includes ``floor_z`` (elevation in the scan's local frame),
    ``inlier_count``, ``wall_score`` (points 0.3–3 m above — higher = more
    likely an indoor floor with lots of walls), and ``axis_idx`` (2=Z-up).

    Use ``floor_z`` values from this list as the ``floor_z`` override in
    POST /api/jobs/:id/reprocess to switch to a different building level.
    """
    job = load_job(job_id)
    if not job.floor_candidates:
        raise HTTPException(404, "Floor level data not available for this job "
                            "(job may pre-date multi-floor support)")
    return job.floor_candidates


@router.post("/{job_id}/reprocess")
async def reprocess_job(job_id: str, body: ReprocessRequest, bg: BackgroundTasks):
    """Re-run room/wall extraction on the already-merged scan (MJ2 + MJ3).

    Skips the expensive scan-loading and merge stages (10–15 min) and
    re-runs only floor detection → wall band → wall planes → plan generation
    → rooms → fixtures.

    The SSE stream (GET /api/jobs/:id/sse) receives progress events with
    stage ``"reprocess"`` while running, and ``"complete"`` when done.

    Body parameters
    ---------------
    floor_z : float | null
        Override the floor elevation (use a ``floor_z`` from GET /floors).
        Null = re-detect automatically.
    band_low_m / band_high_m : float
        Wall-band height offsets above the floor plane.
    min_wall_length_m : float
        Minimum wall segment length kept by the Hough filter.
    hough_threshold : int
        Minimum Hough vote count (lower → more lines, more noise).
    """
    result = _load_result_or_404(job_id)

    merged_ply = Path(result.get("merged_ply", ""))
    if not merged_ply.exists():
        raise HTTPException(
            409,
            "Merged scan not found for this job — cannot re-process. "
            "The raw scan may have been deleted by the storage cleanup cron, "
            "or this job was processed before re-processing support was added.",
        )

    plan_path: Path | None = None
    try:
        plan_path = _find_plan_file(job_id)
    except HTTPException:
        pass   # scan-only mode — plan_path stays None

    # Drop any prior replay buffer so a re-process doesn't immediately replay
    # the last run's "complete" event to fresh subscribers.
    from ..sse import clear_history
    clear_history(job_id)

    bg.add_task(
        reprocess_rooms_guarded,
        job_id,
        merged_ply,
        plan_path,
        body.floor_z,
        body.band_low_m,
        body.band_high_m,
        body.min_wall_length_m,
        body.hough_threshold,
    )

    return {"job_id": job_id, "status": "reprocessing"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_result_or_404(job_id: str) -> dict:
    try:
        return load_result_json(job_id)
    except KeyError:
        raise HTTPException(404, f"Results for job {job_id} not ready or not found")


def _find_plan_file(job_id: str) -> Path:
    upload_dir = uploads_dir(job_id)
    for f in sorted(upload_dir.iterdir()):
        if f.suffix.lower() == ".dxf":
            return f
    raise HTTPException(404, "No DXF plan found for this job (scan-only mode)")


def _find_scan_file(job_id: str) -> Path:
    """Return the first scan file found (used by manual-transform endpoint)."""
    upload_dir = uploads_dir(job_id)
    for f in sorted(upload_dir.iterdir()):
        if f.suffix.lower() in (".las", ".laz", ".ply", ".e57"):
            return f
    raise HTTPException(404, "Scan file not found for this job")


def _find_scan_files(job_id: str) -> list[Path]:
    """Return all scan files for this job."""
    upload_dir = uploads_dir(job_id)
    return sorted(
        f for f in upload_dir.iterdir()
        if f.suffix.lower() in (".las", ".laz", ".ply", ".e57")
    )


def _safe_filename(name: str) -> str:
    # Strip directory components first, then allow only safe characters
    base = Path(name).name
    return "".join(c for c in base if c.isalnum() or c in "._-").replace("..", "_")
