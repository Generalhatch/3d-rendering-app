"""HTTP routes for the Vectorize (scan-to-CAD) pipeline.

Endpoints
---------
- ``POST   /api/vectorize``                     Upload a scan, queue a job
- ``GET    /api/vectorize/{job_id}``            Status + result metadata
- ``GET    /api/vectorize/{job_id}/sse``        Server-Sent Events progress
- ``GET    /api/vectorize/{job_id}/raster``     Slice PNG (cleaned)
- ``GET    /api/vectorize/{job_id}/slice``      Slice PNG (raw, pre-cleanup)
- ``GET    /api/vectorize/{job_id}/overlay``    Detected lines overlay PNG
- ``GET    /api/vectorize/{job_id}/overlay/raw`` Raw detector overlay PNG
- ``GET    /api/vectorize/{job_id}/coverage``   RGBA gaps mask (uncovered walls)
- ``GET    /api/vectorize/{job_id}/dxf``        Download the vectorized DXF
- ``POST   /api/vectorize/{job_id}/reprocess``  Re-run with new params
- ``GET    /api/vectorize/{job_id}/segments``           Editable segment list
- ``GET    /api/vectorize/{job_id}/rejected-segments``  Ghost candidates the pipeline dropped
- ``POST   /api/vectorize/{job_id}/find-duplicates``    Suggest near-duplicate pairs to merge
- ``GET    /api/vectorize/{job_id}/edits/log``          Edit history (append-only)
- ``POST   /api/vectorize/{job_id}/edits``              Save edits + re-emit DXF
- ``POST   /api/vectorize/{job_id}/generate-measurement`` Rebuild BOMA from edited walls
- ``GET    /api/vectorize/{job_id}/scan``               Original uploaded scan file
- ``GET    /api/vectorize``                     List recent jobs
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import (
    APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse

from ..models.vectorize_job import (
    DetectorName,
    DuplicatePair,
    FindDuplicatesRequest,
    FindDuplicatesResponse,
    GenerateMeasurementRequest,
    GenerateMeasurementResponse,
    SaveEditsRequest,
    SaveEditsResponse,
    SheetOverridesRequest,
    SheetOverridesResponse,
    VectorizeJobCreate,
    VectorizeJobDetail,
    VectorizeMetrics,
    VectorizeParams,
    VectorizeReprocessRequest,
    VectorizeStatus,
)
from ..sse import progress_generator
from ..storage import (
    artifacts_dir,
    create_vectorize_job,
    list_vectorize_jobs,
    load_vectorize_job,
    load_vectorize_result,
    results_dir,
    store_upload_deduped,
    update_vectorize_params,
    update_vectorize_result,
    update_vectorize_status,
    uploads_dir,
)
from ..vectorize import edits as edits_mod
from ..vectorize.measure_from_segments import (
    RoomsNotClosedError,
    generate_measurement_from_segments,
)
from ..vectorize.pipeline import run_vectorize_guarded

MAX_SCAN_MB = 2048

router = APIRouter(prefix="/api/vectorize", tags=["vectorize"])

# Supported scan formats for the vectorize pipeline.  Matches the alignment
# pipeline's ingest layer (LAS, LAZ, PLY, E57) — we go through the same loader.
_VALID_SCAN_SUFFIXES = {".las", ".laz", ".ply", ".e57"}


# ── POST /api/vectorize ──────────────────────────────────────────────────────

@router.post("", response_model=VectorizeJobCreate)
async def create_vectorize(
    bg: BackgroundTasks,
    scan: UploadFile = File(..., description="LAS/LAZ/PLY/E57 scan file"),
    # Params come in as form fields rather than a nested JSON blob so the
    # upload + params fit in a single multipart request (browser-friendly).
    # Defaults MUST mirror VectorizeParams.model_fields — keep them in sync.
    elevation_m: float | None = Form(default=None),
    slab_thickness_m: float = Form(default=0.10),
    resolution_m_per_px: float = Form(default=0.01),
    detector: DetectorName = Form(default=DetectorName.both),
    min_wall_length_m: float = Form(default=0.40),
    manhattan_snap: bool = Form(default=True),
    merge_collinear: bool = Form(default=True),
    remove_speckle: bool = Form(default=True),
    multi_elevation: bool = Form(default=True),
    detect_openings: bool = Form(default=True),
    detect_columns: bool = Form(default=True),
    snapping_distance_m: float = Form(default=0.85),
):
    """Upload a scan and start a vectorize job.

    The scan is streamed to disk, validated, and the pipeline is queued via
    :class:`fastapi.BackgroundTasks`.  Returns immediately with the job id —
    poll status via ``GET /api/vectorize/{id}`` or subscribe to live progress
    via ``GET /api/vectorize/{id}/sse``.
    """
    if not scan.filename:
        raise HTTPException(400, "Scan file is required")

    suffix = Path(scan.filename).suffix.lower()
    if suffix not in _VALID_SCAN_SUFFIXES:
        raise HTTPException(
            400,
            f"Unsupported scan format {suffix!r}. "
            f"Supported: {', '.join(sorted(_VALID_SCAN_SUFFIXES))}",
        )

    job_id = str(uuid.uuid4())
    safe_name = _safe_filename(scan.filename)

    scan.file.seek(0, 2)
    size_bytes = scan.file.tell()
    scan.file.seek(0)
    if size_bytes == 0:
        raise HTTPException(400, "Scan file is empty")
    if size_bytes > MAX_SCAN_MB * 1024 * 1024:
        raise HTTPException(
            413,
            f"Scan file is too large ({size_bytes / 1e9:.1f} GB, max "
            f"{MAX_SCAN_MB // 1024} GB)",
        )

    # Stream to the content-addressed store (sha256 computed while writing,
    # no second pass) — identical bytes uploaded twice are stored once, and
    # the job dir gets a symlink to the shared blob.
    scan_path, _scan_hash = store_upload_deduped(scan.file, job_id, safe_name)

    params = VectorizeParams(
        elevation_m=elevation_m,
        slab_thickness_m=slab_thickness_m,
        resolution_m_per_px=resolution_m_per_px,
        detector=detector,
        min_wall_length_m=min_wall_length_m,
        manhattan_snap=manhattan_snap,
        merge_collinear=merge_collinear,
        remove_speckle=remove_speckle,
        multi_elevation=multi_elevation,
        detect_openings=detect_openings,
        detect_columns=detect_columns,
        snapping_distance_m=snapping_distance_m,
    )

    create_vectorize_job(job_id, safe_name, params.model_dump(mode="json"))
    _enqueue_run(bg, job_id, scan_path, params)

    return VectorizeJobCreate(job_id=job_id, status=VectorizeStatus.queued)


# ── GET /api/vectorize/{job_id} ──────────────────────────────────────────────

@router.get("/{job_id}", response_model=VectorizeJobDetail)
async def get_vectorize(job_id: str):
    try:
        record = load_vectorize_job(job_id)
    except KeyError:
        raise HTTPException(404, f"Vectorize job {job_id} not found")

    artifact_dir = artifacts_dir(job_id)
    result_dir = results_dir(job_id)
    metrics: VectorizeMetrics | None = None
    warnings: list = []
    room_confidence: list = []
    if record["result"] is not None:
        m = record["result"].get("metrics")
        if m:
            metrics = VectorizeMetrics.model_validate(m)
        warnings = record["result"].get("warnings") or []
        room_confidence = record["result"].get("room_confidence") or []

    return VectorizeJobDetail(
        job_id=record["job_id"],
        status=VectorizeStatus(record["status"]),
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        scan_filename=record["scan_filename"],
        params=VectorizeParams.model_validate(record["params"]) if record["params"] else None,
        metrics=metrics,
        error_message=record["error_message"],
        warnings=warnings,
        room_confidence=room_confidence,
        has_raster=(artifact_dir / "slice_cleaned.png").exists(),
        has_overlay=(artifact_dir / "overlay.png").exists(),
        has_dxf=(result_dir / "vectorized.dxf").exists(),
        has_coverage=(artifact_dir / "coverage_gaps.png").exists(),
        has_rejected=(result_dir / "rejected_segments.json").exists(),
        has_sheet=(result_dir / "sheet.svg").exists(),
        has_floor_geometry=(result_dir / "floor_geometry.json").exists(),
        has_measurement_report=(result_dir / "measurement_report.json").exists(),
    )


# ── GET /api/vectorize/{job_id}/sse ──────────────────────────────────────────

@router.get("/{job_id}/sse")
async def vectorize_sse(job_id: str):
    return StreamingResponse(
        progress_generator(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Artifact downloads ────────────────────────────────────────────────────────

@router.get("/{job_id}/raster")
async def get_raster(job_id: str):
    """The cleaned raster — what the detector actually saw."""
    path = artifacts_dir(job_id) / "slice_cleaned.png"
    if not path.exists():
        raise HTTPException(404, "Raster not ready yet")
    return FileResponse(str(path), media_type="image/png", filename="slice_cleaned.png")


@router.get("/{job_id}/slice")
async def get_slice_raw(job_id: str):
    """The raw raster before preprocessing — useful for debugging."""
    path = artifacts_dir(job_id) / "slice.png"
    if not path.exists():
        raise HTTPException(404, "Slice not ready yet")
    return FileResponse(str(path), media_type="image/png", filename="slice.png")


@router.get("/{job_id}/overlay")
async def get_overlay(job_id: str):
    """The cleaned raster with regularized lines drawn in red — the operator preview."""
    path = artifacts_dir(job_id) / "overlay.png"
    if not path.exists():
        raise HTTPException(404, "Overlay not ready yet")
    return FileResponse(str(path), media_type="image/png", filename="overlay.png")


@router.get("/{job_id}/overlay/raw")
async def get_overlay_raw(job_id: str):
    """The raw detector output overlaid in orange — for comparing detectors."""
    path = artifacts_dir(job_id) / "overlay_raw.png"
    if not path.exists():
        raise HTTPException(404, "Raw overlay not ready yet")
    return FileResponse(str(path), media_type="image/png", filename="overlay_raw.png")


@router.get("/{job_id}/coverage")
async def get_coverage_gaps(job_id: str):
    """RGBA PNG marking wall pixels that no kept segment covers.

    Transparent everywhere except over "missed wall" pixels, which are drawn
    in semi-opaque red.  The editor blends this directly over the raster as a
    diagnostic overlay — bright red = recall failure (or furniture clutter
    the operator should ignore).
    """
    path = artifacts_dir(job_id) / "coverage_gaps.png"
    if not path.exists():
        raise HTTPException(404, "Coverage diagnostic not ready yet")
    return FileResponse(str(path), media_type="image/png", filename="coverage_gaps.png")


@router.get("/{job_id}/dxf")
async def get_dxf(job_id: str):
    """The vectorized DXF deliverable."""
    path = results_dir(job_id) / "vectorized.dxf"
    if not path.exists():
        raise HTTPException(404, "DXF not ready yet")
    return FileResponse(
        str(path), media_type="application/dxf", filename="vectorized.dxf",
    )


@router.get("/{job_id}/scan")
async def get_vectorize_scan(job_id: str):
    """Download the original uploaded scan for this vectorize job."""
    try:
        record = load_vectorize_job(job_id)
    except KeyError:
        raise HTTPException(404, f"Vectorize job {job_id} not found")
    name = record.get("scan_filename") or ""
    path = uploads_dir(job_id) / name if name else None
    if path is None or not path.exists():
        # Symlink may resolve; also try any file in the upload dir.
        u_dir = uploads_dir(job_id)
        candidates = [p for p in u_dir.iterdir() if p.is_file() or p.is_symlink()] if u_dir.exists() else []
        if not candidates:
            raise HTTPException(404, "Original scan not available for this job")
        path = candidates[0]
        name = path.name
    return FileResponse(
        str(path.resolve()),
        media_type="application/octet-stream",
        filename=name or path.name,
    )


@router.get("/{job_id}/floor-geometry.json")
async def download_floor_geometry(job_id: str):
    """Download ``floor_geometry.json`` as an attachment."""
    path = results_dir(job_id) / "floor_geometry.json"
    if not path.exists():
        raise HTTPException(404, "Floor geometry not available for this job")
    return FileResponse(
        str(path),
        media_type="application/json",
        filename="floor_geometry.json",
    )


# ── Floor plan sheet (Phase 3) ───────────────────────────────────────────────

@router.get("/{job_id}/sheet")
async def get_sheet_svg(job_id: str):
    """The rendered floor plan sheet (SVG).

    Lazily renders on first request for jobs that completed before the
    sheet stage existed (requires ``floor_geometry.json``).
    """
    r_dir = results_dir(job_id)
    path = r_dir / "sheet.svg"
    if not path.exists():
        if not (r_dir / "floor_geometry.json").exists():
            raise HTTPException(404, "Sheet not available — no floor geometry")
        from ..sheet.service import render_job_sheet
        render_job_sheet(r_dir)
    return FileResponse(str(path), media_type="image/svg+xml", filename="sheet.svg")


@router.get("/{job_id}/sheet.pdf")
async def get_sheet_pdf(job_id: str):
    """The rendered floor plan sheet (PDF, converted from the same SVG)."""
    r_dir = results_dir(job_id)
    path = r_dir / "sheet.pdf"
    if not path.exists():
        if not (r_dir / "floor_geometry.json").exists():
            raise HTTPException(404, "Sheet not available — no floor geometry")
        from ..sheet.service import render_job_sheet
        render_job_sheet(r_dir)
    return FileResponse(str(path), media_type="application/pdf", filename="sheet.pdf")


@router.get("/{job_id}/floor-geometry")
async def get_floor_geometry(job_id: str):
    """The canonical FloorGeometry JSON — the sheet editor's data source."""
    path = results_dir(job_id) / "floor_geometry.json"
    if not path.exists():
        raise HTTPException(404, "Floor geometry not available for this job")
    return json.loads(path.read_text())


@router.get("/{job_id}/measurement-report")
async def get_measurement_report(job_id: str):
    """BOMA/REBNY/Gross measurement report JSON — read-only deliverable."""
    path = results_dir(job_id) / "measurement_report.json"
    if not path.exists():
        raise HTTPException(404, "Measurement report not available for this job")
    return json.loads(path.read_text())


@router.get("/{job_id}/measurement-report.pdf")
async def get_measurement_report_pdf(job_id: str):
    """Measurement report PDF (same numbers as the JSON report)."""
    path = results_dir(job_id) / "measurement_report.pdf"
    if not path.exists():
        raise HTTPException(404, "Measurement report PDF not available for this job")
    return FileResponse(
        str(path), media_type="application/pdf", filename="measurement_report.pdf",
    )


@router.get("/{job_id}/sheet/overrides")
async def get_sheet_overrides(job_id: str):
    """Current operator overrides (suite labels, title block, manual grid)."""
    from ..sheet.service import load_overrides
    return load_overrides(results_dir(job_id))


@router.post("/{job_id}/sheet/overrides", response_model=SheetOverridesResponse)
async def save_sheet_overrides(job_id: str, body: SheetOverridesRequest):
    """Persist sheet overrides and re-render sheet.svg / sheet.pdf.

    ``labels`` merges per-room (empty string clears one override);
    ``meta`` and ``manual_grid`` replace wholesale (null removes).
    """
    r_dir = results_dir(job_id)
    if not (r_dir / "floor_geometry.json").exists():
        raise HTTPException(404, "Sheet not available — no floor geometry")

    from ..sheet.service import render_job_sheet, save_overrides
    save_overrides(r_dir, body.model_dump(exclude_unset=True))
    try:
        render = render_job_sheet(r_dir)
    except Exception as exc:
        raise HTTPException(500, f"Sheet re-render failed: {exc}")

    return SheetOverridesResponse(
        job_id=job_id,
        sheet_url=f"/api/vectorize/{job_id}/sheet",
        scale_denominator=render.scale_denominator,
    )


# ── POST /api/vectorize/{job_id}/reprocess ───────────────────────────────────

@router.post("/{job_id}/reprocess", response_model=VectorizeJobCreate)
async def reprocess_vectorize(
    job_id: str, body: VectorizeReprocessRequest, bg: BackgroundTasks,
):
    """Re-run the pipeline on the already-uploaded scan with new parameters.

    Lets the operator iterate on elevation / detector / regularize settings
    without re-uploading a multi-GB scan file.  The new params replace the
    stored params (history is not retained — keep it simple for now).
    """
    try:
        record = load_vectorize_job(job_id)
    except KeyError:
        raise HTTPException(404, f"Vectorize job {job_id} not found")

    upload_dir = uploads_dir(job_id)
    scan_path = upload_dir / record["scan_filename"]
    if not scan_path.exists():
        raise HTTPException(
            409,
            "Scan file no longer exists for this job — it may have been "
            "removed by the storage cleanup cron.  Please re-upload.",
        )

    update_vectorize_params(job_id, body.params.model_dump(mode="json"))
    _enqueue_run(bg, job_id, scan_path, body.params)

    return VectorizeJobCreate(job_id=job_id, status=VectorizeStatus.queued)


# ── Editor endpoints (Phase 3) ───────────────────────────────────────────────

@router.get("/{job_id}/segments")
async def get_segments(job_id: str):
    """Return the current editable segment list for this job.

    Reflects the latest save — if the operator has saved edits, those are
    returned; otherwise the original pipeline output.  Also attaches
    ``dangling_endpoints`` (degree-1 junctions) when the artifact exists so
    the editor can highlight open wall ends without a second round-trip.
    """
    from ..vectorize import topology as topology_mod

    r_dir = results_dir(job_id)
    try:
        payload = edits_mod.load_segments(r_dir)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    payload["dangling_endpoints"] = topology_mod.load_dangling_endpoints(r_dir)
    return payload


@router.get("/{job_id}/rejected-segments")
async def get_rejected_segments(job_id: str):
    """Segments the pipeline's regularizer dropped, with provenance.

    The editor renders these as ghost candidates the operator can rescue —
    useful when the Manhattan filter or short-length cutoff was too
    aggressive.  Empty payload (not 404) if the job ran before the feature
    was added, so the editor can fall back gracefully.
    """
    r_dir = results_dir(job_id)
    path = r_dir / "rejected_segments.json"
    if not path.exists():
        return {"version": 1, "units": "metres", "rejected": []}
    return json.loads(path.read_text())


@router.post("/{job_id}/find-duplicates", response_model=FindDuplicatesResponse)
async def find_duplicates(job_id: str, body: FindDuplicatesRequest | None = None):
    """Suggest pairs of segments that look like near-duplicates of the same wall.

    Reads the *current* (post-edit) ``segments.json`` so suggestions reflect
    everything the operator has done in this session.  Returns a list of
    ``DuplicatePair`` — the editor sidebar walks through them one at a time
    with [Merge] / [Skip] / [Skip all].
    """
    if body is None:
        body = FindDuplicatesRequest()

    r_dir = results_dir(job_id)
    try:
        payload = edits_mod.load_segments(r_dir)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    raw = payload.get("segments", [])
    if len(raw) < 2:
        return FindDuplicatesResponse(suggestions=[])

    import numpy as np
    from ..vectorize import regularize as reg_mod

    arr = np.array(
        [[[s["x1"], s["y1"]], [s["x2"], s["y2"]]] for s in raw], dtype=np.float64,
    )
    pairs = reg_mod.find_near_duplicate_pairs(
        arr,
        perp_distance_m=body.perp_distance_m,
        parallel_tol_deg=body.parallel_tol_deg,
        endpoint_gap_m=body.endpoint_gap_m,
    )

    suggestions: list[DuplicatePair] = []
    for i, j in pairs:
        merged = reg_mod.merge_segment_group(arr[[i, j]])
        # Re-derive the per-pair metrics so the UI can sort / explain them.
        a, b = arr[i], arr[j]
        da = a[1] - a[0]
        db = b[1] - b[0]
        ang_a = float(np.degrees(np.arctan2(da[1], da[0])) % 180.0)
        ang_b = float(np.degrees(np.arctan2(db[1], db[0])) % 180.0)
        d_ang = abs(ang_a - ang_b)
        d_ang = min(d_ang, 180.0 - d_ang)
        len_a = float(np.linalg.norm(da)) or 1.0
        dir_a = da / len_a
        perp = np.array([-dir_a[1], dir_a[0]])
        ca = (a[0] + a[1]) / 2.0
        cb = (b[0] + b[1]) / 2.0
        perp_dist = abs(float(np.dot(cb - ca, perp)))
        t_a0 = float(np.dot(a[0] - ca, dir_a))
        t_a1 = float(np.dot(a[1] - ca, dir_a))
        t_b0 = float(np.dot(b[0] - ca, dir_a))
        t_b1 = float(np.dot(b[1] - ca, dir_a))
        a_lo, a_hi = min(t_a0, t_a1), max(t_a0, t_a1)
        b_lo, b_hi = min(t_b0, t_b1), max(t_b0, t_b1)
        gap = max(0.0, max(a_lo, b_lo) - min(a_hi, b_hi))
        suggestions.append(DuplicatePair(
            a_id=raw[i]["id"],
            b_id=raw[j]["id"],
            perp_distance_m=perp_dist,
            angle_diff_deg=d_ang,
            endpoint_gap_m=gap,
            merged_x1=float(merged[0, 0]),
            merged_y1=float(merged[0, 1]),
            merged_x2=float(merged[1, 0]),
            merged_y2=float(merged[1, 1]),
        ))

    # Show the most-clearly-duplicate pairs first.
    suggestions.sort(key=lambda p: (p.perp_distance_m, p.endpoint_gap_m))
    return FindDuplicatesResponse(suggestions=suggestions)


@router.get("/{job_id}/edits/log")
async def get_edit_log(job_id: str):
    """Return all save events for this job, oldest first.

    Useful for debugging and (later) for training-data export.
    """
    r_dir = results_dir(job_id)
    return {"saves": edits_mod.load_edit_history(r_dir)}


@router.post("/{job_id}/edits", response_model=SaveEditsResponse)
async def save_edits(job_id: str, body: SaveEditsRequest):
    """Persist operator edits + re-emit the DXF.

    The frontend sends the *complete* current segment list (not a diff) plus
    the full event log from this editor session.  Backend snapshots the
    previous state, writes the new one, and returns a versioned DXF URL.
    """
    # Confirm the job exists and completed at least once.
    try:
        record = load_vectorize_job(job_id)
    except KeyError:
        raise HTTPException(404, f"Vectorize job {job_id} not found")
    if record["status"] != "complete":
        raise HTTPException(
            409,
            f"Cannot save edits on a job in status {record['status']!r}; "
            f"wait for the pipeline to finish.",
        )

    r_dir = results_dir(job_id)
    try:
        result = edits_mod.save_edits(r_dir, job_id, body.segments, body.log)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return SaveEditsResponse(
        job_id=job_id,
        edit_version=result.edit_version,
        dxf_url=f"/api/vectorize/{job_id}/dxf",
        segments_saved=result.segments_saved,
    )


@router.post(
    "/{job_id}/generate-measurement",
    response_model=GenerateMeasurementResponse,
)
async def generate_measurement(job_id: str, body: GenerateMeasurementRequest | None = None):
    """Rebuild floor geometry + BOMA from the current (edited) wall network.

    Used when auto room-closing found zero rooms: the operator closes walls
    in the editor, then calls this endpoint.  Does **not** re-ingest the scan.
    """
    body = body or GenerateMeasurementRequest()
    try:
        record = load_vectorize_job(job_id)
    except KeyError:
        raise HTTPException(404, f"Vectorize job {job_id} not found")
    if record["status"] != "complete":
        raise HTTPException(
            409,
            f"Cannot generate measurement on a job in status "
            f"{record['status']!r}; wait for the pipeline to finish.",
        )

    r_dir = results_dir(job_id)
    edit_version: int | None = None
    if body.save_first or body.segments is not None:
        if not body.segments:
            raise HTTPException(
                400,
                "segments required when save_first is true "
                "(or pass segments to measure without a prior save)",
            )
        try:
            saved = edits_mod.save_edits(r_dir, job_id, body.segments, body.log)
            edit_version = saved.edit_version
        except ValueError as e:
            raise HTTPException(400, str(e))

    if not (r_dir / "segments.json").exists():
        raise HTTPException(404, "No segments.json — run or save the job first")

    params = VectorizeParams.model_validate(record["params"] or {})
    try:
        result = generate_measurement_from_segments(
            r_dir,
            job_id,
            segments=body.segments,
            snapping_distance_m=float(params.snapping_distance_m),
        )
    except RoomsNotClosedError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(e),
                "n_walls": e.n_walls,
                "n_junctions": e.n_junctions,
                "n_edges": e.n_edges,
                "dangling_endpoints": e.dangling_endpoints,
            },
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Measurement generation failed: {e}") from e

    # Merge artifact flags + rooms_detected into the stored result so the
    # job-detail API reflects the new deliverables without a full reprocess.
    try:
        stored = load_vectorize_result(job_id)
    except KeyError:
        stored = {"metrics": {}, "params": record.get("params") or {}, "artifacts": {}}
    metrics = dict(stored.get("metrics") or {})
    metrics["rooms_detected"] = result.rooms_detected
    artifacts = dict(stored.get("artifacts") or {})
    artifacts["floor_geometry_json"] = str(r_dir / "floor_geometry.json")
    artifacts["measurement_report_json"] = str(r_dir / "measurement_report.json")
    artifacts["measurement_report_pdf"] = str(r_dir / "measurement_report.pdf")
    if result.has_sheet:
        artifacts["sheet_svg"] = str(r_dir / "sheet.svg")
        artifacts["sheet_pdf"] = str(r_dir / "sheet.pdf")
    warnings = list(stored.get("warnings") or [])
    # Drop a prior rooms_not_closed warning — rooms are closed now.
    warnings = [w for w in warnings if w.get("code") != "rooms_not_closed"]
    warnings.extend(result.warnings)
    stored["metrics"] = metrics
    stored["artifacts"] = artifacts
    stored["warnings"] = warnings
    update_vectorize_result(job_id, stored)

    return GenerateMeasurementResponse(
        job_id=job_id,
        rooms_detected=result.rooms_detected,
        has_floor_geometry=result.has_floor_geometry,
        has_measurement_report=result.has_measurement_report,
        has_sheet=result.has_sheet,
        warnings=result.warnings,
        n_walls=result.n_walls,
        n_junctions=result.n_junctions,
        edit_version=edit_version,
    )


# ── GET /api/vectorize  (list) ───────────────────────────────────────────────

@router.get("")
async def list_vectorize(limit: int = 50):
    """Recent vectorize jobs, newest first.  Used by the history panel."""
    return {"jobs": list_vectorize_jobs(limit=limit)}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _enqueue_run(
    bg: BackgroundTasks, job_id: str, scan_path: Path, params: VectorizeParams,
) -> None:
    """Common queueing logic shared by create + reprocess."""
    a_dir = artifacts_dir(job_id)
    r_dir = results_dir(job_id)

    def _on_complete(result_payload: dict) -> None:
        try:
            update_vectorize_result(job_id, result_payload)
        except Exception:
            # The SSE event still fires; persistence failures are non-fatal.
            import traceback
            traceback.print_exc()

    def _on_error(err: str) -> None:
        try:
            update_vectorize_status(job_id, "failed", error=err)
        except Exception:
            pass

    # Drop any prior replay buffer so a re-process doesn't immediately replay
    # the last run's "complete" event to fresh subscribers.
    from ..sse import clear_history
    clear_history(job_id)

    update_vectorize_status(job_id, "processing")
    bg.add_task(
        run_vectorize_guarded,
        job_id, scan_path, a_dir, r_dir, params,
        _on_complete, _on_error,
    )


def _safe_filename(name: str) -> str:
    """Strip directory components and dangerous characters from an upload filename."""
    base = Path(name).name
    cleaned = "".join(c for c in base if c.isalnum() or c in "._-")
    return cleaned.replace("..", "_") or "scan.laz"
