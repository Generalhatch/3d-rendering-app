"""Main pipeline runner: orchestrates all stages and emits SSE progress.

Two modes:
  - WITH plan (plan_path is not None): full alignment pipeline.
  - SCAN-ONLY (plan_path is None): merge scans, detect walls, generate a
    synthetic 2D floor outline from the wall planes, detect fixtures.
    Alignment is identity (scan is already in its own coordinate frame).
    ~10–15s faster than aligned mode.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np

from ..geometry.classify import COMMON_CATEGORIES
from ..limits import JOB_SEMAPHORE
from ..models.job import JobStatus
from ..storage import (
    uploads_dir, artifacts_dir, results_dir,
    load_result_json, update_job_status, update_job_result,
)
from ..sse import publish
from .ingest import write_decimated_ply
from .merge import merge_scans
from .slicing import detect_all_floors, detect_floor, extract_wall_band
from .segment import extract_wall_planes_with_fallback
from .align import align_scan_to_plan
from .rooms import extract_rooms, extract_wall_lines, compute_match_quality
from .scanplan import (
    generate_plan_from_walls, generate_plan_from_projection,
    generate_rooms_from_scan, generate_building_outline,
)
from .fixtures import detect_fixtures
from .confidence import compute_confidence_pct, compute_scan_only_confidence
from .export import write_aligned_json, write_aligned_dxf
from .ai_review import render_overlay_png


def _emit(job_id: str, stage: str, message: str, progress: float) -> None:
    publish(job_id, stage, message, progress)


def run_pipeline(
    job_id: str,
    scan_paths: list[Path],
    plan_path: Path | None,           # None = scan-only mode
    band_low_m: float = 1.00,
    band_high_m: float = 2.00,
) -> None:
    """Full pipeline with optional DXF plan."""
    t0 = time.time()
    scan_only = plan_path is None

    try:
        update_job_status(job_id, JobStatus.processing)
        artifact_dir = artifacts_dir(job_id)
        num = len(scan_paths)

        # ── 1. Load + merge scans ──────────────────────────────────────────────
        mode_label = "scan-only" if scan_only else "aligned"
        _emit(job_id, "ingest",
              f"Loading {num} scan{'s' if num > 1 else ''} ({mode_label} mode)…", 0.05)

        def _merge_progress(msg: str, p: float):
            _emit(job_id, "ingest", msg, 0.05 + p * 0.15)

        merge_result = merge_scans(scan_paths, voxel_size=0.03, progress_cb=_merge_progress)
        scan_pcd = merge_result.merged

        _emit(
            job_id, "ingest",
            f"{'Merged' if num > 1 else 'Loaded'}: {merge_result.total_points_after:,} pts "
            f"· originals preserved",
            0.20,
        )

        # Registration quality gates: scans that failed the ICP fitness/RMSE
        # gates are excluded from the merged cloud and flagged here — never
        # silently guessed into position.
        unplaced_scan_names = [r.scan_name for r in merge_result.unplaced]
        if unplaced_scan_names:
            _emit(
                job_id, "ingest",
                f"⚠ {len(unplaced_scan_names)} scan(s) could not be reliably "
                f"registered and are excluded: {', '.join(unplaced_scan_names)}",
                0.205,
            )

        # Emit Z-range so we can verify the coordinate frame looks sane
        _pts = np.asarray(scan_pcd.points)
        z_min, z_max = float(_pts[:, 2].min()), float(_pts[:, 2].max())
        y_min, y_max = float(_pts[:, 1].min()), float(_pts[:, 1].max())
        _emit(job_id, "ingest",
              f"Coord range — Z: {z_min:.2f}→{z_max:.2f}m  Y: {y_min:.2f}→{y_max:.2f}m",
              0.21)

        # ── 2. Floor + wall plane detection (needed in both modes) ─────────────
        _emit(job_id, "align", "Detecting floor plane…", 0.24)
        all_floors = detect_all_floors(scan_pcd)
        floor = all_floors[0]
        if len(all_floors) > 1:
            _emit(job_id, "align",
                  f"{len(all_floors)} floor levels detected "
                  f"(elevations: {', '.join(f'{f.floor_z_estimate:.2f}m' for f in all_floors)})",
                  0.26)
        wall_band = extract_wall_band(scan_pcd, floor, band_low_m, band_high_m)

        # Write decimated PLY for browser viewer — filter to wall-height band so
        # the viewer shows only meaningful structure (walls, doors, windows).
        # This removes floor reflections, ceiling, outdoor trees, and scan-sweep
        # artifacts that would otherwise clutter the top-down view.
        decimated_path = artifact_dir / "scan_decimated.ply"
        _viewer_pcd = _wall_height_slice(scan_pcd, floor.floor_z_estimate, floor.axis_idx)
        write_decimated_ply(_viewer_pcd, decimated_path, target_points=max(200_000, num * 5_000))
        del _viewer_pcd

        _emit(job_id, "align", "Detecting wall planes…", 0.30)
        wall_band_pts = len(wall_band.points)
        _emit(job_id, "align", f"Wall band: {wall_band_pts:,} pts between "
              f"{floor.floor_z_estimate + band_low_m:.1f}–{floor.floor_z_estimate + band_high_m:.1f} "
              f"(axis={'Z' if floor.axis_idx == 2 else 'Y'})", 0.32)

        wall_planes, detect_label = extract_wall_planes_with_fallback(
            wall_band,
            up_axis_idx=floor.axis_idx,
            full_cloud=scan_pcd,
            floor_z=floor.floor_z_estimate,
        )

        # ── Per-scan RANSAC fallback ───────────────────────────────────────────
        # Global RANSAC on a merged building-wide cloud typically returns 0 planes:
        # with 60+ rooms at all orientations, no single plane dominates enough to
        # exceed min_inliers.  Per-scan solves this: each file covers 1–2 rooms,
        # RANSAC finds 4–8 crisp vertical planes, and the pre-registered coordinate
        # frame means all planes drop directly into the shared floor plan.
        if not wall_planes and num > 1 and scan_only:
            _emit(job_id, "align",
                  f"Global RANSAC found 0 planes — switching to per-scan detection…", 0.34)
            wall_planes = _per_scan_chest_planes(
                scan_paths,
                floor,
                merge_result.centroid_offset,
                chest_low=band_low_m,
                chest_high=band_high_m,
                emit_cb=lambda msg: _emit(job_id, "align", msg, 0.35),
            )
            detect_label = f"per-scan ({len(wall_planes)} planes from {num} scans)"

        _emit(job_id, "align",
              f"{len(wall_planes)} wall planes detected ({detect_label})", 0.38)

        # ── 3a. WITH PLAN: full alignment ──────────────────────────────────────
        if not scan_only:
            _emit(job_id, "ingest", "Parsing plan…", 0.40)
            plan_lines = extract_wall_lines(str(plan_path))
            if len(plan_lines) == 0:
                raise ValueError("No line entities found in the DXF plan file.")

            _emit(job_id, "align", "Computing principal axes…", 0.44)
            _emit(job_id, "align", "Aligning scan to plan…", 0.50)

            result = align_scan_to_plan(
                scan_pcd, plan_lines,
                band_low_m=band_low_m, band_high_m=band_high_m,
            )
            transformation = result.transformation
            confidence     = result.confidence
            conf_pct       = compute_confidence_pct(confidence)
            residual_rmse  = result.residual_rmse
            num_wall_planes = result.num_wall_planes

            _emit(
                job_id, "align",
                f"Aligned — {conf_pct}% confidence · {residual_rmse * 1000:.1f}mm residual",
                0.60,
            )

            alignment_meta = {
                "transformation": transformation.tolist(),
                "residual_rmse": float(residual_rmse),
                "residual_rmse_mm": float(residual_rmse * 1000),
                "confidence": float(confidence),
                "confidence_pct": conf_pct,
                "inlier_ratio": float(result.inlier_ratio),
                "rotation_candidate_used": int(result.rotation_candidate_used),
                "iterations": int(result.iterations),
                "num_wall_planes": int(num_wall_planes),
                "mode": "aligned",
            }

        # ── 3b. SCAN-ONLY: generate synthetic plan, identity transform ─────────
        else:
            _emit(job_id, "align", "Generating floor plan from scan…", 0.42)

            if len(wall_planes) >= 4:
                # Enough 3D planes to describe the building — use them
                synthetic = generate_plan_from_walls(wall_planes, axis_idx=floor.axis_idx)
                _emit(job_id, "align",
                      f"Floor plan generated — {synthetic.wall_count} wall segments (3D planes)",
                      0.60)
            else:
                # Too few 3D planes — 2D projection gives better coverage
                if wall_planes:
                    _emit(job_id, "align",
                          f"Only {len(wall_planes)} 3D plane(s) — using 2D projection for full coverage…",
                          0.44)
                else:
                    _emit(job_id, "align",
                          "No 3D wall planes — projecting to 2D for floor plan…", 0.44)
                synthetic = generate_plan_from_projection(scan_pcd, floor)
                _emit(job_id, "align",
                      f"Floor plan generated — {synthetic.wall_count} wall segments (2D projection)",
                      0.60)

            plan_lines = synthetic.segments
            transformation = np.eye(4)   # identity — scan IS the plan
            residual_rmse  = 0.0
            num_wall_planes = len(wall_planes)
            scan_plan_source = (
                "3d_planes" if len(wall_planes) >= 4 else "2d_projection"
            )

            # confidence / confidence_pct are filled in AFTER room extraction
            # (compute_scan_only_confidence needs the rooms) — see step 4.
            alignment_meta = {
                "transformation": transformation.tolist(),
                "residual_rmse": 0.0,
                "residual_rmse_mm": 0.0,
                "confidence": None,
                "confidence_pct": None,
                "inlier_ratio": 1.0,
                "rotation_candidate_used": 0,
                "iterations": 0,
                "num_wall_planes": int(num_wall_planes),
                "plan_source": scan_plan_source,
                "mode": "scan_only",
            }

        # ── 4. Room extraction ────────────────────────────────────────────────
        _emit(job_id, "rooms", "Extracting rooms…", 0.64)
        rooms_raw: list[dict] = []
        aligned_pts = np.zeros((0, 3))

        if not scan_only:
            try:
                room_objs = extract_rooms(str(plan_path))
                T = transformation
                pts = np.asarray(wall_band.points)
                pts_h = np.hstack([pts, np.ones((len(pts), 1))])
                aligned_pts = (T @ pts_h.T).T[:, :3]
                room_objs = compute_match_quality(room_objs, aligned_pts, plan_lines)
                rooms_raw = [
                    {
                        "id": r.id, "label": r.label, "category": r.category,
                        "polygon_2d": r.polygon_2d, "centroid": list(r.centroid),
                        "area_m2": float(r.area_m2), "match_quality": r.match_quality,
                        # DXF labels are a trusted category source → full
                        # common-area mapping (corridor, lobby, restroom).
                        "is_common": r.category in COMMON_CATEGORIES,
                    }
                    for r in room_objs
                ]
            except Exception as e:
                _emit(job_id, "rooms", f"Room extraction skipped: {e}", 0.68)
        else:
            # Scan-only: reconstruct rooms from the synthetic plan
            aligned_pts = np.asarray(wall_band.points)   # already in scan frame
            rooms_raw = generate_rooms_from_scan(wall_planes, synthetic, scan_pcd=scan_pcd, floor=floor)  # type: ignore[name-defined]

            # Now we have everything the scan-only confidence model needs.
            # No plan means no ground truth — never claim 100%.
            confidence = compute_scan_only_confidence(
                num_wall_planes=num_wall_planes,
                plan_source=scan_plan_source,
                merge_strategy=merge_result.strategy,
                rooms=rooms_raw,
            )
            conf_pct = compute_confidence_pct(confidence)
            alignment_meta["confidence"] = float(confidence)
            alignment_meta["confidence_pct"] = conf_pct
            _emit(job_id, "rooms",
                  f"Scan-only confidence: {conf_pct}% "
                  f"({num_wall_planes} planes · {merge_result.strategy})", 0.70)

        _emit(job_id, "rooms", f"{len(rooms_raw)} rooms extracted", 0.72)

        # ── 5. Fixture detection ──────────────────────────────────────────────
        _emit(job_id, "fixtures", "Detecting wall fixtures…", 0.75)
        fixtures_raw: list[dict] = []
        try:
            all_pts = np.asarray(scan_pcd.points)
            all_pts_h = np.hstack([all_pts, np.ones((len(all_pts), 1))])
            aligned_all = (transformation @ all_pts_h.T).T[:, :3]
            fixtures = detect_fixtures(aligned_all, wall_planes)
            fixtures_raw = [
                {
                    "id": f.id, "wall_id": f.wall_id,
                    "centroid": list(f.centroid),
                    "bbox_min": list(f.bbox_min), "bbox_max": list(f.bbox_max),
                    "protrusion_depth_m": float(f.protrusion_depth_m),
                    "width_m": float(f.width_m), "height_m": float(f.height_m),
                    "point_count": int(f.point_count), "confidence": float(f.confidence),
                }
                for f in fixtures
            ]
        except Exception as e:
            _emit(job_id, "fixtures", f"Fixture detection skipped: {e}", 0.78)

        _emit(job_id, "fixtures", f"{len(fixtures_raw)} wall fixtures detected", 0.82)

        # ── 6. Building outline (alphashape) ─────────────────────────────────
        _emit(job_id, "export", "Computing building outline…", 0.84)
        building_outline: list[list[float]] = []
        try:
            building_outline = generate_building_outline(scan_pcd, floor)
            _emit(job_id, "export",
                  f"Building outline: {len(building_outline)} vertices", 0.86)
        except Exception as _e:
            _emit(job_id, "export", f"Building outline skipped: {_e}", 0.86)

        # ── 7. Overlay PNG ────────────────────────────────────────────────────
        overlay_path = str(artifact_dir / "overlay.png")
        try:
            scan_pts_2d = (aligned_pts[:, :2] if len(aligned_pts) > 0
                           else np.zeros((0, 2)))
            render_overlay_png(scan_pts_2d, plan_lines, overlay_path)
        except Exception:
            overlay_path = ""

        # ── 8. Export ─────────────────────────────────────────────────────────
        _emit(job_id, "export", "Writing results…", 0.88)
        result_dir = results_dir(job_id)

        # Serialise all detected floor levels for the multi-floor UI (MJ2).
        # wall_score is the count of points 0.3–3.0 m above the floor — used to
        # rank floors and let the user see which level has the most wall coverage.
        _scan_pts = np.asarray(scan_pcd.points)
        floor_candidates_data = [
            {
                "floor_z": float(f.floor_z_estimate),
                "inlier_count": int(f.inlier_count),
                "wall_score": int(
                    ((_scan_pts[:, f.axis_idx] > f.floor_z_estimate + 0.30)
                     & (_scan_pts[:, f.axis_idx] < f.floor_z_estimate + 3.00)).sum()
                ),
                "axis_idx": int(f.axis_idx),
            }
            for f in all_floors
        ]

        result_payload = {
            "job_id": job_id,
            "alignment": alignment_meta,
            "floor_z": float(floor.floor_z_estimate),
            "floor_axis_idx": int(floor.axis_idx),
            # MJ2: all detected floor levels sorted by wall-content score
            "floor_candidates": floor_candidates_data,
            "scan_only": scan_only,
            "num_scans": num,
            "merge_strategy": merge_result.strategy,
            # Per-scan registration outcomes (Phase 2 quality gates).  Scans
            # with placed=False failed the ICP fitness/RMSE gate and are NOT
            # part of the merged geometry — surfaced so the operator can
            # re-scan or manually place them instead of trusting a guess.
            "scan_registrations": [
                r.to_json_dict() for r in merge_result.registrations
            ],
            "unplaced_scans": unplaced_scan_names,
            "rooms": rooms_raw,
            # Phase 2: common areas as a first-class category.  Each room dict
            # carries is_common; this summary feeds the UI and reports.
            "common_areas": {
                "room_ids": [
                    r["id"] for r in rooms_raw if r.get("is_common")
                ],
                "total_m2": round(sum(
                    float(r.get("area_m2", 0.0))
                    for r in rooms_raw if r.get("is_common")
                ), 3),
            },
            "fixtures": fixtures_raw,
            "plan_bounds": _compute_plan_bounds(plan_lines),
            # Store the actual wall segments so the viewer can render them
            "plan_segments": _segments_to_list(plan_lines),
            "overlay_png": overlay_path,
            # Building exterior hull (alphashape concave polygon) — [[x, y], ...]
            "building_outline": building_outline,
            # Path to the merged+decimated PLY — used by the manual-transform
            # endpoint so multi-scan jobs re-align against the full merged cloud,
            # not just the first raw scan file.
            "merged_ply": str(decimated_path),
        }

        # ── 8b. Canonical floor geometry (Phase 1) + closure validation (Phase 2)
        # Emit the standard-agnostic FloorGeometry so the measurement engine
        # (app.measurement) can compute BOMA/REBNY/Gross areas from this job
        # without re-running the pipeline, then run the floor-closure sanity
        # checks (envelope vs rooms, perimeter closure, unscanned gaps).
        if rooms_raw:
            try:
                import json as _json

                from ..geometry.adapters import floor_geometry_from_alignment_rooms
                floor_geo = floor_geometry_from_alignment_rooms(
                    rooms_raw, floor_id=job_id,
                    envelope_xy=(
                        np.asarray(building_outline, dtype=float)
                        if building_outline else None
                    ),
                )
                floor_geo_path = result_dir / "floor_geometry.json"
                floor_geo_path.write_text(
                    _json.dumps(floor_geo.to_json_dict(), indent=2)
                )
                result_payload["floor_geometry_json"] = str(floor_geo_path)

                from ..geometry.validation import validate_floor_closure
                validation = validate_floor_closure(floor_geo)
                result_payload["floor_validation"] = validation.to_json_dict()
                (result_dir / "floor_validation.json").write_text(
                    _json.dumps(result_payload["floor_validation"], indent=2)
                )
                if not validation.passed:
                    n_gaps = len(validation.unscanned_gaps)
                    gap_m2 = sum(g.area_m2 for g in validation.unscanned_gaps)
                    _emit(
                        job_id, "export",
                        f"⚠ Floor-closure checks failed "
                        f"({len(validation.warnings)} warning(s)"
                        + (f", {n_gaps} unscanned gap(s) totalling "
                           f"{gap_m2:.1f} m²" if n_gaps else "")
                        + ") — see floor_validation.json",
                        0.885,
                    )
            except Exception as fg_err:
                _emit(job_id, "export",
                      f"Floor geometry export skipped: {fg_err}", 0.885)

            # ── 8c. Floor plan sheet (Phase 3): SVG + PDF from the
            # canonical FloorGeometry.  Guarded — never fails the job.
            try:
                if "floor_geometry_json" not in result_payload:
                    raise RuntimeError("no canonical floor geometry emitted")
                from ..sheet.service import render_job_sheet
                sheet_render = render_job_sheet(result_dir, floor=floor_geo)
                result_payload["sheet_svg"] = str(result_dir / "sheet.svg")
                result_payload["sheet_pdf"] = str(result_dir / "sheet.pdf")
                _emit(
                    job_id, "export",
                    f"Floor plan sheet rendered at "
                    f"1:{sheet_render.scale_denominator}",
                    0.887,
                )
            except Exception as sheet_err:
                _emit(job_id, "export",
                      f"Sheet rendering skipped: {sheet_err}", 0.887)

        write_aligned_json(result_payload, result_dir / "aligned.json")
        if not scan_only and plan_path is not None:
            write_aligned_dxf(plan_path, transformation, result_dir / "aligned.dxf")

        elapsed = time.time() - t0
        update_job_result(job_id, result_payload, elapsed)

        suffix = (
            f"· {conf_pct}% confidence (scan-only)" if scan_only  # type: ignore[possibly-undefined]
            else f"· {conf_pct}% confidence"  # type: ignore[possibly-undefined]
        )
        _emit(
            job_id, "complete",
            f"Done in {elapsed:.1f}s {suffix} · "
            f"{len(rooms_raw)} rooms · {len(fixtures_raw)} fixtures · "
            f"{num} scan{'s' if num > 1 else ''} merged",
            1.0,
        )

    except Exception as exc:
        import traceback
        err = str(exc)
        traceback.print_exc()
        update_job_status(job_id, JobStatus.failed, error=err)
        _emit(job_id, "error", err, 1.0)


def _compute_plan_bounds(plan_lines: np.ndarray) -> list[float]:
    if len(plan_lines) == 0:
        return [0.0, 0.0, 1.0, 1.0]
    pts = plan_lines.reshape(-1, 2)
    return [
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    ]


def _segments_to_list(plan_lines: np.ndarray) -> list:
    """Convert (N,2,2) or (N,4) segment array to JSON-serialisable list of [[x1,y1],[x2,y2]]."""
    if len(plan_lines) == 0:
        return []
    arr = np.asarray(plan_lines)
    # Normalise to (N, 2, 2)
    if arr.ndim == 2 and arr.shape[1] == 4:
        arr = arr.reshape(-1, 2, 2)
    elif arr.ndim == 3 and arr.shape[1] == 2 and arr.shape[2] == 2:
        pass  # already (N, 2, 2)
    else:
        # Fallback: reshape as pairs of 2D points
        flat = arr.reshape(-1, 2)
        n_seg = len(flat) // 2
        arr = flat[: n_seg * 2].reshape(n_seg, 2, 2)
    return arr.tolist()


def _per_scan_chest_planes(
    scan_paths: list[Path],
    floor: "FloorReference",
    centroid_offset: "np.ndarray",
    chest_low: float = 1.00,
    chest_high: float = 2.00,
    emit_cb: "Callable[[str], None] | None" = None,
) -> "list[WallPlane]":
    """Run RANSAC wall-plane detection on each scan's chest-height band.

    Why this works when global RANSAC fails
    ----------------------------------------
    Global RANSAC on the merged 115m-wide cloud sees hundreds of wall planes at
    all orientations.  Open3D's segment_plane finds only the single most-dominant
    plane per call; after 40 calls it gives up, often returning 0 usable planes.

    Per-scan: each file covers 1–2 rooms → 4–8 cleanly-vertical planes.
    RANSAC trivially finds all of them on the first few iterations.  The result
    is sub-centimeter wall positions limited only by scanner hardware precision,
    not algorithm noise.

    Coordinate frame
    ----------------
    Raw scans are in the pre-registered shared frame.  ``centroid_offset`` is
    the 3D mean that ``_center_cloud`` subtracted during the merge step; applying
    the same subtraction here puts per-scan points in the same centered frame as
    ``floor.floor_z_estimate``.
    """
    import open3d as o3d
    from .ingest import load_point_cloud
    from .segment import extract_wall_planes_with_fallback, WallPlane

    fl_z = floor.floor_z_estimate
    ax   = floor.axis_idx
    all_planes: list[WallPlane] = []

    for i, path in enumerate(scan_paths):
        try:
            if emit_cb:
                emit_cb(f"Per-scan wall detection {i + 1}/{len(scan_paths)}: {path.name}…")
            pcd = load_point_cloud(path)
            pts_raw = np.asarray(pcd.points, dtype=np.float64)
            del pcd

            # Apply the same centering offset used during merge_scans so that
            # fl_z (in the centered frame) can be used directly.
            pts = pts_raw - centroid_offset

            mask = (pts[:, ax] > fl_z + chest_low) & (pts[:, ax] < fl_z + chest_high)
            pts_band = pts[mask]
            if len(pts_band) < 50:
                continue

            band_pcd = o3d.geometry.PointCloud()
            band_pcd.points = o3d.utility.Vector3dVector(pts_band)

            planes, _ = extract_wall_planes_with_fallback(
                band_pcd, up_axis_idx=ax, floor_z=fl_z,
            )
            for p in planes:
                p.id = f"s{i:02d}-{p.id}"
            all_planes.extend(planes)
        except Exception:
            continue

    return all_planes


def _wall_height_slice(
    pcd: object,
    floor_z: float,
    axis_idx: int,
    low_offset: float = 0.20,
    high_offset: float = 2.50,
) -> object:
    """Return only points in the wall-height band for the browser viewer.

    Filters to [floor_z + low_offset, floor_z + high_offset] along the
    vertical axis.  This removes:
      - Floor reflections / laser floor returns
      - Ceiling scans
      - Outdoor trees and sky captured through windows
      - Furniture tops above head height
    leaving mostly wall surfaces, doors, window frames — exactly what you
    need to understand the building layout from above.
    """
    import open3d as o3d
    pts = np.asarray(pcd.points)
    low  = floor_z + low_offset
    high = floor_z + high_offset
    mask = (pts[:, axis_idx] > low) & (pts[:, axis_idx] < high)
    indices = np.where(mask)[0]
    if len(indices) < 1000:
        # Not enough points in band — fall back to full cloud so we show something
        return pcd
    sliced = pcd.select_by_index(indices.tolist())
    return sliced


async def run_pipeline_guarded(
    job_id: str,
    scan_paths: list[Path],
    plan_path: Path | None,
    band_low_m: float = 1.00,
    band_high_m: float = 2.00,
) -> None:
    """Semaphore-guarded async wrapper: at most 2 pipeline runs at once.

    Jobs that arrive while both slots are occupied queue here and run as
    soon as a slot is released.  The job stays in 'queued' status until
    the semaphore is acquired, then transitions to 'processing' inside
    run_pipeline().
    """
    async with JOB_SEMAPHORE:
        await asyncio.to_thread(
            run_pipeline, job_id, scan_paths, plan_path, band_low_m, band_high_m
        )


# ── MJ2 / MJ3: Live Re-processing ────────────────────────────────────────────

def reprocess_rooms(
    job_id: str,
    merged_ply_path: Path,
    plan_path: Path | None,
    floor_z_override: float | None = None,
    band_low_m: float = 1.00,
    band_high_m: float = 2.00,
    min_wall_length_m: float = 2.0,
    hough_threshold: int = 35,
) -> None:
    """Re-run room/wall extraction using the already-merged scan.

    Skips the expensive scan-loading and merge stages (typically 10–15 min)
    and re-runs only:
      floor detection → wall band → wall planes → plan generation → rooms
      → fixtures → building outline → update stored result

    Parameters
    ----------
    job_id : str
        Job to update.  SSE progress events are published on this channel.
    merged_ply_path : Path
        Path to the decimated merged PLY written during the original pipeline run.
    plan_path : Path | None
        DXF plan file (aligned mode) or None (scan-only mode).
    floor_z_override : float | None
        If set, bypasses floor detection and uses this elevation directly.
        Use to switch to a different building level (MJ2 multi-floor).
    band_low_m / band_high_m : float
        Wall-band height offsets above the floor (MJ3 parameter tuning).
    min_wall_length_m : float
        Minimum wall segment length for Hough-line filtering (MJ3).
    hough_threshold : int
        Minimum Hough vote count (MJ3).
    """
    from .ingest import load_point_cloud
    from .slicing import FloorReference

    t0 = time.time()

    try:
        _emit(job_id, "reprocess", "Loading merged point cloud for re-processing…", 0.05)
        scan_pcd = load_point_cloud(merged_ply_path)

        # ── Floor detection (or use provided override) ─────────────────────────
        if floor_z_override is not None:
            # Build a minimal FloorReference from the user-chosen floor elevation.
            # Axis index is inferred from the stored result if available.
            try:
                stored = load_result_json(job_id)
                ax = int(stored.get("floor_axis_idx", 2))
            except Exception:
                ax = 2
            up = np.zeros(3)
            up[ax] = 1.0
            floor = FloorReference(
                plane_eq=np.array([up[0], up[1], up[2], -floor_z_override]),
                up_normal=up,
                floor_z_estimate=floor_z_override,
                inlier_count=0,
                axis_idx=ax,
            )
            _emit(job_id, "reprocess",
                  f"Using selected floor at z={floor_z_override:.2f} m", 0.10)
        else:
            _emit(job_id, "reprocess", "Re-detecting floor plane…", 0.10)
            floor = detect_floor(scan_pcd)
            _emit(job_id, "reprocess",
                  f"Floor detected at z={floor.floor_z_estimate:.2f} m", 0.14)

        # ── Wall band + wall planes ────────────────────────────────────────────
        _emit(job_id, "reprocess", "Extracting wall band…", 0.18)
        wall_band = extract_wall_band(scan_pcd, floor, band_low_m, band_high_m)

        _emit(job_id, "reprocess", "Detecting wall planes…", 0.26)
        wall_planes, detect_label = extract_wall_planes_with_fallback(
            wall_band,
            up_axis_idx=floor.axis_idx,
            full_cloud=scan_pcd,
            floor_z=floor.floor_z_estimate,
        )
        _emit(job_id, "reprocess",
              f"{len(wall_planes)} wall planes ({detect_label})", 0.38)

        # ── Plan generation + room extraction ─────────────────────────────────
        scan_only = plan_path is None
        if scan_only:
            _emit(job_id, "reprocess", "Generating floor plan from scan…", 0.42)
            if len(wall_planes) >= 4:
                synthetic = generate_plan_from_walls(wall_planes, axis_idx=floor.axis_idx)
            else:
                synthetic = generate_plan_from_projection(
                    scan_pcd, floor,
                    min_wall_length_m=min_wall_length_m,
                )
            plan_lines = synthetic.segments
            _emit(job_id, "reprocess",
                  f"Plan: {synthetic.wall_count} wall segments", 0.55)

            _emit(job_id, "reprocess", "Extracting rooms…", 0.60)
            rooms_raw = generate_rooms_from_scan(
                wall_planes, synthetic, scan_pcd=scan_pcd, floor=floor
            )
            aligned_pts = np.asarray(wall_band.points)
            transformation = np.eye(4)

        else:
            _emit(job_id, "reprocess", "Aligning to plan…", 0.42)
            plan_lines_dxf = extract_wall_lines(str(plan_path))
            result_align = align_scan_to_plan(
                scan_pcd, plan_lines_dxf,
                band_low_m=band_low_m, band_high_m=band_high_m,
            )
            transformation = result_align.transformation
            plan_lines = plan_lines_dxf
            conf_pct = compute_confidence_pct(result_align.confidence)
            _emit(job_id, "reprocess",
                  f"Re-aligned: {conf_pct}% confidence", 0.56)

            _emit(job_id, "reprocess", "Extracting rooms…", 0.62)
            room_objs = extract_rooms(str(plan_path))
            T = transformation
            pts = np.asarray(wall_band.points)
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])
            aligned_pts = (T @ pts_h.T).T[:, :3]
            room_objs = compute_match_quality(room_objs, aligned_pts, plan_lines)
            rooms_raw = [
                {
                    "id": r.id, "label": r.label, "category": r.category,
                    "polygon_2d": r.polygon_2d, "centroid": list(r.centroid),
                    "area_m2": float(r.area_m2), "match_quality": r.match_quality,
                    "is_common": r.category in COMMON_CATEGORIES,
                }
                for r in room_objs
            ]

        _emit(job_id, "reprocess", f"{len(rooms_raw)} rooms extracted", 0.68)

        # ── Fixtures ──────────────────────────────────────────────────────────
        _emit(job_id, "reprocess", "Detecting fixtures…", 0.72)
        fixtures_raw: list[dict] = []
        try:
            all_pts = np.asarray(scan_pcd.points)
            all_pts_h = np.hstack([all_pts, np.ones((len(all_pts), 1))])
            aligned_all = (transformation @ all_pts_h.T).T[:, :3]
            fixtures_list = detect_fixtures(aligned_all, wall_planes)
            fixtures_raw = [
                {
                    "id": f.id, "wall_id": f.wall_id,
                    "centroid": list(f.centroid),
                    "bbox_min": list(f.bbox_min), "bbox_max": list(f.bbox_max),
                    "protrusion_depth_m": float(f.protrusion_depth_m),
                    "width_m": float(f.width_m), "height_m": float(f.height_m),
                    "point_count": int(f.point_count), "confidence": float(f.confidence),
                }
                for f in fixtures_list
            ]
        except Exception as e:
            _emit(job_id, "reprocess", f"Fixture detection skipped: {e}", 0.76)

        # ── Building outline ──────────────────────────────────────────────────
        _emit(job_id, "reprocess", "Computing building outline…", 0.80)
        building_outline: list[list[float]] = []
        try:
            building_outline = generate_building_outline(scan_pcd, floor)
        except Exception:
            pass

        # ── Overlay PNG ───────────────────────────────────────────────────────
        artifact_dir = artifacts_dir(job_id)
        overlay_path = str(artifact_dir / "overlay.png")
        try:
            scan_pts_2d = aligned_pts[:, :2] if len(aligned_pts) > 0 else np.zeros((0, 2))
            render_overlay_png(scan_pts_2d, plan_lines, overlay_path)
        except Exception:
            overlay_path = ""

        # ── Merge into stored result ──────────────────────────────────────────
        _emit(job_id, "reprocess", "Saving results…", 0.90)
        try:
            stored = load_result_json(job_id)
        except Exception:
            stored = {}

        stored["rooms"] = rooms_raw
        stored["common_areas"] = {
            "room_ids": [r["id"] for r in rooms_raw if r.get("is_common")],
            "total_m2": round(sum(
                float(r.get("area_m2", 0.0))
                for r in rooms_raw if r.get("is_common")
            ), 3),
        }
        stored["fixtures"] = fixtures_raw
        stored["plan_bounds"] = _compute_plan_bounds(plan_lines)
        stored["plan_segments"] = _segments_to_list(plan_lines)
        stored["building_outline"] = building_outline
        stored["floor_z"] = float(floor.floor_z_estimate)
        stored["floor_axis_idx"] = int(floor.axis_idx)
        if overlay_path:
            stored["overlay_png"] = overlay_path

        if scan_only:
            # Recompute the honest scan-only confidence from the fresh rooms.
            conf = compute_scan_only_confidence(
                num_wall_planes=len(wall_planes),
                plan_source="3d_planes" if len(wall_planes) >= 4 else "2d_projection",
                merge_strategy=str(stored.get("merge_strategy", "single")),
                rooms=rooms_raw,
            )
            stored.setdefault("alignment", {})
            stored["alignment"]["confidence"] = float(conf)
            stored["alignment"]["confidence_pct"] = compute_confidence_pct(conf)
        else:
            stored["alignment"]["transformation"] = transformation.tolist()
            stored["alignment"]["residual_rmse"] = float(result_align.residual_rmse)  # type: ignore[name-defined]
            stored["alignment"]["residual_rmse_mm"] = float(result_align.residual_rmse * 1000)  # type: ignore[name-defined]
            stored["alignment"]["confidence"] = float(result_align.confidence)         # type: ignore[name-defined]
            stored["alignment"]["confidence_pct"] = int(compute_confidence_pct(result_align.confidence))  # type: ignore[name-defined]

        elapsed = time.time() - t0
        update_job_result(job_id, stored, elapsed)

        _emit(
            job_id, "complete",
            f"Re-processing done in {elapsed:.1f}s · "
            f"{len(rooms_raw)} rooms · {len(fixtures_raw)} fixtures",
            1.0,
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _emit(job_id, "error", f"Re-processing failed: {exc}", 1.0)


async def reprocess_rooms_guarded(
    job_id: str,
    merged_ply_path: Path,
    plan_path: Path | None,
    floor_z_override: float | None = None,
    band_low_m: float = 1.00,
    band_high_m: float = 2.00,
    min_wall_length_m: float = 2.0,
    hough_threshold: int = 35,
) -> None:
    """Semaphore-guarded wrapper for reprocess_rooms."""
    async with JOB_SEMAPHORE:
        await asyncio.to_thread(
            reprocess_rooms,
            job_id, merged_ply_path, plan_path,
            floor_z_override, band_low_m, band_high_m,
            min_wall_length_m, hough_threshold,
        )
