"""Main pipeline runner: orchestrates all stages and emits SSE progress."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..models.job import JobStatus
from ..storage import (
    uploads_dir, artifacts_dir, results_dir,
    update_job_status, update_job_result,
)
from ..sse import publish
from .ingest import write_decimated_ply
from .merge import merge_scans
from .slicing import extract_wall_band
from .align import align_scan_to_plan
from .rooms import extract_rooms, extract_wall_lines, compute_match_quality
from .fixtures import detect_fixtures
from .confidence import compute_confidence_pct
from .export import write_aligned_json, write_aligned_dxf
from .ai_review import render_overlay_png


def _emit(job_id: str, stage: str, message: str, progress: float) -> None:
    publish(job_id, stage, message, progress)


def run_pipeline(
    job_id: str,
    scan_paths: list[Path],        # ← now a list (one or many scan files)
    plan_path: Path,
    band_low_m: float = 0.75,
    band_high_m: float = 1.80,
) -> None:
    """Full pipeline: ingest → merge → align → rooms → fixtures → export."""
    t0 = time.time()

    try:
        update_job_status(job_id, JobStatus.processing)
        artifact_dir = artifacts_dir(job_id)

        # ── 1. Load + merge scans ──────────────────────────────────────────────
        num = len(scan_paths)
        if num == 1:
            _emit(job_id, "ingest", "Reading scan…", 0.05)
        else:
            _emit(job_id, "ingest", f"Loading {num} scan files…", 0.05)

        def _merge_progress(msg: str, p: float):
            _emit(job_id, "ingest", msg, 0.05 + p * 0.15)

        merge_result = merge_scans(scan_paths, voxel_size=0.03, progress_cb=_merge_progress)
        scan_pcd = merge_result.merged

        _emit(
            job_id, "ingest",
            f"{'Merged' if num > 1 else 'Loaded'} {num} scan{'s' if num > 1 else ''}: "
            f"{merge_result.total_points_after:,} points "
            f"({'concatenated' if merge_result.strategy == 'concatenate' else 'ICP-registered'})",
            0.20,
        )

        # Write decimated PLY for browser viewer
        decimated_path = artifact_dir / "scan_decimated.ply"
        write_decimated_ply(scan_pcd, decimated_path, target_points=200_000)

        # ── 2. DXF parsing ────────────────────────────────────────────────────
        _emit(job_id, "ingest", "Parsing plan…", 0.22)
        plan_lines = extract_wall_lines(str(plan_path))
        if len(plan_lines) == 0:
            raise ValueError("No line entities found in the DXF plan file.")

        # ── 3. Alignment ──────────────────────────────────────────────────────
        _emit(job_id, "align", "Detecting floor plane…", 0.28)
        _emit(job_id, "align", "Extracting wall band…", 0.33)
        _emit(job_id, "align", "Running RANSAC plane detection…", 0.40)
        _emit(job_id, "align", "Computing principal axes…", 0.48)
        _emit(job_id, "align", "Aligning scan to plan…", 0.52)

        result = align_scan_to_plan(
            scan_pcd,
            plan_lines,
            band_low_m=band_low_m,
            band_high_m=band_high_m,
        )

        _emit(
            job_id, "align",
            f"Alignment complete — {compute_confidence_pct(result.confidence)}% confidence, "
            f"{result.residual_rmse * 1000:.1f}mm residual",
            0.62,
        )

        # ── 4. Room extraction ────────────────────────────────────────────────
        _emit(job_id, "rooms", "Extracting rooms…", 0.65)
        try:
            rooms = extract_rooms(str(plan_path))
        except Exception as e:
            rooms = []
            _emit(job_id, "rooms", f"Room extraction skipped: {e}", 0.68)

        # Per-room match quality
        aligned_pts = np.zeros((0, 3))
        if rooms and len(plan_lines) > 0:
            T = result.transformation
            wall_band = extract_wall_band(scan_pcd, result.floor, band_low_m, band_high_m)
            pts = np.asarray(wall_band.points)
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])
            aligned_pts = (T @ pts_h.T).T[:, :3]
            rooms = compute_match_quality(rooms, aligned_pts, plan_lines)

        _emit(job_id, "rooms", f"{len(rooms)} rooms extracted", 0.72)

        # ── 5. Fixture detection ──────────────────────────────────────────────
        _emit(job_id, "fixtures", "Detecting wall fixtures…", 0.75)
        try:
            all_pts = np.asarray(scan_pcd.points)
            all_pts_h = np.hstack([all_pts, np.ones((len(all_pts), 1))])
            aligned_all = (result.transformation @ all_pts_h.T).T[:, :3]
            fixtures = detect_fixtures(aligned_all, result.wall_planes)
        except Exception as e:
            fixtures = []
            _emit(job_id, "fixtures", f"Fixture detection skipped: {e}", 0.78)

        _emit(job_id, "fixtures", f"{len(fixtures)} wall fixtures detected", 0.82)

        # ── 6. Overlay PNG for AI review ──────────────────────────────────────
        overlay_path = str(artifact_dir / "overlay.png")
        try:
            scan_pts_2d = aligned_pts[:, :2] if len(aligned_pts) > 0 else np.zeros((0, 2))
            render_overlay_png(scan_pts_2d, plan_lines, overlay_path)
        except Exception:
            overlay_path = ""

        # ── 7. Export ─────────────────────────────────────────────────────────
        _emit(job_id, "export", "Writing results…", 0.88)
        result_dir = results_dir(job_id)

        conf_pct = compute_confidence_pct(result.confidence)
        result_payload = {
            "job_id": job_id,
            "alignment": {
                "transformation": result.transformation.tolist(),
                "residual_rmse": float(result.residual_rmse),
                "residual_rmse_mm": float(result.residual_rmse * 1000),
                "confidence": float(result.confidence),
                "confidence_pct": conf_pct,
                "inlier_ratio": float(result.inlier_ratio),
                "rotation_candidate_used": int(result.rotation_candidate_used),
                "iterations": int(result.iterations),
                "num_wall_planes": int(result.num_wall_planes),
            },
            "floor_z": float(result.floor.floor_z_estimate),
            "num_scans": num,
            "merge_strategy": merge_result.strategy,
            "rooms": [
                {
                    "id": r.id,
                    "label": r.label,
                    "category": r.category,
                    "polygon_2d": r.polygon_2d,
                    "centroid": list(r.centroid),
                    "area_m2": float(r.area_m2),
                    "match_quality": r.match_quality,
                }
                for r in rooms
            ],
            "fixtures": [
                {
                    "id": f.id,
                    "wall_id": f.wall_id,
                    "centroid": list(f.centroid),
                    "bbox_min": list(f.bbox_min),
                    "bbox_max": list(f.bbox_max),
                    "protrusion_depth_m": float(f.protrusion_depth_m),
                    "width_m": float(f.width_m),
                    "height_m": float(f.height_m),
                    "point_count": int(f.point_count),
                    "confidence": float(f.confidence),
                }
                for f in fixtures
            ],
            "plan_bounds": _compute_plan_bounds(plan_lines),
            "overlay_png": overlay_path,
        }

        json_path = result_dir / "aligned.json"
        write_aligned_json(result_payload, json_path)
        write_aligned_dxf(plan_path, result.transformation, result_dir / "aligned.dxf")

        elapsed = time.time() - t0
        update_job_result(job_id, result_payload, elapsed)

        _emit(
            job_id, "complete",
            f"Done in {elapsed:.1f}s — {conf_pct}% confidence · "
            f"{len(rooms)} rooms · {len(fixtures)} fixtures · {num} scan{'s' if num > 1 else ''} merged",
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
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    ]
