"""End-to-end vectorize pipeline orchestrator.

Stages
------
  1. ingest       — load the scan into Open3D
  2. floor detect — pick auto elevation if not supplied
  3. slice        — project the slab to a 2D raster
  4. preprocess   — morphology + smoothing on the raster
  5. detect       — classical line detection (Hough / FLD / both)
  6. regularize   — drop short, snap to Manhattan, merge collinear
  7. overlay      — write detection overlay PNG for visual review
  8. dxf          — write the layered DXF deliverable
  9. persist      — write the result JSON sidecar

Emits SSE progress at every stage via :func:`app.sse.publish`.
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ..limits import JOB_SEMAPHORE
from ..models.vectorize_job import VectorizeMetrics, VectorizeParams, VectorizeStatus
from ..sse import publish
from . import classical, dxf_writer, preprocess, regularize, slicer


def _emit(job_id: str, stage: str, message: str, progress: float) -> None:
    publish(job_id, stage, message, progress)


def run_vectorize(
    job_id: str,
    scan_path: Path,
    artifact_dir: Path,
    result_dir: Path,
    params: VectorizeParams,
    on_complete=None,           # callable: (metrics_dict) -> None  (state persistence hook)
    on_error=None,              # callable: (error_message) -> None
) -> None:
    """Run the full vectorize pipeline for one job, blocking.

    Designed to be called from a worker thread (see :func:`run_vectorize_guarded`).
    All exceptions are caught and reported through ``on_error`` + SSE — the
    function never re-raises.
    """
    t0 = time.time()
    try:
        # ── 1. Load point cloud ──────────────────────────────────────────
        _emit(job_id, "ingest", f"Loading scan ({scan_path.name})…", 0.05)
        from ..pipeline.ingest import load_point_cloud   # local import: heavy module
        pcd = load_point_cloud(scan_path)
        raw_points = int(len(pcd.points))
        _emit(job_id, "ingest", f"Loaded {raw_points:,} points", 0.15)

        # ── 2. Decide elevation ──────────────────────────────────────────
        if params.elevation_m is None:
            _emit(job_id, "floor", "Detecting floor plane…", 0.20)
            from ..pipeline.slicing import detect_floor
            floor = detect_floor(pcd)
            chest_offset = 1.40
            elevation = float(floor.floor_z_estimate + chest_offset)
            axis_idx = int(floor.axis_idx)
            _emit(
                job_id, "floor",
                f"Floor at {floor.floor_z_estimate:.2f} m (axis="
                f"{'Z' if axis_idx == 2 else 'Y'}-up) — slicing at "
                f"{elevation:.2f} m (floor + {chest_offset:.2f} m)",
                0.25,
            )
        else:
            elevation = float(params.elevation_m)
            axis_idx = 2  # Default to Z-up when explicit elevation supplied.
            _emit(
                job_id, "floor",
                f"Slicing at user-supplied elevation {elevation:.2f} m",
                0.25,
            )

        # ── 3. Slice ─────────────────────────────────────────────────────
        _emit(job_id, "slice", "Projecting slab to raster…", 0.30)
        slice_result = slicer.slice_to_raster(
            pcd,
            elevation_m=elevation,
            slab_thickness_m=params.slab_thickness_m,
            resolution_m_per_px=params.resolution_m_per_px,
            axis_idx=axis_idx,
        )
        # Free the point cloud — slicer is the last consumer.
        del pcd

        slicer.save_slice(slice_result, artifact_dir, basename="slice")
        _emit(
            job_id, "slice",
            f"Raster: {slice_result.affine.width_px}×{slice_result.affine.height_px} px "
            f"({slice_result.n_points_in_slab:,} pts in slab)",
            0.40,
        )

        # ── 4. Preprocess ────────────────────────────────────────────────
        _emit(job_id, "preprocess", "Cleaning raster…", 0.45)
        cleaned = preprocess.preprocess(slice_result.image)

        # Persist the cleaned raster too — useful for the operator to see what
        # the detector actually saw.
        cleaned_display = cv2.flip(cleaned, 0)
        cv2.imwrite(str(artifact_dir / "slice_cleaned.png"), cleaned_display)

        # ── 5. Detect line segments ──────────────────────────────────────
        _emit(job_id, "detect", f"Detecting lines (detector={params.detector.value})…", 0.55)

        min_len_px = max(
            3, int(round(params.min_wall_length_m / params.resolution_m_per_px))
        )
        hough_params = classical.HoughParams(min_line_length_px=min_len_px)
        fld_params = classical.FldParams(length_threshold=min_len_px)

        if params.detector.value == "hough":
            seg_px = classical.detect_hough(cleaned, hough_params)
        elif params.detector.value == "fld":
            seg_px = classical.detect_fld(cleaned, fld_params)
            if len(seg_px) == 0:
                # ximgproc missing or FLD found nothing — fall back to Hough.
                _emit(job_id, "detect",
                      "FLD returned no segments — falling back to Hough", 0.58)
                seg_px = classical.detect_hough(cleaned, hough_params)
        else:  # "both"
            seg_px = classical.merge_detectors(
                classical.detect_hough(cleaned, hough_params),
                classical.detect_fld(cleaned, fld_params),
            )

        segments_detected = int(len(seg_px))
        _emit(job_id, "detect", f"Detected {segments_detected} raw segments", 0.65)

        # ── 6. Pixel → world, then regularize ───────────────────────────
        _emit(job_id, "regularize", "Snapping + merging segments…", 0.72)
        segments_world = classical.segments_pixels_to_world(seg_px, slice_result.affine)

        reg_params = regularize.RegularizeParams(
            drop_short_below_m=params.min_wall_length_m,
            manhattan_snap=params.manhattan_snap,
            merge_collinear=params.merge_collinear,
        )
        clean_segments = regularize.regularize(segments_world, reg_params)
        segments_after = int(len(clean_segments))
        _emit(
            job_id, "regularize",
            f"{segments_after} clean segments (from {segments_detected})",
            0.80,
        )

        # ── 7. Overlay PNG ───────────────────────────────────────────────
        _emit(job_id, "overlay", "Rendering review overlay…", 0.85)
        # Re-project clean segments back to pixel space so the overlay matches
        # what the operator will see in the DXF (not the raw detector output).
        clean_px = _segments_world_to_pixels(clean_segments, slice_result.affine)
        overlay = classical.render_overlay(cleaned, clean_px, color_bgr=(0, 0, 255), thickness=2)
        cv2.imwrite(str(artifact_dir / "overlay.png"), cv2.flip(overlay, 0))

        # Also save the raw (pre-regularize) detection overlay for comparison.
        raw_overlay = classical.render_overlay(cleaned, seg_px, color_bgr=(0, 165, 255), thickness=1)
        cv2.imwrite(str(artifact_dir / "overlay_raw.png"), cv2.flip(raw_overlay, 0))

        # ── 8. DXF ───────────────────────────────────────────────────────
        _emit(job_id, "dxf", "Writing DXF…", 0.92)
        annotation = (
            f"Stevenson Vectorize | job={job_id} | "
            f"elev={elevation:.2f}m | slab={params.slab_thickness_m:.2f}m | "
            f"res={params.resolution_m_per_px:.3f}m/px | "
            f"detector={params.detector.value} | "
            f"segments_raw={segments_detected} segments_clean={segments_after}"
        )
        dxf_path = result_dir / "vectorized.dxf"
        dxf_writer.write_walls_dxf(clean_segments, dxf_path, annotation_text=annotation)

        # ── 9. Persist metrics ──────────────────────────────────────────
        elapsed = time.time() - t0
        world_w = slice_result.affine.width_px * slice_result.affine.resolution_m_per_px
        world_h = slice_result.affine.height_px * slice_result.affine.resolution_m_per_px

        metrics = VectorizeMetrics(
            elapsed_s=float(elapsed),
            raw_points=raw_points,
            points_in_slab=slice_result.n_points_in_slab,
            raster_width_px=slice_result.affine.width_px,
            raster_height_px=slice_result.affine.height_px,
            raster_world_width_m=float(world_w),
            raster_world_height_m=float(world_h),
            segments_detected=segments_detected,
            segments_after_regularize=segments_after,
            elevation_m=float(elevation),
            detector=params.detector.value,
        )

        result_payload = {
            "metrics": metrics.model_dump(),
            "params": params.model_dump(mode="json"),
            "affine": slice_result.affine.to_json(),
            "elevation_m": float(elevation),
            "axis_idx": int(axis_idx),
            "artifacts": {
                "slice_png": str(artifact_dir / "slice.png"),
                "slice_cleaned_png": str(artifact_dir / "slice_cleaned.png"),
                "overlay_png": str(artifact_dir / "overlay.png"),
                "overlay_raw_png": str(artifact_dir / "overlay_raw.png"),
                "dxf": str(dxf_path),
            },
        }
        (result_dir / "result.json").write_text(json.dumps(result_payload, indent=2))

        if on_complete is not None:
            on_complete(result_payload)

        _emit(
            job_id, "complete",
            f"Done in {elapsed:.1f}s · {segments_after} wall segments · DXF ready",
            1.0,
        )

    except Exception as exc:
        err = str(exc) or exc.__class__.__name__
        traceback.print_exc()
        if on_error is not None:
            try:
                on_error(err)
            except Exception:
                pass
        _emit(job_id, "error", err, 1.0)


def _segments_world_to_pixels(
    segments_world: np.ndarray,
    affine: slicer.RasterAffine,
) -> np.ndarray:
    """Inverse of :func:`classical.segments_pixels_to_world` for overlay rendering."""
    if len(segments_world) == 0:
        return np.zeros((0, 4), dtype=np.int32)
    res = affine.resolution_m_per_px
    seg = segments_world.astype(np.float64)
    x1p = (seg[:, 0, 0] - affine.origin_x) / res
    y1p = (seg[:, 0, 1] - affine.origin_y) / res
    x2p = (seg[:, 1, 0] - affine.origin_x) / res
    y2p = (seg[:, 1, 1] - affine.origin_y) / res
    return np.stack([x1p, y1p, x2p, y2p], axis=1).astype(np.int32)


async def run_vectorize_guarded(
    job_id: str,
    scan_path: Path,
    artifact_dir: Path,
    result_dir: Path,
    params: VectorizeParams,
    on_complete=None,
    on_error=None,
) -> None:
    """Semaphore-guarded async wrapper around :func:`run_vectorize`.

    Re-uses the same ``JOB_SEMAPHORE`` as the alignment pipeline so concurrent
    job count is capped across both pipelines.
    """
    async with JOB_SEMAPHORE:
        await asyncio.to_thread(
            run_vectorize,
            job_id, scan_path, artifact_dir, result_dir, params,
            on_complete, on_error,
        )
