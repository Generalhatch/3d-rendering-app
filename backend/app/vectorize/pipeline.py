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
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ..limits import JOB_SEMAPHORE
from ..models.vectorize_job import VectorizeMetrics, VectorizeParams, VectorizeStatus
from ..sse import publish
from . import (
    classical,
    columns as columns_mod,
    density_slicer as density_slicer_mod,
    dxf_writer,
    envelope as envelope_mod,
    normals as normals_mod,
    openings as openings_mod,
    preprocess,
    regularize,
    slicer,
    walls as walls_mod,
)


# Coverage tolerance: a foreground (wall) pixel counts as "covered" if any
# kept segment passes within this radius.  8 cm at 1 cm/px = 8 px — wide
# enough to absorb both faces of a typical 10–15 cm wall, narrow enough that
# missed walls show up as obvious red blobs in the diagnostic.
COVERAGE_RADIUS_M = 0.08


def _emit(job_id: str, stage: str, message: str, progress: float) -> None:
    publish(job_id, stage, message, progress)


def _compute_coverage(
    cleaned_raster: np.ndarray,
    segments_world: np.ndarray,
    affine: slicer.RasterAffine,
    radius_m: float = COVERAGE_RADIUS_M,
) -> tuple[float, int, int, np.ndarray]:
    """How much of the raster foreground is within ``radius_m`` of any segment.

    Returns ``(coverage_pct, foreground_px, uncovered_px, gaps_mask)`` where
    ``gaps_mask`` is a ``(H, W)`` uint8 image — 255 where a wall pixel is
    *missing* coverage, 0 elsewhere.  The mask is in internal raster
    convention (row 0 = low world Y); persist it Y-flipped for display.
    """
    h, w = cleaned_raster.shape
    fg_mask = cleaned_raster > 0
    foreground_px = int(fg_mask.sum())
    if foreground_px == 0:
        return 1.0, 0, 0, np.zeros_like(cleaned_raster)

    # Rasterize segments into a binary mask (1 px wide) in pixel space.
    seg_mask = np.zeros_like(cleaned_raster)
    if len(segments_world) > 0:
        res = affine.resolution_m_per_px
        for seg in segments_world:
            x1 = int(round((seg[0, 0] - affine.origin_x) / res))
            y1 = int(round((seg[0, 1] - affine.origin_y) / res))
            x2 = int(round((seg[1, 0] - affine.origin_x) / res))
            y2 = int(round((seg[1, 1] - affine.origin_y) / res))
            cv2.line(seg_mask, (x1, y1), (x2, y2), color=255, thickness=1)

    # Distance transform: for every pixel, distance (in px) to the nearest
    # *segment* pixel.  Input must be 0 where segments live, 255 elsewhere.
    inv = np.where(seg_mask > 0, 0, 255).astype(np.uint8)
    dist_px = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
    radius_px = radius_m / affine.resolution_m_per_px

    covered_mask = fg_mask & (dist_px <= radius_px)
    uncovered_mask = fg_mask & (dist_px > radius_px)
    covered_px = int(covered_mask.sum())
    uncovered_px = int(uncovered_mask.sum())
    coverage_pct = covered_px / foreground_px if foreground_px else 1.0

    gaps_mask = np.where(uncovered_mask, 255, 0).astype(np.uint8)
    return coverage_pct, foreground_px, uncovered_px, gaps_mask


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
            # 1.6 m clears desks (~0.75 m), filing cabinets (~1.2 m), and most
            # cubicle dividers (~1.3–1.5 m) while staying below typical ceiling
            # fixtures, beams, and HVAC (≥ 2.1 m).  See VECTORIZE_SETTINGS.md.
            shoulder_offset = 1.60
            floor_z = float(floor.floor_z_estimate)
            elevation = floor_z + shoulder_offset
            axis_idx = int(floor.axis_idx)
            _emit(
                job_id, "floor",
                f"Floor at {floor_z:.2f} m (axis="
                f"{'Z' if axis_idx == 2 else 'Y'}-up) — slicing at "
                f"{elevation:.2f} m (floor + {shoulder_offset:.2f} m)",
                0.25,
            )
        else:
            elevation = float(params.elevation_m)
            axis_idx = 2  # Default to Z-up when explicit elevation supplied.
            # No floor detection ran; assume floor is ~1.6 m below the
            # explicit slice plane (typical shoulder-height default).  Used
            # only by the envelope extractor — slight error here is fine
            # because the envelope band is a 45 cm range, not a sharp plane.
            floor_z = elevation - 1.60
            _emit(
                job_id, "floor",
                f"Slicing at user-supplied elevation {elevation:.2f} m "
                f"(assuming floor at {floor_z:.2f} m for envelope)",
                0.25,
            )

        # ── 2a. Building envelope (priority #1 — external walls) ─────────
        # Extracts the building shell as a single closed polygon via a
        # concave-hull on a floor-level point projection.  Independent of
        # the line detector — locks the exterior wall even if the rest of
        # the pipeline stumbles.  Run BEFORE the vertical-surface filter
        # so the floor-level band still contains floor points (which is
        # what gives the hull its lateral extent).
        envelope_segments_world: np.ndarray = np.zeros((0, 2, 2), dtype=np.float64)
        envelope_result = None
        if params.extract_envelope:
            try:
                _emit(job_id, "envelope", "Extracting building envelope…", 0.255)
                envelope_result = envelope_mod.extract_envelope(
                    pcd, floor_z=floor_z, axis_idx=axis_idx,
                )
                envelope_segments_world = envelope_result.segments
                envelope_mod.save_envelope(
                    envelope_result, result_dir / "envelope.json",
                )
                _emit(
                    job_id, "envelope",
                    f"Envelope: {envelope_result.area_m2:.0f} m² floorplate, "
                    f"{envelope_result.perimeter_m:.1f} m perimeter, "
                    f"{len(envelope_result.segments)} edges "
                    f"(alpha={envelope_result.alpha_used:.2f})",
                    0.265,
                )
            except Exception as env_err:
                _emit(
                    job_id, "envelope",
                    f"Envelope extraction skipped: {env_err}",
                    0.265,
                )

        # ── 2b. Filter to vertical surfaces (point-normal gate) ──────────
        # Estimate per-point normals, keep only points on vertical surfaces.
        # This removes floor returns, ceiling returns, desk tops, cabinet
        # tops, monitor screens, picture frames, and similar horizontal
        # contaminants BEFORE any 2D projection.  Single biggest noise-floor
        # reduction available in the classical-CV pipeline.  See
        # backend/app/vectorize/normals.py for details.
        if params.vertical_surfaces_only:
            _emit(job_id, "normals", "Estimating point normals…", 0.27)
            n_before = int(len(pcd.points))
            try:
                pcd = normals_mod.filter_to_vertical_surfaces(
                    pcd, axis_idx=axis_idx,
                )
                n_after = int(len(pcd.points))
                _emit(
                    job_id, "normals",
                    normals_mod.filter_summary(n_before, n_after),
                    0.29,
                )
                if n_after < 1000:
                    # Filter wiped almost everything — likely an axis mis-detection
                    # or an unusual scan format.  Bail out of the filter rather
                    # than starve the slicer; downstream stages still work on
                    # unfiltered points.
                    raise ValueError(
                        f"only {n_after} points survived vertical-surface filter "
                        f"(input had {n_before:,}) — falling back to unfiltered"
                    )
            except Exception as filt_err:
                # Reload the cloud to recover from any in-place filter damage.
                _emit(
                    job_id, "normals",
                    f"Vertical-surface filter skipped: {filt_err}",
                    0.29,
                )
                pcd = load_point_cloud(scan_path)

        # ── 3. Slice (density-column / single / multi-elevation OR-fused) ─
        if params.use_density_slicer:
            # Vertical-column density score: every XY cell gets the fraction
            # of its vertical wall-band heights that have any point in them.
            # Walls light up (continuous floor-to-ceiling); furniture stays
            # dim.  See backend/app/vectorize/density_slicer.py.
            _emit(
                job_id, "slice",
                f"Building density raster (band {floor_z + 0.30:.2f}–"
                f"{floor_z + 2.20:.2f} m)…",
                0.30,
            )
            density_params = density_slicer_mod.DensitySliceParams(
                resolution_m_per_px=params.resolution_m_per_px,
            )
            slice_result = density_slicer_mod.slice_to_raster_density(
                pcd,
                floor_z=floor_z,
                axis_idx=axis_idx,
                params=density_params,
            )
            # Also persist the raw density image (pre-threshold) for the
            # editor — useful diagnostic of what the model "saw".
            try:
                density_uint8, _ = density_slicer_mod.density_image_uint8(
                    pcd, floor_z=floor_z, axis_idx=axis_idx,
                    params=density_params,
                )
                cv2.imwrite(
                    str(artifact_dir / "slice_density.png"),
                    cv2.flip(density_uint8, 0),
                )
            except Exception:
                pass
        elif params.multi_elevation:
            # ±0.4 m brackets the centre slice — catches walls hidden by tall
            # furniture or doorway headers without straying into ceiling.
            elevations = [elevation - 0.40, elevation, elevation + 0.40]
            _emit(
                job_id, "slice",
                f"Projecting 3 slabs (elev = {elevations[0]:.2f} / "
                f"{elevations[1]:.2f} / {elevations[2]:.2f} m)…",
                0.30,
            )
            slice_result = slicer.slice_to_raster_multi(
                pcd,
                elevations_m=elevations,
                slab_thickness_m=params.slab_thickness_m,
                resolution_m_per_px=params.resolution_m_per_px,
                axis_idx=axis_idx,
            )
        else:
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
        elev_summary = (
            f"3 slabs OR-merged" if params.multi_elevation else "1 slab"
        )
        _emit(
            job_id, "slice",
            f"Raster: {slice_result.affine.width_px}×{slice_result.affine.height_px} px "
            f"({slice_result.n_points_in_slab:,} pts · {elev_summary})",
            0.40,
        )

        # ── 4. Preprocess ────────────────────────────────────────────────
        _emit(job_id, "preprocess", "Cleaning raster…", 0.45)
        # When ``remove_speckle`` is on, run a small morphological OPEN after
        # the gap-bridging CLOSE.  3-px elliptical kernel kills any cluster
        # smaller than ~3 px (≈ 3 cm at 1 cm/px) — that's isolated scanner
        # noise and furniture stippling, well below any real wall thickness.
        preprocess_params = preprocess.PreprocessParams(
            open_kernel_px=3 if params.remove_speckle else 0,
            open_iterations=1 if params.remove_speckle else 0,
        )
        cleaned = preprocess.preprocess(slice_result.image, preprocess_params)

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
        reg_result = regularize.regularize_with_provenance(segments_world, reg_params)
        clean_segments = reg_result.kept
        rejected_segments = reg_result.rejected
        segments_after = int(len(clean_segments))
        _emit(
            job_id, "regularize",
            f"{segments_after} clean segments (from {segments_detected}, "
            f"{len(rejected_segments)} rescuable ghost candidates)",
            0.80,
        )

        # ── 6b. Wall-thickness pairing → double-line entities ────────────
        # Collapse parallel detected segments 7–35 cm apart (the two faces of
        # a wall) into single thickness-aware Wall entities.  Emits both
        # faces on the WALLS_FACES DXF layer plus a centerline on WALLS.
        # See backend/app/vectorize/walls.py — this is the single biggest
        # visual lift toward Image 1 ("looks like CAD, not a sketch").
        wall_pairing = walls_mod.WallPairingResult(
            walls=[],
            face_segments=np.zeros((0, 2, 2), dtype=np.float64),
            centerline_segments=clean_segments,
            median_thickness_m=walls_mod.WallPairingParams().fallback_thickness_m,
            n_paired=0,
            n_unpaired=int(len(clean_segments)),
        )
        if params.pair_walls and len(clean_segments) > 1:
            try:
                wall_pairing = walls_mod.pair_walls(clean_segments)
                _emit(
                    job_id, "regularize",
                    walls_mod.pairing_summary(wall_pairing),
                    0.82,
                )
            except Exception as wp_err:
                _emit(
                    job_id, "regularize",
                    f"Wall pairing skipped: {wp_err}",
                    0.82,
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

        # ── 7b. Coverage diagnostic ──────────────────────────────────────
        # Compute *which raster pixels are far from every kept segment* — the
        # operator-facing "did we miss anything?" signal.  Written as an RGBA
        # PNG (red + alpha where uncovered, transparent elsewhere) so the
        # frontend can blend it directly over the raster without colour maths.
        _emit(job_id, "overlay", "Computing wall-coverage diagnostic…", 0.88)
        cov_pct, fg_px, uncov_px, gaps_mask = _compute_coverage(
            cleaned, clean_segments, slice_result.affine, radius_m=COVERAGE_RADIUS_M,
        )
        rgba = np.zeros((gaps_mask.shape[0], gaps_mask.shape[1], 4), dtype=np.uint8)
        gap_pixels = gaps_mask > 0
        rgba[gap_pixels] = (60, 60, 240, 200)  # BGRA — strong red, semi-opaque
        cv2.imwrite(str(artifact_dir / "coverage_gaps.png"), cv2.flip(rgba, 0))
        _emit(
            job_id, "overlay",
            f"Coverage: {cov_pct*100:.1f}% of {fg_px:,} wall pixels covered "
            f"({uncov_px:,} flagged for operator review)",
            0.90,
        )

        # ── 7c. Opening (door / wall-gap) detection ──────────────────────
        # Why we re-slice above the door header (Phase 5 "E" recall-lift):
        # The wall raster (single- or multi-elevation) sees *closed* door slabs
        # as walls — they fill the raster at shoulder height.  Opening detection
        # needs a slice ABOVE the door so the slab is gone and only the gap
        # remains.  Standard interior doors top out at 2.03 m, so we slice at
        # ``floor + 2.20 m`` (== elevation + 0.60 m for the shoulder default).
        # We re-use the same slab thickness + axis as the main slice.  Cost:
        # ~3–6 s on a typical scan; only paid when ``detect_openings`` is on.
        opening_segs: list[np.ndarray] = []
        opening_raster_label = "shoulder (closed doors fill the gap)"
        if params.detect_openings and len(clean_segments) > 0:
            _emit(job_id, "openings", "Slicing above door header for opening detection…", 0.905)
            door_elevation = elevation + 0.60
            from ..pipeline.ingest import load_point_cloud  # heavy, local import
            # We already freed ``pcd`` after the main slice — reload it.  Only
            # this branch pays the I/O cost, and only when detect_openings is on.
            pcd_for_doors = load_point_cloud(scan_path)
            try:
                door_slice = slicer.slice_to_raster(
                    pcd_for_doors,
                    elevation_m=door_elevation,
                    slab_thickness_m=params.slab_thickness_m,
                    resolution_m_per_px=params.resolution_m_per_px,
                    axis_idx=axis_idx,
                )
            finally:
                del pcd_for_doors

            # Same preprocessing as the wall raster so the perpendicular-band
            # support test sees a comparable signal (closed CLOSE + optional OPEN).
            door_cleaned = preprocess.preprocess(door_slice.image, preprocess_params)

            # The door slice is sliced fresh, so its affine is *almost* identical
            # to the main affine but may differ slightly in canvas extent (the
            # high slice has different point coverage).  We need an affine that
            # the openings sampler can use to index ``door_cleaned`` correctly —
            # use ``door_slice.affine``, which matches its own raster.
            opening_raster_label = (
                f"high slice @ {door_elevation:.2f} m "
                f"({door_slice.affine.width_px}×{door_slice.affine.height_px} px)"
            )

            _emit(job_id, "openings", "Scanning walls for door-shaped gaps…", 0.91)
            detected = openings_mod.detect_openings(
                door_cleaned, clean_segments, door_slice.affine,
            )
            opening_segs = [d.seg for d in detected]
            _emit(
                job_id, "openings",
                f"Detected {len(opening_segs)} opening{'s' if len(opening_segs) != 1 else ''} "
                f"from {opening_raster_label}",
                0.92,
            )
            if opening_segs:
                # Visual sanity-check overlay — walls red, openings yellow,
                # over the door-header slice the detector actually used.
                walls_px = _segments_world_to_pixels(clean_segments, door_slice.affine)
                opens_px = _segments_world_to_pixels(
                    np.array(opening_segs), door_slice.affine,
                )
                ovl = openings_mod.render_openings_overlay(door_cleaned, walls_px, opens_px)
                cv2.imwrite(str(artifact_dir / "overlay_openings.png"), cv2.flip(ovl, 0))
                # Also persist the door slice raster itself so the editor can
                # show it as an alternate backdrop (operator can verify the gaps).
                cv2.imwrite(str(artifact_dir / "slice_doors.png"), cv2.flip(door_cleaned, 0))

        openings_array = (
            np.array(opening_segs, dtype=np.float64)
            if opening_segs
            else np.zeros((0, 2, 2), dtype=np.float64)
        )

        # ── 7d. Column detection ────────────────────────────────────────
        # Columns: isolated, square-ish blobs that survived the morphology
        # pass.  Detection runs on the *wall* raster (not the door slice) —
        # structural columns sit between walls in plan view, so the shoulder-
        # height slice that captures walls also captures columns.
        column_objs: list[columns_mod.DetectedColumn] = []
        if params.detect_columns and len(clean_segments) > 0:
            _emit(job_id, "columns", "Scanning for column-shaped blobs…", 0.925)
            column_objs = columns_mod.detect_columns(
                cleaned, clean_segments, slice_result.affine,
            )
            _emit(
                job_id, "columns",
                f"Detected {len(column_objs)} column{'s' if len(column_objs) != 1 else ''}",
                0.93,
            )
            if column_objs:
                walls_px = _segments_world_to_pixels(clean_segments, slice_result.affine)
                ovl = columns_mod.render_columns_overlay(
                    cleaned, walls_px, column_objs, slice_result.affine,
                )
                cv2.imwrite(str(artifact_dir / "overlay_columns.png"), cv2.flip(ovl, 0))

        columns_array = columns_mod.columns_to_segments(column_objs)

        # ── 7e. Per-layer coverage diagnostic ────────────────────────────
        # ``coverage_by_layer`` is the operator's "did we miss anything?"
        # answer per class.  Today only walls have a meaningful denominator
        # (their raster foreground IS what wall lines explain).  Openings
        # and columns are about *instance count* (how many doors/columns)
        # rather than *pixel coverage*; their numbers in this metric end
        # up dominated by furniture noise and aren't useful to the
        # operator, so we omit them.  The data shape is kept open so
        # future classes (e.g. windows from a second high slice) can plug
        # in when they have a meaningful denominator.
        coverage_by_layer: dict[str, float] = {"walls": float(cov_pct)}

        # ── 8. DXF + segments.json ───────────────────────────────────────
        _emit(job_id, "dxf", "Writing DXF…", 0.94)
        envelope_summary = (
            f" envelope=yes({envelope_result.area_m2:.0f}m²)"
            if envelope_result is not None
            else ""
        )
        annotation = (
            f"Beehive Automations L.L.C. Vectorize | job={job_id} | "
            f"elev={elevation:.2f}m | slab={params.slab_thickness_m:.2f}m | "
            f"res={params.resolution_m_per_px:.3f}m/px | "
            f"detector={params.detector.value} | "
            f"walls={segments_after} openings={len(opening_segs)} "
            f"columns={len(column_objs)}{envelope_summary}"
        )
        dxf_path = result_dir / "vectorized.dxf"
        # Multi-layer DXF: WALLS_EXTERIOR (envelope) + WALLS (centerlines) +
        # WALLS_FACES (paired double-line walls) + OPENINGS + COLUMNS.
        # Reserved-but-unwritten slots (ROOMS, WINDOWS, MEP, TEXT) are still
        # created so CAD operators can drop content onto them.
        dxf_writer.write_dxf(
            {
                "walls_exterior": envelope_segments_world,
                "walls":          wall_pairing.centerline_segments,
                "walls_faces":    wall_pairing.face_segments,
                "openings":       openings_array,
                "columns":        columns_array,
            },
            dxf_path,
            annotation_text=annotation,
        )

        # Persist segments as JSON with stable IDs so the editor (Phase 3) has
        # an addressable source-of-truth.  The DXF is the *deliverable*; this
        # JSON is the *editable record*.  The affine + raster URL are bundled
        # so the editor can render the slice as a backdrop without a second
        # round-trip.
        segments_payload = {
            "version": 1,
            "units": "metres",
            "affine": slice_result.affine.to_json(),
            "raster": {
                "width_px": int(slice_result.affine.width_px),
                "height_px": int(slice_result.affine.height_px),
                "y_flipped_for_display": True,
            },
            "segments": [
                {
                    "id": uuid.uuid4().hex[:12],
                    "layer": "walls_exterior",
                    "x1": float(seg[0, 0]),
                    "y1": float(seg[0, 1]),
                    "x2": float(seg[1, 0]),
                    "y2": float(seg[1, 1]),
                }
                for seg in envelope_segments_world
            ] + [
                {
                    "id": uuid.uuid4().hex[:12],
                    "layer": "walls",
                    "x1": float(seg[0, 0]),
                    "y1": float(seg[0, 1]),
                    "x2": float(seg[1, 0]),
                    "y2": float(seg[1, 1]),
                }
                for seg in wall_pairing.centerline_segments
            ] + [
                {
                    "id": uuid.uuid4().hex[:12],
                    "layer": "walls_faces",
                    "x1": float(seg[0, 0]),
                    "y1": float(seg[0, 1]),
                    "x2": float(seg[1, 0]),
                    "y2": float(seg[1, 1]),
                }
                for seg in wall_pairing.face_segments
            ] + [
                {
                    "id": uuid.uuid4().hex[:12],
                    "layer": "openings",
                    "x1": float(seg[0, 0]),
                    "y1": float(seg[0, 1]),
                    "x2": float(seg[1, 0]),
                    "y2": float(seg[1, 1]),
                }
                for seg in opening_segs
            ] + [
                {
                    "id": uuid.uuid4().hex[:12],
                    "layer": "columns",
                    "x1": float(seg[0, 0]),
                    "y1": float(seg[0, 1]),
                    "x2": float(seg[1, 0]),
                    "y2": float(seg[1, 1]),
                }
                for seg in columns_array
            ],
        }
        (result_dir / "segments.json").write_text(json.dumps(segments_payload, indent=2))

        # Persist the rejected ghost candidates separately so the editor can
        # render them on demand without bloating the primary segments.json.
        rejected_payload = {
            "version": 1,
            "units": "metres",
            "rejected": [
                {
                    "id": f"ghost-{uuid.uuid4().hex[:10]}",
                    "layer": "walls",
                    "x1": float(r.seg[0, 0]),
                    "y1": float(r.seg[0, 1]),
                    "x2": float(r.seg[1, 0]),
                    "y2": float(r.seg[1, 1]),
                    "dropped_by": r.dropped_by,
                    "length_m": float(r.length_m),
                    "angle_deg": float(r.angle_deg),
                }
                for r in rejected_segments
            ],
        }
        (result_dir / "rejected_segments.json").write_text(
            json.dumps(rejected_payload, indent=2)
        )

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
            coverage_pct=float(cov_pct),
            coverage_radius_m=float(COVERAGE_RADIUS_M),
            foreground_px=int(fg_px),
            uncovered_px=int(uncov_px),
            elevations_used_m=list(slice_result.elevations_used or [elevation]),
            openings_detected=len(opening_segs),
            columns_detected=len(column_objs),
            coverage_by_layer=coverage_by_layer,
            walls_paired=int(wall_pairing.n_paired),
            walls_unpaired=int(wall_pairing.n_unpaired),
            walls_median_thickness_m=float(wall_pairing.median_thickness_m),
        )

        artifacts = {
            "slice_png": str(artifact_dir / "slice.png"),
            "slice_cleaned_png": str(artifact_dir / "slice_cleaned.png"),
            "overlay_png": str(artifact_dir / "overlay.png"),
            "overlay_raw_png": str(artifact_dir / "overlay_raw.png"),
            "coverage_gaps_png": str(artifact_dir / "coverage_gaps.png"),
            "dxf": str(dxf_path),
            "segments_json": str(result_dir / "segments.json"),
        }
        if opening_segs:
            artifacts["overlay_openings_png"] = str(artifact_dir / "overlay_openings.png")
        if column_objs:
            artifacts["overlay_columns_png"] = str(artifact_dir / "overlay_columns.png")

        result_payload = {
            "metrics": metrics.model_dump(),
            "params": params.model_dump(mode="json"),
            "affine": slice_result.affine.to_json(),
            "elevation_m": float(elevation),
            "axis_idx": int(axis_idx),
            "artifacts": artifacts,
        }
        (result_dir / "result.json").write_text(json.dumps(result_payload, indent=2))

        if on_complete is not None:
            on_complete(result_payload)

        completion_summary = (
            f"Done in {elapsed:.1f}s · {segments_after} walls"
            + (f" · {len(opening_segs)} openings" if opening_segs else "")
            + (f" · {len(column_objs)} columns" if column_objs else "")
            + " · DXF ready"
        )
        _emit(job_id, "complete", completion_summary, 1.0)

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
