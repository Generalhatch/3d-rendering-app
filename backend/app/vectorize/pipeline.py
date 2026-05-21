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
    ceiling_band_slicer as ceiling_band_mod,
    classical,
    clip_to_envelope as clip_mod,
    columns as columns_mod,
    density_slicer as density_slicer_mod,
    dxf_writer,
    envelope as envelope_mod,
    ingest_downsample as downsample_mod,
    normals as normals_mod,
    openings as openings_mod,
    preprocess,
    regularize,
    slicer,
    topology as topology_mod,
    walls as walls_mod,
    walls_contour as walls_contour_mod,
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
        _emit(job_id, "ingest", f"Loaded {raw_points:,} points", 0.12)

        # ── 1b. Voxel downsample (v4 Phase A — see ACCURACY_TO_CAD_QUALITY_PLAN.md § 0.6) ──
        # Single biggest speed lever in the pipeline.  At 5 mm voxel spacing,
        # walls are still represented by hundreds of points per metre, every
        # door frame by dozens.  Density rasters and alpha-shapes are
        # visually indistinguishable from native-resolution output, but every
        # downstream stage runs 10–50× faster.
        #
        # Done HERE, before envelope/normals/slice, so every other stage
        # benefits.  Envelope works correctly post-downsample because it
        # depends on the *spatial extent* of low-band points, not their
        # density (downsample preserves extent, only thins density).
        downsample_result: downsample_mod.DownsampleResult | None = None
        if params.voxel_downsample_m > 0:
            _emit(job_id, "ingest", "Voxel-downsampling cloud…", 0.135)
            try:
                if getattr(params, "voxel_downsample_auto", True):
                    downsample_result = downsample_mod.voxel_downsample_auto(
                        pcd, initial_voxel_m=params.voxel_downsample_m,
                    )
                else:
                    downsample_result = downsample_mod.voxel_downsample(
                        pcd, voxel_m=params.voxel_downsample_m,
                    )
                pcd = downsample_result.pcd
                _emit(job_id, "ingest", downsample_result.summary(), 0.15)
            except Exception as ds_err:
                # Downsampling should never fail on a non-empty cloud, but
                # if Open3D throws on some pathological input we'd rather
                # eat the runtime hit than abort the entire job.
                _emit(
                    job_id, "ingest",
                    f"Voxel downsample skipped: {ds_err} — running at native resolution",
                    0.15,
                )
        else:
            _emit(job_id, "ingest", f"Loaded {raw_points:,} points (downsample disabled)", 0.15)

        points_after_downsample = int(len(pcd.points))

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
        ceiling_band_points = 0
        wall_mask_pixels_after_gate = 0
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

            # ── 3a. Ceiling-band gate (v4 Phase A.2) ──────────────────────
            # Cloud2BIM 2025 strategy: a 1.9-2.3 m slab is mostly walls,
            # almost no furniture (cubicle stops ≤ 1.5 m, file cabinets ≤
            # 1.8 m), and above the 2.03 m standard door header so closed/
            # open doors don't fill the gap.  ANDing it with the density
            # mask kills tall furniture (bookcases / shelving units) that
            # the density score alone misclassifies as walls.
            if params.use_ceiling_band:
                try:
                    _emit(
                        job_id, "slice",
                        f"Slicing ceiling band ({params.ceiling_band_lo_m:.2f}–"
                        f"{params.ceiling_band_hi_m:.2f} m above floor)…",
                        0.36,
                    )
                    ceiling_params = ceiling_band_mod.CeilingBandParams(
                        z_lo_m=params.ceiling_band_lo_m,
                        z_hi_m=params.ceiling_band_hi_m,
                        resolution_m_per_px=params.resolution_m_per_px,
                    )
                    ceiling_slice = ceiling_band_mod.slice_ceiling_band(
                        pcd,
                        floor_z=floor_z,
                        axis_idx=axis_idx,
                        params=ceiling_params,
                        target_affine=slice_result.affine,
                    )
                    ceiling_band_points = int(ceiling_slice.n_points_in_slab)

                    # AND the two masks.  If the ceiling band is too sparse
                    # the helper returns the density mask unchanged so we
                    # don't silently emit an empty wall raster.
                    pre_and = int((slice_result.image > 0).sum())
                    gated = ceiling_band_mod.gate_with_ceiling(
                        slice_result.image, ceiling_slice.image,
                    )
                    post_and = int((gated > 0).sum())
                    wall_mask_pixels_after_gate = post_and
                    slice_result = slicer.SliceResult(
                        image=gated,
                        affine=slice_result.affine,
                        elevation_m=slice_result.elevation_m,
                        slab_thickness_m=slice_result.slab_thickness_m,
                        axis_idx=slice_result.axis_idx,
                        n_points_in_slab=slice_result.n_points_in_slab,
                        elevations_used=(
                            list(slice_result.elevations_used or [])
                            + list(ceiling_slice.elevations_used or [])
                        ),
                    )
                    # Persist the ceiling-band raster so the editor / debug
                    # tooling can compare it side-by-side with the density.
                    cv2.imwrite(
                        str(artifact_dir / "slice_ceiling.png"),
                        cv2.flip(ceiling_slice.image, 0),
                    )

                    if post_and == pre_and:
                        # Helper fell back — surface that explicitly.
                        _emit(
                            job_id, "slice",
                            f"Ceiling band sparse ({ceiling_band_points:,} pts) "
                            f"— gating skipped, density mask kept",
                            0.40,
                        )
                    else:
                        kept_pct = 100.0 * post_and / max(1, pre_and)
                        _emit(
                            job_id, "slice",
                            f"Ceiling gate: {pre_and:,} → {post_and:,} wall px "
                            f"({kept_pct:.1f}% kept; {ceiling_band_points:,} band pts)",
                            0.40,
                        )
                except Exception as cb_err:
                    _emit(
                        job_id, "slice",
                        f"Ceiling-band slice skipped: {cb_err}",
                        0.40,
                    )
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
        # v4 Phase A: post-voxel-downsample, the cloud is small enough (~3 M
        # points) that keeping it around for openings detection is cheap
        # (~40 MB).  Previously we freed it here and reloaded the entire
        # raw 680 MB file from disk for the door re-slice, adding ~30 s of
        # I/O.  We now keep `pcd` alive until after openings detection runs.

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

        # ── 5. Detect wall geometry ──────────────────────────────────────
        # Two paths:
        #   (a) v4 contour-based extraction (default — Cloud2BIM 2025) →
        #       walls come out as continuous polylines.  Eliminates the
        #       "688 raw segments → 97 final" regularizer attrition that
        #       § 13 of ACCURACY_TO_CAD_QUALITY_PLAN.md identified as the
        #       single biggest source of dropped walls.
        #   (b) Legacy Hough/FLD line detector (kept behind feature flag for
        #       A/B and as a fallback for buildings where contour topology
        #       breaks — large open spaces with very few walls).
        contour_walls_detected = 0
        seg_px = np.zeros((0, 4), dtype=np.int32)

        if params.use_contour_walls:
            _emit(
                job_id, "detect",
                f"Extracting wall contours (eps={params.contour_simplify_eps_m * 100:.1f} cm)…",
                0.55,
            )
            try:
                contour_params = walls_contour_mod.WallContourParams(
                    simplify_eps_m=params.contour_simplify_eps_m,
                    min_perimeter_m=max(params.min_wall_length_m * 2, 0.30),
                    min_segment_length_m=max(params.min_wall_length_m * 0.20, 0.08),
                )
                contour_result = walls_contour_mod.extract_wall_contours(
                    cleaned, slice_result.affine, params=contour_params,
                )
                clean_segments = contour_result.flat_segments
                segments_detected = int(contour_result.n_segments_total)
                contour_walls_detected = int(contour_result.n_contours_kept)
                _emit(
                    job_id, "detect",
                    walls_contour_mod.contour_summary(contour_result),
                    0.65,
                )
                # Contour path has no "rejected" segments — every segment
                # belongs to a kept contour by construction.  The downstream
                # ghost-promotion code still expects a list, so we hand it
                # an empty one rather than skipping the contract.
                rejected_segments = []
            except Exception as cw_err:
                _emit(
                    job_id, "detect",
                    f"Contour walls failed ({cw_err}) — falling back to "
                    f"line detector",
                    0.58,
                )
                # Fall through to the legacy detector path so the run can
                # still produce *something*.  In practice this never fires
                # on real scans; it's a defence-in-depth path for malformed
                # masks.
                params_use_contour = False
            else:
                params_use_contour = True
        else:
            params_use_contour = False

        if not params_use_contour:
            _emit(
                job_id, "detect",
                f"Detecting lines (detector={params.detector.value})…",
                0.55,
            )
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

            # ── 6. Pixel → world, then regularize ───────────────────────
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
            _emit(
                job_id, "regularize",
                f"{int(len(clean_segments))} clean segments (from {segments_detected}, "
                f"{len(rejected_segments)} rescuable ghost candidates)",
                0.80,
            )

        segments_after = int(len(clean_segments))

        # ── 6a. Clip walls to envelope (OPT-IN) ──────────────────────────
        # Some walls survive contour extraction even when they're outside
        # the building shell (window reflections, awnings, parapet
        # edges).  Cheap fix: drop walls whose midpoint sits outside the
        # envelope polygon + a generous buffer.
        #
        # DISABLED BY DEFAULT.  When the envelope is even slightly too
        # tight (DBSCAN dropped some legitimate exterior wall points)
        # this clips real walls and the floor plan collapses.  Enable
        # via params.clip_walls_to_envelope on scans where you've
        # confirmed the envelope is good and you have a parking-lot
        # contamination problem.  Buffer of 1 m is generous — only
        # catches truly external clutter.
        n_clipped_out = 0
        if (getattr(params, "clip_walls_to_envelope", False)
                and envelope_result is not None and len(clean_segments) > 0):
            clip_res = clip_mod.clip_walls_to_envelope(
                clean_segments,
                envelope_result.polygon_xy,
                buffer_m=1.00,
            )
            n_clipped_out = clip_res.n_dropped
            if n_clipped_out > 0:
                clean_segments = clip_res.kept
                segments_after = int(len(clean_segments))
                _emit(job_id, "regularize", clip_res.summary(), 0.81)

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

        # ── 6c. Junction snap + room inference ───────────────────────────
        # Close T-junctions / L-corners that overshoot or undershoot by a
        # few centimetres, then enumerate the planar faces of the wall
        # graph — each interior face is a room.  This is what visually
        # turns a "good sketch" into a CAD-ready floor plan: walls *meet*
        # cleanly and rooms become first-class geometry the operator can
        # label, tag, and area-tabulate.  See backend/app/vectorize/topology.py
        topology_result: topology_mod.TopologyResult | None = None
        room_segments = np.zeros((0, 2, 2), dtype=np.float64)
        if params.infer_rooms and len(wall_pairing.centerline_segments) > 1:
            try:
                # Auto-tune snap tolerance to wall thickness (v4): too-small
                # tolerance leaves T-junctions open; too-large fuses the two
                # faces of one wall into a single junction.  Median wall
                # thickness from the pairing stage is the right scale.
                topology_params = topology_mod.TopologyParams(
                    snap_tol_m=0.0,  # 0 → auto-tune from wall thickness
                    snapping_distance_m=params.snapping_distance_m,
                    wall_thickness_median_m=float(wall_pairing.median_thickness_m),
                )
                topology_result = topology_mod.build_topology(
                    wall_pairing.centerline_segments,
                    params=topology_params,
                )
                _emit(
                    job_id, "regularize",
                    topology_mod.topology_summary(topology_result),
                    0.84,
                )
                room_segments = topology_mod.rooms_to_segments(topology_result.rooms)
            except Exception as tp_err:
                _emit(
                    job_id, "regularize",
                    f"Room inference skipped: {tp_err}",
                    0.84,
                )

        # ── 7. Overlay PNG ───────────────────────────────────────────────
        _emit(job_id, "overlay", "Rendering review overlay…", 0.85)
        # Re-project clean segments back to pixel space so the overlay matches
        # what the operator will see in the DXF (not the raw detector output).
        clean_px = _segments_world_to_pixels(clean_segments, slice_result.affine)
        overlay = classical.render_overlay(cleaned, clean_px, color_bgr=(0, 0, 255), thickness=2)
        cv2.imwrite(str(artifact_dir / "overlay.png"), cv2.flip(overlay, 0))

        # Also save the raw (pre-regularize) detection overlay for comparison.
        # Contour-walls path: no separate "raw" stage exists — show the same
        # walls in orange so the operator can still A/B against the cleaned
        # overlay visually (the orange + red won't differ when contour mode
        # is on, which is itself a useful signal that nothing got dropped).
        raw_px = clean_px if params.use_contour_walls else seg_px
        raw_overlay = classical.render_overlay(cleaned, raw_px, color_bgr=(0, 165, 255), thickness=1)
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
        # v4 (Cloud2BIM-style) simplification:
        # The wall raster IS already the right surface for opening detection
        # when we use the ceiling-band gate.  The ceiling-band slice sits at
        # 1.9-2.3 m above the floor — that's ABOVE the standard 2.03 m door
        # header, so closed-door slabs have already disappeared and only the
        # gap remains.  The AND with the density mask doesn't re-fill the
        # gap because the density slicer also returns "no support" where the
        # door slab is.
        #
        # Therefore: detect openings on the SAME `cleaned` raster the wall
        # extractor used.  This removes:
        #   - The 30-second reload of the raw 680 MB scan (we no longer
        #     `del pcd`, so this isn't needed even as a fallback).
        #   - The duplicate slice + preprocess step (3-6 s saved).
        #   - The elevation-misalignment failure mode (the old re-slice at
        #     `elevation + 0.60` could land below or above doors depending
        #     on the operator's chosen `elevation` knob).
        # Net: openings work on more scans, AND ~40 s faster.
        opening_segs: list[np.ndarray] = []
        # v4 bug fix: the opening detector samples perpendicular to each
        # wall.  If we pass it the CONTOUR FACE segments (which sit on
        # either face of a real wall), the perpendicular band from the
        # outer face hits the inner face of the same wall and reports
        # "supported" — masking every door.  Pass the CENTERLINES from
        # the pairing stage instead; perpendicular sampling from there
        # sees no wall at a door from either side, so the gap is
        # detected.  Falls back to clean_segments only if pairing
        # produced zero centerlines (rare, debug-only).
        walls_for_openings = (
            wall_pairing.centerline_segments
            if (wall_pairing is not None and
                len(wall_pairing.centerline_segments) > 0)
            else clean_segments
        )
        if params.detect_openings and len(walls_for_openings) > 0:
            _emit(
                job_id, "openings",
                "Scanning walls for door-shaped gaps on ceiling-gated raster…",
                0.91,
            )
            try:
                # Increase perpendicular band to half of typical wall
                # thickness — this way the sampler sees both faces of
                # the wall, but at a door it sees NEITHER and correctly
                # registers the gap.
                op_params = openings_mod.OpeningParams(
                    perpendicular_band_m=max(
                        0.08,
                        float(wall_pairing.median_thickness_m) * 0.6,
                    ),
                )
                detected = openings_mod.detect_openings(
                    cleaned, walls_for_openings, slice_result.affine,
                    params=op_params,
                )
            except Exception as op_err:
                _emit(
                    job_id, "openings",
                    f"Opening detection failed: {op_err}", 0.92,
                )
                detected = []

            opening_segs = [d.seg for d in detected]
            _emit(
                job_id, "openings",
                f"Detected {len(opening_segs)} opening"
                f"{'s' if len(opening_segs) != 1 else ''}",
                0.92,
            )
            if opening_segs:
                walls_px = _segments_world_to_pixels(walls_for_openings, slice_result.affine)
                opens_px = _segments_world_to_pixels(
                    np.array(opening_segs), slice_result.affine,
                )
                ovl = openings_mod.render_openings_overlay(cleaned, walls_px, opens_px)
                cv2.imwrite(str(artifact_dir / "overlay_openings.png"), cv2.flip(ovl, 0))

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

        # v4: split round vs rectangular columns.  Rectangular ones still go
        # via the segments array (4 polyline edges); round ones go as DXF
        # CIRCLE entities via the new ``circles_by_class`` parameter to
        # write_dxf.  This matches what commercial CAD drawings do.
        rect_columns = [c for c in column_objs if not c.is_round]
        round_columns = [c for c in column_objs if c.is_round]
        columns_array = columns_mod.columns_to_segments(rect_columns)
        column_circles: list[tuple[tuple[float, float], float]] = []
        for c in round_columns:
            # Diameter = mean(long, short) of the min-area rect; the round
            # classifier already ensured aspect ≤ 1.2 so the rect is nearly
            # square — mean is the correct radius estimate.
            radius = float(0.5 * 0.5 * (c.size_m[0] + c.size_m[1]))
            column_circles.append((c.centre_m, radius))

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
                "rooms":          room_segments,
                "openings":       openings_array,
                "columns":        columns_array,
            },
            dxf_path,
            annotation_text=annotation,
            circles_by_class=(
                {"columns": column_circles} if column_circles else None
            ),
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
                    "layer": "rooms",
                    "x1": float(seg[0, 0]),
                    "y1": float(seg[0, 1]),
                    "x2": float(seg[1, 0]),
                    "y2": float(seg[1, 1]),
                }
                for seg in room_segments
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

        # v4: also persist per-layer JSON files alongside the combined
        # segments.json.  The editor can keep reading segments.json (no
        # frontend changes required), but future code can lazy-load only
        # the layers it needs — useful for very large floor plans where
        # rooms + walls alone are MBs of polylines.  The per-layer files
        # use the same item shape as segments.json's `segments` list, so
        # a JSON reader can be reused.
        per_layer_payload = {
            "version": 1,
            "units": "metres",
            "affine": slice_result.affine.to_json(),
            "raster": {
                "width_px": int(slice_result.affine.width_px),
                "height_px": int(slice_result.affine.height_px),
                "y_flipped_for_display": True,
            },
        }
        layer_buckets: dict[str, list[dict]] = {}
        for entry in segments_payload["segments"]:
            layer_buckets.setdefault(entry["layer"], []).append(entry)
        for layer_name, entries in layer_buckets.items():
            payload = dict(per_layer_payload)
            payload["layer"] = layer_name
            payload["segments"] = entries
            (result_dir / f"segments_{layer_name}.json").write_text(
                json.dumps(payload, indent=2)
            )
        # v4 round columns — separate file because the data shape (centre +
        # radius) doesn't fit the segments schema.
        if column_circles:
            circles_payload = {
                "version": 1,
                "units": "metres",
                "layer": "columns_circles",
                "circles": [
                    {
                        "id": uuid.uuid4().hex[:12],
                        "cx": float(cx),
                        "cy": float(cy),
                        "r": float(r),
                    }
                    for (cx, cy), r in column_circles
                ],
            }
            (result_dir / "segments_columns_circles.json").write_text(
                json.dumps(circles_payload, indent=2)
            )

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
            rooms_detected=(
                len(topology_result.rooms) if topology_result is not None else 0
            ),
            junctions_merged=(
                int(topology_result.n_endpoints_merged)
                if topology_result is not None else 0
            ),
            t_junctions_extended=(
                int(topology_result.n_extended)
                if topology_result is not None else 0
            ),
            # v4 Phase A stats — voxel downsample + ceiling-band gate.
            points_after_downsample=points_after_downsample,
            downsample_voxel_m=(
                float(downsample_result.voxel_m)
                if downsample_result is not None else None
            ),
            ceiling_band_used=bool(params.use_ceiling_band and params.use_density_slicer),
            ceiling_band_n_points=(
                int(ceiling_band_points) if ceiling_band_points else None
            ),
            wall_mask_pixels=(
                int(wall_mask_pixels_after_gate) if wall_mask_pixels_after_gate else None
            ),
            contour_walls_detected=(
                int(contour_walls_detected) if contour_walls_detected else None
            ),
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
        # v4 Phase A: ceiling-band slice (when use_ceiling_band is on).
        ceiling_png = artifact_dir / "slice_ceiling.png"
        if ceiling_png.exists():
            artifacts["slice_ceiling_png"] = str(ceiling_png)
        density_png = artifact_dir / "slice_density.png"
        if density_png.exists():
            artifacts["slice_density_png"] = str(density_png)

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
