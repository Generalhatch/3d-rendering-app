"""Pydantic models for the Vectorize (scan-to-CAD) pipeline.

Separate from the alignment ``job`` models so the two surfaces can evolve
independently — the vectorize pipeline takes a point cloud and an elevation,
detects 2D line geometry on a rasterized slice, and emits a DXF.
"""
from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


class VectorizeStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class DetectorName(str, enum.Enum):
    """Which classical line detector to run.

    - ``hough``: probabilistic Hough transform (``cv2.HoughLinesP``).  Robust,
      well-understood, fast.  Tends to produce many short segments on dense edges.
    - ``fld``:   Fast Line Detector (``cv2.ximgproc.createFastLineDetector``).
      Modern LSD replacement.  Produces cleaner long segments on architectural
      edges; better for the wall use-case.
    - ``both``:  run both and merge — slower but useful for the Phase 0/1 spike
      where we want to compare detector quality.
    """
    hough = "hough"
    fld = "fld"
    both = "both"


class VectorizeParams(BaseModel):
    """Tunable parameters for one vectorize run.

    All distances are in metres.  Defaults are tuned to handle a typical
    interior LiDAR scan at ~1 cm point spacing **without intervention** —
    they prioritise recall + a clean raster backdrop over speed.  Power users
    can dial individual knobs from the side panel.
    """
    elevation_m: Optional[float] = Field(
        default=None,
        description="Absolute elevation of the slice plane in the scan's local "
                    "frame.  If null, the pipeline auto-detects the floor and "
                    "slices at floor_z + 1.6 m — high enough to clear desks, "
                    "filing cabinets, and most cubicle dividers; low enough to "
                    "stay below ducts, beams, and ceiling fixtures.  Bump this "
                    "to 1.7–1.8 m on offices with tall cubicles; drop to 1.2 m "
                    "for low-ceiling industrial spaces.",
    )
    slab_thickness_m: float = Field(
        default=0.10,
        ge=0.02, le=2.0,
        description="Vertical thickness of the slab projected to the raster.  "
                    "Thinner = cleaner cross-section (less furniture/duct "
                    "bleed-through); thicker = more pixels per wall.  10 cm is "
                    "the sweet spot for clean scans; bump to 0.20 m for sparse "
                    "or noisy point clouds.",
    )
    resolution_m_per_px: float = Field(
        default=0.01,
        ge=0.002, le=0.10,
        description="World metres per raster pixel.  5 mm/px is high-quality "
                    "but produces large images; 1 cm/px is the practical default.",
    )
    detector: DetectorName = Field(
        default=DetectorName.both,
        description="Which classical detector to run.  'both' merges Hough + "
                    "FLD for maximum recall (~5 s extra per scan).  'fld' alone "
                    "is faster with slightly fewer fragments.",
    )
    min_wall_length_m: float = Field(
        default=0.40,
        ge=0.05, le=10.0,
        description="Drop any detected segment shorter than this.  0.40 m "
                    "keeps short interior partitions and closet walls; bump "
                    "to 0.80–1.00 m if you only want major structural walls.",
    )
    manhattan_snap: bool = Field(
        default=True,
        description="After detection, filter to segments aligned with the two "
                    "dominant building axes (Manhattan-world).  Disable on "
                    "rotated / curved buildings.",
    )
    merge_collinear: bool = Field(
        default=True,
        description="Merge near-collinear segments that are close end-to-end into "
                    "a single longer segment.  Reduces line count substantially "
                    "on long walls broken by openings.",
    )
    remove_speckle: bool = Field(
        default=True,
        description="Apply a small morphological OPEN (3-px elliptical kernel) "
                    "before line detection — removes isolated scanner noise + "
                    "tiny furniture stippling without harming wall continuity.  "
                    "Safe for any wall ≥ 3 cm thick at 1 cm/px resolution.  "
                    "Disable for very sparse scans where every pixel counts.",
    )
    multi_elevation: bool = Field(
        default=True,
        description="Combine three horizontal slices (elevation − 0.4 m, "
                    "elevation, elevation + 0.4 m) into a single raster before "
                    "line detection.  Dramatically improves recall: walls that "
                    "are hidden at one height (cubicle stops at 1.4 m, doorway "
                    "header at 2.0 m, knee-wall below a counter) get picked up "
                    "from whichever slice does see them.  Cost: +10–15 s slice "
                    "time.  Disable for the original single-slice behaviour.",
    )
    detect_openings: bool = Field(
        default=True,
        description="After wall detection, scan every kept wall for door-shaped "
                    "gaps (0.7–1.2 m wide, away from wall corners) at a high "
                    "slice (elevation + 0.6 m) so closed door slabs disappear "
                    "and only the gap remains.  Emits them on the OPENINGS layer.",
    )
    detect_columns: bool = Field(
        default=True,
        description="After wall detection, find isolated square-ish blobs in the "
                    "raster (20–120 cm bounding diagonal, ≤ 2.5:1 aspect, ≥ 55% "
                    "fill) that aren't explained by any wall — these are usually "
                    "structural columns.  Emits them as 4-vertex rectangle "
                    "footprints on the COLUMNS layer.",
    )

    # ── Accuracy-track parameters (MacBook MVP — see ACCURACY_TO_CAD_QUALITY_PLAN.md § 0.5) ──

    vertical_surfaces_only: bool = Field(
        default=True,
        description="Before slicing, estimate per-point normals and keep only "
                    "points whose normal is nearly horizontal (|n_z| ≤ 0.30) — "
                    "i.e. points on vertical surfaces.  Removes floor + ceiling "
                    "returns, desk tops, monitor screens (mostly), bookshelf "
                    "tops, and any other roughly horizontal surface from the "
                    "input before the raster is built.  Dramatic noise drop on "
                    "office and lab scans.  Disable for scans where many real "
                    "walls are slanted (industrial / agricultural). Adds ~5–15 s "
                    "to the ingest stage.",
    )
    extract_envelope: bool = Field(
        default=True,
        description="In addition to the main detector pass, extract the building "
                    "shell as a single closed polygon via concave-hull / alpha-"
                    "shape on a floor-level point projection.  Emitted on its "
                    "own layer (WALLS_EXTERIOR) so the deliverable can highlight "
                    "the exterior independently.  Locks priority #1 (external "
                    "walls) even if the rest of the detector path stumbles.",
    )
    use_density_slicer: bool = Field(
        default=True,
        description="Use the vertical-column density slicer (every XY cell gets "
                    "a score = fraction of vertical wall-band heights that have "
                    "any point in them) instead of the legacy binary-occupancy "
                    "slicer.  Walls become bright; furniture stays dim.  Strong "
                    "noise reduction for downstream detection.",
    )
    pair_walls: bool = Field(
        default=True,
        description="After regularization, find pairs of parallel detected "
                    "segments 7–35 cm apart (the two faces of a wall) and "
                    "collapse each pair into a single thickness-aware wall.  "
                    "Emits both faces on the WALLS_FACES DXF layer plus a "
                    "centerline on WALLS.  Single-faced walls keep the median "
                    "thickness from the paired set.  This is the visual lift "
                    "that turns the deliverable from a sketch into CAD.",
    )
    infer_rooms: bool = Field(
        default=True,
        description="After wall pairing, snap nearby endpoints into shared "
                    "junctions, extend T-junctions onto their host walls, and "
                    "enumerate the planar faces of the wall graph.  Each "
                    "interior face becomes a candidate room polygon, emitted "
                    "on the ROOMS DXF layer.  Closes the gaps and antenna "
                    "spikes left by the regularizer and turns walls + rooms "
                    "into editable architectural geometry.",
    )

    # ── v4 (Cloud2BIM-style classical track — see ACCURACY_TO_CAD_QUALITY_PLAN.md § 0.6) ──

    voxel_downsample_m: float = Field(
        default=0.005,
        ge=0.0, le=0.05,
        description="Voxel-downsample the input cloud to this XYZ grid spacing "
                    "BEFORE any other stage.  5 mm is the production default — "
                    "preserves every wall feature ≥ 1 cm while cutting downstream "
                    "runtime by 2-50× on large scans depending on native point "
                    "spacing.  Set 0 to disable (debug only).",
    )
    voxel_downsample_auto: bool = Field(
        default=True,
        description="After the initial voxel pass, re-downsample with a larger "
                    "voxel (up to 10 mm — the accuracy ceiling) if the cloud is "
                    "still above 20 M points.  Dense multi-scan mosaics often "
                    "keep ~80 M points at 5 mm; without this the pipeline can "
                    "thrash for hours.  Escalation never exceeds 10 mm so wall "
                    "features ≥ 1 cm and door frames stay intact.  Disable only "
                    "when debugging a scan that needs native density.",
    )
    use_ceiling_band: bool = Field(
        default=True,
        description="Add a complementary slice at 1.9–2.3 m above the floor — well "
                    "above most furniture, still below ducts and beams.  Combined "
                    "with the density slicer via logical AND to gate the wall mask "
                    "(Cloud2BIM 2025 strategy).  Disable for low-ceiling spaces "
                    "(< 2.4 m clearance) where the band may not exist.",
    )
    ceiling_band_lo_m: float = Field(
        default=1.90,
        ge=0.50, le=4.00,
        description="Bottom of the ceiling-adjacent slice (m above floor).  Should "
                    "clear most office furniture (cubicle tops ~1.5 m, file "
                    "cabinets ~1.8 m).",
    )
    ceiling_band_hi_m: float = Field(
        default=2.30,
        ge=0.50, le=4.00,
        description="Top of the ceiling-adjacent slice (m above floor).  Should "
                    "stay below ducts, beams, and ceiling fixtures (typically "
                    "≥ 2.5 m for commercial, ≥ 2.4 m for residential).",
    )
    use_contour_walls: bool = Field(
        default=True,
        description="Use cv2.findContours + Douglas-Peucker on the binary wall mask "
                    "to extract walls as continuous polylines, instead of Hough/FLD "
                    "line segments.  Eliminates the 'walls fragment into many short "
                    "segments' failure mode (Cloud2BIM § 2.7).  Disable to fall back "
                    "to the v3 detector path for A/B comparison.",
    )
    contour_simplify_eps_m: float = Field(
        default=0.02,
        ge=0.005, le=0.10,
        description="Douglas-Peucker tolerance for contour-based wall polylines.  "
                    "2 cm is the Cloud2BIM 2025 default; below it produces jagged "
                    "polylines, above it merges corners.",
    )
    snapping_distance_m: float = Field(
        default=0.85,
        ge=0.05, le=1.50,
        description="Cloud2BIM / Phase-3 corner join: extend each wall along "
                    "its own axis (and trim overshoots) until it meets a host "
                    "or L-partner, but only if the gap is ≤ this distance.  "
                    "0.85 m closes typical 10–80 cm undershoots on real scans "
                    "without bridging most door leaves (~0.9 m).  Drop toward "
                    "0.60 m if doors start closing; raise toward 1.0 m for "
                    "messy contour corners.",
    )
    # ── Phase 4 (robustness + trust) ──

    gravity_relevel: bool = Field(
        default=True,
        description="After floor detection, rotate the cloud so the detected "
                    "floor plane is exactly level (and at height 0) before "
                    "slicing.  Fixes tilted-scanner scans where a horizontal "
                    "slice catches ceiling at one end of the building and "
                    "desks at the other.  Tilts under 0.5° are ignored; tilts "
                    "over 15° are refused (almost certainly a mis-detected "
                    "ramp) and surfaced as a warning.  Only applies when "
                    "elevation_m is auto.",
    )
    adaptive_slice_band: bool = Field(
        default=True,
        description="Measure the actual floor-to-ceiling clearance from the "
                    "height histogram and scale the slice elevation + ceiling "
                    "band down for low-clearance spaces (data centers, old "
                    "basements) where the defaults would straddle the ceiling. "
                    "Tall rooms keep the defaults unchanged.  Only applies "
                    "when elevation_m is auto.",
    )

    clip_walls_to_envelope: bool = Field(
        default=False,
        description="OPT-IN: drop walls whose midpoint sits outside the building "
                    "envelope + 1 m buffer.  Useful for scans with adjacent "
                    "buildings or window reflections that produce phantom walls "
                    "outside the building.  OFF by default — if the envelope is "
                    "slightly too tight, this will clip legitimate walls and the "
                    "floor plan collapses.  Only enable after confirming the "
                    "envelope is good.",
    )


class VectorizeJobCreate(BaseModel):
    """Response after POSTing a new vectorize job."""
    job_id: str
    status: VectorizeStatus


class VectorizeMetrics(BaseModel):
    """Summary numbers about a completed run — surfaced to the UI."""
    elapsed_s: float
    raw_points: int
    points_in_slab: int
    raster_width_px: int
    raster_height_px: int
    raster_world_width_m: float
    raster_world_height_m: float
    segments_detected: int
    segments_after_regularize: int
    elevation_m: float
    detector: str
    # Coverage diagnostic — fraction of raster foreground pixels within
    # COVERAGE_RADIUS_M of a kept segment.  None on legacy jobs that ran
    # before the diagnostic was added.  Range [0.0, 1.0].
    coverage_pct: Optional[float] = None
    coverage_radius_m: Optional[float] = None
    foreground_px: Optional[int] = None
    uncovered_px: Optional[int] = None
    # Multi-elevation bookkeeping.  ``elevations_used_m`` is the full list of
    # slice elevations (one element when multi_elevation is off, three when on).
    elevations_used_m: Optional[list[float]] = None
    # Phase 4: detected openings (doors / wall-gaps).  Null on legacy jobs.
    openings_detected: Optional[int] = None
    # Phase 6 ("A"): detected columns + per-layer coverage diagnostic.
    columns_detected: Optional[int] = None
    coverage_by_layer: Optional[dict[str, float]] = None
    # MacBook MVP (A4): wall-thickness pairing summary.
    walls_paired: Optional[int] = None              # # of double-line walls
    walls_unpaired: Optional[int] = None            # # of single-face walls
    walls_median_thickness_m: Optional[float] = None
    # MacBook MVP (A5): topology / room inference summary.
    rooms_detected: Optional[int] = None            # # of enclosed rooms found
    junctions_merged: Optional[int] = None          # # of endpoints collapsed
    t_junctions_extended: Optional[int] = None      # # of T-junctions closed
    # Phase 2 continuity: residual gap-close + envelope projection.
    gaps_closed: Optional[int] = None               # collinear degree-1 bridges
    envelope_projections: Optional[int] = None      # dangling ends snapped to hull
    dangling_before: Optional[int] = None           # degree-1 count pre-continuity
    dangling_after: Optional[int] = None            # degree-1 count post-continuity
    envelope_edges_injected: Optional[int] = None   # virtual hull edges in topology
    # v4 (Cloud2BIM-style track): voxel downsample + ceiling-band slice stats.
    points_after_downsample: Optional[int] = None   # # of points after voxel grid
    downsample_voxel_m: Optional[float] = None      # voxel size that was applied
    ceiling_band_used: Optional[bool] = None        # True iff use_ceiling_band on
    ceiling_band_n_points: Optional[int] = None     # # of points in the band
    wall_mask_pixels: Optional[int] = None          # FG pixels after density&ceiling AND
    # v4 contour-based wall extraction: # of contours / polylines emitted.
    contour_walls_detected: Optional[int] = None    # # of wall polylines from contours
    # Phase 4: gravity re-level + adaptive slice band + per-room confidence.
    gravity_relevel_applied: Optional[bool] = None  # True iff cloud was releveled
    gravity_tilt_deg: Optional[float] = None        # detected floor-plane tilt
    ceiling_clearance_m: Optional[float] = None     # floor→ceiling from histogram
    slice_band_adjusted: Optional[bool] = None      # True iff bands were scaled down
    rooms_flagged: Optional[int] = None             # # rooms below confidence threshold


class PipelineWarningPayload(BaseModel):
    """One structured pipeline fallback warning (Phase 4).

    ``code`` is a stable snake_case identifier (see
    ``app.vectorize.pipeline_warnings.KNOWN_CODES``); ``stage`` is the SSE
    stage where the fallback fired.
    """
    code: str
    stage: str
    message: str


class RoomConfidencePayload(BaseModel):
    """Per-room trust score (Phase 4) — surfaced in the editor.

    ``polygon`` is the room ring in world metres so the editor can
    highlight flagged rooms without re-deriving them from segments.
    """
    room_index: int
    area_m2: float
    boundary_coverage: float
    snap_correction_m: float
    confidence: float
    flagged: bool
    polygon: list[list[float]]


class VectorizeJobDetail(BaseModel):
    """Full status payload for one vectorize job."""
    job_id: str
    status: VectorizeStatus
    created_at: str
    updated_at: str
    scan_filename: str
    params: Optional[VectorizeParams] = None
    metrics: Optional[VectorizeMetrics] = None
    error_message: Optional[str] = None
    # Phase 4: structured fallback warnings + per-room confidence from
    # result.json.  Empty lists on legacy jobs.
    warnings: list[PipelineWarningPayload] = []
    room_confidence: list[RoomConfidencePayload] = []

    has_raster: bool = False
    has_overlay: bool = False
    has_dxf: bool = False
    has_coverage: bool = False
    has_rejected: bool = False     # True when ``rejected_segments.json`` exists
    has_sheet: bool = False        # True when ``sheet.svg`` exists (Phase 3)
    has_floor_geometry: bool = False   # True when ``floor_geometry.json`` exists
    has_measurement_report: bool = False  # True when ``measurement_report.json`` exists


class VectorizeReprocessRequest(BaseModel):
    """Body for POST /api/vectorize/{id}/reprocess.

    Re-runs the slice+detect+regularize+write stages on the already-uploaded
    scan with a new set of parameters.  Skips re-uploading.
    """
    params: VectorizeParams


# ── Edit-mode (Phase 3) schemas ──────────────────────────────────────────────

class EditableSegment(BaseModel):
    """One wall segment that the operator can review / edit in the browser.

    World coordinates in metres; ``id`` is stable for the lifetime of one
    pipeline run (re-runs invalidate IDs).  ``layer`` lets us extend to
    openings/columns later without changing the wire format.
    """
    id: str
    layer: str = "walls"
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length_m(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return (dx * dx + dy * dy) ** 0.5


class EditEvent(BaseModel):
    """A single operator action.

    Stored as an append-only log alongside the final segment state — the log
    is what feeds Phase 4 training data ("here's what the model produced,
    here's what the operator changed it to").
    """
    op: str
    # known ops:
    #   reject | restore | delete | move_endpoint | snap_manhattan
    #   add            — operator drew a new wall (new_id, to_xy = endpoint-1,
    #                    from_xy = endpoint-0)
    #   merge          — operator merged ``segment_ids`` into a single new
    #                    segment ``new_id`` with endpoints ``from_xy``→``to_xy``
    #   promote_ghost  — operator rescued a pipeline-rejected ghost
    #                    (``new_id`` is the new active id; ``segment_id`` is
    #                    the ghost's original id from rejected_segments.json)
    segment_id: Optional[str] = None
    segment_ids: Optional[list[str]] = None
    new_id: Optional[str] = None                # for add / merge / promote_ghost
    which_endpoint: Optional[int] = None        # 0 or 1 for move_endpoint
    from_xy: Optional[list[float]] = None
    to_xy: Optional[list[float]] = None
    ts_ms: Optional[int] = None                 # client-side timestamp


class RejectedSegmentPayload(BaseModel):
    """A segment the pipeline dropped during regularization.

    Surfaced to the editor so operators can render them as ghost candidates
    and one-click promote any that were over-aggressively filtered.
    """
    id: str
    layer: str = "walls"
    x1: float
    y1: float
    x2: float
    y2: float
    dropped_by: str        # "short" | "manhattan" | "merged"
    length_m: float
    angle_deg: float


class DuplicatePair(BaseModel):
    """One pair of segments the duplicate-finder thinks could be merged.

    ``merged`` is the suggested centreline.  The editor renders it as a
    preview and either commits (merge), skips (this pair), or dismisses all.
    """
    a_id: str
    b_id: str
    perp_distance_m: float
    angle_diff_deg: float
    endpoint_gap_m: float
    merged_x1: float
    merged_y1: float
    merged_x2: float
    merged_y2: float


class FindDuplicatesRequest(BaseModel):
    """Body for POST /api/vectorize/{id}/find-duplicates.

    Operator can dial thresholds from the UI; defaults are tuned for "the
    pipeline's regularizer would not have caught this".
    """
    perp_distance_m: float = 0.15
    parallel_tol_deg: float = 5.0
    endpoint_gap_m: float = 0.50


class FindDuplicatesResponse(BaseModel):
    suggestions: list[DuplicatePair]


class SaveEditsRequest(BaseModel):
    """Body for POST /api/vectorize/{id}/edits.

    The operator's final cleaned segment list plus the full edit log.  Backend
    persists both, re-emits the DXF from the final segments, and returns the
    versioned DXF URL.
    """
    segments: list[EditableSegment]
    log: list[EditEvent] = []


class SaveEditsResponse(BaseModel):
    job_id: str
    edit_version: int                  # monotonic — increments on every save
    dxf_url: str
    segments_saved: int


class GenerateMeasurementRequest(BaseModel):
    """Body for POST /api/vectorize/{id}/generate-measurement.

    When ``segments`` is provided the server saves them first (same as
    POST /edits), then rebuilds floor geometry + BOMA from that wall
    network.  When omitted, uses the current ``segments.json`` on disk.
    """
    segments: Optional[list[EditableSegment]] = None
    log: list[EditEvent] = []
    save_first: bool = False


class GenerateMeasurementResponse(BaseModel):
    job_id: str
    rooms_detected: int
    has_floor_geometry: bool
    has_measurement_report: bool
    has_sheet: bool
    warnings: list[dict] = []
    n_walls: int = 0
    n_junctions: int = 0
    edit_version: Optional[int] = None


# ── Floor plan sheet (Phase 3) schemas ───────────────────────────────────────

class ManualGridSpec(BaseModel):
    """Operator-entered column grid — the fallback when no columns were
    detected (or the fitted grid is wrong).  Offsets are in metres, in the
    grid frame rotated ``rotation_deg`` from world axes."""
    rotation_deg: float = 0.0
    u_offsets_m: list[float] = []      # numbered lines (1, 2, 3…)
    v_offsets_m: list[float] = []      # lettered lines (A, B, C…)


class SheetMetaSpec(BaseModel):
    """Title block fields the operator can fill in."""
    building_name: str = ""
    address: str = ""
    floor_name: str = ""
    north_angle_deg: float = 0.0


class DoorSwingSpec(BaseModel):
    """Per-opening door swing override — lets the operator flip a door the
    scanner couldn't observe.  ``side`` is relative to the opening
    segment's direction ("left" = +90° CCW)."""
    hinge: str = "start"               # "start" | "end"
    side: str = "left"                 # "left" | "right"


class SheetOverridesRequest(BaseModel):
    """Body for POST /api/vectorize/{id}/sheet/overrides.

    All fields optional: ``labels`` / ``door_swings`` merge per-key (empty
    value clears an override); ``meta`` / ``manual_grid`` / ``style``
    replace wholesale (null removes).  ``style`` is "full" or
    "stevenson_minimal".
    """
    labels: Optional[dict[str, str]] = None
    meta: Optional[SheetMetaSpec] = None
    manual_grid: Optional[ManualGridSpec] = None
    style: Optional[str] = None
    door_swings: Optional[dict[str, DoorSwingSpec]] = None


class SheetOverridesResponse(BaseModel):
    job_id: str
    sheet_url: str
    scale_denominator: int
