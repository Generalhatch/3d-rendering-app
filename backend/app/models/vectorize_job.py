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
                    "gaps in the raster (0.7–1.2 m wide, away from wall corners). "
                    "Emits them as an OPENINGS layer in the DXF and as separate "
                    "segments in the editor.  Conservative — won't false-fire on "
                    "wall terminations, but may miss closed-door scans where the "
                    "door slab fills the raster at slice height.",
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

    has_raster: bool = False
    has_overlay: bool = False
    has_dxf: bool = False
    has_coverage: bool = False
    has_rejected: bool = False     # True when ``rejected_segments.json`` exists


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
