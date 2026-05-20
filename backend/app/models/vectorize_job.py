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

    All distances are in metres.  Defaults are chosen to work on a typical
    interior LiDAR scan at ~1 cm point spacing.
    """
    elevation_m: Optional[float] = Field(
        default=None,
        description="Absolute elevation of the slice plane in the scan's local "
                    "frame.  If null, the pipeline auto-detects the floor and "
                    "uses floor_z + 1.4 m (chest height — a clean cross-section "
                    "of wall surface above furniture, below ceiling fixtures).",
    )
    slab_thickness_m: float = Field(
        default=0.20,
        ge=0.02, le=2.0,
        description="Vertical thickness of the slab projected to the raster.  "
                    "Thicker → more pixels per wall (denser image, better Hough "
                    "but blurrier corners).  0.10–0.30 m is the sweet spot.",
    )
    resolution_m_per_px: float = Field(
        default=0.01,
        ge=0.002, le=0.10,
        description="World metres per raster pixel.  5 mm/px is high-quality "
                    "but produces large images; 1 cm/px is the practical default.",
    )
    detector: DetectorName = DetectorName.fld
    min_wall_length_m: float = Field(
        default=0.50,
        ge=0.05, le=10.0,
        description="Drop any detected segment shorter than this.",
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
    op: str                # "reject" | "restore" | "move_endpoint" | "snap_manhattan" | "delete"
    segment_id: Optional[str] = None
    segment_ids: Optional[list[str]] = None
    which_endpoint: Optional[int] = None       # 0 or 1 for move_endpoint
    from_xy: Optional[list[float]] = None
    to_xy: Optional[list[float]] = None
    ts_ms: Optional[int] = None                # client-side timestamp


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
