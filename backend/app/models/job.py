"""Pydantic models for API request/response and internal job state."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    aligned = "aligned"
    failed = "failed"
    approved = "approved"


class JobCreate(BaseModel):
    """Response after POSTing /api/jobs."""
    job_id: str
    status: JobStatus


class AlignmentResultSchema(BaseModel):
    transformation: list[list[float]]   # 4x4 as nested list
    residual_rmse: float = 0.0          # metres
    residual_rmse_mm: float = 0.0       # millimetres (convenience)
    confidence: float = 0.0             # 0.0 – 1.0
    confidence_pct: int = 0             # 0 – 100
    inlier_ratio: float = 0.0
    rotation_candidate_used: int = 0
    iterations: int = 0
    num_wall_planes: int = 0
    mode: str = "aligned"               # "aligned" | "scan_only"


class FloorCandidate(BaseModel):
    """A horizontal plane detected in the scan — represents one building level.

    All values are in the scan's local coordinate frame.
    """
    floor_z: float          # elevation of the floor plane along the vertical axis
    inlier_count: int       # RANSAC inlier count (larger = more confident plane)
    wall_score: int         # number of points 0.3–3.0 m above this level (room signal)
    axis_idx: int           # 2 = Z-up, 1 = Y-up


class ReprocessRequest(BaseModel):
    """Body for POST /api/jobs/:id/reprocess (MJ2 + MJ3).

    Skips the expensive scan-loading and merge/downsample stages and re-runs
    only floor detection, wall extraction, plan generation, and room detection
    on the already-merged point cloud.

    Parameters
    ----------
    floor_z : float | None
        If set, overrides automatic floor detection with this elevation.
        Use one of the values from GET /api/jobs/:id/floors to switch floors.
    band_low_m / band_high_m : float
        Wall-band height offsets above the floor.  Defaults match the pipeline.
    min_wall_length_m : float
        Minimum Hough-line segment length kept as a wall segment.
    hough_threshold : int
        Minimum vote count for Hough lines (lower = more lines, more noise).
    """
    floor_z: Optional[float] = None
    band_low_m: float = 0.75
    band_high_m: float = 1.80
    min_wall_length_m: float = 2.0
    hough_threshold: int = 35


class JobDetail(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    scan_filename: str
    scan_filenames: list[str] = []
    plan_filename: str
    error_message: Optional[str] = None
    result: Optional[AlignmentResultSchema] = None
    scan_only: bool = False
    num_rooms: Optional[int] = None
    num_fixtures: Optional[int] = None
    elapsed_s: Optional[float] = None
    plan_bounds: Optional[list[float]] = None
    floor_z: float = 0.0
    floor_candidates: list[FloorCandidate] = []


class ProgressEvent(BaseModel):
    stage: str
    message: str
    progress: float   # 0.0 – 1.0
    job_id: str
