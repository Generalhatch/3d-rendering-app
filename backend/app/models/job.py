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
    residual_rmse: float                # millimeters
    confidence: float                   # 0.0 – 1.0
    confidence_pct: int                 # 0 – 100
    inlier_ratio: float
    rotation_candidate_used: int
    iterations: int
    num_wall_planes: int


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
    num_rooms: Optional[int] = None
    num_fixtures: Optional[int] = None
    elapsed_s: Optional[float] = None


class ProgressEvent(BaseModel):
    stage: str
    message: str
    progress: float   # 0.0 – 1.0
    job_id: str
