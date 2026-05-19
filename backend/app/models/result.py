"""Full pipeline result stored to disk and returned by GET /api/jobs/{id}."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from .job import AlignmentResultSchema
from .room import RoomSchema
from .fixture import FixtureSchema


class PipelineResult(BaseModel):
    job_id: str
    alignment: AlignmentResultSchema
    rooms: list[RoomSchema]
    fixtures: list[FixtureSchema]
    floor_z: float                      # floor height in scan coords (for viewer)
    plan_bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    ai_review: Optional[dict] = None
