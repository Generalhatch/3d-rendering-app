"""Fixture (wall protrusion) data model."""
from __future__ import annotations

from pydantic import BaseModel


class FixtureSchema(BaseModel):
    id: str
    wall_id: str
    centroid: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    protrusion_depth_m: float
    width_m: float
    height_m: float
    point_count: int
    confidence: float   # 0.0–1.0
