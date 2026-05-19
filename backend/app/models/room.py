"""Room data model."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class RoomSchema(BaseModel):
    id: str
    label: str
    category: str   # office | bathroom | hallway | common | unknown
    polygon_2d: list[tuple[float, float]]
    centroid: tuple[float, float]
    area_m2: float
    match_quality: Optional[float] = None   # 0.0–1.0
