"""Canonical floor geometry model shared by all extractors and the
measurement engine.  See :mod:`app.geometry.model`.
"""
from .model import (  # noqa: F401
    BoundaryBasis,
    BoundaryRules,
    BoundaryTarget,
    ColumnFeature,
    FloorGeometry,
    Opening,
    Penetration,
    Room,
    Wall,
    WallClass,
    WallFaceRef,
    offset_ring,
    polygon_area,
    ring_perimeter,
    ring_signed_area,
)
