"""DIMENSION entities for the DXF deliverable, from measured wall faces.

The reference CAD sheets carry linear dimensions along the building
perimeter.  This module appends native DXF ``DIMENSION`` entities (aligned
dimensions) to the vectorize deliverable, measured on the OUTER face of
each exterior wall — the same face the measurement engine uses for
outside-face standards — so the numbers on the drawing are the numbers in
the report.

Only exterior walls are dimensioned by default: partition dimensions
clutter a full-floor sheet, and interior measurements belong to the
measurement report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..geometry.model import FloorGeometry, Wall, WallClass

DIMENSIONS_LAYER = "DIMENSIONS"


def _interior_centroid(floor: FloorGeometry) -> np.ndarray:
    """A point inside the floor — mean of room-boundary vertices, falling
    back to the envelope, then to wall endpoints."""
    pts: list[np.ndarray] = [
        np.asarray(r.boundary, dtype=float) for r in floor.rooms
    ]
    if not pts and floor.envelope is not None:
        pts = [np.asarray(floor.envelope, dtype=float)]
    if not pts:
        pts = [np.asarray(w.centerline, dtype=float) for w in floor.walls]
    if not pts:
        return np.zeros(2)
    return np.vstack(pts).mean(axis=0)


def outer_face(wall: Wall, interior_point: np.ndarray) -> np.ndarray:
    """The wall face farther from the floor interior — the measured face
    for perimeter dimensions.  Returns a ``(2, 2)`` segment."""
    face_a = np.asarray(wall.face_a, dtype=float)
    face_b = np.asarray(wall.face_b, dtype=float)
    da = float(np.linalg.norm(face_a.mean(axis=0) - interior_point))
    db = float(np.linalg.norm(face_b.mean(axis=0) - interior_point))
    return face_a if da >= db else face_b


def add_wall_dimensions(
    msp,
    floor: FloorGeometry,
    offset_m: float = 0.8,
    min_wall_length_m: float = 1.0,
    layer: str = DIMENSIONS_LAYER,
) -> int:
    """Add one aligned DIMENSION per exterior wall to a modelspace.

    The dimension measures the wall's outer face end-to-end and its
    dimension line is placed ``offset_m`` OUTSIDE the building.  Returns
    the number of dimensions added.
    """
    doc = msp.doc
    if layer not in doc.layers:
        doc.layers.add(name=layer, color=8, lineweight=13)

    interior = _interior_centroid(floor)
    n_added = 0
    for wall in floor.walls:
        if wall.wall_class != WallClass.EXTERIOR:
            continue
        face = outer_face(wall, interior)
        p1, p2 = face[0], face[1]
        length = float(np.linalg.norm(p2 - p1))
        if length < min_wall_length_m:
            continue

        # Outward side: perpendicular of p1→p2 pointing away from interior.
        u = (p2 - p1) / length
        n_left = np.array([-u[1], u[0]])   # left of travel direction
        mid = (p1 + p2) / 2.0
        outward = mid - interior
        distance = offset_m if float(np.dot(n_left, outward)) >= 0 else -offset_m

        dim = msp.add_aligned_dim(
            p1=(float(p1[0]), float(p1[1])),
            p2=(float(p2[0]), float(p2[1])),
            distance=distance,
            dxfattribs={"layer": layer},
        )
        dim.render()
        n_added += 1
    return n_added


def add_dimensions_to_dxf(
    dxf_path: str | Path,
    floor: FloorGeometry,
    offset_m: float = 0.8,
    output_path: Optional[str | Path] = None,
) -> int:
    """Open an existing DXF, append exterior-wall dimensions, save.

    Saves in place unless ``output_path`` is given.  Returns the number of
    DIMENSION entities added.
    """
    import ezdxf

    dxf_path = Path(dxf_path)
    doc = ezdxf.readfile(str(dxf_path))
    n = add_wall_dimensions(doc.modelspace(), floor, offset_m=offset_m)
    doc.saveas(str(Path(output_path) if output_path else dxf_path))
    return n
