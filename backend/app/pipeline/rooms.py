"""DXF room extraction.

Two-pass approach:
  1. Closed LWPOLYLINE/POLYLINE entities → rooms as drawn.
  2. If too few found, polygonize from loose line segments (Shapely).

Labels are matched by point-in-polygon from TEXT/MTEXT entities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import ezdxf
    from shapely.geometry import Polygon, Point, MultiLineString
    from shapely.ops import polygonize, unary_union
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


@dataclass
class Room:
    id: str
    label: str
    category: str   # office | bathroom | hallway | common | unknown
    polygon_2d: list[tuple[float, float]]
    centroid: tuple[float, float]
    area_m2: float
    match_quality: Optional[float] = None


LABEL_PATTERNS: dict[str, list[str]] = {
    "bathroom":  ["rr", "restroom", "bathroom", "toilet", "wc", "lavatory", "lav"],
    "hallway":   ["corridor", "hall", "hallway", "passage", "egress", "stair"],
    "common":    ["lobby", "reception", "lounge", "break", "kitchen", "café", "cafe",
                  "elev", "elevator", "lift", "mail", "copy", "server"],
    "office":    ["office", "conf", "meeting", "board", "room", "suite"],
}


def extract_rooms(dxf_path: str) -> list[Room]:
    if not _SHAPELY:
        raise ImportError("shapely is required for room extraction. pip install shapely")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    closed_polys = _extract_closed_polylines(msp)
    if len(closed_polys) < 2:
        closed_polys = _polygonize_from_lines(msp)

    labels = _extract_text_entities(msp)

    rooms: list[Room] = []
    for i, poly in enumerate(closed_polys):
        label = _match_label_to_polygon(poly, labels) or f"Room-{i + 1:03d}"
        category = _categorize(label)
        centroid = poly.centroid
        rooms.append(Room(
            id=f"room-{i + 1:03d}",
            label=label,
            category=category,
            polygon_2d=[(x, y) for x, y in poly.exterior.coords],
            centroid=(float(centroid.x), float(centroid.y)),
            area_m2=float(poly.area),
        ))
    return rooms


def extract_wall_lines(dxf_path: str) -> np.ndarray:
    """Return all line segments from the DXF as (N, 2, 2) array."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    segments: list[list] = []
    for e in msp.query("LINE"):
        s = e.dxf.start
        en = e.dxf.end
        segments.append([[s[0], s[1]], [en[0], en[1]]])
    for e in msp.query("LWPOLYLINE POLYLINE"):
        pts = [(v[0], v[1]) for v in e.get_points()]
        for a, b in zip(pts, pts[1:]):
            segments.append([list(a), list(b)])
        if e.closed and len(pts) >= 2:
            segments.append([list(pts[-1]), list(pts[0])])
    if not segments:
        return np.zeros((0, 2, 2))
    return np.array(segments, dtype=np.float64)


def compute_match_quality(
    rooms: list[Room],
    aligned_scan_points: np.ndarray,   # (N, 3) in plan coords
    wall_lines: np.ndarray,            # (M, 2, 2)
    tolerance_m: float = 0.05,
) -> list[Room]:
    """For each room, compute the fraction of in-room scan points near a plan wall."""
    from scipy.spatial import cKDTree

    if len(wall_lines) == 0 or len(aligned_scan_points) == 0:
        return rooms

    wall_pts = wall_lines.reshape(-1, 2)
    tree = cKDTree(wall_pts)
    xy = aligned_scan_points[:, :2]

    for room in rooms:
        try:
            poly = Polygon(room.polygon_2d)
            if not poly.is_valid or poly.area < 0.1:
                continue
            # Vectorised bbox pre-filter
            minx, miny, maxx, maxy = poly.bounds
            bbox_mask = (
                (xy[:, 0] >= minx) & (xy[:, 0] <= maxx) &
                (xy[:, 1] >= miny) & (xy[:, 1] <= maxy)
            )
            candidates = xy[bbox_mask]
            if len(candidates) < 20:
                room.match_quality = None
                continue
            in_room = np.array([poly.contains(Point(p)) for p in candidates])
            in_room_pts = candidates[in_room]
            if len(in_room_pts) < 20:
                room.match_quality = None
                continue
            dists, _ = tree.query(in_room_pts, k=1)
            room.match_quality = float((dists < tolerance_m).mean())
        except Exception:
            room.match_quality = None

    return rooms


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_closed_polylines(msp) -> list:
    from shapely.geometry import Polygon
    polys = []
    for e in msp.query("LWPOLYLINE POLYLINE"):
        if not e.closed:
            continue
        try:
            pts = [(v[0], v[1]) for v in e.get_points()]
            if len(pts) < 3:
                continue
            p = Polygon(pts)
            if p.is_valid and p.area > 0.5:
                polys.append(p)
        except Exception:
            continue
    return polys


def _polygonize_from_lines(msp) -> list:
    from shapely.geometry import Polygon, MultiLineString
    from shapely.ops import polygonize, unary_union
    lines = []
    for e in msp.query("LINE"):
        s, en = e.dxf.start, e.dxf.end
        lines.append([(s[0], s[1]), (en[0], en[1])])
    for e in msp.query("LWPOLYLINE POLYLINE"):
        pts = [(v[0], v[1]) for v in e.get_points()]
        for a, b in zip(pts, pts[1:]):
            lines.append([a, b])
        if e.closed and len(pts) >= 2:
            lines.append([pts[-1], pts[0]])
    if not lines:
        return []
    merged = unary_union(MultiLineString(lines))
    return [p for p in polygonize(merged) if p.area > 0.5]


def _extract_text_entities(msp) -> list[tuple[str, tuple[float, float]]]:
    result = []
    for e in msp.query("TEXT"):
        try:
            text = e.dxf.text.strip()
            pos = e.dxf.insert
            result.append((text, (float(pos[0]), float(pos[1]))))
        except Exception:
            pass
    for e in msp.query("MTEXT"):
        try:
            text = e.text.strip()
            pos = e.dxf.insert
            result.append((text, (float(pos[0]), float(pos[1]))))
        except Exception:
            pass
    return result


def _match_label_to_polygon(poly, labels: list[tuple[str, tuple[float, float]]]) -> Optional[str]:
    from shapely.geometry import Point
    for text, pos in labels:
        try:
            if poly.contains(Point(pos)):
                return text
        except Exception:
            pass
    return None


def _categorize(label: str) -> str:
    lower = label.lower()
    for category, patterns in LABEL_PATTERNS.items():
        if any(p in lower for p in patterns):
            return category
    return "unknown"
