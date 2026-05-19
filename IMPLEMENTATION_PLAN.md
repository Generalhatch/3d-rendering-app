# Implementation Plan: Scan-to-Plan Alignment MVP

**Companion to:** PRD.md
**Target:** Localhost demo, 4–6 weeks heads-down (8–10 weeks part-time)
**Stack:** Python + FastAPI backend, React + Vite + Three.js frontend, OpenRouter for AI

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  BROWSER  (localhost:5173)                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  React + Vite + Tailwind                             │  │
│  │  ├─ UploadZone        (drag-drop, file validation)   │  │
│  │  ├─ ProcessingPanel   (job status, logs)             │  │
│  │  ├─ ViewerScene       (Three.js, two point clouds)   │  │
│  │  ├─ SnapAnimation     (tween scan into place)        │  │
│  │  ├─ ConfidencePanel   (score, residual, AI summary)  │  │
│  │  └─ ManualControls    (rotate, translate, snap-help) │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────┬──────────────────────────────────────────────┘
               │  HTTP (multipart upload, SSE for progress)
               ▼
┌─────────────────────────────────────────────────────────────┐
│  BACKEND  (localhost:8000, FastAPI + Uvicorn)               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  /api/jobs           POST  (upload, queue alignment) │  │
│  │  /api/jobs/{id}      GET   (status + result)         │  │
│  │  /api/jobs/{id}/sse  GET   (progress stream)         │  │
│  │  /api/jobs/{id}/scan GET   (decimated PLY for viewer)│  │
│  │  /api/jobs/{id}/plan GET   (parsed DXF as GeoJSON)   │  │
│  │  /api/jobs/{id}/rooms GET  (room polygons + labels)  │  │
│  │  /api/jobs/{id}/fixtures GET (detected protrusions)  │  │
│  │  /api/jobs/{id}/ai   POST  (call OpenRouter review)  │  │
│  │  /api/jobs/{id}/manual POST (apply user transform)   │  │
│  │  /api/jobs/{id}/export GET (aligned.dxf + .json)     │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Pipeline modules (Python)                           │  │
│  │  ├─ ingest.py        (laspy, pye57, ezdxf, open3d)   │  │
│  │  ├─ slicing.py       (floor detection, wall band)    │  │
│  │  ├─ segment.py       (RANSAC plane detection)        │  │
│  │  ├─ axes.py          (principal axis extraction)     │  │
│  │  ├─ align.py         (rotation + ICP + ambiguity)    │  │
│  │  ├─ rooms.py         (DXF → polygons + labels +      │  │
│  │  │                    per-room match quality)        │  │
│  │  ├─ fixtures.py      (off-wall clusters → markers)   │  │
│  │  ├─ confidence.py    (residual → 0–100 score)        │  │
│  │  ├─ export.py        (DXF + JSON output)             │  │
│  │  └─ ai_review.py     (OpenRouter vision call)        │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Storage                                             │  │
│  │  ├─ ./data/uploads/{job_id}/  (raw inputs)           │  │
│  │  ├─ ./data/artifacts/{job_id}/(intermediate PLY, etc)│  │
│  │  ├─ ./data/results/{job_id}/  (aligned outputs)      │  │
│  │  └─ jobs.sqlite     (job metadata, audit log)        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
               │
               ▼ (only outbound network call)
┌─────────────────────────────────────────────────────────────┐
│  OPENROUTER API   https://openrouter.ai/api/v1              │
│  Model: anthropic/claude-sonnet-4.6 (or equivalent vision)  │
│  Use: "Looks right?" review of rendered overlay PNG         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack — Decisions and Rationale

### Backend

| Choice | Why |
|---|---|
| **Python 3.11** | Open3D, PDAL, ezdxf, laspy all have first-class Python bindings. The geometry libraries you actually need are here. |
| **FastAPI + Uvicorn** | Async HTTP, automatic OpenAPI docs (helpful for self-documentation), built-in SSE for progress streaming. |
| **Open3D** | RANSAC, ICP, voxel downsampling, point cloud I/O. The single most useful library for this project. |
| **PDAL** | Optional, for cases Open3D chokes on (very large scans, exotic formats). Use only if needed. |
| **ezdxf** | DXF parsing. Mature, well-documented, handles the variants you'll see. |
| **laspy** | LAS/LAZ reading. Standard. |
| **pye57** | E57 reading (Open3D doesn't support E57 natively). |
| **Shapely** | 2D polygon ops for room extraction — `polygonize` reconstructs rooms from loose line segments, point-in-polygon for per-room scan filtering. |
| **scikit-learn** | `DBSCAN` for fixture clustering — robust density-based clustering for the protrusion detection step. (Open3D has DBSCAN too; sklearn is simpler for the small clusters we work with.) |
| **SQLite** | Zero-setup job state. Path to Postgres later is trivial. |

### Frontend

| Choice | Why |
|---|---|
| **React 18 + Vite** | Fast dev loop, you already know it. |
| **TypeScript** | Catches the type mismatches between the API and the UI that will otherwise eat afternoons. |
| **Three.js** (raw, not react-three-fiber) | More control for the snap animation. R3F would also work; pick based on comfort. |
| **Tailwind** | Demo polish in less time. |
| **TWEEN.js** | Drop-in animation tweening for the snap. ~3KB, does exactly what you need. |
| **Zustand** | Lightweight state. Redux is overkill. |

### AI

| Choice | Why |
|---|---|
| **OpenRouter** | You're already using it. One API, swap models without rewriting. |
| **Vision model:** `anthropic/claude-sonnet-4.6` (or `openai/gpt-4o` as fallback) | Vision capability, JSON output mode, decent latency. Both work — A/B test during week 5. |
| **OCR (v2):** same models work for raster plan extraction | Defer to v2. |

### Dev tooling

- `uv` or `pip-tools` for Python dependency management
- `make` for the demo runner (`make demo` spins up backend and frontend)
- `pytest` for backend unit tests on the alignment math
- Pre-commit hooks: `ruff`, `black`, `prettier`

---

## 3. Repository Structure

```
align-mvp/
├── README.md
├── Makefile
├── .env.example
├── docker-compose.yml          # optional, for one-command demo
│
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── config.py
│   │   ├── routes/
│   │   │   ├── jobs.py
│   │   │   └── export.py
│   │   ├── pipeline/
│   │   │   ├── ingest.py
│   │   │   ├── slicing.py         # floor detection + wall band slice
│   │   │   ├── segment.py
│   │   │   ├── axes.py
│   │   │   ├── align.py
│   │   │   ├── rooms.py           # polygon extraction, labels, per-room QA
│   │   │   ├── fixtures.py        # off-wall cluster detection
│   │   │   ├── confidence.py
│   │   │   ├── export.py
│   │   │   └── ai_review.py
│   │   ├── models/
│   │   │   ├── job.py             # SQLAlchemy / Pydantic
│   │   │   ├── room.py            # Room dataclass / Pydantic schema
│   │   │   ├── fixture.py         # Fixture dataclass / schema
│   │   │   └── result.py
│   │   ├── storage.py             # filesystem helpers
│   │   └── sse.py                 # progress streaming
│   ├── tests/
│   │   ├── test_alignment.py
│   │   ├── test_segment.py
│   │   ├── test_slicing.py
│   │   ├── test_rooms.py
│   │   ├── test_fixtures.py
│   │   └── fixtures/              # tiny synthetic test files (the test files, not the wall fixtures!)
│   └── scripts/
│       └── generate_synthetic_test_data.py
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/
│       │   └── client.ts       # typed API wrapper
│       ├── components/
│       │   ├── UploadZone.tsx
│       │   ├── ProcessingPanel.tsx
│       │   ├── ViewerScene.tsx
│       │   ├── ConfidencePanel.tsx
│       │   ├── RoomDetailPanel.tsx  # shown when a room is selected
│       │   ├── FixturePanel.tsx     # shown when a fixture is selected
│       │   └── ManualControls.tsx
│       ├── three/
│       │   ├── scene.ts             # core Three.js setup
│       │   ├── snapAnimation.ts
│       │   ├── pointCloudLoader.ts
│       │   ├── planLoader.ts
│       │   ├── roomOverlay.ts       # colored fills, labels, hover/click
│       │   ├── roomColors.ts        # category → color mapping
│       │   ├── fixtureMarkers.ts    # disks on wall surfaces, hover/click
│       │   └── fixtureColors.ts     # size-based color mapping
│       └── state/
│           └── jobStore.ts          # Zustand
│
└── data/                       # gitignored
    ├── uploads/
    ├── artifacts/
    ├── results/
    └── jobs.sqlite
```

---

## 4. Backend Implementation

### 4.1 Floor detection and wall band slicing

This step runs before alignment. It isolates a clean horizontal band of the scan that contains just the walls — no floor furniture, no ceiling fixtures, no baseboards. The downstream RANSAC plane detection sees a much cleaner signal.

```python
# backend/app/pipeline/slicing.py

from dataclasses import dataclass
import numpy as np
import open3d as o3d

@dataclass
class FloorReference:
    plane_eq: np.ndarray             # (a, b, c, d) for ax+by+cz+d=0
    up_normal: np.ndarray            # unit vector pointing up (away from floor)
    floor_z_estimate: float          # height of floor in scan coords
    inlier_count: int

def detect_floor(scan: o3d.geometry.PointCloud) -> FloorReference:
    """
    Find the dominant horizontal plane with normal pointing up.
    Robust to Z-up vs Y-up — we test against gravity-aligned axes.
    """
    # Downsample for speed during floor search
    scan_ds = scan.voxel_down_sample(0.1)

    # Try planes; pick the one with the lowest centroid AND a near-vertical normal
    candidates = []
    remaining = scan_ds
    for _ in range(5):  # check up to 5 planes
        if len(remaining.points) < 1000:
            break
        plane, inliers = remaining.segment_plane(
            distance_threshold=0.03, ransac_n=3, num_iterations=1000
        )
        inlier_cloud = remaining.select_by_index(inliers)
        normal = np.array(plane[:3])
        normal = normal / np.linalg.norm(normal)
        # Floor candidate must be nearly vertical-normal (within 15 degrees of any axis)
        for axis_idx in [2, 1]:  # try Z-up first, then Y-up
            axis = np.zeros(3); axis[axis_idx] = 1
            cos = abs(np.dot(normal, axis))
            if cos > 0.96:  # ~16 degrees tolerance
                centroid_z = np.asarray(inlier_cloud.points).mean(axis=0)[axis_idx]
                # Flip normal if needed so it points up
                up = axis if np.dot(normal, axis) > 0 else -axis
                candidates.append((centroid_z, plane, up, len(inliers), axis_idx))
                break
        remaining = remaining.select_by_index(inliers, invert=True)

    if not candidates:
        raise ValueError("No horizontal floor plane detected")

    # Pick the candidate with the lowest centroid (the actual floor, not a desk)
    candidates.sort(key=lambda c: c[0])
    floor_z, plane_eq, up, count, axis_idx = candidates[0]
    return FloorReference(
        plane_eq=np.array(plane_eq),
        up_normal=up,
        floor_z_estimate=floor_z,
        inlier_count=count,
    )

def extract_wall_band(
    scan: o3d.geometry.PointCloud,
    floor: FloorReference,
    band_low_m: float = 0.75,      # 0.75m above floor — clears most furniture
    band_high_m: float = 1.80,     # 1.80m above floor — clears most ceiling features
) -> o3d.geometry.PointCloud:
    """
    Extract the horizontal slice between band_low_m and band_high_m above the floor.
    This is what gets fed into RANSAC for wall detection and alignment.
    Defaults give a ~3.4-foot band at roughly chest height.
    """
    pts = np.asarray(scan.points)
    # Height above floor for each point
    heights = (pts - pts.mean(axis=0)) @ floor.up_normal + floor.floor_z_estimate
    heights = pts @ floor.up_normal  # signed distance along up axis
    floor_h = floor.floor_z_estimate

    in_band = (heights >= floor_h + band_low_m) & (heights <= floor_h + band_high_m)
    sliced = scan.select_by_index(np.where(in_band)[0])
    return sliced
```

The constants `band_low_m` and `band_high_m` are exposed as job parameters in the API so a technician can widen the band for high-ceiling spaces or narrow it for low-ceiling ones.

### 4.2 The alignment pipeline (the core IP)

This is the heart of the project. Everything else is plumbing around it.

```python
# backend/app/pipeline/align.py

from dataclasses import dataclass
import numpy as np
import open3d as o3d

@dataclass
class AlignmentResult:
    transformation: np.ndarray       # 4x4 matrix, scan → plan frame
    residual_rmse: float             # millimeters
    confidence: float                # 0.0 – 1.0
    inlier_ratio: float
    rotation_candidate_used: int     # which of 0/90/180/270
    iterations: int

def align_scan_to_plan(
    scan_pcd: o3d.geometry.PointCloud,
    plan_lines_2d: np.ndarray,       # (N, 2, 2): N line segments
    voxel_size: float = 0.05,        # 5cm voxels
    band_low_m: float = 0.75,
    band_high_m: float = 1.80,
) -> AlignmentResult:
    """
    Deterministic scan-to-plan alignment.
    Pipeline: floor detect → wall band slice → downsample → segment walls →
              principal axes → rotation + translation → ICP → ambiguity resolution.
    """
    from .slicing import detect_floor, extract_wall_band

    # 1. Detect floor and extract the clean wall band
    floor = detect_floor(scan_pcd)
    wall_band = extract_wall_band(scan_pcd, floor, band_low_m, band_high_m)

    # 2. Downsample the wall band for speed
    scan_ds = wall_band.voxel_down_sample(voxel_size)

    # 3. Extract wall planes via iterative RANSAC on the clean slice
    wall_planes = extract_wall_planes(scan_ds, max_planes=30)

    # 4. Compute scan's principal horizontal axes
    scan_axes = principal_axes_from_walls(wall_planes)

    # 5. Compute plan's principal axes from DXF line angles
    plan_axes = principal_axes_from_lines(plan_lines_2d)

    # 6. Initial rotation: align scan_axes to plan_axes
    R_init = rotation_between_axes(scan_axes, plan_axes)

    # 7. Project both to 2D, find centroid translation
    scan_footprint_2d = project_walls_to_2d(wall_planes, R_init)
    plan_footprint_2d = polylines_to_points(plan_lines_2d)
    t_init = centroid_translation(scan_footprint_2d, plan_footprint_2d)

    # 8. Try 4 rotational ambiguity candidates, pick best
    best = None
    for k in range(4):
        R_test = R_init @ rotation_z(k * np.pi / 2)
        T_test = build_transform(R_test, t_init)

        # ICP refinement
        refined = run_icp_2d(scan_footprint_2d, plan_footprint_2d, T_test)

        if best is None or refined.residual_rmse < best.residual_rmse:
            best = refined
            best.rotation_candidate_used = k

    # 9. Confidence from residual + inlier ratio + plane support
    best.confidence = compute_confidence(best, len(wall_planes))

    return best


def compute_confidence(result, num_wall_planes: int) -> float:
    """
    Heuristic confidence in [0, 1]:
      - residual_rmse: lower is better (0mm → 1.0, 50mm → 0.0)
      - inlier_ratio: higher is better
      - num_wall_planes: more walls = more constraints = higher confidence
    Tune weights on real test data in week 5.
    """
    rmse_score = max(0.0, 1.0 - result.residual_rmse / 0.050)
    inlier_score = result.inlier_ratio
    support_score = min(1.0, num_wall_planes / 12.0)
    return 0.5 * rmse_score + 0.3 * inlier_score + 0.2 * support_score
```

### 4.3 Room extraction (the visual richness layer)

This module turns the DXF into a list of `Room` objects with polygons, labels, and per-room match-quality scores. It runs after alignment so the scan is in the plan's coordinate frame.

```python
# backend/app/pipeline/rooms.py

from dataclasses import dataclass
from typing import Optional
import numpy as np
import ezdxf
from shapely.geometry import Polygon, Point, MultiLineString
from shapely.ops import polygonize, unary_union

@dataclass
class Room:
    id: str                  # "room-001", or DXF handle if available
    label: str               # from TEXT/MTEXT, or auto-generated
    category: str            # "office" | "bathroom" | "hallway" | "common" | "unknown"
    polygon_2d: list[tuple[float, float]]  # closed ring, in plan coordinates
    centroid: tuple[float, float]
    area_m2: float
    match_quality: Optional[float] = None  # 0.0–1.0, filled after alignment

# Heuristic category map. Tune from real DXFs in week 4.
LABEL_PATTERNS = {
    "bathroom": ["rr", "restroom", "bathroom", "toilet", "wc", "lavatory"],
    "hallway":  ["corridor", "hall", "hallway", "passage"],
    "common":   ["lobby", "reception", "lounge", "break", "kitchen", "elev"],
    "office":   ["office", "conf", "meeting", "room"],
}

def extract_rooms(dxf_path: str) -> list[Room]:
    """Two-pass extraction: try closed polylines first, fall back to polygonize."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Pass 1: closed LWPOLYLINE / POLYLINE entities are likely rooms as drawn
    closed_polys = _extract_closed_polylines(msp)

    # Pass 2: if too few rooms found, polygonize from line segments
    if len(closed_polys) < 3:
        closed_polys = _polygonize_from_lines(msp)

    # Extract text entities for labels
    labels = _extract_text_entities(msp)

    rooms = []
    for i, poly in enumerate(closed_polys):
        label = _match_label_to_polygon(poly, labels) or f"Room-{i+1:03d}"
        category = _categorize(label)
        centroid = poly.centroid
        rooms.append(Room(
            id=f"room-{i+1:03d}",
            label=label,
            category=category,
            polygon_2d=list(poly.exterior.coords),
            centroid=(centroid.x, centroid.y),
            area_m2=poly.area,
        ))
    return rooms

def _extract_closed_polylines(msp) -> list[Polygon]:
    polys = []
    for e in msp.query("LWPOLYLINE POLYLINE"):
        if e.closed:
            pts = [(v[0], v[1]) for v in e.get_points()]
            if len(pts) >= 3:
                p = Polygon(pts)
                if p.is_valid and p.area > 1.0:  # filter tiny artifacts
                    polys.append(p)
    return polys

def _polygonize_from_lines(msp) -> list[Polygon]:
    """Reconstruct rooms when walls are individual line segments."""
    lines = []
    for e in msp.query("LINE LWPOLYLINE POLYLINE"):
        if e.dxftype() == "LINE":
            lines.append([(e.dxf.start[0], e.dxf.start[1]),
                          (e.dxf.end[0], e.dxf.end[1])])
        else:
            pts = [(v[0], v[1]) for v in e.get_points()]
            for a, b in zip(pts, pts[1:]):
                lines.append([a, b])
    merged = unary_union(MultiLineString(lines))
    polys = [p for p in polygonize(merged) if p.area > 1.0]
    return polys

def _extract_text_entities(msp) -> list[tuple[str, tuple[float, float]]]:
    labels = []
    for e in msp.query("TEXT MTEXT"):
        text = e.dxf.text if e.dxftype() == "TEXT" else e.text
        pos = e.dxf.insert
        labels.append((text.strip(), (pos[0], pos[1])))
    return labels

def _match_label_to_polygon(poly: Polygon, labels) -> Optional[str]:
    for text, pos in labels:
        if poly.contains(Point(pos)):
            return text
    return None

def _categorize(label: str) -> str:
    lower = label.lower()
    for category, patterns in LABEL_PATTERNS.items():
        if any(p in lower for p in patterns):
            return category
    return "unknown"

def compute_match_quality(
    rooms: list[Room],
    aligned_scan_points: np.ndarray,   # (N, 3) in plan coords
    wall_lines: np.ndarray,            # (M, 2, 2) plan wall segments
    tolerance_m: float = 0.05,
) -> list[Room]:
    """For each room, fraction of in-room scan points within tolerance of a plan wall."""
    from scipy.spatial import cKDTree
    wall_pts = wall_lines.reshape(-1, 2)
    tree = cKDTree(wall_pts)

    for room in rooms:
        poly = Polygon(room.polygon_2d)
        xy = aligned_scan_points[:, :2]
        # Vectorized point-in-polygon would be faster; this is the readable version
        in_room = np.array([poly.contains(Point(p)) for p in xy])
        in_room_pts = xy[in_room]
        if len(in_room_pts) < 50:
            room.match_quality = None  # not enough data
            continue
        dists, _ = tree.query(in_room_pts, k=1)
        room.match_quality = float((dists < tolerance_m).mean())
    return rooms
```

### 4.4 Fixture detection (wall protrusions)

Runs after alignment, using the **full scan** — not just the wall band — so fixtures above or below the slice are still caught. For each detected wall plane, we look at scan points that are slightly *in front* of the plane (2cm–30cm away on the room side), cluster them, and emit a fixture record per cluster.

```python
# backend/app/pipeline/fixtures.py

from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.cluster import DBSCAN

@dataclass
class Fixture:
    id: str                          # "fix-001"
    wall_id: str                     # which wall plane it sits on
    centroid: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    protrusion_depth_m: float        # how far it sticks out from the wall
    width_m: float
    height_m: float
    point_count: int
    confidence: float                # 0.0–1.0; high = dense, well-formed cluster

@dataclass
class WallPlane:
    id: str
    plane_eq: np.ndarray             # (a, b, c, d), normal points INTO the room
    inlier_points: np.ndarray        # (N, 3) points on the wall

def detect_fixtures(
    full_scan: np.ndarray,           # (N, 3) full scan in PLAN coordinates (post-alignment)
    wall_planes: list[WallPlane],
    min_protrusion_m: float = 0.02,  # ignore noise <2cm
    max_protrusion_m: float = 0.30,  # ignore furniture >30cm
    min_cluster_points: int = 50,
    cluster_eps_m: float = 0.05,
) -> list[Fixture]:
    """
    For each wall plane, find clusters of points that lie just in front of the wall.
    Each cluster becomes a Fixture. Determinism: same scan + same walls = same fixtures.
    """
    fixtures = []
    fixture_counter = 0

    for wall in wall_planes:
        a, b, c, d = wall.plane_eq
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # Signed distance of every point to this plane.
        # Positive = on the side the normal points to (room side).
        dists = full_scan @ normal + d / np.linalg.norm([a, b, c])

        # Candidates: points 2cm to 30cm in front of the wall
        in_front_mask = (dists >= min_protrusion_m) & (dists <= max_protrusion_m)
        in_front_pts = full_scan[in_front_mask]
        if len(in_front_pts) < min_cluster_points:
            continue

        # Restrict to points whose projection onto the wall actually falls within
        # the wall's extent (don't pick up fixtures attached to a different wall)
        wall_2d_min = wall.inlier_points[:, :2].min(axis=0) - 0.1
        wall_2d_max = wall.inlier_points[:, :2].max(axis=0) + 0.1
        within_wall_extent = (
            (in_front_pts[:, 0] >= wall_2d_min[0]) &
            (in_front_pts[:, 0] <= wall_2d_max[0]) &
            (in_front_pts[:, 1] >= wall_2d_min[1]) &
            (in_front_pts[:, 1] <= wall_2d_max[1])
        )
        candidate_pts = in_front_pts[within_wall_extent]
        if len(candidate_pts) < min_cluster_points:
            continue

        # Cluster the candidates. DBSCAN is good here: density-based, no need to
        # know cluster count in advance, handles irregular shapes (door trims,
        # electrical panels, etc).
        clustering = DBSCAN(eps=cluster_eps_m, min_samples=min_cluster_points).fit(candidate_pts)
        labels = clustering.labels_

        for label in set(labels):
            if label == -1:  # noise
                continue
            cluster_pts = candidate_pts[labels == label]
            if len(cluster_pts) < min_cluster_points:
                continue

            bbox_min = cluster_pts.min(axis=0)
            bbox_max = cluster_pts.max(axis=0)
            centroid = cluster_pts.mean(axis=0)
            # Protrusion depth = max distance from the wall plane within this cluster
            cluster_dists = cluster_pts @ normal + d / np.linalg.norm([a, b, c])
            protrusion = float(cluster_dists.max())

            extent = bbox_max - bbox_min
            # Width = extent along the wall direction; height = extent along up axis (z)
            width = float(np.linalg.norm(extent[:2]))  # approx; refine in v2
            height = float(extent[2])

            # Confidence from point density + bbox compactness
            volume = max(np.prod(extent), 1e-6)
            density = len(cluster_pts) / volume
            confidence = min(1.0, density / 1000.0)  # tune on real data

            fixture_counter += 1
            fixtures.append(Fixture(
                id=f"fix-{fixture_counter:03d}",
                wall_id=wall.id,
                centroid=tuple(centroid.tolist()),
                bbox_min=tuple(bbox_min.tolist()),
                bbox_max=tuple(bbox_max.tolist()),
                protrusion_depth_m=protrusion,
                width_m=width,
                height_m=height,
                point_count=int(len(cluster_pts)),
                confidence=float(confidence),
            ))

    # Optional: post-filter — drop fixtures that are clearly noise
    # (very thin AND very small, e.g. <5cm in every dimension)
    fixtures = [f for f in fixtures
                if not (f.width_m < 0.05 and f.height_m < 0.05 and f.protrusion_depth_m < 0.03)]
    return fixtures
```

A few notes on the parameters worth highlighting in code review:

- **2cm minimum protrusion** filters scanner noise. Real electrical boxes protrude 4–8cm; door trims 2–4cm; thermostats 2–5cm. Pictures hanging on the wall (~1cm) are intentionally below the threshold and won't show up — that's the right behavior; we don't want every framed photo flagged as a fixture.
- **30cm maximum protrusion** prevents whiteboards on rolling stands, cabinets, and large furniture from being marked as wall fixtures. Things sticking out 30cm+ aren't "on the wall."
- **DBSCAN** with `eps=5cm` and `min_samples=50` reliably separates an electrical panel from an adjacent thermostat. Tune on real data in week 5.
- **The "within wall extent" filter** is the subtle one that matters most: it prevents fixtures on one wall from getting attached to a perpendicular wall that the cluster also happens to be "in front of." Look at a corner — a thermostat on Wall A sits 10cm "in front of" Wall A's plane, but it's also "in front of" Wall B's plane mathematically. The filter says: only count this cluster against Wall A if its projection onto Wall A falls within Wall A's actual physical extent.

### 4.5 OpenRouter integration

```python
# backend/app/pipeline/ai_review.py

import base64
import httpx
from typing import Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

async def review_alignment(
    overlay_png_path: str,
    confidence: float,
    residual_mm: float,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4.6",
) -> dict:
    """
    Call OpenRouter vision model to sanity-check the alignment overlay.
    Returns a dict with `summary`, `concerns` (list), and `model_confidence` (0-1).
    NOT used as a measurement — purely advisory.
    """
    with open(overlay_png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = f"""You are reviewing an automated alignment of a LiDAR scan to a building plan.

The deterministic pipeline reports:
- Confidence: {confidence:.0%}
- Residual error: {residual_mm:.1f}mm

The image shows the scan outline (red) overlaid on the plan (black) after alignment.

Respond with ONLY a JSON object:
{{
  "summary": "one sentence describing the quality of the overlay",
  "concerns": ["list", "of", "specific", "mismatches"],
  "model_confidence": 0.0 to 1.0
}}

Your role is advisory only. Do not output measurements or numeric values
other than the confidence score."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "AlignAI MVP",
            },
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        }},
                    ],
                }],
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        return resp.json()
```

### 4.6 API surface (FastAPI sketch)

```python
# backend/app/routes/jobs.py

from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
import uuid

router = APIRouter(prefix="/api/jobs")

@router.post("")
async def create_job(
    bg: BackgroundTasks,
    plan: UploadFile = File(...),
    scan: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())
    await save_uploads(job_id, plan, scan)
    bg.add_task(run_alignment_pipeline, job_id)
    return {"job_id": job_id, "status": "queued"}

@router.get("/{job_id}")
async def get_job(job_id: str):
    return load_job(job_id)

@router.get("/{job_id}/sse")
async def progress_stream(job_id: str):
    return StreamingResponse(
        sse_progress_generator(job_id),
        media_type="text/event-stream",
    )

@router.post("/{job_id}/manual")
async def apply_manual_transform(job_id: str, transform: list[list[float]]):
    """User drags the scan in the UI; recompute residual + confidence."""
    return apply_manual_correction(job_id, transform)

@router.get("/{job_id}/rooms")
async def get_rooms(job_id: str):
    """Room polygons + labels + match quality, for the viewer overlay."""
    return load_rooms(job_id)

@router.get("/{job_id}/fixtures")
async def get_fixtures(job_id: str):
    """Detected wall protrusions with bounding boxes and depth."""
    return load_fixtures(job_id)

@router.post("/{job_id}/ai")
async def trigger_ai_review(job_id: str):
    return await run_ai_review(job_id)

@router.get("/{job_id}/export")
async def export_aligned(job_id: str):
    return build_export_bundle(job_id)
```

---

## 5. Frontend Implementation

### 5.1 The 3D viewer scene

```typescript
// frontend/src/three/scene.ts

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class AlignmentScene {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;

  scanCloud: THREE.Points | null = null;
  planLines: THREE.LineSegments | null = null;

  // Current scan transformation (animated)
  currentTransform = new THREE.Matrix4();

  constructor(canvas: HTMLCanvasElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf8f9fa);

    this.camera = new THREE.PerspectiveCamera(
      45, canvas.clientWidth / canvas.clientHeight, 0.1, 1000
    );
    this.camera.position.set(20, 20, 20);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    this.controls = new OrbitControls(this.camera, canvas);

    // Top-down default view
    this.camera.up.set(0, 0, 1);
    this.controls.target.set(0, 0, 0);
  }

  loadScan(positions: Float32Array, colors?: Float32Array) {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (colors) geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.02,
      color: colors ? 0xffffff : 0xe11d48,  // rose-600
      vertexColors: !!colors,
    });
    this.scanCloud = new THREE.Points(geom, mat);
    this.scene.add(this.scanCloud);
  }

  loadPlan(segments: Float32Array) {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(segments, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x111827 });  // gray-900
    this.planLines = new THREE.LineSegments(geom, mat);
    this.scene.add(this.planLines);
  }

  setScanTransform(matrix: THREE.Matrix4) {
    if (!this.scanCloud) return;
    this.scanCloud.matrix.copy(matrix);
    this.scanCloud.matrixAutoUpdate = false;
  }

  render = () => {
    requestAnimationFrame(this.render);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
```

### 5.2 The snap animation (the wow moment)

```typescript
// frontend/src/three/snapAnimation.ts

import * as THREE from 'three';
import TWEEN from '@tweenjs/tween.js';

interface SnapParams {
  scene: AlignmentScene;
  fromMatrix: THREE.Matrix4;   // unaligned (identity, usually)
  toMatrix: THREE.Matrix4;     // aligned result from backend
  durationMs?: number;
  onComplete?: () => void;
}

export function playSnapAnimation(p: SnapParams) {
  const duration = p.durationMs ?? 1500;

  // Decompose start and end matrices into position/quaternion/scale
  const startPos = new THREE.Vector3();
  const startQuat = new THREE.Quaternion();
  const startScale = new THREE.Vector3();
  p.fromMatrix.decompose(startPos, startQuat, startScale);

  const endPos = new THREE.Vector3();
  const endQuat = new THREE.Quaternion();
  const endScale = new THREE.Vector3();
  p.toMatrix.decompose(endPos, endQuat, endScale);

  const state = {
    t: 0,
    pos: startPos.clone(),
    quat: startQuat.clone(),
  };

  new TWEEN.Tween(state)
    .to({ t: 1 }, duration)
    .easing(TWEEN.Easing.Cubic.InOut)
    .onUpdate(() => {
      const pos = startPos.clone().lerp(endPos, state.t);
      const quat = startQuat.clone().slerp(endQuat, state.t);
      const m = new THREE.Matrix4().compose(pos, quat, startScale);
      p.scene.setScanTransform(m);
    })
    .onComplete(() => p.onComplete?.())
    .start();

  // TWEEN.update() must be called in the render loop
  function animate(time: number) {
    requestAnimationFrame(animate);
    TWEEN.update(time);
  }
  requestAnimationFrame(animate);
}
```

### 5.3 Room overlay (the visual richness)

```typescript
// frontend/src/three/roomOverlay.ts

import * as THREE from 'three';

export interface Room {
  id: string;
  label: string;
  category: 'office' | 'bathroom' | 'hallway' | 'common' | 'unknown';
  polygon_2d: [number, number][];
  centroid: [number, number];
  match_quality: number | null;
}

const CATEGORY_COLORS: Record<string, number> = {
  office:   0x3b82f6,  // blue-500
  bathroom: 0x14b8a6,  // teal-500
  hallway:  0xeab308,  // yellow-500
  common:   0xa855f7,  // purple-500
  unknown:  0x94a3b8,  // slate-400
};

export class RoomOverlay {
  group = new THREE.Group();
  meshes = new Map<string, THREE.Mesh>();
  labels = new Map<string, THREE.Sprite>();
  selected: string | null = null;

  constructor(rooms: Room[], floorZ: number = 0) {
    for (const room of rooms) {
      this.addRoom(room, floorZ);
    }
  }

  private addRoom(room: Room, z: number) {
    // Polygon fill — extrude slightly above floor so it renders cleanly
    const shape = new THREE.Shape(
      room.polygon_2d.map(([x, y]) => new THREE.Vector2(x, y))
    );
    const geom = new THREE.ShapeGeometry(shape);
    geom.translate(0, 0, z + 0.01);
    const mat = new THREE.MeshBasicMaterial({
      color: CATEGORY_COLORS[room.category] ?? CATEGORY_COLORS.unknown,
      transparent: true,
      opacity: 0.25,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData.roomId = room.id;
    this.meshes.set(room.id, mesh);
    this.group.add(mesh);

    // Billboard label
    const sprite = makeTextSprite(room.label);
    sprite.position.set(room.centroid[0], room.centroid[1], z + 0.5);
    this.labels.set(room.id, sprite);
    this.group.add(sprite);
  }

  highlight(roomId: string | null) {
    // Deselect previous
    if (this.selected) {
      const prev = this.meshes.get(this.selected);
      if (prev) (prev.material as THREE.MeshBasicMaterial).opacity = 0.25;
    }
    // Select new
    if (roomId) {
      const mesh = this.meshes.get(roomId);
      if (mesh) (mesh.material as THREE.MeshBasicMaterial).opacity = 0.55;
    }
    this.selected = roomId;
  }

  pickFromRaycast(raycaster: THREE.Raycaster): Room['id'] | null {
    const hits = raycaster.intersectObjects([...this.meshes.values()]);
    return hits[0]?.object.userData.roomId ?? null;
  }
}

function makeTextSprite(text: string): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgba(17, 24, 39, 0.85)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'white';
  ctx.font = '24px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(2.5, 0.625, 1);
  return sprite;
}
```

### 5.4 Fixture markers

Small colored disks rendered on each wall at fixture positions. Disk size encodes the fixture's bounding box; color encodes protrusion depth (deeper = more red).

```typescript
// frontend/src/three/fixtureMarkers.ts

import * as THREE from 'three';

export interface Fixture {
  id: string;
  wall_id: string;
  centroid: [number, number, number];
  bbox_min: [number, number, number];
  bbox_max: [number, number, number];
  protrusion_depth_m: number;
  width_m: number;
  height_m: number;
  point_count: number;
  confidence: number;
}

function depthToColor(depthMeters: number): THREE.Color {
  // 2cm → teal, 10cm → amber, 30cm → red
  const t = Math.min(1, Math.max(0, (depthMeters - 0.02) / 0.28));
  return new THREE.Color().setHSL(0.5 - 0.5 * t, 0.7, 0.55);
}

export class FixtureMarkers {
  group = new THREE.Group();
  meshes = new Map<string, THREE.Mesh>();
  visible = true;

  constructor(fixtures: Fixture[]) {
    for (const fx of fixtures) {
      this.addMarker(fx);
    }
  }

  private addMarker(fx: Fixture) {
    // Radius scaled to bbox extent, clamped so very small fixtures stay visible
    const extent = Math.max(fx.width_m, fx.height_m);
    const radius = Math.max(0.08, Math.min(0.25, extent / 2));

    const geom = new THREE.SphereGeometry(radius, 16, 12);
    const mat = new THREE.MeshBasicMaterial({
      color: depthToColor(fx.protrusion_depth_m),
      transparent: true,
      opacity: 0.85,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(...fx.centroid);
    mesh.userData.fixtureId = fx.id;
    mesh.userData.fixture = fx;

    this.meshes.set(fx.id, mesh);
    this.group.add(mesh);
  }

  setVisible(visible: boolean) {
    this.visible = visible;
    this.group.visible = visible;
  }

  pickFromRaycast(raycaster: THREE.Raycaster): Fixture | null {
    const hits = raycaster.intersectObjects([...this.meshes.values()]);
    return hits[0]?.object.userData.fixture ?? null;
  }
}
```

The marker color scheme is a quiet UX win: a glance tells the technician whether the wall has shallow trim work, mid-depth boxes, or deep protrusions, without having to click anything.

### 5.5 The main app flow

```typescript
// frontend/src/App.tsx (sketch)

function App() {
  const [job, setJob] = useState<Job | null>(null);
  const sceneRef = useRef<AlignmentScene | null>(null);

  const onFilesDropped = async (planFile: File, scanFile: File) => {
    const { job_id } = await api.createJob(planFile, scanFile);
    setJob({ id: job_id, status: 'processing' });

    // Subscribe to SSE progress
    const sse = api.subscribeProgress(job_id, (event) => {
      // Update progress UI
      if (event.type === 'complete') {
        loadResultIntoScene(job_id);
      }
    });
  };

  const loadResultIntoScene = async (jobId: string) => {
    const result = await api.getJob(jobId);
    const scanPoints = await api.getScanDecimated(jobId);   // PLY blob
    const planSegments = await api.getPlanGeoJSON(jobId);
    const rooms = await api.getRooms(jobId);                // Room[]
    const fixtures = await api.getFixtures(jobId);          // Fixture[]

    sceneRef.current!.loadScan(scanPoints);
    sceneRef.current!.loadPlan(planSegments);

    // Start at identity, snap to aligned matrix
    const targetMatrix = new THREE.Matrix4().fromArray(result.transformation.flat());
    playSnapAnimation({
      scene: sceneRef.current!,
      fromMatrix: new THREE.Matrix4(),
      toMatrix: targetMatrix,
      durationMs: 1500,
      onComplete: () => {
        // After the snap settles, fade in the room overlay
        const overlay = new RoomOverlay(rooms);
        sceneRef.current!.addOverlay(overlay);
        sceneRef.current!.bindRoomPicking(overlay, setSelectedRoom);

        // Then fade in fixture markers (slight delay so the eye absorbs rooms first)
        setTimeout(() => {
          const markers = new FixtureMarkers(fixtures);
          sceneRef.current!.addOverlay(markers);
          sceneRef.current!.bindFixturePicking(markers, setSelectedFixture);
        }, 600);
      },
    });

    setJob({ ...job, status: 'aligned', result, rooms, fixtures });
  };

  return (
    <div className="grid grid-cols-[1fr_400px] h-screen">
      <ViewerCanvas onReady={(s) => (sceneRef.current = s)} />
      <SidePanel
        job={job}
        selectedRoom={selectedRoom}
        selectedFixture={selectedFixture}
        onUpload={onFilesDropped}
      />
    </div>
  );
}
```

---

## 6. Week-by-Week Plan (heads-down full-time)

Timeline: **7–8 weeks** heads-down. Slicing adds ~2 days to week 2. Fixture detection adds ~3–4 days, slotted into week 5. Rooms remain in week 4. Demo polish and AI review shift to weeks 6 and 7. Week 8 is integration + client test data + demo prep.

### Week 1 — Foundations
- [ ] Repo scaffolding (backend + frontend, Makefile, .env.example, README skeleton)
- [ ] Backend: FastAPI hello world, SQLite, file upload endpoint
- [ ] Frontend: Vite + React + Tailwind, drag-drop upload component
- [ ] Backend: DXF parsing (ezdxf), produce GeoJSON-like line segments
- [ ] Backend: Point cloud loading (laspy for LAS, Open3D for PLY, pye57 for E57)
- [ ] Backend: Voxel downsampling, produce a smaller PLY for the frontend viewer
- [ ] Frontend: Three.js scene with hardcoded point cloud + lines
- [ ] **Inspect 2–3 real DXFs from the client** to determine: are rooms closed polylines (easy) or loose line segments (need polygonize)? This decision affects week 4 estimate.
- [ ] **Confirm up-axis convention** (Z-up vs Y-up) on at least one real scan. Affects slicing logic.
- [ ] **Milestone:** Drag a plan and a scan in, see them in the 3D viewer (unaligned, side by side or overlaid in their raw positions). No alignment yet.

### Week 2 — Slicing + the geometric core
- [ ] Backend: `slicing.py` — floor plane detection with Z-up/Y-up robustness
- [ ] Backend: `slicing.py` — wall band extraction (configurable low/high bounds)
- [ ] RANSAC plane detection on the wall band (Open3D `segment_plane` in a loop)
- [ ] Vertical-plane filtering (normal z-component near zero, height range)
- [ ] Principal axis extraction from wall normals (angle histogram, peak detection)
- [ ] DXF line angle histogram → plan's principal axes
- [ ] Rotation alignment + centroid translation
- [ ] ICP refinement (`o3d.pipelines.registration.registration_icp`)
- [ ] 4-way ambiguity resolution loop
- [ ] Unit tests with synthetic data: generate a cluttered scan (walls + tables + ceiling fixtures), verify that slicing extracts a clean band and alignment recovers a known transformation
- [ ] **Milestone:** `pytest backend/tests/test_alignment.py` and `test_slicing.py` pass. Run on a real cluttered test file produces a clean transformation matrix.

### Week 3 — Wiring it together
- [ ] Background task runner (FastAPI BackgroundTasks for now)
- [ ] Job state machine in SQLite (queued → processing → aligned → approved)
- [ ] SSE progress stream
- [ ] Confidence scoring heuristic
- [ ] Export: aligned DXF + JSON metadata
- [ ] Frontend: API client (typed)
- [ ] Frontend: Apply backend transformation matrix to the scan in the scene (no animation yet — just snap it instantly)
- [ ] **Milestone:** Upload → see the aligned result (instant). Numbers panel shows confidence and residual.

### Week 4 — Room extraction + visualization
- [ ] Backend: `rooms.py` — closed-polyline extraction path
- [ ] Backend: `rooms.py` — `polygonize` fallback for line-segment DXFs (if needed based on week 1 inspection)
- [ ] Backend: TEXT/MTEXT label association with polygons
- [ ] Backend: Room categorization heuristic from labels
- [ ] Backend: Per-room match-quality computation (scipy cKDTree + point-in-polygon)
- [ ] Backend: `/api/jobs/{id}/rooms` endpoint
- [ ] Frontend: `RoomOverlay` class — colored fills, billboard labels
- [ ] Frontend: Hover detection → tooltip with match quality
- [ ] Frontend: Click detection → camera focus + scan point highlight
- [ ] `test_rooms.py` — unit tests for both extraction paths with synthetic DXFs
- [ ] **Milestone:** Real test DXF produces correctly extracted rooms with labels in the viewer. Click a room → camera frames it.

### Week 5 — Fixture detection
- [ ] Backend: `fixtures.py` — signed-distance filter against wall planes
- [ ] Backend: `fixtures.py` — within-wall-extent filter (so a thermostat on Wall A doesn't get attached to perpendicular Wall B)
- [ ] Backend: DBSCAN clustering with tuned eps and min_samples
- [ ] Backend: Bounding box, centroid, protrusion depth, dimensions per cluster
- [ ] Backend: Confidence scoring per fixture (density + compactness)
- [ ] Backend: `/api/jobs/{id}/fixtures` endpoint
- [ ] Frontend: `FixtureMarkers` class — colored spheres on wall surfaces
- [ ] Frontend: Toggle to show/hide fixtures
- [ ] Frontend: Hover → tooltip with depth, size, point count
- [ ] Frontend: Click → side panel detail with bounding box visualization
- [ ] `test_fixtures.py` — unit tests with synthetic clusters at known locations
- [ ] **Milestone:** Real test scan produces detected fixtures correctly placed on walls in the viewer. False positives <5% on a clean office floor.

### Week 6 — The snap animation + UX polish
- [ ] TWEEN.js integrated, snap animation working smoothly
- [ ] Camera framing — auto-fit to the aligned scene
- [ ] Room overlay fades in after snap completes (not during, to keep the snap clean)
- [ ] Fixture markers fade in ~600ms after rooms (staggered reveal)
- [ ] Opacity slider for the scan
- [ ] Opacity slider for the room overlay
- [ ] Toggle for fixture markers
- [ ] Top-down vs perspective toggle
- [ ] Confidence panel with status colors (green/yellow/red)
- [ ] Per-room match-quality badges in the room detail panel
- [ ] Fixture detail panel (depth, size, confidence)
- [ ] Loading states, error states
- [ ] **Milestone:** The demo flow works end-to-end and feels polished. Show it to one trusted person before the AI step.

### Week 7 — AI review + manual mode
- [ ] Rendered top-down overlay PNG generation (server-side via Open3D or matplotlib) — include rooms AND fixture markers in the rendered image so the vision model sees them
- [ ] OpenRouter integration, JSON response parsing
- [ ] AI review panel in UI (toggleable)
- [ ] Manual nudge mode: drag handles, arrow-key translate, re-evaluate button
- [ ] Apply manual transform → backend recomputes residual, confidence, per-room match quality, AND fixture positions (fixtures move with the scan, since they're in scan coords)
- [ ] **Milestone:** Both happy path AND low-confidence path work cleanly.

### Week 8 — Demo prep + safety nets
- [ ] Test on 5+ real (plan, scan) pairs from the client
- [ ] Tune confidence weights, match-quality thresholds, and fixture detection parameters on real data
- [ ] Test on at least one intentionally cluttered scan (offices with desks, conference rooms with chairs) to verify slicing does its job
- [ ] README: full setup instructions, troubleshooting, demo script
- [ ] Makefile: `make demo` brings up backend and frontend
- [ ] Pre-load 2–3 demo files into a "Try sample" picker for the meeting
- [ ] Record a backup screencast of the demo working, in case live demo breaks
- [ ] **Milestone:** Clean laptop → `make demo` → working app in under 5 minutes.

### If part-time, multiply by ~1.7
A 7–8 week full-time plan stretches to 12–14 weeks at 15–20 hours/week. Wedding on Sept 5 is the dominant constraint — recommend not committing to a demo date before mid-November if going part-time, or scoping the demo specifically for the window after the wedding.

---

## 7. OpenRouter Integration Details

### API key handling
- Store in `.env` (gitignored): `OPENROUTER_API_KEY=sk-or-v1-...`
- Backend reads via `pydantic-settings`.
- Frontend never sees the key — all AI calls proxied through the backend.

### Model selection
- Default: `anthropic/claude-sonnet-4.6` for vision review
- Fallback: `openai/gpt-4o` (set via `AI_REVIEW_MODEL` env var)
- For OCR in v2: same models support image input

### Cost envelope for MVP demo
- ~1 vision call per alignment, ~1500 input tokens (prompt + image), ~200 output tokens
- At Sonnet pricing roughly $0.01–$0.03 per call
- Demo session of 20 alignments: under $1
- Not a budget concern at MVP scale

### Failure handling
- If the OpenRouter call fails or times out (>30s), the main flow continues without the AI summary
- AI summary is rendered separately and never blocks approval

---

## 8. Test Data Strategy

This is the single biggest project risk. Before Week 1:

1. Sign an NDA with the client and request:
   - 5–10 representative (plan DXF, scan LAS/PLY/E57) pairs
   - For each: the alignment their senior technician arrived at (the "ground truth")
   - At least one example of each failure mode they encounter

2. Build a synthetic test data generator (`scripts/generate_synthetic_test_data.py`) for unit tests:
   - Generate a known plan (a rectangle with internal walls)
   - Generate a scan by sampling points on the walls, adding Gaussian noise
   - Apply a known rotation + translation to the scan
   - The pipeline should recover the inverse transformation to within tolerance

3. Reserve 2 of the real pairs as a held-out test set — don't tune confidence weights on these.

---

## 9. Local Setup Instructions (for the README)

```bash
# Prerequisites: Python 3.11+, Node 20+, system PDAL (optional)

git clone <repo>
cd align-mvp
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY

# One command demo
make demo
# This runs:
#   - backend/.venv setup + pip install
#   - frontend/node_modules install
#   - backend on :8000, frontend on :5173
#   - Opens http://localhost:5173

# Or run pieces separately:
make backend
make frontend
```

---

## 10. Demo Script (for the client meeting)

**Setting:** Conference room. Laptop on the table, plugged into the projector. Backend already running. Browser showing `localhost:5173`.

**Beat 1 — Frame the problem (30s).**
> "Right now your team takes 20–45 minutes per building to align a scan to a plan. I want to show you what that looks like in under 30 seconds. This is running locally on my laptop — no cloud, no setup."

**Beat 2 — The upload (10s).**
Drag `demo_plan.dxf` and `demo_scan.las` from the desktop into the upload zone. Files appear with their names and sizes. Hit "Align."

**Beat 3 — Processing (20–25s).**
The progress bar moves through readable stages: "Reading scan..." → "Detecting walls..." → "Computing alignment..." → "Refining..." → "Extracting rooms..."

**Beat 4 — The snap (1.5s).**
The scan rotates and slides into position over the plan. _Pause here_. Let them see it.

**Beat 5 — The rooms reveal (2s).**
The colored room overlays fade in on top of the aligned floor. Conference rooms in blue, bathrooms in teal, hallways in yellow, common areas in purple. Each room labeled with its name pulled directly from their DXF.
> "And we already understand the floor — every room recognized, labeled, and scored for how well the scan matches the plan in that specific room. Hover any room to see its match quality."
Hover over a conference room: "98% — clean alignment." Hover over a bathroom: "94%." This is the moment they realize this isn't just a registration tool, it's the foundation of a per-room data product.

**Beat 6 — The fixtures reveal (2s).**
Small colored disks fade in on the walls — dozens of them. Electrical panels, outlet boxes, door trims, thermostats, fire extinguisher cabinets.
> "And every protrusion on every wall. The contractor added an electrical panel here last quarter — see it? It's right where they installed it, marked automatically. The system used a clean horizontal slice of the scan at chest height for the alignment math, so the tables in conference rooms didn't throw it off, but it still catches every fixture sticking out of any wall."
Hover over a fixture: "8cm protrusion · 0.4m × 0.3m · electrical box." Click to drill in — the side panel shows the bounding box and confidence.

**Beat 7 — The numbers.**
> "97% overall confidence. 6mm residual error. The math is fully deterministic — same input always gives the same answer, which is going to matter when this evidence ends up in a courtroom."

**Beat 8 — Click into a room.**
Click on the conference room. Camera smoothly frames it, scan points within the room highlight, fixtures inside the room emphasize.
> "Today this is visual. In v2, every one of these rooms gets sub-centimeter dimensional measurements, every fixture gets classified — electrical panel vs door trim vs thermostat — and the whole thing gets a forensic chain of custody. But you needed to see the foundation working first."

**Beat 9 — The manual mode.**
> "When confidence is lower, the technician steps in. Let me show you."
Load the second test pair, which is intentionally a harder case. Confidence comes back at 72%. Click manual mode, nudge the scan slightly, hit "Re-evaluate." Confidence climbs to 94%, and watch the per-room match scores and fixture positions update live.

**Beat 10 — The AI review.**
> "And we have an optional second-opinion layer. It's not making measurements — it's just describing what it sees in plain English."
Show the AI summary: _"Aligned well across north and east walls. Slight overhang on the southwest indicating a possible scanning extension into the adjacent suite. 47 fixtures detected, density consistent with a typical office floor."_

**Beat 11 — The roadmap.**
> "This is v1 — registration plus room understanding plus fixture detection. v2 adds dimensional measurements per room, fixture classification with AI, multi-floor stacking, and the forensic chain of custody. v3 is the path to processing your historic backlog at scale. But first I wanted you to see the foundation working."

**Beat 12 — The ask.**
> "What I'd like to walk away from this meeting with: a signed statement of work for v2, and access to 50 representative buildings from your archive to validate accuracy at scale. I'll have a draft SOW to you by Friday."

**Backup:** Screencast of the same flow on `demo_backup.mp4` in case the live demo trips on something. _Always have the backup._

---

## 11. What This Plan Does Not Cover

To set explicit expectations:

- **No production deployment.** This is localhost only.
- **No authentication.** Anyone with access to the laptop has access to the app.
- **No multi-user.** One job at a time.
- **No per-room dimensional measurements.** v1 *visualizes* rooms with colored fills, labels, and a per-room match-quality indicator. It does not produce room areas, wall lengths, or corner coordinates as quantitative outputs. That's v2.
- **No fixture classification.** v1 *detects and marks* protrusions but does not label them ("electrical panel" vs "door trim" vs "thermostat"). v2 adds AI-driven classification.
- **No accurate fixture dimensions or mounting heights.** v1 shows approximate bounding boxes for visualization only.
- **No forensic certification.** That's v3 and needs a Professional Land Surveyor on the team.
- **No DWG.** DXF only. Add a conversion step or scope a separate week if DWG is required.
- **No raster PDF plans.** Vector DXF only. OCR path is a v2 deliverable.
- **No multi-floor.** One floor per job. Multi-floor stacking is v2.

If any of these become requirements during scoping, the timeline and price both grow.

---

## 12. Pricing Framing (Reference, Not Binding)

For the client conversation, three positions:

- **v1 MVP (this document):** $35–55K fixed, 7–8 weeks. Localhost demo, single-floor, **full per-room visualization, clean wall-band slicing, and wall-fixture detection with markers**. No forensic posture, no dimensional measurements, no fixture classification.
- **v2:** $60–110K + retainer, 8–12 weeks after v1. Multi-floor stacking, per-room dimensional measurements, corner extraction, AI-driven fixture classification, accurate fixture dimensions, hosted (not just localhost), basic auth, raster PDF support.
- **Forensic + scale v3:** $120–200K + per-building usage fee, 3–6 months. Court-admissible bundles, validated accuracy, path to 200M-asset backlog.

Lead with v1 in the conversation. Don't quote v3 until they've seen v1 work and asked for more.

---

## 13. First Three Actions

If you greenlight this plan:

1. **Today:** Request 5–10 (plan, scan) test pairs from the client under NDA. This is the gating dependency.
2. **This week:** Stand up the empty repo, `.env`, `make demo` that does nothing yet. Commit it. The skeleton being real makes the project real.
3. **Day 1 of Week 1:** Get DXF parsing and LAS reading working with one test pair. Visualize both in Three.js side-by-side. End of Day 1 = "I can see both files."

From there, everything follows the week-by-week plan above.
