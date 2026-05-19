"""Generate a synthetic 2D floor plan from detected wall planes or 2D projection.

Two strategies:
  1. 3D wall-plane strategy (preferred): RANSAC wall planes → project to XY.
     Works well for typical indoor scans with clear planar walls.
  2. 2D projection strategy (fallback): project all mid-height points to XY,
     then run 2D line RANSAC. Works for open-plan spaces, glass walls, and
     any scan where 3D RANSAC can't find vertical planes.

The resulting line segments are used exactly like DXF wall lines:
  - Rendered as the "plan" in the viewer
  - Fed into room extraction (Shapely polygonize)
  - Used for per-room match quality computation

Since the plan IS derived from the scan, the "alignment" is trivial (identity
transform). The real value is: the technician gets a clean 2D floor outline,
room labels (auto-generated), and fixture markers — with no DXF required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .segment import WallPlane
from .slicing import FloorReference

# FloorReference is used as a type hint in generate_rooms_from_scan


@dataclass
class SyntheticPlan:
    segments: np.ndarray          # (N, 2, 2) wall line segments in XY
    wall_count: int
    bounds: tuple[float, float, float, float]   # minx, miny, maxx, maxy


def generate_plan_from_walls(
    wall_planes: list[WallPlane],
    axis_idx: int = 2,
) -> SyntheticPlan:
    """Project each wall plane's inlier points to 2D and fit a line segment.

    Returns line segments in the same format as extract_wall_lines() so the
    rest of the pipeline treats them identically to a DXF plan.

    ``axis_idx`` matches ``FloorReference.axis_idx``: 2 = Z-up (project to XY),
    1 = Y-up (project to XZ).
    """
    segments: list[list] = []

    for wall in wall_planes:
        pts = wall.inlier_points
        if len(pts) < 10:
            continue

        # Project to the correct 2D plane based on the scan's up-axis.
        # Y-up scans have the floor in the XZ plane — projecting to XY gives
        # wrong geometry (the vertical axis leaks into the plan).
        xy = pts[:, :2] if axis_idx == 2 else pts[:, [0, 2]]

        # PCA to find the dominant axis of the wall in 2D
        centroid = xy.mean(axis=0)
        xy_centered = xy - centroid
        cov = xy_centered.T @ xy_centered / len(xy_centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Eigenvector with largest eigenvalue = direction along the wall
        wall_dir = eigvecs[:, np.argmax(eigvals)]

        # Project all points onto the wall direction
        projections = xy_centered @ wall_dir

        # Segment extent: from min to max projection
        t_min, t_max = float(projections.min()), float(projections.max())

        # Reconstruct start and end points
        start = centroid + wall_dir * t_min
        end   = centroid + wall_dir * t_max

        # Only keep segments longer than 20cm (filter noise/tiny planes)
        length = float(np.linalg.norm(end - start))
        if length < 0.20:
            continue

        segments.append([[float(start[0]), float(start[1])],
                          [float(end[0]),   float(end[1])]])

    if not segments:
        raise ValueError(
            "Could not extract any wall segments from scan. "
            "The scan may be too sparse or have no planar walls."
        )

    arr = np.array(segments, dtype=np.float64)
    all_pts = arr.reshape(-1, 2)
    bounds = (
        float(all_pts[:, 0].min()),
        float(all_pts[:, 1].min()),
        float(all_pts[:, 0].max()),
        float(all_pts[:, 1].max()),
    )

    return SyntheticPlan(segments=arr, wall_count=len(segments), bounds=bounds)


def generate_plan_from_projection(
    scan_pcd: o3d.geometry.PointCloud,
    floor: FloorReference,
    grid_res_m: float = 0.10,
    min_wall_length_m: float = 2.00,
    max_lines: int = 120,
) -> SyntheticPlan:
    """Generate floor plan via occupancy grid → edge extraction → Hough Lines.

    Pipeline:
      1. Select densest 2.5 m height slice (robust to wrong floor detection).
      2. DBSCAN cluster filter — keeps only the main building cluster, removes
         outdoor tree points, scanner path artefacts captured through windows.
      3. Project to 2D → binary occupancy image.
      4. morphologyEx CLOSE to bridge gaps, then fill interior holes.
      5. GaussianBlur → Canny edge detection on the filled shape.
      6. HoughLinesP on edges → clean wall segments.
      7. Manhattan-world orientation filter.
    """
    import cv2
    from scipy import ndimage as ndi

    ds = scan_pcd.voxel_down_sample(0.08)
    pts = np.asarray(ds.points)
    axis = floor.axis_idx

    # ── 1. Pick height band ───────────────────────────────────────────────────
    # Chest-height slice (1.0–2.0 m above floor = 3.3–6.6 ft): strictly above
    # desk/cabinet height (~0.9 m), below ceiling, and completely free of floor
    # returns.  This is the cleanest possible cross-section of wall surface.
    floor_z = floor.floor_z_estimate
    mask = (pts[:, axis] > floor_z + 1.00) & (pts[:, axis] < floor_z + 2.00)
    pts_mid = pts[mask]

    if len(pts_mid) < max(500, len(pts) * 0.01):
        # Floor detection may be on the wrong level; fall back to the densest 1 m slice.
        pts_mid = _dense_z_slice(pts, axis, band_width=1.0)

    if len(pts_mid) < 50:
        pts_mid = pts

    # ── 2. DBSCAN: remove isolated outdoor noise clusters ────────────────────
    # Outdoor trees, exterior terrain scanned through windows, or scanner
    # path artefacts form clusters isolated from the main building interior.
    # Cap at 80K pts before clustering: DBSCAN only needs to locate the main
    # cluster's bounding region so sub-sampling is safe, and avoids the
    # O(n·k) blowup where k = neighbor count is very high at eps=1.5m with
    # a dense scan (500K+ pts at 8 cm voxel → >20 min without the cap).
    _DBSCAN_CAP = 80_000
    if len(pts_mid) > 500:
        try:
            if len(pts_mid) > _DBSCAN_CAP:
                rng = np.random.default_rng(42)
                db_idx = rng.choice(len(pts_mid), size=_DBSCAN_CAP, replace=False)
                pts_for_db = pts_mid[db_idx]
            else:
                pts_for_db = pts_mid
            _tmp = o3d.geometry.PointCloud()
            _tmp.points = o3d.utility.Vector3dVector(pts_for_db)
            db_labels = np.array(_tmp.cluster_dbscan(eps=1.5, min_points=10))
            del _tmp
            if db_labels.max() >= 1:   # at least two clusters found
                valid = db_labels[db_labels >= 0]
                if len(valid) > 0:
                    main_lbl = int(np.bincount(valid).argmax())
                    # Derive the main cluster's XY bounding box and use it to
                    # filter the full (un-capped) pts_mid so no data is lost.
                    main_pts = pts_for_db[db_labels == main_lbl]
                    xy_main = main_pts[:, :2] if axis == 2 else main_pts[:, [0, 2]]
                    mn = xy_main.min(axis=0) - 3.0   # 3 m margin for coverage
                    mx = xy_main.max(axis=0) + 3.0
                    xy_mid = pts_mid[:, :2] if axis == 2 else pts_mid[:, [0, 2]]
                    keep = (
                        (xy_mid[:, 0] >= mn[0]) & (xy_mid[:, 0] <= mx[0]) &
                        (xy_mid[:, 1] >= mn[1]) & (xy_mid[:, 1] <= mx[1])
                    )
                    pts_mid = pts_mid[keep]
        except Exception:
            pass  # cluster_dbscan failure is non-fatal

    # ── 3. Project to 2D → occupancy image ───────────────────────────────────
    xy = pts_mid[:, :2] if axis == 2 else pts_mid[:, [0, 2]]

    x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    xi = np.clip(np.floor((xy[:, 0] - x0) / grid_res_m).astype(np.int32), 0, 4095)
    yi = np.clip(np.floor((xy[:, 1] - y0) / grid_res_m).astype(np.int32), 0, 4095)
    w, h = int(xi.max()) + 1, int(yi.max()) + 1

    img = np.zeros((h, w), dtype=np.uint8)
    img[yi, xi] = 255

    # ── 4. Close gaps + fill interior ────────────────────────────────────────
    # morphologyEx CLOSE = dilate then erode in one operation.
    # Bridges gaps in scan coverage (doorways, windows, sparse areas).
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_close, iterations=3)

    # Fill holes: empty rooms become solid → outer boundary = building outline
    img_filled = ndi.binary_fill_holes(img_closed > 0).astype(np.uint8) * 255

    # ── 5. Smooth → Canny edges ───────────────────────────────────────────────
    # GaussianBlur reduces micro-noise at scan point boundaries, which would
    # otherwise produce many short spurious edge fragments in Canny.
    img_smooth = cv2.GaussianBlur(img_filled, ksize=(5, 5), sigmaX=1.5)
    edges = cv2.Canny(img_smooth, threshold1=40, threshold2=120, L2gradient=True)

    # ── 6. Hough on edge image ────────────────────────────────────────────────
    min_px = max(10, int(min_wall_length_m / grid_res_m))
    gap_px = max(5, int(0.60 / grid_res_m))

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,   # 0.5° angular resolution
        threshold=35,         # votes required — lower than before since Gaussian smoothed edges
        minLineLength=min_px,
        maxLineGap=gap_px,
    )

    segments: list[list] = []
    if raw_lines is not None:
        for line in raw_lines:
            x1p, y1p, x2p, y2p = line[0]
            wx1 = float(x1p) * grid_res_m + x0
            wy1 = float(y1p) * grid_res_m + y0
            wx2 = float(x2p) * grid_res_m + x0
            wy2 = float(y2p) * grid_res_m + y0
            if float(np.hypot(wx2 - wx1, wy2 - wy1)) >= min_wall_length_m:
                segments.append([[wx1, wy1], [wx2, wy2]])

    # Cap total count — take longest segments first
    if len(segments) > max_lines:
        segments.sort(key=lambda s: -np.hypot(s[1][0]-s[0][0], s[1][1]-s[0][1]))
        segments = segments[:max_lines]

    # ── 7. Manhattan-world filter ─────────────────────────────────────────────
    if len(segments) >= 4:
        segments = _filter_to_dominant_orientations(segments, tolerance_deg=15.0)

    # ── Fallback: bounding box ────────────────────────────────────────────────
    if not segments:
        bx0, bx1 = float(xy[:, 0].min()), float(xy[:, 0].max())
        by0, by1 = float(xy[:, 1].min()), float(xy[:, 1].max())
        segments = [
            [[bx0, by0], [bx1, by0]], [[bx1, by0], [bx1, by1]],
            [[bx1, by1], [bx0, by1]], [[bx0, by1], [bx0, by0]],
        ]

    arr = np.array(segments, dtype=np.float64)
    all_pts2 = arr.reshape(-1, 2)
    bounds = (
        float(all_pts2[:, 0].min()), float(all_pts2[:, 1].min()),
        float(all_pts2[:, 0].max()), float(all_pts2[:, 1].max()),
    )
    return SyntheticPlan(segments=arr, wall_count=len(segments), bounds=bounds)


def _dense_z_slice(
    pts: np.ndarray,
    axis: int,
    band_width: float = 2.5,
    step: float = 0.10,
) -> np.ndarray:
    """Return the points inside the densest ``band_width``-wide Z window.

    Slides a window of ``band_width`` metres across the full Z range in
    ``step`` increments and returns the points in the window that contains
    the most points.  This reliably lands on the building interior even when
    floor detection placed the reference at outdoor ground level.
    """
    z = pts[:, axis]
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min <= band_width:
        return pts

    best_low = z_min
    best_count = 0
    for z_start in np.arange(z_min, z_max - band_width, step):
        count = int(((z >= z_start) & (z < z_start + band_width)).sum())
        if count > best_count:
            best_count = count
            best_low = float(z_start)

    mask = (z >= best_low) & (z < best_low + band_width)
    return pts[mask]


def _filter_to_dominant_orientations(
    segments: list[list],
    tolerance_deg: float = 20.0,
) -> list[list]:
    """Keep only segments aligned with the two dominant building orientations.

    Professional buildings follow the Manhattan-world assumption: nearly all
    structural walls are either parallel or perpendicular to one main axis.
    This filters out diagonal noise (furniture edges, scanner path artefacts).
    """
    if not segments:
        return segments

    # Compute angle of each segment in [0°, 180°)
    angles: list[float] = []
    for seg in segments:
        dx = seg[1][0] - seg[0][0]
        dy = seg[1][1] - seg[0][1]
        angle = float(np.degrees(np.arctan2(dy, dx))) % 180.0
        angles.append(angle)

    ang_arr = np.array(angles)

    # Histogram of angles (2° bins) — find the dominant angle
    bins = np.arange(0, 181, 2)
    hist, _ = np.histogram(ang_arr, bins=bins)
    peak_bin = int(np.argmax(hist))
    dominant_angle = float(bins[peak_bin] + 1)  # centre of bin

    # Second dominant: perpendicular (+ 90°), also looking ± 15° of that
    perp_angle = (dominant_angle + 90.0) % 180.0

    tol = tolerance_deg
    kept: list[list] = []
    for seg, ang in zip(segments, angles):
        diff1 = min(abs(ang - dominant_angle), 180 - abs(ang - dominant_angle))
        diff2 = min(abs(ang - perp_angle),    180 - abs(ang - perp_angle))
        if diff1 <= tol or diff2 <= tol:
            kept.append(seg)

    # If filtering removed everything, return original (building might be rotated)
    return kept if kept else segments


def generate_building_outline(
    scan_pcd: o3d.geometry.PointCloud,
    floor: FloorReference,
) -> list[list[float]]:
    """Compute the building exterior footprint via alphashape.

    Projects mid-height scan points to 2D (the same band used for room
    detection) and runs ``alphashape.optimizealpha`` to find the tightest
    concave hull that still forms a single polygon.

    Returns a list of ``[x, y]`` coordinate pairs describing the outer
    boundary in the scan's local coordinate frame.  Returns an empty list
    on failure — the caller should treat this as "outline unavailable".
    """
    try:
        import alphashape
        from shapely.geometry import MultiPolygon, Polygon
    except ImportError:
        return []

    pts = np.asarray(scan_pcd.points)
    axis = floor.axis_idx

    mid_pts = _dense_z_slice(pts, axis, band_width=2.5)
    if len(mid_pts) < 50:
        mid_pts = pts

    # Project to 2D — same plane selection as the rest of the pipeline
    xy_full = mid_pts[:, :2] if axis == 2 else mid_pts[:, [0, 2]]

    # Subsample for speed — alphashape is O(n log n) but still slow on 500k pts
    if len(xy_full) > 20_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xy_full), size=20_000, replace=False)
        xy = xy_full[idx]
    else:
        xy = xy_full

    try:
        # Probe a set of practical fixed alphas first — much faster than
        # optimizealpha's binary search (which can take several minutes on
        # 20K pts).  For typical building footprints alpha 0.3–2.0 works well.
        hull = None
        for trial_alpha in [0.5, 1.0, 2.0, 0.25]:
            try:
                candidate = alphashape.alphashape(xy, trial_alpha)
                if (
                    candidate is not None
                    and not candidate.is_empty
                    and candidate.geom_type in ("Polygon", "MultiPolygon")
                ):
                    hull = candidate
                    break
            except Exception:
                continue
        if hull is None:
            # Only run the slow optimizealpha on small clouds where it's fast.
            if len(xy) <= 5_000:
                alpha_opt = alphashape.optimizealpha(xy)
                hull = alphashape.alphashape(xy, alpha_opt)
            else:
                from shapely.geometry import MultiPoint
                hull = MultiPoint([(float(p[0]), float(p[1])) for p in xy]).convex_hull
    except Exception:
        return []

    # Normalise: always work with the exterior of the largest polygon
    if hull is None or hull.is_empty:
        return []

    if isinstance(hull, MultiPolygon):
        hull = max(hull.geoms, key=lambda p: p.area)

    if not isinstance(hull, Polygon) or hull.is_empty:
        return []

    # Return as a closed list of [x, y] pairs (last point == first point)
    coords = list(hull.exterior.coords)
    return [[float(x), float(y)] for x, y in coords]


def generate_rooms_from_scan(
    wall_planes: list[WallPlane],
    synthetic_plan: SyntheticPlan,
    scan_pcd: o3d.geometry.PointCloud | None = None,
    floor: FloorReference | None = None,
) -> list[dict]:
    """Reconstruct room polygons from the 2D scan projection.

    Strategy (in order of preference):
      1. Occupancy-grid flood-fill — derive rooms directly from where the scan
         has points.  Does NOT require wall segments to form closed loops.
         Works for any layout.
      2. Shapely polygonize — faster but only works when segments close perfectly.
      3. Whole-building bounding box — last resort so the UI always shows something.
    """
    # ── Strategy 1: occupancy grid (preferred) ────────────────────────────────
    if scan_pcd is not None and floor is not None:
        rooms = _rooms_from_occupancy(scan_pcd, floor, synthetic_plan)
        if rooms:
            return rooms

    # ── Strategy 2: polygonize from wall segments ─────────────────────────────
    rooms = _rooms_from_polygonize(synthetic_plan)
    if rooms:
        return rooms

    # ── Strategy 3: whole building as one room ────────────────────────────────
    return _building_as_single_room(synthetic_plan)


# ── Room helpers ──────────────────────────────────────────────────────────────

def _classify_room(area_m2: float, eccentricity: float, solidity: float) -> tuple[str, str, float]:
    """Classify a room from shape metrics. Returns (label, category, confidence: 0.0–1.0).

    Uses watershed regionprops values that are already computed for each room:
      eccentricity  — 0=square, 1=very elongated line (>0.85 suggests corridor)
      solidity      — area / convex_hull_area (1=convex, <0.7=highly irregular)
      area_m2       — floor area in square metres

    Confidence reflects how strongly the metrics match the assigned category.
    A score of 0.95 means the room clearly fits; 0.50 means it is a weak match.
    """
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    # ── Hallway / Corridor ────────────────────────────────────────────────────
    # Key signal: high eccentricity (elongated shape).  Area must be reasonable.
    if eccentricity > 0.85 and area_m2 < 40:
        ecc_conf = _clamp01((eccentricity - 0.85) / 0.12)   # 0 at 0.85 → 1 at 0.97
        area_conf = _clamp01(1.0 - area_m2 / 45.0)
        return "Hallway / Corridor", "hallway", _clamp01(0.55 + ecc_conf * 0.38 + area_conf * 0.07)

    # ── Bathroom / Storage ────────────────────────────────────────────────────
    # Key signal: unusually small floor area.
    if area_m2 < 7.0:
        # Score scales from 0.55 (just-under-7 m²) up to 0.92 (≤2 m²)
        conf = _clamp01(0.55 + (7.0 - area_m2) / 14.0)
        return "Bathroom / Storage", "bathroom", conf

    # ── Open Plan / Lobby ─────────────────────────────────────────────────────
    # Key signals: large area AND convex shape.  Both criteria needed.
    if area_m2 > 80.0 and solidity > 0.85:
        area_conf = _clamp01((area_m2 - 80.0) / 120.0)       # 0 at 80 m² → 1 at 200 m²
        sol_conf  = _clamp01((solidity - 0.85) / 0.14)        # 0 at 0.85 → 1 at 0.99
        return "Open Plan / Lobby", "common", _clamp01(0.60 + area_conf * 0.25 + sol_conf * 0.15)

    # ── Conference Room ───────────────────────────────────────────────────────
    # Medium-large area; confidence grows with size above the threshold.
    if area_m2 > 40.0:
        conf = _clamp01(0.55 + (area_m2 - 40.0) / 100.0)     # 0.55 at 40 m² → ~0.90 at 75 m²
        return "Conference Room", "office", conf

    # ── Irregular Space ───────────────────────────────────────────────────────
    # Low solidity indicates a complex, non-convex polygon.
    if solidity < 0.70:
        conf = _clamp01(0.50 + (0.70 - solidity) / 0.60)      # 0.50 at 0.70 → 0.83 at 0.40
        return "Irregular Space", "unknown", conf

    # ── Office / Meeting ─────────────────────────────────────────────────────
    # Default: regular shape, typical office size (8–30 m²).
    area_fit = 1.0 - abs(area_m2 - 18.0) / 25.0              # peak at 18 m²
    sol_fit  = _clamp01((solidity - 0.60) / 0.35)
    conf = _clamp01(0.50 + max(0.0, area_fit) * 0.25 + sol_fit * 0.15)
    return "Office / Meeting", "office", conf


def _rooms_from_occupancy(
    scan_pcd: o3d.geometry.PointCloud,
    floor: FloorReference,
    synthetic_plan: SyntheticPlan,
    cell_size: float = 0.50,
    min_room_area_m2: float = 4.0,
    max_rooms: int = 80,
) -> list[dict]:
    """Derive rooms from 2D point-cloud occupancy via watershed segmentation.

    Replaces the previous flood-fill + connected-components approach with a
    proper watershed algorithm seeded from distance-transform maxima.

    Why watershed is better:
      - The distance transform peaks naturally find the centre of each room
        (the point furthest from any wall surface).
      - Watershed uses those room-centre seeds to grow outward, so touching
        rooms are cleanly separated along the wall boundary — rather than
        being merged into one blob or arbitrarily split by dilation artefacts.
      - No hand-tuned dilation-iteration count is needed to "bridge" gaps;
        watershed handles arbitrary room shapes and sizes automatically.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        from skimage.feature import peak_local_max
        from skimage.morphology import remove_small_holes
        from skimage.segmentation import watershed
        from skimage.measure import label, regionprops, find_contours, approximate_polygon
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.validation import make_valid
    except ImportError:
        return []

    import cv2

    # Downsample before rasterising — extra density adds nothing to the 50 cm
    # occupancy grid but makes _dense_z_slice and numpy ops slow on 18M-pt clouds.
    _scan_ds = (
        scan_pcd if len(scan_pcd.points) <= 400_000
        else scan_pcd.voxel_down_sample(0.15)
    )
    pts = np.asarray(_scan_ds.points)
    axis = floor.axis_idx

    # Chest-height band: 1.0–2.0 m above floor — same reasoning as
    # generate_plan_from_projection.  Removes floor returns, furniture tops,
    # and ceiling clutter that inflate cell occupancy and blur room boundaries.
    fl_z = floor.floor_z_estimate
    chest_mask = (pts[:, axis] > fl_z + 1.00) & (pts[:, axis] < fl_z + 2.00)
    pts_mid = pts[chest_mask]
    if len(pts_mid) < max(100, len(pts) * 0.01):
        pts_mid = _dense_z_slice(pts, axis, band_width=1.0)
    if len(pts_mid) < 100:
        pts_mid = pts

    xy = pts_mid[:, :2] if axis == 2 else pts_mid[:, [0, 2]]
    if len(xy) < 100:
        return []

    x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    xi = np.clip(((xy[:, 0] - x0) / cell_size).astype(np.int32), 0, 4999)
    yi = np.clip(((xy[:, 1] - y0) / cell_size).astype(np.int32), 0, 4999)
    w = int(xi.max()) + 3
    h = int(yi.max()) + 3

    occ = np.zeros((h, w), dtype=np.uint8)
    occ[yi, xi] = 255

    # ── RG-2: Two-phase morphology — close wall gaps without bridging doorways ──
    # Phase 1 — small kernel + few iterations: only fills intra-wall scatter gaps
    # (<0.75 m at 0.5 m/px).  Standard interior doorways (0.9–1.2 m) are NOT
    # bridged, so watershed can still find room seeds on both sides of a doorway.
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    occ_wall_closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k_small, iterations=2)

    # ── RG-9: Selective fill — avoid phantom rooms in courtyards/outdoor gaps ──
    # binary_fill_holes fills ALL enclosed holes, including large outdoor gaps in
    # U-shaped buildings.  remove_small_holes only fills holes smaller than the
    # area_threshold, so exterior courtyards (typically >200 m²) are left open.
    max_fill_px = int(200.0 / (cell_size ** 2))  # 200 m² expressed in grid cells
    occ_filled = remove_small_holes(
        occ_wall_closed > 0,
        area_threshold=max_fill_px,
        connectivity=2,
    ).astype(np.uint8)

    # ── Keep only the largest connected component ─────────────────────────────
    # Isolated clusters (separate buildings, outdoor structures captured through
    # windows or open doorways) appear as valid occupancy but would produce
    # phantom "floating" rooms.  Label all CCs and keep only the largest one.
    cc_labels, num_cc = label(occ_filled, connectivity=2, return_num=True)
    if num_cc > 1:
        cc_sizes = np.bincount(cc_labels.ravel())
        cc_sizes[0] = 0  # ignore background
        occ_filled = (cc_labels == int(cc_sizes.argmax())).astype(np.uint8)

    # ── Burn wall-plan segments as barriers ────────────────────────────────────
    # The Hough-detected wall segments define where rooms divide, but the
    # occupancy grid only knows where points ARE.  Without burning the walls as
    # background pixels, the distance transform has no internal barriers and
    # watershed merges the whole floor into 1–3 giant blobs instead of rooms.
    if len(synthetic_plan.segments) > 0:
        wall_canvas = np.zeros_like(occ_filled, dtype=np.uint8)
        for seg in synthetic_plan.segments:
            px1 = int(np.clip((float(seg[0][0]) - x0) / cell_size, 0, w - 1))
            py1 = int(np.clip((float(seg[0][1]) - y0) / cell_size, 0, h - 1))
            px2 = int(np.clip((float(seg[1][0]) - x0) / cell_size, 0, w - 1))
            py2 = int(np.clip((float(seg[1][1]) - y0) / cell_size, 0, h - 1))
            cv2.line(wall_canvas, (px1, py1), (px2, py2), 1, thickness=1)
        occ_filled = occ_filled * (1 - wall_canvas)

    # Distance transform: each pixel's value = distance to nearest background.
    # Local maxima = room centres (furthest from all walls).
    dist = distance_transform_edt(occ_filled).astype(np.float32)

    # ── RG-3: Adaptive min_distance — stop suppressing small rooms ───────────
    # The old 3 m minimum distance meant any bathroom peak within 3 m of an
    # office peak was suppressed (bathrooms are typically 1–2 m from walls).
    # Fix: use 1.5 m as the primary minimum, then force-include any peak that
    # sits ≥1.5 m from the nearest wall regardless of its proximity to other
    # peaks — these represent genuinely distinct small rooms.
    occupied_mask = occ_filled.astype(bool)

    # Pass 1: standard peaks with 1.5 m separation (captures most rooms)
    min_dist_px = max(3, int(1.5 / cell_size))
    coords = peak_local_max(dist, min_distance=min_dist_px, labels=occupied_mask)

    # Pass 2: any peak ≥1.5 m from all walls, regardless of inter-peak distance.
    # This ensures small rooms (bathrooms, closets) always get a seed even if
    # they happen to be adjacent to a large room.
    all_coords = peak_local_max(dist, min_distance=1, labels=occupied_mask)
    small_room_threshold_px = max(3, int(1.5 / cell_size))
    if len(all_coords) > 0:
        peak_vals = dist[all_coords[:, 0], all_coords[:, 1]]
        forced = all_coords[peak_vals >= small_room_threshold_px]
        if len(forced) > 0:
            coords = np.unique(np.vstack([coords, forced]), axis=0) if len(coords) > 0 else forced

    if len(coords) == 0:
        return []

    # Build marker image — each seed gets a unique integer ID
    markers = np.zeros_like(occ_filled, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # Watershed: grows each seed outward to fill the floor footprint.
    # Using negative distance so peaks (room centres) are the minima for watershed.
    labeled = watershed(-dist, markers, mask=occupied_mask)

    regions = regionprops(labeled)
    if not regions:
        return []

    rooms: list[dict] = []
    for region in sorted(regions, key=lambda r: -r.area):
        area_m2 = float(region.area) * cell_size ** 2
        if area_m2 < min_room_area_m2:
            continue

        region_mask = (labeled == region.label).astype(np.uint8)
        contours = find_contours(region_mask, 0.5)
        if not contours:
            continue

        # Largest contour = outer boundary
        contour = max(contours, key=len)

        # approximate_polygon: geometrically-correct Douglas-Peucker simplification.
        # tolerance=2.0 at cell_size=0.5m means simplify within 1m — clean building geometry.
        tol_px = max(1.0, 1.0 / cell_size)
        contour_simplified = approximate_polygon(contour, tolerance=tol_px)

        polygon = [
            [float(c[1]) * cell_size + x0, float(c[0]) * cell_size + y0]
            for c in contour_simplified
        ]
        polygon.append(polygon[0])  # close

        # ── RG-6: Validate + repair self-intersecting polygons ────────────────
        # Douglas-Peucker simplification can produce crossing edges on complex
        # L-shaped or U-shaped rooms.  make_valid() fixes these before storage.
        try:
            poly_shapely = ShapelyPolygon(polygon)
            if not poly_shapely.is_valid:
                poly_shapely = make_valid(poly_shapely)
            if poly_shapely.geom_type == "MultiPolygon":
                poly_shapely = max(poly_shapely.geoms, key=lambda p: p.area)
            if poly_shapely.geom_type == "Polygon" and not poly_shapely.is_empty:
                polygon = [[float(x), float(y)] for x, y in poly_shapely.exterior.coords]
        except Exception:
            pass  # keep the simplified polygon if Shapely is unavailable

        # ── RG-5: Representative centroid — guaranteed inside the polygon ─────
        # region.centroid is the arithmetic mean of pixel coordinates; for
        # L-shaped or U-shaped rooms it can fall outside the room.
        # Shapely's representative_point() always returns an interior point.
        cy_px, cx_px = region.centroid
        cx = float(cx_px) * cell_size + x0
        cy = float(cy_px) * cell_size + y0
        try:
            rp = ShapelyPolygon(polygon).representative_point()
            cx, cy = float(rp.x), float(rp.y)
        except Exception:
            pass  # fall back to regionprops centroid

        # ── RG-7: Stable room IDs from centroid hash ─────────────────────────
        # Sequential counters like "room-001" change whenever seed order changes,
        # breaking any user annotation that referenced the old ID.
        # A hash of the rounded centroid gives the same room the same ID across
        # multiple pipeline runs as long as the centroid stays within 0.5 m.
        centroid_hash = abs(hash((round(cx, 0), round(cy, 0)))) % 65536
        room_id = f"room-{centroid_hash:04x}"

        label, category, clf_conf = _classify_room(
            area_m2,
            float(region.eccentricity),
            float(region.solidity),
        )

        # Annotate with shape metadata that helps classify room type
        rooms.append({
            "id": room_id,
            "label": label,
            "category": category,
            "polygon_2d": polygon,
            "centroid": [cx, cy],
            "area_m2": area_m2,
            "match_quality": None,
            # Extra shape properties for future room classification
            "eccentricity": float(region.eccentricity),    # 0=square, 1=line (corridor)
            "orientation_deg": float(np.degrees(region.orientation)),
            "solidity": float(region.solidity),             # 1=convex, <0.8=irregular
            # MJ1: classifier confidence — how strongly the metrics fit this category
            "classification_confidence": round(clf_conf, 3),
        })
        if len(rooms) >= max_rooms:
            break

    return rooms


def _rooms_from_polygonize(synthetic_plan: SyntheticPlan) -> list[dict]:
    """Try Shapely polygonize.  Only works when segments form closed loops."""
    try:
        from shapely.geometry import MultiLineString
        from shapely.ops import polygonize, unary_union
    except ImportError:
        return []

    lines = [
        [(s[0][0], s[0][1]), (s[1][0], s[1][1])]
        for s in synthetic_plan.segments.tolist()
    ]
    if not lines:
        return []

    bx0, by0, bx1, by1 = synthetic_plan.bounds
    plan_area = (bx1 - bx0) * (by1 - by0)
    min_room_area = max(2.0, plan_area * 0.005)

    merged = unary_union(MultiLineString(lines))
    polys = sorted(
        [p for p in polygonize(merged) if p.area >= min_room_area],
        key=lambda p: -p.area,
    )
    if not polys:
        return []

    rooms = []
    for i, poly in enumerate(polys):
        # ── RG-11: representative_point() always inside concave polygons ──────
        # poly.centroid can fall outside L-shaped / U-shaped rooms.
        try:
            rp = poly.representative_point()
            cx, cy = float(rp.x), float(rp.y)
        except Exception:
            c = poly.centroid
            cx, cy = float(c.x), float(c.y)
        # Approximate eccentricity from bounding-box aspect ratio
        minx, miny, maxx, maxy = poly.bounds
        w = maxx - minx
        h = maxy - miny
        aspect = max(w, h) / max(min(w, h), 0.001)
        eccentricity_approx = 1.0 - 1.0 / aspect   # 0=square, 1=very elongated
        solidity = float(poly.area / poly.convex_hull.area) if poly.convex_hull.area > 0 else 1.0
        area_m2 = float(poly.area)
        label, category, clf_conf = _classify_room(area_m2, eccentricity_approx, solidity)
        # Stable ID from centroid hash (matches watershed path — RG-7 parity)
        centroid_hash = abs(hash((round(cx, 0), round(cy, 0)))) % 65536
        rooms.append({
            "id": f"room-{centroid_hash:04x}",
            "label": label,
            "category": category,
            "polygon_2d": list(poly.exterior.coords),
            "centroid": [cx, cy],
            "area_m2": area_m2,
            "match_quality": None,
            "eccentricity": eccentricity_approx,
            "solidity": solidity,
            "orientation_deg": 0.0,
            "classification_confidence": round(clf_conf, 3),
        })
    return rooms


def _building_as_single_room(synthetic_plan: SyntheticPlan) -> list[dict]:
    """Return the whole building bounding box as a single 'Building' room."""
    bx0, by0, bx1, by1 = synthetic_plan.bounds
    area = (bx1 - bx0) * (by1 - by0)
    polygon = [
        [bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1], [bx0, by0],
    ]
    return [{
        "id": "room-001",
        "label": "Building",
        "category": "unknown",
        "polygon_2d": polygon,
        "centroid": [float((bx0 + bx1) / 2), float((by0 + by1) / 2)],
        "area_m2": float(area),
        "match_quality": None,
        "classification_confidence": 0.30,   # fallback path — low confidence
    }]
