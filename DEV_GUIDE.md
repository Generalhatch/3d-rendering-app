# Scan-to-Plan — Development Manifesto

> Last updated: May 2026  
> Sources: live docs for every library (Open3D 0.18, OpenCV 4.13, scikit-image 0.26, scipy 1.13/1.17, Shapely 2.1, alphashape 1.3, laspy 2.5, Three.js r168)

This document is the single source of truth for how to tune, extend, and understand every algorithm in the pipeline. Every variable, every library function, and every "we haven't tried this yet" capability is listed here.

---

## Table of Contents

1. [Architecture & Data Flow](#1-architecture--data-flow)
2. [Stage 1 — Ingest & Merge (laspy + Open3D)](#2-stage-1--ingest--merge)
3. [Stage 2 — Floor Detection (Open3D)](#3-stage-2--floor-detection)
4. [Stage 3 — 3D Wall Planes (Open3D RANSAC)](#4-stage-3--3d-wall-planes)
5. [Stage 4 — 2D Floor Plan (OpenCV + scipy)](#5-stage-4--2d-floor-plan)
6. [Stage 5 — Room Extraction (scikit-image + OpenCV)](#6-stage-5--room-extraction)
7. [Stage 6 — Viewer PLY](#7-stage-6--viewer-ply)
8. [Library A: Open3D 0.18 — Full API](#8-library-a-open3d-018--full-api)
9. [Library B: OpenCV 4.13 — Full API](#9-library-b-opencv-413--full-api)
10. [Library C: scikit-image 0.26 — Full API](#10-library-c-scikit-image-026--full-api)
11. [Library D: scipy.ndimage — Full API](#11-library-d-scipyndimage--full-api)
12. [Library E: Shapely 2.1 — Full API](#12-library-e-shapely-21--full-api)
13. [Library F: alphashape 1.3 — Full API](#13-library-f-alphashape-13--full-api)
14. [Library G: laspy 2.5 — Full API](#14-library-g-laspy-25--full-api)
15. [Library H: Three.js (Frontend)](#15-library-h-threejs-frontend)
16. [Tuning Cheat Sheet](#16-tuning-cheat-sheet)
17. [Upgrade Roadmap](#17-upgrade-roadmap)

---

## 1. Architecture & Data Flow

```
LAZ / LAS / PLY / E57
        │
        ▼
 ┌──────────────────────────────────────────────────┐
 │  Stage 1: Ingest & Merge                         │
 │  merge.py + ingest.py                            │
 │  laspy.read() → o3d.voxel_down_sample()          │
 │  → merge_scans(voxel_size=0.03)                  │
 └──────────────────────────┬───────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────┐
 │  Stage 2: Floor Detection                        │
 │  slicing.py                                      │
 │  o3d.segment_plane() × 12 iterations             │
 │  Score by wall content above → best floor        │
 └──────────────────────────┬───────────────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
           Has DXF plan?          Scan-only?
                    │                 │
                    ▼                 ▼
 ┌───────────────────────┐  ┌─────────────────────────┐
 │ Stage 3: 3D RANSAC    │  │ Stage 4: 2D Floor Plan  │
 │ segment.py            │  │ scanplan.py             │
 │ extract_wall_planes() │  │ occupancy grid →        │
 │ normal-based fallback │  │ Canny → HoughLinesP     │
 └──────────┬────────────┘  └──────────┬──────────────┘
            │                          │
            └──────────┬───────────────┘
                       ▼
 ┌──────────────────────────────────────────────────┐
 │  Stage 5: Room Extraction                        │
 │  scanplan.py                                     │
 │  cv2.dilate → binary_fill_holes →                │
 │  sk_measure.label → find_contours                │
 └──────────────────────────┬───────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────┐
 │  Stage 6: Viewer PLY + Export                    │
 │  runner.py / ingest.py                           │
 │  _wall_height_slice() → write_decimated_ply()    │
 └──────────────────────────────────────────────────┘
```

**Files → functions map**

| File | Key functions |
|---|---|
| `runner.py` | `run_pipeline()`, `_wall_height_slice()` |
| `ingest.py` | `_load_las()`, `write_decimated_ply()` |
| `merge.py` | `merge_scans()`, `_scans_share_coordinate_frame()` |
| `slicing.py` | `detect_floor()`, `extract_wall_band()` |
| `segment.py` | `extract_wall_planes()`, `extract_wall_planes_with_fallback()`, `extract_wall_planes_from_full_cloud()` |
| `scanplan.py` | `generate_plan_from_projection()`, `generate_rooms_from_scan()`, `_rooms_from_occupancy()`, `_dense_z_slice()`, `_filter_to_dominant_orientations()` |
| `align.py` | `align_scan_to_plan()` |
| `fixtures.py` | `detect_fixtures()` |

---

## 2. Stage 1 — Ingest & Merge

**Files:** `backend/app/pipeline/merge.py`, `backend/app/pipeline/ingest.py`

### `merge_scans(scan_paths, voxel_size=0.03, progress_cb=None)`

```python
# runner.py line 64
merge_result = merge_scans(scan_paths, voxel_size=0.03, progress_cb=_merge_progress)
```

| Param | Default | Range | Effect |
|---|---|---|---|
| `voxel_size` | `0.03` m | 0.01–0.15 | Grid cell size for downsampling each scan before and after merge. **Biggest lever for speed vs. detail.** |

| `voxel_size` | Resulting pts (6 scans) | Processing time | Use case |
|---|---|---|---|
| `0.02` | ~40M | ~25 min | Maximum detail |
| `0.03` | ~18M | ~16 min | **Production default** |
| `0.05` | ~7M | ~8 min | Fast iteration |
| `0.10` | ~2M | ~4 min | Debug / prototype |

### `_load_las()` — individual scan loading

Uses **laspy 2.5** — see [Library G](#14-library-g-laspy-25--full-api).

Each scan is loaded with `laspy.read()`, XYZ extracted, downsampled to `voxel_size`, then merged.

**Currently unused but available from each LAS file:**
- `las.intensity` — scan reflectance (0–65535) — can be used to color points or weight wall detection
- `las.classification` — ASPRS classification (2=Ground, 6=Building, etc.) — pre-filter to class 6 to eliminate outdoor terrain automatically
- `las.return_number` / `las.number_of_returns` — multi-return info; last return = ground
- `las.scan_angle` — angle at which laser was fired
- `las.point_source_id` — which scanner/flight strip

**High-value unlock: use `las.classification == 6` to filter to building points before any processing.** This would immediately fix the floor detection problem with outdoor terrain.

```python
# In ingest.py _load_las() — add after reading points:
building_mask = las.classification == 6
if building_mask.sum() > 1000:
    pts = pts[building_mask]
```

### Pre-registration detection (`_scans_share_coordinate_frame`)

Current logic: centroid spread >5m OR voxel overlap → pre-registered (concatenate). Otherwise → ICP registration.

---

## 3. Stage 2 — Floor Detection

**File:** `backend/app/pipeline/slicing.py`

### `detect_floor(scan)` — all variables

```python
# Downsampling before plane search
scan_ds = scan.voxel_down_sample(0.10)   # 10cm for speed

# RANSAC plane fitting — runs up to 12 times
plane, inliers = remaining.segment_plane(
    distance_threshold = 0.03,   # metres — inlier tolerance
    ransac_n           = 3,      # minimum points per RANSAC hypothesis
    num_iterations     = 1000,   # iterations per plane search
)

# Horizontal plane filter
cos_angle > 0.96   # plane normal must be within ~16° of vertical axis

# Scoring: count points 0.30–3.00m above each candidate
((z > floor_h + 0.30) & (z < floor_h + 3.00)).sum()
```

| Variable | Default | ↑ to | ↓ to |
|---|---|---|---|
| `distance_threshold` | `0.03` | `0.05` for rough/worn floors | `0.01` for very flat polished floors |
| `num_iterations` | `1000` | `2000` for accuracy | `500` for speed |
| `cos_angle >` | `0.96` (~16°) | `0.90` (~26°) for sloped ramps | keep at 0.96 for flat floors |
| Score band high | `3.00` m | `3.50` for tall spaces | `2.20` for low ceilings |
| Score band low | `0.30` m | stay | `0.10` if floor has scatter near it |

### `extract_wall_band(scan, floor, band_low_m, band_high_m)` — variables

```python
# runner.py line 85
wall_band = extract_wall_band(scan_pcd, floor, band_low_m=0.75, band_high_m=1.80)
```

| Param | Default | Effect |
|---|---|---|
| `band_low_m` | `0.75` m | Bottom of slice above floor. Below this = furniture legs, baseboards. |
| `band_high_m` | `1.80` m | Top of slice. Above this = ceiling, lighting. |

---

## 4. Stage 3 — 3D Wall Planes

**File:** `backend/app/pipeline/segment.py`

### Progressive fallback attempts

```python
attempts = [
    # (voxel_size, min_inliers, dist_thresh, normal_up_max, min_height_range)
    (0.05, 80,  0.05, 0.30, 0.15),   # Attempt 1: strict
    (0.08, 50,  0.08, 0.40, 0.10),   # Attempt 2: looser
    (0.10, 30,  0.10, 0.50, 0.08),   # Attempt 3: very loose
]
# If all fail → normal-based extraction on full cloud
```

| Param | Meaning | ↑ effect | ↓ effect |
|---|---|---|---|
| `voxel_size` | Grid size for pre-downsampling wall band | Faster, less detail | Slower, more planes found |
| `min_inliers` | Min points on a plane to accept it | Fewer planes, higher quality | More planes, may include noise |
| `dist_thresh` | Max distance from plane to be an inlier | More inliers per plane | Fewer inliers, cleaner planes |
| `normal_up_max` | Max vertical component of plane normal (0=vertical, 1=horizontal) | Allows tilted walls | Only near-vertical walls |
| `min_height_range` | Plane's points must span this height | Filters horizontal furniture tops | May reject short walls |

### Normal-based fallback

```python
ds.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=30)
)
```

| Param | Default | Effect |
|---|---|---|
| `radius` | `0.20` m | Neighbourhood for normal estimation. `0.40` for sparse clouds. |
| `max_nn` | `30` | Max neighbours. More = smoother normals, slower. |

---

## 5. Stage 4 — 2D Floor Plan

**File:** `backend/app/pipeline/scanplan.py` → `generate_plan_from_projection()`

### Complete pipeline inside this function

```python
generate_plan_from_projection(
    scan_pcd,
    floor,
    grid_res_m        = 0.10,   # occupancy grid: metres per pixel
    min_wall_length_m = 2.00,   # discard walls shorter than this
    max_lines         = 120,    # cap total output segments
)
```

#### Step 1: Height band

```python
# Floor-relative (primary)
pts_mid = pts[(pts[:, axis] > floor_z + 0.20) & (pts[:, axis] < floor_z + 2.80)]

# Fallback: densest 2.5m window (if <2% of pts in floor-relative band)
pts_mid = _dense_z_slice(pts, axis, band_width=2.5, step=0.10)
```

| Variable | Default | Raise to | Lower to |
|---|---|---|---|
| `floor_z + 0.20` | 20 cm above floor | `0.40` to skip floor reflections | `0.10` for low-profile objects |
| `floor_z + 2.80` | 2.8 m above floor | `3.50` for warehouses | `2.00` for low ceilings |
| Fallback threshold | `2%` of pts | `5%` for noisy data | `1%` to rely more on floor detection |
| `band_width` | `2.5` m | `3.0` for tall spaces | `2.0` for multi-floor buildings |
| `step` | `0.10` m | stay | `0.05` for more precision |

#### Step 2: Occupancy grid

```python
grid_res_m = 0.10  # 1 pixel = 10 cm
xi = np.floor((xy[:, 0] - x0) / grid_res_m)
```

| Resolution | Px for 100m building | Detail |
|---|---|---|
| `0.05` m | 2000×2000 | Finds walls 10 cm thick |
| `0.10` m | 1000×1000 | **Current default** — 20 cm walls |
| `0.20` m | 500×500 | Fast, misses thin walls |

#### Step 3: Morphological close + fill

```python
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
img_d = cv2.dilate(img, k_close, iterations=3)
img_filled = ndi.binary_fill_holes(img_d > 0)
edges = cv2.Canny(img_filled, threshold1=50, threshold2=150)
```

| Variable | Default | Effect |
|---|---|---|
| Kernel shape | `MORPH_ELLIPSE (5,5)` | Isotropic — no axis preference. Use `MORPH_RECT` for speed. |
| Dilation `iterations` | `3` | Each iteration bridges ~25 cm of gap at 10 cm/px. Raise to `6` for bigger gaps. |
| `Canny threshold1` | `50` | Lower edge acceptance. Reduce to `20` for faint walls. |
| `Canny threshold2` | `150` | Strong edge threshold. Keep ratio ~1:3 with threshold1. |
| Canny `L2gradient` | `False` | Set `True` for more accurate edges on curved walls. |
| Canny `apertureSize` | `3` | Sobel kernel size. `5` or `7` for smoother edges. |

#### Step 4: Hough Lines (cv2.HoughLinesP)

```python
cv2.HoughLinesP(
    edges,
    rho          = 1,            # pixel distance resolution
    theta        = np.pi / 360,  # 0.5° angular resolution
    threshold    = 40,           # minimum votes
    minLineLength= min_px,       # minimum segment length in pixels
    maxLineGap   = gap_px,       # maximum gap to bridge within one segment
)
```

| Parameter | Default | **↑ Raise when** | **↓ Lower when** |
|---|---|---|---|
| `rho` | `1` | — | Speed only (try `2`) |
| `theta` | `π/360` (0.5°) | — | Speed only (`π/180` = 1°) |
| `threshold` | `40` | Too many spurious lines | Missing walls |
| `minLineLength` | `20 px` (2 m) | Short noise segments appear | Walls are short (<2 m) |
| `maxLineGap` | `6 px` (0.6 m) | Wall segments split at doorways | Unrelated walls are being joined |

**Critical note on threshold:** This is the number of edge pixels that must vote for a line in Hough space. With a 10 cm grid and a 2 m wall, a solid wall has ~20 pixels. `threshold=40` means the wall must appear in Hough space at least 40 times — good for walls, eliminates most furniture.

#### Step 5: Manhattan-world filter

```python
_filter_to_dominant_orientations(segments, tolerance_deg=15.0)
```

| Variable | Default | Raise | Lower |
|---|---|---|---|
| `tolerance_deg` | `15°` | `25°` allows diagonal walls | `8°` extremely strict |

**Disable entirely** (return all segments) for buildings without orthogonal wall layouts, or circular/organic architecture.

---

## 6. Stage 5 — Room Extraction

**File:** `backend/app/pipeline/scanplan.py` → `_rooms_from_occupancy()`

```python
_rooms_from_occupancy(
    scan_pcd, floor, synthetic_plan,
    cell_size        = 0.50,   # grid resolution for room detection
    min_room_area_m2 = 10.0,   # minimum m² to call it a room
    max_rooms        = 30,     # cap on rooms returned
)
```

### Morphological dilation for gap bridging

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
occ_d  = cv2.dilate(occ, kernel, iterations=12)
```

At `cell_size=0.50 m`, kernel (7,7), each iteration bridges ~1.75 m:

| Iterations | Total bridge reach | Use when |
|---|---|---|
| `4` | ~7 m | Scans have dense coverage, small gaps |
| `8` | ~14 m | Typical office corridor gaps |
| `12` | ~21 m | **Current default** — rooms 5–10 m apart |
| `20` | ~35 m | Very sparse warehouse scans |

### Contour polygon simplification

```python
step = max(1, len(contour) // 40)   # keep ~40 points per polygon
```

| `// N` | Points per polygon | Quality |
|---|---|---|
| `// 20` | ~20 | Very boxy |
| `// 40` | ~40 | **Current** — clean for UI |
| `// 80` | ~80 | Smooth curves |

**Better alternative:** use `skimage.measure.approximate_polygon()`:
```python
from skimage.measure import approximate_polygon
simplified = approximate_polygon(contour, tolerance=2.0)  # 2px = 1m at 0.5m/px
```

---

## 7. Stage 6 — Viewer PLY

**File:** `backend/app/pipeline/runner.py` → `_wall_height_slice()`

```python
_wall_height_slice(
    scan_pcd, floor_z, axis_idx,
    low_offset  = 0.20,   # metres above floor — cut floor returns
    high_offset = 2.50,   # metres above floor — cut ceiling
)
write_decimated_ply(_viewer_pcd, decimated_path, target_points=200_000)
```

| Param | Default | Raise | Lower |
|---|---|---|---|
| `low_offset` | `0.20` m | `0.40` to cut more floor scatter | `0.10` for very thin wall bases |
| `high_offset` | `2.50` m | `3.50` for tall warehouses | `2.00` for low ceilings |
| `target_points` | `200_000` | `500_000` for richer viewer | `100_000` for faster load |
| Fallback minimum | `1000` pts | increase if viewer shows garbage | — |

---

## 8. Library A: Open3D 0.18 — Full API

**Docs:** https://www.open3d.org/docs/0.18.0/python_api/open3d.geometry.PointCloud.html

### PointCloud — complete method list

#### Downsampling
```python
pcd.voxel_down_sample(voxel_size)
# → PointCloud. Grid-based. Each voxel → 1 point (centroid). Fastest.

pcd.uniform_down_sample(every_k_points)
# → PointCloud. Keep every Nth point. No grid. Use for preview.

pcd.farthest_point_down_sample(num_samples)
# → PointCloud. Maximally spread sample. Best for registration but slow.

pcd.random_down_sample(sampling_ratio)
# → PointCloud. Random fraction 0–1.
```

#### Noise removal — **currently unused, high value**
```python
pcd.remove_statistical_outlier(
    nb_neighbors = 20,    # neighbourhood size for mean distance
    std_ratio    = 2.0,   # points > mean + std_ratio * std are removed
)
# Returns: (cleaned_pcd, list_of_kept_indices)
# USE THIS before merging to remove tree foliage, scan noise, sky points.

pcd.remove_radius_outlier(
    nb_points = 16,       # minimum neighbours within radius
    radius    = 0.10,     # m — search radius
)
# Returns: (cleaned_pcd, list_of_kept_indices)
# More aggressive than statistical. Use for isolated floating points.
```

#### Plane segmentation
```python
plane_model, inliers = pcd.segment_plane(
    distance_threshold = 0.03,  # m — inlier tolerance
    ransac_n           = 3,     # points per minimal model
    num_iterations     = 1000,  # RANSAC iterations
    probability        = 0.999, # desired probability of finding best plane
)
# plane_model = [a, b, c, d] for ax+by+cz+d=0
# inliers = list of point indices
```

#### Planar patch detection — **currently unused**
```python
patches = pcd.detect_planar_patches(
    normal_variance_threshold_deg = 60,   # max variance of normals in patch
    coplanarity_deg               = 75,   # coplanarity threshold
    outlier_ratio                 = 0.75, # max fraction of outliers
    min_plane_edge_length         = 0.0,  # minimum patch edge length (m)
    min_num_points                = 0,    # minimum points per patch
    search_param = KDTreeSearchParamKNN(knn=30),
)
# Returns: List[OrientedBoundingBox] — each box represents one planar patch
# BETTER than segment_plane for building walls — finds all planar regions at once
```

#### Normal estimation
```python
pcd.estimate_normals(
    search_param = o3d.geometry.KDTreeSearchParamHybrid(
        radius = 0.20,    # neighbourhood radius (m)
        max_nn = 30,      # max neighbours
    ),
    fast_normal_computation = True,  # False = more stable, slower
)

pcd.orient_normals_consistent_tangent_plane(k=30)  # make normals consistent
pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
pcd.orient_normals_to_align_with_direction(orientation_reference=[0, 0, 1])
```

#### Clustering — **currently unused**
```python
labels = pcd.cluster_dbscan(
    eps           = 0.50,   # m — neighbourhood radius
    min_points    = 10,     # minimum cluster size
    print_progress= False,
)
# labels[i] = cluster index for point i; -1 = noise
# USE THIS to: separate individual scan clusters, identify room blobs,
# filter out outdoor noise clusters far from building centroid
```

#### Geometry queries
```python
pcd.get_axis_aligned_bounding_box()    # → AxisAlignedBoundingBox (fast)
pcd.get_oriented_bounding_box()        # → OrientedBoundingBox (PCA-based, tight)
pcd.get_minimal_oriented_bounding_box()# → smallest possible OBB
pcd.get_center()                       # → [x, y, z] centroid
pcd.compute_convex_hull()              # → (TriangleMesh, point_indices)
pcd.compute_nearest_neighbor_distance()# → per-point NN distance
pcd.compute_point_cloud_distance(target_pcd)  # → per-point distance to target
pcd.compute_mean_and_covariance()      # → (mean_3d, cov_3x3)
pcd.compute_mahalanobis_distance()     # → per-point Mahalanobis distances
```

#### Visibility / cropping — **currently unused**
```python
pcd.hidden_point_removal(
    camera_location = [0, 0, 10],   # viewpoint position
    radius          = 100,          # sphere radius
)
# Removes points not visible from camera_location
# USE THIS before top-down projection to remove occluded points

pcd.crop(bounding_box, invert=False)
# Crop to axis-aligned or oriented bounding box
```

#### Registration (ICP)
```python
import open3d.pipelines.registration as reg

# Point-to-point ICP
result = reg.registration_icp(
    source, target,
    max_correspondence_distance = 0.05,   # m
    init                        = np.eye(4),
    estimation_method           = reg.TransformationEstimationPointToPoint(),
    criteria                    = reg.ICPConvergenceCriteria(
        relative_fitness = 1e-6,
        relative_rmse    = 1e-6,
        max_iteration    = 30,
    ),
)

# Point-to-plane ICP (more accurate, needs normals)
result = reg.registration_icp(
    source, target,
    max_correspondence_distance = 0.05,
    estimation_method           = reg.TransformationEstimationPointToPlane(),
)

# Colored ICP — uses RGB + geometry (needs colors)
result = reg.registration_colored_icp(
    source, target,
    max_correspondence_distance = 0.05,
    init                        = np.eye(4),
    criteria                    = reg.ICPConvergenceCriteria(max_iteration=50),
    lambda_geometric            = 0.968,  # weight for geometric term (0–1)
)

# result.fitness      — overlap ratio (0–1); higher = better
# result.inlier_rmse  — RMSE of correspondences; lower = better
# result.transformation — 4×4 transform matrix
```

#### Global registration (FPFH + RANSAC) — already used in merge.py
```python
import open3d.pipelines.registration as reg

fpfh = reg.compute_fpfh_feature(
    pcd,
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100),
)

result = reg.registration_ransac_based_on_feature_matching(
    source, target, source_fpfh, target_fpfh,
    mutual_filter                = True,
    max_correspondence_distance  = 0.05,
    estimation_method            = reg.TransformationEstimationPointToPoint(False),
    ransac_n                     = 3,
    checkers = [
        reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        reg.CorrespondenceCheckerBasedOnDistance(0.05),
    ],
    criteria = reg.RANSACConvergenceCriteria(100000, 0.999),
)
```

---

## 9. Library B: OpenCV 4.13 — Full API

**Docs:** https://docs.opencv.org/4.x/

### Image creation and conversion
```python
img = np.zeros((h, w), dtype=np.uint8)   # binary image
img = np.zeros((h, w, 3), dtype=np.uint8)  # colour image
gray = cv2.cvtColor(colour_img, cv2.COLOR_BGR2GRAY)
```

### Blurring / smoothing — **use before Canny for noisy scans**
```python
cv2.GaussianBlur(src, ksize=(5,5), sigmaX=1.0, sigmaY=0.0)
# ksize must be odd. sigmaX/Y in pixels. 0 = compute from ksize.
# USE: pre-smooth occupancy image to reduce edge noise before Canny.

cv2.medianBlur(src, ksize=5)   # ksize must be odd
# Removes salt-and-pepper noise. Good for binary occupancy images.

cv2.blur(src, ksize=(5,5))     # simple box blur
```

### Morphological operations — complete list
```python
# Structuring element shapes
cv2.MORPH_RECT      # rectangle
cv2.MORPH_ELLIPSE   # ellipse — isotropic, preferred
cv2.MORPH_CROSS     # cross shape — good for thin structures

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
# kw, kh must be odd integers. Larger = wider reach per iteration.

# Basic ops
cv2.dilate(src, kernel, iterations=1)    # grow foreground
cv2.erode(src, kernel, iterations=1)     # shrink foreground

# Combined ops (morphologyEx) — more efficient than separate calls
cv2.morphologyEx(src, op, kernel, iterations=1)
# op options:
cv2.MORPH_OPEN     # erode then dilate — removes small protrusions
cv2.MORPH_CLOSE    # dilate then erode — fills small holes ← USE THIS for gap bridging
cv2.MORPH_GRADIENT # dilate minus erode — ring/outline of foreground ← USE FOR WALL BOUNDARIES
cv2.MORPH_TOPHAT   # src minus open — bright spots on dark bg
cv2.MORPH_BLACKHAT # close minus src — dark spots on bright bg
cv2.MORPH_HITMISS  # hit-or-miss transform
```

**Recommended replacement for current dilate+erode sequence:**
```python
# Current (two calls):
img_d = cv2.dilate(img, kernel, iterations=3)
img_e = cv2.erode(img_d, kernel, iterations=1)

# Better (one call, same result):
img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
```

### Edge detection
```python
# Canny — currently used
cv2.Canny(
    image,
    threshold1   = 50,     # lower hysteresis threshold
    threshold2   = 150,    # upper hysteresis threshold
    apertureSize = 3,      # Sobel kernel size: 3, 5, or 7
    L2gradient   = False,  # True = more accurate gradient norm
)

# Sobel — raw gradient
sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)  # x gradient
sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)  # y gradient
magnitude = cv2.magnitude(sobelx, sobely)

# Laplacian
lap = cv2.Laplacian(src, cv2.CV_64F, ksize=3)
```

### Line detection — complete Hough API
```python
# Probabilistic Hough — currently used (returns segments)
lines = cv2.HoughLinesP(
    image,
    rho          = 1,           # px — distance accumulator resolution
    theta        = np.pi/180,   # rad — angle resolution
    threshold    = 40,          # votes required
    minLineLength= 20,          # px — minimum segment length
    maxLineGap   = 6,           # px — max gap within one segment
)
# Returns: array of shape (N, 1, 4) — [[x1,y1,x2,y2], ...]

# Standard Hough — returns infinite lines (better for extending to corners)
lines = cv2.HoughLines(
    image,
    rho       = 1,
    theta     = np.pi/180,
    threshold = 40,
    srn       = 0,   # for multi-scale Hough (0 = standard)
    stn       = 0,
    min_theta = 0,   # minimum angle
    max_theta = np.pi,  # maximum angle
)
# Returns: array of shape (N, 1, 2) — [[rho, theta], ...]
# Convert to line endpoints: x = cos(theta)*rho, y = sin(theta)*rho
```

### Connected components — alternative to scikit-image
```python
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary_image,
    connectivity = 8,      # 4 or 8
    ltype        = cv2.CV_32S,
)
# stats[i] = [x, y, width, height, area] for component i
# centroids[i] = [cx, cy]
# Label 0 = background

# Simple version (no stats)
n_labels, labels = cv2.connectedComponents(binary_image, connectivity=8)
```

### Distance transform — **high value for room centroids**
```python
dist = cv2.distanceTransform(
    binary_image,
    distanceType = cv2.DIST_L2,    # Euclidean — also: DIST_L1, DIST_C
    maskSize     = cv2.DIST_MASK_PRECISE,  # also: DIST_MASK_3, DIST_MASK_5
)
# dist[y,x] = distance from pixel (x,y) to nearest background pixel
# Peak of dist inside a room = room center (furthest from walls)
# USE: find canonical room centroid points for watershed seeds
```

### Watershed — **next big upgrade for rooms**
```python
# Requires: 8-bit 3-channel image + 32-bit marker image
markers = np.zeros_like(binary_image, dtype=np.int32)
# Place seeds: markers[y, x] = room_id (1, 2, 3, ...) at known room centers
# 0 = unknown region, -1 = boundary

cv2.watershed(colour_image_8uc3, markers)
# After call: markers contains room labels, -1 at boundaries
# Boundary pixels (markers == -1) are the walls between rooms
```

**Watershed workflow for room detection:**
1. Create occupancy image
2. Run `distanceTransform` on filled footprint
3. Find local maxima of distance transform → room center seeds
4. Place seeds in marker image
5. Run `watershed` → room boundaries are the walls

### Contour finding
```python
contours, hierarchy = cv2.findContours(
    binary_image,
    mode   = cv2.RETR_EXTERNAL,  # outer contours only
                                  # also: RETR_LIST, RETR_CCOMP, RETR_TREE
    method = cv2.CHAIN_APPROX_SIMPLE,  # compress horizontal/vertical/diagonal runs
                                        # also: CHAIN_APPROX_NONE (all points)
)

# Simplify contour (Douglas-Peucker)
epsilon = 0.02 * cv2.arcLength(contour, closed=True)
approx = cv2.approxPolyDP(contour, epsilon, closed=True)
# epsilon controls simplification tolerance
```

### Drawing (for debug output)
```python
cv2.line(img, pt1, pt2, color, thickness)
cv2.drawContours(img, contours, contourIdx, color, thickness)
cv2.rectangle(img, pt1, pt2, color, thickness)
cv2.circle(img, center, radius, color, thickness)
cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
```

---

## 10. Library C: scikit-image 0.26 — Full API

**Docs:** https://scikit-image.org/docs/stable/api/api.html

### skimage.measure — currently used
```python
from skimage.measure import label, regionprops, find_contours, approximate_polygon

# Label connected components
labeled = label(binary_image, connectivity=2)  # 2=8-connected, 1=4-connected

# Region properties — every available attribute:
for region in regionprops(labeled):
    region.label          # integer label
    region.area           # pixel count
    region.area_bbox      # bounding box area
    region.area_convex    # convex hull area
    region.area_filled    # area with holes filled
    region.bbox           # (min_row, min_col, max_row, max_col)
    region.centroid       # (row, col) — USE THIS for room centers
    region.centroid_local # centroid relative to bbox
    region.coords         # array of pixel coords
    region.eccentricity   # 0=circle, 1=line — USE: corridor detection
    region.equivalent_diameter_area  # diameter of circle with same area
    region.euler_number   # topology (holes)
    region.extent         # area / bbox_area — USE: detect non-rectangular rooms
    region.feret_diameter_max  # longest diameter
    region.inertia_tensor # 2×2 inertia tensor
    region.inertia_tensor_eigvals
    region.intensity_max  # max intensity (if intensity_image provided)
    region.intensity_mean # mean intensity
    region.intensity_median  # median intensity (new in 0.26)
    region.intensity_min
    region.moments        # spatial moments
    region.moments_central
    region.moments_hu     # Hu invariant moments — USE: room shape fingerprinting
    region.moments_normalized
    region.orientation    # angle of major axis — USE: detect corridor direction
    region.perimeter      # perimeter length — USE: room classification
    region.perimeter_crofton  # more accurate perimeter estimate
    region.slice          # slice objects for indexing
    region.solidity       # area / convex_area — USE: detect irregular rooms

# Find contours (marching squares)
contours = find_contours(binary_region, level=0.5)
# level: isovalue; 0.5 for binary images is correct

# Simplify contour — BETTER than current every-Nth subsampling
simplified = approximate_polygon(contour, tolerance=2.0)
# tolerance in pixels — 1.0 = tight, 5.0 = loose
```

### skimage.morphology — available but unused
```python
from skimage import morphology

# Structuring elements (footprints)
morphology.disk(radius)          # filled circle
morphology.square(side)          # filled square
morphology.diamond(radius)       # 45° rotated square
morphology.star(a)               # star-shaped
morphology.octagon(m, n)         # octagon

# Basic morphology
morphology.dilation(image, footprint=disk(3))
morphology.erosion(image, footprint=disk(3))
morphology.opening(image, footprint=disk(3))   # erode then dilate
morphology.closing(image, footprint=disk(3))   # dilate then erode

# Noise removal — **HIGH VALUE**
morphology.remove_small_objects(binary, min_size=64)
# Removes connected components smaller than min_size pixels
# USE: clean tiny scan artifacts before room detection

morphology.remove_small_holes(binary, area_threshold=64)
# Fills holes smaller than area_threshold
# USE: fill small voids inside rooms

# Skeleton / thinning — **HIGH VALUE for wall centerlines**
skel = morphology.skeletonize(binary_image)
# Reduces binary shape to 1-pixel centerline
# USE: find wall centerlines for clean line representation

thin = morphology.thin(binary_image, max_iter=None)
# Alternative to skeletonize — better for branching structures

medial = morphology.medial_axis(binary_image, return_distance=False)
# Returns medial axis — good for finding corridor spines

# Convex hull
hull = morphology.convex_hull_image(binary_image)

# Flood fill
filled = morphology.flood_fill(image, seed_point, new_value)

# Advanced area/diameter morphology
opened = morphology.area_opening(image, area_threshold=64)
closed = morphology.area_closing(image, area_threshold=64)
```

### skimage.segmentation — available but unused
```python
from skimage import segmentation

# Watershed (simpler API than cv2)
from skimage.segmentation import watershed
labels = watershed(distance_image, markers, mask=binary_image)
# distance_image: result of distance_transform_edt
# markers: seed points for each region (integer array)

# Find boundaries between regions
boundaries = segmentation.find_boundaries(label_image, mode='thick')
# mode: 'thick', 'inner', 'outer', 'subpixel'
# USE: extract wall boundaries as binary image

# Expand labels (grow each labeled region)
expanded = segmentation.expand_labels(label_image, distance=10)

# Active contours (snakes) — sophisticated room outline fitting
from skimage.segmentation import active_contour
snake = active_contour(gaussian_image, init_contour, alpha=0.015, beta=10)
```

### skimage.feature — available but unused
```python
from skimage import feature

# Corner detection
corners = feature.corner_harris(gray_image, method='k', k=0.05)
coords  = feature.corner_peaks(corners, min_distance=5)
# USE: find wall corner points for snapping segment endpoints

# Blob detection
blobs = feature.blob_log(gray_image, min_sigma=1, max_sigma=30, num_sigma=10)
# USE: find circular features like columns/pillars in floor plan

# RANSAC (skimage has its own) — line fitting
from skimage.measure import ransac, LineModelND
model, inliers = ransac(data, LineModelND, min_samples=2,
                        residual_threshold=1.0, max_trials=1000)
```

---

## 11. Library D: scipy.ndimage — Full API

**Docs:** https://docs.scipy.org/doc/scipy/reference/ndimage.html

### Currently used
```python
from scipy import ndimage
ndimage.binary_fill_holes(binary_image)
# Fills enclosed holes in a binary image. The only function we currently use.
```

### Morphology (full list)
```python
ndimage.binary_dilation(input, structure=None, iterations=1)
ndimage.binary_erosion(input, structure=None, iterations=1)
ndimage.binary_opening(input, structure=None, iterations=1)
ndimage.binary_closing(input, structure=None, iterations=1)
ndimage.binary_fill_holes(input, structure=None)    # ← currently used
ndimage.binary_hit_or_miss(input, structure1=None, structure2=None)
ndimage.binary_propagation(input, structure=None, mask=None)
ndimage.grey_dilation(input, size=None, footprint=None)
ndimage.grey_erosion(input, size=None, footprint=None)
ndimage.grey_opening(input, size=None, footprint=None)
ndimage.grey_closing(input, size=None, footprint=None)
ndimage.morphological_gradient(input, size=None)   # dilation - erosion
ndimage.morphological_laplace(input, size=None)    # (dilation + erosion) - 2*input
ndimage.white_tophat(input, size=None)             # input - opening
ndimage.black_tophat(input, size=None)             # closing - input
```

### Distance transforms — **HIGH VALUE**
```python
dist, indices = ndimage.distance_transform_edt(
    binary_image,
    sampling     = None,   # voxel spacing; (1.0, 1.0) for uniform
    return_distances = True,
    return_indices   = False,
)
# dist[y,x] = Euclidean distance to nearest background pixel
# Perfect for finding room interior seed points

ndimage.distance_transform_bf(input, metric='euclidean', sampling=None)
# Brute force — slow but exact for any metric

ndimage.distance_transform_cdt(input, metric='chessboard')
# Chamfer distance — fast approximation
```

### Measurements (per-label stats)
```python
ndimage.label(input, structure=None)              # ← superseded by skimage.label
ndimage.center_of_mass(input, labels=None, index=None)  # per-label centroid
ndimage.sum_labels(input, labels=None, index=None)      # per-label sum
ndimage.mean(input, labels=None, index=None)            # per-label mean
ndimage.maximum(input, labels=None, index=None)         # per-label maximum
ndimage.minimum(input, labels=None, index=None)         # per-label minimum
ndimage.variance(input, labels=None, index=None)        # per-label variance
ndimage.standard_deviation(input, labels=None, index=None)
ndimage.extrema(input, labels=None, index=None)         # min, max + positions
ndimage.histogram(input, min, max, bins, labels=None)
ndimage.maximum_position(input, labels=None, index=None) # position of maximum
ndimage.minimum_position(input, labels=None, index=None)
ndimage.watershed_ift(input, markers, structure=None)    # scipy's watershed
```

### Filters
```python
ndimage.gaussian_filter(input, sigma=1.0)          # ← use before Canny
ndimage.median_filter(input, size=3)               # remove salt+pepper noise
ndimage.uniform_filter(input, size=3)              # box blur
ndimage.maximum_filter(input, size=3)              # local max (dilation-like)
ndimage.minimum_filter(input, size=3)              # local min (erosion-like)
ndimage.sobel(input, axis=0)                       # edge detection
ndimage.laplace(input)                             # edge detection
ndimage.convolve(input, weights)                   # custom kernel
ndimage.percentile_filter(input, percentile=50, size=3)
```

---

## 12. Library E: Shapely 2.1 — Full API

**Docs:** https://shapely.readthedocs.io/en/stable/

### Currently used
```python
from shapely.geometry import MultiLineString
from shapely.ops import polygonize, unary_union
```

### Constructive operations — full list
```python
import shapely

# Building outlines — HIGH VALUE
hull = shapely.concave_hull(geometry, ratio=0.0, allow_holes=False)
# ratio: 0.0 = convex hull, 1.0 = tightest concave hull
# REQUIRES GEOS ≥ 3.11 (check: shapely.geos_version)
# USE: generate building outline from projected floor points

hull = shapely.convex_hull(geometry)

# Simplification
simplified = shapely.simplify(geometry, tolerance=0.5)
# tolerance in same units as coordinates (metres)
# Douglas-Peucker algorithm

# Buffering
buffered = shapely.buffer(
    geometry,
    distance,
    quad_segs     = 16,    # segments per quarter circle
    cap_style     = 1,     # 1=round, 2=flat, 3=square
    join_style    = 1,     # 1=round, 2=mitre, 3=bevel
    mitre_limit   = 5.0,
    single_sided  = False,
)
# Negative distance = inward buffer (shrink polygon)

# Snapping — HIGH VALUE for connecting near-touching wall segments
snapped = shapely.snap(geometry_a, geometry_b, tolerance=0.1)
# Snaps geometry_a to geometry_b within tolerance

# Validity repair
fixed = shapely.make_valid(geometry)
normalized = shapely.normalize(geometry)

# Boolean operations
union        = shapely.union(a, b)
intersection = shapely.intersection(a, b)
difference   = shapely.difference(a, b)
sym_diff     = shapely.symmetric_difference(a, b)
```

### Measurement operations
```python
shapely.area(geometry)
shapely.length(geometry)
shapely.distance(a, b)           # minimum distance
shapely.hausdorff_distance(a, b) # max of all min distances
shapely.frechet_distance(a, b)   # path-aware distance
shapely.perimeter(geometry)      # = length for polygons
```

### Set operations on collections
```python
from shapely.ops import unary_union, polygonize, split, nearest_points, snap

unary_union([geom1, geom2, ...])           # merge all geometries
polygonize(multilinestring)                # form polygons from lines
list(polygonize_full(lines))               # returns (polys, dangles, cuts, invalids)
nearest_points(geom_a, geom_b)            # nearest point pair
```

### Predicates (boolean tests)
```python
shapely.is_valid(geom)
shapely.is_empty(geom)
shapely.intersects(a, b)
shapely.contains(a, b)
shapely.within(a, b)
shapely.covers(a, b)
shapely.crosses(a, b)
shapely.touches(a, b)
```

---

## 13. Library F: alphashape 1.3 — Full API

**Docs:** https://alphashape.readthedocs.io/en/latest/

### Functions
```python
import alphashape

# Main function
shape = alphashape.alphashape(points, alpha=None)
# points: list of (x,y) tuples, np.ndarray, MultiPoint, or GeoDataFrame
# alpha: float — controls concavity
#   alpha = 0   → convex hull
#   alpha = 0.5 → moderately concave
#   alpha = 5.0 → tight concave hull
#   alpha = None → auto-optimize (slow, calls optimizealpha)
# Returns: Shapely Polygon, LineString, or Point

# Auto-optimize alpha (finds tightest hull containing all points in one polygon)
optimal_alpha = alphashape.optimizealpha(
    points,
    max_iterations = 10000,
    lower          = 0.0,
    upper          = float('inf'),
    silent         = False,
)
# Returns: float — use this with alphashape(points, optimal_alpha)
# Returns 0 on failure (safe — falls back to convex hull)

# Low-level utilities
simplices = alphashape.alphasimplices(points)   # Delaunay simplices + circumradii
cc = alphashape.circumcenter(triangle_points)    # circumcenter
cr = alphashape.circumradius(triangle_points)    # circumradius
```

### When to use alphashape vs. alternatives

| Situation | Best choice |
|---|---|
| Need exact building outline, points are dense | `alphashape.optimizealpha()` then `alphashape()` |
| Fast rough outline, building is convex | `shapely.convex_hull()` |
| Building has courtyards / holes | `shapely.concave_hull(ratio=0.3, allow_holes=True)` |
| Need to match existing wall segments | `shapely.buffer(union_of_segments, distance=0.5)` |

### Recommended usage for building outline
```python
import alphashape
import numpy as np

# Use projected 2D floor points
pts_2d = [(x, y) for x, y in xy_array]

# Auto-tune alpha — ~5 seconds for 100K points
alpha = alphashape.optimizealpha(pts_2d)
outline = alphashape.alphashape(pts_2d, alpha)

# outline is a Shapely Polygon
print(f"Building area: {outline.area:.1f} m²")
print(f"Building perimeter: {outline.length:.1f} m")
```

---

## 14. Library G: laspy 2.5 — Full API

**Docs:** https://laspy.readthedocs.io/en/latest/

### Reading files
```python
import laspy

# Read entire file into memory
las = laspy.read("scan.laz")   # auto-detects LAS/LAZ, decompresses LAZ

# Streaming read (large files that don't fit in RAM)
with laspy.open("scan.laz") as reader:
    for chunk in reader.chunk_iterator(chunk_size=1_000_000):
        pts = np.stack([chunk.x, chunk.y, chunk.z], axis=1)
        # process chunk...
```

### Standard point dimensions — **all available on every LAS file**
```python
las.x           # float64 array — easting
las.y           # float64 array — northing
las.z           # float64 array — elevation

las.intensity          # uint16 — scan reflectance (0–65535)
las.return_number      # uint8 — which return this point is (1st, 2nd, ...)
las.number_of_returns  # uint8 — total returns for this pulse
las.scan_direction_flag # bool — scanner moving left/right
las.edge_of_flight_line # bool — last point before line change
las.classification     # uint8 — ASPRS classification:
#   0=Never classified, 1=Unassigned, 2=Ground, 3=Low vegetation
#   4=Medium vegetation, 5=High vegetation, 6=Building
#   7=Noise, 8=Reserved, 9=Water, 10=Rail, 11=Road surface
#   12=Reserved, 13=Wire guard, 14=Wire conductor, 15=Transmission tower
#   16=Wire connector, 17=Bridge deck, 18=High noise
las.synthetic          # bool — point was synthetically created
las.key_point          # bool — point is a model key-point
las.withheld           # bool — point should be ignored
las.overlap            # bool — point is in overlap region
las.scan_angle         # int16 — scan angle rank/angle step
las.user_data          # uint8 — user-defined
las.point_source_id    # uint16 — file origin / flight strip ID
las.gps_time           # float64 — GPS timestamp

# RGB colour (LAS format ≥ 2)
las.red    # uint16
las.green  # uint16
las.blue   # uint16

# NIR (LAS format 8)
las.nir    # uint16 — near-infrared
```

### **HIGH VALUE: classification-based filtering**
```python
# Filter to building points only — eliminates terrain, vegetation, noise
las = laspy.read("scan.laz")
pts = np.stack([las.x, las.y, las.z], axis=1)
colors = np.stack([las.red, las.green, las.blue], axis=1) if las.point_format.has_rgb else None

# Keep only Building (6) and optionally Bridge (17)
building_mask = las.classification == 6
if building_mask.sum() > 100:
    pts    = pts[building_mask]
    if colors is not None:
        colors = colors[building_mask]
    print(f"  → {len(pts):,} building points (filtered from {len(las.x):,} total)")
```

### Header information
```python
las.header.version                     # LAS version (e.g. "1.4")
las.header.point_format                # point format number
las.header.point_count                 # total points
las.header.x_scale, las.header.x_offset  # coordinate precision
las.header.mins, las.header.maxs       # bounding box [xmin,ymin,zmin], [xmax,ymax,zmax]
las.header.creation_year
```

### Writing
```python
new_las = laspy.create(point_format=6, file_version="1.4")
new_las.x = pts[:, 0]
new_las.y = pts[:, 1]
new_las.z = pts[:, 2]
new_las.write("output.las")
```

### Extra bytes / custom dimensions
```python
# Check if extra dims exist
print(las.point_format.extra_dims_names)

# Add extra dimension
las.add_extra_dims([
    laspy.ExtraBytesParams(name="height_above_ground", type=np.float32)
])
```

---

## 15. Library H: Three.js (Frontend)

**Docs:** https://threejs.org/docs/  
**Files:** `frontend/src/three/scene.ts`, `frontend/src/components/ViewerScene.tsx`

### PLYLoader
```javascript
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const loader = new PLYLoader();
loader.load(url, (geometry) => {
    // geometry is BufferGeometry
    // geometry.attributes.position  — Float32 XYZ
    // geometry.attributes.color     — Float32 RGB (if present in PLY)
    // geometry.attributes.normal    — Float32 (if present)
});

// Custom property mapping
loader.setPropertyNameMapping({ diffuse_red: 'red', diffuse_green: 'green', diffuse_blue: 'blue' });

// Parse from ArrayBuffer directly
const geometry = loader.parse(arrayBuffer);
```

**PLY format requirements for Three.js:**
- Properties must be `float` (32-bit), NOT `double` (64-bit) — already fixed
- Colour properties: `red`, `green`, `blue` as `uchar` (0–255)
- Binary little-endian is fastest to parse

### PointsMaterial — all properties
```javascript
const mat = new THREE.PointsMaterial({
    size:           0.12,    // world-space point size
    sizeAttenuation: true,   // false = fixed pixel size regardless of distance
    color:          0x38bdf8, // base colour (multiplied with vertexColors)
    vertexColors:   true,    // use per-point colour from geometry
    map:            null,    // texture for point shape (circle sprite, etc.)
    alphaMap:       null,    // grayscale texture controlling opacity per point
    transparent:    true,
    opacity:        0.85,
    depthWrite:     false,   // prevents z-fighting with floor plane
    depthTest:      true,
    blending:       THREE.NormalBlending,  // also: AdditiveBlending for glow effect
    fog:            true,
    toneMapped:     true,
    alphaTest:      0.0,     // discard points below this alpha
    alphaToCoverage: false,  // MSAA-based alpha (WebGL2 only)
});
```

### Points object
```javascript
const points = new THREE.Points(geometry, material);
points.renderOrder = 0;       // higher = rendered later (on top)
points.frustumCulled = true;  // false = always render even if out of view
points.matrixAutoUpdate = true;
points.visible = true;
```

### Height-based colour gradient (currently uses flat colour)
```javascript
// In scene.ts — add height colouring when no vertex colours in PLY:
const positions = geometry.attributes.position.array;
const colours   = new Float32Array(positions.length);  // RGB per point

const zMin = /* compute from positions */;
const zMax = /* compute from positions */;

for (let i = 0; i < positions.length; i += 3) {
    const t = (positions[i + 2] - zMin) / (zMax - zMin);  // 0=floor, 1=ceiling
    // Blue → cyan → green gradient
    colours[i]     = 0;          // R
    colours[i + 1] = t;          // G  
    colours[i + 2] = 1 - t * 0.5; // B
}

geometry.setAttribute('color', new THREE.BufferAttribute(colours, 3));
material.vertexColors = true;
```

### LineSegments for plan display
```javascript
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.Float32BufferAttribute(flat_pts, 3));

const material = new THREE.LineBasicMaterial({
    color:       0xffffff,
    linewidth:   2,    // NOTE: WebGL ignores linewidth > 1 on most platforms
    transparent: false,
    depthTest:   true,
});

const lines = new THREE.LineSegments(geometry, material);
lines.renderOrder = 1;  // render on top of point cloud
```

**For thick lines on WebGL:** Use `THREE.Line2` from `three/addons/lines/Line2.js` which works around the WebGL linewidth limitation:
```javascript
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const mat = new LineMaterial({ color: 0xffffff, linewidth: 3 }); // pixels, not world units
```

---

## 16. Tuning Cheat Sheet

Edit these two files — uvicorn auto-reloads on save.

### `backend/app/pipeline/runner.py`

```python
# Line 64 — scan density (biggest speed lever)
merge_scans(scan_paths, voxel_size=0.03)  # 0.05 for 2× faster, less detail

# Line 85 — RANSAC wall band height
extract_wall_band(scan_pcd, floor, band_low_m=0.75, band_high_m=1.80)
# Raise band_high_m to 2.50 for taller buildings

# Line 92-93 — viewer PLY height + density
_wall_height_slice(..., low_offset=0.20, high_offset=2.50)
write_decimated_ply(..., target_points=200_000)  # 500K for richer view
```

### `backend/app/pipeline/scanplan.py`

```python
# generate_plan_from_projection() function signature
grid_res_m        = 0.10   # 0.05 = 2× detail, 4× slower
min_wall_length_m = 2.00   # 1.00 = finds shorter walls
max_lines         = 120    # 60 = fewer, 200 = more

# Height band — inside function body
floor_z + 0.20  →  floor_z + 2.80    # room height band

# Canny thresholds — search for "cv2.Canny"
threshold1 = 50   # lower = more edges
threshold2 = 150  # keep ~3× ratio to threshold1

# Hough — search for "HoughLinesP"
threshold    = 40   # ↑ = fewer lines  ↓ = more lines
maxLineGap   = 6    # px, 1px = 10cm at default resolution

# Manhattan filter — search for "_filter_to_dominant_orientations"
tolerance_deg = 15.0   # ↑ allows diagonal walls

# Room occupancy — _rooms_from_occupancy function signature
cell_size         = 0.50    # 0.30 = finer, 0.75 = coarser + bridges bigger gaps
min_room_area_m2  = 10.0    # ↓ to 5.0 to find small bathrooms
iterations (dilate) = 12    # ↑ to 20 for very spread-out scans
```

### `backend/app/pipeline/slicing.py`

```python
# detect_floor scoring band — search for "_wall_content_score"
((z > floor_h + 0.30) & (z < floor_h + 3.00))
# Change 3.00 → 2.20 for low-ceiling buildings
```

---

## 17. Upgrade Roadmap

Prioritised by impact vs. effort.

### 🔴 High impact, low effort

**A. Use LAS classification to filter outdoor terrain**
```python
# ingest.py — add to _load_las()
building_mask = las.classification == 6  # ASPRS "Building"
if building_mask.sum() > 100:
    pts = pts[building_mask]
```
This single change would fix every floor detection failure caused by outdoor terrain.

**B. Use `skimage.measure.approximate_polygon` for room contours**
```python
# scanplan.py — replace current step subsampling
from skimage.measure import approximate_polygon
contour_simplified = approximate_polygon(contour, tolerance=2.0)
```
Gives geometrically correct simplification instead of lossy subsampling.

**C. Add `cv2.GaussianBlur` before Canny**
```python
# scanplan.py generate_plan_from_projection() — before Canny
img_smooth = cv2.GaussianBlur(img_filled, ksize=(5,5), sigmaX=1.5)
edges = cv2.Canny(img_smooth, 50, 150)
```
Smoother edges → fewer fragmented Hough segments.

**D. Replace suboptimal dilation with `morphologyEx(MORPH_CLOSE)`**
```python
# One call instead of dilate + erode:
img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
```

### 🟡 High impact, medium effort

**E. `pcd.detect_planar_patches()` instead of iterative `segment_plane()`**
```python
patches = scan_pcd.detect_planar_patches(
    normal_variance_threshold_deg=60,
    coplanarity_deg=75,
    outlier_ratio=0.75,
)
# Returns all planar regions at once — faster and more complete than loop
```

**F. Watershed for room segmentation**
```python
# In _rooms_from_occupancy():
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

dist = distance_transform_edt(interior > 0)
peaks = peak_local_max(dist, min_distance=10, labels=interior > 0)
markers = np.zeros_like(interior, dtype=np.int32)
for i, (r, c) in enumerate(peaks):
    markers[r, c] = i + 1
labels = watershed(-dist, markers, mask=interior > 0)
```

**G. alphashape for building outline display**
```python
import alphashape
alpha = alphashape.optimizealpha(pts_2d[:1000])  # sample for speed
outline = alphashape.alphashape(pts_2d, alpha)
# Use as the building boundary polygon — shown as outer ring in viewer
```

**H. Statistical outlier removal before merge**
```python
# merge.py — add before concatenation
for i, cloud in enumerate(clouds):
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    clouds[i] = cl
```

### 🟢 Lower priority, future work

**I. Coloured ICP when LAZ files have RGB data**
```python
result = o3d.pipelines.registration.registration_colored_icp(
    source, target, max_correspondence_distance=0.05
)
```

**J. DBSCAN to identify and filter outdoor noise clusters**
```python
labels = scan_pcd.cluster_dbscan(eps=1.0, min_points=50)
labels = np.asarray(labels)
# Find largest cluster — that's the building interior
main_cluster = np.bincount(labels[labels >= 0]).argmax()
scan_pcd = scan_pcd.select_by_index(np.where(labels == main_cluster)[0])
```

**K. Three.js Line2 for thick plan lines on all platforms**
Replaces `THREE.LineSegments` with `THREE.Line2` to get actual 3px lines on all GPUs (WebGL ignores `linewidth > 1`).

**L. `hidden_point_removal` for cleaner top-down viewer**
```python
pcd_visible, pt_map = scan_pcd.hidden_point_removal(
    camera_location=[centroid_x, centroid_y, 100],  # top-down camera
    radius=500,
)
```
