# Scan-to-Plan Alignment — Full Codebase Audit & Implementation Plan

> Generated: May 2026  
> Author: AI Audit Pass  
> Covers: all 51 source files across `backend/`, `frontend/`, and `shared/`  
> Last Updated: May 19 2026 — Phase 6 complete (45/45 items through Phase 6)

---

## 1. Overview

### Purpose
A desktop web application that ingests multi-file LiDAR scans (LAS/LAZ/PLY/E57), generates a 2D floor plan from the 3D point cloud data, and optionally aligns the generated plan to an uploaded DXF architectural drawing.

### Technology Stack
| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript (strict), Vite, Tailwind CSS, Three.js |
| State | Zustand (UI state), direct `useState` hooks |
| Backend | FastAPI + Python 3.12, Uvicorn |
| Processing | Open3D 0.18, NumPy, OpenCV 4.13, scikit-image 0.26, scipy, Shapely 2.1, laspy 2.5, alphashape 1.3 |
| Database | SQLite (via `sqlite3` stdlib) |
| Transport | REST + SSE (Server-Sent Events for progress) |
| AI | OpenRouter (claude-sonnet-4-6) for alignment review |

### Architecture Summary
```
Browser (React + Three.js)
    │
    ├── POST /api/jobs              → Upload scans + optional DXF
    ├── GET  /api/jobs/:id/sse      → Real-time progress (SSE, auto-reconnect)
    ├── GET  /api/jobs/:id/scan     → Decimated PLY (Three.js viewer)
    ├── GET  /api/jobs/:id/plan     → Wall segments as GeoJSON
    ├── GET  /api/jobs/:id/rooms    → Room polygons
    ├── GET  /api/jobs/:id/fixtures → Fixture markers
    ├── GET  /api/jobs/:id/outline  → Building exterior alphashape hull (NEW)
    └── GET  /api/jobs/:id/overlay  → Scan-vs-plan overlay PNG

Pipeline (runs in BackgroundTasks, max 2 concurrent via semaphore):
  ingest.py  →  merge.py  →  slicing.py  →  segment.py
       └── (scan-only) scanplan.py  →  fixtures.py  →  alphashape  →  runner.py → SQLite

Cleanup (background cron, every 24 h):
  storage.py cleanup_old_jobs() → removes raw uploads for jobs > N days old
```

---

## 2. Current Feature Status

### 2.1 File Structure
```
3d-rendering-app/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app init + CORS
│   │   ├── config.py               pydantic-settings config
│   │   ├── storage.py              SQLite + filesystem helpers
│   │   ├── sse.py                  Server-Sent Events pub/sub
│   │   ├── models/
│   │   │   ├── job.py              JobStatus, JobDetail, JobCreate
│   │   │   ├── room.py             RoomSchema
│   │   │   ├── fixture.py          FixtureSchema
│   │   │   └── result.py           AlignmentResult
│   │   ├── routes/
│   │   │   ├── jobs.py             Main API endpoints
│   │   │   └── export.py           JSON/DXF export
│   │   └── pipeline/
│   │       ├── runner.py           Orchestrator — calls all stages
│   │       ├── ingest.py           LAS/LAZ/PLY/E57 → Open3D PCD  ← UPDATED
│   │       ├── merge.py            Multi-scan merge + ICP          ← UPDATED
│   │       ├── slicing.py          Floor detection + wall band
│   │       ├── segment.py          3D RANSAC wall planes
│   │       ├── scanplan.py         2D plan + room generation        ← UPDATED
│   │       ├── align.py            Scan-to-DXF alignment
│   │       ├── rooms.py            DXF room extraction
│   │       ├── fixtures.py         Wall fixture detection
│   │       ├── confidence.py       Alignment quality scoring
│   │       ├── export.py           Output file writing
│   │       └── ai_review.py        OpenRouter vision review
│   ├── tests/                      Jest unit tests
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 Root component + data loading
│   │   ├── api/client.ts           Typed API client
│   │   ├── state/jobStore.ts       Zustand store
│   │   ├── three/
│   │   │   ├── scene.ts            Three.js AlignmentScene class
│   │   │   ├── pointCloudLoader.ts PLY loader
│   │   │   ├── planLoader.ts       GeoJSON → Float32Array
│   │   │   ├── roomOverlay.ts      Room polygon rendering
│   │   │   ├── fixtureMarkers.ts   Fixture 3D markers
│   │   │   ├── snapAnimation.ts    Alignment snap animation
│   │   │   ├── roomColors.ts       Color config
│   │   │   └── fixtureColors.ts    Color config
│   │   └── components/
│   │       ├── UploadZone.tsx      Drag-and-drop file input
│   │       ├── ProcessingPanel.tsx SSE progress log display
│   │       ├── ViewerScene.tsx     Canvas mount + scene lifecycle
│   │       ├── ConfidencePanel.tsx Results + controls + AI review
│   │       ├── RoomDetailPanel.tsx Clicked room details
│   │       ├── FixturePanel.tsx    Clicked fixture details
│   │       └── ManualControls.tsx  Manual transform controls
│   └── vite.config.ts
└── DEV_GUIDE.md                    Library parameter reference
```

### 2.2 Supported Features
#### Core Features
- [x] Upload multiple LAS/LAZ scan files (no DXF required)
- [x] Upload optional DXF floor plan for aligned mode
- [x] Multi-scan merge with pre-registration detection
- [x] Statistical outlier removal per scan  ← NEW
- [x] LAS classification-based filtering (class 6 = building)  ← NEW
- [x] Floor plane detection (scoring by wall content)
- [x] 3D RANSAC wall plane extraction
- [x] 2D floor plan generation (Hough Lines on occupancy grid)
- [x] DBSCAN outdoor noise cluster removal  ← NEW
- [x] GaussianBlur + morphologyEx for cleaner edges  ← NEW
- [x] Watershed-based room segmentation  ← NEW
- [x] approximate_polygon contour simplification  ← NEW
- [x] Fixture detection
- [x] Real-time SSE progress updates
- [x] Three.js point cloud viewer with room overlays
- [x] Manual transform override
- [x] OpenRouter AI alignment review
- [x] Export as JSON + DXF

#### Features Added (Phase 1–3)
- [x] File size validation (2 GB limit + HTTP 413)  ← Phase 1
- [x] Concurrent job queue limit (semaphore, max 2)  ← Phase 2
- [x] Job history panel (localStorage sidebar)  ← Phase 2
- [x] Room type classification from shape/eccentricity  ← Phase 2
- [x] SSE auto-reconnect with exponential backoff  ← Phase 2
- [x] Band height controls in upload UI  ← Phase 2
- [x] Overlay PNG wired to AI review  ← Phase 2
- [x] Automatic storage cleanup (configurable age, 24 h cron)  ← Phase 3
- [x] alphashape building outline overlay  ← Phase 3
- [x] Height-gradient colour on point cloud when no vertex RGB  ← Phase 3

#### Features Not Yet Implemented
- [ ] Watershed distance transform for fixture detection
- [ ] Multi-floor support
- [ ] Live re-processing with updated parameters

### 2.3 API Endpoints
| Method | Endpoint | Purpose | Status |
|---|---|---|---|
| POST | `/api/jobs` | Create job + start pipeline | ✅ Working |
| GET | `/api/jobs/:id` | Job metadata + status | ✅ Working |
| GET | `/api/jobs/:id/sse` | Real-time progress stream | ✅ Working |
| GET | `/api/jobs/:id/scan` | Decimated PLY for viewer | ✅ Working |
| GET | `/api/jobs/:id/plan` | Wall segments as GeoJSON | ✅ Working |
| GET | `/api/jobs/:id/rooms` | Room polygons | ✅ Working |
| GET | `/api/jobs/:id/fixtures` | Fixture markers | ✅ Working |
| GET | `/api/jobs/:id/outline` | Building exterior hull (GeoJSON Polygon) | ✅ Working (Phase 3) |
| GET | `/api/jobs/:id/overlay` | Scan-vs-plan overlay PNG | ✅ Working (Phase 2) |
| POST | `/api/jobs/:id/ai` | OpenRouter alignment review | ✅ Working |
| POST | `/api/jobs/:id/manual` | Apply manual transform (full merged cloud) | ✅ Working (fixed Phase 2) |
| POST | `/api/jobs/:id/approve` | Mark job approved | ✅ Working |
| GET | `/api/jobs/:id/export/json` | Export aligned.json | ✅ Working |
| GET | `/api/jobs/:id/export/dxf` | Export aligned.dxf | ✅ Working |

### 2.4 Database Schema
```sql
-- Table: jobs (SQLite, file: data/jobs.sqlite)
CREATE TABLE jobs (
    job_id          TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'queued',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    scan_filename   TEXT NOT NULL DEFAULT '',  -- legacy: first scan only
    scan_filenames  TEXT NOT NULL DEFAULT '[]',-- JSON array of all scan names
    plan_filename   TEXT NOT NULL DEFAULT '',
    error_message   TEXT,
    result_json     TEXT,   -- full pipeline result as JSON blob
    elapsed_s       REAL
)
-- NOTE: No separate rooms/fixtures tables. All stored in result_json blob.
```

---

## 3. Infrastructure Issues

### Critical Issues 🔴

#### C1. No upload file size limit → OOM / DoS risk
- **Location:** `backend/app/routes/jobs.py:61–69`
- **Description:** Files are read into memory with `await scan_file.read()` with no size check. A 10 GB LAZ file will be loaded entirely into RAM before any processing starts.
- **Impact:** Server OOM, process kill, job failure. Can be exploited accidentally by users with large scans.
- **Recommended Fix:**
```python
# BEFORE — in create_job_endpoint
content = await scan_file.read()
if len(content) == 0:
    raise HTTPException(400, ...)

# AFTER — add size guard before reading
MAX_SCAN_BYTES = 2 * 1024 ** 3  # 2 GB
scan_file.file.seek(0, 2)  # seek to end
file_size = scan_file.file.tell()
scan_file.file.seek(0)
if file_size > MAX_SCAN_BYTES:
    raise HTTPException(413, f"Scan file too large: {file_size / 1e9:.1f} GB (max 2 GB)")
content = await scan_file.read()
```

#### C2. No concurrent job limit → RAM exhaustion with multiple users
- **Location:** `backend/app/routes/jobs.py:72`
- **Description:** Every POST to `/api/jobs` immediately fires a `BackgroundTask`. With 6 scans at 3cm voxel, one job uses ~1.5 GB RAM. Three simultaneous jobs = OOM.
- **Impact:** Server crash for all users while heavy jobs run.
- **Recommended Fix:**
```python
# In app/main.py — add a semaphore gate
import asyncio
_JOB_SEMAPHORE = asyncio.Semaphore(2)  # max 2 concurrent pipeline jobs

# In runner.py — wrap run_pipeline with the semaphore
async def run_pipeline_guarded(*args, **kwargs):
    async with _JOB_SEMAPHORE:
        await asyncio.to_thread(run_pipeline, *args, **kwargs)
```

#### C3. Memory leak — resize event listener never removed
- **Location:** `frontend/src/three/scene.ts:48`
- **Description:** `window.addEventListener('resize', ...)` registers a new handler every time `AlignmentScene` is constructed. `dispose()` only stops the render loop — it never calls `window.removeEventListener`.
- **Impact:** Each "New job" click adds another resize handler. After 5 new jobs, 5 listeners fire on every resize, each calling `this.renderer.setSize()` on a potentially stale canvas.
- **Recommended Fix:**
```typescript
// BEFORE
window.addEventListener('resize', () => this._onResize(canvas));

// AFTER — store the handler and remove it in dispose()
private _resizeHandler: () => void;

constructor(canvas: HTMLCanvasElement) {
  // ...
  this._resizeHandler = () => this._onResize(canvas);
  window.addEventListener('resize', this._resizeHandler);
}

dispose() {
  this.stopRenderLoop();
  window.removeEventListener('resize', this._resizeHandler);
  canvas.removeEventListener('click', this._onClick);
  this.renderer.dispose();
}
```

#### C4. Canvas click handler never removed
- **Location:** `frontend/src/three/scene.ts:47`
- **Description:** `canvas.addEventListener('click', this._onClick)` added in constructor. Never removed in `dispose()`. Same memory / duplicate-handler issue as C3.
- **Fix:** Add `canvas.removeEventListener('click', this._onClick)` to `dispose()`. Requires storing the canvas reference.

#### C5. Manual transform endpoint broken for multi-scan jobs
- **Location:** `backend/app/routes/jobs.py:276–280` (`_find_scan_file`)
- **Description:** The `/manual` endpoint re-runs alignment using a single scan file. For jobs with 6 merged scans, it uses only the first file — completely ignoring the merged point cloud. The result is a transform optimized against 1/6th of the data.
- **Impact:** Manual re-alignment produces incorrect results for multi-scan jobs.
- **Recommended Fix:** Store the merged+processed PLY path in the job result, and reload that rather than re-processing raw scans:
```python
# In runner.py — add to result_payload:
"merged_ply": str(decimated_path),  # viewer PLY = pre-processed merged cloud

# In jobs.py — manual endpoint reads merged PLY instead of raw scan:
merged_ply = Path(result.get("merged_ply", ""))
if merged_ply.exists():
    scan_pcd = load_point_cloud(merged_ply)
else:
    scan_pcd = load_point_cloud(_find_scan_file(job_id))
```

---

### High Priority Issues 🟡

#### H1. `floor_z` hardcoded to 0 in App.tsx → room overlay at wrong height
- **Location:** `frontend/src/App.tsx:102`
- **Description:** `floor_z: 0` is passed directly. The actual floor Z comes from `result_payload["floor_z"]` in runner.py but is never pulled into `App.tsx`.
- **Impact:** Room overlays are placed at Z=0 regardless of actual floor height. For scans where the floor is at Z=-3m, the overlays float visibly above the point cloud.
- **Recommended Fix:**
```typescript
// BEFORE
const fullResult: FullResult = {
  floor_z: 0,   // ← WRONG
  ...
};

// AFTER — read from jobDetail which is already fetched
const fullResult: FullResult = {
  floor_z: (jobDetail as any).floor_z ?? 0,
  ...
};

// Also in storage.py load_job() — expose floor_z from result_json:
floor_z = parsed.get("floor_z", 0.0)
// Add to JobDetail model and return it
```

#### H2. New room fields missing from TypeScript interface
- **Location:** `frontend/src/api/client.ts:25–33`
- **Description:** The watershed upgrade adds `eccentricity`, `orientation_deg`, and `solidity` to every room dict. The TypeScript `Room` interface doesn't declare these. They're silently stripped when TypeScript processes the JSON, so room classification logic (corridor detection etc.) can't use them.
- **Recommended Fix:**
```typescript
// In frontend/src/api/client.ts
export interface Room {
  id: string;
  label: string;
  category: 'office' | 'bathroom' | 'hallway' | 'common' | 'unknown';
  polygon_2d: [number, number][];
  centroid: [number, number];
  area_m2: number;
  match_quality: number | null;
  // Shape analysis from watershed regionprops
  eccentricity?: number;       // 0=square, 1=line (use >0.8 to classify as corridor)
  orientation_deg?: number;    // major axis angle
  solidity?: number;           // 1=convex, <0.7=highly irregular
}
```

#### H3. GPU memory leak — material not disposed on scene clear
- **Location:** `frontend/src/App.tsx:298–303` (`_clearScene`)
- **Description:** `_clearScene` calls `geometry.dispose()` but never disposes the `material`. GPU texture/shader memory leaks on every "New job" click.
- **Recommended Fix:**
```typescript
// BEFORE
function _clearScene(scene: AlignmentScene) {
  if (scene.scanCloud) {
    scene.scene.remove(scene.scanCloud);
    scene.scanCloud.geometry.dispose();  // ← missing material dispose
    scene.scanCloud = null;
  }
}

// AFTER
function _clearScene(scene: AlignmentScene) {
  if (scene.scanCloud) {
    scene.scene.remove(scene.scanCloud);
    scene.scanCloud.geometry.dispose();
    (scene.scanCloud.material as THREE.Material).dispose();
    scene.scanCloud = null;
  }
  if (scene.planLines) {
    scene.scene.remove(scene.planLines);
    scene.planLines.geometry.dispose();
    (scene.planLines.material as THREE.Material).dispose();
    scene.planLines = null;
  }
}
```

#### H4. SQLite concurrent write risk — no connection pool
- **Location:** `backend/app/storage.py:14–16`
- **Description:** `check_same_thread=False` is set but no mutex protects concurrent writes. FastAPI background tasks run in a thread pool. Two jobs completing simultaneously can cause `database is locked` errors.
- **Recommended Fix:** Add a `threading.Lock` at module level:
```python
import threading
_DB_LOCK = threading.Lock()

def _conn() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # WAL mode allows concurrent reads
    conn.execute("PRAGMA busy_timeout=5000")   # 5s timeout instead of immediate fail
    return conn
```

#### H5. `_2d_line_ransac` is dead code — 87 lines never called
- **Location:** `backend/app/pipeline/scanplan.py:285–369`
- **Description:** After switching to OpenCV HoughLinesP, `_2d_line_ransac` is never called. Dead code bloats the module and confuses future maintainers.
- **Recommended Fix:** Delete lines 285–369 of `scanplan.py`. The function is fully superseded.

#### H6. Overlay PNG generated but never displayed in UI
- **Location:** `backend/app/pipeline/runner.py:248–254` + frontend
- **Description:** `render_overlay_png()` writes a PNG to `artifacts_dir` and the path is stored in `result_payload["overlay_png"]`. The frontend `ConfidencePanel` has no UI to display or link to it.
- **Impact:** ~2–5 seconds of pipeline time wasted generating an asset nobody sees.
- **Options:** Either wire it into the AI Review section, or skip generation until it's used.

#### H7. No file type validation beyond extension
- **Location:** `backend/app/routes/jobs.py:63–68`
- **Description:** Extension checking (`f.suffix.lower()` in `load_point_cloud`) is the only validation. A malformed or misnamed file crashes deep in the pipeline, leaving the job stuck in "processing" status with a cryptic error.
- **Recommended Fix:** Add magic-byte validation before writing to disk:
```python
LAS_MAGIC = b"LASF"
PLY_MAGIC = b"ply"

def _validate_scan_magic(content: bytes, filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext in (".las", ".laz") and not content[:4] == LAS_MAGIC:
        raise HTTPException(400, f"{filename} is not a valid LAS/LAZ file")
    if ext == ".ply" and not content[:3] == PLY_MAGIC:
        raise HTTPException(400, f"{filename} is not a valid PLY file")
```

---

### Low Priority Issues 🟢

#### L1. SSE connection has no reconnect logic
- **Location:** `frontend/src/api/client.ts:139–154`
- **Description:** `EventSource.onerror` isn't passed to a reconnect handler. If the connection drops mid-processing, the frontend silently stops receiving progress, and the job appears hung forever.
- **Fix:** Add an `onerror` → auto-reconnect with backoff, or poll `GET /jobs/:id` as fallback.

#### L2. Progress logs grow unbounded in Zustand store
- **Location:** `frontend/src/state/jobStore.ts:90–97`
- **Description:** Every SSE event appends to `progressLogs` array with no cap. A 15-minute pipeline job can generate 200+ log entries, all held in memory.
- **Fix:** Cap at last 100 entries: `progressLogs: [...state.progressLogs.slice(-99), newEntry]`

#### L3. `band_low_m` / `band_high_m` not exposed in frontend
- **Location:** `frontend/src/api/client.ts:101–113`
- **Description:** The API accepts `band_low_m` and `band_high_m` parameters. The frontend always sends the defaults (0.75 / 1.80). Users with unusually low ceilings or raised floors have no way to adjust these.
- **Fix:** Add two number inputs to `UploadZone` or an "Advanced settings" section.

#### L4. `_safe_filename` permits path traversal with repeated `..`
- **Location:** `backend/app/routes/jobs.py:294–295`
- **Description:** `"".join(c for c in name if c.isalnum() or c in "._-")` permits `..` since both `.` chars pass individually. Combined with job-id subdirectory, this is low risk but should use `Path(name).name` instead.
- **Fix:** `return Path(name).name.replace("..", "_")`

#### L5. No job history UI
- **Description:** Users can only access the current job. Refreshing the page loses the job ID. No list of past jobs is shown.
- **Fix:** Store recent job IDs in `localStorage` and show a job history panel.

#### L6. Processing overlay message says "Aligning scan to plan…" for scan-only jobs
- **Location:** `frontend/src/App.tsx:204`
- **Description:** The spinner text is always "Aligning scan to plan…" regardless of mode.
- **Fix:** `{isProcessing ? (scanMode ? 'Generating floor plan…' : 'Aligning scan to plan…') : null}`

---

## 4. Bugs Identified

### Confirmed Bugs 🐛

#### Bug 1: Manual transform uses single scan for multi-scan jobs
- **Location:** `backend/app/routes/jobs.py:200, 276–280`
- **Steps to Reproduce:**
  1. Upload 6 LAZ scan files (no DXF)
  2. Wait for processing to complete
  3. Use manual transform controls
  4. POST to `/api/jobs/:id/manual` with a new matrix
- **Expected:** Re-alignment computed on full merged 6-scan cloud
- **Actual:** Re-alignment computed on the first scan file only
- **Severity:** High
- **Fix:** See Critical Issue C5 above

#### Bug 2: Room overlays at wrong height (Z=0) for non-zero floor scans
- **Location:** `frontend/src/App.tsx:102`
- **Steps to Reproduce:**
  1. Upload scans where the building floor is at Z≠0 (e.g., UTM-projected scans before centering)
  2. Complete processing
- **Expected:** Room polygons appear at floor height, aligned with point cloud
- **Actual:** Room polygons rendered at Z=0, floating above or below the point cloud
- **Severity:** Medium (always affects scan-only mode since floor_z is never passed)
- **Fix:** See High Priority Issue H1

#### Bug 3: `_clear_scene` leaves GPU materials allocated
- **Location:** `frontend/src/App.tsx:298–303`
- **Steps to Reproduce:**
  1. Complete a job (point cloud visible)
  2. Click "New job" 5 times
  3. Check browser GPU memory in DevTools
- **Expected:** Memory returns to baseline
- **Actual:** GPU memory grows ~10–20 MB per cycle (unreclaimed `PointsMaterial` + `LineBasicMaterial`)
- **Severity:** Medium (affects long sessions / frequent users)
- **Fix:** See High Priority Issue H3

#### Bug 4: Resize listener accumulates across "New job" clicks
- **Location:** `frontend/src/three/scene.ts:48`
- **Severity:** Medium
- **Fix:** See Critical Issue C3

### Potential Bugs ⚠️

#### PB1: `AlignmentScene.fitToScene()` includes room overlay and fixture markers in bbox
- **Location:** `frontend/src/three/scene.ts:146`
- `new THREE.Box3().setFromObject(this.scene)` captures everything including room polygon meshes, which are flat (Z≈0). This causes the camera to position itself to fit the room overlays rather than the 3D point cloud, which can make the view feel zoomed out.
- **Suggested Fix:** Compute bounding box only from `this.scanCloud` if it exists.

#### PB2: Hough line tolerance filter disabled for <4 segments
- **Location:** `backend/app/pipeline/scanplan.py` — `_filter_to_dominant_orientations` check `if len(segments) >= 4`
- If only 2–3 Hough segments are found (very sparse scan), the Manhattan filter is skipped entirely. Those 2–3 segments could be at random angles. This is correct behaviour but should emit a log warning so it's visible in the progress stream.

---

## 5. Feature Recommendations

### Quick Wins (1–2 hours each)

#### QW1. Room category from shape metrics
- **Description:** Use the `eccentricity` and `area_m2` already returned by watershed to auto-classify rooms before they display.
- **User Value:** "Hallway / Corridor" shows correctly instead of sequential cycling through the label list
- **Implementation:**
```python
# In _rooms_from_occupancy(), replace label cycling with:
def _classify_room(area_m2: float, eccentricity: float) -> tuple[str, str]:
    if eccentricity > 0.85:
        return "Hallway / Corridor", "hallway"
    if area_m2 < 8:
        return "Bathroom / Storage", "bathroom"
    if area_m2 > 60:
        return "Open Plan / Lobby", "common"
    return "Office / Meeting", "office"
```
- **Files:** `backend/app/pipeline/scanplan.py`

#### QW2. Expose band height in UI (Advanced settings accordion)
- **Description:** Let users set `band_low_m` and `band_high_m` before upload
- **User Value:** Fix wall detection for non-standard ceiling heights without code changes
- **Files:** `frontend/src/components/UploadZone.tsx`, `frontend/src/api/client.ts`

#### QW3. Cap progress log entries
- **Description:** `progressLogs.slice(-100)` in Zustand reducer
- **User Value:** Prevents UI lag on long jobs
- **Files:** `frontend/src/state/jobStore.ts:90`

#### QW4. Fix processing spinner label for scan-only mode
- **Description:** Show "Generating floor plan…" when no DXF uploaded
- **Files:** `frontend/src/App.tsx:204`

#### QW5. Delete `_2d_line_ransac` dead code
- **Description:** 87 lines of unused code. Remove entirely.
- **Files:** `backend/app/pipeline/scanplan.py:285–369`

---

### Medium Features (2–5 days each)

#### M1. alphashape building outline overlay
- **Description:** Generate the building exterior hull using `alphashape.optimizealpha()` and render it as a distinct colour line over the point cloud. Replaces the bounding-box fallback as a visible building footprint.
- **Value:** Gives users an immediate visual of the full building shape, even before rooms are refined.
- **Files:** `scanplan.py` (add `_generate_building_outline()`), `scene.ts` (add `loadOutline()`), `planLoader.ts`

#### M2. Job history in sidebar
- **Description:** Store recent job IDs in `localStorage`. Show a list of the last 10 jobs with status, scan count, and timestamp.
- **Value:** Users can return to a previous scan without re-uploading.
- **Files:** `frontend/src/state/jobStore.ts`, new `JobHistoryPanel.tsx`

#### M3. Automatic storage cleanup
- **Description:** Jobs older than N days (configurable) should have their raw uploads deleted. Results can be kept. Add a background cron task.
- **Value:** Prevents disk exhaustion on long-running servers.
- **Files:** `backend/app/storage.py`, `backend/app/main.py` (startup event)

#### M4. Height-gradient point colouring when no vertex colors
- **Description:** When PLY has no RGB (scans without colour), apply a programmatic height-based gradient in Three.js instead of flat cyan. Already planned in `scene.ts` comments.
- **Value:** Gives structural depth cues without RGB data — users can see floor vs. ceiling heights.
- **Files:** `frontend/src/three/scene.ts:70–87`

#### M5. SSE auto-reconnect
- **Description:** Detect `EventSource` errors and reconnect with exponential backoff. Fall back to polling `GET /jobs/:id` if SSE is unavailable.
- **Value:** Processing jobs survive temporary network drops / server restarts without appearing hung.
- **Files:** `frontend/src/api/client.ts`

---

### Major Features (1–2 weeks each)

#### MJ1. Room classification using ML / rule engine
- **Description:** Use room shape properties (area, eccentricity, solidity, adjacency) plus a rule engine or small classifier to output meaningful room types with confidence scores.
- **Value:** "Conference Room (87%)" is far more useful than "Office / Meeting #3"

#### MJ2. Multi-floor support
- **Description:** Detect multiple floor planes (currently only the highest-scoring one is used) and allow users to select which floor to analyse.
- **Value:** Buildings with basement + ground + upper floors currently only process one level.

#### MJ3. Live re-processing with updated parameters
- **Description:** Expose a "Re-run with new settings" button that re-triggers just the plan/room stages (not the expensive merge/downsample) with updated parameters.
- **Value:** Users can tune Hough threshold, min_wall_length, etc. without waiting 15 minutes.

---

## 6. Implementation Guide

### Priority 1: Security & Stability (implement before any user demos)

#### Step 1.1 — File size limit (`backend/app/routes/jobs.py`)
```python
# Add after FastAPI imports at top of jobs.py
MAX_SCAN_MB = 2048   # 2 GB

# Replace the bare content = await scan_file.read() block:
scan_file.file.seek(0, 2)
size_bytes = scan_file.file.tell()
scan_file.file.seek(0)
if size_bytes > MAX_SCAN_MB * 1024 * 1024:
    raise HTTPException(413, f"File too large ({size_bytes / 1e9:.1f} GB, max {MAX_SCAN_MB / 1024} GB)")
content = await scan_file.read()
```

#### Step 1.2 — SQLite WAL mode + busy timeout (`backend/app/storage.py`)
```python
def _conn() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
```

#### Step 1.3 — Fix resize + click memory leaks (`frontend/src/three/scene.ts`)
```typescript
private _resizeHandler!: () => void;
private _canvas!: HTMLCanvasElement;

constructor(canvas: HTMLCanvasElement) {
  this._canvas = canvas;
  // ... existing setup ...
  this._resizeHandler = () => this._onResize(canvas);
  window.addEventListener('resize', this._resizeHandler);
  canvas.addEventListener('click', this._onClick);
}

dispose() {
  this.stopRenderLoop();
  window.removeEventListener('resize', this._resizeHandler);
  this._canvas.removeEventListener('click', this._onClick);
  this.renderer.dispose();
}
```

#### Step 1.4 — Fix material dispose in `_clearScene` (`frontend/src/App.tsx`)
```typescript
function _clearScene(scene: AlignmentScene) {
  const disposeObj = (obj: THREE.Points | THREE.LineSegments | null) => {
    if (!obj) return;
    scene.scene.remove(obj);
    obj.geometry.dispose();
    (Array.isArray(obj.material) ? obj.material : [obj.material])
      .forEach((m) => m.dispose());
  };
  disposeObj(scene.scanCloud); scene.scanCloud = null;
  disposeObj(scene.planLines); scene.planLines = null;
  if (scene.roomOverlay) { scene.scene.remove(scene.roomOverlay.group); scene.roomOverlay = null; }
  if (scene.fixtureMarkers) { scene.scene.remove(scene.fixtureMarkers.group); scene.fixtureMarkers = null; }
}
```

---

### Priority 2: Data Correctness (fix before processing real client scans)

#### Step 2.1 — Pass `floor_z` from backend to frontend

**`backend/app/storage.py`** — expose `floor_z` in `load_job`:
```python
floor_z = parsed.get("floor_z", 0.0)
# Add floor_z to JobDetail model and return it
```

**`backend/app/models/job.py`** — add field:
```python
floor_z: float = 0.0
```

**`frontend/src/api/client.ts`** — add to `JobDetail`:
```typescript
floor_z?: number;
```

**`frontend/src/App.tsx:102`** — use actual value:
```typescript
floor_z: jobDetail.floor_z ?? 0,
```

#### Step 2.2 — Add new Room fields to TypeScript interface
```typescript
// frontend/src/api/client.ts — Room interface
eccentricity?: number;
orientation_deg?: number;
solidity?: number;
```

#### Step 2.3 — Fix manual transform endpoint for multi-scan jobs
Add `merged_ply` path to `result_payload` in `runner.py`:
```python
result_payload = {
  ...
  "merged_ply": str(decimated_path),   # ← add this line
}
```
Then in `jobs.py` manual endpoint, prefer `merged_ply` over raw scan file.

---

### Priority 3: Quality of Life

#### Step 3.1 — Delete dead code (`scanplan.py`)
Remove `_2d_line_ransac` function (lines 285–369). Run `npm run test:run` + `cd server && npm test` to confirm nothing breaks.

#### Step 3.2 — Cap progress logs
```typescript
// frontend/src/state/jobStore.ts — appendLog reducer
appendLog: (event) =>
  set((state) => ({
    progressLogs: [
      ...state.progressLogs.slice(-99),  // keep last 100
      { stage: event.stage, message: event.message, progress: event.progress, timestamp: Date.now() },
    ],
    overallProgress: event.progress,
  })),
```

#### Step 3.3 — Room auto-classification from shape
```python
# backend/app/pipeline/scanplan.py — replace label cycling
def _classify_room(area_m2: float, eccentricity: float, solidity: float) -> tuple[str, str]:
    if eccentricity > 0.85:
        return "Hallway / Corridor", "hallway"
    if area_m2 < 7.0:
        return "Bathroom / Storage", "bathroom"
    if area_m2 > 50.0 and solidity > 0.85:
        return "Open Plan / Lobby", "common"
    if area_m2 > 20.0:
        return "Conference Room", "office"
    return "Office", "office"

# Then in _rooms_from_occupancy():
label, category = _classify_room(area_m2, region.eccentricity, region.solidity)
rooms.append({
  ...,
  "label": label,
  "category": category,
  ...
})
```

#### Step 3.4 — Fix fitToScene to use only scan cloud bounds
```typescript
// frontend/src/three/scene.ts — fitToScene()
fitToScene() {
  const target = this.scanCloud ?? this.planLines;
  if (!target) return;
  const box = new THREE.Box3().setFromObject(target);   // ← only scan/plan, not overlays
  // ... rest unchanged
}
```

---

## 7. Testing Checklist

### Unit Tests Needed
- [ ] `test_ingest.py` — LAS with class 6 points: verify non-building pts filtered
- [ ] `test_ingest.py` — LAS with no class 6 points: verify all points kept (no regression)
- [ ] `test_ingest.py` — LAS with intensity but no RGB: verify intensity colors applied
- [ ] `test_merge.py` — statistical outlier removal: inject 10 known outlier points, verify removed
- [ ] `test_scanplan.py` — DBSCAN filter: two clusters, verify only main cluster kept
- [ ] `test_scanplan.py` — watershed: known point grid, verify rooms separated at walls
- [ ] `test_scanplan.py` — `_classify_room`: test each category threshold
- [ ] `test_storage.py` — concurrent writes: 2 threads calling `update_job_result`, verify no corruption

### Integration Tests Needed
- [ ] E2E scan-only with 6 LAZ files: verify >0 rooms detected
- [ ] E2E scan-only with single LAZ: verify plan segments and PLY viewer loads
- [ ] E2E aligned mode: verify confidence_pct returned in expected range
- [ ] Memory: three consecutive "New job" clicks, verify GPU memory doesn't grow

### Manual Verification Checklist
- [ ] Upload 6 LAZ files → pipeline completes → rooms displayed
- [ ] Upload LAZ with DXF → alignment confidence shown → export works
- [ ] "New job" × 5 → no console errors → GPU memory stable
- [ ] Resize browser window → viewer re-renders correctly (not duplicated)
- [ ] Drop connection mid-processing → re-connect → job still shows result

---

## 8. Performance Metrics

### Current Performance (M1 Max, 32 GB, 6 scans)
| Stage | Time | Notes |
|---|---|---|
| Load + downsample 6 LAZ | ~8–10 min | laspy read + voxel 3cm |
| Statistical outlier removal | +~2 min | NEW — per-scan |
| Merge + center | ~30 sec | concatenate |
| Floor detection | ~45 sec | 12 RANSAC iterations |
| Wall band + RANSAC | ~1 min | 3 attempts |
| 2D projection + Hough | ~15 sec | includes DBSCAN filter |
| Watershed room detection | ~5 sec | |
| PLY write | ~20 sec | 200K pts |
| **Total** | **~13–16 min** | scan-only mode |

### Target Performance
| Stage | Target | Method |
|---|---|---|
| Load + downsample | <6 min | Use `laspy.open()` chunked read for large files |
| RANSAC wall | <30 sec | Use `detect_planar_patches()` (single call vs. 12 iterations) |
| Total scan-only | <10 min | Chunked read + planar patches |

---

## 9. Next Steps

### Phase 1 — Completed (May 19 2026) ✅

1. [x] Fix C3: resize listener leak (`scene.ts`)
2. [x] Fix H3: material dispose in `_clearScene`
3. [x] Fix H1: `floor_z` passed correctly to frontend
4. [x] Fix H2: new Room TypeScript fields
5. [x] Fix L6: spinner label for scan-only mode
6. [x] Fix C1: file size validation (2 GB limit + HTTP 413)
7. [x] Fix C4: canvas click handler cleanup
8. [x] Fix H4: SQLite WAL + busy timeout
9. [x] Fix H5: delete `_2d_line_ransac` dead code (87 lines removed)
10. [x] Fix H3: GPU material dispose verified (geometry + material array-safe)
11. [x] Fix H7: file magic-byte validation (LAS/PLY)
12. [x] Implement QW3/L2: cap progress logs at 100 entries
13. [x] Fix L4: `_safe_filename` path traversal hardened
14. [x] Fix PB1: `fitToScene` uses scan cloud bounds only (not room overlays)

### Phase 2 — Completed (May 19 2026) ✅
15. [x] Fix C2: concurrent job semaphore (max 2 pipelines — OOM risk)
16. [x] Fix C5: manual transform uses merged PLY for multi-scan jobs
17. [x] Fix H6: wire overlay PNG to AI review UI
18. [x] Fix L1: SSE auto-reconnect with exponential backoff
19. [x] Implement QW1: room auto-classification from shape metrics
20. [x] Implement L3/QW2: expose `band_low_m` / `band_high_m` in upload UI
21. [x] Implement L5/M2: job history panel (localStorage + sidebar)

### Phase 3 — Completed (May 19 2026) ✅
22. [x] Implement M1: alphashape building outline overlay
23. [x] Implement M3: automatic storage cleanup (configurable age)
24. [x] Implement M4: height-gradient point colours (no-RGB fallback)

### Phase 4 — Deep LiDAR Fixes (Section 11) — Completed (May 19 2026) ✅
25. [x] Fix LAZ-1/2: classification filter wrong for interior scans
26. [x] Fix LAZ-3: intensity percentile normalization
27. [x] Fix LAZ-4: RGB 8-bit vs 16-bit auto-detection
28. [x] Fix LAZ-5: chunked streaming for large LAS files
29. [x] Fix LAZ-6: analytical voxel estimate (remove 20× binary search)
30. [x] Fix LAZ-7: E57 reads all scans, not just index 0
31. [x] Fix LAZ-8: log LAS header info for coordinate debugging

### Phase 5 — Room Generation Fixes (Section 12) — Completed (May 19 2026) ✅
32. [x] Fix RG-4: min room area 10 → 4 m² (bathrooms always missing now)
33. [x] Fix RG-8: room labels from shape metrics (done in Phase 2 / QW1)
34. [x] Fix RG-2: morphology close bridges doorways (reduce iterations)
35. [x] Fix RG-3: watershed `min_distance` suppresses small rooms
36. [x] Fix RG-5: centroid outside non-convex rooms
37. [x] Fix RG-7: stable room IDs across runs (centroid-hash)
38. [x] Fix RG-9: `binary_fill_holes` creates phantom outdoor rooms
39. [x] Fix RG-1: Y-up scan wall plane projects to wrong plane
40. [x] Fix RG-6: polygon self-intersection validation
41. [x] Fix RG-10/11: polygonize fallback path missing shape metadata + centroid

### Phase 6 — Major Features — Completed (May 19 2026) ✅
42. [x] Implement MJ1: room ML classification with confidence scores
43. [x] Implement MJ2: multi-floor support
44. [x] Implement MJ3: live re-processing with updated parameters
45. [x] Write unit tests from section 7

---

## 10. Progress Tracker

> Last updated: May 19 2026 — Phase 6 complete (45/45 items through Phase 6)

| ID | Issue | Priority | Est. Effort | Status |
|---|---|---|---|---|
| C1 | File size limit (2 GB + HTTP 413) | 🔴 Critical | 30 min | ✅ Done |
| C2 | Concurrent job limit | 🔴 Critical | 2 hr | ✅ Done |
| C3 | Resize listener leak | 🔴 Critical | 30 min | ✅ Done |
| C4 | Click listener leak | 🔴 Critical | 15 min | ✅ Done |
| C5 | Manual transform multi-scan | 🔴 Critical | 3 hr | ✅ Done |
| H1 | floor_z propagation | 🟡 High | 1 hr | ✅ Done |
| H2 | Room TypeScript fields | 🟡 High | 15 min | ✅ Done |
| H3 | GPU material dispose | 🟡 High | 30 min | ✅ Done |
| H4 | SQLite WAL mode | 🟡 High | 15 min | ✅ Done |
| H5 | Delete dead code (`_2d_line_ransac`) | 🟡 High | 15 min | ✅ Done |
| H6 | Wire overlay PNG to UI | 🟡 High | 2 hr | ✅ Done |
| H7 | File magic validation | 🟡 High | 1 hr | ✅ Done |
| L1 | SSE reconnect | 🟢 Low | 3 hr | ✅ Done |
| L2 | Cap progress logs | 🟢 Low | 10 min | ✅ Done |
| L3 | band_low/high in UI | 🟢 Low | 2 hr | ✅ Done |
| L4 | Safe filename path traversal | 🟢 Low | 10 min | ✅ Done |
| L5 | Job history panel | 🟢 Low | 1 day | ✅ Done |
| L6 | Spinner label (scan-only vs aligned) | 🟢 Low | 10 min | ✅ Done |
| QW1 | Room auto-classification | Quick Win | 1 hr | ✅ Done |
| QW5 | Delete dead code | Quick Win | 15 min | ✅ Done |
| PB1 | fitToScene uses scan cloud bounds only | Quick Win | 15 min | ✅ Done |
| M1 | alphashape outline | Medium | 2 days | ✅ Done |
| M2 | Job history | Medium | 2 days | ✅ Done (L5) |
| M3 | Storage cleanup | Medium | 1 day | ✅ Done |
| M4 | Height gradient colours | Medium | 3 hr | ✅ Done |
| M5 | SSE auto-reconnect | Medium | 3 hr | ✅ Done (L1) |
| LAZ-1/2 | Classification filter wrong for interior scans | 🔴 Critical | 1 hr | ✅ Done |
| LAZ-3 | Intensity percentile normalization | 🟡 High | 30 min | ✅ Done |
| LAZ-4 | RGB 8-bit vs 16-bit auto-detection | 🟡 High | 30 min | ✅ Done |
| LAZ-5 | Chunked streaming for large LAS files | 🟡 High | 3 hr | ✅ Done |
| LAZ-6 | Analytical voxel estimate (remove 20× binary search) | 🟢 Low | 30 min | ✅ Done |
| LAZ-7 | E57 reads all scans, not just index 0 | 🟡 High | 1 hr | ✅ Done |
| LAZ-8 | Log LAS header info for coord debugging | 🟢 Low | 15 min | ✅ Done |
| RG-4 | min_room_area 10 → 4 m² (bathrooms missing) | 🔴 Critical | 15 min | ✅ Done |
| RG-8 | Room labels from shape metrics, not cycling list | 🔴 Critical | 1 hr | ✅ Done (QW1) |
| RG-2 | Morphology close bridges doorways | 🟡 High | 1 hr | ✅ Done |
| RG-3 | Watershed min_distance suppresses small rooms | 🟡 High | 1 hr | ✅ Done |
| RG-5 | Centroid outside non-convex rooms | 🟡 High | 1 hr | ✅ Done |
| RG-7 | Unstable room IDs across runs | 🟡 High | 30 min | ✅ Done |
| RG-9 | binary_fill_holes phantom outdoor rooms | Medium | 2 hr | ✅ Done |
| RG-1 | Y-up scan wall plane projects to wrong plane | Medium | 30 min | ✅ Done |
| RG-6 | Polygon self-intersection validation | Medium | 1 hr | ✅ Done |
| RG-10/11 | Polygonize fallback missing shape metadata + centroid | 🟢 Low | 1 hr | ✅ Done |
| RG-12 | Coordinate frame not documented in export | 🟢 Low | 1 hr | ☐ TODO |
| MJ1 | Room ML classification with confidence scores | Major | 2 weeks | ✅ Done |
| MJ2 | Multi-floor support (detect_all_floors + /floors + /reprocess) | Major | 2 weeks | ✅ Done |
| MJ3 | Live re-processing (POST /reprocess + Re-process UI) | Major | 1 week | ✅ Done |
| Tests | Unit tests: test_scanplan_classify, test_scanplan, test_storage_concurrent | Section 7 | 2 hr | ✅ Done |

---

## 11. LAZ / LiDAR File Processing — Deep Analysis

Every issue in this section maps to a specific line in `backend/app/pipeline/ingest.py` or `merge.py`.

---

### 11.1 Classification Filter Threshold Is Wrong

**Location:** `ingest.py:46`

```python
# CURRENT — incorrect threshold
building_pts = int((cls == 6).sum())
if building_pts >= 500:
    mask = cls == 6
```

**Problem:** 500 is an absolute count with no reference to the total. For a 7 million point scan, 500 building-classified points is 0.007% of the file — nearly nothing. A scan could have 600 points spuriously classified as class 6 (scanner startup artefacts, stray returns) while the remaining 7 million interior points are class 0 (Unclassified). Our code would then keep only those 600 spurious points and discard all real data.

**A deeper problem:** ASPRS class 6 ("Building") is defined for **exterior aerial/mobile mapping** workflows (drone surveys, mobile LiDAR vans). For **interior terrestrial scans** from Leica RTC360, FARO Focus, Matterport Pro3, or NavVis — which is exactly the equipment your users have — points are almost universally classified as class 0 (Never Classified) or class 1 (Unclassified). The class 6 filter is designed for the wrong workflow and will silently do nothing for 95%+ of interior scans while potentially causing data loss for the remaining 5%.

**Recommended Fix:**
```python
# AFTER — percentage-based, gracefully handles interior scans
try:
    cls = np.asarray(las.classification, dtype=np.int32)
    n_class6 = int((cls == 6).sum())
    pct_class6 = n_class6 / max(n_total, 1)

    if pct_class6 >= 0.10:
        # Clearly an exterior survey with classification — use class 6
        mask = cls == 6
    elif pct_class6 >= 0.02:
        # Some classification exists — use building + unclassified
        mask = (cls == 6) | (cls == 1) | (cls == 0)
    # else: interior scan with no meaningful classification — keep all points
except Exception:
    pass  # no classification dimension — keep all points
```

**Also consider filtering out known-noise classes regardless:**
```python
# Always remove ASPRS noise classes if classification exists
NOISE_CLASSES = {7, 18}   # Low noise (7), High noise (18)
if cls is not None:
    noise_mask = ~np.isin(cls, list(NOISE_CLASSES))
    mask = mask & noise_mask
```

---

### 11.2 Intensity Normalization Produces Dark/Invisible Colors

**Location:** `ingest.py:76`

```python
# CURRENT — divides by max, collapses dynamic range
intens_norm = intens / max(float(intens.max()), 1.0)
```

**Problem 1 — Outlier single bright return:** One scanner startup pulse or retroreflector hit produces an intensity value of 65535. Every other point gets normalized relative to that one outlier, compressing all values to 0.00–0.03 range. The result is a viewer that shows nearly-black points for the entire scan.

**Problem 2 — 8-bit vs 16-bit intensity range:** LAS format specifies `intensity` as uint16 (0–65535). But many older or budget scanners output 8-bit values (0–255) stored in the uint16 field. The current code divides by 65535, so an 8-bit intensity of 128 becomes 0.002 — essentially invisible. You'd need to detect the actual range and normalize accordingly.

**Recommended Fix — percentile normalization:**
```python
# AFTER — robust normalization ignoring outliers
if not color_set:
    try:
        intens = np.asarray(las.intensity, dtype=np.float64)[mask]
        if intens.max() > intens.min():
            # Clip to 2nd–98th percentile to eliminate outlier returns
            p2  = np.percentile(intens, 2)
            p98 = np.percentile(intens, 98)
            intens_norm = np.clip((intens - p2) / max(p98 - p2, 1.0), 0.0, 1.0)
        else:
            intens_norm = np.zeros_like(intens)
        r_ch = intens_norm * 0.40
        g_ch = intens_norm * 0.85
        b_ch = np.clip(0.55 + intens_norm * 0.45, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(np.vstack([r_ch, g_ch, b_ch]).T)
    except Exception:
        pass
```

---

### 11.3 RGB Colors Can Be Near-Zero When Scanner Uses 8-bit Range

**Location:** `ingest.py:65–68`

```python
# CURRENT — always divides by 65535
r = np.asarray(las.red,   dtype=np.float64)[mask] / 65535.0
g = np.asarray(las.green, dtype=np.float64)[mask] / 65535.0
b = np.asarray(las.blue,  dtype=np.float64)[mask] / 65535.0
```

**Problem:** If the scanner outputs 8-bit RGB (max value 255) stored in a uint16 LAS field, dividing by 65535 yields colors ~0.004 — essentially black in the viewer. Leica RTC360 exports 8-bit color. FARO Focus X330 exports 16-bit. NavVis VLX exports 8-bit. There is no standard.

**Recommended Fix — auto-detect bit depth:**
```python
r_raw = np.asarray(las.red,   dtype=np.float64)[mask]
g_raw = np.asarray(las.green, dtype=np.float64)[mask]
b_raw = np.asarray(las.blue,  dtype=np.float64)[mask]

# Auto-detect: if max channel value ≤ 255, it's 8-bit despite uint16 container
max_val = max(r_raw.max(), g_raw.max(), b_raw.max())
scale = 255.0 if max_val <= 255.0 else 65535.0

r = r_raw / scale
g = g_raw / scale
b = b_raw / scale
pcd.colors = o3d.utility.Vector3dVector(np.vstack([r, g, b]).T)
color_set = True
```

---

### 11.4 `laspy.read()` Loads Entire File Into RAM

**Location:** `ingest.py:32`

```python
las = laspy.read(str(path))  # entire file into RAM before any processing
```

**Problem:** `laspy.read()` reads the complete file into memory before returning. For a 2 GB LAZ file this means 2 GB lives in RAM simultaneously with the Open3D PointCloud objects being built from it — peak usage ~4 GB for a single large scan file, before any downsampling occurs.

**Recommended Fix — chunked streaming for large files:**
```python
import os

def _load_las(path: Path) -> o3d.geometry.PointCloud:
    import laspy

    file_size = os.path.getsize(path)
    CHUNK_THRESHOLD = 500 * 1024 * 1024  # 500 MB

    if file_size > CHUNK_THRESHOLD:
        return _load_las_chunked(path)

    las = laspy.read(str(path))
    # ... existing logic


def _load_las_chunked(path: Path) -> o3d.geometry.PointCloud:
    """Stream-read large LAS/LAZ files in 1M-point chunks to control peak RAM."""
    import laspy
    pts_chunks: list[np.ndarray] = []
    intens_chunks: list[np.ndarray] = []

    with laspy.open(str(path)) as reader:
        for chunk in reader.chunk_iterator(chunk_size=1_000_000):
            # Apply classification filter per chunk
            cls = np.asarray(chunk.classification, dtype=np.int32)
            mask = np.ones(len(chunk.x), dtype=bool)
            n_class6 = (cls == 6).sum()
            if n_class6 / max(len(cls), 1) >= 0.10:
                mask = cls == 6

            pts_chunks.append(np.vstack([
                np.asarray(chunk.x)[mask],
                np.asarray(chunk.y)[mask],
                np.asarray(chunk.z)[mask],
            ]).T.astype(np.float64))
            # ... accumulate intensity/color per chunk

    pts = np.vstack(pts_chunks)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd
```

---

### 11.5 E57 Loader Only Reads First Scan

**Location:** `ingest.py:91`

```python
data = e57.read_scan(0, ...)  # always scan index 0
```

**Problem:** E57 files frequently contain multiple scans registered together (Leica Cyclone Register 360 exports all scans in a single E57). `read_scan(0)` only reads the first one. For a building with 10 registered scan positions in one E57 file, we'd process ~10% of the data.

**Recommended Fix:**
```python
def _load_e57(path: Path) -> o3d.geometry.PointCloud:
    import pye57
    e57 = pye57.E57(str(path))
    all_pts: list[np.ndarray] = []

    for i in range(e57.scan_count):
        try:
            data = e57.read_scan(i, intensity=True, colors=True,
                                  ignore_missing_fields=True)
            pts = np.vstack([
                data["cartesianX"], data["cartesianY"], data["cartesianZ"]
            ]).T
            all_pts.append(pts.astype(np.float64))
        except Exception:
            continue  # skip malformed individual scans

    if not all_pts:
        raise ValueError(f"No readable scans in E57 file: {path}")

    combined = np.vstack(all_pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined)
    return pcd
```

---

### 11.6 `_estimate_voxel_for_target` Runs 20 Expensive Voxel Downsamples

**Location:** `ingest.py:159–172`

```python
for _ in range(20):
    mid = (lo + hi) / 2.0
    ds = pcd.voxel_down_sample(mid)   # expensive operation × 20
```

**Problem:** Binary search for target voxel size calls `voxel_down_sample` 20 times. For a 7M point cloud, each call takes ~0.5s, meaning this binary search alone costs ~10s. The function is called inside `write_decimated_ply` which runs after the full pipeline — adding 10s of pure overhead at the end.

**Recommended Fix — analytical estimate, only one voxel call:**
```python
def _estimate_voxel_for_target(pcd: o3d.geometry.PointCloud, target: int) -> float:
    """Analytical estimate: voxel size scales ~as (N/target)^(1/3) for uniform clouds."""
    n = len(pcd.points)
    if n <= target:
        return 0.001   # no downsampling needed

    pts = np.asarray(pcd.points)
    bbox = pts.max(axis=0) - pts.min(axis=0)
    volume = float(np.prod(bbox))
    # Points/volume density → voxel size needed for target points
    density = n / max(volume, 1e-9)
    voxel_estimate = (1.0 / density * (n / target)) ** (1/3)

    # One verification downsample (single call instead of 20)
    ds = pcd.voxel_down_sample(voxel_estimate)
    ratio = len(ds.points) / max(target, 1)
    if ratio > 1.2:
        voxel_estimate *= ratio ** (1/3)  # adjust if estimate was too small
    return voxel_estimate
```

---

### 11.7 Missing: Return Number Filtering

**Available attribute:** `las.return_number`, `las.number_of_returns`

**Not currently used.** Last returns in a multi-return pulse penetrate surfaces (ground through gaps, signals that passed through glass). For interior LiDAR scanning this is less relevant, but for exterior facade scanning:
- First returns = building exterior surface (what you want for walls)
- Last returns = ground, interior objects seen through doorways/windows

**High-value addition for exterior/mobile LiDAR scans:**
```python
# In _load_las(), after the classification mask:
try:
    ret_num = np.asarray(las.return_number, dtype=np.int32)
    n_ret = np.asarray(las.number_of_returns, dtype=np.int32)
    # First returns of single-return pulses = most reliable wall surfaces
    # Include: return_number == 1 (first return) OR single-return pulses
    wall_return_mask = (ret_num == 1) | (n_ret == 1)
    if wall_return_mask.sum() / max(n_total, 1) >= 0.40:
        mask = mask & wall_return_mask
except Exception:
    pass
```

---

### 11.8 Missing: Header Validation and Coordinate Precision Logging

**Available:** `las.header.x_scale`, `las.header.x_offset`, `las.header.point_count`, `las.header.mins`, `las.header.maxs`

No header information is currently logged or validated. This makes it very hard to diagnose coordinate system issues.

**Recommended addition:**
```python
# In _load_las(), add logging after read:
try:
    h = las.header
    print(f"  LAS {h.version}: {h.point_count:,} pts, "
          f"scale={h.x_scale:.6f}, "
          f"bbox X:{h.mins[0]:.1f}→{h.maxs[0]:.1f} "
          f"Y:{h.mins[1]:.1f}→{h.maxs[1]:.1f} "
          f"Z:{h.mins[2]:.1f}→{h.maxs[2]:.1f}")
except Exception:
    pass
```

This single line tells you immediately if a scan is in UTM (X=500000), geographic (X=-117.5), or local origin (X=0) — critical for diagnosing coordinate frame issues.

---

### 11.9 LAZ Processing — Summary of Issues

| ID | Issue | Location | Impact | Effort |
|---|---|---|---|---|
| LAZ-1 | Classification threshold uses absolute count | `ingest.py:46` | Silent data loss for non-class-6 interior scans | 30 min |
| LAZ-2 | Interior terrestrial scans never have class 6 | `ingest.py:43–48` | Filter does nothing OR discards all data | 1 hr |
| LAZ-3 | Intensity normalization collapses from outliers | `ingest.py:76` | Viewer shows all-dark points | 30 min |
| LAZ-4 | RGB detection doesn't handle 8-bit scanners | `ingest.py:65–68` | Colors nearly black for Leica RTC360 | 30 min |
| LAZ-5 | `laspy.read()` loads entire file into RAM | `ingest.py:32` | OOM for >2 GB scans | 3 hr |
| LAZ-6 | Binary search = 20 voxel downsamples | `ingest.py:163–170` | +10s overhead per job at export | 30 min |
| LAZ-7 | E57 only reads scan index 0 | `ingest.py:91` | 90% data loss for multi-scan E57 | 1 hr |
| LAZ-8 | No header validation/logging | `ingest.py:32` | Impossible to debug coordinate issues | 15 min |

---

## 12. Room Generation — Deep Analysis

Every issue in this section maps to a specific line in `backend/app/pipeline/scanplan.py`.

---

### 12.1 `generate_plan_from_walls` Projects Wrong Plane for Y-Up Scans

**Location:** `scanplan.py:53`

```python
# CURRENT — always projects to XY regardless of scan up-axis
xy = pts[:, :2]   # takes columns 0 (X) and 1 (Y)
```

**Problem:** For Y-up scans (`floor.axis_idx == 1`), the floor lies in the XZ plane, not the XY plane. Column 1 (Y) is the vertical axis — projecting to XY would be projecting walls onto the wrong plane, producing meaningless room shapes.

Compare with `generate_plan_from_projection` line 156 which correctly handles this:
```python
xy = pts_mid[:, :2] if axis == 2 else pts_mid[:, [0, 2]]   # correct
```

`generate_plan_from_walls` is the preferred path (used when ≥4 wall planes detected), meaning every Y-up scan that successfully finds wall planes generates a broken floor plan silently.

**Fix:**
```python
# In generate_plan_from_walls() — add axis parameter and use it
def generate_plan_from_walls(
    wall_planes: list[WallPlane],
    axis_idx: int = 2,        # ← add this
) -> SyntheticPlan:
    ...
    for wall in wall_planes:
        pts = wall.inlier_points
        # Project to the correct 2D plane
        xy = pts[:, :2] if axis_idx == 2 else pts[:, [0, 2]]   # ← fix this
```

And in `runner.py` pass `axis_idx=floor.axis_idx`.

---

### 12.2 `morphologyEx CLOSE iterations=6` Bridges Doorways

**Location:** `scanplan.py:480`

```python
occ_closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel, iterations=6)
```

**The math:** At `cell_size=0.50m`, kernel `(5,5)` Ellipse, `iterations=6`:
- Each ellipse kernel extends ~2.5 pixels from center
- At 0.50m/px this is a 1.25m reach per iteration
- With 6 iterations in CLOSE: max bridge = ~3.75m

**Problem:** Standard interior doorways are 0.9m–1.2m wide. At 3.75m bridge capability, the CLOSE operation fills every doorway, making adjacent rooms appear as one solid connected area before the watershed seeds are placed. The watershed will then try to separate them, but since the doorway region is solid and contributes to both rooms' distance transforms, the room boundary might be placed in the middle of what should be a doorway — or two rooms with a doorway might only get one seed and merge into a single room.

**Evidence:** This is why users saw "0 rooms extracted" in earlier runs — the massive dilation (old code was 12 iterations) merged everything into one blob before room extraction.

**Recommended Fix — two-phase morphology:**
```python
# Phase 1: Close small intra-wall gaps (scan point scatter)
# Use small kernel + few iterations → bridges gaps <0.3m only
k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
occ_wall_closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k_small, iterations=2)

# Phase 2: Fill room interiors (the rooms themselves have no scan points inside)
occ_filled = binary_fill_holes(occ_wall_closed > 0).astype(np.uint8)
# Do NOT add another morphological operation between fill and watershed.
# Let the distance transform handle room separation naturally.
```

With `(3,3)` kernel, `iterations=2`: bridge = ~0.75m at 0.5m/px. This bridges intra-wall point scatter but NOT doorways.

---

### 12.3 Watershed `min_distance=3m` Suppresses Small Rooms

**Location:** `scanplan.py:491`

```python
min_dist_px = max(4, int(3.0 / cell_size))   # = 6 pixels at cell_size=0.5
coords = peak_local_max(dist, min_distance=min_dist_px, ...)
```

**Problem:** `peak_local_max` with `min_distance=6px` (3m) suppresses any distance-transform peak that's within 3m of a larger peak. This means:
- A bathroom (3×2m, max distance from walls ≈ 1m) has a peak only 1px high
- The adjacent office (5×6m) has a peak 2.5px high
- If the bathroom is within 3m of the office, the bathroom peak is suppressed
- Watershed places no seed in the bathroom → it merges into the office

**Recommended Fix — adaptive min_distance from peak heights:**
```python
# Find all peaks with no minimum distance first
all_coords = peak_local_max(dist, min_distance=1, labels=occ_filled.astype(bool))
if len(all_coords) == 0:
    return []

# Get peak values (room "spaciousness" = distance to nearest wall)
peak_vals = dist[all_coords[:, 0], all_coords[:, 1]]

# Adaptive min_distance: peaks must be at least 1.5m apart
# (smaller than a bathroom ~2m)
min_dist_px = max(3, int(1.5 / cell_size))
coords = peak_local_max(dist, min_distance=min_dist_px, labels=occ_filled.astype(bool))
```

Or even better — use a threshold on peak value to include small room peaks:
```python
# Also include any peak that represents a room >1.5m from walls
# regardless of proximity to other peaks (these are distinct rooms)
small_room_threshold_px = int(1.5 / cell_size)  # 1.5m = smallest valid room half-width
forced_seeds = all_coords[peak_vals >= small_room_threshold_px]
coords = np.vstack([coords, forced_seeds]) if len(coords) > 0 else forced_seeds
coords = np.unique(coords, axis=0)
```

---

### 12.4 `min_room_area_m2 = 10.0` Is Too Large — Misses Bathrooms

**Location:** `scanplan.py:429`

```python
min_room_area_m2: float = 10.0,
```

**Real room sizes:**
| Room type | Typical area |
|---|---|
| Small bathroom | 3–5 m² |
| Large bathroom | 6–9 m² |
| Small office | 9–12 m² |
| Standard office | 12–20 m² |
| Conference room | 20–50 m² |

At 10.0 m², **every bathroom and many small offices are silently discarded**. Users see 2 rooms instead of 8 and think the pipeline is broken.

**Recommended Fix:**
```python
# Lower threshold to 4.0 m² to capture bathrooms
# and add a comment explaining the reasoning
min_room_area_m2: float = 4.0,   # 4m² ≈ 2×2m smallest valid room
```

The tradeoff is more noise regions (scan artefacts) being classified as rooms. These can be filtered by a `min_room_pts` check: require at least 50 scan points inside the region to distinguish real rooms from morphological fill artefacts.

---

### 12.5 Room Centroids Can Be Outside Non-Convex Rooms

**Location:** `scanplan.py:545–547`

```python
cy_px, cx_px = region.centroid    # scikit-image regionprops centroid
cx = float(cx_px) * cell_size + x0
cy = float(cy_px) * cell_size + y0
```

**Problem:** `region.centroid` computes the arithmetic mean of all pixel coordinates. For L-shaped rooms (common in offices with alcoves), the centroid falls in empty space — outside the room polygon entirely. The UI then draws a label or places a click target at a location that's visually wrong.

**Recommended Fix — use Shapely's `representative_point()`:**
```python
# After building the polygon, use Shapely to find a guaranteed interior point
try:
    from shapely.geometry import Polygon
    poly_shapely = Polygon(polygon)
    if poly_shapely.is_valid and not poly_shapely.is_empty:
        rep = poly_shapely.representative_point()  # always inside the polygon
        cx, cy = float(rep.x), float(rep.y)
    # else: fall back to regionprops centroid
except Exception:
    pass   # use regionprops centroid as fallback
```

---

### 12.6 Room Polygons Are Never Validated for Self-Intersections

**Location:** `scanplan.py:537` (after `approximate_polygon`)

`approximate_polygon` with `tolerance=tol_px` simplifies contours using Douglas-Peucker. For complex L-shaped or U-shaped rooms, high tolerance values can produce self-intersecting polygons (where simplification causes two contour edges to cross). Self-intersecting polygons cause:
- Incorrect area calculations (the polygon area formula gives wrong results)
- Frontend Three.js triangulation failures (invalid geometry)
- Shapely operations to fail silently

**Recommended Fix:**
```python
# After building polygon, validate and repair with Shapely
try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    poly_check = Polygon(polygon)
    if not poly_check.is_valid:
        poly_check = make_valid(poly_check)
        # Re-extract coords if make_valid changed geometry type
        if poly_check.geom_type == 'Polygon':
            polygon = list(poly_check.exterior.coords)
        elif poly_check.geom_type == 'MultiPolygon':
            # Take largest sub-polygon
            largest = max(poly_check.geoms, key=lambda p: p.area)
            polygon = list(largest.exterior.coords)
except Exception:
    pass  # keep original polygon if Shapely unavailable
```

---

### 12.7 Room IDs Are Sequential — Change Every Run

**Location:** `scanplan.py:552`

```python
"id": f"room-{idx + 1:03d}",   # sequential counter
```

**Problem:** Watershed seed ordering is not deterministic — it depends on `peak_local_max` output order which depends on the exact distance transform values. Every pipeline run on the same scan produces `room-001` through `room-008` assigned to different physical rooms. If a user labels "room-003" as "Server Room", re-processing makes "room-003" a different physical room.

**Recommended Fix — centroid-based deterministic ID:**
```python
# Generate ID from rounded centroid — same physical room always gets same ID
centroid_hash = abs(hash((round(cx, 0), round(cy, 0)))) % 65536
"id": f"room-{centroid_hash:04x}",   # e.g., "room-3fa2"
```

This isn't perfect (hash collisions possible) but gives stable IDs across runs for rooms whose centroid doesn't move more than 0.5m between processing runs.

---

### 12.8 Room Labels Still Sequential From Static List — Ignoring Shape Data

**Location:** `scanplan.py:553`

```python
_ROOM_LABELS = ["Office / Meeting", "Hallway / Corridor", "Common / Lobby", ...]
"label": _ROOM_LABELS[idx % len(_ROOM_LABELS)],   # cycles through list
```

Despite now having `eccentricity`, `solidity`, and `area_m2` available from the watershed regionprops, labels are still assigned by cycling through the static list in creation order. Room 1 (largest) = "Office / Meeting", room 2 = "Hallway / Corridor", regardless of whether room 2 is actually a large open lobby.

This was identified in QW1 of the main audit. The shape data is right there — it just needs to be used.

**Recommended Fix (copy into code):**
```python
def _classify_room(area_m2: float, eccentricity: float, solidity: float) -> tuple[str, str]:
    """Classify room by shape metrics. Returns (label, category)."""
    if eccentricity > 0.85 and area_m2 < 40:
        return "Hallway / Corridor", "hallway"
    if area_m2 < 7.0:
        return "Bathroom / Storage", "bathroom"
    if area_m2 > 80.0 and solidity > 0.85:
        return "Open Plan / Lobby", "common"
    if area_m2 > 40.0:
        return "Conference Room", "office"
    if solidity < 0.70:
        return "Irregular Space", "unknown"
    return "Office / Meeting", "office"
```

---

### 12.9 `binary_fill_holes` Incorrectly Bridges Disconnected Building Wings

**Location:** `scanplan.py:483`

```python
occ_filled = binary_fill_holes(occ_closed > 0).astype(np.uint8)
```

**Problem:** `scipy.ndimage.binary_fill_holes` fills ALL topologically enclosed holes in the binary image. If a building has two wings connected by an outdoor walkway (or two scanned areas separated by an unscanned courtyard), `binary_fill_holes` treats the gap between them as an enclosed hole and fills it — creating a phantom room in the courtyard. Any outdoor gap fully enclosed by scanned area becomes a false room.

**Example scenario:** A U-shaped building with an interior courtyard. The courtyard is surrounded by scanned walls on three sides. `binary_fill_holes` fills the courtyard, creates a phantom region, and watershed produces a "room" for the outdoor space.

**Recommended Fix — use `cv2.floodFill` from known interior points instead:**
```python
# Instead of binary_fill_holes, use scan point density to identify interior
# Flood-fill only from cells that have actual scan points (not from all gaps)
from scipy.ndimage import binary_fill_holes

# Alternative: use morphological closing to fill only gaps smaller than max_room_fill
# Large gaps = courtyards/outside. Small gaps = room interiors (max_room_fill_m2 = 200m²)
max_fill_px = int(200 / (cell_size ** 2))  # 200m² in pixels
# Use skimage morphology area-based closing (fills holes smaller than max size)
from skimage.morphology import remove_small_holes
occ_filled = remove_small_holes(
    occ_closed > 0,
    area_threshold=max_fill_px,  # only fill interior holes, not large exterior gaps
    connectivity=2,
).astype(np.uint8)
```

---

### 12.10 `_rooms_from_polygonize` Centroid Uses Non-Interior Point

**Location:** `scanplan.py:604`

```python
c = poly.centroid    # Shapely centroid — can be outside for concave polygons
rooms.append({
    ...,
    "centroid": [float(c.x), float(c.y)],
})
```

Same issue as 12.5 — use `poly.representative_point()` which is guaranteed to be inside the polygon.

---

### 12.11 `_rooms_from_polygonize` Missing Shape Metadata

**Location:** `scanplan.py:603–613`

`_rooms_from_polygonize` (the fallback strategy) does not add `eccentricity`, `solidity`, or `orientation_deg` to the room dict. If watershed fails and polygonize succeeds, room classification via shape metrics won't work and the frontend will hit `eccentricity: undefined`.

**Fix:** compute shape metrics from Shapely polygon:
```python
from shapely.geometry import Polygon
import math

poly_shapely = Polygon(poly.exterior.coords)
# Eccentricity approximation from bounding box aspect ratio
minx, miny, maxx, maxy = poly_shapely.bounds
w = maxx - minx
h = maxy - miny
aspect = max(w, h) / max(min(w, h), 0.001)
eccentricity_approx = 1.0 - 1.0 / aspect   # 0=square, 1=very elongated

rooms.append({
    ...,
    "eccentricity": eccentricity_approx,
    "solidity": float(poly.area / poly_shapely.convex_hull.area),
    "orientation_deg": 0.0,  # not computed in polygonize path
})
```

---

### 12.12 Room Polygon Coordinate System Not Documented / Inconsistent

**Location:** `scanplan.py:539–542` vs `_rooms_from_polygonize:609`

Watershed rooms return `[x, y]` coordinates in the scan's centered local frame (meters from cloud centroid). Polygonize rooms also return the same. But the `_building_as_single_room` fallback in `scanplan.py:621–626` returns the plan bounding box, which is also in the same frame.

**The problem:** The coordinate frame is never documented in the data schema. Room polygons are rendered by the frontend as 2D meshes at `floor_z` height in the Three.js scene. The scene uses the **same** centered coordinate frame as the scan data — so this works correctly currently. However:

1. If a user exports `aligned.json` and opens it in another tool, they won't know the coordinates are relative to the cloud centroid
2. Areas computed from polygon coordinates are in scan-space meters, which is correct only if the scan scale is 1.0 (it always is for real LiDAR, but worth noting)

**Recommendation:** Add a `coordinate_frame` field to the export:
```python
result_payload["coordinate_frame"] = {
    "type": "local_centered",
    "origin_in_utm": [float(centroid_x), float(centroid_y), float(centroid_z)]
    # populated from merge_result if we save the centroid offset
}
```

---

### 12.13 Room Generation — Summary of Issues

| ID | Issue | Location | Impact | Effort |
|---|---|---|---|---|
| RG-1 | `generate_plan_from_walls` ignores axis_idx → wrong plane for Y-up | `scanplan.py:53` | Silent wrong results for Y-up scans | 30 min |
| RG-2 | `MORPH_CLOSE iterations=6` bridges doorways | `scanplan.py:480` | Adjacent rooms merge | 1 hr |
| RG-3 | `min_distance=3m` suppresses small room peaks | `scanplan.py:491` | Bathrooms merged into adjacent offices | 1 hr |
| RG-4 | `min_room_area_m2=10.0` filters all bathrooms | `scanplan.py:429` | Missing room types in output | 15 min |
| RG-5 | Centroid outside non-convex rooms | `scanplan.py:545` | Labels/click targets in wrong position | 1 hr |
| RG-6 | No polygon self-intersection validation | `scanplan.py:537` | Invalid geometry → Three.js errors | 1 hr |
| RG-7 | Sequential room IDs change every run | `scanplan.py:552` | Unstable room identity | 30 min |
| RG-8 | Labels still cycle from static list | `scanplan.py:553` | Wrong room type displayed | 1 hr |
| RG-9 | `binary_fill_holes` creates phantom outdoor rooms | `scanplan.py:483` | False rooms in courtyards/gaps | 2 hr |
| RG-10 | Polygonize path missing shape metadata | `scanplan.py:603` | TypeScript type error on fallback path | 30 min |
| RG-11 | Polygonize centroid outside concave polygons | `scanplan.py:604` | Wrong label placement | 30 min |
| RG-12 | Coordinate frame not documented in export | `scanplan.py:539` | External tools can't interpret coordinates | 1 hr |

---

### 12.14 Room Generation — Recommended Priority Order

These 12 items rank by how often they'd silently affect real scans:

```
CRITICAL (affects every run):
  RG-4  min_room_area=10 — every bathroom is missing RIGHT NOW
  RG-8  labels still cycling — room types are always wrong RIGHT NOW

HIGH (affects many buildings):
  RG-2  doorway bridging — connected rooms may merge
  RG-3  small room peak suppression — bathrooms missed even if RG-4 is fixed
  RG-5  centroid outside room — broken UI labels for L-shaped rooms
  RG-7  unstable IDs — user annotations lost on reprocess

MEDIUM:
  RG-9  phantom outdoor rooms — affects U-shaped buildings / courtyards
  RG-6  polygon self-intersection — rare but causes viewer crashes
  RG-1  Y-up scan wall plane axis — only affects Y-up scans (rarer)

LOW:
  RG-10 polygonize shape metadata — only hits fallback path
  RG-11 polygonize centroid — only hits fallback path
  RG-12 coordinate frame doc — developer/integration concern
```
