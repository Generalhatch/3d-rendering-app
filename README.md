# AlignAI — Scan-to-Plan Alignment

Automatically aligns 3D LiDAR scans to an architectural floor plan (DXF) — or generates a floor plan directly from the scan when no DXF is available. Returns confidence scores, room overlays, wall-fixture markers, and an optional AI sanity check.

**Status:** MVP — localhost demo only. No auth, no cloud deployment.

---

## Quick Start

```bash
# Prerequisites: Python 3.11+, Node 20+
git clone <repo> && cd alignai
cp .env.example .env          # optionally add OPENROUTER_API_KEY

npm run setup                 # one-time: Python venv + node_modules
npm run dev                   # starts backend :8000 + frontend :5173
```

Open **http://localhost:5173** and drop your files in.

---

## Daily Dev Commands

These are the only commands you need day-to-day:

```bash
npm run dev          # start everything (recommended)
npm run dev:be       # backend only  → http://localhost:8000
npm run dev:fe       # frontend only → http://localhost:5173
```

`npm run dev` runs both processes in one terminal with color-coded, labeled output:

```
[BE] INFO:  Uvicorn running on http://0.0.0.0:8000
[FE]   VITE v5.4.21  ready in 491ms
[BE] INFO:  POST /api/jobs 200
```

- `[BE]` lines are cyan (backend/Python), `[FE]` lines are magenta (frontend/React).
- Backend has `--reload`: any Python file save auto-restarts it instantly.
- Frontend has Vite HMR: any React/TS save reflects in the browser instantly.
- `Ctrl+C` kills both.

---

## Two Modes

### With a DXF plan (full alignment)
Drop one `.dxf` + one or more `.laz` files → the scan aligns to the plan, you get a confidence %, room labels from the DXF, and a snap animation.

### Without a DXF plan (scan-only)
Drop only `.laz` files → the pipeline generates a 2D floor plan directly from the detected wall planes. Faster (~15s vs ~30s), room labels are auto-generated ("Space 01", "Space 02"…).

---

## Multi-Scan Support

Each room of a floor is often a separate `.laz` file. Drop all of them at once — the pipeline merges them into a single unified point cloud before processing.

- Files are automatically deduplicated by name.
- Memory-safe: each scan is downsampled immediately after loading (3cm voxel), full-res originals are preserved untouched in `data/uploads/`.
- Pre-registered scans (already in a common site frame) are detected automatically and concatenated. Unregistered scans are aligned via FPFH+RANSAC+ICP.

---

## Reading Logs

### Backend errors
Watch the `[BE]` lines in your terminal. Every Python exception prints a full traceback there:

```
[BE] ERROR:   Exception in ASGI application
[BE] Traceback (most recent call last):
[BE]   File ".../pipeline/runner.py", line 42, in run_pipeline
[BE]     merge_result = merge_scans(scan_paths, ...)
[BE] ValueError: Could not extract any wall segments from scan.
```

### Frontend errors
Open browser DevTools (`Cmd+Option+I`):

| Tab | What to check |
|---|---|
| **Console** | JS errors, unhandled promise rejections |
| **Network** | Click any red request → **Response** tab shows the raw FastAPI error |

Quick filter: type `/api` in the Network filter box to show only backend requests.

---

## Setup From Scratch

```bash
# One-time full setup
npm run setup

# Or individually
npm run setup:be    # Python venv + pip install
npm run setup:fe    # npm install in frontend/
```

---

## Running Tests

```bash
# Backend unit tests (pytest)
cd backend && .venv/bin/pytest tests/ -v

# Generate synthetic test data (DXF + PLY) for testing without real scans
cd backend && .venv/bin/python scripts/generate_synthetic_test_data.py

# Frontend type-check
cd frontend && npx tsc --noEmit
```

---

## Environment Variables

Copy `.env.example` to `.env`. The app works without any API keys — the AI review step is the only optional feature that needs one.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | (none) | Optional — AI vision review of the alignment overlay |
| `AI_REVIEW_MODEL` | `anthropic/claude-sonnet-4-6` | Vision model used for the AI review |
| `BACKEND_PORT` | `8000` | FastAPI / Uvicorn port |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed frontend origins |

---

## Project Structure

```
alignai/
├── backend/                   FastAPI + Open3D (Python 3.11)
│   ├── app/
│   │   ├── pipeline/          Core processing stages
│   │   │   ├── merge.py       Multi-scan loading + registration
│   │   │   ├── ingest.py      LAS/LAZ/PLY/E57 → Open3D
│   │   │   ├── slicing.py     Floor detection + wall band extraction
│   │   │   ├── segment.py     RANSAC wall plane detection
│   │   │   ├── align.py       Principal axes + ICP alignment
│   │   │   ├── scanplan.py    Generate floor plan from scan (no-DXF mode)
│   │   │   ├── rooms.py       DXF room polygon extraction
│   │   │   ├── fixtures.py    Wall protrusion detection (DBSCAN)
│   │   │   ├── confidence.py  Alignment confidence scoring
│   │   │   ├── ai_review.py   OpenRouter vision review
│   │   │   ├── export.py      JSON + DXF export
│   │   │   └── runner.py      Pipeline orchestrator (calls all of the above)
│   │   ├── routes/            FastAPI route handlers
│   │   ├── models/            Pydantic schemas
│   │   ├── storage.py         SQLite job metadata + filesystem
│   │   └── sse.py             Server-Sent Events (progress streaming)
│   ├── tests/                 pytest unit tests
│   └── scripts/               generate_synthetic_test_data.py
│
├── frontend/                  React 18 + Vite + Three.js + Tailwind
│   └── src/
│       ├── components/        UploadZone, ConfidencePanel, ProcessingLog, etc.
│       ├── three/             Three.js scene, room overlays, fixture markers
│       ├── state/             Zustand store
│       └── api/client.ts      All backend API calls
│
├── data/                      Runtime data (gitignored)
│   ├── uploads/               Original scan + plan files (never modified)
│   ├── artifacts/             Decimated PLY for viewer, overlay PNG
│   └── results/               aligned.json, aligned.dxf
│
├── package.json               Root — dev scripts + concurrently
└── .env                       Secrets (copy from .env.example)
```

---

## File Formats

| Type | Formats |
|---|---|
| Plan | `.dxf` — LINE, LWPOLYLINE, POLYLINE, TEXT, MTEXT |
| Scan | `.las`, `.laz`, `.ply`, `.e57` |

`.dwg` and raster PDF are planned for v2.

---

## Troubleshooting

**`npm run dev` — backend says `Application startup complete` but browser shows a network error**
The frontend proxies `/api` → `http://localhost:8000`. Make sure both are running (you should see both `[BE]` and `[FE]` lines).

**`pip install open3d` fails**
Open3D requires Python 3.11 and a recent pip:
```bash
python3 --version          # must be 3.11.x
pip install --upgrade pip setuptools wheel
npm run setup:be
```

**`No horizontal floor plane detected`**
The scan may be very sparse or oriented Y-up. The pipeline auto-detects both Z-up and Y-up, but a near-empty scan will fail. Check the Processing Log for the point count — you need at least ~10k points.

**`Could not extract any wall segments from scan` (scan-only mode)**
The pipeline couldn't find planar walls. This happens if the scan is an outdoor scene, a single room with very few points, or a highly curved space. Try uploading a DXF plan instead, which bypasses wall detection.

**AI review not appearing**
Add `OPENROUTER_API_KEY` to `.env` and restart the backend. The alignment pipeline works completely without it — the AI review is an optional final step.

**Old job data causing weird results after code changes**
```bash
npm run clean:data    # wipes uploads/, artifacts/, results/, jobs.sqlite
```

**Upload fails with a 500 / "No space left on device"**
Your disk is full. LiDAR scans are large (500–700 MB each). Check space and clean up:
```bash
npm run disk          # shows free space + data/ folder size
npm run clean:data    # frees all stored job data (originals are untouched)
```
Each job you run keeps the original scans in `data/uploads/`. Delete old jobs there when you're done with them.
