# AlignAI — Scan-to-Plan Alignment MVP

Automatically aligns a 3D LiDAR scan of a building floor to its architectural plan (DXF) in under 30 seconds. Displays the result with a confidence score, a cinematic snap animation, color-coded room overlays, and wall-fixture markers.

**Status:** MVP — localhost demo only. No auth, no cloud, no production deployment.

---

## What It Does

1. Drag a plan (`*.dxf`) and a LiDAR scan (`*.las`, `*.ply`, or `*.e57`) into the browser.
2. The pipeline detects floors, extracts a clean wall-height slice, runs RANSAC plane detection, aligns via principal axes + ICP, and resolves 90° rotational ambiguity.
3. The scan snaps into place over the plan with a 1.5s animation.
4. Each room renders with a translucent colored fill, a label from the DXF, and a per-room match-quality badge.
5. Wall fixtures (panels, outlets, trims, thermostats) are detected and marked as colored disks.
6. Optionally, an AI vision model reviews the overlay and gives a plain-English sanity check.

---

## Quick Start

```bash
# Prerequisites: Python 3.11+, Node 20+
git clone <repo> && cd align-mvp
cp .env.example .env
# (optional) Edit .env to add OPENROUTER_API_KEY for the AI review step

make demo
# Opens http://localhost:5173
```

`make demo` installs all dependencies, starts the backend on `:8000` and the frontend on `:5173`.

---

## Manual Setup

```bash
# Backend (FastAPI + Open3D)
make install-backend
make backend

# Frontend (React + Vite + Three.js)
make install-frontend
make frontend
```

---

## Running Tests

```bash
make test-backend            # pytest backend/tests/
make generate-test-data      # generate synthetic PLY + DXF test fixtures
```

---

## Environment Variables

See `.env.example`. The only required variable for the AI review step is `OPENROUTER_API_KEY`. Everything else has sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | (none) | API key for optional AI review. App works without it. |
| `AI_REVIEW_MODEL` | `anthropic/claude-sonnet-4-6` | Vision model for overlay review |
| `BACKEND_PORT` | `8000` | FastAPI port |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed frontend origins |

---

## Architecture

```
frontend/   React + Vite + Tailwind + Three.js (port 5173)
backend/    FastAPI + Uvicorn + Open3D (port 8000)
data/       uploads/, artifacts/, results/, jobs.sqlite  (gitignored)
```

See `IMPLEMENTATION_PLAN.md` for full architecture details and the week-by-week build plan.

---

## Troubleshooting

**`make demo` fails on `pip install open3d`**
Open3D requires Python 3.11 and a recent pip. Try: `pip install --upgrade pip setuptools wheel` then retry.

**Frontend blank after `make frontend`**
Check that the backend is running on port 8000. The frontend proxies `/api` to `http://localhost:8000`.

**`No horizontal floor plane detected` error**
The scan may be Y-up instead of Z-up. The pipeline auto-detects both, but a very sparse scan might fail. Check the scan's point count in the processing log.

**AI review not appearing**
Add `OPENROUTER_API_KEY` to `.env` and click the "AI Review" button in the confidence panel. The main alignment flow works without it.

---

## File Formats Supported

| Type | Formats |
|---|---|
| Plan | DXF (vector — LWPOLYLINE, LINE, POLYLINE, TEXT, MTEXT) |
| Scan | LAS, LAZ, PLY, E57 |

DWG and raster PDF are v2 items.
