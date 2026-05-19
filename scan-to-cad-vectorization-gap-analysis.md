# Scan-to-CAD Vectorization — Implementation Plan

> **Source roadmap:** `Scan-to-CAD_Vectorization_Roadmap.docx` (Stevenson Systems, May 2026 — Draft for internal review)
> **Codebase analyzed:** `/Users/tannerhatch/3d-rendering-app` (the "AlignAI" MVP)
> **Decision date:** May 2026
> **Owner:** Tanner Hatch
> **Status:** Active plan — updated after the strategic decisions below

---

## 0. Strategic Direction (Read this first)

### The root problem we are solving (for Stevenson)

> Manual tracing of rasterized scan slices into CAD geometry is the dominant cost driver in their scan-to-CAD deliverables. Throughput scales linearly with operator hours.

Anything we build is judged on one metric: **does it reduce operator hours per project?** Everything else is supporting infrastructure.

### Decisions made

| Decision | Choice | Rationale |
|---|---|---|
| **Operational surface** | Web app only (extend AlignAI) | No BricsCAD plugin. Plugin's only real win is reducing operator context-switching, which is solvable cheaply later via a 50-line LISP helper if data shows demand |
| **Distribution shape** | SaaS-ready from day one, single-tenant in practice until customer #2 | Stevenson is customer #1 / design partner; architecture should allow customer #2 without a rewrite |
| **CAD vendor coupling** | None. Output is clean DXF that any CAD app can open | Keeps the product CAD-agnostic and resellable |
| **First detection approach** | Classical CV baseline before any ML investment | De-risks Phase 1, generates a usable tool while ML is being trained, gives us a measurable accuracy floor |
| **Training-data approach** | Stevenson's historical archive is the moat; CubiCasa5K for pre-training | Roadmap was right about this — capture it from day one |
| **Mode of delivery** | Phased; every phase ships a thing an operator can use end-to-end | Avoids long invisible workstreams |

### Things to handle outside the codebase

These are not engineering tasks but they block the longer-term plan:

1. **IP / contract paragraph with Stevenson.** Before money or substantial deliverables change hands, get a one-paragraph email confirmation that the contractor retains rights to the core engine and Stevenson receives a license. Required to keep the SaaS path open.
2. **Pricing model exploration.** Per-scan? Per-month? Per-seat? Talk to 3–5 plausible second customers (architecture firms, scan-to-BIM service shops, surveyors) before Phase 5 (SaaS hardening) so the metering design isn't guessed.
3. **Accuracy threshold definition.** Roadmap flags this as open. Pick a number ("we ship when line-segment IoU at 30 mm tolerance ≥ 70 % on held-out Stevenson projects") before Phase 2 (eval harness) — otherwise the harness has no target.

---

## 1. Current State vs. Target State

### Tech stack — what exists, what we're adding

| Layer | Currently in repo | Target | Action |
|---|---|---|---|
| Frontend | React 18 + Vite + Three.js + Tailwind + Zustand (`frontend/`) | Same — add a "Vectorize" mode | Extend, don't replace |
| Backend API | FastAPI + Uvicorn + SSE (`backend/app/`) | Same — add `/api/vectorize/*` routes | Extend, don't replace |
| Background jobs | In-process `BackgroundTasks` + `JOB_SEMAPHORE` (`backend/app/limits.py`) | Same now; Redis/RQ when customer count > 1 | Extend now, swap later |
| Storage | SQLite + local filesystem | SQLite now; Postgres + S3-compatible storage at Phase 5 | Abstract behind a storage interface now so the swap is cheap |
| Auth | None | None now; Supabase Auth (or Clerk) at Phase 5 | Add a stub `current_user` dependency now that always returns `org_id="stevenson"` so multi-tenancy isn't a later refactor |
| Point-cloud ingestion | Open3D + laspy + pye57 (`backend/app/pipeline/ingest.py`) | Same — reuse | Reuse |
| Raster slicer | Inline in `scanplan.py`, not persisted | Standalone module, persisted PNG + affine | Build (Phase 0) |
| Line detection | `cv2.Canny` + `cv2.HoughLinesP` only, on synthetic occupancy | LSD + ED-Lines + Hough side-by-side, on real rasters | Build (Phase 0) |
| Vectorization / regularization | PCA wall fit only (`scanplan.py:67-83`) | Snap/ortho/merge/close-loops | Build (Phase 1) |
| DXF writer | Annotates input DXF only (`export.py:20-50`) | Writes detected geometry on configurable layers | Build (Phase 0/1) |
| ML stack | None | PyTorch + segmentation-models-pytorch + ONNX inference | Build (Phase 4) |
| Data preparation | None | Crawler, raster-vector aligner, classification tagger, splitter | Build (Phase 2) |
| Evaluation harness | None | Line IoU, precision/recall, FP/FN, regression gate | Build (Phase 2) |
| Operator review tooling | Read-only viewer + per-room/fixture panels | Interactive edit: accept/reject/drag/snap/batch/undo | Build (Phase 3) |
| Telemetry | None | Sentry + PostHog, hours-saved tracking | Build (Phase 5) |
| BricsCAD plugin | None | None — defer indefinitely | Skip unless data demands it (see Phase 6.5 optional) |

### Reusable building blocks already in the repo

These shorten every phase below:

| Block | Location | Reuse |
|---|---|---|
| FastAPI app + lifespan + CORS | `backend/app/main.py` | Add new routes alongside existing ones |
| SSE progress bus | `backend/app/sse.py` | Same `publish()` + `progress_generator()` for vectorize jobs |
| SQLite job model + filesystem layout | `backend/app/storage.py` | Reuse for vectorize jobs; new table next to `jobs` |
| Concurrency cap (`JOB_SEMAPHORE`) | `backend/app/limits.py` | Apply to vectorize workers too |
| Periodic cleanup loop | `backend/app/main.py:_cleanup_loop` | Already handles per-job artifact cleanup |
| Cloud point ingest (LAS/LAZ/PLY/E57) | `backend/app/pipeline/ingest.py` | Phase 0 spike feeds this into the slicer |
| OpenCV occupancy + morphology + Hough | `backend/app/pipeline/scanplan.py:194-224` | Extract into standalone slicer + classical detector |
| `ezdxf` + DXF parsing helpers | `pyproject.toml` + `rooms.py` | DXF writer reuses |
| React + Vite + Tailwind + Zustand + Three.js | `frontend/` | Add a Vectorize tab alongside the existing alignment view |
| Pytest infrastructure | `backend/tests/` | Add `tests/vectorize/`, `tests/eval/` siblings |

---

## 2. Phased Implementation Plan

Each phase delivers a usable deliverable, explicitly states whether it advances the **root problem** (reducing Stevenson operator hours), and lists what's intentionally *not* in scope.

### Phase 0 — Feasibility spike (3–5 days)

**Goal:** Prove that classical CV gets us anywhere on Stevenson's actual scan data before any further investment.

**Deliverable:** A CLI that takes one of the `source-files/*.laz` files and produces:
- `slice.png` — the rasterized 2D plan slice
- `overlay_hough.png` — slice with Hough-detected lines in red
- `overlay_lsd.png` — slice with LSD-detected lines in red
- `vectorized.dxf` — a clean DXF on a `WALLS` layer, openable in BricsCAD

**Files to create:**
```
backend/app/vectorize/__init__.py
backend/app/vectorize/slicer.py          # .laz → raster PNG + affine sidecar
backend/app/vectorize/classical.py       # raster → segments (Hough, LSD)
backend/app/vectorize/dxf_writer.py      # segments → clean DXF with $INSUNITS
backend/app/vectorize/spike.py           # CLI entry: python -m backend.app.vectorize.spike
backend/tests/vectorize/test_slicer.py
backend/tests/vectorize/test_dxf_writer.py
```

**Files to modify:**
```
backend/pyproject.toml                    # swap opencv-python-headless → opencv-contrib-python-headless
                                          # (LSD lives in main cv2; ED-Lines lives in ximgproc contrib)
```

**Success criteria:**
- [ ] One `.laz` from `source-files/` runs to completion in under 60 seconds on a developer laptop
- [ ] Generated DXF opens cleanly in BricsCAD with units (`$INSUNITS`) preserved
- [ ] At least 50 % of obvious walls in the slice are represented by a line segment in the DXF (eyeballed, not measured — measurement comes in Phase 2)
- [ ] You can hand the DXF to a Stevenson operator and they can recognize what building it is

**Solves root problem?** Not yet. This phase tells us how much further the project has to go. If classical alone gets us 80 %, ML is polish. If it gets us 30 %, ML is the headline.

**Not in scope:**
- Web UI changes
- ML / training
- Door / window / column / MEP detection (walls only)
- Multi-floor / multi-scan handling
- Regularization beyond raw detector output
- OCR

**Effort:** 1 engineer × 3–5 focused days.

---

### Phase 1 — Web FE Vectorize mode (2–3 weeks)

**Goal:** A Stevenson operator can open AlignAI, drop a `.laz` file, pick an elevation, and download a usable DXF. This is the first shippable version that actually saves hours.

**Deliverable:** A "Vectorize" tab in the existing AlignAI UI:
1. Upload `.laz` (reuse `UploadZone`)
2. Pick elevation (slider, default = floor + 1.4 m)
3. Click "Vectorize" → job runs, SSE progress streams to UI
4. Result page shows: raster preview, detected lines overlay, downloadable DXF
5. "Re-run with different parameters" button (elevation, slab thickness, detector choice)

**Files to create:**
```
backend/app/routes/vectorize.py           # POST /api/vectorize, GET /api/vectorize/{id}, SSE
backend/app/vectorize/pipeline.py         # the orchestrator (ingest → slice → detect → regularize → write)
backend/app/vectorize/regularize.py       # merge collinear, snap to Manhattan, close short gaps
backend/app/vectorize/preprocess.py       # adaptive threshold, morphology on rasters
backend/app/models/vectorize_job.py       # Pydantic schemas
backend/tests/vectorize/test_pipeline.py
backend/tests/vectorize/test_regularize.py

frontend/src/components/vectorize/
  VectorizeTab.tsx                        # top-level tab component
  VectorizeUploadZone.tsx                 # specialized for single-file workflow
  ElevationPicker.tsx                     # slider + numeric input
  VectorizeResult.tsx                     # raster + overlay + DXF download
  VectorizeParams.tsx                     # re-run controls
frontend/src/api/vectorize.ts             # typed client for /api/vectorize/*
```

**Files to modify:**
```
backend/app/main.py                       # include the new router
backend/app/storage.py                    # add vectorize_jobs table
frontend/src/App.tsx                      # add Vectorize tab to the layout
frontend/src/state/jobStore.ts            # add vectorize mode to UI state
```

**Success criteria:**
- [ ] Operator can complete the full upload → download loop without engineering intervention
- [ ] End-to-end run on a typical `.laz` (~700 MB) finishes in under 90 seconds on a developer laptop
- [ ] Output DXF imports into BricsCAD with at least 50 % of walls present and recognizable
- [ ] Operator running a real project reports time saved vs. fully-manual tracing (target: at least 30 % time reduction even at classical accuracy)

**Solves root problem?** **Yes — first time.** Even at modest detection accuracy (60–70 %), if the operator can correct rather than trace from scratch, this is the first phase where Stevenson actually saves hours.

**Not in scope:**
- ML
- Interactive in-browser correction (operator corrects in BricsCAD)
- Multi-tenant auth
- Door/window/column detection
- Evaluation harness (Phase 2)

**Effort:** 1 engineer × 2–3 weeks. Most of the time is the UI; the pipeline reuses Phase 0 work.

---

### Phase 2 — Data preparation + evaluation harness (2–3 weeks)

**Goal:** Stop guessing. Catalog Stevenson's archive into a training/eval corpus, and instrument every change with hard accuracy numbers.

**Deliverable:** Two tools.

**Tool A — Dataset manifest:**
- Crawl Stevenson's project archive (path provided by Stevenson once available)
- For each (raster slice, traced DWG) pair, write a row to `data/datasets/v1/manifest.parquet`:
  - `project_id`, `slice_path`, `traced_dwg_path`, `sha256` of each, `elevation_m`, `scanner_model` (if known), `quality_grade` (manual A/B/C tag), `train_val_test_split`
- Persist the parsed traced DXF as a normalized "ground truth" JSON for fast eval re-runs

**Tool B — Evaluation harness:**
- `python -m backend.eval.runner --pipeline-version v0.1 --dataset v1 --split test`
- For each test-set pair: run the pipeline, score the output against ground truth
- Metrics: line-segment IoU at multiple tolerances (15 mm / 30 mm / 50 mm), precision, recall, F1, false-positive line count, false-negative line count
- Write results to `data/eval/runs/{run_id}.json` and append a row to `data/eval/index.parquet`
- Generate `eval_report.html` showing trend over time (run-over-run comparison)

**Files to create:**
```
backend/data_prep/__init__.py
backend/data_prep/crawler.py              # walk a project root, emit candidate (raster, dwg) pairs
backend/data_prep/aligner.py              # homography between raster and traced DWG when origins differ
backend/data_prep/tagger.py               # classification tags from DWG layer names
backend/data_prep/grader.py               # quality grading (CLI or heuristics)
backend/data_prep/splitter.py             # train/val/test split (project-level to avoid leakage)
backend/data_prep/build_dataset.py        # CLI

backend/eval/__init__.py
backend/eval/metrics.py                   # line IoU, precision/recall, FP/FN
backend/eval/runner.py                    # CLI runner
backend/eval/report.py                    # HTML report generator
backend/eval/regression_gate.py           # CI gate: fail PR if regressed
backend/tests/eval/test_metrics.py
backend/tests/eval/test_runner.py
```

**Files to modify:**
```
backend/pyproject.toml                    # add pandas, pyarrow, jinja2 (for the HTML report)
.github/workflows/                        # (if/when CI is set up) add eval regression check
```

**Success criteria:**
- [ ] Manifest exists for ≥ 20 historical Stevenson projects
- [ ] Eval harness produces a single accuracy number for any pipeline version
- [ ] Re-running yesterday's pipeline yields the same number (deterministic)
- [ ] The Phase 1 classical pipeline is the v0.1 baseline; you have a concrete number on the test set
- [ ] Quality threshold for "production-ready" is decided and recorded in the harness

**Solves root problem?** Indirectly — it doesn't reduce operator hours by itself, but every subsequent phase becomes measurable. Without this, Phase 3 and Phase 4 are flying blind.

**Not in scope:**
- ML training (depends on this; comes in Phase 4)
- Reviewing accuracy in the UI (Phase 3)
- Aggregated multi-customer dataset (Phase 5+)

**Effort:** 1 engineer × 2–3 weeks. Data quality dominates the schedule; if Stevenson's archive is messy this could double.

**Dependency:** Stevenson must give us read access to (or copies of) 20+ paired (raster, traced DWG) historical projects. This is the longest-pole-in-the-tent for the entire roadmap.

---

### Phase 3 — Operator review UX in the browser (2–3 weeks)

**Goal:** Even when detection accuracy is medium, operators can clean up the result inside the web app fast enough that they don't need to re-do work in BricsCAD.

**Deliverable:** An interactive editor in the Vectorize tab:
- Render raster underlay + detected lines on top
- Click line → select; Shift+click → multi-select
- Delete key → reject; Enter → accept
- Drag line endpoints; OSNAP-style snap to other endpoints / grid / right angles
- "Batch reject lines shorter than X" / "Snap all selected to nearest Manhattan axis"
- Undo / redo
- "Export DXF" button only enables once everything is accepted or rejected
- Stores the operator's edit history as part of the job artifact (becomes future training data — feeds Phase 4)

**Files to create:**
```
frontend/src/components/vectorize/editor/
  EditorCanvas.tsx                        # Three.js or Canvas 2D editor surface
  LineSegment.tsx                         # selectable, draggable line primitive
  Toolbar.tsx                             # tools: select, snap, ortho-snap, delete, accept-all
  HistoryPanel.tsx                        # undo/redo stack visualization
  BatchOpsPanel.tsx                       # batch reject/accept by length, orientation, etc.
frontend/src/state/editorStore.ts         # Zustand store for editor state (selection, history)

backend/app/routes/vectorize.py           # add PUT /api/vectorize/{id}/edits (persist edit history)
backend/app/vectorize/edits.py            # apply edit log → final DXF
```

**Success criteria:**
- [ ] Operator can take a job with 70 % detection accuracy and reach a clean DXF in under 5 minutes per slice
- [ ] All edits are persisted; reopening the job 1 hour later restores the in-progress state
- [ ] Edit history is queryable (becomes the corrections dataset feeding Phase 4)

**Solves root problem?** **Big jump.** Combined with Phase 1, this is the version of the product that wins or loses Stevenson's adoption. A "perfect" detection model is worthless if correction is painful; a "70 % detection + 5-min editor" beats a "95 % detection + 30-min correction in BricsCAD."

**Not in scope:**
- Door / window / column editing (single feature class — walls only)
- ML
- Multi-user editing

**Effort:** 1 engineer × 2–3 weeks. The editor surface is the substantial work; the persistence layer is small.

---

### Phase 4 — ML segmentation (6–10 weeks)

**Goal:** Push detection accuracy from "decent classical" to "near operator quality" by training a model on Stevenson's archive.

**Deliverable:**
- A semantic segmentation model that, per pixel, predicts class ∈ {background, wall, opening, column}
- Hybrid pipeline: classical detection finds candidate lines, ML labels each pixel; lines without ML wall-pixel support are dropped; ML wall pixels without a classical line spawn new lines
- All quality changes measured against the Phase 2 eval harness

**Files to create:**
```
backend/ml/__init__.py
backend/ml/datasets/cubicasa5k.py         # download / parse for pre-training
backend/ml/datasets/stevenson_v1.py       # consumes data/datasets/v1/
backend/ml/models/unet.py                 # baseline U-Net
backend/ml/models/factory.py              # model registry
backend/ml/train/train.py                 # PyTorch Lightning trainer
backend/ml/train/config/walls_v1.yaml     # experiment config
backend/ml/inference/segment.py           # ONNX inference
backend/ml/checkpoints/                   # gitignored, tracked via S3/DVC
backend/ml/scripts/export_onnx.py         # checkpoint → onnx

backend/app/vectorize/hybrid.py           # fuse classical + ML
backend/app/vectorize/pipeline.py         # add --mode classical|ml|hybrid
```

**Files to modify:**
```
backend/pyproject.toml                    # add [project.optional-dependencies] ml = [torch, lightning, onnxruntime, segmentation-models-pytorch]
                                          # keep core install slim — ml stays optional
backend/eval/runner.py                    # accept --mode argument
```

**Success criteria:**
- [ ] U-Net trained on Stevenson v1 dataset reaches ≥ 80 % wall IoU on held-out test set at 30 mm tolerance
- [ ] Hybrid pipeline beats classical-only on every metric in the eval harness
- [ ] CPU inference under 15 s per slice; GPU under 3 s
- [ ] Model can be exported to ONNX and loaded by the inference module without PyTorch installed (keeps production deployment small)

**Solves root problem?** **Biggest single quality jump.** Likely doubles the operator-hour savings achieved in Phase 3 alone.

**Not in scope:**
- Multi-class beyond walls (deferred to Phase 6)
- Training on cross-customer data (deferred to Phase 5+ once tenants exist)
- OCR

**Effort:** 1 engineer × 6–10 weeks. Schedule depends entirely on data quality and how much the model architecture has to iterate. GPU access required.

**Dependency:** GPU workstation (RTX 4090 / A6000) or cloud GPU rental.

---

### Phase 5 — SaaS hardening (3–4 weeks)

**Goal:** Open the door to a second customer. The product works for Stevenson by end of Phase 3 / 4; this phase makes it possible to onboard a second customer without code changes.

**Deliverable:**
- Real auth (Supabase Auth or Clerk); every API call scoped to `org_id`
- Postgres replaces SQLite; per-table Row Level Security
- S3 / Supabase Storage replaces local disk for uploads / artifacts / results
- Real job queue (Redis + RQ) replaces in-process background tasks
- Stripe integration: subscription + usage-metered billing
- Self-serve sign-up + magic-link login
- Basic org admin: user invites, usage dashboard, billing portal
- Cloud deployment (Fly.io or Render for API + worker, Vercel for FE, Modal or Banana for GPU inference)
- Sentry + PostHog wired up

**Files to create:**
```
backend/app/auth/__init__.py
backend/app/auth/dependencies.py          # current_user, current_org dependencies
backend/app/auth/middleware.py
backend/app/storage/__init__.py           # abstract storage interface
backend/app/storage/local.py              # current behaviour
backend/app/storage/s3.py                 # new
backend/app/billing/__init__.py
backend/app/billing/stripe_client.py
backend/app/billing/metering.py
backend/app/queue/__init__.py
backend/app/queue/redis_rq.py             # replaces in-process BackgroundTasks
infra/
  fly.toml or render.yaml
  docker/api.Dockerfile
  docker/worker.Dockerfile
  docker/inference.Dockerfile
.github/workflows/deploy.yml

frontend/src/auth/                        # sign-in, sign-up, magic-link flows
frontend/src/billing/                     # plan picker, billing portal redirect
frontend/src/admin/                       # org users, usage, settings
```

**Files to modify:**
```
backend/app/main.py                       # add auth middleware, swap storage backend by config
backend/app/routes/vectorize.py           # add org_id scoping to every query
backend/app/storage.py                    # migrate SQLite → Postgres via Alembic
all routes                                # add Depends(current_user)
all tests                                 # add fixtures for auth + org
```

**Success criteria:**
- [ ] Second test org can sign up, upload, vectorize, download — never sees Stevenson's data
- [ ] Stripe test-mode subscription works end-to-end (sign up, pick plan, hit usage limit, upgrade)
- [ ] Deployed to cloud with a public URL
- [ ] No engineering touch required to onboard a new customer

**Solves root problem?** Not directly for Stevenson (they're already saving hours by end of Phase 3 / 4). This phase unlocks the business model — the ability to sell to customers #2, #3, #4.

**Not in scope:**
- Marketing site (separate repo / Webflow / etc.)
- Customer support tooling
- Multi-region data residency
- SOC 2 / HIPAA

**Effort:** 1 engineer × 3–4 weeks.

**Prerequisite:** the contract / IP paragraph with Stevenson is in place (see Section 0).

---

### Phase 6 — Continuous improvement (ongoing, starts after Phase 5)

**Goal:** Make the product self-improving and broaden detection coverage.

**Deliverables (each is an independent sub-project):**

- **6.1 Correction feedback loop:** Operator edits captured in Phase 3 are appended to `data/datasets/corrections/{date}/`. Weekly retraining cron picks them up. Each customer's corrections improve only their model (no cross-contamination) until customers opt into a shared "global" model.
- **6.2 Additional feature classes:** Train detectors for openings (doors/windows), columns, MEP runs. Each is a new output channel on the segmentation model + a new layer in the DXF output.
- **6.3 OCR (Tesseract → PaddleOCR if needed):** Detect text in slices — room labels, dimensions, callouts. Write to a `TEXT` layer.
- **6.4 Multi-floor / batch processing:** Process all floors of a building in one upload; merge outputs.
- **6.5 Telemetry-driven product decisions:** Time-saved-per-job metric, detector quality by project type, operator correction patterns → backlog for next quarter.

### Phase 6.5 — *Optional* CAD-side helper (only if data demands it)

**Trigger:** Operator survey at end of Phase 3 shows consistent complaints about "switching to the browser to grab the DXF, then importing into BricsCAD, then fixing the layers."

**Deliverable:** A 50-line LISP file (`bricscad-helpers.lsp`) that customers paste into their BricsCAD startup script. Adds one command:
```
SC2C_FETCH    ; pulls the latest completed job for the logged-in user
              ; via the API, INSERTs the DXF using a saved layer map
```

LISP is portable to AutoCAD with minimal changes; ship one file per CAD vendor.

**Effort:** 2–3 days.

**Not a plugin.** No installer, no .NET, no SDK skill required.

---

## 3. Phase Map at a Glance

```
Phase 0  ─┬─ 3-5 days ──── classical CV spike (CLI only)
          │
Phase 1  ─┼─ 2-3 weeks ─── Vectorize tab in web UI ★ first hours saved
          │
Phase 2  ─┼─ 2-3 weeks ─── data prep + eval harness (no UX delta)
          │                     ↓ needs Stevenson archive access
Phase 3  ─┼─ 2-3 weeks ─── interactive editor in browser ★ adoption-defining
          │
Phase 4  ─┼─ 6-10 weeks ── ML segmentation ★ accuracy jump
          │                     ↓ needs GPU
Phase 5  ─┼─ 3-4 weeks ─── SaaS hardening (auth/billing/cloud) ★ second customer possible
          │                     ↓ needs IP paragraph with Stevenson
Phase 6  ─┴─ ongoing ────── corrections feedback / more classes / OCR / telemetry

(Phase 6.5: optional LISP helper, only if operators ask for it)
```

Cumulative engineering time end of Phase 5: ~18–25 weeks (4–6 months) of focused single-engineer work.

End of Phase 1 (~3 weeks): Stevenson saves measurable hours per project.
End of Phase 3 (~7–9 weeks): Stevenson's adoption decision is made.
End of Phase 4 (~13–19 weeks): accuracy is at "near-operator-quality."
End of Phase 5 (~16–23 weeks): can sign up customer #2.

---

## 4. Architecture (target end-of-Phase-5)

```
┌───────────────────────────────────────────────────────────────────┐
│ Browser (any customer, any OS)                                    │
│   React 18 + Vite + Tailwind + Zustand + Three.js                 │
│   ├─ AlignTab        (legacy AlignAI — keep working)              │
│   ├─ VectorizeTab    (Phase 1+)                                   │
│   │  ├─ UploadZone                                                │
│   │  ├─ ParametersPanel                                           │
│   │  ├─ ProgressViewer                                            │
│   │  └─ EditorCanvas  (Phase 3+ interactive review)               │
│   ├─ Auth            (Phase 5: sign-in, sign-up, magic link)      │
│   └─ Admin           (Phase 5: org users, usage, billing portal)  │
└───────────────┬───────────────────────────────────────────────────┘
                │  HTTPS  |  SSE progress  |  signed-URL uploads
                ▼
┌───────────────────────────────────────────────────────────────────┐
│ FastAPI app (API)                                                 │
│   /api/align/*       legacy alignment endpoints                   │
│   /api/vectorize/*   new (Phase 1+)                               │
│   /api/auth/*        (Phase 5)                                    │
│   /api/billing/*     (Phase 5)                                    │
│   /api/admin/*       (Phase 5)                                    │
│   Middleware: auth (Phase 5), org_id scoping, rate limit, CORS    │
└───────────────┬───────────────────────────────────────────────────┘
                │  enqueue
                ▼
┌───────────────────────────────────────────────────────────────────┐
│ Redis + RQ worker pool (Phase 5 — in-process before then)         │
│   Runs the vectorize pipeline:                                    │
│   ingest → slice → preprocess → classical OR hybrid → regularize  │
│   → write DXF → persist artifacts → emit completion event         │
└──────┬───────────────────────────────────┬────────────────────────┘
       │                                   │
       ▼                                   ▼
┌──────────────────────────┐  ┌────────────────────────────────────┐
│ ML inference service     │  │ Postgres + S3-compatible storage   │
│ (Phase 4+)               │  │   jobs, edits, datasets, billing   │
│   ONNX runtime on GPU    │  │   org_id RLS on every table        │
│   (Modal / Banana / etc) │  │   (SQLite + local fs before P5)    │
└──────────────────────────┘  └────────────────────────────────────┘

(Plus Sentry, PostHog, Stripe webhooks from Phase 5)
```

---

## 5. Issues to Address Along the Way

These are bugs / friction points already present that will hurt the new path if left alone. Each is tied to the phase that should fix it.

| # | Issue | Location | Fix in phase |
|---|---|---|---|
| 1 | `write_aligned_dxf` doesn't write detected geometry — only annotates input DXF | `backend/app/pipeline/export.py:20-50` | Phase 0 — replaced by new `dxf_writer.py`; legacy fn stays for AlignAI but is renamed |
| 2 | `scanplan.py` clips occupancy raster to 4096 px — silently truncates large buildings | `backend/app/pipeline/scanplan.py` (np.clip 0,4095) | Phase 0 — new slicer has no fixed cap, sizes to fit |
| 3 | `opencv-python-headless` lacks `ximgproc` (ED-Lines) | `backend/pyproject.toml:22` | Phase 0 — swap to `opencv-contrib-python-headless` |
| 4 | In-process `BackgroundTasks` won't survive single-machine cloud deploy | `backend/app/routes/jobs.py:113` | Phase 5 — Redis + RQ |
| 5 | Local-only storage paths everywhere (`data/uploads/`, `data/results/`) | `backend/app/storage.py` | Phase 5 — abstract behind storage interface; S3 backend |
| 6 | No `org_id` anywhere — every query implicitly trusts the request | `backend/app/storage.py`, all routes | Phase 5 — add `current_user` / `current_org` dependencies, RLS at DB layer |
| 7 | No magic-byte validation for image uploads (only LAS/PLY) | `backend/app/routes/jobs.py:19-25` | Phase 1 — add PNG/TIFF magic-byte check for raster uploads |
| 8 | DXF unit preservation is incidental — no explicit handling | new code (Phase 0+) | Phase 0 — `dxf_writer` sets `$INSUNITS` explicitly |
| 9 | Single monolithic pipeline orchestrator (`runner.py`) | `backend/app/pipeline/runner.py` | Phase 1 — new vectorize pipeline doesn't reuse this; alignment pipeline keeps it |
| 10 | No correlation between operator edits and detector output (no learning loop) | n/a | Phase 3 captures edits; Phase 6.1 wires them into retraining |

---

## 6. Testing Strategy

**Phase 0:**
- Unit: slicer round-trip (raster pixel → world coord); DXF writer schema (open with `ezdxf` and assert entity counts); classical detectors return ≥3 segments on a 3-line synthetic image.
- Manual: run the spike on 3 different `.laz` files; eyeball outputs.

**Phase 1:**
- Unit: pipeline stage isolation; API contract (FastAPI test client); React component snapshot tests.
- Integration: end-to-end via test client (upload → poll → download); SSE event stream completes.
- Manual: Stevenson operator runs the tool on one real project, times themselves.

**Phase 2:**
- Unit: metric determinism (same inputs → same numbers); edge cases (empty prediction, empty ground truth, exact match).
- Integration: harness runs end-to-end against the v1 dataset; report HTML renders.
- Regression gate: CI fails if any released pipeline version drops accuracy on the v1 test set.

**Phase 3:**
- Unit: edit log application order-independence where applicable; undo/redo invariants.
- Integration: full edit session round-trip (open → edit → save → reopen → same state).
- Manual: timed operator session — start with a v0 detection, end with a clean DXF, record minutes.

**Phase 4:**
- Unit: model loading; ONNX vs. PyTorch parity (within numerical tolerance); dataset class balance.
- Integration: train smoke (1 epoch on tiny subset → loss decreases); inference smoke (random image → expected output shape).
- Regression: hybrid pipeline beats classical on the v1 test set on every metric. If not, no merge.

**Phase 5:**
- Unit: auth dependency; org scoping in every query (write a test fixture with 2 orgs → assert one can't read the other's).
- Integration: full sign-up → vectorize → download in a hermetic test env.
- Manual: penetration check — try to access another org's data via direct API calls.

---

## 7. Open Questions to Resolve Before / During the Plan

Before Phase 1:
- Where do we keep Stevenson's `source-files/` and any future archive data — local only, or do we already need S3 to avoid disk pressure on the dev laptop?

Before Phase 2:
- What "production-ready" accuracy threshold? (Answer drives the regression gate.)
- Can Stevenson grant us read access to 20+ paired (raster, traced DWG) historical projects, including any client-confidentiality constraints?

Before Phase 4:
- Buy a GPU workstation (RTX 4090 ~$2k, A6000 ~$5k) or rent cloud GPU (~$1/hr on-demand)? Buy is cheaper past ~2000 training hours.
- Do we pre-train on CubiCasa5K, or skip straight to Stevenson data? (Recommended: pre-train; data is free and we already plan to.)

Before Phase 5:
- IP paragraph with Stevenson (Section 0).
- Pricing model (talk to 3-5 plausible second customers).
- Hosting target (Fly.io / Render / Vercel + Modal — pick once based on cost projections).
- Auth provider (Supabase vs. Clerk vs. Auth.js).
- DB host (Supabase Postgres vs. Neon vs. RDS).

Throughout:
- Track time-saved-per-project for every Stevenson job from Phase 1 onward. This is the root-problem metric; everything else is supporting.

---

## 8. Appendix — Reusable code inventory

Already in the repo, immediately useful for the new pipeline:

| Block | File | Reused in phase |
|---|---|---|
| Occupancy-grid logic | `backend/app/pipeline/scanplan.py:194-205` | Phase 0 slicer |
| Canny + HoughLinesP wiring | `backend/app/pipeline/scanplan.py:211-224` | Phase 0 classical detector |
| Floor detection (auto-elevation) | `backend/app/pipeline/slicing.py` | Phase 1 elevation default |
| LAS/LAZ/PLY/E57 loader | `backend/app/pipeline/ingest.py` | Phase 0 input |
| Decimated-PLY writer (binary little-endian float32) | `backend/app/pipeline/ingest.py:52-113` | Pattern for compact artifact serialization |
| FastAPI app factory + lifespan + CORS | `backend/app/main.py` | Phase 1 routes plug in here |
| SSE progress bus | `backend/app/sse.py` | Phase 1 progress streaming |
| SQLite job model | `backend/app/storage.py` | Phase 1 vectorize_jobs table |
| Concurrency cap | `backend/app/limits.py` | Phase 1 worker cap |
| Pytest infrastructure | `backend/tests/` | Phases 0+ — add `tests/vectorize/` sibling |
| React+Vite+Tailwind+Zustand+Three.js | `frontend/` | Phases 1+ — new tab alongside existing |
| Upload zone + processing panel + job history | `frontend/src/components/` | Phase 1 reuse for the new tab |

---

*This is a living plan. Check off success criteria as each phase completes; revisit Section 0 decisions at every phase boundary.*
