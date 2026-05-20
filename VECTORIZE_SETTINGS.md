# Vectorize — Settings & Viewer Reference

A practical guide to every knob on the Vectorize panel and every view mode in the result viewer. Read top-to-bottom the first time; after that use it as a lookup.

> **What "Vectorize" does in one sentence**
> Take a 3D scan (`.las` / `.laz` / `.ply` / `.e57`), slice a thin horizontal slab out of it at roughly chest height, flatten that slab into a 2D image, find straight wall segments in the image, clean them up, and write the result as a layered DXF.

---

## Table of contents

1. [Pipeline at a glance](#pipeline-at-a-glance)
2. [Input — the scan upload](#input--the-scan-upload)
3. [Parameters panel](#parameters-panel)
    - [Elevation (m)](#elevation-m)
    - [Line detector](#line-detector)
    - [Min wall length (m)](#min-wall-length-m)
    - [Snap to building axes (Manhattan)](#snap-to-building-axes-manhattan)
    - [Merge collinear segments](#merge-collinear-segments)
    - [Advanced — Slab thickness (m)](#advanced--slab-thickness-m)
    - [Advanced — Resolution (m/px)](#advanced--resolution-mpx)
4. [Viewer modes — Clean / Raw / Raster](#viewer-modes--clean--raw--raster)
5. [Which knob affects which stage](#which-knob-affects-which-stage)
6. [Tuning workflow](#tuning-workflow)
7. [Defaults & valid ranges](#defaults--valid-ranges)
8. [Where these live in the code](#where-these-live-in-the-code)

---

## Pipeline at a glance

```
scan file
   │
   ▼
1. ingest        load point cloud (Open3D)
2. floor detect  pick auto elevation (if not supplied)
3. slice         project a horizontal slab → 2D raster      ← Slab thickness, Resolution, Elevation
4. preprocess    morphology + smoothing on the raster
5. detect        find line segments on the cleaned raster   ← Line detector, Min wall length
6. regularize    cull short, snap to axes, merge collinear  ← Min wall length, Manhattan, Merge
7. overlay       render review PNGs (Clean & Raw)
8. dxf           write the layered DXF deliverable
9. persist       write result.json + segments.json
```

The viewer's three buttons show artifacts written at stages 4, 5, and 7.

---

## Input — the scan upload

A single point-cloud file in one of: `.las`, `.laz`, `.ply`, `.e57`.

Drag onto the drop zone or click to browse. Only one file per job. Everything below describes how that file gets turned into lines.

---

## Parameters panel

### Elevation (m)

The absolute height of the horizontal slice plane through the scan, in metres in the scan's local coordinate frame. The pipeline grabs a thin slab of points centred on this height and projects it down into the 2D image that line detection runs on.

| Value | What happens |
|---|---|
| `auto (floor + 1.4)` *(default)* | The pipeline runs floor detection, then slices at `floor_z + 1.4 m`. That's roughly chest height — intentionally above furniture and baseboards, below most ceiling fixtures, so the cross-section is mostly clean wall. |
| Explicit number | Skips floor detection and slices at exactly that elevation. Use this when auto picks a bad plane (multi-storey scans, low-poly ceilings being mistaken for the floor, or you want a slice through a specific floor of the building). |

**When to override auto:**
- Multi-storey scans where you want a specific floor.
- Scans where furniture or a mezzanine is being detected as the floor.
- You've found a "sweet spot" elevation visually and want to lock it in for re-runs.

---

### Line detector

Which classical line-detection algorithm runs on the cleaned raster.

| Option | Algorithm | Character | When to use |
|---|---|---|---|
| **FLD** *(default, recommended)* | `cv2.ximgproc.createFastLineDetector` — modern LSD replacement | Fewer, longer, cleaner segments on architectural edges | Almost always — best wall fidelity for interior scans |
| **Hough** | `cv2.HoughLinesP` — probabilistic Hough transform | Very robust, very fast, fragments long walls into many short pieces | When FLD misses obvious walls, or as a sanity check |
| **Both** | Run FLD + Hough, merge the outputs | Slower; more fragments going into regularization | Diagnostic only — useful for comparing detector quality side-by-side |

> **Fallback**: if FLD returns zero segments (e.g. `opencv-contrib` isn't installed), the pipeline automatically falls back to Hough for that run and logs it.

---

### Min wall length (m)

Two-job slider:

1. **At detection time** — converted to pixels (`min_wall_length_m / resolution_m_per_px`, floored at 3 px) and passed to the detector as the shortest segment it's allowed to return.
2. **At regularization time** — used again as a hard cull in world units. Anything shorter than this after merging is dropped.

| Setting | Result |
|---|---|
| Low (0.10–0.50 m) | Keeps door jambs, columns, half-walls, partitions, closet returns. More clutter. |
| Default (~1.40 m) | Balanced for residential interiors — keeps real walls, drops most furniture-edge noise. |
| High (2.00–5.00 m) | Structural walls only. Very clean output, but you'll lose short real walls (closets, niches). |

**Rule of thumb**: start at default. Raise if the Clean overlay looks noisy. Lower if you're missing legitimate short walls.

---

### Snap to building axes (Manhattan)

After detection, finds the two dominant orientations across all detected segments, then **drops anything that isn't aligned with one of them** (within a tolerance) and snaps the survivors to be perfectly axis-aligned.

| State | Behaviour | Best for |
|---|---|---|
| **On** *(default)* | Output looks "drafted" — only the two strong building axes survive. Kills diagonal noise from furniture, plants, scan artefacts, reflections. | Typical rectangular buildings |
| **Off** | Every angle the detector found is preserved | Rotated buildings, curved walls, organic shapes, sites with multiple non-orthogonal wings |

> Note: "Manhattan" still works for buildings *rotated as a whole* — the two dominant axes don't have to be world-axis-aligned, they just have to exist. Turn it off when the building genuinely has more than two wall orientations.

---

### Merge collinear segments

After Manhattan snapping, fuses segments that are nearly parallel, nearly collinear, and close end-to-end into one longer segment.

| State | Behaviour |
|---|---|
| **On** *(default)* | A long wall the detector broke into 5–10 pieces (because of a doorway, picture frame, or scan gap) becomes a single line. Typical 2–5× reduction in segment count. |
| **Off** | Every fragment preserved — useful only for QA when you want to see exactly what the detector returned. |

Leave on for normal use.

---

### Advanced — Slab thickness (m)

How tall (in metres) the horizontal slab of points around the elevation plane is, before being squashed flat into the 2D raster.

| Value | Effect |
|---|---|
| Thinner (≤ 0.10 m) | Crisp corners, clean separation between rooms. Sparse scans may have so few points per pixel that walls become dashed and the detector misses them. |
| **0.20 m** *(default)* | Sweet spot. Dense enough for reliable detection, thin enough for clean corners. |
| Thicker (≥ 0.30 m) | Dense, bright raster — detector sees walls reliably. But corners get rounded, and overhead beams / door tops / stairs can poke into the slab and show up as fake walls. |

---

### Advanced — Resolution (m/px)

World metres represented by one raster pixel.

| Value | Effect |
|---|---|
| 0.005 (5 mm/px) | High fidelity — preserves thin walls and tight features. Raster gets ~4× larger than 1 cm/px, detection slows, noisy scans look noisier. |
| **0.010 (1 cm/px)** *(default)* | Production default. Fast; captures everything thicker than ~2 cm. |
| 0.050 (5 cm/px) | Very coarse — only structural walls survive. Useful for previews of huge buildings. |

> **Coupling warning**: this also indirectly changes the *minimum-length floor* the detector is given, because `Min wall length` (metres) is converted to pixels using this resolution. If you lower resolution sharply, double-check that small walls aren't being silently dropped.

---

## Viewer modes — Clean / Raw / Raster

The three buttons in the top-right of the result canvas. They show different artifacts from the same run so you can compare what the detector saw vs. what made it into the DXF.

| Button | Image | Colour | What it shows |
|---|---|---|---|
| **Clean** | `overlay.png` | Red lines | Final, regularized segments drawn on the cleaned raster. **These are the exact lines that end up in the downloaded DXF.** |
| **Raw** | `overlay_raw.png` | Orange lines | The detector's raw output drawn on the cleaned raster — *before* regularization (no short-segment cull, no Manhattan snap, no merge). |
| **Raster** | `slice_cleaned.png` | — | The cleaned 2D slice itself, with no overlays. The literal image the detector ran on. |

### How to use them

**Clean — "is the result good?"**
Your primary view. If it looks right, hit Download DXF. If it looks wrong, switch to Raw to diagnose.

**Raw — "what did the detector find?"**
If Clean is missing walls or has too few segments:
- Correct walls in orange that Clean is missing → regularize is too aggressive (lower Min wall length, disable Manhattan, etc.).
- Walls in the raster that have *no* orange on them → the detector itself is failing (switch detector, change resolution, thicken slab).
- Lots of orange noise on furniture → expected; that's exactly what Manhattan + min-length cull.

**Raster — "is the input even usable?"**
If Raw is also missing walls:
- Walls should appear as connected, continuous bright pixel runs. If they're dotted, the scan is too sparse for the current resolution/slab settings.
- Furniture poking through? Lower the slab thickness or shift elevation.
- Walls invisible entirely? Raise slab thickness or pick a different elevation.

> Buttons grey out until their artifact exists on disk (`has_raster` / `has_overlay`), so during a running job they light up in order: **Raster** first, then **Raw + Clean** once the overlay stage finishes.

---

## Which knob affects which stage

| Knob | Stages it influences |
|---|---|
| Elevation | Floor detect → Slice |
| Slab thickness | Slice |
| Resolution | Slice (raster size) → Detect (via px conversion of min length) |
| Line detector | Detect |
| Min wall length | Detect (lower bound) → Regularize (cull) |
| Manhattan snap | Regularize |
| Merge collinear | Regularize |

---

## Tuning workflow

A reliable order for dialling in a new scan:

1. **Run defaults first.** FLD, ~1.4 m min length, Manhattan on, merge on, slab 0.20 m, resolution 0.01 m/px.
2. **Open the Clean view.** If it looks good → done, download DXF.
3. **Open the Raster view first** if Clean looks bad. Decide whether the *input image* is usable.
    - Walls dotted / missing → adjust slab thickness, elevation, or resolution. Don't touch detector settings yet.
4. **Open the Raw view** if the Raster looks fine but Clean is sparse.
    - Real walls in orange that didn't make it to Clean → loosen regularize: lower Min wall length, or disable Manhattan if the building is rotated/curved.
    - Walls in the raster with no orange at all → switch detector (FLD ↔ Hough), or raise resolution / slab.
5. **Use the Result panel numbers** as a feedback signal. The ratio `segments_after_regularize / segments_detected` tells you how aggressive your cleanup is:
    - Tiny ratio → cleanup is killing too much.
    - Both numbers huge → detector + raster are noisy; tighten upstream.

---

## Defaults & valid ranges

| Parameter | Default | Min | Max | Unit |
|---|---|---|---|---|
| `elevation_m` | auto (floor + 1.4) | — | — | m |
| `slab_thickness_m` | 0.20 | 0.02 | 2.00 | m |
| `resolution_m_per_px` | 0.010 | 0.002 | 0.100 | m/px |
| `detector` | `fld` | — | — | enum: `fld` / `hough` / `both` |
| `min_wall_length_m` | 0.50 (panel slider default may differ) | 0.05 | 10.00 | m |
| `manhattan_snap` | `true` | — | — | bool |
| `merge_collinear` | `true` | — | — | bool |

Validated server-side by Pydantic in `backend/app/models/vectorize_job.py`.

---

## Where these live in the code

| Concern | File |
|---|---|
| UI for the settings panel | `frontend/src/components/vectorize/VectorizePanel.tsx` |
| UI for the Clean / Raw / Raster buttons | `frontend/src/components/vectorize/VectorizeViewer.tsx` |
| Frontend types & API client | `frontend/src/api/vectorize.ts` |
| Frontend store | `frontend/src/state/vectorizeStore.ts` |
| Pydantic schemas + defaults + ranges | `backend/app/models/vectorize_job.py` |
| End-to-end pipeline orchestrator | `backend/app/vectorize/pipeline.py` |
| Slice → raster | `backend/app/vectorize/slicer.py` |
| Raster cleanup | `backend/app/vectorize/preprocess.py` |
| Hough / FLD / merge | `backend/app/vectorize/classical.py` |
| Min-length cull + Manhattan + merge | `backend/app/vectorize/regularize.py` |
| DXF emit | `backend/app/vectorize/dxf_writer.py` |

---

*Last updated: 2026-05-19 — kept in sync with the parameter schema in `backend/app/models/vectorize_job.py`. If you add or rename a knob there, update this doc in the same PR.*
