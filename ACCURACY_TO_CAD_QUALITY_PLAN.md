# Accuracy-to-CAD-Quality Plan

> **Goal:** Close the gap between what we generate today (Image 2 — noisy raster + fragmented cyan centerlines + scattered "column" boxes) and what clients deliver (Image 1 — clean, drafted, double-line walls with proper openings, room enclosure, and CAD layer hygiene).
>
> **Scope:** This plan supersedes Path A/B/C/D decisions in `scan-to-cad-vectorization-gap-analysis.md` for the next 6–10 weeks of engineering, then merges back into the roadmap. Phases 0–7 (already shipped) are assumed and not re-litigated.
>
> **Owner:** Tanner Hatch
> **Drafted:** 2026-05-20
> **Status:** v3 — **MacBook MVP track is now the active execution plan** (§ 0.5). GPU refinement appendix added (§ 11). Confirmed priorities locked: external walls first, internal walls second, columns/pillars third. Demo-quality target reachable without any training data, on the MacBook alone — see § 0.5. Post-contract refinement on dual-3090 server — see § 11.
>
> **Confirmed priorities** (from product discussion 2026-05-20):
> 1. **Generation/rendering quality is the top priority** — operator UX track is de-emphasised until the pipeline is producing demo-quality output.
> 2. **External walls (building envelope) first** — must be visually clean and dimensionally accurate.
> 3. **Internal walls second** — close to external quality, but acceptable to have a small visible quality delta.
> 4. **Pillars / columns third** — must not contaminate walls or rooms; some operator cleanup acceptable.
> 5. **Demo-quality without Stevenson archive** — Tier 1 + Tier 2 (pretrained ML, no fine-tune) must be enough to demo. Tier 3 follows the demo, not precedes it.

---

## 0.5. ACTIVE EXECUTION PLAN — MacBook MVP track (no training, no GPU, ~1 week)

**Strategy (confirmed 2026-05-20):** Ship a great MVP on MacBook alone to close a contract. Use the contract scan archive + dual-3090 server to refine to operator parity afterward (§ 11). The MVP must be commercially clean — no license-tainted weights, no training data we don't already own.

**What "MacBook MVP" means in code:**

| Item | Tier 1 ref | Library / cost | Implementation status |
|---|---|---|---|
| **A1 Point-normal filtering** (biggest single missed win) | new | Open3D (already in deps), MIT-equiv | day-1 morning |
| **A3 Building-envelope extraction** | § 1.2 | `alphashape` + `shapely` (already in deps) | day-1 afternoon |
| **A2 Vertical-column density slicer** | § 1.1 | numpy + Open3D (already in deps) | day-2 |
| **A4 Wall-thickness pairing + double-line emission** | § 1.3 | numpy geometry (no deps) | day-3 |
| **A5 Junction snap + cycle-based room inference** | § 1.4 | `networkx` (BSD — add to deps) | day-4 |
| **B1 DeepLSD line detector drop-in** (MIT pretrained, no domain gap) | § 3 Tier 2 alt | `cvg/DeepLSD` (MIT) + ONNX | day-5 |
| **C1–C5 Self-supervised classifier on operator edit log** | new | LightGBM (Apache 2.0) | day-6 |
| **A8 Column hardening + A9 curved-wall arc fit + A7 skeletonize** | § 1.5–1.7 | scikit-image (BSD) | day-7 |

**Why this works without training data:**

1. The biggest accuracy lifts are **all classical CV** — point-normal filtering, density slicing, wall pairing, envelope extraction, junction snap. No model required.
2. **DeepLSD** is MIT-licensed code AND weights, trained on the Wireframe dataset. It learned "what a line looks like", which is content-agnostic — no domain gap when applied to LiDAR density images.
3. **The edit log** in `data/results/*/segments_edited.json` is already labelled training data we own. A LightGBM classifier on handcrafted features trains in seconds on a MacBook and encodes operator domain expertise into the pipeline.

**Estimated outcome (no training data, MacBook-only):**

| Priority | External walls | Internal walls | Columns | Rooms |
|---|---|---|---|---|
| Target (vs. Image 1) | **90–95 %** | 75–88 % | 80–90 % | 75–85 % enclosure |
| Today | ~30 % | ~25 % | ~10 % (over-detecting 5–10×) | 0 % (no inference) |

**Demo readiness:** External walls match Image 1 at a glance. Internal walls look correct with a handful of furniture-against-wall artifacts that the operator can wave off as "honest as-built data — one or two clicks of cleanup." Columns and rooms make the deliverable look like CAD, not a sketch.

**Honest gaps that will remain** until § 11 GPU refinement:

- Furniture against a wall = thicker wall artifact (no semantic disambiguation without ML).
- Heavily occluded walls = single-faced (no hallucination without ML).
- Stevenson-specific drawing-style polish (line weights, layer naming) — operator-applied for now.
- Door swings, casework, fixture symbols — out of scope for MVP.

**Day-by-day execution plan:**

| Day | Code | Outcome |
|---|---|---|
| **1 AM** | `ingest.py` — estimate normals; `slicer.py` — `vertical_only` filter | Floor / ceiling / desk-top noise gone before any other stage |
| **1 PM** | `envelope.py` — concave hull of low-Z point projection; `WALLS_EXTERIOR` layer | Building shell pops in the demo |
| **2** | `density_slicer.py` — vertical-column ratio raster behind feature flag | Walls bright; furniture faint |
| **3** | `walls.py` — pair parallel detections; `WALLS_FACES` layer | Double-line walls = "looks like CAD" |
| **4** | `topology.py` — junction snap + cycle-based rooms; `ROOMS` layer | Plan respects rooms |
| **5** | `ml/deeplsd_detector.py` — ONNX DeepLSD inference; new detector option | Cleaner line quality |
| **6** | `ml/edit_classifier.py` — LightGBM train + inference | Operator's prior corrections inform new runs |
| **7** | `columns.py` polish + `arcs.py` + `skeleton.py` | Pillars ≤ 1/100m², curved exteriors preserved |

**This is the immediate work plan. The next major section (§ 0 TL;DR) covers the broader strategy; everything from § 3 onward describes the post-MVP path (GPU refinement, multi-task models, fine-tune).**

---

## 0. TL;DR — what's wrong and what we'll do

| Layer | What's wrong today | What changes |
|---|---|---|
| **Slicing** | Multi-elevation OR of chest-height slabs (1.07 / 1.47 / 1.87 m). At those heights *everything* — cubicles, filing cabinets, monitors, shelves, partitions — paints into the raster identically to walls. | Switch the primary raster to a **vertical-column density map**: project the *number of points in a vertical wall band* per XY cell. Walls become tall, dense columns. Furniture becomes short, sparse columns. They separate cleanly. |
| **Raster representation** | Binary occupancy (any point = white pixel). Every dense blob looks identical. | **8-bit density image** + percentile normalisation. The "is this a wall?" signal becomes a brightness gradient, not a hard threshold. |
| **Wall detection** | Hough/FLD line fit on a noisy binary raster → 672 raw segments → 142 walls @ 70 % coverage. Lines float as centerlines with no thickness. | Two-stage: (1) ML **semantic mask** (CubiCasa5K-pretrained U-Net or HEAT/RoomFormer pretrained on Structured3D) gates the classical detector; (2) **wall-thickness pairing** collapses parallel detected edges into a single wall entity with thickness, then re-emits as **double-line polygons** matching CAD convention. |
| **Topology** | Floating segments. No corners, no T/L junctions, no enclosed rooms. | **Corner/junction snap** + **room polygon inference** (cycle finding on the wall graph). Final output is room-enclosing closed polylines, not a soup of lines. |
| **Columns** | Anything isolated and dense scores as a column → 19–58 false positives per scan. | Suppress detector inside semantic non-wall mask; require **structural-grid corroboration** (column has to repeat at regular intervals OR sit at a wall-axis intersection). |
| **Validation** | Visual inspection only. No held-out ground truth, no regression gate. | **Eval harness** lands *before* Tier 2. Every change is measured in segment-IoU, recall, precision, and pixel coverage against a labelled fixture set. |
| **Operator UX** | Editor exists (Phases 3 / 3.5 / 3.7 / 4–7) but operator still spends time fixing what the pipeline gets wrong. | Targeted ease-of-use track (§ 6) — one-click "convert centerlines to double-line walls", "infer rooms", "fix junctions", and an "AI review" pass that flags only the segments the model is unsure about. |

**Realistic outcome bands** (against client-quality reference DXFs), broken out by wall class per the confirmed priorities:

| Track | Effort | External walls (#1) | Internal walls (#2) | Columns/pillars (#3) |
|---|---|---|---|---|
| Tier 1 — classical-CV ceiling | 1.5–2 weeks | **80–90 %** (very strong — density slicer + alpha-shape envelope is purpose-built for shell extraction) | 60–70 % | 70–80 % (grid corroboration kills false positives) |
| Tier 2 — pretrained ML drop-in | +2.5–3.5 weeks | **90–95 %** (demo-clean) | 80–90 % (demo-acceptable) | 80–90 % |
| Tier 3 — domain fine-tune (post-demo) | +4–6 weeks (after IP signed) | 97–99 % | 95 %+ | 95 %+ |

**Demo target:** Tier 1 + Tier 2 land before the client demo. External walls look like Image 1. Internal walls look "right" with small visible delta from operator-traced. Columns are clean and few. **Tier 3 follows the demo**, fine-tunes on Stevenson labels to reach operator parity.

---

## 1. Anatomy of the goal image

The first screenshot is a typical Revit/AutoCAD **as-built floor plan** at LOD 200, exported to DXF/PDF for review (the "Crop" button hints at PlanGrid / BlueBeam / Procore). Reading it carefully tells us what the pipeline must eventually produce:

| Feature in Image 1 | Meaning | What our pipeline needs |
|---|---|---|
| Two parallel cyan lines per wall, ~10–20 cm apart | **Wall objects with thickness** — interior partition (~10 cm), demising wall (~15 cm), exterior shell (~25–30 cm) | Wall thickness inference + offset-polyline emission |
| Cyan lines meet cleanly at T / L / cross junctions | **Walls are connected, not floating** | Junction snap + topology graph |
| Red highlight strip on one wall | Operator-added emphasis (a callout for the trade) | Layer slots for client annotations |
| Yellow lines at lower-left | **Stair flights** | Stair layer (deferred — out of scope for v1) |
| White lines inside rooms | Door swings, casework, fixtures, plumbing | Multi-class detection (we ship walls + openings + columns today; doors/windows/casework are next) |
| Suite numbers (e.g. "SUITE 220") | **Room labels** | Room inference (cycle finding) gives us the polygons; labels are operator-entered |
| Grid letters along bottom ("A", "B", "C") | **Structural column grid** | Grid line inference (a clean way to suppress 90 % of our false-positive columns) |
| Curved blue line on the right edge | **Curved exterior wall** | Curve detection (out of scope v1 — Manhattan-only is acceptable for the demo) |
| Solid dark background with dashed cyan beyond crop | **Hidden / above-cut content** | Cosmetic layer (Phase ∞) |

**The non-negotiable visual gates** for a client-quality result are: (i) double-line walls, (ii) clean junctions, (iii) enclosed rooms, (iv) no furniture leaking into wall geometry. Everything else is icing.

---

## 2. Anatomy of our current output (Image 2 + repo metrics)

Two recent runs from `data/results/`:

| Run | `coverage_pct` | `segments_detected` → `_after_regularize` | `columns_detected` | Elevations used (m) |
|---|---|---|---|---|
| `69914141…` | **69.6 %** | 672 → 142 | **19** | 1.07 / 1.47 / 1.87 |
| `35c4b2a7…` | **42.3 %** | 892 → 176 | **58** | 0.31 / 0.71 / 1.11 |

A 42 % coverage means we're **failing to explain over half the foreground pixels** the slicer painted. The 58-column count on a 1000 m² building is ~10× what's plausible — that's furniture-blob escape. Both runs match what's in Image 2: a noisy raster with broken cyan walls and scattered pink boxes.

### Root causes, traced to the code

1. **Slicer paints furniture identically to walls** — `slicer.py::slice_to_raster_multi` projects *every* point in the slab as a white pixel. A monitor, a filing cabinet, and a wall are all "white" if any point lands in the slab. **Multi-elevation makes this worse**, because OR-ing three slabs triples the furniture contamination. The shoulder-height defaults (`floor + 1.6 m ± 0.4`) intersect exactly the furniture band.
   *File:* `backend/app/vectorize/slicer.py` lines 95–210, 213–324
   *Config:* `VectorizeParams` defaults in `backend/app/models/vectorize_job.py`

2. **Preprocess CLOSE bridges noise as readily as walls** — `preprocess.PreprocessParams(close_kernel_px=3, close_iterations=2)` runs a 3-px CLOSE twice. That connects furniture stippling into wall-shaped blobs, which then get detected.
   *File:* `backend/app/vectorize/preprocess.py` lines 18–73

3. **Line detector runs on the noise** — FLD/Hough don't know "wall" from "filing cabinet edge"; they find any contrast edge. With 558 K foreground pixels in run 2, the detector returns 892 segments, most of which are furniture edges.
   *File:* `backend/app/vectorize/classical.py`

4. **Regularizer keeps everything Manhattan-aligned, including noise** — Manhattan snap drops *misaligned* furniture edges but keeps furniture edges that happen to align with the building axes (which is most of them — desks and monitors are axis-aligned too). 176 final segments for a 1000 m² building is ~3× too many.
   *File:* `backend/app/vectorize/regularize.py`

5. **Single centerline per wall, no thickness** — `dxf_writer.write_dxf` emits each segment as a single DXF LINE entity. CAD operators draw walls as parallel offset polylines bounding a wall body. This is the largest *visual* gap on its own.
   *File:* `backend/app/vectorize/dxf_writer.py`

6. **No topology** — segments are independent. No corner inference, no T-junction snap, no room-cycle finding. Floating walls don't enclose anything.

7. **Column detector triggers on furniture** — `columns.py::detect_columns` keeps blobs with 20–120 cm bounding diagonal, ≤ 2.5:1 aspect, ≥ 55 % fill *anywhere* the wall mask doesn't already explain. That description matches a printer, a stand-up desk, a planter, a vending machine, a chair cluster, a server rack…
   *File:* `backend/app/vectorize/columns.py`

8. **No semantic gate** — every downstream stage trusts that "white pixel = wall". The fundamental fix is to add a **"is this pixel a wall?"** classifier before the detector.

---

## 3. The plan, three tiers

Each tier is shippable on its own. Tier 1 unblocks the demo. Tier 2 is where output starts to look like Image 1. Tier 3 is where it crosses operator-parity for the client's domain.

### Tier 1 — Classical CV ceiling (1.5–2 weeks)

Everything below is doable with the libraries already in `pyproject.toml` (Open3D, OpenCV, NumPy) plus one new dependency (`alphashape` or `shapely.concave_hull`). No new model weights, no GPU. Closes most of the *raster-quality* gap.

**Re-ordered by the confirmed priority** (external walls first, internal walls second, columns third):

| # | Item | Serves priority | Why first/early |
|---|---|---|---|
| 1.1 | Vertical-column density slicer | All three | Foundational — without it, every downstream stage sees furniture as walls. |
| 1.2 | Building-envelope extraction (NEW) | External walls (#1) | Cheap classical pass that locks the building shell as a single closed polygon — guarantees the exterior wall is correct *even if everything else fails*. |
| 1.3 | Wall-thickness pairing → double-line emission | External + internal | The single biggest visual lift toward Image 1. |
| 1.4 | Junction snap + cycle-based room inference | Internal walls (#2) | Internal walls need topology to look like rooms, not floating sticks. |
| 1.5 | Column detector hardened | Columns (#3) | Last because it's already partially working; tightening it cleans up the existing output. |
| 1.6 | Skeletonize before detect | Quality polish | Removes the parallel-fragment artefact you see in `overlay_raw.png` today. |
| 1.7 | Curved-wall opt-out | External walls (curved shells) | Image 1 has a curved exterior — Manhattan snap drops it today. |

#### 1.1 Vertical-column density slicer (biggest single win)

**Replace** the current "project all points in a slab to a binary raster" with "for each XY cell, count how many points fall in the vertical wall band (`floor + 0.3 m` to `floor + 2.2 m`), then divide by that band's height". Output is a 32-bit float density image in points-per-vertical-metre.

Why this works: a wall has points at every height from skirting board to ceiling — its vertical column is **tall and dense**. A filing cabinet has points only between 0.3 m and 1.2 m — its vertical column is **short and dense at the bottom, empty above**. A desk has points only at 0.75 m — its column is **sparse**. The ratio of "filled vertical extent" to "total band" cleanly separates walls (≥ 80 %) from furniture (≤ 40 %).

Concretely:
- Bin points into a 3-D voxel grid at XY = `resolution_m_per_px`, Z = 0.1 m.
- Per (X, Y) cell, count occupied Z bins between `floor + 0.3 m` and `floor + 2.2 m`.
- Cell value = `occupied_z_bins / total_z_bins` ∈ [0, 1].
- Clip to the 98th percentile, scale to uint8, threshold at 0.6 to produce a binary wall-mask raster.

**Why we believe this works:** the dynamic-layer-extraction paper (*Dynamic Layer Extraction for Floor Plan Generation from Point Clouds*, Springer 2024) reports +5 % precision and +33 % recall over single-slab methods using this exact strategy on mobile-mapper point clouds. Paul Bourke's industry-standard scan-to-plan method ("Plans and elevations from Laser Scans") also uses 2D point density histograms rather than discrete slicing — for the same reason.

**Code locations:**
- New module: `backend/app/vectorize/density_slicer.py`
- Update: `backend/app/vectorize/pipeline.py::run_vectorize` — swap `slicer.slice_to_raster_multi` with new path behind a feature flag (`use_density_slicer: bool = True` in `VectorizeParams`).
- Preserve the old slicer for A/B comparison until eval-harness blesses the switch.

**Expected result:** raster goes from "every point is a white pixel" to "walls are bright, furniture is faint, empty is black". Detector hit-rate goes up; false-positive segments go way down.

#### 1.2 Building-envelope extraction (priority #1 — external walls)

A dedicated classical pass that finds the **building shell** as a single closed polygon, independently of the line-detector path. This guarantees a clean exterior wall even if the rest of the pipeline misbehaves.

Algorithm:
1. **Floor-level point projection.** From the raw point cloud (not the density raster), grab points at `floor + 0.15 m → floor + 0.60 m` — well below most furniture, above the scanner's floor-noise band. Project to XY.
2. **Concave hull.** Use `shapely.concave_hull(points, ratio=0.05)` or `alphashape.alphashape(points, alpha)` with α auto-tuned by binary search until the hull has the largest connected interior area. This is the **outer building footprint**.
3. **Polygon simplification.** Apply `shapely.simplify(tolerance=0.10)` (10 cm) to drop scanner-noise notches.
4. **Manhattan-align if the building is rectilinear.** Detect the two dominant edge orientations on the envelope; snap edges within 8° to those axes.
5. **Wall-pair the envelope.** Re-slice points at floor+1.4 m within 50 cm of the envelope polygon — that's the *inner face* of the exterior wall. The envelope is the outer face. The two faces give exterior wall thickness directly.
6. **Emit on a dedicated layer.** New DXF layer `WALLS_EXTERIOR` (ACI 5 / blue, lineweight 0.50 mm) — heavier than interior walls, matching CAD convention.

**Why this is its own pass:** the building shell is the most important feature in the deliverable, and it's the *easiest* feature to extract robustly. Treating it as a special case (instead of relying on the generic detector picking it up) makes the demo-critical layer essentially never fail.

**Cost:** ~5–10 s on a typical scan (concave hull is the bottleneck; one-shot, cached).

**Code locations:**
- New module: `backend/app/vectorize/envelope.py` — `extract_building_envelope(pcd, floor_z) -> ExteriorPolygon`
- Update: `pipeline.py` — runs envelope extraction *before* the main detector, persists `envelope.json` artifact.
- New layer: `WALLS_EXTERIOR` (split from `WALLS` so the demo can highlight it).
- Editor: render exterior walls in a heavier blue to match CAD convention.

#### 1.3 Wall-thickness pairing → double-line emission

After detection + regularization, **pair parallel segments that are 7–35 cm apart and overlap significantly** into a single `Wall` entity with `centerline`, `thickness`, and two `face` polylines.

Algorithm:
1. For each kept segment, sort all other segments by perpendicular distance.
2. For each pair within `(0.07 m, 0.35 m)` perp distance and `> 60 %` length overlap and `< 2°` parallel: register as `wall_faces`.
3. Wall thickness = perp distance. Centerline = midline.
4. Unpaired segments fall back to "single-face wall" with the global median thickness as default (operator can adjust).
5. **DXF emit:** new `WALLS_FACES` layer with the actual face polylines, `WALLS_CENTERLINES` layer kept for the editor.

**Why this matters:** even before any room inference, the deliverable instantly looks like Image 1 — double-line walls with consistent thickness. This is the single biggest *visual* lift in the plan.

**Code locations:**
- New module: `backend/app/vectorize/walls.py` — `pair_into_walls(segments) -> list[Wall]`
- Update: `backend/app/vectorize/dxf_writer.py` — emit both layers.
- New layer slot: `WALLS_FACES` (ACI 7 / white, lineweight 0.35 mm).

#### 1.4 Junction snap + cycle-based room inference (priority #2 — internal walls)

Once we have double-line walls:
1. **Junction snap.** Extend every face polyline by 30 cm at each end, find intersections with other face polylines, snap endpoints to the nearest intersection within 25 cm (cap by perpendicular search). Use a KD-tree.
2. **Build wall graph.** Nodes = junctions. Edges = wall face polylines.
3. **Cycle finding.** Run a planar-graph minimum-cycle-basis decomposition (NetworkX has this out of the box: `nx.minimum_cycle_basis`). Each cycle = a room polygon.
4. **Emit rooms.** New `ROOMS` layer with `LWPOLYLINE` per room; expose room polygons in `result.json` so the editor can label them.

**Code locations:**
- New module: `backend/app/vectorize/topology.py` — `snap_junctions`, `build_wall_graph`, `infer_rooms`.
- Editor (Phase 3+ already built): add a "Rooms" layer pill with hover-to-highlight.

#### 1.5 Column detector hardened (priority #3)

Three constraints on the existing column detector:
- **Mask out semantic-wall pixels.** Today the column detector subtracts walls with 12 cm slop; tighten to a binary mask derived from the new density slicer (anything `density ≥ 0.6 wall threshold` is suppressed).
- **Structural-grid corroboration.** Find lines that pass through ≥ 3 column candidates within 30 cm perpendicular and parallel to a building axis. Keep only columns that sit on such a line. Inspired by the grid letters in Image 1 (A / B / C / D) — real columns repeat.
- **Minimum density requirement.** A column should be ≥ 90 % full in its bounding rectangle. Today's 55 % bar is too lax.

Expected outcome: column count drops from 19–58 to 0–6 on residential, 4–12 on commercial. False positives near zero.

**Code:** `backend/app/vectorize/columns.py` — tighten params, add `_corroborate_grid()`.

#### 1.6 Detector polish — skeletonize before Hough

For thin walls (single-pixel-wide raster runs after the density threshold), **skeletonize** the wall mask first (`skimage.morphology.skeletonize`) and run the detector on the skeleton. This eliminates the "many-parallel-fragments" output you see in `overlay_raw.png` today.

**Code:** `backend/app/vectorize/preprocess.py` — add `skeletonize` flag (off by default initially; on by default once stable). Or new file `backend/app/vectorize/skeleton.py`.

#### 1.7 Curved-wall tolerance (priority #1 — external walls; small but visible)

The exterior in Image 1 is curved. Today's Manhattan snap drops it. Add a **per-segment Manhattan opt-out**: any segment that fits a circular arc with RMS < 5 cm over its length is preserved as an arc (DXF `ARC` entity) and excluded from Manhattan snap. Use `cv2.fitEllipse` on the contour, or RANSAC circle fit.

**Code:** new `backend/app/vectorize/arcs.py`; `regularize.py` calls it before Manhattan snap.

#### Tier 1 acceptance criteria

Listed per the confirmed priority order:

**Priority #1 — external walls:**
- Building envelope polygon present, closed, and within 10 cm RMS of the visible exterior in the raster.
- Exterior wall thickness inferred and consistent within ±3 cm along the shell.
- Curved exterior segments preserved as DXF `ARC` entities (not Manhattan-snapped to false right-angles).
- Visually: the `WALLS_EXTERIOR` layer alone, opened in CAD, would be recognisable as the building.

**Priority #2 — internal walls:**
- Coverage (`coverage_pct`) ≥ 85 % overall (currently ~50 %).
- `segments_after_regularize` drops by 40–60 % (fewer, more meaningful entities).
- ≥ 80 % of rooms in the scan are enclosed by a closed polygon in the `ROOMS` layer.
- Double-line walls present in `WALLS_FACES` layer with consistent thickness per wall (≤ 3 cm variance along a single wall).

**Priority #3 — columns/pillars:**
- `columns_detected` ≤ 1 per 100 m² of net floor area (vs. current 10×–20× that rate).
- Zero columns inside the building envelope at clearly-non-structural locations.

**Operational:**
- DXF opens cleanly in BricsCAD / AutoCAD with layers: `WALLS_EXTERIOR`, `WALLS_INTERIOR`, `WALLS_FACES`, `WALLS_CENTERLINES`, `ROOMS`, `OPENINGS`, `COLUMNS`, `WINDOWS`.
- Pipeline runtime ≤ 90 s on a 1000 m² scan on a 2024 MacBook Pro.

---

### Tier 2 — Pretrained ML drop-in (2.5–3.5 weeks)

Tier 1 closes the *raster-quality* gap. Tier 2 closes the *semantic* gap — the pipeline finally understands "is this pixel a wall?" beyond density heuristics.

> **Prerequisite:** Eval harness (§ 4) must ship before Tier 2 starts. Without quantitative ground truth, we cannot tell if a model helped or hurt.

#### 2.1 Wall segmentation: CubiCasa5K-pretrained U-Net (one week)

A pretrained checkpoint exists on Hugging Face: `Yytsi/floorplan-to-3d-walls` — ResNet-34 encoder + U-Net decoder, 4 classes (floor / wall / door / window), mIoU 0.983 on CubiCasa5K validation. ImageNet-pretrained backbone.

**Integration:**
1. Add `segmentation-models-pytorch` and `torch` (CPU build) to `backend/pyproject.toml`.
2. New module `backend/app/vectorize/ml/wall_segmenter.py` — loads the HF checkpoint, exposes `segment(density_image: np.ndarray) -> WallMask`.
3. **Bridge to the density slicer:** the model expects a 512 × 512 RGB image. Replicate our uint8 density image into 3 channels and letterbox to 512² preserving aspect, run, upsample mask back to native resolution.
4. **Two-stage detection:** use the wall mask to *gate* the classical detector (`cleaned_mask = density_mask & wall_seg_mask`). Hough/FLD now runs only on the union of evidence.
5. **Door / window heads come free** — we get those segmentation masks as bonus for the existing OPENINGS / WINDOWS layers (Phase 5 + reserved Window layer).
6. ONNX-export the model for inference speed (≈ 15 s/scan on CPU is acceptable; ≈ 0.5 s with M-series GPU).

**Code:**
- New: `backend/app/vectorize/ml/wall_segmenter.py`
- New: `backend/app/vectorize/ml/__init__.py` (loader + ONNX runtime)
- Update: `pipeline.py` — call `wall_segmenter.segment()` after preprocess, AND it with the density-thresholded mask before detection.
- Bundle weights into `backend/app/vectorize/ml/weights/` (≈ 90 MB; download on first run).

**Domain gap caveat:** CubiCasa5K is rasterized architectural plans, not LiDAR density images. The model will work but with a 5–15 % mIoU degradation vs. its in-domain numbers. The fix is Tier 3 (fine-tune). This still beats anything we can do classically.

#### 2.2 Room / corner reconstruction: RoomFormer or HEAT (1–1.5 weeks)

Both consume a 256 × 256 density map and emit room polygons end-to-end. RoomFormer (CVPR 2023, github.com/ywyue/RoomFormer) is fast (0.01 s inference), watertight by construction, pretrained on Structured3D + SceneCAD.

**Why we want this even after Tier 1's cycle-based room inference:**
- Tier 1 needs perfect junction snap to find cycles. If a junction is 26 cm apart instead of 25 cm, the cycle breaks. RoomFormer doesn't care — it predicts the polygon directly from pixels.
- RoomFormer predicts **room-by-room** semantics. Two adjacent rooms separated by a thin wall are not "one big polygon with a wall sticking through" (the Tier-1 failure mode); they're two polygons.
- The corner / angle accuracy (F1 91.7 % / 89.3 % per CAGE paper) is at operator level.

**Integration as a "second opinion":**
1. Run RoomFormer on the density map (after the U-Net wall mask gates the input).
2. For each predicted room polygon, snap to the Tier 1 wall faces within 30 cm.
3. Where ML and classical disagree by more than 30 cm, **defer to the classical wall** (we trust geometry over an out-of-domain pretrained model — Stevenson commercial scans aren't Structured3D synthetic scenes).
4. Use ML predictions to *promote* low-confidence classical walls and to *demote* segments that fall inside predicted room interiors.

**Alternative:** HEAT (CVPR 2022, github.com/woodfrog/heat) — slightly older, predicts corners + edge classification. Choose RoomFormer if domain transfer looks OK in our eval harness; HEAT if RoomFormer's room-decoder hallucinates rooms that aren't there.

**Newer option to monitor:** CAGE (NeurIPS 2025) — current SOTA with F1 99.1 / 91.7 / 89.3 (rooms / corners / angles). If a clean inference checkpoint releases by the time we ship Tier 2.2, switch to it.

**Code:**
- New: `backend/app/vectorize/ml/room_decoder.py`
- New: `backend/app/vectorize/ml/weights/roomformer_stru3d.pth` (download script)
- Update: `pipeline.py` — runs after the wall-segmenter mask + classical detector, fuses results.

#### 2.3 ML-aware confidence + AI review

`pipeline/ai_review.py` already exists as a skeleton. Wire it up properly:
- For each emitted segment, attach `confidence ∈ [0, 1]` computed as:
  - 0.5 × (segment lies in ML wall mask)
  - 0.3 × (corroborated by detector at multiple elevations)
  - 0.2 × (paired into a thickness-bound wall)
- Segments with `confidence < 0.6` get rendered as **amber** in the editor (vs. emerald for high-confidence).
- The editor's "AI review" mode auto-jumps from one low-confidence segment to the next, asking the operator to keep / drop / redraw.
- Average operator review time drops from "look at the whole plan" to "look at the 8 segments the model isn't sure about".

#### Tier 2 acceptance criteria

- Wall-segment IoU (at 30 mm tolerance) ≥ 0.70 vs. labelled fixture set.
- Coverage ≥ 92 %.
- Visual side-by-side: indistinguishable from operator-traced for residential layouts at a glance; commercial layouts within "looks right with 5 minutes of cleanup".
- Inference cost ≤ 60 s on a 1000 m² scan on a 2024 MacBook Pro (M-series), no GPU required.
- Editor "AI review" pass takes the operator ≤ 5 minutes for a typical 1000 m² scan.

---

### Tier 3 — Domain fine-tune on Stevenson archive (4–6 weeks)

The cap on Tier 2 is the domain gap — pretrained models were trained on synthetic Structured3D or rasterized CubiCasa5K plans, not on Stevenson's real LiDAR density images. Fine-tuning closes that gap.

#### 3.1 Data prep

- **IP paragraph signed** with Stevenson — this is a blocker, called out in the existing gap-analysis doc as a non-engineering item.
- Pull 30–80 historical Stevenson projects with: (i) raw point cloud, (ii) operator-edited final DXF.
- Build the ground-truth labelling script:
  - Rasterize each operator DXF at our pipeline's resolution.
  - Auto-align with the matching point-cloud density image (translate + rotate + scale; bracket the search).
  - Per-pixel labels: `floor`, `wall`, `door`, `window`, `column`, `furniture`, `background`.
- Manual QA: every project gets a 5-minute human pass to fix mis-alignment / label errors. Build a dead-simple labelling UI on top of the existing editor — operator drags a slider to align, hits "OK".

#### 3.2 Fine-tune

- Workstation GPU (Apple M3 Max → ~50 min/epoch on this dataset, or a single RTX-class cloud GPU at ≈ $0.50/hr).
- Fine-tune the CubiCasa5K U-Net on Stevenson labels (50 epochs, mixed precision, on-the-fly augmentation: rotation 0–90°, noise injection, partial occlusion simulation).
- Re-evaluate on a 5-project held-out test set.

#### 3.3 Continuous-improvement loop

- Editor already logs operator edits (Phase 3's edit log).
- Every saved edit becomes a candidate training example: if operator deleted a segment we predicted, that's a "false positive" label; if they drew a segment we didn't, that's a "false negative".
- Monthly: append the new label deltas, re-fine-tune, deploy.
- Eval harness gates deployment: model only promoted if no metric drops > 1 % vs. previous version.

#### Tier 3 acceptance criteria

- Wall-segment IoU ≥ 0.85 at 30 mm tolerance on held-out Stevenson scans.
- Operator review time ≤ 3 minutes per 1000 m² scan.
- < 1 % operator-rejected wall predictions on a representative project.

---

## 4. Eval harness — must precede Tier 2

Without numbers, "the multi-elevation change made things better" is wishful thinking. This is the single most important *non-shipped* item in the existing roadmap (Path C in the prior gap-analysis).

### 4.1 Fixture set

Three populations:

1. **Synthetic.** Generate procedural floor plans (rooms + corridors + 2× furniture density) with known ground-truth wall segments. We control the noise so we can measure pipeline degradation curves. Already 80 % built in `backend/scripts/generate_synthetic_test_data.py`.
2. **Internal small-scale.** 3–5 short scans we've collected with hand-traced ground truth (~1 hour of operator time each).
3. **Stevenson archive (post-IP).** Once IP is signed, the real domain test set. Until then, fixtures 1 + 2 are the bar.

### 4.2 Metrics

| Metric | Definition | Target (Tier 1) | Target (Tier 2) | Target (Tier 3) |
|---|---|---|---|---|
| Segment IoU @ 15 mm | Per-wall intersection over union with ground-truth segment, 15 mm endpoint tolerance | 0.55 | 0.70 | 0.85 |
| Segment IoU @ 30 mm | Same, 30 mm | 0.65 | 0.80 | 0.90 |
| Wall precision | True walls / detected walls | 0.85 | 0.92 | 0.97 |
| Wall recall | Detected walls / true walls | 0.80 | 0.90 | 0.95 |
| Coverage % | Foreground raster pixels within 8 cm of an emitted wall | 0.85 | 0.92 | 0.95 |
| Column FP rate | False-positive columns / true columns | ≤ 0.5 | ≤ 0.2 | ≤ 0.05 |
| Operator review minutes | Median operator time to ship a clean DXF | 8 min | 5 min | 3 min |

### 4.3 Harness implementation

- New CLI: `python -m backend.eval.runner --pipeline v0.4 --dataset internal`
- HTML report at `data/eval/<run>/report.html` with side-by-side ground-truth vs. predicted overlays.
- CI gate: pipeline change cannot land if any metric drops > 2 % vs. main.

**Code:**
- New module: `backend/app/eval/`
- New CLI: `backend/app/eval/__main__.py`
- New fixtures: `backend/tests/fixtures/eval/`

### 4.4 Why this is the gating item

Three of the four "easy" Tier 1 wins (density slicer, wall pairing, column hardening) interact in non-obvious ways. Without a single number to optimise, we'll over-tune one and break another. The harness lets us run the four changes in any order and know which combinations are net positive.

---

## 5. Phasing and dependency order

The schedule is sequenced so the **client demo can happen at end of Week 6** with Tier 1 + Tier 2 shipped. Tier 3 fine-tune is post-demo and gated on Stevenson archive access.

| Week | Track | Deliverable | External walls | Internal walls | Columns |
|---|---|---|---|---|---|
| **0 (this week)** | Plan + eval | Eval harness scaffold + synthetic fixture generator | — | — | — |
| **1** | Tier 1.1 + 1.2 | Density slicer · **building-envelope extraction** · A/B in eval harness | ✓ first cut | — | — |
| **1–2** | Tier 1.3 + 1.7 | Wall-thickness pairing + double-line emission · curved-wall opt-out | ✓ refined | ✓ first cut | — |
| **2** | Tier 1.4 + 1.5 | Junction snap + room inference · column hardening | — | ✓ refined | ✓ refined |
| **2–3** | Tier 1.6 polish | Skeletonize · confidence scoring stub · regression test pass | ✓ polish | ✓ polish | ✓ polish |
| **3** | **Gate: Tier 1 release** | All Tier 1 acceptance criteria met | 80–90 % | 60–70 % | 70–80 % |
| **4** | Tier 2.1 | CubiCasa5K U-Net integration · ONNX export · mask-gated detection | ✓ ML-gated | ✓ ML-gated | ✓ ML-gated |
| **5** | Tier 2.2 + 2.3 | RoomFormer / HEAT integration · confidence-based AI review | ✓ refined | ✓ refined | ✓ refined |
| **6** | **Gate: Tier 2 release — demo-ready** | Output looks like Image 1 | 90–95 % | 80–90 % | 80–90 % |
| **— DEMO HAPPENS —** | | | | | |
| **6+** | Stevenson IP track | Paragraph signed (in parallel with Tier 2 if possible) | | | |
| **7+** | Stevenson archive ingest | Label-generation pipeline · operator QA UI | | | |
| **8–12** | Tier 3 | Fine-tune on Stevenson labels · continuous-improvement loop | 97–99 % | 95 %+ | 95 %+ |

**The demo gate (end of Week 6) is the only thing that must be hit before Stevenson data lands.** If Tier 2 lands faster than 3 weeks, the demo accelerates. If it lands slower, Tier 1 + a slightly worse internal-walls layer is still a credible demo (external walls and shell will already be at 80–90 %).

**Hard dependencies:**
- Tier 2 cannot start without eval harness running on a non-trivial fixture set.
- Tier 3 cannot start without IP paragraph + Stevenson archive access.
- Wall-pairing depends on density slicer landing first (otherwise pair detection sees furniture as walls).
- Room inference depends on wall-pairing (rooms are bounded by face polylines, not centerlines).

**Soft dependencies (i.e. desirable but can be parallelised):**
- AI review depends on confidence scoring, which is easier after Tier 2 model is online but can use Tier 1 heuristics in the meantime.
- Curved walls can ship any time after wall-pairing.

---

## 6. Ease-of-use track (DE-PRIORITISED until after Tier 2)

> **Confirmed priority:** "Quality of the renderings/scan visualisation is our first most important." The editor UX track is therefore deferred until *after* the demo. The items below stay in the plan as a reference for what's next after Tier 2 ships, but no engineering time competes with the accuracy work until then.

Tier 1 and Tier 2 reduce the *number* of edits the operator has to make. The UX track reduces the *time per edit*. These can ship independently of the accuracy work — but later.

| Improvement | Effort | Value |
|---|---|---|
| **One-click "convert centerlines to double-line walls"** | 0.5 day | Operator can switch deliverable style without re-running the pipeline. Useful for client meetings. |
| **"Infer rooms" button** | 0.5 day | Runs the topology pass in the editor (Tier 1 1.3). Operator can iterate on wall geometry then re-infer rooms. |
| **"Heal junctions" button** | 1 day | Walks every endpoint, finds the nearest perpendicular wall, snaps if within 30 cm. Eliminates the "almost-touching" T-junctions. |
| **Auto-resolution suggestion** | 0.5 day | Based on raster size + point count, suggest 5 mm/px for small scans, 10 mm/px for medium, 20 mm/px for huge. Operator no longer has to think. |
| **One-shot "AI review"** | 1 day (Tier 2 dependency) | Walks operator through low-confidence segments only. Skip / keep / fix per segment. |
| **DXF preview in browser** | 2 days | Use `dxf-viewer` so the operator sees the actual delivered file in the tab before download. Avoids "looks fine in editor, broken in CAD" surprises. |
| **Project memory** | 1 day | Remember last-used resolution / detector / Manhattan setting per scan name. Re-run with one click on a fresh upload. |
| **Hotkey legend always visible (or `?`)** | 0.5 day | Lower onboarding friction. Phase 7 added a lot of hotkeys; surface them. |

**Code locations:**
- Frontend hotkeys + UI: `frontend/src/components/vectorize/`
- Backend topology actions: new endpoints in `backend/app/routes/vectorize.py` for "heal_junctions" and "infer_rooms" so they don't require a full re-run.

---

## 7. Success criteria (what "done" looks like)

The user asked for *success, accuracy, and ease of use*. These are the gates:

**Accuracy (the screenshot question):**
- A scan that today produces Image 2 produces, after Tier 2:
  - Double-line walls with median thickness 10–15 cm (residential) or 15–25 cm (commercial).
  - Enclosed room polygons.
  - ≤ 1 false-positive column per 100 m².
  - ≥ 92 % of wall raster covered.
  - No furniture leaking into walls.

**Success (operator metric):**
- Operator review time on a 1000 m² scan ≤ 5 minutes after Tier 2; ≤ 3 minutes after Tier 3.
- Operator can ship a Stevenson-style DXF without leaving the browser.

**Ease of use:**
- A first-time user uploads, hits "Vectorize" with defaults, and gets a result that looks like Image 1 within 60 seconds.
- No mandatory parameter tuning. The advanced panel is for power users only.
- One-click "make it look like a CAD deliverable" — the double-line walls + rooms toggle.

**Operational:**
- Pipeline runs ≤ 60 s on a 1000 m² scan on a 2024 MacBook Pro.
- Memory peak ≤ 4 GB.
- ONNX inference paths so a future Linux/Docker production env doesn't drag PyTorch.

---

## 8. Risks and decisions you need to make

### Decisions (resolved on 2026-05-20)

1. **Image 1 is the client's FINAL deliverable** — the accuracy bar is "match this". Tier 2 is the demo target.
2. **External walls > internal walls > columns** — confirmed priority order. Engineering reflects this.
3. **Stevenson data after the demo** — Tier 1 + Tier 2 must reach demo-quality on synthetic + internal fixtures alone. Tier 3 is post-demo.
4. **Editor UX is deferred** until accuracy is at Tier 2.

### Remaining decisions

1. **Compute budget for Tier 3.** Tier 2 inference fits on CPU. Tier 3 fine-tune wants a workstation GPU or cloud credit. Cost: $2 K hardware *or* ~$50–200 of cloud GPU spend per fine-tune cycle. Worth confirming before week 6.
2. **Demo dataset.** Which scan(s) is the demo run on? Need to pick this ASAP because Tier 1 acceptance is measured on it. If we don't have a "demo scan" lined up, the eval harness's internal fixture set becomes the de facto demo target.
3. **Visual style of the deliverable.** Image 1 shows specific layer colours (cyan walls, yellow stairs, red highlights). Should the DXF default to those colours, or to a more neutral palette the client overlays their own theme on? Worth a 5-minute decision before week 4.

### Technical risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| CubiCasa5K U-Net fails to transfer to LiDAR density images | Medium | Fine-tune as Tier 3 (planned). Fallback: hand-tune the density-image preprocessing to better mimic CubiCasa rasterizations. |
| RoomFormer hallucinates rooms on commercial layouts (large open spaces, atriums) | Medium | Use ML predictions as suggestions, not ground truth; classical wall geometry wins ties. |
| Wall-pairing fails on single-faced walls (scanner could only see one side, e.g. exterior of building) | Medium | Fallback to "single-face wall + median thickness from paired walls". |
| Junction snap creates phantom rooms when walls almost-cross but shouldn't | Low | Tune `endpoint_extension_m` to 0.20 m initially; if false rooms appear, drop. Eval harness catches this. |
| Density slicer over-rejects partial-height walls (kneewalls, half-height partitions) | Medium | Add a second "low-band" raster at `floor + 0.3 m → floor + 1.0 m` and combine with the main raster for half-height walls. |
| ONNX export of segmentation models is finicky | Low | If it fails, ship PyTorch CPU runtime; Docker image stays under 2 GB. |

### Schedule risk

- The plan assumes 6–10 weeks of focused engineering. If Stevenson sessions or other client work interrupt, slip Tier 3 first, then Tier 2 polish; never slip the eval harness.

---

## 9. What to do tomorrow

The first three days of work, ordered around the external-walls-first priority:

1. **Day 1.** Build the eval harness skeleton + synthetic fixture generator (extend `backend/scripts/generate_synthetic_test_data.py`). Land a metrics CLI that produces a number on the current pipeline against at least one fixture. **Pick the demo scan** so the harness measures progress against it specifically.

2. **Day 2 — external walls #1.** Implement the **building-envelope extraction** (§ 1.2) — concave hull of low-Z point projection. Render the resulting polygon on top of the current overlay. This produces the demo screenshot's first "ooh" moment: the building shell pops out cleanly even before anything else changes.

3. **Day 3 — external walls #2.** Implement the **vertical-column density slicer** (§ 1.1) behind a feature flag. A/B vs. binary slicer on the demo scan. Compare visually: density map should already make external walls look like thick bright bands.

4. **Day 4 — external walls #3.** **Wall-thickness pairing** (§ 1.3) — drop the new `WALLS_FACES` DXF layer. Open output in BricsCAD. The screenshot test (does the exterior shell look like Image 1?) starts to pass.

By end of week 1, external walls should be 70–80 % of the way to demo quality on the chosen demo scan, before any ML work begins. Everything after that is incremental against the eval harness numbers — no more visual-only iteration.

---

## 10. Cross-references

- Existing gap-analysis (decisions, phases 0–7): `scan-to-cad-vectorization-gap-analysis.md`
- Parameter reference (current state): `VECTORIZE_SETTINGS.md`
- Initial implementation plan (pre-MVP): `IMPLEMENTATION_PLAN.md`
- Code audit: `AUDIT.md`
- Dev guide: `DEV_GUIDE.md`

**Files that will be added by this plan:**

```
backend/app/vectorize/
  density_slicer.py         # Tier 1.1
  envelope.py               # Tier 1.2 — building envelope (priority #1)
  walls.py                  # Tier 1.3 — pairing into thickness-aware walls
  topology.py               # Tier 1.4 — junctions + rooms (priority #2)
  arcs.py                   # Tier 1.7 — curved walls (priority #1 — curved shells)
  skeleton.py               # Tier 1.6 — skeletonize before detect
  ml/
    __init__.py
    wall_segmenter.py       # Tier 2.1
    room_decoder.py         # Tier 2.2
    weights/                # downloaded on first run
backend/app/eval/
  __init__.py
  __main__.py               # CLI
  metrics.py
  fixtures.py
  report.py
backend/scripts/
  generate_synthetic_test_data.py   # extend (exists)
  download_model_weights.py         # new (Tier 2)
backend/tests/fixtures/eval/        # ground-truth labels
frontend/src/components/vectorize/
  RoomsLayer.tsx                    # editor for inferred rooms
  AiReviewMode.tsx                  # walk low-confidence segments
```

---

*This is a plan, not a contract. Numbers and timelines are estimates with substantial uncertainty bands. The eval harness exists in this plan precisely so we replace estimates with measurements as soon as the work starts.*

---

## 11. Final Refinement on Dual 3090 (Post-Contract)

> **Trigger:** Contract signed with a client (Stevenson or first paying customer). Their historical scan archive becomes the training corpus. The MVP shipped per § 0.5 has demonstrated demo-quality output to close the contract. This section describes what we do *after* that to reach operator-parity quality.
>
> **Hardware:** Dual NVIDIA RTX 3090, 48 GB total VRAM, on the user's own server. **No cloud GPU spend required for any of this.**
>
> **All code remains MIT/Apache/BSD licensed. All training data is either commercially clean (ProcTHOR-10K Apache 2.0) or owned by us (client archive, our procedural generator, operator edit log).**

### 11.1 Synthetic LiDAR simulator (Phase R1, ~1 week)

**Why:** Off-the-shelf models (CubiCasa, RoomFormer) have a *domain gap* — they were trained on rasterized floor plans or synthetic top-down renderings, not LiDAR density images. The fix is to generate training data that matches our inference distribution exactly.

**Tech:** Open3D ray-casting on ProcTHOR-10K meshes + our existing procedural generator. For each scene:

1. Place 4–8 virtual scanner positions inside the scene.
2. Cast rays from each position to mesh surfaces with realistic LiDAR artifacts:
   - ±2 cm Gaussian range noise
   - Random occlusion holes (5–20 % of rays dropped)
   - Multi-scan registration jitter (±1 cm per scan position)
   - Varying point density per surface (closer surfaces = denser)
3. Output a synthetic LAS file → ingest through the **actual production pipeline** → produces a density image identical in distribution to real scans.
4. Use ProcTHOR's ground-truth meshes to compute per-pixel labels: `{void, wall, floor, ceiling, door, window, furniture}`, plus per-pixel wall-thickness and an "is-exterior" channel.

**Output:** ~50 000 (density_image, label_mask) pairs cached locally to `data/training/synthetic/`.

**Code:** new `backend/app/vectorize/ml/simulator/` package — `raycast.py`, `scene_loader.py`, `label_generator.py`, `augmentations.py`. All MIT.

### 11.2 Multi-task wall segmenter (Phase R2, ~1 week training)

**Architecture:** ResNet-50 (ImageNet init) + U-Net decoder via `segmentation_models_pytorch` (MIT). Single backbone, four output heads:

| Head | Output | Loss |
|---|---|---|
| Semantic | 7-class per-pixel mask (void/wall/floor/ceiling/door/window/furniture) | Tversky (α=0.7, β=0.3) — penalises false positives more than false negatives |
| Wall-thickness | per-pixel cm (regression, only valid on wall pixels) | Smooth L1, masked to wall pixels |
| Exterior-wall | binary per-pixel (is this pixel part of the building shell?) | BCE |
| Corner heatmap | per-pixel "is corner" probability | Focal |

**Training on dual 3090:**
- Batch size 64 (32 per GPU via DDP), 50 epochs.
- Mixed precision (`torch.cuda.amp`).
- ~8–12 hours total wall-clock on dual 3090.
- Augmentations: rotation 0–90°, scale 0.8×–1.2×, random crop, noise injection, partial occlusion simulation, intensity jitter.
- Track per-class IoU; gate model release on `val_wall_iou ≥ 0.85`.
- Export to ONNX → ships back to the MacBook pipeline. Inference 5–10 s on CPU.

**Code:** new `backend/scripts/train_wall_segmenter.py`, `backend/app/vectorize/ml/wall_segmenter.py`. ONNX weights checked into LFS at `backend/app/vectorize/ml/weights/wall_segmenter_v1.onnx`.

### 11.3 Room polygon decoder (Phase R3, ~1.5 weeks)

**Why:** Tier-1 cycle-based room inference (§ 1.4) needs junctions to be within 25 cm of each other for cycles to close. On complex layouts this fails. A learned room-polygon decoder doesn't depend on perfect junction snap.

**Architecture:** Custom RoomFormer-style transformer decoder. Reuses RoomFormer's MIT-licensed code (architectures are MIT) but **retrained from scratch on our synthetic data** — no Structured3D license taint anywhere.

- Two-level query: polygon-level + corner-level (per RoomFormer paper).
- Variable-length polygon output.
- Trained on the same simulator output as 11.2.

**Training:** ~24 hours on dual 3090.

**Inference:** ≤ 1 s on CPU via ONNX. Drops into pipeline after the segmenter.

### 11.4 Client-data fine-tune (Phase R4, ~1 week per client)

**Once a contract is signed:**

1. Ingest the client's historical scan archive into `data/training/client/<client_slug>/`.
2. Auto-align operator-edited DXFs with the corresponding density images:
   - Rasterize each DXF at our pipeline's resolution.
   - 2D ICP between DXF raster and density-image raster.
   - Per-pixel labels derived from DXF layer assignments.
3. **Operator QA pass:** new endpoint in the existing editor renders the auto-alignment for a human sanity check. Operator drags a slider to fix alignment, hits "OK". ~5 min/project × 30–80 projects = 2.5–7 hours total operator time per client.
4. Fine-tune the segmenter + room decoder for 10–20 epochs on client data with a low learning rate (1e-5). Mix in 20 % synthetic data per batch to prevent catastrophic forgetting.
5. ~2–4 hours total training on dual 3090.

**Per-client checkpoints:** `backend/app/vectorize/ml/weights/wall_segmenter_<client>_v<N>.onnx`. Each client's pipeline auto-selects their checkpoint if present, falls back to the base model otherwise.

### 11.5 Continuous-improvement loop (Phase R5, ongoing)

The editor already has a full edit log (`segments_edited.json` + `EditEvent` schema in `models/vectorize_job.py`). Wire this into the training pipeline:

| Trigger | Action |
|---|---|
| Operator saves a project | Append `(density_image, predicted_mask, operator_corrected_mask)` to `data/training/online/<client_slug>/` |
| Weekly | Compute deltas; if ≥ 5 new projects exist for a client, queue a fine-tune cycle |
| Fine-tune completes | Run eval harness against held-out test set; promote to production only if no metric drops > 1 % |
| Monthly | Run on the full corpus (synthetic + ProcTHOR + all clients) to refresh the base model |

**Effort:** ~1 day to wire up, then automated.

### 11.6 Additional data sources (commercially-clean, optional)

| Dataset | License | Use |
|---|---|---|
| **ProcTHOR-10K** | Apache 2.0 | Base pretrain — 10K diverse synthetic floor plans |
| Our procedural generator | Ours | Targeted edge cases: long corridors, atriums, T-junctions, curves |
| **HM3D** (Habitat-Matterport) | CC-BY 4.0 (commercial OK with attribution) | Real-building geometry diversity |
| **Habitat-Sim sample scenes** | Various — case-by-case | Realistic indoor scenes for rendering |
| Stevenson archive | Per Stevenson contract | Domain-specific gold for client fine-tune |
| ZInD (Zillow Indoor) | Registration-gated; verify ToU before use | Optional residential diversity |

**Excluded for license reasons:** CubiCasa5K (CC-BY-NC), Structured3D (non-commercial research), HEAT pretrained weights (GPL + non-commercial), Yytsi/floorplan-to-3d-walls (data-tainted from CubiCasa5K).

### 11.7 IFC export (parallel deliverable)

Independent of the ML work but enabled by it (walls now have semantic class + thickness + opening associations):

- New module `backend/app/vectorize/ifc_writer.py` using `ifcopenshell` (LGPL — fine for our use; we link, not modify).
- Emits IFC 4 with `IfcWall`, `IfcDoor`, `IfcWindow`, `IfcColumn`, `IfcSpace` entities.
- Walls have thickness, materials, opening associations. Spaces have room labels.
- Stevenson and any AEC customer gets BIM-grade output, not just CAD lines. Most competitors only ship DXF.

### 11.8 Differentiators — what makes the refined product "revolutionary"

1. **Sim2Real domain randomization for LiDAR density images** — almost no published work does this for our specific input distribution. We'd be first-to-market.
2. **Multi-task single-model architecture** — walls, doors, windows, columns, thickness, exteriors, corners all from one forward pass.
3. **Continuous learning from operator edits** — competitors ship static models. We improve weekly without any new training data we don't already own.
4. **DXF + IFC dual export** — BIM-grade output, not just CAD lines.
5. **Per-client fine-tuned weights** — each client's pipeline learns their specific drawing conventions.
6. **Defensible IP moat** — code stays MIT (great for hiring + community), the synthetic LiDAR simulator scene library + trained weights + edit-log corpus are our private assets.

### 11.9 Timeline summary (post-MVP)

| Phase | Effort | Hardware | Cost |
|---|---|---|---|
| MacBook MVP (§ 0.5) | ~1 week | MacBook | $0 |
| **— DEMO, CONTRACT SIGNED —** | | | |
| R1 — Synthetic LiDAR simulator | ~1 week | MacBook + 3090 (data prep) | $0 |
| R2 — Multi-task segmenter training | ~1 week | Dual 3090 | $0 (electricity ~$2) |
| R3 — Room polygon decoder | ~1.5 weeks | Dual 3090 | $0 |
| R4 — Client-data fine-tune | ~1 week per client | Dual 3090 | $0 |
| R5 — Continuous-improvement loop | ~1 day setup, then automated | Dual 3090 (scheduled) | $0 |
| R7 — IFC export | ~1 week (parallel) | MacBook | $0 |
| **Total to operator-parity** | ~6 weeks post-contract | All in-house | **~$2 electricity** |

**The MVP closes the contract. The refinement track makes the product defensible long-term.**

---

*Last revised 2026-05-20 evening. v3: added MacBook MVP track and GPU refinement appendix in response to "ship something to close the contract, refine after."*

