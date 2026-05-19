# PRD: Scan-to-Plan Alignment MVP

**Status:** Draft v1
**Owner:** Tanner Hatch (contractor)
**Target:** Localhost demo, single-floor end-to-end
**Working name:** _AlignAI_ (placeholder — rename before client delivery)

---

## 1. Summary

A desktop web application that automatically aligns a 3D LiDAR scan of an **entire building floor** — including every room, hallway, bathroom, and common area — to its corresponding 2D architectural plan (DXF), then displays the result with a confidence score and a cinematic "snap-to-place" animation. After alignment, the floor is rendered with each room visually distinguished (color-coded fills, room labels pulled from the DXF, per-room match-quality indicators) and **wall fixtures detected and marked** — electrical panels, outlet boxes, door frame trims, and other protrusions that stick out of the wall plane. The alignment math itself uses a **clean horizontal "wall band" slice** of the scan (typically chest height) to avoid being thrown off by floor furniture, ceiling fixtures, or other real-world clutter.

The current manual workflow takes ~30 minutes per building floor and is performed by 25 employees at the client's company. The MVP reduces this to under 5 minutes per floor with a deterministic geometric core, optionally assisted by a vision model for sanity-checking.

**Scope clarification:** v1 covers a single floor of a single building with full per-room visualization and wall-fixture *detection and marking*. v1 does *not* produce per-room dimensional measurements or *classified* fixture identification (e.g. "this specific protrusion is a 200-amp electrical panel"). Measurement work and AI-driven fixture classification are v2.

The MVP runs entirely on localhost (frontend + backend on the contractor's laptop) for live demo purposes. Production deployment is out of scope for this phase.

---

## 2. Problem

The client's team currently aligns LiDAR scans to building plans by hand: opening both files in CAD software, eyeballing landmarks (corners, doorways, stairwells), and incrementally rotating and translating the scan until it visually matches the plan. The work has three failure modes:

1. **Time.** 20–45 minutes per building floor depending on complexity.
2. **Variance.** Two technicians given the same scan and plan produce slightly different alignments. There is no quantified "this is the right answer" baseline.
3. **No audit trail.** No record of why a given alignment was chosen, what was tried, or how confident the technician was.

Long-term, the client wants this output usable in forensic and legal contexts (insurance disputes, construction litigation, post-incident reconstruction). The manual workflow doesn't meet the reproducibility bar that requires.

---

## 3. Users

| Persona | Role | Primary need |
|---|---|---|
| **Alignment Technician** | Day-to-day operator. Currently doing this manually. | A tool that does 95% of the work and lets them approve/correct the 5% |
| **Project Manager** | Oversees throughput, signs off on completed buildings | Confidence scores, batch status, exception flagging |
| **Forensic Reviewer** (future) | Downstream consumer of the data when it appears in a case | Provenance, reproducibility, error bounds. _Not in MVP scope._ |

The MVP targets the Alignment Technician persona. Everything else is v2+.

---

## 4. Goals

- **G1:** A technician can drag a plan file and a scan file into the app and see a correctly aligned result in under 30 seconds.
- **G2:** Each alignment produces a quantified confidence score (0–100) and residual error in millimeters.
- **G3:** When confidence is high (>90%), the technician approves with one click.
- **G4:** When confidence is low (<90%), the technician can manually nudge the alignment and the app records the correction.
- **G5:** The full pipeline is deterministic — same inputs always produce the same alignment.
- **G6:** A complete demo runs on a single laptop with no internet dependency except for the optional AI review call.
- **G7:** Each room on the aligned floor is visually distinguished — color-coded fill, label from the DXF, and a per-room match-quality badge — so the technician (and the client during the demo) can read the floor's structure at a glance.
- **G8:** A technician can click any room in the viewer to focus on it (camera frames the room, scan points within that room are highlighted) without changing the underlying alignment.
- **G9:** The alignment pipeline uses a clean horizontal "wall band" slice of the scan (default ~0.75m to ~1.8m above floor — roughly a 3.5-foot band at chest height) so that floor furniture, ceiling fixtures, baseboards, and other vertical clutter do not degrade plane detection or alignment quality.
- **G10:** After alignment, wall protrusions (electrical panels, outlet boxes, door frame trims, fire extinguisher cabinets, thermostats, anything sticking out of the wall plane by 2cm+) are detected geometrically and marked on the wall with visible indicators showing position and approximate size.

---

## 5. Non-Goals (v1)

Explicitly out of scope to keep MVP tight:

- Multi-floor stacking (separate scans of different floors of the same building)
- **Multi-scan registration for isolated rooms** (combining scans with no shared visual content — requires room-topology graph matching, shape/wall-length heuristics). _Scans with doorway overlap or from professional pre-registered exports are fully supported in v1._
- Per-room **dimensional measurements** (room areas, wall lengths, corner coordinates). Per-room *visualization* IS in v1; quantitative measurements are v2.
- **Classified fixture identification** ("this is an electrical panel" vs "this is a door trim"). Fixture *detection and marking* IS in v1; AI-driven labeling is v2.
- **Accurate fixture dimensions** (exact size, mounting heights, model identification). Approximate bounding boxes are shown in v1 for visualization only.
- Wall corner extraction with reported coordinates
- Forensic evidence bundling, cryptographic signing, audit trail
- User accounts, multi-tenancy, role-based permissions
- Production deployment, cloud hosting, scaling
- DWG support — DXF only for v1 (DWG requires a separate conversion toolchain)
- Raster-only PDF plans — vector DXF only for v1 (raster path is a v2 add-on using vision OCR)
- Batch processing of >1 building at a time
- Persistent project history beyond the current session

**Shipped beyond original v1 scope (added during development):**

- **Scan-only mode** — users can upload LiDAR scans with no DXF plan. The pipeline auto-generates a 2D floor plan via wall-plane detection + 2D projection line RANSAC. Confidence is 100% (the plan IS the scan). This enables the "no DXF required" demo flow.
- **Multi-scan merging** — up to N LAZ/LAS/PLY/E57 files per job, one per room. Pre-registered scans are detected (centroid spread + spatial overlap check) and concatenated. Unregistered scans fall back to FPFH + ICP pairwise registration.
- **Height-filtered viewer PLY** — only wall-band points (0.2–2.5m above floor) are sent to the browser viewer, removing floor reflections, ceiling data, and outdoor trees captured through windows.

These are now in scope for v1 and implemented.

---

## 6. User Flows

### Flow A — Happy path

1. Technician opens app at `localhost:5173`.
2. Drag-and-drops `plan.dxf` and `scan.las` (or `.ply`, `.e57`) into the upload zone.
3. App shows ingestion progress (~5s for parsing, ~10–20s for plane detection, ~3s for room extraction, ~5s for fixture detection).
4. App computes alignment and displays:
   - 3D viewer showing scan and plan overlaid, with the snap animation playing once
   - Each room rendered with a translucent colored fill and a label pulled from the DXF (e.g. "Conf 204", "RR-M", "Lobby")
   - Wall fixtures marked with small colored disks (electrical panels, outlet boxes, door trims, etc.) at their detected positions
   - Confidence: **97%**, Residual error: **6.2mm**
   - "Aligned in 18.7s · 12 rooms · 47 wall fixtures detected"
5. Technician reviews the overlay (toggle scan opacity slider, toggle fixture visibility, rotate view, hover a room to see per-room match quality, hover a fixture to see its bounding box and protrusion depth).
6. Technician clicks any room → camera frames it, scan points within that room highlight.
7. Technician clicks **Approve**.
8. App generates a downloadable result file (`aligned.dxf` or `aligned.json` with the 4x4 transformation matrix, room polygons, fixture locations, and metadata).

### Flow B — Low confidence

1. Same upload, but confidence comes back at 64%.
2. App displays the alignment with a yellow warning banner: "Manual review recommended."
3. Optional AI review section shows: _"Northeast quadrant matches well, but the southwest corner of the scan extends beyond the plan boundary by ~1.5m. Possible curved wall not in plan, or scan includes adjacent space."_
4. Technician toggles to manual mode:
   - Drag-rotate handle on the scan
   - Arrow keys to translate
   - "Snap to nearest wall" helper
5. Technician adjusts, clicks **Re-evaluate** → new confidence score.
6. Approves once satisfied.

### Flow C — Failure

1. Files upload, but plane detection finds no dominant wall directions (highly curved building or sparse scan).
2. App displays: "Could not auto-align. Manual mode only."
3. Technician falls back to fully manual placement, app records the result.

---

## 7. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Ingest DXF plans (vector lines, polylines representing walls) | Must |
| FR-2 | Ingest LiDAR scans in LAS, LAZ, PLY, and E57 formats | Must |
| FR-3 | Detect planar surfaces in the scan via RANSAC | Must |
| FR-4 | Filter detected planes to vertical (wall) planes within a height range | Must |
| FR-5 | Compute the scan's principal horizontal axes from wall normals | Must |
| FR-6 | Compute the plan's principal horizontal axes from DXF line angles | Must |
| FR-7 | Apply rotation alignment + centroid translation between scan and plan | Must |
| FR-8 | Refine alignment via ICP (Iterative Closest Point) | Must |
| FR-9 | Test 4 rotational ambiguity candidates (0°/90°/180°/270°), pick lowest residual | Must |
| FR-10 | Return a 4x4 transformation matrix, confidence score, and residual error | Must |
| FR-11 | Display the scan and plan in a 3D viewer, both pre- and post-alignment | Must |
| FR-12 | Animate the snap from initial to aligned position (smooth tween, ~1.5s) | Must — this is the wow |
| FR-13 | Provide a scan opacity slider, view rotation controls, view reset | Must |
| FR-14 | Provide manual nudge controls (rotate handle, translate via arrow keys) | Must |
| FR-15 | Re-evaluate confidence after manual adjustment | Must |
| FR-16 | Call an OpenRouter vision model for "looks right" sanity check on the aligned overlay | Should |
| FR-17 | Export the aligned result as JSON (matrix + metadata) and as a transformed DXF | Must |
| FR-18 | Show processing logs in a collapsible panel for debugging | Should |
| FR-19 | OCR fallback for plans provided as raster PDF (optional, behind a feature flag) | Could (v2 candidate) |
| FR-20 | Extract room polygons from the DXF — closed polylines and/or polygon reconstruction from wall line segments | Must |
| FR-21 | Extract room labels from DXF TEXT and MTEXT entities; associate each label with its containing room polygon | Must |
| FR-22 | Render each room as a translucent colored fill in the 3D viewer, with a billboarded label above it | Must |
| FR-23 | For each room, compute a per-room match-quality score (fraction of in-room scan points within tolerance of a plan wall); display on hover | Should |
| FR-24 | Click a room to focus the camera on it and highlight its scan points; click empty space to deselect | Should |
| FR-25 | Detect the floor plane and compute height-above-floor for every scan point | Must |
| FR-26 | Extract a clean horizontal "wall band" slice (default 0.75m–1.8m above floor, configurable) for use in plane detection and alignment | Must |
| FR-27 | Detect wall protrusions: cluster points that lie 2cm–30cm in front of any detected wall plane, output bounding box, centroid, and protrusion depth per cluster | Must |
| FR-28 | Render fixture markers in the 3D viewer (small colored disks on the wall surface) at detected positions; hover to see depth and size | Must |
| FR-29 | Provide a toggle to show/hide fixture markers without recomputing | Should |
| FR-30 | Per-wall fixture counts available in the export JSON | Should |

---

## 8. Non-Functional Requirements

- **Determinism.** The alignment pipeline must be deterministic. Given identical input bytes, the output transformation matrix must match bit-for-bit across runs. This rules out using AI models in the measurement path; vision is review-only. The wall band slicing and fixture detection are also deterministic — same input bytes always produce the same detected fixtures.
- **Performance.** Total time from file drop to displayed result: under 30 seconds for a typical 50,000-square-foot office floor with a 5M-point scan (this includes the fixture detection pass). Under 60 seconds for the largest expected case.
- **Accuracy target.** On Manhattan-axis buildings (90% of office stock), residual error after ICP should be under 10mm RMS. This is not yet a forensic claim — that requires validation against ground truth in v2.
- **Robustness to clutter.** The alignment must succeed on scans containing typical office contents (desks, chairs, cabinets, ceiling HVAC, hanging lights) by using the clean wall band slice. Test data must include cluttered real-world scans, not just empty rooms.
- **Robustness.** App must not crash on malformed inputs. Bad files produce a clear error message and graceful failure.
- **Local-only.** Backend and frontend both run on `localhost`. The only outbound network call is to OpenRouter for the optional AI review step. The user must be able to demo the app on a hotel WiFi with the AI step disabled.

---

## 9. Technical Approach (Summary)

Detailed in the Implementation Plan document. High level:

- **Geometric core:** Python service using Open3D (point cloud ops, RANSAC, ICP), ezdxf (DXF parsing), laspy (LAS reading), pye57 (E57 reading). FastAPI for the HTTP interface.
- **Frontend:** React + Vite + Three.js. Tailwind for styling. The 3D viewer is a Three.js scene with two point cloud objects whose transformation matrices are tweened.
- **AI integration:** OpenRouter API. Vision-capable model called with the rendered top-down overlay image; returns a confidence interpretation and natural-language summary. Optional and behind a toggle.
- **State:** SQLite for job metadata (overkill but cheap to add and useful for the audit story in v2). Local filesystem for uploaded files and intermediate artifacts.

---

## 10. Success Metrics

For the MVP demo:

- ✅ Drag-and-drop a real test pair, see an aligned result with the snap animation in under 30 seconds.
- ✅ Confidence score and residual error display correctly.
- ✅ Each room on the floor renders with a colored fill, label, and per-room match badge on hover.
- ✅ Wall fixtures (electrical panels, outlet boxes, door trims, etc.) are detected and rendered as visible markers on the walls.
- ✅ Clicking a room focuses the camera and highlights its scan points.
- ✅ Manual nudge mode works smoothly.
- ✅ Export produces a valid JSON + transformed DXF.
- ✅ AI review step (when enabled) produces a sensible English summary.
- ✅ Runs on a fresh laptop with one setup command.

Stretch metric (for the sales conversation, not the engineering deliverable):

- 90%+ accuracy on a curated test set of 10 real (plan, scan) pairs supplied by the client before development starts. "Accuracy" here means: the alignment matches what an experienced technician would produce, within 25mm.

---

## 11. Roadmap Beyond MVP

These are not commitments — they're framing for the client conversation about what the larger contract unlocks.

**v2 (next 2–4 weeks after MVP signoff):**
- Multi-floor stacking
- **Room-topology stitching for isolated scans** — when scans have no overlap (truly independent room captures), use wall-length + doorway-opening graph matching to stitch rooms together without ICP. Requires: (1) detecting wall openings (doorways) in each room scan, (2) building a topology graph, (3) combinatorial search for the layout that satisfies all adjacency constraints. ~3–4 weeks engineering effort.
- Per-room **dimensional measurements** — room areas, wall lengths, corner coordinates with reported error bounds (the room polygons from v1 become the input to per-room geometric extraction)
- Wall corner extraction with reported coordinates
- **AI-driven fixture classification** — vision model labels detected protrusions (electrical panel, outlet, door trim, thermostat, fire extinguisher, HVAC return, etc.) with confidence scores
- **Accurate fixture dimensions and mounting heights** — measurements taken from the detected bounding boxes
- Raster PDF plan support via vision OCR

**v3 (forensic posture, 1–2 months):**
- Cryptographic chain-of-custody for every alignment
- Versioned, hash-pinned pipeline configurations
- Exportable evidence bundles
- Calibration validation against ground-truth reference rooms
- SOP document signed by a Professional Land Surveyor

**v4 (scale, 2–3 months):**
- Cloud deployment
- Batch ingestion API
- COPC / EPT cloud-native storage
- Worker pool for parallel processing
- Path to the client's 200M-asset historic backlog

---

## 12. Open Questions

These need answers from the client before or during week 1 of development:

1. **What file formats are actually in use?** DXF vs DWG, LAS vs E57 vs proprietary. If anything other than DXF + LAS/PLY, scope adjusts.
2. **Do they have test data?** Need 5–10 (plan, scan) pairs under NDA before coding starts. These are the validation set.
3. **What's the typical scan size?** Point counts. Affects performance targets.
4. **What scanners do they use?** Affects expected noise levels and density.
5. **What's their definition of "successful alignment"?** Is it visual approval by a senior technician, or a numeric tolerance, or both?
6. **Is there a coordinate system convention?** Z-up vs Y-up, units, georeferenced or local.
7. **Buildings with curved walls or non-Manhattan geometry — how common?** Determines how much the failure path matters.
8. **How cleanly are rooms drawn in the DXFs?** Closed polylines per room (easy to extract) vs. individual line segments that happen to enclose space (requires polygon reconstruction). Look at 2–3 representative DXFs in a viewer before committing to the week 4 timeline.
9. **Do the DXFs reliably contain room labels as TEXT/MTEXT entities?** If labels are missing or inconsistent, FR-21 degrades to "auto-generated room IDs" rather than meaningful names.
10. **What is the up-axis convention in their scans?** Z-up is standard for most LiDAR formats but some workflows use Y-up. Affects floor detection and the wall-band slice computation. Confirm in week 1 with one real scan.
11. **What is the typical ceiling height in the buildings they scan?** Determines the upper bound of the wall band slice. Default is 1.8m above floor, but if they often scan retail or industrial spaces with much higher ceilings the slice band can be widened.
12. **How dense are wall fixtures in their typical buildings?** Office floors might have 30–60 fixtures (outlets, panels, thermostats, fire extinguishers). Industrial or medical floors could have 200+. Affects performance and visual clutter — may need fixture clustering or zoom-based level-of-detail in v1.5 if density is very high.

---

## 13. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Test data formats turn out to be DWG or proprietary | Medium | High | Confirm in kickoff. Budget 1–2 weeks if conversion toolchain needed. |
| Manhattan assumption fails on >10% of their buildings | Medium | Medium | MVP can punt to manual mode. v2 adds non-orthogonal alignment. |
| Snap animation harder than expected | Low | Medium | Pre-built tweening libraries (TWEEN.js); start frontend in week 1 to surface issues. |
| Client wants v2 features in MVP scope | High | High | This PRD explicitly enumerates v1 vs v2. Hold the line in the SOW. |
| Performance worse than 30s on large scans | Medium | Medium | Downsample to voxel grid before plane detection. PDAL streaming if needed. |
| AI review step adds variance perceived as unreliability | Low | Low | Hide AI score behind a toggle, default off for the demo if it misbehaves. |
| Room extraction harder than expected — DXFs use loose line segments instead of closed polylines | Medium | Medium | Inspect 2–3 real DXFs before starting week 4. If reconstruction is needed, use Shapely's `polygonize`. Worst case, slip week 4 by 2–3 days. |
| DXF lacks room labels — FR-21 degrades | Medium | Low | Fall back to auto-generated IDs (Room-01, Room-02). Note for client during demo. |
| Fixtures and wall surface aren't cleanly separable (noisy scans, scanner-close-to-wall reflections) | Medium | Medium | Tune protrusion threshold per scanner type. Conservative defaults: 2cm minimum protrusion. Allow technician to disable fixture detection per-job if it's noisy. |
| Floor plane detection fails on sloped floors (loading docks, ramps) | Low | Medium | Fallback: use multi-plane floor detection or assume floor at z=0 if no dominant horizontal plane found. Note in error log. |
| Wall band slice cuts off rooms with raised platforms or sunken areas | Low | Low | Configurable per-job slice band. Document this as a "tunable parameter" for the technician. |

---

## 14. Definition of Done

The MVP is "done" when:

1. A clean laptop with the repo cloned can run `make demo` (or equivalent) and have the app live on `localhost:5173` within 5 minutes.
2. The supplied test pair aligns correctly with the snap animation playing.
3. Alignment uses the wall band slice and is robust to typical scan clutter (desks, ceiling fixtures, etc.).
4. Rooms are extracted from the DXF, rendered with colored fills and labels, and respond to hover/click.
5. Per-room match-quality badges display sensible values.
6. Wall fixtures are detected and rendered as markers on the walls; hover shows depth and approximate size.
7. Fixture markers can be toggled on/off without recomputation.
8. Manual nudge mode works and updates the confidence score.
9. Export produces a valid `aligned.json` (including room polygons and fixture locations) and `aligned.dxf`.
10. AI review step toggles cleanly on and off without breaking the main flow.
11. The README documents setup, demo flow, and troubleshooting for at least the top 3 failure cases.
