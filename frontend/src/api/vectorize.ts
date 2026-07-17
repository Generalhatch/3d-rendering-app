/**
 * Typed client for the /api/vectorize/* surface (scan-to-CAD pipeline).
 *
 * Kept separate from ``api/client.ts`` (alignment) so the two surfaces can
 * evolve independently and the typings don't leak across.
 */

const BASE = '/api/vectorize';

export type VectorizeStatus = 'queued' | 'processing' | 'complete' | 'failed';
export type DetectorName = 'hough' | 'fld' | 'both';

export interface VectorizeParams {
  /** Absolute elevation (m) at which to slice; null = auto (floor + 1.6 m). */
  elevation_m: number | null;
  slab_thickness_m: number;
  resolution_m_per_px: number;
  detector: DetectorName;
  min_wall_length_m: number;
  manhattan_snap: boolean;
  merge_collinear: boolean;
  /** Apply a 3-px morphological OPEN before detection (kills speckle). */
  remove_speckle: boolean;
  /** OR-fuse 3 slabs (elev ± 0.4 m) into the raster for better recall. */
  multi_elevation: boolean;
  /** Detect door-shaped gaps in each wall and emit them on an OPENINGS layer. */
  detect_openings: boolean;
  /** Detect column-shaped blobs in the raster and emit them on a COLUMNS layer. */
  detect_columns: boolean;
  /**
   * How far a free wall end may extend / trim to meet a host or L-partner
   * (Cloud2BIM § 2.8 / Phase 3 corner finish).  Metres.
   */
  snapping_distance_m: number;
}

/** Layer ID — drives DXF layer name + on-canvas colour. */
export type SegmentLayer =
  | 'walls'
  | 'walls_exterior'
  | 'walls_faces'
  | 'rooms'
  | 'openings'
  | 'windows'
  | 'columns'
  | 'mep'
  | string;

export interface VectorizeMetrics {
  elapsed_s: number;
  raw_points: number;
  points_in_slab: number;
  raster_width_px: number;
  raster_height_px: number;
  raster_world_width_m: number;
  raster_world_height_m: number;
  segments_detected: number;
  segments_after_regularize: number;
  elevation_m: number;
  detector: string;
  /** Fraction [0, 1] of raster foreground within COVERAGE_RADIUS_M of a segment.
   *  Null on jobs that completed before the diagnostic was added. */
  coverage_pct: number | null;
  coverage_radius_m: number | null;
  foreground_px: number | null;
  uncovered_px: number | null;
  /** All elevations OR-fused into the raster.  Length 1 = single-slice, 3 = multi. */
  elevations_used_m: number[] | null;
  /** Doors / wall-gaps detected by Phase 4 opening detector.  Null on legacy jobs. */
  openings_detected: number | null;
  /** Structural columns detected by Phase 6 column detector.  Null on legacy jobs. */
  columns_detected: number | null;
  /**
   * Per-layer foreground-coverage percentage (0..1).  ``walls`` always present;
   * ``openings`` / ``columns`` present when their respective detectors ran.
   */
  coverage_by_layer: Record<string, number> | null;
  /** Phase 4: True iff the cloud was rotated so the floor plane is level. */
  gravity_relevel_applied: boolean | null;
  /** Detected floor-plane tilt (degrees).  Null on legacy jobs. */
  gravity_tilt_deg: number | null;
  /** Floor-to-ceiling clearance measured from the height histogram (m). */
  ceiling_clearance_m: number | null;
  /** True iff the slice bands were scaled down for a low ceiling. */
  slice_band_adjusted: boolean | null;
  /** Number of rooms flagged below the confidence threshold. */
  rooms_flagged: number | null;
  /** Rooms inferred by topology (0 = not closed — use Generate BOMA). */
  rooms_detected: number | null;
  walls_paired: number | null;
  walls_unpaired: number | null;
  walls_median_thickness_m: number | null;
  junctions_merged: number | null;
  t_junctions_extended: number | null;
  /** Phase 2: collinear degree-1 gaps bridged during continuity. */
  gaps_closed: number | null;
  /** Phase 2: dangling ends projected onto the building envelope. */
  envelope_projections: number | null;
  /** Phase 2: degree-1 junction count before continuity. */
  dangling_before: number | null;
  /** Phase 2: degree-1 junction count after continuity. */
  dangling_after: number | null;
  /** Phase 2: envelope ring edges injected into the topology graph. */
  envelope_edges_injected: number | null;
  points_after_downsample: number | null;
  downsample_voxel_m: number | null;
}

/** One structured pipeline warning — every silent fallback surfaces here.
 *  Codes are stable identifiers (see backend pipeline_warnings.KNOWN_CODES). */
export interface PipelineWarning {
  code: string;
  stage: string;
  message: string;
}

/** Per-room trust score (Phase 4).  Presentation-only — geometry and areas
 *  are computed elsewhere and never altered by confidence. */
export interface RoomConfidence {
  room_index: number;
  area_m2: number;
  /** Fraction [0, 1] of the room boundary supported by observed raster. */
  boundary_coverage: number;
  /** Mean distance (m) room vertices moved during topology snapping. */
  snap_correction_m: number;
  /** Combined trust score [0, 1]. */
  confidence: number;
  /** True = below threshold; the operator should review this room. */
  flagged: boolean;
  /** Room polygon (world metres) for hover-highlighting in the editor. */
  polygon: Array<[number, number]>;
}

export interface VectorizeJobDetail {
  job_id: string;
  status: VectorizeStatus;
  created_at: string;
  updated_at: string;
  scan_filename: string;
  params: VectorizeParams | null;
  metrics: VectorizeMetrics | null;
  error_message: string | null;
  /** Structured fallback warnings from the pipeline run (empty on clean runs). */
  warnings: PipelineWarning[];
  /** Per-room confidence scores; empty when no rooms were inferred. */
  room_confidence: RoomConfidence[];
  has_raster: boolean;
  has_overlay: boolean;
  has_dxf: boolean;
  has_coverage: boolean;
  /** True when ``rejected_segments.json`` exists — editor enables the ghost layer. */
  has_rejected: boolean;
  /** True when ``sheet.svg`` exists — the Phase 3 floor plan sheet. */
  has_sheet: boolean;
  /** True when ``floor_geometry.json`` exists — sheet can be (re)rendered. */
  has_floor_geometry: boolean;
  /** True when ``measurement_report.json`` exists (BOMA/REBNY/Gross). */
  has_measurement_report: boolean;
}

export interface VectorizeJobCreate {
  job_id: string;
  status: VectorizeStatus;
}

export interface VectorizeProgressEvent {
  stage: string;
  message: string;
  progress: number;
  job_id: string;
}

// ── Editor (Phase 3) ────────────────────────────────────────────────────────

export interface EditableSegment {
  id: string;
  /** Logical class — controls DXF layer, render colour, and operator filters. */
  layer: SegmentLayer;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RasterAffine {
  origin_x: number;
  origin_y: number;
  resolution_m_per_px: number;
  width_px: number;
  height_px: number;
}

/** Degree-1 topology junction — an open wall end the operator should close. */
export interface DanglingEndpoint {
  x: number;
  y: number;
  degree: number;
}

export interface SegmentsPayload {
  version: number;
  units: string;
  edit_version?: number;
  affine?: RasterAffine;
  raster?: { width_px: number; height_px: number; y_flipped_for_display: boolean };
  segments: EditableSegment[];
  /** Open wall ends from the last topology pass (empty when rooms closed). */
  dangling_endpoints?: DanglingEndpoint[];
}

/** Structured 422 body from Generate BOMA when rooms still aren't closed. */
export interface RoomsNotClosedDetail {
  message: string;
  n_walls?: number;
  n_junctions?: number;
  n_edges?: number;
  dangling_endpoints?: DanglingEndpoint[];
}

export class VectorizeApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, body: string) {
    let detail: unknown = body;
    let message = `vectorize API ${status}: ${body}`;
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (parsed && typeof parsed === 'object' && 'detail' in parsed) {
        detail = parsed.detail;
        if (detail && typeof detail === 'object' && detail !== null && 'message' in detail) {
          message = String((detail as { message: unknown }).message);
        } else if (typeof detail === 'string') {
          message = detail;
        }
      }
    } catch {
      // keep raw body message
    }
    super(message);
    this.name = 'VectorizeApiError';
    this.status = status;
    this.detail = detail;
  }
}

export type EditOp =
  | 'reject'
  | 'restore'
  | 'move_endpoint'
  | 'snap_manhattan'
  | 'delete'
  | 'add'
  | 'merge'
  | 'promote_ghost';

export interface EditEvent {
  op: EditOp;
  segment_id?: string;
  segment_ids?: string[];
  new_id?: string;
  which_endpoint?: 0 | 1;
  from_xy?: [number, number];
  to_xy?: [number, number];
  ts_ms?: number;
}

export interface SaveEditsResponse {
  job_id: string;
  edit_version: number;
  dxf_url: string;
  segments_saved: number;
}

export interface GenerateMeasurementRequest {
  segments?: EditableSegment[];
  log?: EditEvent[];
  save_first?: boolean;
}

export interface GenerateMeasurementResponse {
  job_id: string;
  rooms_detected: number;
  has_floor_geometry: boolean;
  has_measurement_report: boolean;
  has_sheet: boolean;
  warnings: PipelineWarning[];
  n_walls: number;
  n_junctions: number;
  edit_version: number | null;
}

/** A segment the pipeline's regularizer dropped — rescuable as a ghost in the editor. */
export interface RejectedSegmentPayload {
  id: string;
  layer: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Why the pipeline dropped it. */
  dropped_by: 'short' | 'manhattan' | 'merged' | string;
  length_m: number;
  angle_deg: number;
}

export interface RejectedSegmentsPayload {
  version: number;
  units: string;
  rejected: RejectedSegmentPayload[];
}

/** One pair of segments the duplicate-finder suggests merging together. */
export interface DuplicatePair {
  a_id: string;
  b_id: string;
  perp_distance_m: number;
  angle_diff_deg: number;
  endpoint_gap_m: number;
  merged_x1: number;
  merged_y1: number;
  merged_x2: number;
  merged_y2: number;
}

export interface FindDuplicatesResponse {
  suggestions: DuplicatePair[];
}

export interface FindDuplicatesRequest {
  /** Max perpendicular distance between centres of two parallel segments. */
  perp_distance_m?: number;
  /** Max difference in angle (deg) for two segments to be considered parallel. */
  parallel_tol_deg?: number;
  /** Max collinear gap between the two segments' projected ranges. */
  endpoint_gap_m?: number;
}

// ── Floor plan sheet (Phase 3) ──────────────────────────────────────────────

/** One room from the canonical floor geometry — the sheet editor's data. */
export interface FloorGeometryRoom {
  id: string;
  label: string;
  category: string;
  suite_id: string | null;
  is_common: boolean;
  area_m2: number;
}

export interface FloorGeometryPayload {
  version: number;
  units: string;
  floor_id: string;
  source: string;
  building_name: string;
  floor_name: string;
  rooms: FloorGeometryRoom[];
  columns: unknown[];
}

export interface ManualGridSpec {
  rotation_deg: number;
  u_offsets_m: number[];
  v_offsets_m: number[];
}

export interface SheetMetaSpec {
  building_name: string;
  address: string;
  floor_name: string;
  north_angle_deg: number;
}

/** Per-opening door swing override ("left" = +90° CCW of the opening
 *  segment direction). */
export interface DoorSwingSpec {
  hinge: 'start' | 'end';
  side: 'left' | 'right';
}

export type SheetStyle = 'full' | 'stevenson_minimal';

/** All fields optional; labels/door_swings merge per-key,
 *  meta/manual_grid/style replace. */
export interface SheetOverrides {
  labels?: Record<string, string>;
  meta?: SheetMetaSpec | null;
  manual_grid?: ManualGridSpec | null;
  style?: SheetStyle | null;
  door_swings?: Record<string, DoorSwingSpec> | null;
}

export interface SheetOverridesResponse {
  job_id: string;
  sheet_url: string;
  scale_denominator: number;
}

/** One suite row from the measurement report (server-computed areas). */
export interface AreaValuePayload {
  value_m2: number;
  value_sqft: number;
  standard: string;
  method: string;
  component: string;
}

export interface MeasurementSuiteRow {
  suite_id: string;
  label: string;
  room_ids: string[];
  usable: AreaValuePayload;
  rentable: AreaValuePayload;
}

export interface MeasurementStandardBlock {
  standard: string;
  method: string;
  floor_id: string;
  load_factor: number;
  floor_usable: AreaValuePayload;
  floor_rentable: AreaValuePayload;
  suites: MeasurementSuiteRow[];
  notes: string[];
}

export interface MeasurementReportPayload {
  version: number;
  generated_at: string;
  floor_id: string;
  building_name: string;
  floor_name: string;
  units: { area: string; secondary: string };
  measurements: MeasurementStandardBlock[];
}

/**
 * Tuned to handle a typical interior LiDAR scan at ~1 cm point spacing
 * without operator intervention.  Keep in sync with
 * ``backend/app/models/vectorize_job.py::VectorizeParams``.
 */
export const DEFAULT_VECTORIZE_PARAMS: VectorizeParams = {
  elevation_m: null,            // auto = floor_z + 1.6 m
  slab_thickness_m: 0.10,
  resolution_m_per_px: 0.01,
  detector: 'both',
  min_wall_length_m: 0.40,
  manhattan_snap: true,
  merge_collinear: true,
  remove_speckle: true,
  multi_elevation: true,
  detect_openings: true,
  detect_columns: true,
  // Phase 3 corner join — 0.85 m closes typical undershoot without bridging
  // most door leaves.  Keep in sync with backend VectorizeParams default.
  snapping_distance_m: 0.85,
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new VectorizeApiError(res.status, body);
  }
  return res.json() as Promise<T>;
}

export const vectorizeApi = {
  /**
   * Upload a scan and start a vectorize job.
   *
   * Uses ``XMLHttpRequest`` (instead of ``fetch``) so the caller can observe
   * upload-byte progress — large LiDAR scans take long enough to upload that
   * a frozen-at-0% progress bar feels broken.  ``onUploadProgress`` is invoked
   * with a fraction in ``[0, 1]`` and a final ``1.0`` once bytes finish
   * streaming (before the server response).
   */
  create: (
    scan: File,
    params: VectorizeParams,
    onUploadProgress?: (fraction: number) => void,
  ): Promise<VectorizeJobCreate> => {
    const fd = new FormData();
    fd.append('scan', scan);
    if (params.elevation_m !== null) {
      fd.append('elevation_m', String(params.elevation_m));
    }
    fd.append('slab_thickness_m', String(params.slab_thickness_m));
    fd.append('resolution_m_per_px', String(params.resolution_m_per_px));
    fd.append('detector', params.detector);
    fd.append('min_wall_length_m', String(params.min_wall_length_m));
    fd.append('manhattan_snap', String(params.manhattan_snap));
    fd.append('merge_collinear', String(params.merge_collinear));
    fd.append('remove_speckle', String(params.remove_speckle));
    fd.append('multi_elevation', String(params.multi_elevation));
    fd.append('detect_openings', String(params.detect_openings));
    fd.append('detect_columns', String(params.detect_columns));
    fd.append('snapping_distance_m', String(params.snapping_distance_m));

    return new Promise<VectorizeJobCreate>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', BASE);
      xhr.responseType = 'text';

      if (onUploadProgress) {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && e.total > 0) {
            onUploadProgress(Math.min(1, e.loaded / e.total));
          }
        };
        xhr.upload.onload = () => onUploadProgress(1);
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText) as VectorizeJobCreate);
          } catch (e) {
            reject(new Error(`vectorize API: bad JSON response (${(e as Error).message})`));
          }
        } else {
          reject(new Error(`vectorize API ${xhr.status}: ${xhr.responseText || xhr.statusText}`));
        }
      };
      xhr.onerror = () => reject(new Error('vectorize API: network error during upload'));
      xhr.onabort = () => reject(new Error('vectorize API: upload aborted'));

      xhr.send(fd);
    });
  },

  get: (jobId: string) => request<VectorizeJobDetail>(`/${jobId}`),

  reprocess: (jobId: string, params: VectorizeParams) =>
    request<VectorizeJobCreate>(`/${jobId}/reprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params }),
    }),

  list: () => request<{ jobs: Array<{ job_id: string; status: VectorizeStatus; created_at: string; scan_filename: string }> }>(''),

  rasterUrl: (jobId: string) => `${BASE}/${jobId}/raster`,
  sliceUrl: (jobId: string) => `${BASE}/${jobId}/slice`,
  overlayUrl: (jobId: string) => `${BASE}/${jobId}/overlay`,
  overlayRawUrl: (jobId: string) => `${BASE}/${jobId}/overlay/raw`,
  coverageUrl: (jobId: string) => `${BASE}/${jobId}/coverage`,
  dxfUrl: (jobId: string) => `${BASE}/${jobId}/dxf`,
  scanUrl: (jobId: string) => `${BASE}/${jobId}/scan`,
  floorGeometryDownloadUrl: (jobId: string) => `${BASE}/${jobId}/floor-geometry.json`,
  sheetUrl: (jobId: string) => `${BASE}/${jobId}/sheet`,
  sheetPdfUrl: (jobId: string) => `${BASE}/${jobId}/sheet.pdf`,
  measurementReportUrl: (jobId: string) => `${BASE}/${jobId}/measurement-report`,
  measurementReportPdfUrl: (jobId: string) => `${BASE}/${jobId}/measurement-report.pdf`,

  /** Fetch sheet SVG as text for inline rendering (crisp at any zoom). */
  fetchSheetSvg: async (jobId: string, version = 0): Promise<string> => {
    const res = await fetch(`${BASE}/${jobId}/sheet?v=${version}`);
    if (!res.ok) {
      const body = await res.text().catch(() => res.statusText);
      throw new Error(`vectorize API ${res.status}: ${body}`);
    }
    return res.text();
  },

  /** Canonical floor geometry (rooms + suites) — drives the sheet editor. */
  getFloorGeometry: (jobId: string) =>
    request<FloorGeometryPayload>(`/${jobId}/floor-geometry`),

  getSheetOverrides: (jobId: string) =>
    request<SheetOverrides>(`/${jobId}/sheet/overrides`),

  getMeasurementReport: (jobId: string) =>
    request<MeasurementReportPayload>(`/${jobId}/measurement-report`),

  /** Persist sheet overrides (suite labels / title block / manual grid) and
   *  re-render sheet.svg + sheet.pdf server-side. */
  saveSheetOverrides: (jobId: string, overrides: SheetOverrides) =>
    request<SheetOverridesResponse>(`/${jobId}/sheet/overrides`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(overrides),
    }),

  getSegments: (jobId: string) => request<SegmentsPayload>(`/${jobId}/segments`),

  /** Pipeline-rejected ghost candidates (segments the regularizer dropped). */
  getRejectedSegments: (jobId: string) =>
    request<RejectedSegmentsPayload>(`/${jobId}/rejected-segments`),

  /** Ask the backend to scan the current segment list for near-duplicate pairs. */
  findDuplicates: (jobId: string, body?: FindDuplicatesRequest) =>
    request<FindDuplicatesResponse>(`/${jobId}/find-duplicates`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    }),

  saveEdits: (jobId: string, segments: EditableSegment[], log: EditEvent[]) =>
    request<SaveEditsResponse>(`/${jobId}/edits`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segments, log }),
    }),

  /** Rebuild floor geometry + BOMA from the current (edited) wall network. */
  generateMeasurement: (jobId: string, body?: GenerateMeasurementRequest) =>
    request<GenerateMeasurementResponse>(`/${jobId}/generate-measurement`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    }),

  subscribeProgress: (
    jobId: string,
    onEvent: (event: VectorizeProgressEvent) => void,
    onError?: (err: Event) => void,
  ): { close: () => void } => {
    let es: EventSource | null = null;
    let closed = false;
    let retryDelay = 1000;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (closed) return;
      es = new EventSource(`${BASE}/${jobId}/sse`);

      es.onmessage = (e) => {
        retryDelay = 1000;
        try {
          onEvent(JSON.parse(e.data) as VectorizeProgressEvent);
        } catch {
          // ignore
        }
      };

      es.onerror = (err) => {
        if (closed) return;
        es?.close();
        es = null;
        onError?.(err);
        retryTimer = setTimeout(() => {
          retryDelay = Math.min(retryDelay * 2, 30_000);
          connect();
        }, retryDelay);
      };
    };

    connect();
    return {
      close: () => {
        closed = true;
        if (retryTimer !== null) clearTimeout(retryTimer);
        es?.close();
        es = null;
      },
    };
  },
};
