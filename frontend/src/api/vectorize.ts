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
}

/** Layer ID — drives DXF layer name + on-canvas colour. */
export type SegmentLayer = 'walls' | 'openings' | 'columns' | 'mep' | string;

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
  has_raster: boolean;
  has_overlay: boolean;
  has_dxf: boolean;
  has_coverage: boolean;
  /** True when ``rejected_segments.json`` exists — editor enables the ghost layer. */
  has_rejected: boolean;
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

export interface SegmentsPayload {
  version: number;
  units: string;
  edit_version?: number;
  affine?: RasterAffine;
  raster?: { width_px: number; height_px: number; y_flipped_for_display: boolean };
  segments: EditableSegment[];
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
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new Error(`vectorize API ${res.status}: ${body}`);
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
