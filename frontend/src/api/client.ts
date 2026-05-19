/**
 * Typed API client — all backend calls go through here.
 */

const BASE = '/api';

export interface JobCreate {
  job_id: string;
  status: string;
}

export interface AlignmentResult {
  transformation: number[][];
  residual_rmse: number;
  residual_rmse_mm: number;
  confidence: number;
  confidence_pct: number;
  inlier_ratio: number;
  rotation_candidate_used: number;
  iterations: number;
  num_wall_planes: number;
  mode?: 'aligned' | 'scan_only';
}

export interface Room {
  id: string;
  label: string;
  category: 'office' | 'bathroom' | 'hallway' | 'common' | 'unknown';
  polygon_2d: [number, number][];
  centroid: [number, number];
  area_m2: number;
  match_quality: number | null;
  // Shape analysis from watershed regionprops
  eccentricity?: number;       // 0=square, 1=line (>0.8 suggests corridor)
  orientation_deg?: number;    // major axis angle
  solidity?: number;           // 1=convex, <0.7=highly irregular
  // MJ1: classifier confidence — 0.30 (weak) to 0.99 (strong match)
  classification_confidence?: number;
}

/** One detected floor level in the scan (MJ2 multi-floor support). */
export interface FloorCandidate {
  floor_z: number;        // elevation in scan's local coordinate frame (m)
  inlier_count: number;   // RANSAC inlier count
  wall_score: number;     // points 0.3–3 m above — higher = indoor floor with walls
  axis_idx: number;       // 2 = Z-up, 1 = Y-up
}

/** Body for POST /api/jobs/:id/reprocess (MJ2 + MJ3). */
export interface ReprocessParams {
  floor_z?: number | null;        // null = auto-detect
  band_low_m?: number;            // default 0.75
  band_high_m?: number;           // default 1.80
  min_wall_length_m?: number;     // default 2.0
  hough_threshold?: number;       // default 35
}

export interface Fixture {
  id: string;
  wall_id: string;
  centroid: [number, number, number];
  bbox_min: [number, number, number];
  bbox_max: [number, number, number];
  protrusion_depth_m: number;
  width_m: number;
  height_m: number;
  point_count: number;
  confidence: number;
}

export interface JobDetail {
  job_id: string;
  status: 'queued' | 'processing' | 'aligned' | 'failed' | 'approved';
  created_at: string;
  updated_at: string;
  scan_filename: string;
  plan_filename: string;
  scan_only?: boolean;
  error_message?: string | null;
  result?: AlignmentResult | null;
  plan_bounds?: [number, number, number, number] | null;
  num_rooms?: number | null;
  num_fixtures?: number | null;
  elapsed_s?: number | null;
  floor_z?: number;
  floor_candidates?: FloorCandidate[];   // MJ2: all detected levels
}

export interface FullResult {
  job_id: string;
  alignment: AlignmentResult & { mode?: 'aligned' | 'scan_only' };
  floor_z: number;
  scan_only?: boolean;
  rooms: Room[];
  fixtures: Fixture[];
  plan_bounds: [number, number, number, number];
  overlay_png?: string;
  elapsed_s?: number;
  num_scans?: number;
  merge_strategy?: string;
  // MJ2: multiple floor levels detected in the scan
  floor_candidates?: FloorCandidate[];
}

export interface ProgressEvent {
  stage: string;
  message: string;
  progress: number;
  job_id: string;
}

export interface AiReview {
  summary: string;
  concerns: string[];
  model_confidence: number | null;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  createJob: async (
    scanFiles: File | File[],
    planFile?: File | null,
    bandLow = 0.75,
    bandHigh = 1.80,
  ): Promise<JobCreate> => {
    const fd = new FormData();
    if (planFile) fd.append('plan', planFile);
    const scansArray = Array.isArray(scanFiles) ? scanFiles : [scanFiles];
    for (const scan of scansArray) fd.append('scans', scan);
    fd.append('band_low_m', String(bandLow));
    fd.append('band_high_m', String(bandHigh));
    return request<JobCreate>('/jobs', { method: 'POST', body: fd });
  },

  getJob: (jobId: string) => request<JobDetail>(`/jobs/${jobId}`),

  getRooms: (jobId: string) => request<{ rooms: Room[] }>(`/jobs/${jobId}/rooms`),

  getFixtures: (jobId: string) => request<{ fixtures: Fixture[] }>(`/jobs/${jobId}/fixtures`),

  getPlanGeoJSON: (jobId: string) => request<{ type: string; features: unknown[] }>(`/jobs/${jobId}/plan`),

  getOutlineGeoJSON: (jobId: string) =>
    request<{ type: string; features: Array<{ geometry: { type: string; coordinates: number[][][] } }> }>(
      `/jobs/${jobId}/outline`,
    ),

  getScanUrl: (jobId: string) => `${BASE}/jobs/${jobId}/scan`,

  requestAiReview: (jobId: string) => request<{ ai_review: AiReview }>(`/jobs/${jobId}/ai`, { method: 'POST' }),

  approveJob: (jobId: string) => request<{ status: string }>(`/jobs/${jobId}/approve`, { method: 'POST' }),

  applyManualTransform: (jobId: string, transformation: number[][]) =>
    request<{ transformation: number[][]; confidence: number; confidence_pct: number; residual_rmse_mm: number }>(
      `/jobs/${jobId}/manual`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ transformation }) },
    ),

  exportJsonUrl: (jobId: string) => `${BASE}/jobs/${jobId}/export/json`,
  exportDxfUrl: (jobId: string) => `${BASE}/jobs/${jobId}/export/dxf`,
  getOverlayUrl: (jobId: string) => `${BASE}/jobs/${jobId}/overlay`,

  /** MJ2: list all detected floor levels for this job. */
  getFloors: (jobId: string) => request<FloorCandidate[]>(`/jobs/${jobId}/floors`),

  /** MJ2/MJ3: trigger re-processing with new parameters. */
  reprocessJob: (jobId: string, params: ReprocessParams) =>
    request<{ job_id: string; status: string }>(
      `/jobs/${jobId}/reprocess`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      },
    ),

  subscribeProgress: (
    jobId: string,
    onEvent: (event: ProgressEvent) => void,
    onError?: (err: Event) => void,
  ): { close: () => void } => {
    let es: EventSource | null = null;
    let closed = false;
    let retryDelay = 1000;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (closed) return;
      es = new EventSource(`${BASE}/jobs/${jobId}/sse`);

      es.onmessage = (e) => {
        retryDelay = 1000; // reset backoff on successful message
        try {
          onEvent(JSON.parse(e.data) as ProgressEvent);
        } catch {
          // ignore parse errors
        }
      };

      es.onerror = (err) => {
        if (closed) return;
        es?.close();
        es = null;
        if (onError) onError(err);
        // Exponential backoff: 1s → 2s → 4s → 8s → 16s → 30s max
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
