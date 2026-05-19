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
}

export interface Room {
  id: string;
  label: string;
  category: 'office' | 'bathroom' | 'hallway' | 'common' | 'unknown';
  polygon_2d: [number, number][];
  centroid: [number, number];
  area_m2: number;
  match_quality: number | null;
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
  error_message?: string | null;
  result?: AlignmentResult | null;
  num_rooms?: number | null;
  num_fixtures?: number | null;
  elapsed_s?: number | null;
}

export interface FullResult {
  job_id: string;
  alignment: AlignmentResult;
  floor_z: number;
  rooms: Room[];
  fixtures: Fixture[];
  plan_bounds: [number, number, number, number];
  overlay_png?: string;
  elapsed_s?: number;
  num_scans?: number;
  merge_strategy?: string;
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
  createJob: async (planFile: File, scanFiles: File | File[], bandLow = 0.75, bandHigh = 1.80): Promise<JobCreate> => {
    const fd = new FormData();
    fd.append('plan', planFile);
    const scansArray = Array.isArray(scanFiles) ? scanFiles : [scanFiles];
    for (const scan of scansArray) {
      fd.append('scans', scan);
    }
    fd.append('band_low_m', String(bandLow));
    fd.append('band_high_m', String(bandHigh));
    return request<JobCreate>('/jobs', { method: 'POST', body: fd });
  },

  getJob: (jobId: string) => request<JobDetail>(`/jobs/${jobId}`),

  getRooms: (jobId: string) => request<{ rooms: Room[] }>(`/jobs/${jobId}/rooms`),

  getFixtures: (jobId: string) => request<{ fixtures: Fixture[] }>(`/jobs/${jobId}/fixtures`),

  getPlanGeoJSON: (jobId: string) => request<{ type: string; features: unknown[] }>(`/jobs/${jobId}/plan`),

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

  subscribeProgress: (
    jobId: string,
    onEvent: (event: ProgressEvent) => void,
    onError?: (err: Event) => void,
  ): EventSource => {
    const es = new EventSource(`${BASE}/jobs/${jobId}/sse`);
    es.onmessage = (e) => {
      try {
        onEvent(JSON.parse(e.data) as ProgressEvent);
      } catch {
        // ignore parse errors
      }
    };
    if (onError) es.onerror = onError;
    return es;
  },
};
