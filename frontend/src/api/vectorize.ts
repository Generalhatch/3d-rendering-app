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
  /** Absolute elevation (m) at which to slice; null = auto (floor + 1.4 m). */
  elevation_m: number | null;
  slab_thickness_m: number;
  resolution_m_per_px: number;
  detector: DetectorName;
  min_wall_length_m: number;
  manhattan_snap: boolean;
  merge_collinear: boolean;
}

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

export const DEFAULT_VECTORIZE_PARAMS: VectorizeParams = {
  elevation_m: null,
  slab_thickness_m: 0.20,
  resolution_m_per_px: 0.01,
  detector: 'fld',
  min_wall_length_m: 0.50,
  manhattan_snap: true,
  merge_collinear: true,
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
  create: async (scan: File, params: VectorizeParams): Promise<VectorizeJobCreate> => {
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
    return request<VectorizeJobCreate>('', { method: 'POST', body: fd });
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
  dxfUrl: (jobId: string) => `${BASE}/${jobId}/dxf`,

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
