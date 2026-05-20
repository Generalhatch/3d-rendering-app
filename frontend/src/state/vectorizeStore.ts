/**
 * Zustand store for the Vectorize (scan-to-CAD) tab.
 *
 * Kept separate from ``jobStore`` so the two surfaces never accidentally
 * share state.  The alignment store has 50+ fields specific to its 3D
 * workflow; mixing them would hurt both.
 *
 * The active ``jobId`` is mirrored to ``localStorage`` so a refresh
 * resurrects the current job (the editor can rehydrate from the backend).
 * Heavy state — progress logs, the in-memory segment list — is intentionally
 * NOT persisted; that comes from the backend on rehydrate.
 */
import { create } from 'zustand';
import {
  type VectorizeJobDetail,
  type VectorizeParams,
  type VectorizeProgressEvent,
  DEFAULT_VECTORIZE_PARAMS,
} from '../api/vectorize';

const ACTIVE_JOB_KEY = 'alignai_vectorize_active_job';

function loadActiveJob(): string | null {
  try {
    return localStorage.getItem(ACTIVE_JOB_KEY);
  } catch {
    return null;
  }
}

function persistActiveJob(jobId: string | null): void {
  try {
    if (jobId === null) localStorage.removeItem(ACTIVE_JOB_KEY);
    else localStorage.setItem(ACTIVE_JOB_KEY, jobId);
  } catch {
    // localStorage unavailable
  }
}

export type VectorizePhase = 'idle' | 'uploading' | 'processing' | 'complete' | 'failed';

export interface VectorizeLogEntry {
  stage: string;
  message: string;
  progress: number;
  timestamp: number;
}

interface VectorizeState {
  phase: VectorizePhase;
  jobId: string | null;
  scanFile: File | null;
  params: VectorizeParams;
  job: VectorizeJobDetail | null;
  logs: VectorizeLogEntry[];
  overallProgress: number;
  /** Which overlay to show in the viewer. */
  overlayMode: 'clean' | 'raw' | 'raster';

  setPhase: (phase: VectorizePhase) => void;
  setJobId: (id: string | null) => void;
  setScanFile: (f: File | null) => void;
  setParams: (params: VectorizeParams) => void;
  patchParams: (patch: Partial<VectorizeParams>) => void;
  setJob: (job: VectorizeJobDetail | null) => void;
  appendLog: (event: VectorizeProgressEvent) => void;
  /** Update the progress bar without appending a log entry — used for upload bytes. */
  setOverallProgress: (progress: number) => void;
  clearLogs: () => void;
  setOverlayMode: (mode: 'clean' | 'raw' | 'raster') => void;
  reset: () => void;
}

const initialState = {
  // ``phase`` starts as 'idle' on a fresh page but flips to 'processing' or
  // 'complete' once we rehydrate the persisted jobId (handled by
  // ``VectorizePanel`` on mount).
  phase: 'idle' as VectorizePhase,
  jobId: loadActiveJob(),
  scanFile: null,
  params: DEFAULT_VECTORIZE_PARAMS,
  job: null,
  logs: [] as VectorizeLogEntry[],
  overallProgress: 0,
  overlayMode: 'clean' as const,
};

export const useVectorizeStore = create<VectorizeState>((set) => ({
  ...initialState,

  setPhase: (phase) => set({ phase }),
  setJobId: (jobId) => {
    persistActiveJob(jobId);
    set({ jobId });
  },
  setScanFile: (scanFile) => set({ scanFile }),
  setParams: (params) => set({ params }),
  patchParams: (patch) => set((s) => ({ params: { ...s.params, ...patch } })),
  setJob: (job) => set({ job }),
  appendLog: (event) =>
    set((s) => ({
      logs: [
        ...s.logs.slice(-99),
        {
          stage: event.stage,
          message: event.message,
          progress: event.progress,
          timestamp: Date.now(),
        },
      ],
      // Never let upload-byte progress get overwritten by a stale, lower
      // backend value if events arrive out of order; progress should be
      // monotonic.
      overallProgress: Math.max(s.overallProgress, event.progress),
    })),
  setOverallProgress: (overallProgress) =>
    set((s) => ({ overallProgress: Math.max(s.overallProgress, overallProgress) })),
  clearLogs: () => set({ logs: [], overallProgress: 0 }),
  setOverlayMode: (overlayMode) => set({ overlayMode }),
  reset: () => {
    persistActiveJob(null);
    set({ ...initialState, jobId: null });
  },
}));
