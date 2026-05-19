/**
 * Zustand store for the Vectorize (scan-to-CAD) tab.
 *
 * Kept separate from ``jobStore`` so the two surfaces never accidentally
 * share state.  The alignment store has 50+ fields specific to its 3D
 * workflow; mixing them would hurt both.
 */
import { create } from 'zustand';
import {
  type VectorizeJobDetail,
  type VectorizeParams,
  type VectorizeProgressEvent,
  DEFAULT_VECTORIZE_PARAMS,
} from '../api/vectorize';

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
  clearLogs: () => void;
  setOverlayMode: (mode: 'clean' | 'raw' | 'raster') => void;
  reset: () => void;
}

const initialState = {
  phase: 'idle' as VectorizePhase,
  jobId: null,
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
  setJobId: (jobId) => set({ jobId }),
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
      overallProgress: event.progress,
    })),
  clearLogs: () => set({ logs: [], overallProgress: 0 }),
  setOverlayMode: (overlayMode) => set({ overlayMode }),
  reset: () => set({ ...initialState }),
}));
