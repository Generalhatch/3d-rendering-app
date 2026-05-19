import { create } from 'zustand';
import type { JobDetail, Room, Fixture, AlignmentResult, ProgressEvent, AiReview, FullResult, FloorCandidate } from '../api/client';

export interface ProgressLog {
  stage: string;
  message: string;
  progress: number;
  timestamp: number;
}

export interface JobHistoryEntry {
  jobId: string;
  status: string;
  createdAt: string;
  scanCount: number;
  scanOnly: boolean;
  elapsed_s?: number | null;
}

const HISTORY_KEY = 'alignai_job_history';
const MAX_HISTORY = 10;

function loadHistory(): JobHistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '[]') as JobHistoryEntry[];
  } catch {
    return [];
  }
}

function saveHistory(entries: JobHistoryEntry[]): void {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
  } catch {
    // localStorage unavailable (private browsing, storage full, etc.)
  }
}

export type AppPhase =
  | 'idle'
  | 'uploading'
  | 'processing'
  | 'aligned'
  | 'approved'
  | 'failed';

interface JobState {
  phase: AppPhase;
  jobId: string | null;
  job: JobDetail | null;
  fullResult: FullResult | null;
  rooms: Room[];
  fixtures: Fixture[];
  selectedRoomId: string | null;
  selectedFixtureId: string | null;
  progressLogs: ProgressLog[];
  overallProgress: number;
  showFixtures: boolean;
  showRooms: boolean;
  scanOpacity: number;
  roomOpacity: number;
  aiReview: AiReview | null;
  aiReviewLoading: boolean;
  isManualMode: boolean;
  isScanOnly: boolean;
  jobHistory: JobHistoryEntry[];
  // MJ2: detected floor levels for the current job
  floorCandidates: FloorCandidate[];
  // MJ2/MJ3: true while a re-process is running
  isReprocessing: boolean;

  // Actions
  setPhase: (phase: AppPhase) => void;
  setJobId: (id: string) => void;
  setJob: (job: JobDetail) => void;
  setFullResult: (result: FullResult) => void;
  setRooms: (rooms: Room[]) => void;
  setFixtures: (fixtures: Fixture[]) => void;
  selectRoom: (id: string | null) => void;
  selectFixture: (id: string | null) => void;
  appendLog: (event: ProgressEvent) => void;
  setOverallProgress: (p: number) => void;
  toggleFixtures: () => void;
  toggleRooms: () => void;
  setScanOpacity: (v: number) => void;
  setRoomOpacity: (v: number) => void;
  setAiReview: (r: AiReview | null) => void;
  setAiReviewLoading: (loading: boolean) => void;
  setManualMode: (on: boolean) => void;
  setScanOnly: (v: boolean) => void;
  addJobToHistory: (entry: JobHistoryEntry) => void;
  setFloorCandidates: (floors: FloorCandidate[]) => void;
  setReprocessing: (v: boolean) => void;
  reset: () => void;
}

const initialState = {
  phase: 'idle' as AppPhase,
  jobId: null,
  job: null,
  fullResult: null,
  rooms: [],
  fixtures: [],
  selectedRoomId: null,
  selectedFixtureId: null,
  progressLogs: [],
  overallProgress: 0,
  showFixtures: true,
  showRooms: true,
  scanOpacity: 0.8,
  roomOpacity: 0.25,
  aiReview: null,
  aiReviewLoading: false,
  isManualMode: false,
  isScanOnly: false,
  jobHistory: loadHistory(),
  floorCandidates: [] as FloorCandidate[],
  isReprocessing: false,
};

export const useJobStore = create<JobState>((set) => ({
  ...initialState,

  setPhase: (phase) => set({ phase }),
  setJobId: (jobId) => set({ jobId }),
  setJob: (job) => set({ job }),
  setFullResult: (fullResult) => set({ fullResult }),
  setRooms: (rooms) => set({ rooms }),
  setFixtures: (fixtures) => set({ fixtures }),
  selectRoom: (selectedRoomId) => set({ selectedRoomId, selectedFixtureId: null }),
  selectFixture: (selectedFixtureId) => set({ selectedFixtureId, selectedRoomId: null }),
  appendLog: (event) =>
    set((state) => ({
      progressLogs: [
        ...state.progressLogs.slice(-99),
        { stage: event.stage, message: event.message, progress: event.progress, timestamp: Date.now() },
      ],
      overallProgress: event.progress,
    })),
  setOverallProgress: (overallProgress) => set({ overallProgress }),
  toggleFixtures: () => set((s) => ({ showFixtures: !s.showFixtures })),
  toggleRooms: () => set((s) => ({ showRooms: !s.showRooms })),
  setScanOpacity: (scanOpacity) => set({ scanOpacity }),
  setRoomOpacity: (roomOpacity) => set({ roomOpacity }),
  setAiReview: (aiReview) => set({ aiReview }),
  setAiReviewLoading: (aiReviewLoading) => set({ aiReviewLoading }),
  setManualMode: (isManualMode) => set({ isManualMode }),
  setScanOnly: (isScanOnly) => set({ isScanOnly }),
  addJobToHistory: (entry) =>
    set((state) => {
      const filtered = state.jobHistory.filter((h) => h.jobId !== entry.jobId);
      const updated = [entry, ...filtered].slice(0, MAX_HISTORY);
      saveHistory(updated);
      return { jobHistory: updated };
    }),
  setFloorCandidates: (floorCandidates) => set({ floorCandidates }),
  setReprocessing: (isReprocessing) => set({ isReprocessing }),
  reset: () => set({ ...initialState, jobHistory: loadHistory() }),
}));
