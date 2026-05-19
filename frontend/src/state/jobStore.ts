import { create } from 'zustand';
import type { JobDetail, Room, Fixture, AlignmentResult, ProgressEvent, AiReview, FullResult } from '../api/client';

export interface ProgressLog {
  stage: string;
  message: string;
  progress: number;
  timestamp: number;
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
        ...state.progressLogs,
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
  reset: () => set(initialState),
}));
