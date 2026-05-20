/**
 * Zustand store for the Vectorize editor (Phase 3).
 *
 * Holds the live segment list, selection set, and a snapshot-based undo/redo
 * stack.  Also accumulates an append-only edit log so the eventual "Save"
 * action can hand the backend the *what the operator actually did* trail —
 * that log becomes Phase 4 training signal.
 *
 * Design choices
 * --------------
 * - **Snapshot history, not command pattern.**  For a few hundred segments and
 *   relatively coarse operations (delete N, move endpoint, etc.) snapshotting
 *   the segment list is bytes-cheap and code-cheap.  We trim to ``MAX_HISTORY``
 *   to bound memory.
 * - **`status: 'active' | 'rejected'`** instead of physically deleting.  Lets
 *   us "undo a delete" trivially and persists the operator's full intent to
 *   the backend log.
 * - **All coordinates in world metres.**  The editor canvas does the pixel
 *   conversion via the affine from the job result.
 */
import { create } from 'zustand';
import type { EditEvent, EditableSegment } from '../api/vectorize';

export type SegmentStatus = 'active' | 'rejected';

export interface EditorSegment extends EditableSegment {
  status: SegmentStatus;
}

const MAX_HISTORY = 100;

interface HistorySnapshot {
  segments: EditorSegment[];
  selection: string[];        // IDs
}

interface EditorState {
  /** ID of the vectorize job this editor is currently bound to. */
  jobId: string | null;
  /** Whether segments have been loaded for ``jobId``. */
  loaded: boolean;
  /** Whether the operator has unsaved changes since last save / load. */
  dirty: boolean;
  /** Saved status from the last successful save. */
  lastSavedAt: number | null;
  lastSavedEditVersion: number | null;

  segments: EditorSegment[];
  selectedIds: Set<string>;
  hoverId: string | null;
  /** Append-only operator action log — sent with the save POST. */
  editLog: EditEvent[];

  history: HistorySnapshot[];
  future: HistorySnapshot[];

  // ── Lifecycle ──────────────────────────────────────────────────────────────
  bindJob: (jobId: string) => void;
  loadSegments: (segments: EditableSegment[]) => void;
  reset: () => void;

  // ── Selection ──────────────────────────────────────────────────────────────
  selectOnly: (id: string) => void;
  toggleSelected: (id: string) => void;
  selectAll: () => void;
  clearSelection: () => void;
  setHover: (id: string | null) => void;

  // ── Edits ──────────────────────────────────────────────────────────────────
  rejectSelected: () => void;
  restoreSelected: () => void;
  deleteSelected: () => void;        // permanent delete (also rejected for history)
  moveEndpoint: (id: string, which: 0 | 1, to: [number, number]) => void;
  snapSelectedToManhattan: () => void;

  // ── History ────────────────────────────────────────────────────────────────
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  // ── Save bookkeeping ───────────────────────────────────────────────────────
  markSaved: (editVersion: number) => void;
}

const initialState = {
  jobId: null as string | null,
  loaded: false,
  dirty: false,
  lastSavedAt: null as number | null,
  lastSavedEditVersion: null as number | null,
  segments: [] as EditorSegment[],
  selectedIds: new Set<string>(),
  hoverId: null as string | null,
  editLog: [] as EditEvent[],
  history: [] as HistorySnapshot[],
  future: [] as HistorySnapshot[],
};

function snapshot(state: Pick<EditorState, 'segments' | 'selectedIds'>): HistorySnapshot {
  return {
    // Shallow copy of each segment is fine — they're plain data objects we
    // never mutate in place (every editor action produces a new array).
    segments: state.segments.map((s) => ({ ...s })),
    selection: Array.from(state.selectedIds),
  };
}

function pushHistory<T extends Pick<EditorState, 'segments' | 'selectedIds' | 'history' | 'future'>>(
  state: T,
): Pick<EditorState, 'history' | 'future'> {
  const next = [...state.history, snapshot(state)].slice(-MAX_HISTORY);
  return { history: next, future: [] };
}

export const useEditorStore = create<EditorState>((set, get) => ({
  ...initialState,

  bindJob: (jobId) => {
    set({
      ...initialState,
      jobId,
      // Re-create the Set so we don't share mutable state with other resets.
      selectedIds: new Set<string>(),
    });
  },

  loadSegments: (segments) => {
    set({
      segments: segments.map((s) => ({ ...s, status: 'active' as const })),
      selectedIds: new Set<string>(),
      hoverId: null,
      editLog: [],
      history: [],
      future: [],
      dirty: false,
      loaded: true,
    });
  },

  reset: () =>
    set({ ...initialState, selectedIds: new Set<string>() }),

  selectOnly: (id) =>
    set({ selectedIds: new Set([id]) }),

  toggleSelected: (id) =>
    set((s) => {
      const next = new Set(s.selectedIds);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { selectedIds: next };
    }),

  selectAll: () =>
    set((s) => ({
      selectedIds: new Set(s.segments.filter((x) => x.status === 'active').map((x) => x.id)),
    })),

  clearSelection: () =>
    set({ selectedIds: new Set<string>() }),

  setHover: (hoverId) => set({ hoverId }),

  rejectSelected: () =>
    set((s) => {
      if (s.selectedIds.size === 0) return {};
      const ids = Array.from(s.selectedIds);
      const idSet = s.selectedIds;
      const hist = pushHistory(s);
      const segments = s.segments.map((seg) =>
        idSet.has(seg.id) && seg.status === 'active'
          ? { ...seg, status: 'rejected' as const }
          : seg,
      );
      const log = [
        ...s.editLog,
        ...ids.map(
          (id): EditEvent => ({ op: 'reject', segment_id: id, ts_ms: Date.now() }),
        ),
      ];
      return {
        segments,
        selectedIds: new Set<string>(),
        editLog: log,
        dirty: true,
        ...hist,
      };
    }),

  restoreSelected: () =>
    set((s) => {
      if (s.selectedIds.size === 0) return {};
      const ids = Array.from(s.selectedIds);
      const idSet = s.selectedIds;
      const hist = pushHistory(s);
      const segments = s.segments.map((seg) =>
        idSet.has(seg.id) && seg.status === 'rejected'
          ? { ...seg, status: 'active' as const }
          : seg,
      );
      const log = [
        ...s.editLog,
        ...ids.map(
          (id): EditEvent => ({ op: 'restore', segment_id: id, ts_ms: Date.now() }),
        ),
      ];
      return { segments, editLog: log, dirty: true, ...hist };
    }),

  deleteSelected: () => {
    // Phase 3 keeps "delete" === "reject" so undo can restore.  When we
    // serialise to the backend we only send `active` segments; rejected ones
    // live on in the edit log as evidence the operator considered them.
    get().rejectSelected();
  },

  moveEndpoint: (id, which, to) =>
    set((s) => {
      const target = s.segments.find((seg) => seg.id === id);
      if (!target) return {};
      const from: [number, number] = which === 0 ? [target.x1, target.y1] : [target.x2, target.y2];
      if (from[0] === to[0] && from[1] === to[1]) return {};
      const hist = pushHistory(s);
      const segments = s.segments.map((seg) =>
        seg.id === id
          ? which === 0
            ? { ...seg, x1: to[0], y1: to[1] }
            : { ...seg, x2: to[0], y2: to[1] }
          : seg,
      );
      const log: EditEvent[] = [
        ...s.editLog,
        {
          op: 'move_endpoint',
          segment_id: id,
          which_endpoint: which,
          from_xy: from,
          to_xy: to,
          ts_ms: Date.now(),
        },
      ];
      return { segments, editLog: log, dirty: true, ...hist };
    }),

  snapSelectedToManhattan: () =>
    set((s) => {
      if (s.selectedIds.size === 0) return {};
      const idSet = s.selectedIds;
      const targets = s.segments.filter(
        (seg) => idSet.has(seg.id) && seg.status === 'active',
      );
      if (targets.length === 0) return {};

      // Find dominant axis among the selected segments (length-weighted angle hist).
      // Same algorithm as the backend's regularize.snap_to_manhattan, just in TS.
      const TOLERANCE_DEG = 12;
      const bins = new Array<number>(90).fill(0); // 0..180 in 2° bins
      for (const seg of targets) {
        const dx = seg.x2 - seg.x1;
        const dy = seg.y2 - seg.y1;
        const len = Math.hypot(dx, dy);
        const angle = ((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 180;
        bins[Math.floor(angle / 2)] += len;
      }
      const peakBin = bins.indexOf(Math.max(...bins));
      const dominantAngle = peakBin * 2 + 1;
      const perpAngle = (dominantAngle + 90) % 180;

      const angleDiff = (a: number, b: number) => {
        const d = Math.abs(a - b);
        return Math.min(d, 180 - d);
      };

      const hist = pushHistory(s);
      const segments = s.segments.map((seg) => {
        if (!idSet.has(seg.id) || seg.status !== 'active') return seg;
        const dx = seg.x2 - seg.x1;
        const dy = seg.y2 - seg.y1;
        const len = Math.hypot(dx, dy);
        if (len < 1e-9) return seg;
        const ang = ((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 180;
        let snap: number;
        if (angleDiff(ang, dominantAngle) <= TOLERANCE_DEG) snap = dominantAngle;
        else if (angleDiff(ang, perpAngle) <= TOLERANCE_DEG) snap = perpAngle;
        else return seg;
        const cx = (seg.x1 + seg.x2) / 2;
        const cy = (seg.y1 + seg.y2) / 2;
        const rad = (snap * Math.PI) / 180;
        const hx = (Math.cos(rad) * len) / 2;
        const hy = (Math.sin(rad) * len) / 2;
        return { ...seg, x1: cx - hx, y1: cy - hy, x2: cx + hx, y2: cy + hy };
      });
      const log: EditEvent[] = [
        ...s.editLog,
        {
          op: 'snap_manhattan',
          segment_ids: targets.map((t) => t.id),
          ts_ms: Date.now(),
        },
      ];
      return { segments, editLog: log, dirty: true, ...hist };
    }),

  undo: () =>
    set((s) => {
      if (s.history.length === 0) return {};
      const prev = s.history[s.history.length - 1];
      const newHistory = s.history.slice(0, -1);
      const currentSnap = snapshot(s);
      return {
        segments: prev.segments,
        selectedIds: new Set(prev.selection),
        history: newHistory,
        future: [...s.future, currentSnap],
        dirty: true,
      };
    }),

  redo: () =>
    set((s) => {
      if (s.future.length === 0) return {};
      const next = s.future[s.future.length - 1];
      const newFuture = s.future.slice(0, -1);
      const currentSnap = snapshot(s);
      return {
        segments: next.segments,
        selectedIds: new Set(next.selection),
        history: [...s.history, currentSnap],
        future: newFuture,
        dirty: true,
      };
    }),

  canUndo: () => get().history.length > 0,
  canRedo: () => get().future.length > 0,

  markSaved: (editVersion) =>
    set({
      lastSavedAt: Date.now(),
      lastSavedEditVersion: editVersion,
      dirty: false,
      editLog: [],
    }),
}));
