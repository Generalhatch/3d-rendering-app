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
import type {
  DuplicatePair,
  EditEvent,
  EditableSegment,
  RejectedSegmentPayload,
  SegmentLayer,
} from '../api/vectorize';

/** Display style for one layer.  Drives canvas colour + toolbar pill. */
export interface LayerStyle {
  label: string;
  color: string;        // canvas stroke
  selectedColor: string;
  hoverColor: string;
  dashArray?: string;
  /**
   * Stroke-width multiplier applied to the canvas line.  Default 1.0.
   * Walls (centerlines) use 0.6 so they don't compete visually with the
   * thicker WALLS_FACES double-line — the eye reads structure first.
   */
  weight?: number;
}

/**
 * Layer presentation map.  Used by the editor canvas, toolbar, and stats —
 * adding a new class is a one-line change here.  Colours kept consistent
 * with the DXF default-layer colour-index choices in :data:`DEFAULT_LAYER_MAP`.
 */
export const LAYER_STYLES: Record<SegmentLayer, LayerStyle> = {
  walls: {
    label: 'Walls (centerlines)',
    color: '#34d399',          // emerald-400 — matches DXF ACI=3 (green)
    selectedColor: '#fde047',  // yellow-300 — high-contrast selection
    hoverColor: '#67e8f9',     // cyan-300 — subtle pre-click affordance
    weight: 0.6,               // thin: walls_faces carry the visual structure
  },
  walls_exterior: {
    label: 'Exterior shell',
    color: '#3b82f6',          // blue-500 — heavier than interior walls, matches DXF ACI=5
    selectedColor: '#fde047',
    hoverColor: '#93c5fd',
  },
  walls_faces: {
    label: 'Wall faces (double-line)',
    color: '#34d399',          // emerald-400 — primary visual structure (matches CAD-style thick walls)
    selectedColor: '#fde047',
    hoverColor: '#67e8f9',
    weight: 1.1,               // slightly bolder so paired faces dominate the render
  },
  rooms: {
    label: 'Rooms',
    color: '#f97316',          // orange-500 — matches DXF ACI=4 (cyan family, varies by viewer)
    selectedColor: '#fde047',
    hoverColor: '#fdba74',
    dashArray: '8 4',          // dashed so rooms read as enclosure, not lines
  },
  openings: {
    label: 'Openings',
    color: '#facc15',          // yellow-400 — matches DXF ACI=2 (yellow)
    selectedColor: '#fb923c',  // orange-400 — distinct from wall selection
    hoverColor: '#fef08a',
    dashArray: '6 3',          // dashed so closed/open doors read at a glance
  },
  windows: {
    label: 'Windows',
    color: '#60a5fa',          // blue-400 — matches DXF ACI=5 (blue)
    selectedColor: '#fb923c',
    hoverColor: '#93c5fd',
    dashArray: '2 4',          // dotted so windows visually differ from doors
  },
  columns: {
    label: 'Columns',
    color: '#f472b6',          // pink-400 — matches DXF ACI=1 (red) family
    selectedColor: '#f43f5e',
    hoverColor: '#fbcfe8',
  },
  mep: {
    label: 'MEP',
    color: '#a78bfa',
    selectedColor: '#c4b5fd',
    hoverColor: '#ddd6fe',
  },
};

/** Resolve a style for any layer, falling back to walls if unknown. */
export function styleForLayer(layer: SegmentLayer): LayerStyle {
  return LAYER_STYLES[layer] ?? LAYER_STYLES.walls;
}

export type SegmentStatus = 'active' | 'rejected';

export interface EditorSegment extends EditableSegment {
  status: SegmentStatus;
}

/**
 * Editor interaction mode.  Drives cursor + pointer event semantics.
 * Phase 6/D: ``measure`` is the click-to-click ruler.
 */
export type EditorMode = 'select' | 'draw' | 'measure';

/** A locked or in-progress measurement (world coords). */
export interface Measurement {
  ax: number;
  ay: number;
  bx: number;
  by: number;
}

const MAX_HISTORY = 100;

interface HistorySnapshot {
  segments: EditorSegment[];
  selection: string[];        // IDs
}

/**
 * Generate a fresh segment ID.  Stable for the lifetime of a draw/promote
 * action; persisted in segments.json on save.
 *
 * ``crypto.randomUUID`` is available in every supported browser; the fallback
 * is only here so unit-test environments without ``crypto`` don't explode.
 */
function newSegmentId(): string {
  try {
    return crypto.randomUUID().slice(0, 12);
  } catch {
    return Math.random().toString(36).slice(2, 14);
  }
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

  // ── Editor mode (select vs draw) ───────────────────────────────────────────
  mode: EditorMode;
  setMode: (m: EditorMode) => void;

  // ── Layer model (Phase 4) ─────────────────────────────────────────────────
  /** Which layer new draw-mode segments land on (default 'walls'). */
  drawLayer: SegmentLayer;
  setDrawLayer: (layer: SegmentLayer) => void;
  /** Per-layer visibility — hidden layers don't render or hit-test. */
  layerVisibility: Record<SegmentLayer, boolean>;
  toggleLayerVisible: (layer: SegmentLayer) => void;
  setLayerVisible: (layer: SegmentLayer, visible: boolean) => void;

  // ── Phase D: "show original" overlay ──────────────────────────────────────
  /** Snapshot of what was loaded — for diffing operator edits against pipeline. */
  originalSegments: EditorSegment[];
  showOriginal: boolean;
  toggleShowOriginal: () => void;

  // ── Phase D: measurement tool ─────────────────────────────────────────────
  /** Latest locked measurement (one only, replaces the previous). */
  measurement: Measurement | null;
  /** First click while in measure mode; cleared on second click or Esc. */
  measureAnchor: [number, number] | null;
  setMeasurement: (m: Measurement | null) => void;
  setMeasureAnchor: (p: [number, number] | null) => void;

  // ── Phase D: auto-save bookkeeping ────────────────────────────────────────
  autoSaveEnabled: boolean;
  setAutoSaveEnabled: (on: boolean) => void;
  /** Last auto-save error message (sticky toast in surface).  Null = OK. */
  autoSaveError: string | null;
  setAutoSaveError: (msg: string | null) => void;

  // ── Ghost layer (pipeline-rejected segments the operator can rescue) ───────
  ghosts: RejectedSegmentPayload[];
  ghostsVisible: boolean;
  setGhosts: (g: RejectedSegmentPayload[]) => void;
  toggleGhosts: () => void;
  /** Promote a ghost candidate to an active segment.  Returns the new id. */
  promoteGhost: (ghost: RejectedSegmentPayload) => string;

  // ── Duplicate finder ───────────────────────────────────────────────────────
  mergeSuggestions: DuplicatePair[];
  mergeCursor: number;                        // index into mergeSuggestions
  setMergeSuggestions: (s: DuplicatePair[]) => void;
  clearMergeSuggestions: () => void;
  /** Commit the suggestion at ``mergeCursor``. */
  acceptCurrentMergeSuggestion: () => void;
  /** Skip the current suggestion, advance cursor. */
  skipCurrentMergeSuggestion: () => void;

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
  /**
   * Add a brand-new segment on the requested ``layer`` (defaults to the
   * current ``drawLayer``).  Logged with op='add'.  Returns the new id.
   */
  addSegment: (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    layer?: SegmentLayer,
  ) => string;
  /** Merge ``ids`` into a single best-fit centreline.  Logged with op='merge'. */
  mergeSegments: (ids: string[]) => void;
  /** Merge the currently-selected segments (no-op if < 2 active selected). */
  mergeSelected: () => void;

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
  mode: 'select' as EditorMode,
  drawLayer: 'walls' as SegmentLayer,
  layerVisibility: {
    walls: true, openings: true, windows: true, columns: true,
  } as Record<SegmentLayer, boolean>,
  originalSegments: [] as EditorSegment[],
  showOriginal: false,
  measurement: null as Measurement | null,
  measureAnchor: null as [number, number] | null,
  autoSaveEnabled: true,
  autoSaveError: null as string | null,
  ghosts: [] as RejectedSegmentPayload[],
  ghostsVisible: false,
  mergeSuggestions: [] as DuplicatePair[],
  mergeCursor: 0,
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
    const fresh = segments.map((s) => ({ ...s, status: 'active' as const }));
    set({
      segments: fresh,
      // Snapshot the just-loaded segments so "Show original" can render the
      // pre-edit pipeline output behind the operator's working set.  We deep-
      // copy because the active list will mutate in place over time.
      originalSegments: fresh.map((s) => ({ ...s })),
      selectedIds: new Set<string>(),
      hoverId: null,
      editLog: [],
      history: [],
      future: [],
      dirty: false,
      loaded: true,
      measurement: null,
      measureAnchor: null,
      autoSaveError: null,
    });
  },

  reset: () =>
    set({ ...initialState, selectedIds: new Set<string>() }),

  // ── Editor mode ──────────────────────────────────────────────────────────
  setMode: (mode) => set({ mode }),

  // ── Layer model ──────────────────────────────────────────────────────────
  setDrawLayer: (drawLayer) => set({ drawLayer }),

  toggleLayerVisible: (layer) =>
    set((s) => ({
      layerVisibility: { ...s.layerVisibility, [layer]: !(s.layerVisibility[layer] ?? true) },
    })),

  setLayerVisible: (layer, visible) =>
    set((s) => ({ layerVisibility: { ...s.layerVisibility, [layer]: visible } })),

  // ── Phase D: show-original overlay ───────────────────────────────────────
  toggleShowOriginal: () => set((s) => ({ showOriginal: !s.showOriginal })),

  // ── Phase D: measurement tool ────────────────────────────────────────────
  setMeasurement: (measurement) => set({ measurement }),
  setMeasureAnchor: (measureAnchor) => set({ measureAnchor }),

  // ── Phase D: auto-save bookkeeping ───────────────────────────────────────
  setAutoSaveEnabled: (autoSaveEnabled) => set({ autoSaveEnabled }),
  setAutoSaveError: (autoSaveError) => set({ autoSaveError }),

  // ── Ghost layer ──────────────────────────────────────────────────────────
  setGhosts: (ghosts) => set({ ghosts }),

  toggleGhosts: () => set((s) => ({ ghostsVisible: !s.ghostsVisible })),

  promoteGhost: (ghost) => {
    const newId = newSegmentId();
    set((s) => {
      const hist = pushHistory(s);
      const seg: EditorSegment = {
        id: newId,
        layer: ghost.layer ?? 'walls',
        x1: ghost.x1, y1: ghost.y1, x2: ghost.x2, y2: ghost.y2,
        status: 'active',
      };
      const segments = [...s.segments, seg];
      // Remove the ghost from the rescuable list so it can't be re-promoted.
      const ghosts = s.ghosts.filter((g) => g.id !== ghost.id);
      const log: EditEvent[] = [
        ...s.editLog,
        {
          op: 'promote_ghost',
          segment_id: ghost.id,
          new_id: newId,
          from_xy: [ghost.x1, ghost.y1],
          to_xy: [ghost.x2, ghost.y2],
          ts_ms: Date.now(),
        },
      ];
      return {
        segments,
        ghosts,
        selectedIds: new Set([newId]),
        editLog: log,
        dirty: true,
        ...hist,
      };
    });
    return newId;
  },

  // ── Duplicate finder ─────────────────────────────────────────────────────
  setMergeSuggestions: (mergeSuggestions) =>
    set({ mergeSuggestions, mergeCursor: 0 }),

  clearMergeSuggestions: () =>
    set({ mergeSuggestions: [], mergeCursor: 0 }),

  acceptCurrentMergeSuggestion: () => {
    const { mergeSuggestions, mergeCursor, mergeSegments: doMerge, clearMergeSuggestions } = get();
    if (mergeCursor >= mergeSuggestions.length) return;
    const pair = mergeSuggestions[mergeCursor];
    doMerge([pair.a_id, pair.b_id]);
    // Advance to the next suggestion; auto-clear when exhausted.
    set((s) => {
      const next = s.mergeCursor + 1;
      if (next >= s.mergeSuggestions.length) {
        clearMergeSuggestions();
        return {};
      }
      return { mergeCursor: next };
    });
  },

  skipCurrentMergeSuggestion: () =>
    set((s) => {
      const next = s.mergeCursor + 1;
      if (next >= s.mergeSuggestions.length) {
        return { mergeSuggestions: [], mergeCursor: 0 };
      }
      return { mergeCursor: next };
    }),

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

  addSegment: (x1, y1, x2, y2, layer) => {
    const newId = newSegmentId();
    set((s) => {
      const hist = pushHistory(s);
      const seg: EditorSegment = {
        id: newId,
        layer: layer ?? s.drawLayer ?? 'walls',
        x1, y1, x2, y2,
        status: 'active',
      };
      const segments = [...s.segments, seg];
      const log: EditEvent[] = [
        ...s.editLog,
        {
          op: 'add',
          new_id: newId,
          from_xy: [x1, y1],
          to_xy: [x2, y2],
          ts_ms: Date.now(),
        },
      ];
      return {
        segments,
        selectedIds: new Set([newId]),
        editLog: log,
        dirty: true,
        ...hist,
      };
    });
    return newId;
  },

  mergeSegments: (ids) =>
    set((s) => {
      if (ids.length < 2) return {};
      const idSet = new Set(ids);
      const candidates = s.segments.filter(
        (seg) => idSet.has(seg.id) && seg.status === 'active',
      );
      if (candidates.length < 2) return {};
      // Merging across layers is almost always a mistake (e.g. an opening
      // accidentally gets selected with two walls).  Restrict to the layer
      // of the first selected segment.
      const targetLayer = candidates[0].layer ?? 'walls';
      const members = candidates.filter((seg) => (seg.layer ?? 'walls') === targetLayer);
      if (members.length < 2) return {};

      // Length-weighted average direction → endpoints from projected extents
      // along that direction.  Same algorithm as the backend's
      // ``merge_segment_group``; kept here so the editor can preview without
      // a round trip.
      let sumDx = 0;
      let sumDy = 0;
      let totalLen = 0;
      const centres: [number, number][] = [];
      for (const m of members) {
        const dx = m.x2 - m.x1;
        const dy = m.y2 - m.y1;
        const len = Math.hypot(dx, dy);
        if (len < 1e-9) continue;
        // Flip second-and-later segments to align with the first segment's heading.
        const flip =
          sumDx === 0 && sumDy === 0
            ? 1
            : (dx * sumDx + dy * sumDy) >= 0
              ? 1
              : -1;
        sumDx += dx * flip;
        sumDy += dy * flip;
        totalLen += len;
        centres.push([(m.x1 + m.x2) / 2, (m.y1 + m.y2) / 2]);
      }
      if (totalLen < 1e-9) return {};
      const mag = Math.hypot(sumDx, sumDy) || 1;
      const ux = sumDx / mag;
      const uy = sumDy / mag;

      let cxAcc = 0;
      let cyAcc = 0;
      for (let i = 0; i < members.length; i++) {
        const len = Math.hypot(members[i].x2 - members[i].x1, members[i].y2 - members[i].y1);
        cxAcc += centres[i][0] * len;
        cyAcc += centres[i][1] * len;
      }
      const cx = cxAcc / totalLen;
      const cy = cyAcc / totalLen;

      const ts: number[] = [];
      for (const m of members) {
        ts.push((m.x1 - cx) * ux + (m.y1 - cy) * uy);
        ts.push((m.x2 - cx) * ux + (m.y2 - cy) * uy);
      }
      const tMin = Math.min(...ts);
      const tMax = Math.max(...ts);
      const nx1 = cx + ux * tMin;
      const ny1 = cy + uy * tMin;
      const nx2 = cx + ux * tMax;
      const ny2 = cy + uy * tMax;

      const newId = newSegmentId();
      const hist = pushHistory(s);
      const merged: EditorSegment = {
        id: newId,
        layer: members[0].layer ?? 'walls',
        x1: nx1, y1: ny1, x2: nx2, y2: ny2,
        status: 'active',
      };
      // Replace members with the merged segment; preserve original order by
      // dropping members and appending the new one at the position of the
      // first member.
      const removed = new Set(members.map((m) => m.id));
      const firstIdx = s.segments.findIndex((seg) => removed.has(seg.id));
      const filtered = s.segments.filter((seg) => !removed.has(seg.id));
      const segments =
        firstIdx >= 0
          ? [...filtered.slice(0, firstIdx), merged, ...filtered.slice(firstIdx)]
          : [...filtered, merged];

      const log: EditEvent[] = [
        ...s.editLog,
        {
          op: 'merge',
          segment_ids: members.map((m) => m.id),
          new_id: newId,
          from_xy: [nx1, ny1],
          to_xy: [nx2, ny2],
          ts_ms: Date.now(),
        },
      ];
      return {
        segments,
        selectedIds: new Set([newId]),
        editLog: log,
        dirty: true,
        ...hist,
      };
    }),

  mergeSelected: () => {
    const { selectedIds, segments, mergeSegments: doMerge } = get();
    const ids = segments
      .filter((seg) => selectedIds.has(seg.id) && seg.status === 'active')
      .map((seg) => seg.id);
    if (ids.length >= 2) doMerge(ids);
  },

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
