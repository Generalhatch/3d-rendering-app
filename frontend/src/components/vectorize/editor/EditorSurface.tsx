/**
 * Top-level orchestration for the editor.
 *
 * Owns the load-segments-from-API lifecycle, mounts ``EditorCanvas`` and
 * ``EditorToolbar``, and wires global keyboard shortcuts (Delete, Esc, ⌘Z,
 * etc.) at the document level.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useEditorStore } from '../../../state/editorStore';
import { useVectorizeStore } from '../../../state/vectorizeStore';
import {
  VectorizeApiError,
  vectorizeApi,
  type DanglingEndpoint,
  type RasterAffine,
  type RoomsNotClosedDetail,
  type SegmentLayer,
} from '../../../api/vectorize';
import { EditorCanvas } from './EditorCanvas';
import { EditorToolbar } from './EditorToolbar';
import { computeDanglingEndpoints } from './danglingEndpoints';

interface Props {
  jobId: string;
}

export function EditorSurface({ jobId }: Props) {
  const bindJob = useEditorStore((s) => s.bindJob);
  const loadSegments = useEditorStore((s) => s.loadSegments);
  const markSaved = useEditorStore((s) => s.markSaved);
  const segments = useEditorStore((s) => s.segments);
  const selectedIds = useEditorStore((s) => s.selectedIds);
  const editLog = useEditorStore((s) => s.editLog);
  const dirty = useEditorStore((s) => s.dirty);
  const lastSavedEditVersion = useEditorStore((s) => s.lastSavedEditVersion);
  const loaded = useEditorStore((s) => s.loaded);
  const boundJobId = useEditorStore((s) => s.jobId);

  const undo = useEditorStore((s) => s.undo);
  const redo = useEditorStore((s) => s.redo);
  const rejectSelected = useEditorStore((s) => s.rejectSelected);
  const clearSelection = useEditorStore((s) => s.clearSelection);
  const selectAll = useEditorStore((s) => s.selectAll);
  const mergeSelected = useEditorStore((s) => s.mergeSelected);

  const mode = useEditorStore((s) => s.mode);
  const setMode = useEditorStore((s) => s.setMode);
  const setDrawLayer = useEditorStore((s) => s.setDrawLayer);
  const ghosts = useEditorStore((s) => s.ghosts);
  const setGhosts = useEditorStore((s) => s.setGhosts);
  const toggleGhosts = useEditorStore((s) => s.toggleGhosts);
  const setMergeSuggestions = useEditorStore((s) => s.setMergeSuggestions);

  // Phase D
  const autoSaveEnabled = useEditorStore((s) => s.autoSaveEnabled);
  const autoSaveError = useEditorStore((s) => s.autoSaveError);
  const setAutoSaveError = useEditorStore((s) => s.setAutoSaveError);
  const toggleShowOriginal = useEditorStore((s) => s.toggleShowOriginal);
  const setMeasurement = useEditorStore((s) => s.setMeasurement);
  const setMeasureAnchor = useEditorStore((s) => s.setMeasureAnchor);
  const measurement = useEditorStore((s) => s.measurement);
  const measureAnchor = useEditorStore((s) => s.measureAnchor);

  const [affine, setAffine] = useState<RasterAffine | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [finding, setFinding] = useState(false);
  const [findError, setFindError] = useState<string | null>(null);
  const [generatingBoma, setGeneratingBoma] = useState(false);
  const [bomaError, setBomaError] = useState<string | null>(null);
  const [bomaSuccess, setBomaSuccess] = useState<string | null>(null);
  const [serverDangling, setServerDangling] = useState<DanglingEndpoint[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  // Pull `has_coverage` from the rehydrated job detail so we only render the
  // diagnostic toggle when the artifact actually exists (legacy jobs lack it).
  const hasCoverage = useVectorizeStore((s) => s.job?.has_coverage ?? false);
  const hasRejected = useVectorizeStore((s) => s.job?.has_rejected ?? false);
  const hasMeasurementReport = useVectorizeStore((s) => s.job?.has_measurement_report ?? false);
  const hasFloorGeometry = useVectorizeStore((s) => s.job?.has_floor_geometry ?? false);
  const roomsDetected = useVectorizeStore((s) => s.job?.metrics?.rooms_detected ?? null);
  const setJob = useVectorizeStore((s) => s.setJob);
  const bumpSheetVersion = useVectorizeStore((s) => s.bumpSheetVersion);
  const setViewMode = useVectorizeStore((s) => s.setViewMode);
  const setSidebarTab = useVectorizeStore((s) => s.setSidebarTab);
  const setLayerVisible = useEditorStore((s) => s.setLayerVisible);

  // Demote the blue exterior-shell scribble when rooms aren't closed — green
  // walls + raster are the trusted draft.  Power users can re-enable via Layers.
  const exteriorUnreliable = !hasFloorGeometry || roomsDetected === 0;
  useEffect(() => {
    if (exteriorUnreliable) {
      setLayerVisible('walls_exterior', false);
    }
  }, [exteriorUnreliable, setLayerVisible, jobId]);

  // Rooms layer: only auto-show when rings closed and floor geometry exists
  // (isoperimetric filter already dropped spiky candidates server-side).
  // Keep hidden when rooms_detected is 0 / null so orange garbage stays off.
  useEffect(() => {
    if (roomsDetected != null && roomsDetected > 0 && hasFloorGeometry) {
      setLayerVisible('rooms', true);
    } else {
      setLayerVisible('rooms', false);
    }
  }, [roomsDetected, hasFloorGeometry, setLayerVisible, jobId]);

  // ── Load segments + affine from backend ────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setLoadError(null);
    if (boundJobId !== jobId) bindJob(jobId);

    (async () => {
      try {
        const payload = await vectorizeApi.getSegments(jobId);
        if (cancelled) return;
        if (!payload.affine) {
          setLoadError(
            'This job was created by an older pipeline version that did not ' +
              'persist the raster affine.  Re-run the job to use the editor.',
          );
          return;
        }
        setAffine(payload.affine);
        loadSegments(payload.segments);
        setServerDangling(payload.dangling_endpoints ?? []);

        // Also fetch the pipeline-rejected ghost candidates if the artifact
        // exists for this job.  Failure is non-fatal — the editor still works
        // without them; the ghosts toggle just stays disabled.
        if (hasRejected) {
          try {
            const rej = await vectorizeApi.getRejectedSegments(jobId);
            if (!cancelled) setGhosts(rej.rejected ?? []);
          } catch {
            // ignore — legacy jobs simply have no ghost layer
          }
        }
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : 'Failed to load segments');
        }
      }
    })();

    return () => { cancelled = true; };
    // We intentionally do NOT include bindJob/loadSegments/setGhosts in deps
    // — they're stable Zustand setters that we only want to fire when jobId
    // (or has_rejected) changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, hasRejected]);

  // ── Save handler (shared by manual save + auto-save) ──────────────────
  const handleSave = useCallback(async (opts?: { silent?: boolean }) => {
    if (saving || !dirty) return;
    setSaving(true);
    if (!opts?.silent) setSaveError(null);
    try {
      const active = segments.filter((s) => s.status === 'active').map((s) => ({
        id: s.id,
        layer: s.layer,
        x1: s.x1,
        y1: s.y1,
        x2: s.x2,
        y2: s.y2,
      }));
      const response = await vectorizeApi.saveEdits(jobId, active, editLog);
      markSaved(response.edit_version);
      // Clear any prior auto-save error toast on success.
      if (autoSaveError) setAutoSaveError(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Save failed';
      if (opts?.silent) {
        setAutoSaveError(msg);
      } else {
        setSaveError(msg);
      }
    } finally {
      setSaving(false);
    }
  }, [saving, dirty, segments, editLog, jobId, markSaved, autoSaveError, setAutoSaveError]);

  const handleGenerateBoma = useCallback(async () => {
    if (generatingBoma || saving) return;
    setGeneratingBoma(true);
    setBomaError(null);
    setBomaSuccess(null);
    try {
      const active = segments.filter((s) => s.status === 'active').map((s) => ({
        id: s.id,
        layer: s.layer,
        x1: s.x1,
        y1: s.y1,
        x2: s.x2,
        y2: s.y2,
      }));
      const response = await vectorizeApi.generateMeasurement(jobId, {
        segments: active,
        log: editLog,
        save_first: dirty,
      });
      if (response.edit_version != null) {
        markSaved(response.edit_version);
      }
      const refreshed = await vectorizeApi.get(jobId);
      setJob(refreshed);
      bumpSheetVersion();
      setBomaSuccess(
        `BOMA ready — ${response.rooms_detected} room(s) measured`,
      );
      setServerDangling([]);
      setSidebarTab('deliver');
      setViewMode('sheet');
    } catch (err) {
      if (err instanceof VectorizeApiError && err.status === 422) {
        const detail = err.detail as RoomsNotClosedDetail | string;
        if (detail && typeof detail === 'object') {
          if (Array.isArray(detail.dangling_endpoints)) {
            setServerDangling(detail.dangling_endpoints);
          }
          setBomaError(detail.message || err.message);
        } else {
          setBomaError(err.message);
        }
      } else {
        setBomaError(err instanceof Error ? err.message : 'Generate BOMA failed');
      }
    } finally {
      setGeneratingBoma(false);
    }
  }, [
    generatingBoma, saving, segments, editLog, dirty, jobId, markSaved,
    setJob, bumpSheetVersion, setSidebarTab, setViewMode,
  ]);

  // ── Auto-save loop ─────────────────────────────────────────────────────
  // Every 30 s, if (a) auto-save is enabled, (b) we have unsaved changes, and
  // (c) no save is currently in flight, fire a silent save.  We avoid auto-
  // saving the first 5 s after a manual edit to give operators a chance to
  // undo without us racing them.
  useEffect(() => {
    if (!autoSaveEnabled) return;
    const interval = window.setInterval(() => {
      if (dirty && !saving) {
        void handleSave({ silent: true });
      }
    }, 30_000);
    return () => window.clearInterval(interval);
  }, [autoSaveEnabled, dirty, saving, handleSave]);

  // ── Find-duplicates handler ────────────────────────────────────────────
  const handleFindDuplicates = useCallback(async () => {
    if (finding) return;
    setFinding(true);
    setFindError(null);
    try {
      // Walk the *unsaved* state by passing the operator a quick warning if
      // they haven't saved — the backend reads from segments.json, which
      // reflects the last save only.  If dirty, ask them to save first so
      // results match what they see on screen.
      if (dirty) {
        const ok = confirm(
          'You have unsaved changes.  The duplicate finder works on the last ' +
          'saved segment list.  Save now to include your changes?',
        );
        if (ok) await handleSave();
      }
      const response = await vectorizeApi.findDuplicates(jobId);
      setMergeSuggestions(response.suggestions);
      if (response.suggestions.length === 0) {
        setFindError('No near-duplicate pairs found.');
        // Auto-clear after 3 s so the toast doesn't linger.
        setTimeout(() => setFindError(null), 3000);
      }
    } catch (err) {
      setFindError(err instanceof Error ? err.message : 'Find-duplicates failed');
    } finally {
      setFinding(false);
    }
  }, [finding, dirty, handleSave, jobId, setMergeSuggestions]);

  // ── Keyboard shortcuts ─────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return;
      }
      const meta = e.metaKey || e.ctrlKey;

      if (meta && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault(); undo(); return;
      }
      if (meta && (e.key.toLowerCase() === 'y' || (e.shiftKey && e.key.toLowerCase() === 'z'))) {
        e.preventDefault(); redo(); return;
      }
      if (meta && e.key.toLowerCase() === 'a') {
        e.preventDefault(); selectAll(); return;
      }
      if (meta && e.key.toLowerCase() === 's') {
        e.preventDefault(); void handleSave(); return;
      }
      if (e.key === 'Escape') {
        // Esc precedence: clear measurement-in-progress → clear locked
        // measurement → exit draw/measure mode → clear selection.
        e.preventDefault();
        if (measureAnchor) { setMeasureAnchor(null); return; }
        if (measurement) { setMeasurement(null); return; }
        if (mode === 'draw' || mode === 'measure') setMode('select');
        else clearSelection();
        return;
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault(); rejectSelected(); return;
      }
      // ── Mode toggles (no modifier) — only when not typing ────────────
      if (!meta && !e.shiftKey && !e.altKey) {
        const k = e.key.toLowerCase();
        // D / O / W / C = enter draw-mode on the named layer.  Pressing the
        // same key again exits draw mode.  Switching keys swaps the layer.
        const layerByKey: Record<string, SegmentLayer> = {
          d: 'walls', o: 'openings', w: 'windows', c: 'columns',
        };
        if (k in layerByKey) {
          e.preventDefault();
          const layer = layerByKey[k];
          if (mode === 'draw' && useEditorStore.getState().drawLayer === layer) {
            setMode('select');
          } else {
            setDrawLayer(layer);
            setMode('draw');
          }
          return;
        }
        if (k === 'r') {
          // R = ruler / measure mode toggle.
          e.preventDefault();
          setMode(mode === 'measure' ? 'select' : 'measure');
          return;
        }
        if (k === 'g') {
          if (ghosts.length > 0) { e.preventDefault(); toggleGhosts(); }
          return;
        }
        if (k === 'm') {
          if (selectedIds.size >= 2) { e.preventDefault(); mergeSelected(); }
          return;
        }
        if (k === 'f') {
          e.preventDefault(); void handleFindDuplicates(); return;
        }
        if (e.key === ' ') {
          // Space = toggle "show original" overlay.  Cheap diff against the
          // pre-edit baseline without leaving keyboard.
          e.preventDefault(); toggleShowOriginal(); return;
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [
    undo, redo, selectAll, clearSelection, rejectSelected, handleSave,
    handleFindDuplicates, mergeSelected, mode, setMode, setDrawLayer,
    ghosts.length, toggleGhosts, selectedIds.size,
    measureAnchor, measurement, setMeasureAnchor, setMeasurement,
    toggleShowOriginal,
  ]);

  // ── Derived state for the toolbar ──────────────────────────────────────
  const segmentCounts = useMemo(() => {
    let active = 0;
    let rejected = 0;
    const byLayer: Record<SegmentLayer, number> = { walls: 0, openings: 0 };
    for (const s of segments) {
      if (s.status === 'active') {
        active++;
        const layer = s.layer ?? 'walls';
        byLayer[layer] = (byLayer[layer] ?? 0) + 1;
      } else if (s.status === 'rejected') {
        rejected++;
      }
    }
    return {
      active,
      rejected,
      selected: selectedIds.size,
      ghosts: ghosts.length,
      byLayer,
    };
  }, [segments, selectedIds, ghosts.length]);

  const liveDangling = useMemo(
    () => (dirty ? computeDanglingEndpoints(segments) : []),
    [dirty, segments],
  );
  const danglingCount = dirty && liveDangling.length > 0
    ? liveDangling.length
    : serverDangling.length;
  const showCloseGapsHint = exteriorUnreliable
    && (segmentCounts.byLayer.walls ?? 0) >= 3;

  // ── Render ─────────────────────────────────────────────────────────────
  if (loadError) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-3 p-8 text-center">
        <div className="text-4xl">⚠</div>
        <div className="text-sm text-rose-300">{loadError}</div>
      </div>
    );
  }
  if (!loaded || !affine) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-600 gap-3">
        <div className="w-8 h-8 rounded-full border-2 border-emerald-400 border-t-transparent animate-spin" />
        <div className="text-sm">Loading editor…</div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative h-full w-full">
      <EditorCanvas
        jobId={jobId}
        affine={affine}
        rasterUrl={vectorizeApi.rasterUrl(jobId)}
        coverageUrl={hasCoverage ? vectorizeApi.coverageUrl(jobId) : null}
        serverDangling={serverDangling}
        showDangling={exteriorUnreliable}
        liveDangling={dirty}
      />
      <EditorToolbar
        saving={saving}
        onSave={handleSave}
        lastSavedVersion={lastSavedEditVersion}
        dirty={dirty}
        segmentCounts={segmentCounts}
        onFindDuplicates={handleFindDuplicates}
        finding={finding}
        onGenerateBoma={handleGenerateBoma}
        generatingBoma={generatingBoma}
        hasMeasurementReport={hasMeasurementReport}
        exteriorUnreliable={exteriorUnreliable}
        showCloseGapsHint={showCloseGapsHint}
        danglingCount={danglingCount}
      />
      {bomaSuccess && !bomaError && (
        <div className="absolute bottom-12 left-3 rounded-lg bg-emerald-950/80 border border-emerald-700 px-3 py-2 text-xs text-emerald-200 backdrop-blur-sm flex items-center gap-2">
          <span>{bomaSuccess}</span>
          <button
            onClick={() => setBomaSuccess(null)}
            className="text-emerald-300 hover:text-emerald-100 text-[10px] underline"
          >
            dismiss
          </button>
        </div>
      )}
      {bomaError && (
        <div className="absolute bottom-12 left-3 max-w-md rounded-lg bg-rose-950/80 border border-rose-700 px-3 py-2 text-xs text-rose-200 backdrop-blur-sm">
          {bomaError}
        </div>
      )}
      {saveError && !bomaError && (
        <div className="absolute bottom-12 left-3 rounded-lg bg-rose-950/80 border border-rose-700 px-3 py-2 text-xs text-rose-200 backdrop-blur-sm">
          {saveError}
        </div>
      )}
      {autoSaveError && !saveError && !bomaError && (
        <div className="absolute bottom-12 left-3 rounded-lg bg-amber-950/80 border border-amber-700 px-3 py-2 text-xs text-amber-200 backdrop-blur-sm flex items-center gap-2">
          <span>Auto-save failed: {autoSaveError}</span>
          <button
            onClick={() => setAutoSaveError(null)}
            className="text-amber-300 hover:text-amber-100 text-[10px] underline"
          >
            dismiss
          </button>
        </div>
      )}
      {findError && (
        <div className="absolute bottom-12 right-3 rounded-lg bg-amber-950/80 border border-amber-700 px-3 py-2 text-xs text-amber-200 backdrop-blur-sm">
          {findError}
        </div>
      )}
    </div>
  );
}
