/**
 * Top-level orchestration for the editor.
 *
 * Owns the load-segments-from-API lifecycle, mounts ``EditorCanvas`` and
 * ``EditorToolbar``, and wires global keyboard shortcuts (Delete, Esc, ⌘Z,
 * etc.) at the document level.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useEditorStore } from '../../../state/editorStore';
import { vectorizeApi, type RasterAffine } from '../../../api/vectorize';
import { EditorCanvas } from './EditorCanvas';
import { EditorToolbar } from './EditorToolbar';

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

  const [affine, setAffine] = useState<RasterAffine | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

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
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : 'Failed to load segments');
        }
      }
    })();

    return () => { cancelled = true; };
    // We intentionally do NOT include bindJob/loadSegments in deps — they're
    // stable Zustand setters that we only want to fire when jobId changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  // ── Save handler ───────────────────────────────────────────────────────
  const handleSave = useCallback(async () => {
    if (saving || !dirty) return;
    setSaving(true);
    setSaveError(null);
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
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  }, [saving, dirty, segments, editLog, jobId, markSaved]);

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
        e.preventDefault(); clearSelection(); return;
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault(); rejectSelected(); return;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [undo, redo, selectAll, clearSelection, rejectSelected, handleSave]);

  // ── Derived state for the toolbar ──────────────────────────────────────
  const segmentCounts = useMemo(() => {
    let active = 0;
    let rejected = 0;
    for (const s of segments) {
      if (s.status === 'active') active++;
      else if (s.status === 'rejected') rejected++;
    }
    return { active, rejected, selected: selectedIds.size };
  }, [segments, selectedIds]);

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
      />
      <EditorToolbar
        saving={saving}
        onSave={handleSave}
        lastSavedVersion={lastSavedEditVersion}
        dirty={dirty}
        segmentCounts={segmentCounts}
      />
      {saveError && (
        <div className="absolute bottom-12 left-3 rounded-lg bg-rose-950/80 border border-rose-700 px-3 py-2 text-xs text-rose-200 backdrop-blur-sm">
          {saveError}
        </div>
      )}
    </div>
  );
}
