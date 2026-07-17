/**
 * Editor toolbar — batch operations, undo/redo, save & export.
 *
 * Floats over the editor canvas at the top-left.  Stays out of the way until
 * the operator selects something, then surfaces actionable controls.
 */
import { LAYER_STYLES, styleForLayer, useEditorStore } from '../../../state/editorStore';
import type { SegmentLayer } from '../../../api/vectorize';

interface Props {
  saving: boolean;
  onSave: () => void;
  /** ``null`` until the operator has done at least one save. */
  lastSavedVersion: number | null;
  dirty: boolean;
  segmentCounts: {
    active: number;
    rejected: number;
    selected: number;
    ghosts: number;
    /** Per-layer active-segment counts (for the layer-toggle pills). */
    byLayer: Record<SegmentLayer, number>;
  };
  /** Trigger duplicate-finder (handled by EditorSurface — needs the API call). */
  onFindDuplicates: () => void;
  /** True while the duplicate-finder request is in flight. */
  finding: boolean;
  /** Rebuild floor geometry + BOMA from the current wall network. */
  onGenerateBoma: () => void;
  generatingBoma: boolean;
  /** True when the job already has a measurement report (button still re-runs). */
  hasMeasurementReport: boolean;
  /** True when exterior shell is unreliable (rooms not closed) — demote label. */
  exteriorUnreliable?: boolean;
  /** True when walls exist but rooms aren't closed — show close-gaps hint. */
  showCloseGapsHint?: boolean;
  /** Count of dangling wall endpoints (for the close-gaps hint). */
  danglingCount?: number;
}

export function EditorToolbar({
  saving, onSave, lastSavedVersion, dirty, segmentCounts, onFindDuplicates, finding,
  onGenerateBoma, generatingBoma, hasMeasurementReport,
  exteriorUnreliable = false, showCloseGapsHint = false, danglingCount = 0,
}: Props) {
  const undo = useEditorStore((s) => s.undo);
  const redo = useEditorStore((s) => s.redo);
  const canUndo = useEditorStore((s) => s.history.length > 0);
  const canRedo = useEditorStore((s) => s.future.length > 0);

  const rejectSelected = useEditorStore((s) => s.rejectSelected);
  const restoreSelected = useEditorStore((s) => s.restoreSelected);
  const snapManhattan = useEditorStore((s) => s.snapSelectedToManhattan);
  const clearSelection = useEditorStore((s) => s.clearSelection);
  const selectAll = useEditorStore((s) => s.selectAll);
  const mergeSelected = useEditorStore((s) => s.mergeSelected);

  const mode = useEditorStore((s) => s.mode);
  const setMode = useEditorStore((s) => s.setMode);
  const ghostsVisible = useEditorStore((s) => s.ghostsVisible);
  const toggleGhosts = useEditorStore((s) => s.toggleGhosts);

  const drawLayer = useEditorStore((s) => s.drawLayer);
  const setDrawLayer = useEditorStore((s) => s.setDrawLayer);
  const layerVisibility = useEditorStore((s) => s.layerVisibility);
  const toggleLayerVisible = useEditorStore((s) => s.toggleLayerVisible);

  // Phase D wiring
  const showOriginal = useEditorStore((s) => s.showOriginal);
  const toggleShowOriginal = useEditorStore((s) => s.toggleShowOriginal);
  const autoSaveEnabled = useEditorStore((s) => s.autoSaveEnabled);
  const setAutoSaveEnabled = useEditorStore((s) => s.setAutoSaveEnabled);
  const lastSavedAt = useEditorStore((s) => s.lastSavedAt);
  const originalSegments = useEditorStore((s) => s.originalSegments);

  const anySelected = segmentCounts.selected > 0;
  const canMerge = segmentCounts.selected >= 2;
  const drawing = mode === 'draw';

  // Layer pills: walls + openings + windows + columns first (always shown),
  // then any other class the operator's segment list contains.
  const layerOrder: SegmentLayer[] = ['walls', 'openings', 'windows', 'columns'];
  for (const key of Object.keys(segmentCounts.byLayer ?? {})) {
    if (!layerOrder.includes(key)) layerOrder.push(key);
  }

  // "Show original" — only meaningful if the operator has actually edited
  // away from the baseline.  Disable when ``originalSegments`` is empty.
  const canShowOriginal = originalSegments.length > 0;

  return (
    <div className="absolute top-3 left-3 flex flex-col gap-2 pointer-events-auto">
      {/* Counts strip — one chip per layer + rejected + selected. */}
      <div className="rounded-lg bg-gray-900/85 backdrop-blur-sm border border-gray-700 px-3 py-1.5 text-[11px] text-gray-400 font-mono flex items-center gap-3">
        {layerOrder.map((layer) => {
          const n = segmentCounts.byLayer?.[layer] ?? 0;
          if (n === 0 && layer !== 'walls') return null;  // only show empty walls (anchors the strip)
          const style = styleForLayer(layer);
          return (
            <span key={`count-${layer}`}>
              <span style={{ color: style.color }}>{n}</span> {style.label.toLowerCase()}
            </span>
          );
        }).filter(Boolean).reduce<React.ReactNode[]>((acc, node, i) => {
          if (i > 0) acc.push(<span key={`sep-${i}`} className="text-gray-700">·</span>);
          acc.push(node);
          return acc;
        }, [])}
        <span className="text-gray-700">·</span>
        <span><span className="text-rose-400">{segmentCounts.rejected}</span> rejected</span>
        {segmentCounts.ghosts > 0 && (
          <>
            <span className="text-gray-700">·</span>
            <span><span className="text-violet-300">{segmentCounts.ghosts}</span> ghosts</span>
          </>
        )}
        <span className="text-gray-700">·</span>
        <span>
          <span className={segmentCounts.selected > 0 ? 'text-amber-300' : 'text-gray-600'}>
            {segmentCounts.selected}
          </span>{' '}
          selected
        </span>
      </div>

      {/* Layer visibility pills — click toggles render + hit-test for that layer. */}
      <div className="rounded-lg bg-gray-900/85 backdrop-blur-sm border border-gray-700 p-1 flex items-center gap-1">
        <span className="text-[10px] uppercase tracking-wide text-gray-500 px-2">Layers</span>
        {layerOrder.map((layer) => {
          const style = styleForLayer(layer);
          const n = segmentCounts.byLayer?.[layer] ?? 0;
          const visible = layerVisibility[layer] !== false;
          const isExterior = layer === 'walls_exterior';
          const label = isExterior && exteriorUnreliable
            ? 'Exterior shell (unreliable)'
            : style.label;
          const title = isExterior && exteriorUnreliable
            ? `Envelope hull is unreliable until rooms close — currently ${visible ? 'shown' : 'hidden'} (${n} segments). Opt-in for power users.`
            : `Toggle ${style.label} visibility — currently ${visible ? 'shown' : 'hidden'} (${n} segments)`;
          return (
            <button
              key={`vis-${layer}`}
              onClick={() => toggleLayerVisible(layer)}
              disabled={n === 0 && layer !== 'walls' && layer !== 'openings'}
              title={title}
              className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-all ${
                visible
                  ? 'text-gray-100'
                  : 'text-gray-600 line-through bg-gray-950 border-gray-800'
              }`}
              style={visible ? {
                backgroundColor: style.color + '22',
                borderColor: style.color + '88',
                color: style.color,
              } : undefined}
            >
              {label} <span className="opacity-60">·{n}</span>
            </button>
          );
        })}
      </div>

      {showCloseGapsHint && (
        <div className="rounded-lg bg-amber-950/90 backdrop-blur-sm border border-amber-700/80 px-3 py-1.5 text-[11px] text-amber-100 max-w-md">
          Close open wall ends
          {danglingCount > 0 ? ` (${danglingCount} open)` : ''}
          {' '}→ Generate BOMA. Green walls + scan underlay are the trusted draft.
        </div>
      )}

      {/* Tool selector + draw-layer selector + ghosts + duplicates + measure — the "fly-through" row */}
      <div className="rounded-lg bg-gray-900/85 backdrop-blur-sm border border-gray-700 p-1 flex items-center gap-1">
        <ToolBtn
          onClick={() => setMode(drawing ? 'select' : 'draw')}
          label={drawing ? `✎ ${styleForLayer(drawLayer).label.toLowerCase().replace(/s$/, '')}` : '✎ Draw'}
          hint={`Draw a new ${styleForLayer(drawLayer).label.toLowerCase().replace(/s$/, '')} (D=wall, O=opening, W=window, C=column).  Click-drag in the canvas to sketch.`}
          variant={drawing ? 'primary' : 'default'}
        />
        {drawing && (
          <div className="flex items-center gap-1 pl-1 pr-2 border-l border-gray-700 ml-1">
            <span className="text-[10px] uppercase tracking-wide text-gray-500">on</span>
            {(['walls', 'openings', 'windows', 'columns'] as SegmentLayer[]).map((layer) => {
              const style = LAYER_STYLES[layer];
              const active = drawLayer === layer;
              return (
                <button
                  key={`draw-${layer}`}
                  onClick={() => setDrawLayer(layer)}
                  className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-all ${
                    active ? 'text-white' : 'text-gray-400 border-gray-700 hover:text-gray-200'
                  }`}
                  style={active ? {
                    backgroundColor: style.color,
                    borderColor: style.color,
                  } : undefined}
                  title={`Switch new-draw target to ${style.label.toLowerCase()}`}
                >
                  {style.label}
                </button>
              );
            })}
          </div>
        )}
        <ToolBtn
          onClick={() => setMode(mode === 'measure' ? 'select' : 'measure')}
          label={mode === 'measure' ? '📏 Measuring' : '📏 Ruler'}
          hint="Measure distance between two points.  Click 2 points; snaps to endpoints/walls. (R)"
          variant={mode === 'measure' ? 'primary' : 'default'}
        />
        <ToolBtn
          onClick={toggleGhosts}
          disabled={segmentCounts.ghosts === 0}
          label={ghostsVisible ? '◉ Ghosts' : '○ Ghosts'}
          hint={segmentCounts.ghosts === 0
            ? 'No pipeline-rejected candidates for this job'
            : `Show ${segmentCounts.ghosts} pipeline-rejected ghost candidates.  Click a ghost to promote it. (G)`}
          variant={ghostsVisible ? 'primary' : 'default'}
        />
        <ToolBtn
          onClick={toggleShowOriginal}
          disabled={!canShowOriginal}
          label={showOriginal ? '◉ Original' : '○ Original'}
          hint={canShowOriginal
            ? 'Overlay the pipeline\'s pre-edit output in faint grey so you can see what you changed. (\u00A0)'
            : 'Nothing loaded yet'}
          variant={showOriginal ? 'primary' : 'default'}
        />
        <Divider />
        <ToolBtn
          onClick={onFindDuplicates}
          disabled={finding || segmentCounts.active < 2}
          label={finding ? '⏳ Scanning…' : '⚲ Find duplicates'}
          hint="Scan the current segment list for near-duplicate pairs the regularizer was too conservative to merge. (F)"
        />
        <ToolBtn
          onClick={mergeSelected}
          disabled={!canMerge}
          label="⨉ Merge"
          hint={canMerge
            ? `Merge ${segmentCounts.selected} selected segments into one centreline (M).  Merging is restricted to a single layer.`
            : 'Select 2+ segments on the same layer to merge'}
        />
      </div>

      {/* Edit ops */}
      <div className="rounded-lg bg-gray-900/85 backdrop-blur-sm border border-gray-700 p-1 flex gap-1">
        <ToolBtn onClick={undo} disabled={!canUndo} label="↶" hint="Undo (⌘Z)" />
        <ToolBtn onClick={redo} disabled={!canRedo} label="↷" hint="Redo (⇧⌘Z)" />
        <Divider />
        <ToolBtn onClick={selectAll} label="All" hint="Select all active (⌘A)" />
        <ToolBtn onClick={clearSelection} disabled={!anySelected} label="None" hint="Clear selection (Esc)" />
        <Divider />
        <ToolBtn
          onClick={rejectSelected}
          disabled={!anySelected}
          label="Reject"
          hint="Mark selected as rejected (Delete)"
          variant="danger"
        />
        <ToolBtn
          onClick={restoreSelected}
          disabled={!anySelected}
          label="Restore"
          hint="Bring back rejected selected"
        />
        <ToolBtn
          onClick={snapManhattan}
          disabled={!anySelected}
          label="Snap"
          hint="Snap selected to dominant building axes"
        />
      </div>

      {/* Save + auto-save indicator */}
      <div className="rounded-lg bg-gray-900/85 backdrop-blur-sm border border-gray-700 p-1 flex items-center gap-2">
        <button
          onClick={onSave}
          disabled={saving || !dirty || generatingBoma}
          className={`
            px-3 py-1.5 rounded text-xs font-semibold transition-colors
            ${dirty && !saving && !generatingBoma
              ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
              : 'bg-gray-800 text-gray-500 cursor-not-allowed'}
          `}
        >
          {saving ? 'Saving…' : '💾 Save & Re-export DXF'}
        </button>
        <button
          onClick={onGenerateBoma}
          disabled={generatingBoma || saving || (segmentCounts.byLayer?.walls ?? 0) < 3}
          title={
            (segmentCounts.byLayer?.walls ?? 0) < 3
              ? 'Need at least 3 active walls to form rooms'
              : hasMeasurementReport
                ? 'Re-run topology + BOMA from the current wall network'
                : 'Close open wall ends, then generate floor geometry + BOMA'
          }
          className={`
            px-3 py-1.5 rounded text-xs font-semibold transition-colors
            ${!generatingBoma && !saving && (segmentCounts.byLayer?.walls ?? 0) >= 3
              ? 'bg-amber-600 hover:bg-amber-500 text-white'
              : 'bg-gray-800 text-gray-500 cursor-not-allowed'}
          `}
        >
          {generatingBoma
            ? 'Generating BOMA…'
            : hasMeasurementReport
              ? '↻ Re-generate BOMA'
              : 'Generate BOMA'}
        </button>
        <span className="text-[11px] text-gray-500 pr-2">
          {dirty
            ? 'unsaved changes'
            : lastSavedVersion !== null
              ? `saved v${lastSavedVersion}${lastSavedAt ? ` · ${relativeTime(lastSavedAt)}` : ''}`
              : 'pristine'}
        </span>
        <Divider />
        <button
          onClick={() => setAutoSaveEnabled(!autoSaveEnabled)}
          className={`px-2 py-1 rounded text-[10px] font-medium transition-colors ${
            autoSaveEnabled ? 'bg-emerald-950 text-emerald-300 hover:bg-emerald-900' : 'bg-gray-800 text-gray-500 hover:bg-gray-700'
          }`}
          title="Toggle auto-save (every 30 s while editing). Always-on by default."
        >
          {autoSaveEnabled ? '⟳ Auto-save: on' : '⊘ Auto-save: off'}
        </button>
      </div>
    </div>
  );
}

function ToolBtn({
  onClick, disabled, label, hint, variant = 'default',
}: {
  onClick: () => void;
  disabled?: boolean;
  label: string;
  hint?: string;
  variant?: 'default' | 'danger' | 'primary';
}) {
  const classes = disabled
    ? 'text-gray-700 cursor-not-allowed'
    : variant === 'danger'
      ? 'text-rose-300 hover:bg-rose-950/40'
      : variant === 'primary'
        ? 'bg-sky-600 hover:bg-sky-500 text-white'
        : 'text-gray-300 hover:bg-gray-800';
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={hint}
      className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${classes}`}
    >
      {label}
    </button>
  );
}

function Divider() {
  return <span className="w-px bg-gray-700 my-1" />;
}

/** Human-friendly "12 s ago" / "3 m ago" suffix for the saved indicator. */
function relativeTime(ts: number): string {
  const delta_s = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (delta_s < 60) return `${delta_s}s ago`;
  const delta_m = Math.floor(delta_s / 60);
  if (delta_m < 60) return `${delta_m}m ago`;
  const delta_h = Math.floor(delta_m / 60);
  return `${delta_h}h ago`;
}
