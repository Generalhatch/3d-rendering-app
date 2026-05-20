/**
 * Editor toolbar — batch operations, undo/redo, save & export.
 *
 * Floats over the editor canvas at the top-left.  Stays out of the way until
 * the operator selects something, then surfaces actionable controls.
 */
import { useEditorStore } from '../../../state/editorStore';

interface Props {
  saving: boolean;
  onSave: () => void;
  /** ``null`` until the operator has done at least one save. */
  lastSavedVersion: number | null;
  dirty: boolean;
  segmentCounts: { active: number; rejected: number; selected: number };
}

export function EditorToolbar({ saving, onSave, lastSavedVersion, dirty, segmentCounts }: Props) {
  const undo = useEditorStore((s) => s.undo);
  const redo = useEditorStore((s) => s.redo);
  const canUndo = useEditorStore((s) => s.history.length > 0);
  const canRedo = useEditorStore((s) => s.future.length > 0);

  const rejectSelected = useEditorStore((s) => s.rejectSelected);
  const restoreSelected = useEditorStore((s) => s.restoreSelected);
  const snapManhattan = useEditorStore((s) => s.snapSelectedToManhattan);
  const clearSelection = useEditorStore((s) => s.clearSelection);
  const selectAll = useEditorStore((s) => s.selectAll);

  const anySelected = segmentCounts.selected > 0;

  return (
    <div className="absolute top-3 left-3 flex flex-col gap-2 pointer-events-auto">
      {/* Counts strip */}
      <div className="rounded-lg bg-gray-900/85 backdrop-blur-sm border border-gray-700 px-3 py-1.5 text-[11px] text-gray-400 font-mono flex items-center gap-3">
        <span><span className="text-emerald-400">{segmentCounts.active}</span> walls</span>
        <span className="text-gray-700">·</span>
        <span><span className="text-rose-400">{segmentCounts.rejected}</span> rejected</span>
        <span className="text-gray-700">·</span>
        <span>
          <span className={segmentCounts.selected > 0 ? 'text-amber-300' : 'text-gray-600'}>
            {segmentCounts.selected}
          </span>{' '}
          selected
        </span>
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

      {/* Save */}
      <div className="rounded-lg bg-gray-900/85 backdrop-blur-sm border border-gray-700 p-1 flex items-center gap-2">
        <button
          onClick={onSave}
          disabled={saving || !dirty}
          className={`
            px-3 py-1.5 rounded text-xs font-semibold transition-colors
            ${dirty && !saving
              ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
              : 'bg-gray-800 text-gray-500 cursor-not-allowed'}
          `}
        >
          {saving ? 'Saving…' : '💾 Save & Re-export DXF'}
        </button>
        <span className="text-[11px] text-gray-500 pr-2">
          {dirty
            ? 'unsaved changes'
            : lastSavedVersion !== null
              ? `saved v${lastSavedVersion}`
              : 'pristine'}
        </span>
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
  variant?: 'default' | 'danger';
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={hint}
      className={`px-2.5 py-1 rounded text-xs font-medium transition-colors
        ${disabled
          ? 'text-gray-700 cursor-not-allowed'
          : variant === 'danger'
            ? 'text-rose-300 hover:bg-rose-950/40'
            : 'text-gray-300 hover:bg-gray-800'}`}
    >
      {label}
    </button>
  );
}

function Divider() {
  return <span className="w-px bg-gray-700 my-1" />;
}
