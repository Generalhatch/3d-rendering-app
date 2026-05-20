/**
 * Top-level layout for the Vectorize (scan-to-CAD) surface.
 *
 * Same overall shape as the alignment ``App.tsx`` — main area on the left,
 * sidebar on the right.  The main area swaps between:
 *
 *   - ``VectorizeViewer`` (idle / processing / failed) — static image with overlay-mode toggles
 *   - ``EditorSurface``  (complete)                    — interactive line editor
 */
import { VectorizePanel } from './components/vectorize/VectorizePanel';
import { VectorizeViewer } from './components/vectorize/VectorizeViewer';
import { EditorSurface } from './components/vectorize/editor/EditorSurface';
import { useVectorizeStore } from './state/vectorizeStore';

export default function VectorizeApp() {
  const phase = useVectorizeStore((s) => s.phase);
  const jobId = useVectorizeStore((s) => s.jobId);
  const showEditor = phase === 'complete' && jobId !== null;

  return (
    <div className="grid h-full" style={{ gridTemplateColumns: '1fr 380px' }}>
      <div className="relative overflow-hidden">
        {showEditor && jobId ? <EditorSurface jobId={jobId} /> : <VectorizeViewer />}
      </div>
      <div className="flex flex-col border-l border-gray-700 bg-gray-900 overflow-y-auto side-panel-scroll">
        <VectorizePanel />
      </div>
    </div>
  );
}
