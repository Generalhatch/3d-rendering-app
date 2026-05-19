/**
 * Top-level layout for the Vectorize (scan-to-CAD) surface.
 *
 * Same overall shape as the alignment ``App.tsx`` — main area on the left,
 * sidebar on the right — but the main area is a 2D image viewer instead of a
 * Three.js scene, and the sidebar contains the vectorize-specific upload /
 * params / progress / result UI.
 */
import { VectorizePanel } from './components/vectorize/VectorizePanel';
import { VectorizeViewer } from './components/vectorize/VectorizeViewer';

export default function VectorizeApp() {
  return (
    <div className="grid h-full" style={{ gridTemplateColumns: '1fr 380px' }}>
      <div className="relative overflow-hidden">
        <VectorizeViewer />
      </div>
      <div className="flex flex-col border-l border-gray-700 bg-gray-900 overflow-y-auto side-panel-scroll">
        <VectorizePanel />
      </div>
    </div>
  );
}
