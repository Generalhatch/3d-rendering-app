/**
 * Top-level shell that switches between the two product surfaces:
 *
 *   - Align     — 3D scan ↔ existing CAD plan alignment
 *   - Vectorize — raster slice → DXF line geometry (Phase 1 scope)
 *
 * Both surfaces share the same window chrome (a thin top tab strip) but own
 * their own layouts and Zustand stores.  Wiring the switcher at the root
 * means neither App needs to know the other exists.
 */
import { useState } from 'react';
import App from './App';
import VectorizeApp from './VectorizeApp';

type Mode = 'align' | 'vectorize';

const TAB_STORAGE_KEY = 'alignai_active_mode';

function loadInitialMode(): Mode {
  try {
    const stored = localStorage.getItem(TAB_STORAGE_KEY);
    if (stored === 'align' || stored === 'vectorize') return stored;
  } catch {
    // localStorage unavailable
  }
  return 'align';
}

export default function RootApp() {
  const [mode, setMode] = useState<Mode>(loadInitialMode);

  const switchMode = (next: Mode) => {
    setMode(next);
    try {
      localStorage.setItem(TAB_STORAGE_KEY, next);
    } catch {
      // ignore
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-950">
      {/* Tab strip */}
      <div className="flex items-center border-b border-gray-800 bg-gray-950">
        <TabButton
          active={mode === 'align'}
          onClick={() => switchMode('align')}
          label="Align"
          sub="scan ↔ plan"
        />
        <TabButton
          active={mode === 'vectorize'}
          onClick={() => switchMode('vectorize')}
          label="Vectorize"
          sub="scan → DXF"
        />
        <div className="flex-1" />
        <div className="px-4 text-[11px] text-gray-600 font-mono select-none">
          stevenson-vectorize · phase 1
        </div>
      </div>

      {/* Active surface */}
      <div className="flex-1 min-h-0">
        {mode === 'align' ? <App /> : <VectorizeApp />}
      </div>
    </div>
  );
}

function TabButton({
  active, onClick, label, sub,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  sub?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        px-4 py-2 text-sm font-semibold border-b-2 transition-colors
        ${active
          ? 'border-emerald-400 text-white'
          : 'border-transparent text-gray-500 hover:text-gray-300'}
      `}
    >
      {label}
      {sub && <span className="ml-2 text-[10px] font-normal text-gray-600">{sub}</span>}
    </button>
  );
}
