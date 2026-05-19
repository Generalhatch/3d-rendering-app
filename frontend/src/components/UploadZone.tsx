import { useCallback, useState } from 'react';

interface UploadZoneProps {
  onFilesReady: (scans: File[], plan: File | null, bandLow: number, bandHigh: number) => void;
  disabled?: boolean;
}

export function UploadZone({ onFilesReady, disabled }: UploadZoneProps) {
  const [planFile, setPlanFile] = useState<File | null>(null);
  const [scanFiles, setScanFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [bandLow, setBandLow] = useState(0.75);
  const [bandHigh, setBandHigh] = useState(1.80);

  const addFiles = useCallback(
    (files: File[]) => {
      if (disabled) return;
      let newPlan: File | null = null;
      const newScans: File[] = [];
      for (const f of files) {
        const ext = f.name.split('.').pop()?.toLowerCase() ?? '';
        if (ext === 'dxf') newPlan = f;
        else if (['las', 'laz', 'ply', 'e57'].includes(ext)) newScans.push(f);
      }
      if (newPlan) setPlanFile(newPlan);
      if (newScans.length > 0) {
        setScanFiles((prev) => {
          // Deduplicate by name
          const existing = new Set(prev.map((f) => f.name));
          const fresh = newScans.filter((f) => !existing.has(f.name));
          return [...prev, ...fresh];
        });
      }
    },
    [disabled],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      addFiles(Array.from(e.dataTransfer.files));
    },
    [addFiles],
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    addFiles(Array.from(e.target.files));
    e.target.value = '';
  };

  const removeScan = (name: string) =>
    setScanFiles((prev) => prev.filter((f) => f.name !== name));

  const handleAlign = () => {
    if (scanFiles.length > 0) onFilesReady(scanFiles, planFile, bandLow, bandHigh);
  };

  const ready = scanFiles.length > 0 && !disabled;

  return (
    <div className="flex flex-col gap-3">
      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        className={`
          flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed
          p-6 text-center transition-colors cursor-pointer
          ${dragging ? 'border-blue-400 bg-blue-950/30' : 'border-gray-600 bg-gray-800/40 hover:border-gray-500'}
          ${disabled ? 'opacity-50 pointer-events-none' : ''}
        `}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          multiple
          accept=".dxf,.las,.laz,.ply,.e57"
          className="hidden"
          onChange={handleFileInput}
        />
        <div className="text-2xl">📁</div>
        <div>
          <p className="text-sm font-medium text-gray-200">Drop files here or click to browse</p>
          <p className="text-xs text-gray-500 mt-0.5">
            Plan: .dxf · Scans: .las / .laz / .ply / .e57 (multiple OK)
          </p>
        </div>
      </div>

      {/* Plan slot — optional */}
      <SingleFileSlot
        label="Plan (DXF) — optional"
        sublabel={planFile ? undefined : "Without a plan, floor layout is generated from the scan"}
        file={planFile}
        accept=".dxf"
        onFile={setPlanFile}
        color="blue"
        icon="📐"
      />

      {/* No-plan info badge */}
      {!planFile && scanFiles.length > 0 && (
        <div className="rounded-lg bg-blue-950/30 border border-blue-800 px-3 py-2 text-xs text-blue-300">
          No DXF? No problem — floor plan will be generated directly from the scan.
        </div>
      )}

      {/* Scan files list */}
      <div className="rounded-xl border border-gray-700 bg-gray-800/30 overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
          <span className="text-xs font-semibold text-rose-300">
            Scans ({scanFiles.length})
          </span>
          <button
            className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            onClick={(e) => {
              e.stopPropagation();
              const inp = document.createElement('input');
              inp.type = 'file';
              inp.multiple = true;
              inp.accept = '.las,.laz,.ply,.e57';
              inp.onchange = () => {
                if (inp.files) addFiles(Array.from(inp.files));
              };
              inp.click();
            }}
          >
            + Add scans
          </button>
        </div>

        {scanFiles.length === 0 ? (
          <div className="px-3 py-3 text-xs text-gray-500">
            No scans yet — drop .laz files here
          </div>
        ) : (
          <div className="divide-y divide-gray-700/50 max-h-48 overflow-y-auto side-panel-scroll">
            {scanFiles.map((f) => (
              <div key={f.name} className="flex items-center gap-2 px-3 py-2 text-xs">
                <span className="text-rose-400 flex-shrink-0">🔴</span>
                <div className="min-w-0 flex-1">
                  <div className="text-gray-200 truncate font-medium">{f.name}</div>
                  <div className="text-gray-500">{(f.size / 1024 / 1024).toFixed(1)} MB</div>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); removeScan(f.name); }}
                  className="text-gray-600 hover:text-gray-300 flex-shrink-0 ml-1"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Scan count badge */}
      {scanFiles.length > 1 && (
        <div className="rounded-lg bg-rose-950/30 border border-rose-800 px-3 py-2 text-xs text-rose-300">
          {scanFiles.length} scans will be merged into one unified floor point cloud
        </div>
      )}

      {/* Advanced settings accordion */}
      <div className="rounded-xl border border-gray-700 bg-gray-800/30 overflow-hidden">
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold text-gray-500 hover:text-gray-300 transition-colors"
        >
          <span>Advanced settings</span>
          <span className="text-gray-600">{showAdvanced ? '▲' : '▼'}</span>
        </button>
        {showAdvanced && (
          <div className="px-3 pb-3 space-y-3 border-t border-gray-700">
            <p className="text-xs text-gray-500 pt-2">
              Wall detection band — vertical range above floor to use for wall extraction.
              Adjust for non-standard ceiling heights or raised floors.
            </p>
            <BandInput
              label="Band low (m)"
              value={bandLow}
              min={0.1}
              max={bandHigh - 0.1}
              step={0.05}
              onChange={setBandLow}
            />
            <BandInput
              label="Band high (m)"
              value={bandHigh}
              min={bandLow + 0.1}
              max={5.0}
              step={0.05}
              onChange={setBandHigh}
            />
          </div>
        )}
      </div>

      {/* Align button */}
      <button
        onClick={handleAlign}
        disabled={!ready}
        className={`
          w-full py-3 rounded-lg font-semibold text-sm transition-all
          ${ready
            ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/30'
            : 'bg-gray-700 text-gray-500 cursor-not-allowed'}
        `}
      >
        {disabled
          ? 'Processing…'
          : !ready
          ? 'Drop scan files to begin'
          : planFile
          ? `⚡ Align${scanFiles.length > 1 ? ` (${scanFiles.length} scans)` : ''}`
          : `🔍 Generate Floor Plan${scanFiles.length > 1 ? ` (${scanFiles.length} scans)` : ''}`}
      </button>
    </div>
  );
}

function SingleFileSlot({
  label, sublabel, file, accept, onFile, color, icon,
}: {
  label: string;
  sublabel?: string;
  file: File | null;
  accept: string;
  onFile: (f: File | null) => void;
  color: 'blue' | 'rose';
  icon: string;
}) {
  const colorMap = {
    blue: 'border-blue-700 bg-blue-950/30 text-blue-300',
    rose: 'border-rose-700 bg-rose-950/30 text-rose-300',
  };
  return (
    <div
      className={`
        flex items-center gap-2 rounded-lg border p-2 text-xs cursor-pointer
        ${file ? colorMap[color] : 'border-gray-700 bg-gray-800/30 text-gray-500'}
      `}
      onClick={(e) => {
        e.stopPropagation();
        const inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = accept;
        inp.onchange = () => { if (inp.files?.[0]) onFile(inp.files[0]); };
        inp.click();
      }}
    >
      <span className="text-base">{icon}</span>
      <div className="min-w-0 flex-1">
        <div className="font-semibold">{label}</div>
        <div className="truncate">
          {file ? file.name : (sublabel ?? 'Not selected')}
        </div>
        {file && <div className="text-gray-500">{(file.size / 1024 / 1024).toFixed(1)} MB</div>}
      </div>
      {file && (
        <button
          onClick={(e) => { e.stopPropagation(); onFile(null); }}
          className="text-gray-500 hover:text-gray-300 ml-1"
        >
          ✕
        </button>
      )}
    </div>
  );
}

function BandInput({
  label, value, min, max, step, onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-28 flex-shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 accent-blue-500"
      />
      <span className="text-xs text-gray-300 w-10 text-right font-mono">{value.toFixed(2)}</span>
    </div>
  );
}
