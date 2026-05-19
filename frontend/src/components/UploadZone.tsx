import { useCallback, useState } from 'react';

interface UploadZoneProps {
  onFilesReady: (plan: File, scan: File) => void;
  disabled?: boolean;
}

export function UploadZone({ onFilesReady, disabled }: UploadZoneProps) {
  const [planFile, setPlanFile] = useState<File | null>(null);
  const [scanFile, setScanFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      const files = Array.from(e.dataTransfer.files);
      _classify(files, setPlanFile, setScanFile);
    },
    [disabled],
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const files = Array.from(e.target.files);
    _classify(files, setPlanFile, setScanFile);
    e.target.value = '';
  };

  const handleAlign = () => {
    if (planFile && scanFile) onFilesReady(planFile, scanFile);
  };

  const ready = !!planFile && !!scanFile && !disabled;

  return (
    <div className="flex flex-col gap-4">
      <div
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        className={`
          relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed
          p-8 text-center transition-colors cursor-pointer
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
        <div className="text-3xl">📁</div>
        <div>
          <p className="text-sm font-medium text-gray-200">Drop files here or click to browse</p>
          <p className="text-xs text-gray-500 mt-1">Plan: .dxf — Scan: .las / .laz / .ply / .e57</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <FileSlot label="Plan" file={planFile} accept=".dxf" onFile={setPlanFile} color="blue" />
        <FileSlot label="Scan" file={scanFile} accept=".las,.laz,.ply,.e57" onFile={setScanFile} color="rose" />
      </div>

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
        {disabled ? 'Processing…' : ready ? '⚡ Align' : 'Drop a plan + scan to begin'}
      </button>
    </div>
  );
}

function FileSlot({
  label, file, accept, onFile, color,
}: {
  label: string;
  file: File | null;
  accept: string;
  onFile: (f: File | null) => void;
  color: 'blue' | 'rose';
}) {
  const colorMap = {
    blue: 'border-blue-700 bg-blue-950/30 text-blue-300',
    rose: 'border-rose-700 bg-rose-950/30 text-rose-300',
  };
  return (
    <div
      className={`
        flex items-center gap-2 rounded-lg border p-2 text-xs
        ${file ? colorMap[color] : 'border-gray-700 bg-gray-800/30 text-gray-500'}
      `}
      onClick={(e) => {
        e.stopPropagation();
        const inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = accept;
        inp.onchange = () => {
          if (inp.files?.[0]) onFile(inp.files[0]);
        };
        inp.click();
      }}
    >
      <span className="text-base">{file ? '✓' : '○'}</span>
      <div className="min-w-0">
        <div className="font-semibold">{label}</div>
        <div className="truncate">{file ? file.name : 'Not selected'}</div>
        {file && (
          <div className="text-gray-500">{(file.size / 1024 / 1024).toFixed(1)} MB</div>
        )}
      </div>
    </div>
  );
}

function _classify(
  files: File[],
  setPlan: (f: File) => void,
  setScan: (f: File) => void,
) {
  for (const f of files) {
    const ext = f.name.split('.').pop()?.toLowerCase() ?? '';
    if (ext === 'dxf') setPlan(f);
    else if (['las', 'laz', 'ply', 'e57'].includes(ext)) setScan(f);
  }
}
