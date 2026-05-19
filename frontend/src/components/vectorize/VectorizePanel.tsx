/**
 * Sidebar UI for the Vectorize tab.
 *
 * Composes upload + params + live progress + result panels in one flow.
 * The companion ``VectorizeViewer`` renders the raster / overlay in the main
 * area; this file owns everything else.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useVectorizeStore } from '../../state/vectorizeStore';
import { vectorizeApi, type DetectorName, type VectorizeParams } from '../../api/vectorize';

const VALID_EXTENSIONS = ['.las', '.laz', '.ply', '.e57'];

export function VectorizePanel() {
  const {
    phase, jobId, scanFile, params, job, logs, overallProgress,
    setPhase, setJobId, setScanFile, patchParams, setParams,
    setJob, appendLog, clearLogs, reset,
  } = useVectorizeStore();

  const sseHandle = useRef<{ close: () => void } | null>(null);

  // Tear down SSE when component unmounts or job changes
  useEffect(() => () => sseHandle.current?.close(), []);

  const submit = useCallback(async () => {
    if (!scanFile) return;
    clearLogs();
    setPhase('uploading');
    try {
      const created = await vectorizeApi.create(scanFile, params);
      setJobId(created.job_id);
      setPhase('processing');

      sseHandle.current?.close();
      sseHandle.current = vectorizeApi.subscribeProgress(
        created.job_id,
        (event) => {
          appendLog(event);
          if (event.stage === 'complete') {
            sseHandle.current?.close();
            void refreshJob(created.job_id);
          } else if (event.stage === 'error') {
            sseHandle.current?.close();
            setPhase('failed');
            void refreshJob(created.job_id);
          }
        },
      );
    } catch (err) {
      setPhase('failed');
      appendLog({
        stage: 'error',
        message: err instanceof Error ? err.message : 'Upload failed',
        progress: 1,
        job_id: 'local',
      });
    }
  }, [scanFile, params, setPhase, setJobId, appendLog, clearLogs]);

  const refreshJob = useCallback(async (id: string) => {
    try {
      const detail = await vectorizeApi.get(id);
      setJob(detail);
      if (detail.status === 'complete') setPhase('complete');
      else if (detail.status === 'failed') setPhase('failed');
    } catch {
      setPhase('failed');
    }
  }, [setJob, setPhase]);

  const reprocess = useCallback(async () => {
    if (!jobId) return;
    clearLogs();
    setPhase('processing');
    try {
      await vectorizeApi.reprocess(jobId, params);
      sseHandle.current?.close();
      sseHandle.current = vectorizeApi.subscribeProgress(
        jobId,
        (event) => {
          appendLog(event);
          if (event.stage === 'complete') {
            sseHandle.current?.close();
            void refreshJob(jobId);
          } else if (event.stage === 'error') {
            sseHandle.current?.close();
            setPhase('failed');
            void refreshJob(jobId);
          }
        },
      );
    } catch (err) {
      setPhase('failed');
      appendLog({
        stage: 'error',
        message: err instanceof Error ? err.message : 'Re-process failed',
        progress: 1,
        job_id: jobId,
      });
    }
  }, [jobId, params, setPhase, appendLog, clearLogs, refreshJob]);

  const startNew = () => {
    sseHandle.current?.close();
    reset();
  };

  return (
    <div className="flex flex-col gap-4 p-5 text-gray-200">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-white">Vectorize</h1>
          <p className="text-xs text-gray-500">Scan → CAD line geometry (DXF)</p>
        </div>
        {phase !== 'idle' && (
          <button
            onClick={startNew}
            className="text-xs text-gray-500 hover:text-gray-300"
          >
            New job
          </button>
        )}
      </div>

      {/* Upload zone — only while idle / failed before a job exists */}
      {(phase === 'idle' || (phase === 'failed' && !jobId)) && (
        <VectorizeUpload
          file={scanFile}
          onFile={setScanFile}
          disabled={phase !== 'idle'}
        />
      )}

      {/* Params — always shown, but disabled while running */}
      <ParamsPanel
        params={params}
        onPatch={patchParams}
        onReset={() => setParams({ ...params, elevation_m: null })}
        disabled={phase === 'uploading' || phase === 'processing'}
      />

      {/* Action button */}
      {!jobId && (
        <button
          onClick={submit}
          disabled={!scanFile || phase === 'uploading'}
          className={`
            w-full py-3 rounded-lg font-semibold text-sm transition-all
            ${scanFile
              ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/30'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'}
          `}
        >
          {phase === 'uploading' ? 'Uploading…' : '⚡ Vectorize'}
        </button>
      )}

      {/* Re-run with new params */}
      {jobId && (phase === 'complete' || phase === 'failed') && (
        <button
          onClick={reprocess}
          className="w-full py-2.5 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white font-semibold text-sm transition-all"
        >
          ↻ Re-run with current params
        </button>
      )}

      {/* Progress */}
      {(phase === 'processing' || phase === 'uploading' || logs.length > 0) && (
        <ProgressLog progress={overallProgress} logs={logs} phase={phase} />
      )}

      {/* Result */}
      {phase === 'complete' && job?.metrics && jobId && (
        <ResultPanel jobId={jobId} job={job} />
      )}

      {/* Error */}
      {phase === 'failed' && (
        <div className="rounded-lg bg-rose-950/40 border border-rose-800 px-3 py-2 text-xs text-rose-300">
          {job?.error_message ?? logs[logs.length - 1]?.message ?? 'Pipeline failed — check backend logs'}
        </div>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function VectorizeUpload({
  file, onFile, disabled,
}: {
  file: File | null;
  onFile: (f: File | null) => void;
  disabled?: boolean;
}) {
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files: File[]) => {
    if (disabled) return;
    const valid = files.find((f) => {
      const ext = '.' + (f.name.split('.').pop()?.toLowerCase() ?? '');
      return VALID_EXTENSIONS.includes(ext);
    });
    if (valid) onFile(valid);
  };

  return (
    <div
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(Array.from(e.dataTransfer.files));
      }}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onClick={() => {
        const inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = VALID_EXTENSIONS.join(',');
        inp.onchange = () => { if (inp.files?.[0]) handleFiles([inp.files[0]]); };
        inp.click();
      }}
      className={`
        flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed
        p-5 text-center transition-colors cursor-pointer
        ${dragging
          ? 'border-emerald-400 bg-emerald-950/30'
          : file
            ? 'border-emerald-700 bg-emerald-950/20'
            : 'border-gray-600 bg-gray-800/40 hover:border-gray-500'}
        ${disabled ? 'opacity-50 pointer-events-none' : ''}
      `}
    >
      <div className="text-2xl">{file ? '✅' : '📁'}</div>
      {file ? (
        <div className="w-full">
          <p className="text-sm font-medium text-emerald-200 truncate">{file.name}</p>
          <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
          <button
            onClick={(e) => { e.stopPropagation(); onFile(null); }}
            className="mt-1 text-[11px] text-gray-500 hover:text-gray-300"
          >
            ✕ remove
          </button>
        </div>
      ) : (
        <div>
          <p className="text-sm font-medium text-gray-200">Drop a scan to vectorize</p>
          <p className="text-xs text-gray-500 mt-0.5">.las · .laz · .ply · .e57 — one file</p>
        </div>
      )}
    </div>
  );
}

function ParamsPanel({
  params, onPatch, onReset, disabled,
}: {
  params: VectorizeParams;
  onPatch: (patch: Partial<VectorizeParams>) => void;
  onReset: () => void;
  disabled?: boolean;
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <div className={`rounded-xl border border-gray-700 bg-gray-800/30 overflow-hidden ${disabled ? 'opacity-60 pointer-events-none' : ''}`}>
      <div className="px-3 py-2 border-b border-gray-700 text-xs font-semibold text-emerald-300">
        Parameters
      </div>

      <div className="px-3 py-3 space-y-3">
        {/* Elevation */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-300">Elevation (m)</label>
            {params.elevation_m !== null && (
              <button
                onClick={onReset}
                className="text-[11px] text-gray-500 hover:text-gray-300"
                title="Auto-detect floor + 1.4 m"
              >
                use auto
              </button>
            )}
          </div>
          <div className="flex items-center gap-2">
            <input
              type="number"
              step={0.05}
              value={params.elevation_m ?? ''}
              placeholder="auto (floor + 1.4)"
              onChange={(e) =>
                onPatch({ elevation_m: e.target.value === '' ? null : parseFloat(e.target.value) })
              }
              className="flex-1 px-2 py-1 rounded bg-gray-900 border border-gray-700 text-xs text-gray-100 placeholder-gray-600"
            />
          </div>
        </div>

        {/* Detector */}
        <div className="space-y-1">
          <label className="text-xs font-medium text-gray-300">Line detector</label>
          <div className="flex gap-1">
            {(['fld', 'hough', 'both'] as DetectorName[]).map((d) => (
              <button
                key={d}
                onClick={() => onPatch({ detector: d })}
                className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-colors
                  ${params.detector === d
                    ? 'bg-emerald-600 text-white'
                    : 'bg-gray-900 text-gray-400 hover:text-gray-200 border border-gray-700'}`}
              >
                {d === 'fld' ? 'FLD' : d === 'hough' ? 'Hough' : 'Both'}
              </button>
            ))}
          </div>
          <p className="text-[11px] text-gray-500">
            {params.detector === 'fld' && 'Fast Line Detector — clean long segments. Recommended.'}
            {params.detector === 'hough' && 'Probabilistic Hough — robust, more fragments.'}
            {params.detector === 'both' && 'Merge both detectors — slower, useful for comparison.'}
          </p>
        </div>

        {/* Min wall length */}
        <SliderRow
          label="Min wall length (m)"
          value={params.min_wall_length_m}
          min={0.10} max={5.0} step={0.10}
          onChange={(v) => onPatch({ min_wall_length_m: v })}
        />

        {/* Toggles */}
        <Toggle
          label="Snap to building axes (Manhattan)"
          help="Filters non-orthogonal noise. Disable on rotated/curved buildings."
          checked={params.manhattan_snap}
          onChange={(v) => onPatch({ manhattan_snap: v })}
        />
        <Toggle
          label="Merge collinear segments"
          help="Fuse near-touching parallel segments into one wall."
          checked={params.merge_collinear}
          onChange={(v) => onPatch({ merge_collinear: v })}
        />

        {/* Advanced */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="w-full flex items-center justify-between text-xs text-gray-500 hover:text-gray-300 py-1"
          >
            <span>Advanced settings</span>
            <span>{showAdvanced ? '▲' : '▼'}</span>
          </button>
          {showAdvanced && (
            <div className="space-y-3 pt-2">
              <SliderRow
                label="Slab thickness (m)"
                value={params.slab_thickness_m}
                min={0.05} max={1.00} step={0.05}
                onChange={(v) => onPatch({ slab_thickness_m: v })}
              />
              <SliderRow
                label="Resolution (m/px)"
                value={params.resolution_m_per_px}
                min={0.002} max={0.05} step={0.002}
                onChange={(v) => onPatch({ resolution_m_per_px: v })}
                fixed={3}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SliderRow({
  label, value, min, max, step, onChange, fixed = 2,
}: {
  label: string;
  value: number;
  min: number; max: number; step: number;
  onChange: (v: number) => void;
  fixed?: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-36 flex-shrink-0">{label}</span>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 accent-emerald-500"
      />
      <span className="text-xs text-gray-300 w-12 text-right font-mono">{value.toFixed(fixed)}</span>
    </div>
  );
}

function Toggle({
  label, help, checked, onChange,
}: {
  label: string; help?: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-2 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 accent-emerald-500"
      />
      <div className="flex-1">
        <div className="text-xs text-gray-200">{label}</div>
        {help && <div className="text-[11px] text-gray-500">{help}</div>}
      </div>
    </label>
  );
}

function ProgressLog({
  progress, logs, phase,
}: {
  progress: number;
  logs: { stage: string; message: string; progress: number; timestamp: number }[];
  phase: string;
}) {
  // Auto-scroll to latest log
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [logs.length]);

  const isRunning = phase === 'processing' || phase === 'uploading';
  const recent = useMemo(() => logs.slice(-20), [logs]);

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800/30 overflow-hidden">
      <div className="px-3 py-2 border-b border-gray-700 flex items-center justify-between">
        <span className="text-xs font-semibold text-emerald-300">
          {isRunning ? 'Processing…' : 'Log'}
        </span>
        <span className="text-xs font-mono text-gray-400">{(progress * 100).toFixed(0)}%</span>
      </div>
      <div className="h-1.5 bg-gray-900">
        <div
          className="h-full bg-emerald-500 transition-all duration-300"
          style={{ width: `${progress * 100}%` }}
        />
      </div>
      <div
        ref={scrollRef}
        className="max-h-48 overflow-y-auto side-panel-scroll px-3 py-2 space-y-1 text-[11px] font-mono"
      >
        {recent.map((log, i) => (
          <div key={i} className="text-gray-400">
            <span className="text-emerald-400 mr-2">{log.stage}</span>
            {log.message}
          </div>
        ))}
        {recent.length === 0 && (
          <div className="text-gray-600 italic">Waiting for progress…</div>
        )}
      </div>
    </div>
  );
}

function ResultPanel({ jobId, job }: { jobId: string; job: NonNullable<ReturnType<typeof useVectorizeStore.getState>['job']> }) {
  const m = job.metrics!;
  return (
    <div className="rounded-xl border border-emerald-800 bg-emerald-950/20 overflow-hidden">
      <div className="px-3 py-2 border-b border-emerald-800 text-xs font-semibold text-emerald-300">
        Result · {m.elapsed_s.toFixed(1)} s
      </div>
      <div className="px-3 py-3 space-y-2 text-xs">
        <Stat label="Wall segments" value={m.segments_after_regularize} hint={`(${m.segments_detected} raw)`} />
        <Stat label="Elevation" value={`${m.elevation_m.toFixed(2)} m`} />
        <Stat label="Detector" value={m.detector.toUpperCase()} />
        <Stat label="Raster" value={`${m.raster_width_px}×${m.raster_height_px} px`} hint={`(${m.raster_world_width_m.toFixed(1)}×${m.raster_world_height_m.toFixed(1)} m)`} />
        <Stat label="Points in slab" value={m.points_in_slab.toLocaleString()} hint={`(${m.raw_points.toLocaleString()} total)`} />

        <a
          href={vectorizeApi.dxfUrl(jobId)}
          download
          className="block mt-3 w-full text-center py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-sm transition-colors"
        >
          ⬇ Download DXF
        </a>
      </div>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-gray-400">{label}</span>
      <span className="text-gray-100 font-mono">
        {value} {hint && <span className="text-gray-500">{hint}</span>}
      </span>
    </div>
  );
}
