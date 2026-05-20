/**
 * Sidebar UI for the Vectorize tab.
 *
 * Composes upload + params + live progress + result panels in one flow.
 * The companion ``VectorizeViewer`` renders the raster / overlay in the main
 * area; this file owns everything else.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
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

  // ── Rehydrate on mount ───────────────────────────────────────────────────
  // If a jobId is persisted (from a previous session or page refresh) but the
  // in-memory phase is still 'idle', fetch the job and resume the right state:
  //   - complete  → flip phase to 'complete' so the editor surface mounts
  //   - failed    → flip phase to 'failed' with the error message
  //   - other     → flip to 'processing' and re-subscribe to SSE
  // Done exactly once per mount; the empty deps array is intentional.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!jobId || phase !== 'idle') return;
    let cancelled = false;
    (async () => {
      try {
        const detail = await vectorizeApi.get(jobId);
        if (cancelled) return;
        setJob(detail);
        if (detail.status === 'complete') {
          setPhase('complete');
        } else if (detail.status === 'failed') {
          setPhase('failed');
        } else {
          setPhase('processing');
          // Re-attach SSE so the still-running job streams progress here too.
          sseHandle.current?.close();
          sseHandle.current = vectorizeApi.subscribeProgress(jobId, (event) => {
            appendLog(event);
            if (event.stage === 'complete') {
              sseHandle.current?.close();
              void refreshJob(jobId);
            } else if (event.stage === 'error') {
              sseHandle.current?.close();
              setPhase('failed');
              void refreshJob(jobId);
            }
          });
        }
      } catch {
        // Persisted job no longer exists on backend (cleanup, etc.) — clear it.
        reset();
      }
    })();
    return () => { cancelled = true; };
    // We deliberately run this once on mount.
  }, []);

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
    <div className={`rounded-xl border border-gray-700 bg-gray-800/30 ${disabled ? 'opacity-60 pointer-events-none' : ''}`}>
      <div className="px-3 py-2 border-b border-gray-700 text-xs font-semibold text-emerald-300">
        Parameters
      </div>

      <div className="px-3 py-3 space-y-3">
        {/* Elevation */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-gray-300 inline-flex items-center">
              Elevation (m)
              <InfoTip>
                <p className="font-semibold text-gray-100 mb-1">What it does</p>
                <p>The height of the horizontal slice through your scan that we look for walls in.</p>
                <p className="mt-2"><span className="text-emerald-300">Leave blank (auto):</span> we detect the floor and slice at floor + 1.4&nbsp;m (chest height) — above furniture, below ceiling fixtures.</p>
                <p className="mt-1"><span className="text-emerald-300">Set a number:</span> slice at exactly that elevation. Use for multi-storey scans or when auto picks the wrong floor.</p>
              </InfoTip>
            </label>
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
          <p className="text-[11px] text-gray-500">
            Leave blank to auto-pick chest height above the detected floor.
          </p>
        </div>

        {/* Detector */}
        <div className="space-y-1">
          <label className="text-xs font-medium text-gray-300 inline-flex items-center">
            Line detector
            <InfoTip>
              <p className="font-semibold text-gray-100 mb-1">Which algorithm finds the walls</p>
              <p><span className="text-emerald-300">FLD</span> — modern, clean long segments. Best for most architectural scans. <span className="text-gray-400">(default)</span></p>
              <p className="mt-1"><span className="text-emerald-300">Hough</span> — robust classic, fragments long walls into many short pieces. Try if FLD misses obvious walls.</p>
              <p className="mt-1"><span className="text-emerald-300">Both</span> — runs FLD + Hough and merges. Slower; mainly for comparing quality.</p>
            </InfoTip>
          </label>
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
          help="Drop any wall shorter than this. Lower = keeps short walls / clutter, higher = structural walls only."
          tip={
            <>
              <p className="font-semibold text-gray-100 mb-1">Shortest wall we'll keep</p>
              <p>Anything shorter than this — in real-world metres — is dropped from the final DXF.</p>
              <p className="mt-2"><span className="text-emerald-300">Low (0.10–0.50&nbsp;m):</span> keeps door jambs, columns, closet returns. More noise.</p>
              <p className="mt-1"><span className="text-emerald-300">Default (~1.4&nbsp;m):</span> balanced — real walls, most furniture-edge noise removed.</p>
              <p className="mt-1"><span className="text-emerald-300">High (2–5&nbsp;m):</span> structural walls only. Very clean, but you'll lose short real walls.</p>
            </>
          }
        />

        {/* Toggles */}
        <Toggle
          label="Snap to building axes (Manhattan)"
          help="Filters non-orthogonal noise. Disable on rotated/curved buildings."
          checked={params.manhattan_snap}
          onChange={(v) => onPatch({ manhattan_snap: v })}
          tip={
            <>
              <p className="font-semibold text-gray-100 mb-1">Force walls onto two axes</p>
              <p>After detection, we find the two dominant wall directions and drop everything that isn't aligned with them — then snap survivors perfectly straight.</p>
              <p className="mt-2"><span className="text-emerald-300">On:</span> output looks "drafted." Kills diagonal noise from furniture, plants, reflections. Best for normal rectangular buildings (even if rotated as a whole).</p>
              <p className="mt-1"><span className="text-emerald-300">Off:</span> every angle is kept. Use for curved walls, organic shapes, or buildings with more than two wall orientations.</p>
            </>
          }
        />
        <Toggle
          label="Merge collinear segments"
          help="Fuse near-touching parallel segments into one wall."
          checked={params.merge_collinear}
          onChange={(v) => onPatch({ merge_collinear: v })}
          tip={
            <>
              <p className="font-semibold text-gray-100 mb-1">Glue broken walls back together</p>
              <p>A long wall the detector broke into 5–10 pieces (because of a doorway or scan gap) becomes a single line. Typical 2–5× drop in segment count.</p>
              <p className="mt-2"><span className="text-emerald-300">On (default):</span> always recommended.</p>
              <p className="mt-1"><span className="text-emerald-300">Off:</span> useful only for QA, to see exactly what the detector returned.</p>
            </>
          }
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
                help="How thick a horizontal slice we flatten into an image. 0.20 m is the sweet spot."
                tip={
                  <>
                    <p className="font-semibold text-gray-100 mb-1">Slice thickness around the elevation</p>
                    <p>We grab a horizontal slab this tall, centred on the elevation plane, then squash it down into a 2D image for the detector to work on.</p>
                    <p className="mt-2"><span className="text-emerald-300">Thinner (≤ 0.10&nbsp;m):</span> crisper corners, but sparse scans go dashed and walls get missed.</p>
                    <p className="mt-1"><span className="text-emerald-300">0.20&nbsp;m (default):</span> the sweet spot.</p>
                    <p className="mt-1"><span className="text-emerald-300">Thicker (≥ 0.30&nbsp;m):</span> dense image, reliable detection, but rounded corners and possible fake walls from beams/door tops.</p>
                  </>
                }
              />
              <SliderRow
                label="Resolution (m/px)"
                value={params.resolution_m_per_px}
                min={0.002} max={0.05} step={0.002}
                onChange={(v) => onPatch({ resolution_m_per_px: v })}
                fixed={3}
                help="How many metres each pixel represents. 0.010 (1 cm/px) is the practical default."
                tip={
                  <>
                    <p className="font-semibold text-gray-100 mb-1">Image resolution</p>
                    <p>How much real-world space one pixel represents.</p>
                    <p className="mt-2"><span className="text-emerald-300">0.005 (5&nbsp;mm/px):</span> high fidelity, preserves thin walls. ~4× larger image, slower, noisier.</p>
                    <p className="mt-1"><span className="text-emerald-300">0.010 (1&nbsp;cm/px, default):</span> production default. Captures anything ≥ 2&nbsp;cm thick.</p>
                    <p className="mt-1"><span className="text-emerald-300">0.050 (5&nbsp;cm/px):</span> very coarse — structural walls only. Good for previews of huge buildings.</p>
                    <p className="mt-2 text-amber-300/90">Lowering this also raises the effective minimum wall length — short walls may disappear.</p>
                  </>
                }
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SliderRow({
  label, value, min, max, step, onChange, fixed = 2, help, tip,
}: {
  label: string;
  value: number;
  min: number; max: number; step: number;
  onChange: (v: number) => void;
  fixed?: number;
  help?: string;
  tip?: ReactNode;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-400 w-36 flex-shrink-0 inline-flex items-center">
          {label}
          {tip && <InfoTip>{tip}</InfoTip>}
        </span>
        <input
          type="range"
          min={min} max={max} step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="flex-1 accent-emerald-500"
        />
        <span className="text-xs text-gray-300 w-12 text-right font-mono">{value.toFixed(fixed)}</span>
      </div>
      {help && <p className="text-[11px] text-gray-500 pl-0">{help}</p>}
    </div>
  );
}

function Toggle({
  label, help, checked, onChange, tip,
}: {
  label: string;
  help?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  tip?: ReactNode;
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
        <div className="text-xs text-gray-200 inline-flex items-center">
          {label}
          {tip && <InfoTip>{tip}</InfoTip>}
        </div>
        {help && <div className="text-[11px] text-gray-500">{help}</div>}
      </div>
    </label>
  );
}

/**
 * Small `?` icon that reveals a tooltip on hover or keyboard focus.
 *
 * The tooltip is rendered via a React portal to ``document.body`` and
 * positioned with ``fixed`` coordinates derived from the trigger's bounding
 * rect. This is necessary because the side panel that hosts these controls
 * uses ``overflow-y-auto`` — which per CSS spec forces ``overflow-x: hidden``
 * — so any in-DOM tooltip extending past the panel edge would get clipped.
 *
 * Behaviour:
 *  - Opens on mouse hover OR keyboard focus.
 *  - Prefers to sit ABOVE the icon (so it doesn't get hidden behind a finger
 *    on touch devices, and so it doesn't push the rest of the panel).
 *  - Horizontally clamped to the viewport with an 8 px margin.
 *  - Width clamps to 16 rem or the viewport, whichever is smaller.
 */
const TOOLTIP_WIDTH_PX = 256;
const TOOLTIP_MARGIN_PX = 8;
const TOOLTIP_GAP_PX = 8;

function InfoTip({ children }: { children: ReactNode }) {
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number; placement: 'top' | 'bottom' }>({
    top: 0, left: 0, placement: 'top',
  });

  // Recompute position whenever the tooltip opens (or the viewport scrolls /
  // resizes while it's open). useLayoutEffect avoids a one-frame flash at
  // (0,0) before the position is applied.
  useLayoutEffect(() => {
    if (!open || !triggerRef.current) return;

    const compute = () => {
      const trigger = triggerRef.current;
      const tooltip = tooltipRef.current;
      if (!trigger) return;
      const rect = trigger.getBoundingClientRect();
      const tooltipHeight = tooltip?.offsetHeight ?? 100;
      const viewportW = window.innerWidth;
      const viewportH = window.innerHeight;

      // Horizontal: anchor the tooltip's right edge to the icon's right edge,
      // then clamp inside the viewport with an 8 px margin on each side.
      let left = rect.right - TOOLTIP_WIDTH_PX;
      const minLeft = TOOLTIP_MARGIN_PX;
      const maxLeft = viewportW - TOOLTIP_WIDTH_PX - TOOLTIP_MARGIN_PX;
      if (left < minLeft) left = minLeft;
      if (left > maxLeft) left = Math.max(minLeft, maxLeft);

      // Vertical: prefer above the icon. If there isn't room above, flip
      // below.
      let placement: 'top' | 'bottom' = 'top';
      let top = rect.top - tooltipHeight - TOOLTIP_GAP_PX;
      if (top < TOOLTIP_MARGIN_PX) {
        placement = 'bottom';
        top = rect.bottom + TOOLTIP_GAP_PX;
        // If even below would overflow, clamp.
        if (top + tooltipHeight > viewportH - TOOLTIP_MARGIN_PX) {
          top = Math.max(TOOLTIP_MARGIN_PX, viewportH - tooltipHeight - TOOLTIP_MARGIN_PX);
        }
      }

      setPos({ top, left, placement });
    };

    compute();
    // Run a second time after a tick so we measure the tooltip's actual
    // height (the first pass might use the fallback 100 px).
    const id = requestAnimationFrame(compute);

    window.addEventListener('scroll', compute, true);
    window.addEventListener('resize', compute);
    return () => {
      cancelAnimationFrame(id);
      window.removeEventListener('scroll', compute, true);
      window.removeEventListener('resize', compute);
    };
  }, [open]);

  // Close on Escape for keyboard users.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  return (
    <>
      <span
        ref={triggerRef}
        tabIndex={0}
        role="button"
        aria-label="More info"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="ml-1 inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-gray-700 text-[9px] font-bold text-gray-300 cursor-help select-none align-middle hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/60"
      >
        ?
      </span>
      {open && createPortal(
        <div
          ref={tooltipRef}
          role="tooltip"
          style={{
            position: 'fixed',
            top: pos.top,
            left: pos.left,
            width: `min(${TOOLTIP_WIDTH_PX}px, calc(100vw - ${TOOLTIP_MARGIN_PX * 2}px))`,
            zIndex: 9999,
          }}
          className="pointer-events-none rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-[11px] leading-snug text-gray-300 shadow-xl"
        >
          {children}
        </div>,
        document.body,
      )}
    </>
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
