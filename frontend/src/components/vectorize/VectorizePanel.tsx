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
import {
  vectorizeApi,
  type DetectorName,
  type FloorGeometryRoom,
  type MeasurementReportPayload,
  type PipelineWarning,
  type RoomConfidence,
  type SheetOverrides,
  type SheetStyle,
  type VectorizeParams,
} from '../../api/vectorize';

const VALID_EXTENSIONS = ['.las', '.laz', '.ply', '.e57'];

/** Compact Upload → BOMA checklist driven by existing job/editor state. */
function WorkflowChecklist({
  phase,
  jobId,
  hasRaster,
  wallsCount,
  hasFloorGeometry,
  hasMeasurementReport,
  hasDxf,
  variant = 'default',
}: {
  phase: string;
  jobId: string | null;
  hasRaster: boolean;
  wallsCount: number;
  hasFloorGeometry: boolean;
  hasMeasurementReport: boolean;
  hasDxf: boolean;
  variant?: 'default' | 'sheet';
}) {
  const uploaded = Boolean(jobId) || phase === 'uploading' || phase === 'processing' || phase === 'complete';
  const viewing = (hasRaster || phase === 'complete') && phase !== 'idle';
  const wallsReady = phase === 'complete' && wallsCount > 0;
  const gapsClosed = hasFloorGeometry;
  const bomaReady = hasMeasurementReport;
  const downloadReady = bomaReady && (hasDxf || hasMeasurementReport);

  const display = [
    { id: 'upload', label: 'Upload', done: uploaded && phase !== 'idle' },
    { id: 'view', label: 'View scan', done: viewing && (phase === 'complete' || phase === 'processing') },
    { id: 'walls', label: 'Generate Walls', done: wallsReady },
    { id: 'gaps', label: 'Close gaps', done: gapsClosed },
    { id: 'boma', label: 'Generate BOMA', done: bomaReady },
    { id: 'download', label: 'Download', done: downloadReady },
  ];
  // Cascade: later completion implies earlier steps are done.
  if (wallsReady) {
    display[0].done = true;
    display[1].done = true;
  }
  if (gapsClosed) {
    display[0].done = true;
    display[1].done = true;
    display[2].done = true;
  }
  if (bomaReady) {
    for (const s of display) {
      if (s.id !== 'download') s.done = true;
    }
  }
  const firstOpen = display.findIndex((s) => !s.done);

  const box =
    variant === 'sheet'
      ? 'rounded-lg border border-gray-200 bg-white p-3 space-y-2'
      : 'rounded-lg border border-gray-700 bg-gray-800 p-3 space-y-2';
  const titleCls =
    variant === 'sheet'
      ? 'text-[11px] uppercase tracking-wider text-gray-500 font-semibold'
      : 'text-[11px] uppercase tracking-wider text-gray-300 font-semibold';
  const doneCls = variant === 'sheet' ? 'text-emerald-700' : 'text-emerald-400';
  const curCls = variant === 'sheet' ? 'text-amber-800 font-semibold' : 'text-amber-300 font-semibold';
  const todoCls = variant === 'sheet' ? 'text-gray-400' : 'text-gray-400';

  return (
    <section className={box}>
      <h3 className={titleCls}>Path to BOMA</h3>
      <ol className="space-y-1">
        {display.map((s, i) => {
          const current = firstOpen === i;
          return (
            <li
              key={s.id}
              className={`flex items-center gap-2 text-xs ${
                s.done ? doneCls : current ? curCls : todoCls
              }`}
            >
              <span className="w-4 text-center font-mono text-[10px] opacity-70">
                {s.done ? '✓' : current ? '→' : String(i + 1)}
              </span>
              <span>{s.label}</span>
            </li>
          );
        })}
      </ol>
      {!gapsClosed && wallsReady && (
        <p className={`text-[11px] leading-snug ${variant === 'sheet' ? 'text-amber-900/80' : 'text-amber-200/80'}`}>
          Green walls + scan underlay are the trusted draft. Close open ends
          (coral markers), then Generate BOMA. Exterior shell is optional until rooms close.
        </p>
      )}
    </section>
  );
}

interface PanelProps {
  /** ``sheet`` variant is used in the slide-over deliver sidebar. */
  variant?: 'default' | 'sheet';
}

export function VectorizePanel({ variant = 'default' }: PanelProps) {
  const {
    phase, jobId, scanFile, params, job, logs, overallProgress,
    sidebarTab, setSidebarTab,
    setPhase, setJobId, setScanFile, patchParams, setParams,
    setJob, appendLog, setOverallProgress, clearLogs, reset, bumpSheetVersion,
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
      // Map upload bytes to the first 5 % of the overall bar. The backend's
      // first real emit ("Loading scan…") lands at 0.05, so the handoff is
      // seamless — bar never goes backwards.
      const created = await vectorizeApi.create(scanFile, params, (fraction) => {
        setOverallProgress(fraction * 0.05);
      });
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
  }, [scanFile, params, setPhase, setJobId, appendLog, setOverallProgress, clearLogs]);

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

  const startNew = useCallback(() => {
    // If a job is mid-flight or there are unsaved results visible, confirm
    // before nuking — otherwise just reset silently.
    const hasSomething = jobId !== null || phase !== 'idle' || scanFile !== null;
    const isRunning = phase === 'uploading' || phase === 'processing';
    if (hasSomething && isRunning) {
      const ok = window.confirm(
        'A job is currently running. Clear it and start over? The backend job will keep running but you will lose progress in this view.',
      );
      if (!ok) return;
    }
    sseHandle.current?.close();
    sseHandle.current = null;
    reset();
  }, [jobId, phase, scanFile, reset]);

  return (
    <div className={`flex flex-col gap-5 p-5 ${variant === 'sheet' ? 'text-gray-800' : 'text-gray-100'}`}>
      <div className="flex items-center justify-between gap-3">
        <div>
          <h1 className={`text-lg font-semibold tracking-tight ${variant === 'sheet' ? 'text-gray-900' : 'text-white'}`}>
            {variant === 'sheet' ? 'Deliver' : 'Vectorize'}
          </h1>
          <p className={`text-xs mt-0.5 ${variant === 'sheet' ? 'text-gray-600' : 'text-gray-400'}`}>
            {variant === 'sheet' ? 'Sheet metadata, numbering, export' : 'Scan → CAD line geometry (DXF)'}
          </p>
        </div>
        {variant === 'default' && (phase !== 'idle' || jobId !== null || scanFile !== null) && (
          <button
            onClick={startNew}
            title="Clear everything and start a new submission"
            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md border border-gray-600 bg-gray-800 hover:bg-gray-700 hover:border-gray-500 text-xs font-medium text-gray-100 transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {phase === 'complete' && jobId && (
        <SidebarTabBar tab={sidebarTab} onTab={setSidebarTab} variant={variant} />
      )}

      {(phase === 'processing' || phase === 'complete' || phase === 'uploading') && (
        <WorkflowChecklist
          phase={phase}
          jobId={jobId}
          hasRaster={Boolean(job?.has_raster)}
          wallsCount={job?.metrics?.segments_after_regularize ?? 0}
          hasFloorGeometry={Boolean(job?.has_floor_geometry)}
          hasMeasurementReport={Boolean(job?.has_measurement_report)}
          hasDxf={Boolean(job?.has_dxf)}
          variant={variant}
        />
      )}

      {phase === 'complete' && jobId && sidebarTab === 'deliver' ? (
        <SheetDeliverPanel jobId={jobId} job={job} onSaved={bumpSheetVersion} />
      ) : (
        <>
      {/* Upload zone — only while idle / failed before a job exists */}
      {(phase === 'idle' || (phase === 'failed' && !jobId)) && (
        <VectorizeUpload
          file={scanFile}
          onFile={setScanFile}
          disabled={phase !== 'idle'}
        />
      )}

      {/* Params — hidden in deliver tab on completed jobs */}
      {!(phase === 'complete' && sidebarTab === 'deliver') && (
        <ParamsPanel
          params={params}
          onPatch={patchParams}
          onReset={() => setParams({ ...params, elevation_m: null })}
          disabled={phase === 'uploading' || phase === 'processing'}
        />
      )}

      {/* Action button */}
      {!jobId && (
        <button
          onClick={submit}
          disabled={!scanFile || phase === 'uploading'}
          className={`
            w-full py-3 rounded-lg font-semibold text-sm transition-all
            ${scanFile
              ? 'bg-emerald-700 hover:bg-emerald-600 text-white'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'}
          `}
        >
          {phase === 'uploading' ? 'Uploading…' : 'Vectorize'}
        </button>
      )}

      {/* Re-run with new params */}
      {jobId && (phase === 'complete' || phase === 'failed') && sidebarTab === 'process' && (
        <button
          onClick={reprocess}
          className="w-full py-2.5 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white font-semibold text-sm transition-all"
        >
          Re-run with current params
        </button>
      )}

      {/* Progress */}
      {(phase === 'processing' || phase === 'uploading' || logs.length > 0) && (
        <ProgressLog progress={overallProgress} logs={logs} phase={phase} />
      )}

      {/* Result */}
      {phase === 'complete' && job?.metrics && jobId && sidebarTab === 'process' && (
        <ResultPanel jobId={jobId} job={job} />
      )}

      {/* Phase 4: structured pipeline warnings (fallbacks that fired) */}
      {phase === 'complete' && sidebarTab === 'process' && (job?.warnings?.length ?? 0) > 0 && (
        <WarningsPanel warnings={job!.warnings} />
      )}

      {/* Phase 4: per-room confidence */}
      {phase === 'complete' && sidebarTab === 'process' && (job?.room_confidence?.length ?? 0) > 0 && (
        <RoomConfidencePanel rooms={job!.room_confidence} />
      )}
        </>
      )}

      {/* Error */}
      {phase === 'failed' && (
        <div className="rounded-lg bg-rose-950/40 border border-rose-800 px-3 py-2 text-xs text-rose-300">
          {job?.error_message ?? logs[logs.length - 1]?.message ?? 'Pipeline failed — check backend logs'}
        </div>
      )}

      {/* Always-visible bottom clear — only when there's actually something
          to clear. Mirrors the header button but harder to miss. */}
      {variant === 'default' && (jobId !== null || phase !== 'idle' || scanFile !== null) && (
        <button
          onClick={startNew}
          className="w-full py-2.5 rounded-lg border border-gray-600 bg-gray-800 hover:bg-gray-700 hover:border-gray-500 text-sm font-medium text-gray-100 transition-colors"
        >
          Clear and start new
        </button>
      )}
    </div>
  );
}

// ── Deliver / process tabs + sheet deliverables ───────────────────────────────

function SidebarTabBar({
  tab, onTab, variant,
}: {
  tab: 'deliver' | 'process';
  onTab: (t: 'deliver' | 'process') => void;
  variant: 'default' | 'sheet';
}) {
  const base = variant === 'sheet'
    ? 'flex rounded-lg overflow-hidden border border-gray-300 text-xs font-medium bg-white'
    : 'flex rounded-lg overflow-hidden border border-gray-700 text-xs font-medium';
  const active = variant === 'sheet'
    ? 'flex-1 px-3 py-2 bg-gray-800 text-white'
    : 'flex-1 px-3 py-2 bg-emerald-700 text-white';
  const idle = variant === 'sheet'
    ? 'flex-1 px-3 py-2 text-gray-600 hover:bg-gray-50'
    : 'flex-1 px-3 py-2 text-gray-400 hover:bg-gray-800';
  return (
    <div className={base}>
      <button type="button" onClick={() => onTab('deliver')} className={tab === 'deliver' ? active : idle}>
        Deliver
      </button>
      <button type="button" onClick={() => onTab('process')} className={tab === 'process' ? active : idle}>
        Process
      </button>
    </div>
  );
}

interface MetaDraft {
  building_name: string;
  address: string;
  floor_name: string;
  north_angle_deg: number;
}

const EMPTY_META: MetaDraft = {
  building_name: '',
  address: '',
  floor_name: '',
  north_angle_deg: 0,
};

function parseOffsets(text: string): number[] {
  return text
    .split(/[,\s]+/)
    .map((t) => Number(t))
    .filter((v) => Number.isFinite(v));
}

function SheetDeliverPanel({
  jobId,
  job,
  onSaved,
}: {
  jobId: string;
  job: ReturnType<typeof useVectorizeStore.getState>['job'];
  onSaved: () => void;
}) {
  const [rooms, setRooms] = useState<FloorGeometryRoom[]>([]);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [meta, setMeta] = useState<MetaDraft>(EMPTY_META);
  const [gridU, setGridU] = useState('');
  const [gridV, setGridV] = useState('');
  const [gridRotation, setGridRotation] = useState('0');
  const [style, setStyle] = useState<SheetStyle>('stevenson_minimal');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<MeasurementReportPayload | null>(null);
  const [reportTab, setReportTab] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // Floor geometry is only present after rooms close (auto or Generate BOMA).
        // Fetching it unconditionally caused a noisy 404 on incomplete jobs.
        if (job?.has_floor_geometry) {
          const [geo, overrides] = await Promise.all([
            vectorizeApi.getFloorGeometry(jobId),
            vectorizeApi.getSheetOverrides(jobId),
          ]);
          if (cancelled) return;
          setRooms(geo.rooms);
          setLabels(overrides.labels ?? {});
          if (overrides.meta) setMeta({ ...EMPTY_META, ...overrides.meta });
          if (overrides.manual_grid) {
            setGridU(overrides.manual_grid.u_offsets_m.join(', '));
            setGridV(overrides.manual_grid.v_offsets_m.join(', '));
            setGridRotation(String(overrides.manual_grid.rotation_deg));
          }
          if (overrides.style) setStyle(overrides.style);
          setError(null);
        } else {
          const overrides = await vectorizeApi.getSheetOverrides(jobId);
          if (cancelled) return;
          setRooms([]);
          setLabels(overrides.labels ?? {});
          if (overrides.meta) setMeta({ ...EMPTY_META, ...overrides.meta });
          if (overrides.manual_grid) {
            setGridU(overrides.manual_grid.u_offsets_m.join(', '));
            setGridV(overrides.manual_grid.v_offsets_m.join(', '));
            setGridRotation(String(overrides.manual_grid.rotation_deg));
          }
          if (overrides.style) setStyle(overrides.style);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Sheet data unavailable');
      }
    })();
    return () => { cancelled = true; };
  }, [jobId, job?.has_floor_geometry]);

  useEffect(() => {
    if (!job?.has_measurement_report) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await vectorizeApi.getMeasurementReport(jobId);
        if (!cancelled) setReport(data);
      } catch {
        if (!cancelled) setReport(null);
      }
    })();
    return () => { cancelled = true; };
  }, [jobId, job?.has_measurement_report]);

  const handleSave = useCallback(async () => {
    if (saving) return;
    setSaving(true);
    setError(null);
    try {
      const overrides: SheetOverrides = { labels, meta, style };
      const u = parseOffsets(gridU);
      const v = parseOffsets(gridV);
      if (u.length > 0 || v.length > 0) {
        overrides.manual_grid = {
          rotation_deg: Number(gridRotation) || 0,
          u_offsets_m: u,
          v_offsets_m: v,
        };
      } else {
        overrides.manual_grid = null;
      }
      await vectorizeApi.saveSheetOverrides(jobId, overrides);
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  }, [saving, labels, meta, style, gridU, gridV, gridRotation, jobId, onSaved]);

  const suites = rooms.filter((r) => !r.is_common);
  const standard = report?.measurements[reportTab];

  return (
    <div className="flex flex-col gap-4 text-xs">
      {!job?.has_floor_geometry && (
        <section className="rounded-lg border border-amber-300 bg-amber-50 p-3 space-y-1.5">
          <h3 className="text-[11px] uppercase tracking-wider text-amber-800 font-semibold">
            Rooms not closed yet
          </h3>
          <p className="text-amber-900/80 leading-snug">
            The wall network did not enclose any rooms, so BOMA and the floor
            sheet are not available yet.  In the editor, close open wall ends
            (coral markers), then click{' '}
            <span className="font-semibold">Generate BOMA</span>.  Green walls
            + the scan underlay are the trusted draft — the blue exterior shell
            is optional until rooms close.
          </p>
        </section>
      )}

      <section className="rounded-lg border border-gray-200 bg-white p-3 space-y-2">
        <h3 className="text-[11px] uppercase tracking-wider text-gray-500">Downloads</h3>
        <div className="flex flex-col gap-1.5">
          {job?.scan_filename && (
            <a
              href={vectorizeApi.scanUrl(jobId)}
              download={job.scan_filename}
              className="block w-full text-center py-1.5 rounded-md border border-gray-200 hover:bg-gray-50 text-gray-700"
            >
              Original scan ({job.scan_filename})
            </a>
          )}
          {job?.has_dxf && (
            <a
              href={vectorizeApi.dxfUrl(jobId)}
              download
              className="block w-full text-center py-1.5 rounded-md border border-gray-200 hover:bg-gray-50 text-gray-700"
            >
              DXF
            </a>
          )}
          {job?.has_overlay && (
            <a
              href={vectorizeApi.overlayUrl(jobId)}
              download
              className="block w-full text-center py-1.5 rounded-md border border-gray-200 hover:bg-gray-50 text-gray-700"
            >
              Overlay PNG
            </a>
          )}
          {job?.has_raster && (
            <a
              href={vectorizeApi.rasterUrl(jobId)}
              download
              className="block w-full text-center py-1.5 rounded-md border border-gray-200 hover:bg-gray-50 text-gray-700"
            >
              Raster PNG
            </a>
          )}
          {job?.has_floor_geometry && (
            <a
              href={vectorizeApi.floorGeometryDownloadUrl(jobId)}
              download
              className="block w-full text-center py-1.5 rounded-md border border-gray-200 hover:bg-gray-50 text-gray-700"
            >
              Floor geometry JSON
            </a>
          )}
          {job?.has_measurement_report && (
            <>
              <a
                href={vectorizeApi.measurementReportUrl(jobId)}
                download
                className="block w-full text-center py-1.5 rounded-md border border-gray-200 hover:bg-gray-50 text-gray-700"
              >
                BOMA report JSON
              </a>
              <a
                href={vectorizeApi.measurementReportPdfUrl(jobId)}
                download
                className="block w-full text-center py-1.5 rounded-md border border-gray-200 hover:bg-gray-50 text-gray-700"
              >
                BOMA report PDF
              </a>
            </>
          )}
          {job?.has_sheet && (
            <>
              <a
                href={vectorizeApi.sheetUrl(jobId)}
                download
                className="block w-full text-center py-1.5 rounded-md border border-gray-200 hover:bg-gray-50 text-gray-700"
              >
                Sheet SVG
              </a>
              <a
                href={vectorizeApi.sheetPdfUrl(jobId)}
                download
                className="block w-full text-center py-1.5 rounded-md border border-gray-300 hover:bg-gray-50 text-gray-700 font-medium"
              >
                Sheet PDF
              </a>
            </>
          )}
        </div>
      </section>

      <section className="rounded-lg border border-gray-200 bg-white p-3 space-y-2">
        <h3 className="text-[11px] uppercase tracking-wider text-gray-500">Sheet style</h3>
        <div className="flex gap-2">
          {(
            [
              ['stevenson_minimal', 'Minimal'],
              ['full', 'Title block'],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => setStyle(value)}
              className={`flex-1 rounded-md px-2 py-1.5 border text-center ${
                style === value
                  ? 'bg-gray-800 border-gray-800 text-white'
                  : 'bg-white border-gray-200 text-gray-600 hover:border-gray-300'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-gray-200 bg-white p-3 space-y-2">
        <h3 className="text-[11px] uppercase tracking-wider text-gray-500">Building metadata</h3>
        <input
          value={meta.building_name}
          onChange={(e) => setMeta({ ...meta, building_name: e.target.value })}
          placeholder="Building name"
          className="w-full rounded-md bg-white border border-gray-200 px-2 py-1.5 text-gray-800 focus:border-gray-400 outline-none"
        />
        <input
          value={meta.address}
          onChange={(e) => setMeta({ ...meta, address: e.target.value })}
          placeholder="Address (footer centre line)"
          className="w-full rounded-md bg-white border border-gray-200 px-2 py-1.5 text-gray-800 focus:border-gray-400 outline-none"
        />
        <input
          value={meta.floor_name}
          onChange={(e) => setMeta({ ...meta, floor_name: e.target.value })}
          placeholder="Floor (e.g. Level 3)"
          className="w-full rounded-md bg-white border border-gray-200 px-2 py-1.5 text-gray-800 focus:border-gray-400 outline-none"
        />
      </section>

      <section className="rounded-lg border border-gray-200 bg-white p-3 space-y-2">
        <h3 className="text-[11px] uppercase tracking-wider text-gray-500">Suite numbering</h3>
        {suites.length === 0 && <p className="text-gray-500">No suites detected.</p>}
        <div className="flex flex-col gap-1.5 max-h-40 overflow-y-auto">
          {suites.map((room) => (
            <div key={room.id} className="flex items-center gap-2">
              <span className="w-20 truncate text-gray-500" title={room.id}>
                {room.label || room.id.slice(0, 8)}
              </span>
              <input
                value={labels[room.id] ?? ''}
                onChange={(e) => setLabels({ ...labels, [room.id]: e.target.value })}
                placeholder="320"
                className="flex-1 rounded-md bg-white border border-gray-200 px-2 py-1 text-gray-800 focus:border-gray-400 outline-none"
              />
              <span className="text-gray-400 w-14 text-right tabular-nums">
                {room.area_m2.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-gray-500">Bare digits render as &quot;Suite N&quot; on the sheet.</p>
      </section>

      <section className="rounded-lg border border-gray-200 bg-white p-3 space-y-2">
        <h3 className="text-[11px] uppercase tracking-wider text-gray-500">Manual column grid</h3>
        <p className="text-gray-500">Comma-separated offsets in metres (optional).</p>
        <label className="flex items-center gap-2">
          <span className="w-16 text-gray-500">Numbered</span>
          <input
            value={gridU}
            onChange={(e) => setGridU(e.target.value)}
            placeholder="0, 6, 12"
            className="flex-1 rounded-md border border-gray-200 px-2 py-1 focus:border-gray-400 outline-none"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="w-16 text-gray-500">Lettered</span>
          <input
            value={gridV}
            onChange={(e) => setGridV(e.target.value)}
            placeholder="0, 5, 10"
            className="flex-1 rounded-md border border-gray-200 px-2 py-1 focus:border-gray-400 outline-none"
          />
        </label>
      </section>

      {job?.has_measurement_report && (
        <section className="rounded-lg border border-gray-200 bg-white p-3 space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-[11px] uppercase tracking-wider text-gray-500">Measurement report</h3>
            <a
              href={vectorizeApi.measurementReportPdfUrl(jobId)}
              download
              className="text-[11px] text-gray-600 hover:text-gray-900 underline"
            >
              PDF
            </a>
          </div>
          {report ? (
            <>
              <div className="flex flex-wrap gap-1">
                {report.measurements.map((m, i) => (
                  <button
                    key={`${m.standard}-${m.method}`}
                    type="button"
                    onClick={() => setReportTab(i)}
                    className={`px-2 py-0.5 rounded border text-[11px] ${
                      reportTab === i
                        ? 'bg-gray-800 text-white border-gray-800'
                        : 'border-gray-200 text-gray-600'
                    }`}
                  >
                    {m.standard}{m.method ? ` ${m.method}` : ''}
                  </button>
                ))}
              </div>
              {standard && (
                <div className="overflow-x-auto">
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="text-left text-gray-500 border-b border-gray-100">
                        <th className="py-1 pr-2">Suite</th>
                        <th className="py-1 pr-2 text-right">Usable</th>
                        <th className="py-1 text-right">Rentable</th>
                      </tr>
                    </thead>
                    <tbody>
                      {standard.suites.map((s) => (
                        <tr key={s.suite_id} className="border-b border-gray-50">
                          <td className="py-1 pr-2">{s.label}</td>
                          <td className="py-1 pr-2 text-right tabular-nums">
                            {s.usable.value_m2.toFixed(1)} m²
                          </td>
                          <td className="py-1 text-right tabular-nums">
                            {s.rentable.value_m2.toFixed(1)} m²
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="mt-2 text-gray-500 tabular-nums">
                    Floor usable {standard.floor_usable.value_m2.toFixed(1)} m² ·
                    rentable {standard.floor_rentable.value_m2.toFixed(1)} m²
                  </p>
                </div>
              )}
            </>
          ) : (
            <p className="text-gray-500">Loading report…</p>
          )}
        </section>
      )}

      {error && <p className="text-rose-600">{error}</p>}
      <button
        type="button"
        onClick={handleSave}
        disabled={saving || !job?.has_floor_geometry}
        className="rounded-lg bg-gray-800 hover:bg-gray-700 disabled:opacity-50 text-white font-medium py-2.5"
      >
        {saving ? 'Re-rendering…' : 'Save and re-render sheet'}
      </button>
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
        p-6 text-center transition-colors cursor-pointer
        ${dragging
          ? 'border-emerald-400 bg-emerald-900/40'
          : file
            ? 'border-emerald-500 bg-emerald-950/40'
            : 'border-gray-500 bg-gray-800 hover:border-gray-400 hover:bg-gray-800/80'}
        ${disabled ? 'opacity-50 pointer-events-none' : ''}
      `}
    >
      <div className="text-2xl font-light text-gray-300">{file ? 'Scan' : '+'}</div>
      {file ? (
        <div className="w-full">
          <p className="text-sm font-medium text-emerald-200 truncate">{file.name}</p>
          <p className="text-xs text-gray-400">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
          <button
            onClick={(e) => { e.stopPropagation(); onFile(null); }}
            className="mt-1 text-[11px] text-gray-400 hover:text-white underline-offset-2 hover:underline"
          >
            remove
          </button>
        </div>
      ) : (
        <div>
          <p className="text-sm font-medium text-white">Drop a scan to vectorize</p>
          <p className="text-xs text-gray-400 mt-1">.las · .laz · .ply · .e57 — one file</p>
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
    <div className={`rounded-xl border border-gray-700 bg-gray-800 ${disabled ? 'opacity-60 pointer-events-none' : ''}`}>
      <div className="px-3.5 py-2.5 border-b border-gray-700 text-xs font-semibold uppercase tracking-wide text-emerald-400">
        Parameters
      </div>

      <div className="px-3.5 py-4 space-y-4">
        {/* Elevation */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-100 inline-flex items-center">
              Elevation (m)
              <InfoTip>
                <p className="font-semibold text-gray-100 mb-1">What it does</p>
                <p>The height of the horizontal slice through your scan that we look for walls in.</p>
                <p className="mt-2"><span className="text-emerald-300">Leave blank (auto):</span> we detect the floor and slice at floor + 1.6&nbsp;m (shoulder height) — above desks, filing cabinets, and most cubicle dividers, below ducts and ceiling fixtures.</p>
                <p className="mt-1"><span className="text-emerald-300">Set a number:</span> slice at exactly that elevation. Use for multi-storey scans or when auto picks the wrong floor.</p>
              </InfoTip>
            </label>
            {params.elevation_m !== null && (
              <button
                onClick={onReset}
                className="text-[11px] text-gray-400 hover:text-white"
                title="Auto-detect floor + 1.6 m"
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
              placeholder="auto (floor + 1.6)"
              onChange={(e) =>
                onPatch({ elevation_m: e.target.value === '' ? null : parseFloat(e.target.value) })
              }
              className="flex-1 px-2.5 py-2 rounded-md bg-gray-950 border border-gray-600 text-sm text-white placeholder-gray-500 focus:border-emerald-500 focus:outline-none"
            />
          </div>
          <p className="text-xs text-gray-400 leading-snug">
            Leave blank to auto-pick shoulder height above the detected floor.
          </p>
        </div>

        {/* Detector */}
        <div className="space-y-1.5">
          <label className="text-sm font-medium text-gray-100 inline-flex items-center">
            Line detector
            <InfoTip>
              <p className="font-semibold text-gray-100 mb-1">Which algorithm finds the walls</p>
              <p><span className="text-emerald-300">FLD</span> — modern, clean long segments. Best for most architectural scans. <span className="text-gray-400">(default)</span></p>
              <p className="mt-1"><span className="text-emerald-300">Hough</span> — robust classic, fragments long walls into many short pieces. Try if FLD misses obvious walls.</p>
              <p className="mt-1"><span className="text-emerald-300">Both</span> — runs FLD + Hough and merges. Slower; mainly for comparing quality.</p>
            </InfoTip>
          </label>
          <div className="flex gap-1.5">
            {(['fld', 'hough', 'both'] as DetectorName[]).map((d) => (
              <button
                key={d}
                onClick={() => onPatch({ detector: d })}
                className={`flex-1 px-2 py-2 rounded-md text-xs font-semibold transition-colors
                  ${params.detector === d
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'bg-gray-950 text-gray-300 hover:text-white border border-gray-600 hover:border-gray-500'}`}
              >
                {d === 'fld' ? 'FLD' : d === 'hough' ? 'Hough' : 'Both'}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-400 leading-snug">
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
              <p className="mt-2"><span className="text-emerald-300">Low (0.10–0.30&nbsp;m):</span> keeps door jambs, narrow closet returns, scanner stubs. More noise.</p>
              <p className="mt-1"><span className="text-emerald-300">0.40&nbsp;m (default):</span> balanced — keeps short partitions and closet walls, drops most fragment noise.</p>
              <p className="mt-1"><span className="text-emerald-300">High (1.0–5.0&nbsp;m):</span> structural walls only. Very clean DXF, but you'll lose interior partitions and short real walls.</p>
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
        <Toggle
          label="Remove speckle"
          help="Pre-filter isolated scanner noise + furniture stippling before detection."
          checked={params.remove_speckle}
          onChange={(v) => onPatch({ remove_speckle: v })}
          tip={
            <>
              <p className="font-semibold text-gray-100 mb-1">Clean up the raster before detection</p>
              <p>Runs a small morphological OPEN (3-px elliptical kernel) on the raster — any blob smaller than ~3&nbsp;cm is erased. Wipes out scanner lint and the stippling left by chairs, plants, monitors, and other furniture that intersected the slab.</p>
              <p className="mt-2"><span className="text-emerald-300">On (default):</span> dramatically fewer "white-without-overlay" zones; cleaner DXF.</p>
              <p className="mt-1"><span className="text-emerald-300">Off:</span> only useful on very sparse scans where every pixel matters, or when debugging why the detector missed something.</p>
              <p className="mt-1 text-gray-500">Safe for any real wall ≥ 3&nbsp;cm thick at 1&nbsp;cm/px resolution.</p>
            </>
          }
        />
        <Toggle
          label="Multi-elevation slicing"
          help="OR-merge three slabs (elevation ± 0.4 m) for higher wall recall."
          checked={params.multi_elevation}
          onChange={(v) => onPatch({ multi_elevation: v })}
          tip={
            <>
              <p className="font-semibold text-gray-100 mb-1">Catch walls that one slice would miss</p>
              <p>Instead of slicing the cloud at a single height, we slice at <b>three</b> — elevation − 0.4&nbsp;m, elevation, and elevation + 0.4&nbsp;m — then bitwise-OR the rasters before detection. Any pixel that's a wall at <i>any</i> of those heights becomes a wall pixel in the merged image.</p>
              <p className="mt-2"><span className="text-emerald-300">On (default):</span> recovers walls hidden by tall furniture (cubicle dividers, filing cabinets), captures doorway headers, and picks up half-height partitions. Typical recall lift: +10–20 % wall pixels covered.</p>
              <p className="mt-1"><span className="text-emerald-300">Off:</span> classic single-slice. ~10–15 s faster, but you'll miss walls that aren't visible at the chosen height.</p>
              <p className="mt-1 text-gray-500">Open the "Show coverage gaps" overlay in the editor to verify recall after a run.</p>
            </>
          }
        />
        <Toggle
          label="Detect openings (doors)"
          help="Re-slice above the door header and scan each wall for door-shaped gaps."
          checked={params.detect_openings}
          onChange={(v) => onPatch({ detect_openings: v })}
          tip={
            <>
              <p className="font-semibold text-gray-100 mb-1">Promote the deliverable past walls-only</p>
              <p>After regularization, we slice the cloud once more at <b>elevation&nbsp;+&nbsp;0.6&nbsp;m</b> (above the standard 2.03&nbsp;m door header) so closed-door slabs disappear from the raster, then sample every kept wall in 2&nbsp;cm steps. Runs of pixels with no wall support that are <b>0.7–1.2&nbsp;m</b> wide and sit at least 30&nbsp;cm away from a wall endpoint become openings — they land on the <span className="font-mono text-amber-300">OPENINGS</span> layer in the DXF (yellow) and as dashed yellow segments in the editor.</p>
              <p className="mt-2"><span className="text-emerald-300">On (default):</span> recovers both open and closed doors.  Adds ~3–6&nbsp;s to the run.</p>
              <p className="mt-1"><span className="text-emerald-300">Off:</span> wall-only output, identical to pre-Phase-4 behaviour.</p>
              <p className="mt-1 text-gray-500">Operators can still add or edit openings manually in the editor — press <span className="font-mono text-gray-300">O</span> to draw a new opening.</p>
            </>
          }
        />
        <Toggle
          label="Detect columns"
          help="Find isolated square-ish blobs that aren't walls — usually structural columns."
          checked={params.detect_columns}
          onChange={(v) => onPatch({ detect_columns: v })}
          tip={
            <>
              <p className="font-semibold text-gray-100 mb-1">Pick up the posts between the walls</p>
              <p>After subtracting any pixel within ~12&nbsp;cm of a kept wall from the raster, we run connected-components on what's left. Blobs that are <b>20–120&nbsp;cm</b> across, ≤&nbsp;2.5:1 aspect, and at least 55% filled within their bounding box land on the <span className="font-mono text-pink-300">COLUMNS</span> layer as a 4-vertex rectangle footprint.</p>
              <p className="mt-2"><span className="text-emerald-300">On (default):</span> commercial/industrial spaces get their structural posts auto-detected.</p>
              <p className="mt-1"><span className="text-emerald-300">Off:</span> skip the post-pass.  Operator can still draw columns manually with <span className="font-mono text-gray-300">C</span>.</p>
              <p className="mt-1 text-gray-500">Residential scans usually find 0 columns — that's expected and correct.</p>
            </>
          }
        />

        {/* Advanced */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="w-full flex items-center justify-between text-xs font-medium text-gray-300 hover:text-white py-1.5"
          >
            <span>Advanced settings</span>
            <span className="text-gray-400">{showAdvanced ? '▲' : '▼'}</span>
          </button>
          {showAdvanced && (
            <div className="space-y-3 pt-2">
              <SliderRow
                label="Corner join (m)"
                value={params.snapping_distance_m}
                min={0.30} max={1.20} step={0.05}
                onChange={(v) => onPatch({ snapping_distance_m: v })}
                help="How far free wall ends may extend or trim to meet at L/T corners. 0.85 m is the production default."
                tip={
                  <>
                    <p className="font-semibold text-gray-100 mb-1">Corner join / snapping distance</p>
                    <p>
                      After walls are detected, each free end is allowed to grow
                      (or trim) along its own axis until it hits another wall —
                      the Cloud2BIM / BricsCAD-style extend-trim step that closes
                      open corners.
                    </p>
                    <p className="mt-2"><span className="text-emerald-300">0.60&nbsp;m:</span> conservative — fewer false door closes, more open corners.</p>
                    <p className="mt-1"><span className="text-emerald-300">0.85&nbsp;m (default):</span> closes typical 10–80&nbsp;cm undershoot on real scans.</p>
                    <p className="mt-1"><span className="text-emerald-300">1.00–1.20&nbsp;m:</span> aggressive — use when corners still gap; may bridge narrow openings.</p>
                  </>
                }
              />
              <SliderRow
                label="Slab thickness (m)"
                value={params.slab_thickness_m}
                min={0.05} max={1.00} step={0.05}
                onChange={(v) => onPatch({ slab_thickness_m: v })}
                help="How thick a horizontal slice we flatten into an image. 0.10 m is the sweet spot for clean scans."
                tip={
                  <>
                    <p className="font-semibold text-gray-100 mb-1">Slice thickness around the elevation</p>
                    <p>We grab a horizontal slab this tall, centred on the elevation plane, then squash it down into a 2D image for the detector to work on.</p>
                    <p className="mt-2"><span className="text-emerald-300">0.10&nbsp;m (default):</span> crisp cross-section of wall surfaces; least bleed-through from furniture / ducts.</p>
                    <p className="mt-1"><span className="text-emerald-300">Thinner (≤ 0.05&nbsp;m):</span> sparse scans go dashed and short walls get missed.</p>
                    <p className="mt-1"><span className="text-emerald-300">Thicker (0.20–0.30&nbsp;m):</span> denser image, more reliable on noisy/sparse scans, at the cost of rounded corners and fake walls from beams/door tops.</p>
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
    <div className="space-y-1.5">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-100 w-40 flex-shrink-0 inline-flex items-center gap-0.5">
          {label}
          {tip && <InfoTip>{tip}</InfoTip>}
        </span>
        <input
          type="range"
          min={min} max={max} step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="flex-1 accent-emerald-500 h-2"
        />
        <span className="text-sm text-white w-12 text-right font-mono tabular-nums">{value.toFixed(fixed)}</span>
      </div>
      {help && <p className="text-xs text-gray-400 leading-snug">{help}</p>}
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
    <label className="flex items-start gap-2.5 cursor-pointer rounded-md px-1 py-1 -mx-1 hover:bg-gray-700/50">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 rounded border-gray-500 accent-emerald-500"
      />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-gray-100 inline-flex items-center gap-0.5">
          {label}
          {tip && <InfoTip>{tip}</InfoTip>}
        </div>
        {help && <div className="text-xs text-gray-400 leading-snug mt-0.5">{help}</div>}
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
        className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full bg-gray-600 text-[10px] font-bold text-white cursor-help select-none align-middle hover:bg-emerald-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/60"
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
          className="pointer-events-none rounded-md border border-gray-600 bg-gray-950 px-3 py-2.5 text-xs leading-snug text-gray-100 shadow-xl"
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

  const isUploading = phase === 'uploading';
  const isProcessing = phase === 'processing';
  const isRunning = isUploading || isProcessing;
  const recent = useMemo(() => logs.slice(-20), [logs]);

  const headerLabel = isUploading
    ? 'Uploading…'
    : isProcessing
    ? 'Processing…'
    : 'Log';

  // Empty-state copy depends on which stage we're in. During upload the bar is
  // already moving (driven by XHR bytes), so don't say "waiting".
  const emptyLabel = isUploading
    ? `Uploading scan… ${(progress * 100).toFixed(0)}%`
    : isProcessing
    ? 'Starting pipeline…'
    : 'Waiting for progress…';

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 overflow-hidden">
      <div className="px-3 py-2.5 border-b border-gray-700 flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-emerald-400">{headerLabel}</span>
        <span className="text-xs font-mono text-gray-300">{(progress * 100).toFixed(0)}%</span>
      </div>
      <div className="h-1.5 bg-gray-950 relative overflow-hidden">
        <div
          className="h-full bg-emerald-500 transition-all duration-300"
          style={{ width: `${progress * 100}%` }}
        />
        {/* Indeterminate shimmer while we're between bytes-uploaded and the
            first backend emit — covers the brief gap where the bar would
            otherwise look frozen. */}
        {isRunning && progress < 0.05 && (
          <div className="absolute inset-0 pointer-events-none animate-progress-shimmer bg-gradient-to-r from-transparent via-emerald-400/40 to-transparent" />
        )}
      </div>
      <div
        ref={scrollRef}
        className="max-h-48 overflow-y-auto side-panel-scroll px-3 py-2 space-y-1 text-xs font-mono"
      >
        {recent.map((log, i) => (
          <div key={i} className="text-gray-300">
            <span className="text-emerald-400 mr-2">{log.stage}</span>
            {log.message}
          </div>
        ))}
        {recent.length === 0 && (
          <div className="text-gray-500 italic">{emptyLabel}</div>
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
        {typeof m.openings_detected === 'number' && (
          <Stat
            label="Openings"
            value={m.openings_detected}
            hint={m.openings_detected === 0
              ? '(none detected — closed doors at scan time, or none in this area)'
              : '(doors / door-shaped wall gaps)'}
          />
        )}
        {typeof m.columns_detected === 'number' && (
          <Stat
            label="Columns"
            value={m.columns_detected}
            hint={m.columns_detected === 0
              ? '(none detected — typical for residential / un-columned spaces)'
              : '(structural posts, isolated from walls)'}
          />
        )}
        {/* Per-layer coverage strip: only show when the pipeline emits
            coverage for more than one class.  Today only `walls` carries a
            meaningful denominator, so this stays hidden by default but the
            strip is ready as soon as windows/columns get a real metric. */}
        {m.coverage_by_layer && Object.keys(m.coverage_by_layer).length > 1 && (
          <PerLayerCoverageStat coverage={m.coverage_by_layer} />
        )}
        {m.coverage_pct !== null && m.coverage_pct !== undefined && (
          <CoverageStat
            pct={m.coverage_pct}
            radius_m={m.coverage_radius_m ?? null}
            foreground={m.foreground_px ?? null}
            uncovered={m.uncovered_px ?? null}
            multi={(m.elevations_used_m?.length ?? 1) > 1}
          />
        )}
        <Stat
          label="Elevation"
          value={
            m.elevations_used_m && m.elevations_used_m.length > 1
              ? `${m.elevations_used_m.map((e) => e.toFixed(2)).join(' / ')} m`
              : `${m.elevation_m.toFixed(2)} m`
          }
          hint={m.elevations_used_m && m.elevations_used_m.length > 1 ? '(3-slab OR-merged)' : undefined}
        />
        {m.gravity_relevel_applied === true && (
          <Stat
            label="Gravity re-level"
            value={`${(m.gravity_tilt_deg ?? 0).toFixed(1)}°`}
            hint="(scanner tilt corrected before slicing)"
          />
        )}
        {typeof m.ceiling_clearance_m === 'number' && (
          <Stat
            label="Ceiling clearance"
            value={`${m.ceiling_clearance_m.toFixed(2)} m`}
            hint={m.slice_band_adjusted ? '(low — slice bands scaled down)' : undefined}
          />
        )}
        <Stat label="Detector" value={m.detector.toUpperCase()} />
        <Stat label="Raster" value={`${m.raster_width_px}×${m.raster_height_px} px`} hint={`(${m.raster_world_width_m.toFixed(1)}×${m.raster_world_height_m.toFixed(1)} m)`} />
        <Stat label="Points in slab" value={m.points_in_slab.toLocaleString()} hint={`(${m.raw_points.toLocaleString()} total)`} />
        {typeof m.rooms_detected === 'number' && (
          <Stat
            label="Rooms"
            value={m.rooms_detected}
            hint={m.rooms_detected === 0
              ? '(none closed — use editor → Generate BOMA)'
              : undefined}
          />
        )}

        <div className="mt-3 space-y-1.5">
          <a
            href={vectorizeApi.dxfUrl(jobId)}
            download
            className="block w-full text-center py-2.5 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white font-semibold text-sm transition-colors"
          >
            Download DXF
          </a>
          <a
            href={vectorizeApi.scanUrl(jobId)}
            download={job.scan_filename}
            className="block w-full text-center py-2 rounded-lg border border-emerald-800 text-emerald-200 hover:bg-emerald-950/40 text-sm transition-colors"
          >
            Download original scan
          </a>
          {job.has_measurement_report && (
            <a
              href={vectorizeApi.measurementReportPdfUrl(jobId)}
              download
              className="block w-full text-center py-2 rounded-lg border border-emerald-800 text-emerald-200 hover:bg-emerald-950/40 text-sm transition-colors"
            >
              Download BOMA PDF
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Wall-coverage stat with a colour-coded pill.
 *
 * The metric: of every "wall-foreground" pixel in the raster, what fraction
 * lies within COVERAGE_RADIUS_M of a kept segment.  It's a recall *proxy*,
 * not a perfect signal — the denominator includes any non-wall clutter
 * (furniture, ducts) that survived speckle removal, so 100 % is only
 * achievable on completely sterile scans.
 *
 * Threshold logic depends on which slicing mode produced the raster:
 *
 *   single-slice (fewer, cleaner pixels in the denominator):
 *      ≥ 85 % green · 70–85 % amber · < 70 % red
 *
 *   multi-slice (3 slabs OR-merged: ~40 % more pixels — includes more
 *   furniture at the ±0.4 m offsets, so a slightly lower % is normal):
 *      ≥ 70 % green · 55–70 % amber · < 55 % red
 *
 * In both modes, the visual overlay in the editor is the ground truth —
 * toggle "Show coverage gaps" to see whether the uncovered pixels are
 * furniture (ignore) or actually-missed walls (manually add).
 */
/**
 * Per-layer coverage strip — one chip per detected class.  Surfaces the
 * answer to "did the pipeline miss anything on each layer?" at a glance.
 *
 * Note: the per-layer percentage compares each layer's segments to *its own*
 * foreground (walls → wall raster, openings → door-header raster, columns →
 * wall-subtracted residual).  Don't compare across layers — they're measured
 * against different denominators.
 */
function PerLayerCoverageStat({ coverage }: { coverage: Record<string, number> }) {
  const layerLabels: Record<string, string> = {
    walls: 'Walls',
    openings: 'Openings',
    columns: 'Columns',
    windows: 'Windows',
  };
  return (
    <div>
      <div className="text-gray-400 mb-1">Per-layer coverage</div>
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(coverage).map(([layer, pct]) => {
          const pctInt = Math.round(pct * 1000) / 10;
          const tone =
            pct >= 0.70 ? 'text-emerald-200 bg-emerald-900/40 border-emerald-700/60'
            : pct >= 0.50 ? 'text-amber-200 bg-amber-900/40 border-amber-700/60'
            : 'text-rose-200 bg-rose-950/60 border-rose-800/70';
          return (
            <span
              key={layer}
              className={`px-1.5 py-0.5 rounded border text-[10px] font-mono flex items-baseline gap-1 ${tone}`}
              title={`${layerLabels[layer] ?? layer}: ${pctInt.toFixed(1)} % of the layer's foreground pixels are within 8 cm of an emitted segment.`}
            >
              <span className="text-gray-300 normal-case">{layerLabels[layer] ?? layer}</span>
              <span>{pctInt.toFixed(1)}%</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

function CoverageStat({
  pct, radius_m, foreground, uncovered, multi,
}: { pct: number; radius_m: number | null; foreground: number | null; uncovered: number | null; multi: boolean }) {
  const pctInt = Math.round(pct * 1000) / 10;
  const [greenAt, amberAt] = multi ? [0.70, 0.55] : [0.85, 0.70];
  const tone =
    pct >= greenAt ? 'text-emerald-300 bg-emerald-900/40 border-emerald-700/60'
    : pct >= amberAt ? 'text-amber-300 bg-amber-900/40 border-amber-700/60'
    : 'text-rose-300 bg-rose-950/50 border-rose-800/60';
  const hint = uncovered !== null && foreground !== null
    ? `(${uncovered.toLocaleString()} of ${foreground.toLocaleString()} px)`
    : undefined;
  const tooltip = [
    radius_m !== null && `Coverage radius: ${(radius_m * 100).toFixed(0)} cm`,
    multi
      ? 'Multi-slice mode: 70 %+ is healthy. Lower coverage just means the ±0.4 m slabs picked up more furniture — toggle "Show coverage gaps" in the editor to see whether the red blobs are real walls or furniture.'
      : 'Single-slice mode: 85 %+ is healthy. Lower coverage usually means the detector missed walls — toggle "Show coverage gaps" in the editor.',
  ].filter(Boolean).join('\n');
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-gray-400" title={tooltip}>
        Wall coverage
      </span>
      <span className="font-mono flex items-baseline gap-2">
        <span className={`px-1.5 py-0.5 rounded border text-[10px] leading-none ${tone}`}>
          {pctInt.toFixed(1)}%
        </span>
        {hint && <span className="text-gray-600 text-[10px]">{hint}</span>}
      </span>
    </div>
  );
}

/**
 * Structured pipeline warnings — the honest-fallback report (Phase 4).
 *
 * Every entry is a place where the pipeline degraded gracefully instead of
 * failing (ceiling gate skipped, Manhattan snap bailed, …).  Zero warnings =
 * clean run = this panel doesn't render at all.
 */
function WarningsPanel({ warnings }: { warnings: PipelineWarning[] }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="rounded-xl border border-amber-800/70 bg-amber-950/20 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full px-3 py-2 border-b border-amber-800/70 flex items-center justify-between text-xs font-semibold text-amber-300"
      >
        <span>⚠ Pipeline warnings · {warnings.length}</span>
        <span>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="px-3 py-2 space-y-2 text-[11px]">
          {warnings.map((w, i) => (
            <div key={i} className="flex gap-2">
              <span
                className="flex-shrink-0 px-1.5 py-0.5 h-fit rounded border border-amber-700/60 bg-amber-900/40 text-amber-200 font-mono text-[10px]"
                title={`Stage: ${w.stage}`}
              >
                {w.code}
              </span>
              <span className="text-gray-300 leading-snug">{w.message}</span>
            </div>
          ))}
          <p className="text-gray-400 pt-1">
            Each warning is a fallback that fired during the run — the output is
            still usable, but these areas deserve a closer look in the editor.
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Per-room confidence list (Phase 4).
 *
 * Trust score per inferred room from boundary coverage (how much of the room
 * outline is supported by observed scan data) and snap correction (how far
 * topology moved the corners).  Flagged rooms sort first so the operator's
 * eye lands on what needs review.
 */
function RoomConfidencePanel({ rooms }: { rooms: RoomConfidence[] }) {
  const [open, setOpen] = useState(true);
  const nFlagged = rooms.filter((r) => r.flagged).length;
  const sorted = useMemo(
    () => [...rooms].sort((a, b) => Number(b.flagged) - Number(a.flagged) || a.confidence - b.confidence),
    [rooms],
  );
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full px-3 py-2.5 border-b border-gray-700 flex items-center justify-between text-xs font-semibold"
      >
        <span className={nFlagged > 0 ? 'text-amber-300' : 'text-emerald-400'}>
          Room confidence · {rooms.length} room{rooms.length === 1 ? '' : 's'}
          {nFlagged > 0 && ` · ${nFlagged} flagged`}
        </span>
        <span className="text-gray-400">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="px-3 py-2 space-y-1.5 text-[11px]">
          {sorted.map((r) => {
            const pct = Math.round(r.confidence * 100);
            const tone = r.flagged
              ? 'border-rose-800/70 bg-rose-950/40'
              : r.confidence >= 0.85
                ? 'border-emerald-800/60 bg-emerald-950/20'
                : 'border-amber-800/60 bg-amber-950/20';
            const barTone = r.flagged
              ? 'bg-rose-500'
              : r.confidence >= 0.85 ? 'bg-emerald-500' : 'bg-amber-500';
            return (
              <div
                key={r.room_index}
                className={`rounded-lg border px-2 py-1.5 ${tone}`}
                title={
                  `Boundary coverage: ${(r.boundary_coverage * 100).toFixed(0)} % of the room outline is supported by scan data.\n` +
                  `Snap correction: corners moved ${(r.snap_correction_m * 100).toFixed(1)} cm on average during topology cleanup.`
                }
              >
                <div className="flex items-baseline justify-between">
                  <span className="text-gray-200 font-medium">
                    Room {r.room_index + 1}
                    {r.flagged && <span className="ml-1.5 text-rose-300">⚑ review</span>}
                  </span>
                  <span className="font-mono text-gray-300">
                    {r.area_m2.toFixed(1)} m² · {pct}%
                  </span>
                </div>
                <div className="mt-1 h-1 rounded bg-gray-900 overflow-hidden">
                  <div className={`h-full ${barTone}`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
          <p className="text-gray-400 pt-1">
            Confidence combines boundary coverage (scan support along the room
            outline) and how far corners were moved during cleanup. Flagged
            rooms should be verified in the editor before measuring.
          </p>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-gray-300">{label}</span>
      <span className="text-white font-mono">
        {value} {hint && <span className="text-gray-400">{hint}</span>}
      </span>
    </div>
  );
}
