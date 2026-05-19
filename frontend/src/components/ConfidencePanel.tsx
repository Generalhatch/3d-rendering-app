import { useState } from 'react';
import { useJobStore } from '../state/jobStore';
import { api } from '../api/client';
import type { ReprocessParams } from '../api/client';

interface ConfidencePanelProps {
  /** Called when the user triggers re-processing. App.tsx subscribes to SSE + reloads scene. */
  onReprocess?: (params: ReprocessParams) => void;
}

export function ConfidencePanel({ onReprocess }: ConfidencePanelProps) {
  const {
    fullResult, phase, jobId,
    showFixtures, showRooms, toggleFixtures, toggleRooms,
    scanOpacity, setScanOpacity,
    roomOpacity, setRoomOpacity,
    aiReview, aiReviewLoading, setAiReview, setAiReviewLoading,
    setPhase,
    floorCandidates, isReprocessing,
  } = useJobStore();

  const [showAiSection, setShowAiSection] = useState(false);
  const [showReprocessSection, setShowReprocessSection] = useState(false);
  // Re-process parameter state (MJ2 + MJ3)
  const [selectedFloorZ, setSelectedFloorZ] = useState<number | null>(null);
  const [bandLow, setBandLow] = useState(0.75);
  const [bandHigh, setBandHigh] = useState(1.80);
  const [minWallLength, setMinWallLength] = useState(2.0);
  const [houghThreshold, setHoughThreshold] = useState(35);

  if (phase !== 'aligned' && phase !== 'approved') return null;

  const result = fullResult;
  if (!result) return null;

  const a = result.alignment;
  if (!a) return null;

  const confPct = a.confidence_pct ?? 0;
  const confColor =
    confPct >= 90 ? 'text-green-400' :
    confPct >= 70 ? 'text-yellow-400' :
    'text-red-400';
  const confBg =
    confPct >= 90 ? 'bg-green-900/30 border-green-700' :
    confPct >= 70 ? 'bg-yellow-900/30 border-yellow-700' :
    'bg-red-900/30 border-red-700';

  const handleAiReview = async () => {
    if (!jobId) return;
    setAiReviewLoading(true);
    try {
      const { ai_review } = await api.requestAiReview(jobId);
      setAiReview(ai_review);
    } catch (e) {
      setAiReview({ summary: 'AI review failed.', concerns: [], model_confidence: null });
    } finally {
      setAiReviewLoading(false);
    }
  };

  const handleApprove = async () => {
    if (!jobId) return;
    await api.approveJob(jobId);
    setPhase('approved');
  };

  const handleReprocess = () => {
    if (!jobId || !onReprocess) return;
    onReprocess({
      floor_z: selectedFloorZ ?? undefined,
      band_low_m: bandLow,
      band_high_m: bandHigh,
      min_wall_length_m: minWallLength,
      hough_threshold: houghThreshold,
    });
  };

  const scanOnly = result.scan_only ?? a.mode === 'scan_only';

  return (
    <div className="space-y-4">
      {/* Confidence badge */}
      <div className={`rounded-xl border p-4 ${scanOnly ? 'bg-blue-950/30 border-blue-700' : confBg}`}>
        <div className="flex items-start justify-between">
          <div>
            {scanOnly ? (
              <>
                <div className="text-lg font-bold text-blue-300">Scan-Only Mode</div>
                <div className="text-xs text-gray-400 mt-0.5">Floor plan generated from scan</div>
              </>
            ) : (
              <>
                <div className={`text-3xl font-bold ${confColor}`}>{confPct}%</div>
                <div className="text-xs text-gray-400 mt-0.5">Alignment Confidence</div>
              </>
            )}
          </div>
          <div className="text-right text-xs text-gray-400 space-y-1">
            <div>{result.rooms?.length ?? 0} rooms</div>
            <div>{result.fixtures?.length ?? 0} fixtures</div>
            {result.num_scans && result.num_scans > 1 && (
              <div className="text-rose-300">{result.num_scans} scans merged</div>
            )}
            {!scanOnly && (
              <div>{(a.residual_rmse_mm ?? 0).toFixed(1)} mm residual</div>
            )}
          </div>
        </div>

        {!scanOnly && confPct < 90 && (
          <div className="mt-3 rounded-lg bg-yellow-900/30 border border-yellow-700 px-3 py-2 text-xs text-yellow-300">
            ⚠ Manual review recommended
          </div>
        )}
      </div>

      {/* View controls */}
      <div className="rounded-xl border border-gray-700 bg-gray-800/40 p-4 space-y-3">
        <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider">View Controls</div>

        <SliderRow label="Scan opacity" value={scanOpacity} onChange={setScanOpacity} />
        <SliderRow label="Room opacity" value={roomOpacity} onChange={setRoomOpacity} />

        <div className="flex gap-2 pt-1">
          <ToggleButton active={showRooms} onClick={toggleRooms} label="Rooms" />
          <ToggleButton active={showFixtures} onClick={toggleFixtures} label="Fixtures" />
        </div>
      </div>

      {/* Export */}
      {jobId && (
        <div className="rounded-xl border border-gray-700 bg-gray-800/40 p-4 space-y-2">
          <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Export</div>
          <div className="flex gap-2">
            <a
              href={api.exportJsonUrl(jobId)}
              download="aligned.json"
              className="flex-1 text-center py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-xs font-medium transition-colors"
            >
              aligned.json
            </a>
            <a
              href={api.exportDxfUrl(jobId)}
              download="aligned.dxf"
              className="flex-1 text-center py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-xs font-medium transition-colors"
            >
              aligned.dxf
            </a>
          </div>
        </div>
      )}

      {/* AI Review */}
      <div className="rounded-xl border border-gray-700 bg-gray-800/40 overflow-hidden">
        <button
          onClick={() => setShowAiSection((v) => !v)}
          className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-gray-400 uppercase tracking-wider hover:bg-gray-700/30 transition-colors"
        >
          <span>AI Review</span>
          <span>{showAiSection ? '▲' : '▼'}</span>
        </button>
        {showAiSection && (
          <div className="px-4 pb-4 space-y-3">
            {/* Overlay image — gives the user visual context for what the AI reviews */}
            {jobId && (
              <img
                src={api.getOverlayUrl(jobId)}
                alt="Scan-to-plan overlay"
                className="w-full rounded-lg border border-gray-700 bg-gray-900"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
            )}
            {!aiReview ? (
              <button
                onClick={handleAiReview}
                disabled={aiReviewLoading}
                className="w-full py-2 rounded-lg bg-purple-700 hover:bg-purple-600 disabled:opacity-50 text-sm font-medium transition-colors"
              >
                {aiReviewLoading ? 'Asking AI…' : 'Request AI Review'}
              </button>
            ) : (
              <div className="space-y-2 text-xs">
                <p className="text-gray-200">{aiReview.summary}</p>
                {aiReview.concerns.length > 0 && (
                  <ul className="list-disc pl-4 space-y-1 text-yellow-300">
                    {aiReview.concerns.map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                )}
                {aiReview.model_confidence !== null && (
                  <p className="text-gray-400">
                    Model confidence: {Math.round((aiReview.model_confidence ?? 0) * 100)}%
                  </p>
                )}
                <button
                  onClick={handleAiReview}
                  disabled={aiReviewLoading}
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Re-run
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Re-process Settings — MJ2 (floor picker) + MJ3 (parameter tuning) */}
      {onReprocess && (
        <div className="rounded-xl border border-gray-700 bg-gray-800/40 overflow-hidden">
          <button
            onClick={() => setShowReprocessSection((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-gray-400 uppercase tracking-wider hover:bg-gray-700/30 transition-colors"
          >
            <span>Re-process Settings</span>
            <span>{showReprocessSection ? '▲' : '▼'}</span>
          </button>

          {showReprocessSection && (
            <div className="px-4 pb-4 space-y-4">
              {/* Floor selector — only shown when multiple levels detected */}
              {floorCandidates.length > 1 && (
                <div className="space-y-2">
                  <div className="text-xs font-medium text-gray-400">Floor Level</div>
                  <div className="space-y-1">
                    {floorCandidates.map((fc, i) => {
                      const isSelected = selectedFloorZ === fc.floor_z;
                      const isActive = selectedFloorZ === null && i === 0;
                      return (
                        <button
                          key={fc.floor_z}
                          onClick={() => setSelectedFloorZ(isActive && selectedFloorZ === null ? fc.floor_z : (isSelected ? null : fc.floor_z))}
                          className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs transition-colors ${
                            isSelected || (i === 0 && selectedFloorZ === null)
                              ? 'bg-blue-700/50 border border-blue-600 text-blue-200'
                              : 'bg-gray-700/40 border border-gray-600 text-gray-300 hover:bg-gray-600/40'
                          }`}
                        >
                          <span className="font-medium">
                            {i === 0 ? 'Ground Floor' : `Level ${i + 1}`}
                          </span>
                          <span className="text-gray-400">
                            z = {fc.floor_z.toFixed(2)} m
                            {i === 0 && selectedFloorZ === null && (
                              <span className="ml-1 text-blue-400">(current)</span>
                            )}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Wall band parameters */}
              <div className="space-y-2">
                <div className="text-xs font-medium text-gray-400">Wall Band Height</div>
                <SliderRow
                  label={`Band low: ${bandLow.toFixed(2)} m`}
                  value={bandLow}
                  min={0.1} max={1.5} step={0.05}
                  onChange={setBandLow}
                  showPct={false}
                />
                <SliderRow
                  label={`Band high: ${bandHigh.toFixed(2)} m`}
                  value={bandHigh}
                  min={1.0} max={3.0} step={0.05}
                  onChange={setBandHigh}
                  showPct={false}
                />
              </div>

              {/* Hough / plan parameters */}
              <div className="space-y-2">
                <div className="text-xs font-medium text-gray-400">Plan Generation</div>
                <SliderRow
                  label={`Min wall length: ${minWallLength.toFixed(1)} m`}
                  value={minWallLength}
                  min={0.5} max={6.0} step={0.5}
                  onChange={setMinWallLength}
                  showPct={false}
                />
                <SliderRow
                  label={`Hough threshold: ${houghThreshold}`}
                  value={houghThreshold}
                  min={10} max={80} step={5}
                  onChange={(v) => setHoughThreshold(Math.round(v))}
                  showPct={false}
                />
              </div>

              <button
                onClick={handleReprocess}
                disabled={isReprocessing}
                className="w-full py-2 rounded-lg bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium transition-colors"
              >
                {isReprocessing ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-3.5 h-3.5 rounded-full border-2 border-indigo-300 border-t-transparent animate-spin" />
                    Re-processing…
                  </span>
                ) : (
                  '↻ Re-run Processing'
                )}
              </button>

              {isReprocessing && (
                <p className="text-xs text-center text-gray-500">
                  Processing in background — progress shown in the log above
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Approve */}
      {phase !== 'approved' && (
        <button
          onClick={handleApprove}
          className="w-full py-3 rounded-lg bg-green-700 hover:bg-green-600 font-semibold text-sm transition-colors"
        >
          ✓ Approve Alignment
        </button>
      )}
      {phase === 'approved' && (
        <div className="w-full py-3 rounded-lg bg-green-900/40 border border-green-700 text-center font-semibold text-sm text-green-400">
          ✓ Approved
        </div>
      )}
    </div>
  );
}

function SliderRow({
  label, value, onChange,
  min = 0, max = 1, step = 0.05, showPct = true,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  showPct?: boolean;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>{label}</span>
        {showPct && <span>{Math.round(value * 100)}%</span>}
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-blue-500"
      />
    </div>
  );
}

function ToggleButton({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`
        flex-1 py-1.5 rounded-lg text-xs font-medium transition-colors
        ${active ? 'bg-blue-700 text-white' : 'bg-gray-700 text-gray-400'}
      `}
    >
      {label}
    </button>
  );
}
