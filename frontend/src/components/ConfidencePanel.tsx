import { useState } from 'react';
import { useJobStore } from '../state/jobStore';
import { api } from '../api/client';

export function ConfidencePanel() {
  const {
    fullResult, phase, jobId,
    showFixtures, showRooms, toggleFixtures, toggleRooms,
    scanOpacity, setScanOpacity,
    roomOpacity, setRoomOpacity,
    aiReview, aiReviewLoading, setAiReview, setAiReviewLoading,
    setPhase,
  } = useJobStore();

  const [showAiSection, setShowAiSection] = useState(false);

  if (phase !== 'aligned' && phase !== 'approved') return null;

  const result = fullResult;
  if (!result) return null;

  const a = result.alignment;
  const confPct = a.confidence_pct;
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

  return (
    <div className="space-y-4">
      {/* Confidence badge */}
      <div className={`rounded-xl border p-4 ${confBg}`}>
        <div className="flex items-start justify-between">
          <div>
            <div className={`text-3xl font-bold ${confColor}`}>{confPct}%</div>
            <div className="text-xs text-gray-400 mt-0.5">Alignment Confidence</div>
          </div>
          <div className="text-right text-xs text-gray-400 space-y-1">
            <div>{a.residual_rmse_mm.toFixed(1)} mm residual</div>
            <div>{result.rooms.length} rooms</div>
            <div>{result.fixtures.length} fixtures</div>
          </div>
        </div>

        {confPct < 90 && (
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

function SliderRow({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>{label}</span>
        <span>{Math.round(value * 100)}%</span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
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
