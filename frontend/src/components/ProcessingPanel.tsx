import { useState } from 'react';
import { useJobStore } from '../state/jobStore';

export function ProcessingPanel() {
  const { progressLogs, overallProgress, phase } = useJobStore();
  const [expanded, setExpanded] = useState(true);

  const isProcessing = phase === 'processing' || phase === 'uploading';

  if (phase === 'idle') return null;

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800/50 overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-gray-200 hover:bg-gray-700/30 transition-colors"
      >
        <span className="flex items-center gap-2">
          {isProcessing ? (
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-blue-400 animate-pulse" />
          ) : phase === 'failed' ? (
            <span className="text-red-400">✕</span>
          ) : (
            <span className="text-green-400">✓</span>
          )}
          Processing Log
        </span>
        <span className="text-gray-500">{expanded ? '▲' : '▼'}</span>
      </button>

      {/* Progress bar */}
      <div className="h-1 bg-gray-700">
        <div
          className={`h-full transition-all duration-300 ${
            phase === 'failed' ? 'bg-red-500' : 'bg-blue-500'
          }`}
          style={{ width: `${Math.round(overallProgress * 100)}%` }}
        />
      </div>

      {expanded && (
        <div className="px-4 py-2 max-h-40 overflow-y-auto side-panel-scroll space-y-1">
          {progressLogs.length === 0 ? (
            <p className="text-xs text-gray-500 py-2">Waiting for progress…</p>
          ) : (
            [...progressLogs].reverse().map((log, i) => (
              <div key={i} className="flex items-start gap-2 text-xs">
                <span className={`mt-0.5 font-mono text-gray-500`}>
                  {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </span>
                <span
                  className={
                    log.stage === 'error' ? 'text-red-400' :
                    log.stage === 'complete' ? 'text-green-400' :
                    'text-gray-300'
                  }
                >
                  {log.message}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
