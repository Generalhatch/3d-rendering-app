import { useJobStore } from '../state/jobStore';
import type { JobHistoryEntry } from '../state/jobStore';
import { api } from '../api/client';

interface JobHistoryPanelProps {
  onResumeJob: (jobId: string) => void;
}

export function JobHistoryPanel({ onResumeJob }: JobHistoryPanelProps) {
  const { jobHistory } = useJobStore();

  if (jobHistory.length === 0) return null;

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800/30 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Recent Jobs
        </span>
        <span className="text-xs text-gray-600">{jobHistory.length}</span>
      </div>

      <div className="divide-y divide-gray-700/50 max-h-64 overflow-y-auto side-panel-scroll">
        {jobHistory.map((entry) => (
          <HistoryRow key={entry.jobId} entry={entry} onResume={onResumeJob} />
        ))}
      </div>
    </div>
  );
}

function HistoryRow({
  entry,
  onResume,
}: {
  entry: JobHistoryEntry;
  onResume: (jobId: string) => void;
}) {
  const date = new Date(entry.createdAt);
  const dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  const timeStr = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });

  const statusColor: Record<string, string> = {
    aligned: 'text-green-400',
    approved: 'text-green-300',
    processing: 'text-yellow-400',
    queued: 'text-blue-400',
    failed: 'text-red-400',
  };

  const statusDot: Record<string, string> = {
    aligned: 'bg-green-500',
    approved: 'bg-green-400',
    processing: 'bg-yellow-400 animate-pulse',
    queued: 'bg-blue-400 animate-pulse',
    failed: 'bg-red-500',
  };

  const color = statusColor[entry.status] ?? 'text-gray-400';
  const dot = statusDot[entry.status] ?? 'bg-gray-500';

  return (
    <div className="flex items-center gap-2 px-3 py-2.5 group hover:bg-gray-700/30 transition-colors">
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${dot}`} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className={`text-xs font-medium ${color}`}>
            {entry.status.charAt(0).toUpperCase() + entry.status.slice(1)}
          </span>
          <span className="text-xs text-gray-600">·</span>
          <span className="text-xs text-gray-500">
            {entry.scanOnly ? 'Scan-only' : 'Aligned'}
          </span>
          {entry.scanCount > 1 && (
            <>
              <span className="text-xs text-gray-600">·</span>
              <span className="text-xs text-rose-400">{entry.scanCount} scans</span>
            </>
          )}
        </div>
        <div className="text-xs text-gray-600 mt-0.5">
          {dateStr} {timeStr}
          {entry.elapsed_s != null && (
            <span className="ml-1.5">{entry.elapsed_s.toFixed(0)}s</span>
          )}
        </div>
      </div>

      {(entry.status === 'aligned' || entry.status === 'approved') && (
        <button
          onClick={() => onResume(entry.jobId)}
          className="text-xs text-blue-400 hover:text-blue-300 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
          title="Reload this job"
        >
          Load
        </button>
      )}

      <a
        href={api.exportJsonUrl(entry.jobId)}
        download="aligned.json"
        onClick={(e) => e.stopPropagation()}
        className="text-xs text-gray-600 hover:text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
        title="Download JSON export"
      >
        ↓
      </a>
    </div>
  );
}
