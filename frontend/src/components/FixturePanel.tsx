import { useJobStore } from '../state/jobStore';

export function FixturePanel() {
  const { selectedFixtureId, fixtures, selectFixture } = useJobStore();

  const fixture = fixtures.find((f) => f.id === selectedFixtureId);
  if (!fixture) return null;

  const depthCm = fixture.protrusion_depth_m * 100;
  const depthColor =
    depthCm < 5  ? 'text-teal-400' :
    depthCm < 15 ? 'text-yellow-400' :
    'text-red-400';

  const confPct = Math.round(fixture.confidence * 100);

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800/50 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-sm text-gray-100">Wall Fixture</span>
        <button
          onClick={() => selectFixture(null)}
          className="text-gray-500 hover:text-gray-300 text-xs"
        >
          ✕
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <Stat label="ID" value={fixture.id} />
        <Stat label="Wall" value={fixture.wall_id} />
        <Stat
          label="Protrusion"
          value={`${depthCm.toFixed(1)} cm`}
          valueClass={depthColor}
        />
        <Stat label="Confidence" value={`${confPct}%`} />
        <Stat label="Width" value={`${(fixture.width_m * 100).toFixed(0)} cm`} />
        <Stat label="Height" value={`${(fixture.height_m * 100).toFixed(0)} cm`} />
        <Stat label="Point count" value={fixture.point_count.toLocaleString()} />
      </div>

      {/* Depth bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Protrusion depth</span>
          <span className={depthColor}>{depthCm.toFixed(1)} cm</span>
        </div>
        <div className="h-1.5 rounded-full bg-gray-700">
          <div
            className={`h-full rounded-full ${
              depthCm < 5 ? 'bg-teal-500' :
              depthCm < 15 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${Math.min(100, (depthCm / 30) * 100)}%` }}
          />
        </div>
      </div>

      <div className="text-xs text-gray-500 font-mono">
        @ ({fixture.centroid.map((v) => v.toFixed(2)).join(', ')})
      </div>
    </div>
  );
}

function Stat({ label, value, valueClass = 'text-gray-200' }: { label: string; value: string; valueClass?: string }) {
  return (
    <div>
      <div className="text-gray-500">{label}</div>
      <div className={`font-medium ${valueClass}`}>{value}</div>
    </div>
  );
}
