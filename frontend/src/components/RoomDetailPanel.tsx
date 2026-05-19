import { useJobStore } from '../state/jobStore';
import { CATEGORY_COLORS, CATEGORY_LABELS } from '../three/roomColors';
import * as THREE from 'three';

export function RoomDetailPanel() {
  const { selectedRoomId, rooms, selectRoom } = useJobStore();

  const room = rooms.find((r) => r.id === selectedRoomId);
  if (!room) return null;

  const colorHex = `#${new THREE.Color(CATEGORY_COLORS[room.category] ?? CATEGORY_COLORS.unknown).getHexString()}`;
  const mq = room.match_quality;
  const mqColor =
    mq === null ? 'text-gray-400' :
    mq >= 0.90 ? 'text-green-400' :
    mq >= 0.70 ? 'text-yellow-400' :
    'text-red-400';

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800/50 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: colorHex }} />
          <span className="font-semibold text-sm text-gray-100 truncate">{room.label}</span>
        </div>
        <button
          onClick={() => selectRoom(null)}
          className="text-gray-500 hover:text-gray-300 text-xs"
        >
          ✕
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <Stat label="Category" value={CATEGORY_LABELS[room.category] ?? 'Unknown'} />
        <Stat label="Area" value={`${room.area_m2.toFixed(1)} m²`} />
        <Stat
          label="Match Quality"
          value={mq !== null ? `${Math.round(mq * 100)}%` : 'N/A'}
          valueClass={mqColor}
        />
        <Stat label="Room ID" value={room.id} />
        {room.classification_confidence !== undefined && (
          <Stat
            label="Type Confidence"
            value={`${Math.round(room.classification_confidence * 100)}%`}
            valueClass={
              room.classification_confidence >= 0.80 ? 'text-green-400' :
              room.classification_confidence >= 0.60 ? 'text-yellow-400' :
              'text-gray-400'
            }
          />
        )}
        {room.eccentricity !== undefined && (
          <Stat
            label="Shape"
            value={
              room.eccentricity > 0.80 ? 'Elongated' :
              room.eccentricity < 0.30 ? 'Compact' : 'Regular'
            }
          />
        )}
      </div>

      {mq !== null && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-500">
            <span>Scan-to-plan match</span>
            <span className={mqColor}>{Math.round(mq * 100)}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-gray-700">
            <div
              className={`h-full rounded-full ${
                mq >= 0.90 ? 'bg-green-500' :
                mq >= 0.70 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${Math.round(mq * 100)}%` }}
            />
          </div>
        </div>
      )}

      {room.classification_confidence !== undefined && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-500">
            <span>Classification confidence</span>
            <span className={
              room.classification_confidence >= 0.80 ? 'text-green-400' :
              room.classification_confidence >= 0.60 ? 'text-yellow-400' :
              'text-gray-400'
            }>
              {Math.round(room.classification_confidence * 100)}%
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-gray-700">
            <div
              className={`h-full rounded-full ${
                room.classification_confidence >= 0.80 ? 'bg-green-500' :
                room.classification_confidence >= 0.60 ? 'bg-yellow-500' : 'bg-gray-500'
              }`}
              style={{ width: `${Math.round(room.classification_confidence * 100)}%` }}
            />
          </div>
        </div>
      )}
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
