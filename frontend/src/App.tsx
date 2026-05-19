import { useRef, useCallback, useEffect } from 'react';
import * as THREE from 'three';
import { AlignmentScene } from './three/scene';
import { RoomOverlay } from './three/roomOverlay';
import { FixtureMarkers } from './three/fixtureMarkers';
import { playSnapAnimation } from './three/snapAnimation';
import { loadPointCloudFromUrl } from './three/pointCloudLoader';
import { planGeoJSONToSegments } from './three/planLoader';
import { useJobStore } from './state/jobStore';
import { api } from './api/client';
import type { FullResult, Room, Fixture } from './api/client';

import { UploadZone } from './components/UploadZone';
import { ProcessingPanel } from './components/ProcessingPanel';
import { ViewerScene } from './components/ViewerScene';
import { ConfidencePanel } from './components/ConfidencePanel';
import { RoomDetailPanel } from './components/RoomDetailPanel';
import { FixturePanel } from './components/FixturePanel';
import { ManualControls } from './components/ManualControls';
import { CATEGORY_COLORS, CATEGORY_LABELS } from './three/roomColors';
import { DEPTH_COLOR_LEGEND } from './three/fixtureColors';

export default function App() {
  const sceneRef = useRef<AlignmentScene | null>(null);
  const roomOverlayRef = useRef<RoomOverlay | null>(null);
  const fixtureMarkersRef = useRef<FixtureMarkers | null>(null);

  const {
    phase, setPhase, setJobId, appendLog, reset,
    setFullResult, setRooms, setFixtures,
    selectedRoomId, selectedFixtureId,
    showFixtures, showRooms, scanOpacity, roomOpacity,
    jobId,
  } = useJobStore();

  // Sync scene visibility/opacity when store changes
  useEffect(() => {
    sceneRef.current?.setScanOpacity(scanOpacity);
  }, [scanOpacity]);

  useEffect(() => {
    roomOverlayRef.current?.setVisible(showRooms);
    sceneRef.current?.setRoomOpacity(roomOpacity);
  }, [showRooms, roomOpacity]);

  useEffect(() => {
    fixtureMarkersRef.current?.setVisible(showFixtures);
  }, [showFixtures]);

  useEffect(() => {
    roomOverlayRef.current?.highlight(selectedRoomId);
  }, [selectedRoomId]);

  useEffect(() => {
    fixtureMarkersRef.current?.highlight(selectedFixtureId);
  }, [selectedFixtureId]);

  const loadResultIntoScene = useCallback(async (jId: string) => {
    const scene = sceneRef.current;
    if (!scene) return;

    try {
      // Load scan PLY
      const scanData = await loadPointCloudFromUrl(api.getScanUrl(jId));
      scene.loadScan(scanData.positions, scanData.colors);

      // Load plan lines
      const geojson = await api.getPlanGeoJSON(jId);
      const segments = planGeoJSONToSegments(geojson as Parameters<typeof planGeoJSONToSegments>[0]);
      scene.loadPlan(segments);

      // Load full result
      const jobDetail = await api.getJob(jId);
      // Re-fetch the rooms and fixtures
      const [roomsResp, fixturesResp] = await Promise.all([
        api.getRooms(jId),
        api.getFixtures(jId),
      ]);

      const rooms: Room[] = roomsResp.rooms;
      const fixtures: Fixture[] = fixturesResp.fixtures;

      // Build the FullResult from jobDetail
      const fullResult: FullResult = {
        job_id: jId,
        alignment: jobDetail.result!,
        floor_z: 0,
        rooms,
        fixtures,
        plan_bounds: [0, 0, 100, 100],
      };
      setFullResult(fullResult);
      setRooms(rooms);
      setFixtures(fixtures);

      // Fit camera first (unaligned view)
      scene.fitToScene();

      // The snap: scan starts at identity, moves to alignment matrix
      const flatMatrix = jobDetail.result!.transformation.flat();
      const targetMatrix = new THREE.Matrix4().fromArray(flatMatrix).transpose();

      playSnapAnimation({
        scene,
        fromMatrix: new THREE.Matrix4(),  // identity = unaligned
        toMatrix: targetMatrix,
        durationMs: 1500,
        onComplete: () => {
          // Rooms fade in after snap
          const overlay = new RoomOverlay(rooms, fullResult.floor_z);
          roomOverlayRef.current = overlay;
          scene.addRoomOverlay(overlay);

          // Fixtures fade in shortly after
          setTimeout(() => {
            const markers = new FixtureMarkers(fixtures);
            fixtureMarkersRef.current = markers;
            scene.addFixtureMarkers(markers);
          }, 600);

          setPhase('aligned');
        },
      });
    } catch (err) {
      console.error('Failed to load result into scene', err);
    }
  }, [setFullResult, setRooms, setFixtures, setPhase]);

  const handleFilesReady = useCallback(async (planFile: File, scanFiles: File[]) => {
    reset();
    setPhase('uploading');

    try {
      const { job_id } = await api.createJob(planFile, scanFiles);
      setJobId(job_id);
      setPhase('processing');

      const sse = api.subscribeProgress(job_id, (event) => {
        appendLog(event);
        if (event.stage === 'complete') {
          sse.close();
          loadResultIntoScene(job_id);
        } else if (event.stage === 'error') {
          sse.close();
          setPhase('failed');
        }
      });
    } catch (err) {
      setPhase('failed');
      appendLog({
        stage: 'error',
        message: err instanceof Error ? err.message : 'Upload failed',
        progress: 1,
        job_id: '',
      });
    }
  }, [reset, setPhase, setJobId, appendLog, loadResultIntoScene]);

  const isProcessing = phase === 'processing' || phase === 'uploading';

  return (
    <div className="grid h-screen" style={{ gridTemplateColumns: '1fr 380px' }}>
      {/* 3D Viewer */}
      <div className="relative overflow-hidden">
        <ViewerScene onReady={(s) => { sceneRef.current = s; }} />

        {/* Top-right viewer controls */}
        <div className="absolute top-3 right-3 flex gap-2">
          <ViewBtn onClick={() => sceneRef.current?.setTopDown()} label="⊤ Top" />
          <ViewBtn onClick={() => sceneRef.current?.fitToScene()} label="⊞ Fit" />
        </div>

        {/* Bottom-left legend */}
        {(phase === 'aligned' || phase === 'approved') && (
          <div className="absolute bottom-4 left-4 space-y-3">
            <RoomLegend />
            <FixtureLegend />
          </div>
        )}

        {/* Processing overlay */}
        {isProcessing && phase !== 'aligned' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900/60 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-4">
              <div className="w-12 h-12 rounded-full border-4 border-blue-400 border-t-transparent animate-spin" />
              <p className="text-blue-300 font-medium text-sm">Aligning scan to plan…</p>
            </div>
          </div>
        )}
      </div>

      {/* Side panel */}
      <div className="flex flex-col border-l border-gray-700 bg-gray-900 overflow-y-auto side-panel-scroll">
        <div className="p-5 space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-lg font-bold text-white">AlignAI</h1>
              <p className="text-xs text-gray-500">Scan-to-Plan Alignment</p>
            </div>
            {phase !== 'idle' && (
              <button
                onClick={() => { reset(); sceneRef.current && _clearScene(sceneRef.current); }}
                className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                New job
              </button>
            )}
          </div>

          {/* Upload zone — always shown until aligned */}
          {(phase === 'idle' || phase === 'uploading' || phase === 'failed') && (
            <UploadZone onFilesReady={handleFilesReady} disabled={isProcessing} />
          )}

          {/* Processing log */}
          <ProcessingPanel />

          {/* Confidence + controls */}
          <ConfidencePanel />

          {/* Room detail */}
          {selectedRoomId && <RoomDetailPanel />}

          {/* Fixture detail */}
          {selectedFixtureId && <FixturePanel />}

          {/* Manual controls */}
          {(phase === 'aligned' || phase === 'approved') && (
            <ManualControls scene={sceneRef.current} />
          )}
        </div>
      </div>
    </div>
  );
}

function ViewBtn({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 rounded-lg bg-gray-900/70 hover:bg-gray-800/90 border border-gray-600 text-xs text-gray-300 backdrop-blur-sm transition-colors"
    >
      {label}
    </button>
  );
}

function RoomLegend() {
  const entries = Object.entries(CATEGORY_LABELS);
  return (
    <div className="rounded-lg bg-gray-900/80 border border-gray-700 px-3 py-2 backdrop-blur-sm space-y-1">
      {entries.map(([cat, label]) => (
        <div key={cat} className="flex items-center gap-2 text-xs">
          <span
            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: `#${new THREE.Color(CATEGORY_COLORS[cat]).getHexString()}` }}
          />
          <span className="text-gray-300">{label}</span>
        </div>
      ))}
    </div>
  );
}

function FixtureLegend() {
  return (
    <div className="rounded-lg bg-gray-900/80 border border-gray-700 px-3 py-2 backdrop-blur-sm space-y-1">
      <div className="text-xs text-gray-500 font-medium mb-1">Fixture depth</div>
      {DEPTH_COLOR_LEGEND.map((d) => (
        <div key={d.label} className="flex items-center gap-2 text-xs">
          <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
          <span className="text-gray-300">{d.label}</span>
        </div>
      ))}
    </div>
  );
}

function _clearScene(scene: AlignmentScene) {
  if (scene.scanCloud) { scene.scene.remove(scene.scanCloud); scene.scanCloud = null; }
  if (scene.planLines) { scene.scene.remove(scene.planLines); scene.planLines = null; }
  if (scene.roomOverlay) { scene.scene.remove(scene.roomOverlay.group); scene.roomOverlay = null; }
  if (scene.fixtureMarkers) { scene.scene.remove(scene.fixtureMarkers.group); scene.fixtureMarkers = null; }
}
