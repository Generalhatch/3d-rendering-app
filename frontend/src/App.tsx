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
import type { FullResult, Room, Fixture, ReprocessParams } from './api/client';

import { UploadZone } from './components/UploadZone';
import { ProcessingPanel } from './components/ProcessingPanel';
import { ViewerScene } from './components/ViewerScene';
import { ConfidencePanel } from './components/ConfidencePanel';
import { RoomDetailPanel } from './components/RoomDetailPanel';
import { FixturePanel } from './components/FixturePanel';
import { ManualControls } from './components/ManualControls';
import { JobHistoryPanel } from './components/JobHistoryPanel';
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
    jobId, isScanOnly, setScanOnly, addJobToHistory,
    setFloorCandidates, setReprocessing,
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
    if (!scene) {
      console.error('[viewer] sceneRef is null — ViewerScene not mounted yet');
      return;
    }

    try {
      // ── 1. Load decimated PLY ──────────────────────────────────────────────
      let scanData: Awaited<ReturnType<typeof loadPointCloudFromUrl>>;
      try {
        scanData = await loadPointCloudFromUrl(api.getScanUrl(jId));
        console.log(`[viewer] PLY loaded: ${scanData.positions.length / 3} pts, colors=${!!scanData.colors}`);
      } catch (e) {
        console.error('[viewer] PLY load failed', e);
        throw e;
      }
      scene.loadScan(scanData.positions, scanData.colors);

      // ── 2. Load plan GeoJSON + building outline (in parallel) ─────────────
      let segments: Float32Array;
      const [planResult, outlineResult] = await Promise.allSettled([
        api.getPlanGeoJSON(jId),
        api.getOutlineGeoJSON(jId),
      ]);

      if (planResult.status === 'fulfilled') {
        segments = planGeoJSONToSegments(planResult.value as Parameters<typeof planGeoJSONToSegments>[0]);
        console.log(`[viewer] plan loaded: ${segments.length / 6} line segments`);
      } else {
        console.error('[viewer] plan GeoJSON failed', planResult.reason);
        segments = new Float32Array(0);
      }
      scene.loadPlan(segments);

      if (outlineResult.status === 'fulfilled') {
        const feature = outlineResult.value.features?.[0];
        const ring = feature?.geometry?.coordinates?.[0];
        if (ring && ring.length > 0) {
          const outlinePts = new Float32Array(ring.length * 3);
          ring.forEach(([x, y], i) => {
            outlinePts[i * 3]     = x;
            outlinePts[i * 3 + 1] = y;
            outlinePts[i * 3 + 2] = 0;  // floor_z applied inside loadOutline
          });
          console.log(`[viewer] outline loaded: ${ring.length} vertices`);
          scene.loadOutline(outlinePts, 0);  // floor_z set after jobDetail fetch below
        }
      } else {
        console.debug('[viewer] building outline not available for this job');
      }

      // ── 3. Fetch job metadata + rooms/fixtures ─────────────────────────────
      const [jobDetail, roomsResp, fixturesResp] = await Promise.all([
        api.getJob(jId),
        api.getRooms(jId),
        api.getFixtures(jId),
      ]);

      const rooms: Room[] = roomsResp.rooms;
      const fixtures: Fixture[] = fixturesResp.fixtures;

      const floorZ = jobDetail.floor_z ?? 0;

      // Re-apply correct floor_z to the outline now that we have it from jobDetail
      if (outlineResult.status === 'fulfilled') {
        const feature = outlineResult.value.features?.[0];
        const ring = feature?.geometry?.coordinates?.[0];
        if (ring && ring.length > 0) {
          const outlinePts = new Float32Array(ring.length * 3);
          ring.forEach(([x, y], i) => {
            outlinePts[i * 3]     = x;
            outlinePts[i * 3 + 1] = y;
            outlinePts[i * 3 + 2] = 0;
          });
          scene.loadOutline(outlinePts, floorZ);
        }
      }

      const fullResult: FullResult = {
        job_id: jId,
        alignment: jobDetail.result!,
        floor_z: floorZ,
        scan_only: jobDetail.scan_only ?? false,
        rooms,
        fixtures,
        plan_bounds: (jobDetail.plan_bounds as [number, number, number, number]) ?? [0, 0, 100, 100],
      };
      setFullResult(fullResult);
      setRooms(rooms);
      setFixtures(fixtures);

      // Update history entry with final status + elapsed time
      addJobToHistory({
        jobId: jId,
        status: jobDetail.status,
        createdAt: jobDetail.created_at,
        scanCount: (jobDetail as { scan_filenames?: string[] }).scan_filenames?.length ?? 1,
        scanOnly: jobDetail.scan_only ?? false,
        elapsed_s: jobDetail.elapsed_s,
      });

      // MJ2: populate floor candidates in store so ConfidencePanel can show them
      if (jobDetail.floor_candidates && jobDetail.floor_candidates.length > 0) {
        setFloorCandidates(jobDetail.floor_candidates);
      } else {
        // Fallback: fetch from the dedicated endpoint for older jobs
        try {
          const floors = await api.getFloors(jId);
          setFloorCandidates(floors);
        } catch {
          setFloorCandidates([]);
        }
      }

      // ── 4. Fit camera to the loaded scene ─────────────────────────────────
      scene.fitToScene();

      // ── 5. Snap animation (identity→alignment transform) ──────────────────
      const flatMatrix = jobDetail.result!.transformation.flat();
      const targetMatrix = new THREE.Matrix4().fromArray(flatMatrix).transpose();

      playSnapAnimation({
        scene,
        fromMatrix: new THREE.Matrix4(),
        toMatrix: targetMatrix,
        durationMs: 1500,
        onComplete: () => {
          // Re-fit camera after scan has been transformed to its aligned position
          scene.fitToScene();

          const overlay = new RoomOverlay(rooms, fullResult.floor_z);
          roomOverlayRef.current = overlay;
          scene.addRoomOverlay(overlay);

          setTimeout(() => {
            const markers = new FixtureMarkers(fixtures);
            fixtureMarkersRef.current = markers;
            scene.addFixtureMarkers(markers);
          }, 600);

          setPhase('aligned');
        },
      });
    } catch (err) {
      console.error('[viewer] loadResultIntoScene failed', err);
      setPhase('failed');
    }
  }, [setFullResult, setRooms, setFixtures, setPhase, addJobToHistory, setFloorCandidates]);

  const handleFilesReady = useCallback(async (scanFiles: File[], planFile: File | null, bandLow: number, bandHigh: number) => {
    reset();
    setScanOnly(planFile === null);
    setPhase('uploading');

    try {
      const { job_id } = await api.createJob(scanFiles, planFile, bandLow, bandHigh);
      setJobId(job_id);
      setPhase('processing');

      // Record as "processing" in history immediately so it shows up even if the
      // user refreshes before it completes
      addJobToHistory({
        jobId: job_id,
        status: 'processing',
        createdAt: new Date().toISOString(),
        scanCount: scanFiles.length,
        scanOnly: planFile === null,
      });

      const sse = api.subscribeProgress(job_id, (event) => {
        appendLog(event);
        if (event.stage === 'complete') {
          sse.close();
          loadResultIntoScene(job_id);
        } else if (event.stage === 'error') {
          sse.close();
          setPhase('failed');
          addJobToHistory({
            jobId: job_id,
            status: 'failed',
            createdAt: new Date().toISOString(),
            scanCount: scanFiles.length,
            scanOnly: planFile === null,
          });
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
  }, [reset, setPhase, setJobId, appendLog, loadResultIntoScene, setScanOnly, addJobToHistory]);

  // Called when the user clicks "Load" on a history entry to restore a past job
  const handleResumeJob = useCallback(async (resumeJobId: string) => {
    reset();
    setJobId(resumeJobId);
    setPhase('processing');
    try {
      const jobDetail = await api.getJob(resumeJobId);
      setScanOnly(jobDetail.scan_only ?? false);
      await loadResultIntoScene(resumeJobId);
    } catch {
      setPhase('failed');
    }
  }, [reset, setJobId, setPhase, setScanOnly, loadResultIntoScene]);

  // MJ2/MJ3: trigger re-processing with new parameters, subscribe to SSE, reload scene on complete
  const handleReprocess = useCallback(async (params: ReprocessParams) => {
    if (!jobId) return;
    setReprocessing(true);
    try {
      await api.reprocessJob(jobId, params);
      // Subscribe to SSE for progress events — same channel as the original pipeline
      const sse = api.subscribeProgress(jobId, (event) => {
        appendLog(event);
        if (event.stage === 'complete') {
          sse.close();
          setReprocessing(false);
          // Reload rooms, plan, and outline from the updated result
          loadResultIntoScene(jobId);
        } else if (event.stage === 'error') {
          sse.close();
          setReprocessing(false);
        }
      });
    } catch (err) {
      setReprocessing(false);
      appendLog({
        stage: 'error',
        message: err instanceof Error ? err.message : 'Re-process request failed',
        progress: 1,
        job_id: jobId,
      });
    }
  }, [jobId, setReprocessing, appendLog, loadResultIntoScene]);

  const isProcessing = phase === 'processing' || phase === 'uploading';
  const spinnerLabel = isScanOnly ? 'Generating floor plan…' : 'Aligning scan to plan…';

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
            <OutlineLegend />
          </div>
        )}

        {/* Processing overlay */}
        {isProcessing && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900/60 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-4">
              <div className="w-12 h-12 rounded-full border-4 border-blue-400 border-t-transparent animate-spin" />
              <p className="text-blue-300 font-medium text-sm">{spinnerLabel}</p>
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

          {/* Job history — shown only on idle/failed so it doesn't compete with results */}
          {(phase === 'idle' || phase === 'failed') && (
            <JobHistoryPanel onResumeJob={handleResumeJob} />
          )}

          {/* Processing log */}
          <ProcessingPanel />

          {/* Confidence + controls */}
          <ConfidencePanel onReprocess={handleReprocess} />

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

function OutlineLegend() {
  return (
    <div className="rounded-lg bg-gray-900/80 border border-gray-700 px-3 py-2 backdrop-blur-sm">
      <div className="flex items-center gap-2 text-xs">
        <span className="w-4 h-0.5 flex-shrink-0" style={{ backgroundColor: '#f59e0b' }} />
        <span className="text-gray-300">Building outline</span>
      </div>
    </div>
  );
}

function _clearScene(scene: AlignmentScene) {
  const disposeObj = (obj: THREE.Points | THREE.LineSegments | THREE.LineLoop | null) => {
    if (!obj) return;
    scene.scene.remove(obj);
    obj.geometry.dispose();
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    mats.forEach((m) => (m as THREE.Material).dispose());
  };
  disposeObj(scene.scanCloud);
  scene.scanCloud = null;
  disposeObj(scene.planLines);
  scene.planLines = null;
  disposeObj(scene.outlineLines);
  scene.outlineLines = null;
  if (scene.roomOverlay) {
    scene.scene.remove(scene.roomOverlay.group);
    scene.roomOverlay = null;
  }
  if (scene.fixtureMarkers) {
    scene.scene.remove(scene.fixtureMarkers.group);
    scene.fixtureMarkers = null;
  }
}
