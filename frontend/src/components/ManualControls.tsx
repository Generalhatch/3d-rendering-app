import { useState } from 'react';
import * as THREE from 'three';
import { useJobStore } from '../state/jobStore';
import type { AlignmentScene } from '../three/scene';
import { api } from '../api/client';

interface ManualControlsProps {
  scene: AlignmentScene | null;
}

export function ManualControls({ scene }: ManualControlsProps) {
  const { isManualMode, setManualMode, jobId, fullResult, setFullResult } = useJobStore();
  const [nudgeStep, setNudgeStep] = useState(0.1); // meters
  const [rotateStep, setRotateStep] = useState(1.0); // degrees
  const [reevaluating, setReevaluating] = useState(false);
  const [lastResult, setLastResult] = useState<{ confidence_pct: number; residual_rmse_mm: number } | null>(null);

  if (!isManualMode) {
    return (
      <button
        onClick={() => setManualMode(true)}
        className="w-full py-2.5 rounded-lg border border-gray-600 bg-gray-800/40 hover:bg-gray-700/50 text-sm text-gray-300 transition-colors"
      >
        Manual Adjust Mode
      </button>
    );
  }

  const getCurrentMatrix = () => {
    if (!scene?.scanCloud) return new THREE.Matrix4();
    return scene.scanCloud.matrix.clone();
  };

  const applyDelta = (delta: THREE.Matrix4) => {
    if (!scene?.scanCloud) return;
    const current = getCurrentMatrix();
    const next = delta.multiply(current);
    scene.setScanTransform(next);
  };

  const nudge = (dx: number, dy: number) => {
    const delta = new THREE.Matrix4().makeTranslation(dx, dy, 0);
    applyDelta(delta);
  };

  const rotate = (deg: number) => {
    const rad = (deg * Math.PI) / 180;
    // Rotate around scene center
    const box = new THREE.Box3().setFromObject(scene!.scene);
    const center = new THREE.Vector3();
    box.getCenter(center);
    const toPivot = new THREE.Matrix4().makeTranslation(-center.x, -center.y, 0);
    const rot = new THREE.Matrix4().makeRotationZ(rad);
    const fromPivot = new THREE.Matrix4().makeTranslation(center.x, center.y, 0);
    const delta = fromPivot.multiply(rot).multiply(toPivot);
    applyDelta(delta);
  };

  const handleReEvaluate = async () => {
    if (!jobId || !scene?.scanCloud) return;
    setReevaluating(true);
    try {
      const m = getCurrentMatrix();
      const t = m.elements;
      // Convert THREE Matrix4 (column-major) to row-major 4x4 for API
      const matrix = [
        [t[0], t[4], t[8],  t[12]],
        [t[1], t[5], t[9],  t[13]],
        [t[2], t[6], t[10], t[14]],
        [t[3], t[7], t[11], t[15]],
      ];
      const result = await api.applyManualTransform(jobId, matrix);
      setLastResult({ confidence_pct: result.confidence_pct, residual_rmse_mm: result.residual_rmse_mm });
      if (fullResult) {
        setFullResult({
          ...fullResult,
          alignment: {
            ...fullResult.alignment,
            transformation: result.transformation,
            confidence: result.confidence,
            confidence_pct: result.confidence_pct,
            residual_rmse_mm: result.residual_rmse_mm,
          },
        });
      }
    } catch (e) {
      console.error('Re-evaluate failed', e);
    } finally {
      setReevaluating(false);
    }
  };

  return (
    <div className="rounded-xl border border-amber-700 bg-amber-950/20 p-4 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-amber-300">Manual Mode</span>
        <button
          onClick={() => setManualMode(false)}
          className="text-xs text-gray-400 hover:text-gray-200"
        >
          Exit
        </button>
      </div>

      {/* Translate controls */}
      <div className="space-y-2">
        <div className="text-xs text-gray-400">Translate (step: {nudgeStep}m)</div>
        <div className="grid grid-cols-3 gap-1">
          <div />
          <ArrowBtn label="▲" onClick={() => nudge(0, nudgeStep)} />
          <div />
          <ArrowBtn label="◄" onClick={() => nudge(-nudgeStep, 0)} />
          <ArrowBtn label="·" onClick={() => {}} />
          <ArrowBtn label="►" onClick={() => nudge(nudgeStep, 0)} />
          <div />
          <ArrowBtn label="▼" onClick={() => nudge(0, -nudgeStep)} />
          <div />
        </div>
        <input
          type="range"
          min={0.01}
          max={1.0}
          step={0.01}
          value={nudgeStep}
          onChange={(e) => setNudgeStep(parseFloat(e.target.value))}
          className="w-full accent-amber-500"
        />
      </div>

      {/* Rotate controls */}
      <div className="space-y-2">
        <div className="text-xs text-gray-400">Rotate (step: {rotateStep}°)</div>
        <div className="flex gap-2">
          <button
            onClick={() => rotate(-rotateStep)}
            className="flex-1 py-2 rounded-lg bg-amber-900/40 hover:bg-amber-800/50 text-sm font-medium border border-amber-800"
          >
            ↺ CCW
          </button>
          <button
            onClick={() => rotate(rotateStep)}
            className="flex-1 py-2 rounded-lg bg-amber-900/40 hover:bg-amber-800/50 text-sm font-medium border border-amber-800"
          >
            ↻ CW
          </button>
        </div>
        <input
          type="range"
          min={0.1}
          max={10}
          step={0.1}
          value={rotateStep}
          onChange={(e) => setRotateStep(parseFloat(e.target.value))}
          className="w-full accent-amber-500"
        />
      </div>

      {/* Re-evaluate */}
      <button
        onClick={handleReEvaluate}
        disabled={reevaluating}
        className="w-full py-2 rounded-lg bg-amber-700 hover:bg-amber-600 disabled:opacity-50 text-sm font-semibold transition-colors"
      >
        {reevaluating ? 'Re-evaluating…' : '⟳ Re-evaluate'}
      </button>

      {lastResult && (
        <div className="text-xs text-center text-gray-400">
          After: <span className="text-amber-300 font-semibold">{lastResult.confidence_pct}%</span>
          {' '}confidence · {lastResult.residual_rmse_mm.toFixed(1)} mm
        </div>
      )}
    </div>
  );
}

function ArrowBtn({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-sm font-medium transition-colors"
    >
      {label}
    </button>
  );
}
