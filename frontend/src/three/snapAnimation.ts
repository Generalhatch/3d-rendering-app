import * as THREE from 'three';
import TWEEN from '@tweenjs/tween.js';
import type { AlignmentScene } from './scene';

interface SnapParams {
  scene: AlignmentScene;
  fromMatrix: THREE.Matrix4;
  toMatrix: THREE.Matrix4;
  durationMs?: number;
  onComplete?: () => void;
}

export function playSnapAnimation(params: SnapParams): void {
  const duration = params.durationMs ?? 1500;

  const startPos = new THREE.Vector3();
  const startQuat = new THREE.Quaternion();
  const startScale = new THREE.Vector3();
  params.fromMatrix.decompose(startPos, startQuat, startScale);

  const endPos = new THREE.Vector3();
  const endQuat = new THREE.Quaternion();
  const endScale = new THREE.Vector3();
  params.toMatrix.decompose(endPos, endQuat, endScale);

  const state = { t: 0 };

  new TWEEN.Tween(state)
    .to({ t: 1 }, duration)
    .easing(TWEEN.Easing.Cubic.InOut)
    .onUpdate(() => {
      const pos = startPos.clone().lerp(endPos, state.t);
      const quat = startQuat.clone().slerp(endQuat, state.t);
      const m = new THREE.Matrix4().compose(pos, quat, startScale);
      params.scene.setScanTransform(m);
    })
    .onComplete(() => {
      // Set the final exact matrix
      params.scene.setScanTransform(params.toMatrix);
      params.onComplete?.();
    })
    .start();

  // TWEEN update loop — stops once complete
  let running = true;
  function loop(time: number) {
    if (!running) return;
    TWEEN.update(time);
    if (state.t < 1) {
      requestAnimationFrame(loop);
    } else {
      running = false;
    }
  }
  requestAnimationFrame(loop);
}
