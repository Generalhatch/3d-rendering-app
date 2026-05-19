/**
 * Load a PLY binary file into a Float32Array of positions (and optionally colors).
 * We fetch the PLY from the server, then parse it client-side.
 *
 * For simplicity, this uses Three.js's built-in PLYLoader.
 */
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

export interface PointCloudData {
  positions: Float32Array;
  colors?: Float32Array;
}

export async function loadPointCloudFromUrl(url: string): Promise<PointCloudData> {
  return new Promise((resolve, reject) => {
    const loader = new PLYLoader();
    loader.load(
      url,
      (geometry: THREE.BufferGeometry) => {
        const positions = geometry.getAttribute('position');
        const posArray = new Float32Array(positions.array as ArrayLike<number>);

        let colors: Float32Array | undefined;
        const colorAttr = geometry.getAttribute('color');
        if (colorAttr) {
          colors = new Float32Array(colorAttr.array as ArrayLike<number>);
        }

        resolve({ positions: posArray, colors });
      },
      undefined,
      (err: unknown) => reject(err instanceof Error ? err : new Error(String(err))),
    );
  });
}
