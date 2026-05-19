import * as THREE from 'three';
import type { Fixture } from '../api/client';

function depthToColor(depthM: number): THREE.Color {
  // 2cm → teal (hue 0.5), 10cm → amber (hue 0.1), 30cm → red (hue 0.0)
  const t = Math.min(1, Math.max(0, (depthM - 0.02) / 0.28));
  return new THREE.Color().setHSL(0.5 - 0.5 * t, 0.85, 0.55);
}

export class FixtureMarkers {
  group = new THREE.Group();
  private meshes = new Map<string, THREE.Mesh>();
  private selectedId: string | null = null;

  constructor(fixtures: Fixture[]) {
    for (const fx of fixtures) {
      this._addMarker(fx);
    }
  }

  private _addMarker(fx: Fixture) {
    const extent = Math.max(fx.width_m, fx.height_m, 0.01);
    const radius = Math.max(0.06, Math.min(0.20, extent / 2));

    const geom = new THREE.SphereGeometry(radius, 12, 8);
    const mat = new THREE.MeshBasicMaterial({
      color: depthToColor(fx.protrusion_depth_m),
      transparent: true,
      opacity: 0.85,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(fx.centroid[0], fx.centroid[1], fx.centroid[2]);
    mesh.userData.fixtureId = fx.id;
    mesh.userData.fixture = fx;
    this.meshes.set(fx.id, mesh);
    this.group.add(mesh);
  }

  highlight(fixtureId: string | null) {
    if (this.selectedId) {
      const prev = this.meshes.get(this.selectedId);
      if (prev) {
        (prev.material as THREE.MeshBasicMaterial).opacity = 0.85;
        prev.scale.setScalar(1.0);
      }
    }
    if (fixtureId) {
      const mesh = this.meshes.get(fixtureId);
      if (mesh) {
        (mesh.material as THREE.MeshBasicMaterial).opacity = 1.0;
        mesh.scale.setScalar(1.6);
      }
    }
    this.selectedId = fixtureId;
  }

  setVisible(visible: boolean) {
    this.group.visible = visible;
  }

  pickFromRaycast(raycaster: THREE.Raycaster): Fixture | null {
    const hits = raycaster.intersectObjects([...this.meshes.values()]);
    return (hits[0]?.object.userData.fixture as Fixture | undefined) ?? null;
  }
}
