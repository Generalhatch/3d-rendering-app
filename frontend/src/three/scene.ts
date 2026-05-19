import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { RoomOverlay } from './roomOverlay';
import type { FixtureMarkers } from './fixtureMarkers';

export class AlignmentScene {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;

  scanCloud: THREE.Points | null = null;
  planLines: THREE.LineSegments | null = null;
  outlineLines: THREE.LineLoop | null = null;
  roomOverlay: RoomOverlay | null = null;
  fixtureMarkers: FixtureMarkers | null = null;

  private _animFrameId = 0;
  private _onRoomPick: ((id: string | null) => void) | null = null;
  private _onFixturePick: ((id: string | null) => void) | null = null;
  private _raycaster = new THREE.Raycaster();
  private _pointer = new THREE.Vector2();
  private _resizeHandler!: () => void;
  private _canvas!: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement) {
    this._canvas = canvas;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111827);

    this.camera = new THREE.PerspectiveCamera(
      45,
      canvas.clientWidth / canvas.clientHeight,
      0.01,
      2000,
    );
    this.camera.position.set(0, 0, 80);
    this.camera.up.set(0, 1, 0);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    // Default to top-down view
    this.controls.enableRotate = true;

    this._resizeHandler = () => this._onResize(canvas);
    canvas.addEventListener('click', this._onClick);
    window.addEventListener('resize', this._resizeHandler);
  }

  loadScan(positions: Float32Array, colors?: Float32Array) {
    if (this.scanCloud) {
      this.scene.remove(this.scanCloud);
      this.scanCloud.geometry.dispose();
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    let mat: THREE.PointsMaterial;
    if (colors) {
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      mat = new THREE.PointsMaterial({
        size: 0.12,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        sizeAttenuation: true,
        depthWrite: false,
      });
    } else {
      // No vertex colors — compute a height-gradient so the user can perceive
      // floor vs. wall vs. ceiling depth without any RGB scanner data.
      // Gradient: deep navy (floor) → sky blue (low wall) → cyan (mid wall) → pale blue-white (ceiling)
      geom.computeBoundingBox();
      const bbox = geom.boundingBox!;
      const zMin = bbox.min.z;
      const zRange = Math.max(bbox.max.z - zMin, 0.001);

      const n = positions.length / 3;
      const gradientColors = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const z = positions[i * 3 + 2];
        const t = Math.max(0, Math.min(1, (z - zMin) / zRange));
        const [r, g, b] = _heightToColor(t);
        gradientColors[i * 3]     = r;
        gradientColors[i * 3 + 1] = g;
        gradientColors[i * 3 + 2] = b;
      }
      geom.setAttribute('color', new THREE.BufferAttribute(gradientColors, 3));

      mat = new THREE.PointsMaterial({
        size: 0.12,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        sizeAttenuation: true,
        depthWrite: false,
      });
    }

    this.scanCloud = new THREE.Points(geom, mat);
    this.scene.add(this.scanCloud);
  }

  loadPlan(segments: Float32Array) {
    if (this.planLines) {
      this.scene.remove(this.planLines);
      this.planLines.geometry.dispose();
    }
    if (segments.length === 0) return;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(segments, 3));
    // Bright white plan lines so wall outlines are clearly legible over the point cloud
    const mat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
    this.planLines = new THREE.LineSegments(geom, mat);
    // Render plan lines on top of the point cloud
    this.planLines.renderOrder = 1;
    this.scene.add(this.planLines);
  }

  loadOutline(polygon: Float32Array, floorZ = 0) {
    if (this.outlineLines) {
      this.scene.remove(this.outlineLines);
      this.outlineLines.geometry.dispose();
      (this.outlineLines.material as THREE.Material).dispose();
      this.outlineLines = null;
    }
    if (polygon.length < 6) return;  // need at least 3 XYZ vertices

    const geom = new THREE.BufferGeometry();
    // Lift the outline slightly above floor_z so it's never clipped by the cloud
    const lifted = new Float32Array(polygon.length);
    for (let i = 0; i < polygon.length; i += 3) {
      lifted[i]     = polygon[i];
      lifted[i + 1] = polygon[i + 1];
      lifted[i + 2] = floorZ + 0.05;
    }
    geom.setAttribute('position', new THREE.BufferAttribute(lifted, 3));
    // Bright amber outline — distinct from white plan lines and room fill colours
    const mat = new THREE.LineBasicMaterial({ color: 0xf59e0b, linewidth: 2 });
    this.outlineLines = new THREE.LineLoop(geom, mat);
    this.outlineLines.renderOrder = 2;
    this.scene.add(this.outlineLines);
  }

  setScanTransform(matrix: THREE.Matrix4) {
    if (!this.scanCloud) return;
    this.scanCloud.matrixAutoUpdate = false;
    this.scanCloud.matrix.copy(matrix);
    this.scanCloud.matrixWorldNeedsUpdate = true;
  }

  setScanOpacity(opacity: number) {
    if (!this.scanCloud) return;
    (this.scanCloud.material as THREE.PointsMaterial).opacity = opacity;
  }

  addRoomOverlay(overlay: RoomOverlay) {
    if (this.roomOverlay) this.scene.remove(this.roomOverlay.group);
    this.roomOverlay = overlay;
    this.scene.add(overlay.group);
  }

  addFixtureMarkers(markers: FixtureMarkers) {
    if (this.fixtureMarkers) this.scene.remove(this.fixtureMarkers.group);
    this.fixtureMarkers = markers;
    this.scene.add(markers.group);
  }

  setRoomOpacity(opacity: number) {
    this.roomOverlay?.setOpacity(opacity);
  }

  bindRoomPicking(cb: (id: string | null) => void) {
    this._onRoomPick = cb;
  }

  bindFixturePicking(cb: (id: string | null) => void) {
    this._onFixturePick = cb;
  }

  fitToScene() {
    const target = this.scanCloud ?? this.planLines;
    if (!target) return;
    const box = new THREE.Box3().setFromObject(target);
    if (box.isEmpty()) return;
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);
    // For floor-plan scans (Z-up, XY spread), the widest dimensions are X and Y.
    // Position camera above the scene looking straight down.
    const maxDim = Math.max(size.x, size.y);
    const fov = this.camera.fov * (Math.PI / 180);
    const dist = (maxDim / 2) / Math.tan(fov / 2) * 1.5;
    this.camera.position.set(center.x, center.y, center.z + dist);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(center);
    this.controls.target.copy(center);
    this.controls.update();
  }

  setTopDown() {
    const box = new THREE.Box3().setFromObject(this.scene);
    if (box.isEmpty()) return;
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y) * 1.5;
    const fov = this.camera.fov * (Math.PI / 180);
    const dist = (maxDim / 2) / Math.tan(fov / 2);
    this.camera.position.set(center.x, center.y, center.z + dist);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(center);
    this.controls.target.copy(center);
    this.controls.update();
  }

  startRenderLoop() {
    const animate = () => {
      this._animFrameId = requestAnimationFrame(animate);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    animate();
  }

  stopRenderLoop() {
    cancelAnimationFrame(this._animFrameId);
  }

  dispose() {
    this.stopRenderLoop();
    window.removeEventListener('resize', this._resizeHandler);
    this._canvas.removeEventListener('click', this._onClick);
    this.renderer.dispose();
  }

  private _onClick = (event: MouseEvent) => {
    const canvas = this.renderer.domElement;
    const rect = canvas.getBoundingClientRect();
    this._pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this._pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this._raycaster.setFromCamera(this._pointer, this.camera);

    // Check fixture markers first (smaller targets, should have priority)
    if (this.fixtureMarkers && this._onFixturePick) {
      const fixture = this.fixtureMarkers.pickFromRaycast(this._raycaster);
      if (fixture) {
        this._onFixturePick(fixture.id);
        return;
      }
    }

    // Then rooms
    if (this.roomOverlay && this._onRoomPick) {
      const roomId = this.roomOverlay.pickFromRaycast(this._raycaster);
      this._onRoomPick(roomId);
    }
  };

  private _onResize = (canvas: HTMLCanvasElement) => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };
}

// Height-gradient color stops: floor (dark navy) → wall mid (cyan) → ceiling (pale blue-white)
const _HEIGHT_STOPS: Array<{ t: number; r: number; g: number; b: number }> = [
  { t: 0.00, r: 0.07, g: 0.12, b: 0.40 },  // floor: deep navy
  { t: 0.30, r: 0.10, g: 0.47, b: 0.82 },  // low wall: medium blue
  { t: 0.60, r: 0.22, g: 0.74, b: 0.88 },  // mid wall: cyan
  { t: 1.00, r: 0.78, g: 0.94, b: 1.00 },  // ceiling: pale blue-white
];

function _heightToColor(t: number): [number, number, number] {
  for (let i = 1; i < _HEIGHT_STOPS.length; i++) {
    const lo = _HEIGHT_STOPS[i - 1];
    const hi = _HEIGHT_STOPS[i];
    if (t <= hi.t) {
      const f = (t - lo.t) / (hi.t - lo.t);
      return [
        lo.r + f * (hi.r - lo.r),
        lo.g + f * (hi.g - lo.g),
        lo.b + f * (hi.b - lo.b),
      ];
    }
  }
  const last = _HEIGHT_STOPS[_HEIGHT_STOPS.length - 1];
  return [last.r, last.g, last.b];
}
