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
  roomOverlay: RoomOverlay | null = null;
  fixtureMarkers: FixtureMarkers | null = null;

  private _animFrameId = 0;
  private _onRoomPick: ((id: string | null) => void) | null = null;
  private _onFixturePick: ((id: string | null) => void) | null = null;
  private _raycaster = new THREE.Raycaster();
  private _pointer = new THREE.Vector2();

  constructor(canvas: HTMLCanvasElement) {
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

    canvas.addEventListener('click', this._onClick);
    window.addEventListener('resize', () => this._onResize(canvas));
  }

  loadScan(positions: Float32Array, colors?: Float32Array) {
    if (this.scanCloud) {
      this.scene.remove(this.scanCloud);
      this.scanCloud.geometry.dispose();
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (colors) geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.05,
      color: colors ? 0xffffff : 0xe11d48,
      vertexColors: !!colors,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true,
    });
    this.scanCloud = new THREE.Points(geom, mat);
    this.scene.add(this.scanCloud);
  }

  loadPlan(segments: Float32Array) {
    if (this.planLines) {
      this.scene.remove(this.planLines);
      this.planLines.geometry.dispose();
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(segments, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0xd1d5db, linewidth: 1 });
    this.planLines = new THREE.LineSegments(geom, mat);
    this.scene.add(this.planLines);
  }

  setScanTransform(matrix: THREE.Matrix4) {
    if (!this.scanCloud) return;
    this.scanCloud.matrixAutoUpdate = false;
    this.scanCloud.matrix.copy(matrix);
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
    const box = new THREE.Box3().setFromObject(this.scene);
    if (box.isEmpty()) return;
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = this.camera.fov * (Math.PI / 180);
    const dist = (maxDim / 2) / Math.tan(fov / 2) * 1.5;
    this.camera.position.set(center.x, center.y + dist * 0.01, center.z + dist);
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
