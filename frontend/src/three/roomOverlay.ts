import * as THREE from 'three';
import type { Room } from '../api/client';
import { CATEGORY_COLORS } from './roomColors';

export class RoomOverlay {
  group = new THREE.Group();
  private meshes = new Map<string, THREE.Mesh>();
  private labels = new Map<string, THREE.Sprite>();
  private selected: string | null = null;
  private baseOpacity = 0.25;

  constructor(rooms: Room[], floorZ = 0) {
    for (const room of rooms) {
      this._addRoom(room, floorZ);
    }
  }

  private _addRoom(room: Room, z: number) {
    if (room.polygon_2d.length < 3) return;

    const shape = new THREE.Shape(
      room.polygon_2d.map(([x, y]) => new THREE.Vector2(x, y)),
    );
    const geom = new THREE.ShapeGeometry(shape);
    // Shift to floor level + tiny offset to avoid z-fighting with plan lines
    geom.translate(0, 0, z + 0.02);

    const color = CATEGORY_COLORS[room.category] ?? CATEGORY_COLORS.unknown;
    const mat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: this.baseOpacity,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData.roomId = room.id;
    mesh.userData.room = room;
    this.meshes.set(room.id, mesh);
    this.group.add(mesh);

    // Billboard label
    const sprite = _makeTextSprite(room.label, color);
    sprite.position.set(room.centroid[0], room.centroid[1], z + 0.5);
    sprite.userData.roomId = room.id;
    this.labels.set(room.id, sprite);
    this.group.add(sprite);
  }

  highlight(roomId: string | null) {
    if (this.selected) {
      const prev = this.meshes.get(this.selected);
      if (prev) (prev.material as THREE.MeshBasicMaterial).opacity = this.baseOpacity;
    }
    if (roomId) {
      const mesh = this.meshes.get(roomId);
      if (mesh) (mesh.material as THREE.MeshBasicMaterial).opacity = 0.55;
    }
    this.selected = roomId;
  }

  setOpacity(opacity: number) {
    this.baseOpacity = opacity;
    for (const [id, mesh] of this.meshes) {
      if (id !== this.selected) {
        (mesh.material as THREE.MeshBasicMaterial).opacity = opacity;
      }
    }
  }

  setVisible(visible: boolean) {
    this.group.visible = visible;
  }

  pickFromRaycast(raycaster: THREE.Raycaster): string | null {
    const hits = raycaster.intersectObjects([...this.meshes.values()]);
    return (hits[0]?.object.userData.roomId as string | undefined) ?? null;
  }

  getRoomAt(id: string): Room | undefined {
    return this.meshes.get(id)?.userData.room as Room | undefined;
  }
}

function _makeTextSprite(text: string, color: number): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d')!;

  // Background pill
  const c = new THREE.Color(color);
  ctx.fillStyle = `rgba(${Math.round(c.r * 255)}, ${Math.round(c.g * 255)}, ${Math.round(c.b * 255)}, 0.75)`;
  _roundRect(ctx, 4, 4, canvas.width - 8, canvas.height - 8, 8);
  ctx.fill();

  // Text
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 20px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  // Truncate long labels
  const label = text.length > 18 ? text.slice(0, 17) + '…' : text;
  ctx.fillText(label, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(3.0, 0.75, 1);
  return sprite;
}

function _roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
