/**
 * Client-side degree-1 endpoint detection for live editor feedback.
 *
 * Mirrors the server topology snap idea at a coarse level: cluster wall
 * endpoints within ``snapTolM``, then treat clusters with a single incident
 * segment end as dangling.  Used while the operator edits so markers update
 * without a topology round-trip; server ``dangling_endpoints`` remain the
 * source of truth on load / Generate-BOMA failure.
 */
import type { DanglingEndpoint, EditableSegment } from '../../../api/vectorize';

/** Default snap radius — matches topology auto-tune floor (~0.10–0.15 m). */
export const DEFAULT_DANGLING_SNAP_TOL_M = 0.12;

export function computeDanglingEndpoints(
  segments: Array<Pick<EditableSegment, 'layer' | 'x1' | 'y1' | 'x2' | 'y2'> & { status?: string }>,
  snapTolM: number = DEFAULT_DANGLING_SNAP_TOL_M,
): DanglingEndpoint[] {
  const ends: Array<{ x: number; y: number }> = [];
  for (const s of segments) {
    if (s.status === 'rejected') continue;
    if ((s.layer ?? 'walls') !== 'walls') continue;
    ends.push({ x: s.x1, y: s.y1 }, { x: s.x2, y: s.y2 });
  }
  if (ends.length === 0) return [];

  const tol2 = snapTolM * snapTolM;
  const parent = ends.map((_, i) => i);

  const find = (i: number): number => {
    let root = i;
    while (parent[root] !== root) root = parent[root];
    let cur = i;
    while (parent[cur] !== root) {
      const next = parent[cur];
      parent[cur] = root;
      cur = next;
    }
    return root;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };

  for (let i = 0; i < ends.length; i++) {
    for (let j = i + 1; j < ends.length; j++) {
      const dx = ends[i].x - ends[j].x;
      const dy = ends[i].y - ends[j].y;
      if (dx * dx + dy * dy <= tol2) union(i, j);
    }
  }

  const clusters = new Map<number, { sx: number; sy: number; n: number }>();
  for (let i = 0; i < ends.length; i++) {
    const root = find(i);
    const c = clusters.get(root) ?? { sx: 0, sy: 0, n: 0 };
    c.sx += ends[i].x;
    c.sy += ends[i].y;
    c.n += 1;
    clusters.set(root, c);
  }

  const out: DanglingEndpoint[] = [];
  for (const c of clusters.values()) {
    // One incident endpoint → degree 1.  Two endpoints in the same cluster
    // usually means two walls already meet (or a zero-length stub).
    if (c.n === 1) {
      out.push({
        x: Math.round((c.sx / c.n) * 1e4) / 1e4,
        y: Math.round((c.sy / c.n) * 1e4) / 1e4,
        degree: 1,
      });
    }
  }
  return out;
}
