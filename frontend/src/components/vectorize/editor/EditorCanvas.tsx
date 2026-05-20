/**
 * The interactive editor surface (Phase 3 MVP).
 *
 * Renders the cleaned raster as a backdrop and overlays each segment as an
 * SVG element that the operator can click, multi-select, delete (which marks
 * "rejected" and is undoable), and drag endpoints on.
 *
 * Coordinate spaces (read this before touching anything):
 *   - World coords:   metres, what segments store.  Y up = north.
 *   - Raster pixel:   the raster PNG's pixel grid.  The on-disk PNG was
 *                     Y-flipped for display, so raster row 0 = high world Y.
 *                     Conversion: pixel_col = (x_world - origin_x) / res
 *                                 pixel_row = H - 1 - (y_world - origin_y) / res
 *   - SVG viewBox:    same as raster pixel space — the SVG's viewBox is
 *                     "0 0 W H".  Lets us draw segments in pixel coords and
 *                     <image> the raster at (0,0,W,H) without any maths.
 *   - Screen pixel:   what the user's mouse moves in.  Mapped to viewBox via
 *                     the wrapper <g>'s pan/zoom transform.
 *
 * The mouse-coord-to-world conversion is implemented in :func:`screenToWorld`
 * below — change one, change the other.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useEditorStore, type EditorSegment } from '../../../state/editorStore';
import { vectorizeApi, type RasterAffine } from '../../../api/vectorize';

const HIT_TOLERANCE_PX = 6;          // mouse must be within this many svg-pixels of a segment to hit it
const ENDPOINT_PICK_PX = 10;         // distance to grab an endpoint vs the body of the segment
const SNAP_RADIUS_PX = 12;           // snap radius (in svg-pixels) when dragging an endpoint

interface Props {
  jobId: string;
  affine: RasterAffine;
  rasterUrl: string;
}

export function EditorCanvas({ jobId: _jobId, affine, rasterUrl }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const W = affine.width_px;
  const H = affine.height_px;
  const RES = affine.resolution_m_per_px;
  const OX = affine.origin_x;
  const OY = affine.origin_y;

  const segments = useEditorStore((s) => s.segments);
  const selectedIds = useEditorStore((s) => s.selectedIds);
  const hoverId = useEditorStore((s) => s.hoverId);
  const selectOnly = useEditorStore((s) => s.selectOnly);
  const toggleSelected = useEditorStore((s) => s.toggleSelected);
  const clearSelection = useEditorStore((s) => s.clearSelection);
  const setHover = useEditorStore((s) => s.setHover);
  const moveEndpoint = useEditorStore((s) => s.moveEndpoint);

  // ── World ↔ SVG-pixel conversions ───────────────────────────────────────
  const worldToPx = useCallback(
    (xw: number, yw: number): [number, number] => {
      const col = (xw - OX) / RES;
      const row = H - 1 - (yw - OY) / RES;
      return [col, row];
    },
    [OX, OY, RES, H],
  );

  const pxToWorld = useCallback(
    (col: number, row: number): [number, number] => {
      const xw = OX + col * RES;
      const yw = OY + (H - 1 - row) * RES;
      return [xw, yw];
    },
    [OX, OY, RES, H],
  );

  // ── Pan/zoom of the SVG content via a wrapper <g transform=...> ─────────
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });
  const [panning, setPanning] = useState(false);
  const panStart = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);

  // Fit the raster into the container on first mount and on resize.
  const fit = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const c = el.getBoundingClientRect();
    const pad = 40;
    const scale = Math.min((c.width - pad) / W, (c.height - pad) / H);
    setView({
      x: (c.width - W * scale) / 2,
      y: (c.height - H * scale) / 2,
      scale,
    });
  }, [W, H]);

  useEffect(() => { fit(); }, [fit]);
  useEffect(() => {
    const onResize = () => fit();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [fit]);

  // Wheel = zoom toward cursor.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const dz = -e.deltaY * 0.0015;
      const newScale = Math.max(0.05, Math.min(40, view.scale * (1 + dz)));
      const ratio = newScale / view.scale;
      setView({
        x: cx - (cx - view.x) * ratio,
        y: cy - (cy - view.y) * ratio,
        scale: newScale,
      });
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, [view]);

  // ── Coordinate conversion: screen pixel → SVG-pixel (i.e. raster col/row) ──
  const screenToSvg = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      const el = containerRef.current;
      if (!el) return [0, 0];
      const rect = el.getBoundingClientRect();
      const lx = clientX - rect.left;
      const ly = clientY - rect.top;
      const col = (lx - view.x) / view.scale;
      const row = (ly - view.y) / view.scale;
      return [col, row];
    },
    [view],
  );

  // ── Hit-testing ────────────────────────────────────────────────────────
  /**
   * Returns the nearest segment within HIT_TOLERANCE_PX of (col,row) in svg
   * pixels, or null.  Also returns which endpoint (if any) is being grabbed.
   * Hit tolerance is divided by view.scale so it stays mouse-pixel-constant.
   */
  const hitTest = useCallback(
    (col: number, row: number): { segId: string; endpoint: 0 | 1 | null; distance: number } | null => {
      const tol = HIT_TOLERANCE_PX / view.scale;
      const endTol = ENDPOINT_PICK_PX / view.scale;
      let best: { segId: string; endpoint: 0 | 1 | null; distance: number } | null = null;

      for (const seg of segments) {
        if (seg.status !== 'active') continue;
        const [x1, y1] = worldToPx(seg.x1, seg.y1);
        const [x2, y2] = worldToPx(seg.x2, seg.y2);

        const d1 = Math.hypot(col - x1, row - y1);
        const d2 = Math.hypot(col - x2, row - y2);

        if (d1 < endTol && (!best || d1 < best.distance)) {
          best = { segId: seg.id, endpoint: 0, distance: d1 };
        }
        if (d2 < endTol && (!best || d2 < best.distance)) {
          best = { segId: seg.id, endpoint: 1, distance: d2 };
        }
        if (best && best.endpoint !== null) continue;

        // Distance from point to line segment.
        const dx = x2 - x1;
        const dy = y2 - y1;
        const lenSq = dx * dx + dy * dy;
        if (lenSq < 1e-9) continue;
        let t = ((col - x1) * dx + (row - y1) * dy) / lenSq;
        t = Math.max(0, Math.min(1, t));
        const px = x1 + t * dx;
        const py = y1 + t * dy;
        const dist = Math.hypot(col - px, row - py);
        if (dist < tol && (!best || dist < best.distance)) {
          best = { segId: seg.id, endpoint: null, distance: dist };
        }
      }
      return best;
    },
    [segments, view.scale, worldToPx],
  );

  // ── Drag state for endpoint editing ─────────────────────────────────────
  const [dragging, setDragging] = useState<
    | null
    | { segId: string; endpoint: 0 | 1; cursorPx: [number, number] }
  >(null);

  // ── Pointer event handlers ──────────────────────────────────────────────
  const onPointerDown = (e: React.PointerEvent) => {
    // Right click / middle click = pan
    if (e.button === 1 || e.button === 2) {
      setPanning(true);
      panStart.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
      e.preventDefault();
      return;
    }
    if (e.button !== 0) return;

    const [col, row] = screenToSvg(e.clientX, e.clientY);
    const hit = hitTest(col, row);

    if (!hit) {
      // Empty click → pan with left-button if not on a segment, then clear selection on up.
      setPanning(true);
      panStart.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
      return;
    }

    if (hit.endpoint !== null && !e.shiftKey) {
      // Begin endpoint drag.
      setDragging({ segId: hit.segId, endpoint: hit.endpoint, cursorPx: [col, row] });
      selectOnly(hit.segId);
      return;
    }

    if (e.shiftKey) toggleSelected(hit.segId);
    else selectOnly(hit.segId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (panning && panStart.current) {
      // Snapshot the drag origin before invoking setState — React 18 can defer
      // the updater past the time onPointerUp nulls panStart.current.
      const start = panStart.current;
      const dx = e.clientX - start.x;
      const dy = e.clientY - start.y;
      setView((v) => ({ ...v, x: start.vx + dx, y: start.vy + dy }));
      return;
    }

    if (dragging) {
      const [col, row] = screenToSvg(e.clientX, e.clientY);
      // Snap to nearest other-segment endpoint within SNAP_RADIUS_PX.
      const snap = nearestEndpoint(col, row, segments, dragging.segId, worldToPx, SNAP_RADIUS_PX / view.scale);
      setDragging({ ...dragging, cursorPx: snap ? snap : [col, row] });
      return;
    }

    // Hover highlight.
    const [col, row] = screenToSvg(e.clientX, e.clientY);
    const hit = hitTest(col, row);
    setHover(hit ? hit.segId : null);
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (dragging) {
      const [col, row] = dragging.cursorPx;
      const [xw, yw] = pxToWorld(col, row);
      moveEndpoint(dragging.segId, dragging.endpoint, [xw, yw]);
      setDragging(null);
      panStart.current = null;
      setPanning(false);
      return;
    }

    if (panning) {
      // Decide: was this a click on empty space (deselect) or an actual drag (keep selection)?
      const moved =
        panStart.current &&
        (Math.abs(e.clientX - panStart.current.x) > 3 ||
          Math.abs(e.clientY - panStart.current.y) > 3);
      if (!moved && e.button === 0) clearSelection();
      panStart.current = null;
      setPanning(false);
    }
  };

  // ── Render-time projections ─────────────────────────────────────────────
  const projected = useMemo(() => {
    return segments.map((seg) => {
      const [x1p, y1p] = worldToPx(seg.x1, seg.y1);
      const [x2p, y2p] = worldToPx(seg.x2, seg.y2);
      return { seg, x1: x1p, y1: y1p, x2: x2p, y2: y2p };
    });
  }, [segments, worldToPx]);

  // ── Cursor logic ────────────────────────────────────────────────────────
  const cursor = panning
    ? 'grabbing'
    : dragging
      ? 'crosshair'
      : hoverId
        ? 'pointer'
        : 'grab';

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full overflow-hidden bg-gray-950 select-none"
      style={{ cursor }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={onPointerUp}
      onContextMenu={(e) => e.preventDefault()}
    >
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: W * view.scale,
          height: H * view.scale,
          transform: `translate(${view.x}px, ${view.y}px)`,
          transformOrigin: '0 0',
          pointerEvents: 'none', // pointer events handled on the container; SVG is purely visual
        }}
      >
        <image href={rasterUrl} x={0} y={0} width={W} height={H} preserveAspectRatio="none" />

        {/* Faint border so the raster bounds are visible even when zoomed in. */}
        <rect x={0} y={0} width={W} height={H} fill="none" stroke="#1f2937" strokeWidth={1 / view.scale} />

        {/* Rejected segments — drawn faded behind the active ones. */}
        {projected
          .filter((p) => p.seg.status === 'rejected')
          .map((p) => (
            <line
              key={p.seg.id}
              x1={p.x1} y1={p.y1} x2={p.x2} y2={p.y2}
              stroke="#dc2626"
              strokeOpacity={0.35}
              strokeDasharray={`${4 / view.scale} ${4 / view.scale}`}
              strokeWidth={1 / view.scale}
            />
          ))}

        {/* Active segments. */}
        {projected
          .filter((p) => p.seg.status === 'active')
          .map((p) => {
            const isSelected = selectedIds.has(p.seg.id);
            const isHover = hoverId === p.seg.id;
            const stroke = isSelected ? '#facc15' : isHover ? '#34d399' : '#22c55e';
            const sw = (isSelected ? 2.5 : isHover ? 2 : 1.5) / view.scale;
            return (
              <g key={p.seg.id}>
                <line
                  x1={p.x1} y1={p.y1} x2={p.x2} y2={p.y2}
                  stroke={stroke}
                  strokeWidth={sw}
                  strokeLinecap="round"
                />
                {isSelected && (
                  <>
                    <Endpoint cx={p.x1} cy={p.y1} scale={view.scale} active={dragging?.segId === p.seg.id && dragging.endpoint === 0} />
                    <Endpoint cx={p.x2} cy={p.y2} scale={view.scale} active={dragging?.segId === p.seg.id && dragging.endpoint === 1} />
                  </>
                )}
              </g>
            );
          })}

        {/* Ghost line during endpoint drag — shows where the endpoint will land on release. */}
        {dragging && (() => {
          const target = segments.find((s) => s.id === dragging.segId);
          if (!target) return null;
          const fixedIdx = dragging.endpoint === 0 ? 1 : 0;
          const [fx, fy] = fixedIdx === 0
            ? worldToPx(target.x1, target.y1)
            : worldToPx(target.x2, target.y2);
          const [cx, cy] = dragging.cursorPx;
          return (
            <g pointerEvents="none">
              <line
                x1={fx} y1={fy} x2={cx} y2={cy}
                stroke="#f59e0b"
                strokeWidth={2 / view.scale}
                strokeDasharray={`${3 / view.scale} ${3 / view.scale}`}
              />
              <circle cx={cx} cy={cy} r={5 / view.scale} fill="#f59e0b" />
            </g>
          );
        })()}
      </svg>

      {/* Top-right view controls. */}
      <div className="absolute top-3 right-3 flex flex-col gap-2 items-end pointer-events-auto">
        <button
          onClick={fit}
          className="px-3 py-1.5 rounded-lg bg-gray-900/80 hover:bg-gray-800 border border-gray-700 text-xs text-gray-300 backdrop-blur-sm transition-colors"
          title="Fit raster to viewport (F)"
        >
          ⊞ Fit
        </button>
        <div className="px-3 py-1 rounded-md bg-gray-900/80 backdrop-blur-sm border border-gray-700 text-[11px] text-gray-400 font-mono">
          {(view.scale * 100).toFixed(0)}%
        </div>
      </div>

      {/* Bottom-left help. */}
      <div className="absolute bottom-3 left-3 rounded-lg bg-gray-900/80 backdrop-blur-sm border border-gray-700 px-3 py-1.5 text-[11px] text-gray-400 pointer-events-none">
        click select · shift+click multi · delete = reject · drag endpoints · scroll = zoom · right-drag = pan
      </div>
    </div>
  );
}

function Endpoint({ cx, cy, scale, active }: { cx: number; cy: number; scale: number; active: boolean }) {
  return (
    <circle
      cx={cx}
      cy={cy}
      r={(active ? 6 : 4) / scale}
      fill={active ? '#f59e0b' : '#facc15'}
      stroke="#0f172a"
      strokeWidth={1 / scale}
    />
  );
}

/**
 * Snap target: nearest endpoint of any *other* active segment to (col, row).
 * Returns its pixel position or null if nothing is within radius.
 */
function nearestEndpoint(
  col: number,
  row: number,
  segments: EditorSegment[],
  excludeId: string,
  worldToPx: (x: number, y: number) => [number, number],
  radius: number,
): [number, number] | null {
  let best: { d: number; pt: [number, number] } | null = null;
  for (const seg of segments) {
    if (seg.id === excludeId || seg.status !== 'active') continue;
    const [x1, y1] = worldToPx(seg.x1, seg.y1);
    const [x2, y2] = worldToPx(seg.x2, seg.y2);
    const d1 = Math.hypot(col - x1, row - y1);
    if (d1 < radius && (!best || d1 < best.d)) best = { d: d1, pt: [x1, y1] };
    const d2 = Math.hypot(col - x2, row - y2);
    if (d2 < radius && (!best || d2 < best.d)) best = { d: d2, pt: [x2, y2] };
  }
  return best ? best.pt : null;
}
