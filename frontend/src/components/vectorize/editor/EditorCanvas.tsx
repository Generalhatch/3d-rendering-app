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
import {
  styleForLayer,
  useEditorStore,
  type EditorSegment,
} from '../../../state/editorStore';
import type { RasterAffine, RejectedSegmentPayload } from '../../../api/vectorize';

const HIT_TOLERANCE_PX = 6;          // mouse must be within this many svg-pixels of a segment to hit it
const ENDPOINT_PICK_PX = 10;         // distance to grab an endpoint vs the body of the segment
const SNAP_RADIUS_PX = 12;           // snap radius (in svg-pixels) for endpoint-to-endpoint
const BODY_SNAP_RADIUS_PX = 10;      // additional snap radius for endpoint-to-wall-body
const AXIS_SNAP_TOLERANCE_DEG = 8;   // during draw, snap to nearest cardinal axis if within this many degrees
const GHOST_HIT_RADIUS_PX = 6;       // hit radius for clicking a ghost candidate

interface Props {
  jobId: string;
  affine: RasterAffine;
  rasterUrl: string;
  /** Optional coverage-gaps RGBA overlay (red where uncovered). */
  coverageUrl?: string | null;
}

export function EditorCanvas({ jobId: _jobId, affine, rasterUrl, coverageUrl }: Props) {
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
  const mode = useEditorStore((s) => s.mode);
  const setMode = useEditorStore((s) => s.setMode);
  const addSegment = useEditorStore((s) => s.addSegment);
  const drawLayer = useEditorStore((s) => s.drawLayer);
  const layerVisibility = useEditorStore((s) => s.layerVisibility);
  const ghosts = useEditorStore((s) => s.ghosts);
  const ghostsVisible = useEditorStore((s) => s.ghostsVisible);
  const promoteGhost = useEditorStore((s) => s.promoteGhost);
  const mergeSuggestions = useEditorStore((s) => s.mergeSuggestions);
  const mergeCursor = useEditorStore((s) => s.mergeCursor);

  // Phase D state
  const originalSegments = useEditorStore((s) => s.originalSegments);
  const showOriginal = useEditorStore((s) => s.showOriginal);
  const measurement = useEditorStore((s) => s.measurement);
  const measureAnchor = useEditorStore((s) => s.measureAnchor);
  const setMeasurement = useEditorStore((s) => s.setMeasurement);
  const setMeasureAnchor = useEditorStore((s) => s.setMeasureAnchor);

  /** Helper: is this layer currently visible?  Defaults to true. */
  const isLayerVisible = useCallback(
    (layer: string) => layerVisibility[layer] !== false,
    [layerVisibility],
  );

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
        if (!isLayerVisible(seg.layer ?? 'walls')) continue;
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
    [segments, view.scale, worldToPx, isLayerVisible],
  );

  /** Hit-test ghosts only (rejected candidates).  Returns the topmost ghost
   *  whose body is within ``GHOST_HIT_RADIUS_PX`` of the cursor, or null. */
  const hitTestGhost = useCallback(
    (col: number, row: number): RejectedSegmentPayload | null => {
      const tol = GHOST_HIT_RADIUS_PX / view.scale;
      let best: { d: number; ghost: RejectedSegmentPayload } | null = null;
      for (const g of ghosts) {
        const [x1, y1] = worldToPx(g.x1, g.y1);
        const [x2, y2] = worldToPx(g.x2, g.y2);
        const dx = x2 - x1;
        const dy = y2 - y1;
        const lenSq = dx * dx + dy * dy;
        if (lenSq < 1e-9) continue;
        let t = ((col - x1) * dx + (row - y1) * dy) / lenSq;
        t = Math.max(0, Math.min(1, t));
        const px = x1 + t * dx;
        const py = y1 + t * dy;
        const d = Math.hypot(col - px, row - py);
        if (d < tol && (!best || d < best.d)) best = { d, ghost: g };
      }
      return best?.ghost ?? null;
    },
    [ghosts, view.scale, worldToPx],
  );

  // ── Drag state for endpoint editing ─────────────────────────────────────
  const [dragging, setDragging] = useState<
    | null
    | { segId: string; endpoint: 0 | 1; cursorPx: [number, number] }
  >(null);

  // ── Draw-new-wall state ────────────────────────────────────────────────
  // ``drawAnchor`` is the svg-pixel position where the draw began; the live
  // endpoint follows the cursor in ``drawCursor`` (snapped to the same kinds
  // of targets as endpoint-drag, plus axis snap to nearest cardinal).
  const [drawAnchor, setDrawAnchor] = useState<[number, number] | null>(null);
  const [drawCursor, setDrawCursor] = useState<[number, number] | null>(null);

  // ── Measure-mode live cursor (svg pixels) ──────────────────────────────
  // After click 1 in measure mode, mouse move updates this so the operator
  // sees the candidate ruler before committing click 2.
  const [measureLive, setMeasureLive] = useState<[number, number] | null>(null);

  // ── Coverage overlay toggle (the "did we miss anything?" diagnostic) ────
  const [showCoverage, setShowCoverage] = useState(false);

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

    // ── Measure mode: click 1 = anchor, click 2 = lock + report ─────────
    if (mode === 'measure') {
      const snapped = snapDrawPoint(col, row, segments, null, worldToPx, view.scale);
      if (!measureAnchor) {
        setMeasureAnchor(snapped);
        setMeasurement(null);
      } else {
        const [ax, ay] = pxToWorld(measureAnchor[0], measureAnchor[1]);
        const [bx, by] = pxToWorld(snapped[0], snapped[1]);
        setMeasurement({ ax, ay, bx, by });
        setMeasureAnchor(null);
      }
      return;
    }

    // ── Draw mode: anchor + start sketching a new wall ───────────────────
    if (mode === 'draw') {
      const snapped = snapDrawPoint(col, row, segments, null, worldToPx, view.scale);
      setDrawAnchor(snapped);
      setDrawCursor(snapped);
      return;
    }

    // ── Ghost click takes priority over normal hit test ─────────────────
    if (ghostsVisible) {
      const ghost = hitTestGhost(col, row);
      if (ghost) {
        promoteGhost(ghost);
        return;
      }
    }

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

    if (drawAnchor) {
      const [col, row] = screenToSvg(e.clientX, e.clientY);
      const snapped = snapDrawPoint(col, row, segments, drawAnchor, worldToPx, view.scale);
      setDrawCursor(snapped);
      return;
    }

    if (mode === 'measure' && measureAnchor) {
      const [col, row] = screenToSvg(e.clientX, e.clientY);
      const snapped = snapDrawPoint(col, row, segments, measureAnchor, worldToPx, view.scale);
      setMeasureLive(snapped);
      return;
    }

    if (dragging) {
      const [col, row] = screenToSvg(e.clientX, e.clientY);
      // Snap to: (a) nearest other-segment endpoint, then (b) nearest other-
      // segment body (project onto line).  Endpoint wins on ties.
      const epSnap = nearestEndpoint(
        col, row, segments, dragging.segId, worldToPx, SNAP_RADIUS_PX / view.scale,
      );
      const bodySnap = epSnap
        ? null
        : nearestSegmentBody(
            col, row, segments, dragging.segId, worldToPx, BODY_SNAP_RADIUS_PX / view.scale,
          );
      setDragging({ ...dragging, cursorPx: epSnap ?? bodySnap ?? [col, row] });
      return;
    }

    // Hover highlight.
    const [col, row] = screenToSvg(e.clientX, e.clientY);
    const hit = hitTest(col, row);
    setHover(hit ? hit.segId : null);
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (drawAnchor && drawCursor) {
      const [c1, r1] = drawAnchor;
      const [c2, r2] = drawCursor;
      // Discard near-zero-length draws (operator misclick).
      const lenPx = Math.hypot(c2 - c1, r2 - r1);
      if (lenPx >= 6 / view.scale) {
        const [x1w, y1w] = pxToWorld(c1, r1);
        const [x2w, y2w] = pxToWorld(c2, r2);
        addSegment(x1w, y1w, x2w, y2w, drawLayer);
      }
      setDrawAnchor(null);
      setDrawCursor(null);
      return;
    }

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

  const projectedGhosts = useMemo(() => {
    if (!ghostsVisible) return [] as { ghost: RejectedSegmentPayload; x1: number; y1: number; x2: number; y2: number }[];
    return ghosts.map((g) => {
      const [x1, y1] = worldToPx(g.x1, g.y1);
      const [x2, y2] = worldToPx(g.x2, g.y2);
      return { ghost: g, x1, y1, x2, y2 };
    });
  }, [ghosts, ghostsVisible, worldToPx]);

  // Active merge suggestion preview (the pair the operator is about to act on).
  const currentSuggestion = useMemo(() => {
    if (!mergeSuggestions.length || mergeCursor >= mergeSuggestions.length) return null;
    const pair = mergeSuggestions[mergeCursor];
    const a = segments.find((s) => s.id === pair.a_id);
    const b = segments.find((s) => s.id === pair.b_id);
    if (!a || !b) return null;
    const [ax1, ay1] = worldToPx(a.x1, a.y1);
    const [ax2, ay2] = worldToPx(a.x2, a.y2);
    const [bx1, by1] = worldToPx(b.x1, b.y1);
    const [bx2, by2] = worldToPx(b.x2, b.y2);
    const [mx1, my1] = worldToPx(pair.merged_x1, pair.merged_y1);
    const [mx2, my2] = worldToPx(pair.merged_x2, pair.merged_y2);
    return { pair, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, mx1, my1, mx2, my2 };
  }, [mergeSuggestions, mergeCursor, segments, worldToPx]);

  // ── Esc cancels in-progress draw (without leaving draw mode) ────────────
  useEffect(() => {
    if (!drawAnchor) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setDrawAnchor(null);
        setDrawCursor(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [drawAnchor]);

  // Clear the measure live-cursor whenever the anchor goes away (after lock
  // or Esc) so the dashed ruler doesn't ghost on screen.
  useEffect(() => {
    if (!measureAnchor) setMeasureLive(null);
  }, [measureAnchor]);

  // ── Cursor logic ────────────────────────────────────────────────────────
  const cursor = panning
    ? 'grabbing'
    : drawAnchor || mode === 'draw' || mode === 'measure'
      ? 'crosshair'
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

        {/* Coverage-gaps overlay: red where wall pixels weren't covered by any kept segment. */}
        {showCoverage && coverageUrl && (
          <image
            href={coverageUrl}
            x={0} y={0}
            width={W} height={H}
            preserveAspectRatio="none"
            style={{ mixBlendMode: 'screen' }}
          />
        )}

        {/* Faint border so the raster bounds are visible even when zoomed in. */}
        <rect x={0} y={0} width={W} height={H} fill="none" stroke="#1f2937" strokeWidth={1 / view.scale} />

        {/* "Show original" — faint grey ghost of the pipeline's just-loaded
            output, so the operator can see at a glance how their edits diverge. */}
        {showOriginal && originalSegments.map((o) => {
          const [x1, y1] = worldToPx(o.x1, o.y1);
          const [x2, y2] = worldToPx(o.x2, o.y2);
          return (
            <line
              key={`orig-${o.id}`}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="#94a3b8"
              strokeOpacity={0.4}
              strokeWidth={1.5 / view.scale}
              strokeDasharray={`${2 / view.scale} ${3 / view.scale}`}
            />
          );
        })}

        {/* Pipeline-rejected ghost candidates — render faintly, clickable to promote. */}
        {projectedGhosts.map(({ ghost, x1, y1, x2, y2 }) => (
          <line
            key={`ghost-${ghost.id}`}
            x1={x1} y1={y1} x2={x2} y2={y2}
            stroke={ghostStroke(ghost.dropped_by)}
            strokeOpacity={0.55}
            strokeDasharray={`${5 / view.scale} ${3 / view.scale}`}
            strokeWidth={1.5 / view.scale}
          />
        ))}

        {/* Operator-rejected segments — drawn faded behind the active ones. */}
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

        {/* Active segments — colour, dash, and stroke pulled from the layer style. */}
        {projected
          .filter((p) => p.seg.status === 'active' && isLayerVisible(p.seg.layer ?? 'walls'))
          .map((p) => {
            const isSelected = selectedIds.has(p.seg.id);
            const isHover = hoverId === p.seg.id;
            const style = styleForLayer(p.seg.layer ?? 'walls');
            const stroke = isSelected ? style.selectedColor : isHover ? style.hoverColor : style.color;
            const sw = (isSelected ? 2.8 : isHover ? 2.2 : 1.8) / view.scale;
            const dash = style.dashArray
              ? style.dashArray
                  .split(/\s+/)
                  .map((n) => (Number(n) / view.scale).toString())
                  .join(' ')
              : undefined;
            return (
              <g key={p.seg.id}>
                <line
                  x1={p.x1} y1={p.y1} x2={p.x2} y2={p.y2}
                  stroke={stroke}
                  strokeWidth={sw}
                  strokeLinecap="round"
                  strokeDasharray={dash}
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

        {/* Live preview of the new segment being drawn — uses the active draw-layer style. */}
        {drawAnchor && drawCursor && (() => {
          const previewStyle = styleForLayer(drawLayer);
          const previewDash = previewStyle.dashArray
            ? previewStyle.dashArray
                .split(/\s+/)
                .map((n) => (Number(n) / view.scale).toString())
                .join(' ')
            : undefined;
          return (
            <g pointerEvents="none">
              <line
                x1={drawAnchor[0]} y1={drawAnchor[1]}
                x2={drawCursor[0]} y2={drawCursor[1]}
                stroke={previewStyle.color}
                strokeWidth={2.5 / view.scale}
                strokeLinecap="round"
                strokeDasharray={previewDash}
              />
              <circle cx={drawAnchor[0]} cy={drawAnchor[1]} r={4 / view.scale} fill={previewStyle.color} />
              <circle cx={drawCursor[0]} cy={drawCursor[1]} r={5 / view.scale} fill={previewStyle.color} />
            </g>
          );
        })()}

        {/* Measure-mode preview line (anchor → live cursor) */}
        {mode === 'measure' && measureAnchor && measureLive && (
          <g pointerEvents="none">
            <line
              x1={measureAnchor[0]} y1={measureAnchor[1]}
              x2={measureLive[0]} y2={measureLive[1]}
              stroke="#fb923c"
              strokeWidth={2 / view.scale}
              strokeDasharray={`${4 / view.scale} ${3 / view.scale}`}
            />
            <circle cx={measureAnchor[0]} cy={measureAnchor[1]} r={4 / view.scale} fill="#fb923c" />
            <circle cx={measureLive[0]} cy={measureLive[1]} r={4 / view.scale} fill="#fb923c" stroke="#0f172a" strokeWidth={1 / view.scale} />
          </g>
        )}

        {/* Locked measurement (solid ruler with endpoint pips) */}
        {measurement && (() => {
          const [ax, ay] = worldToPx(measurement.ax, measurement.ay);
          const [bx, by] = worldToPx(measurement.bx, measurement.by);
          return (
            <g pointerEvents="none">
              <line
                x1={ax} y1={ay} x2={bx} y2={by}
                stroke="#fb923c"
                strokeWidth={2 / view.scale}
              />
              <circle cx={ax} cy={ay} r={4 / view.scale} fill="#fb923c" stroke="#0f172a" strokeWidth={1 / view.scale} />
              <circle cx={bx} cy={by} r={4 / view.scale} fill="#fb923c" stroke="#0f172a" strokeWidth={1 / view.scale} />
            </g>
          );
        })()}

        {/* Merge-suggestion highlight: the two source segments + the proposed centreline. */}
        {currentSuggestion && (
          <g pointerEvents="none">
            <line
              x1={currentSuggestion.ax1} y1={currentSuggestion.ay1}
              x2={currentSuggestion.ax2} y2={currentSuggestion.ay2}
              stroke="#a855f7" strokeWidth={3 / view.scale}
              strokeOpacity={0.85}
            />
            <line
              x1={currentSuggestion.bx1} y1={currentSuggestion.by1}
              x2={currentSuggestion.bx2} y2={currentSuggestion.by2}
              stroke="#a855f7" strokeWidth={3 / view.scale}
              strokeOpacity={0.85}
            />
            <line
              x1={currentSuggestion.mx1} y1={currentSuggestion.my1}
              x2={currentSuggestion.mx2} y2={currentSuggestion.my2}
              stroke="#fbbf24" strokeWidth={2 / view.scale}
              strokeDasharray={`${6 / view.scale} ${4 / view.scale}`}
            />
          </g>
        )}
      </svg>

      {/* Top-right view controls. */}
      <div className="absolute top-3 right-3 flex flex-col gap-2 items-end pointer-events-auto">
        <button
          onClick={fit}
          className="px-3 py-1.5 rounded-lg bg-gray-900/80 hover:bg-gray-800 border border-gray-700 text-xs text-gray-300 backdrop-blur-sm transition-colors"
          title="Fit raster to viewport"
        >
          ⊞ Fit
        </button>
        {coverageUrl && (
          <button
            onClick={() => setShowCoverage((v) => !v)}
            className={`px-3 py-1.5 rounded-lg border text-xs backdrop-blur-sm transition-colors ${
              showCoverage
                ? 'bg-rose-600/80 hover:bg-rose-600 border-rose-500 text-white'
                : 'bg-gray-900/80 hover:bg-gray-800 border-gray-700 text-gray-300'
            }`}
            title="Highlight raster pixels not covered by any kept segment — i.e. potentially missed walls"
          >
            {showCoverage ? '◉ Hide coverage gaps' : '○ Show coverage gaps'}
          </button>
        )}
        <div className="px-3 py-1 rounded-md bg-gray-900/80 backdrop-blur-sm border border-gray-700 text-[11px] text-gray-400 font-mono">
          {(view.scale * 100).toFixed(0)}%
        </div>
      </div>

      {/* Bottom-left help — mode-aware. */}
      <div className="absolute bottom-3 left-3 rounded-lg bg-gray-900/80 backdrop-blur-sm border border-gray-700 px-3 py-1.5 text-[11px] text-gray-400 pointer-events-none">
        {mode === 'draw' ? (
          <span>
            <span className="text-sky-300">draw mode</span> · click-drag = new {styleForLayer(drawLayer).label.toLowerCase()} ·
            {' '}snaps to endpoints/walls/axes · D=wall · O=opening · W=window · C=column · esc cancel
          </span>
        ) : mode === 'measure' ? (
          <span>
            <span className="text-orange-300">measure mode</span> · click 2 points · snaps to endpoints/walls · R exits · esc clears
          </span>
        ) : (
          <span>click select · shift+click multi · delete = reject · drag endpoints · scroll = zoom · right-drag = pan · D wall · O opening · W window · C column · R ruler · M merge · G ghosts · F duplicates</span>
        )}
      </div>

      {/* Draw-mode banner at top centre — colour matches the active draw layer. */}
      {mode === 'draw' && (() => {
        const drawStyle = styleForLayer(drawLayer);
        return (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 pointer-events-auto">
            <div
              className="rounded-full backdrop-blur-sm border px-4 py-1.5 text-xs text-white font-semibold flex items-center gap-3 shadow-lg"
              style={{
                backgroundColor: drawStyle.color + 'e6',  // ~90% opacity
                borderColor: drawStyle.color,
              }}
            >
              <span>✎ Drawing new {drawStyle.label.toLowerCase().replace(/s$/, '')}</span>
              <button
                onClick={() => { setMode('select'); setDrawAnchor(null); setDrawCursor(null); }}
                className="text-white/90 hover:text-white text-[10px] underline"
              >
                exit (D)
              </button>
            </div>
          </div>
        );
      })()}

      {/* Merge-suggestion footer when there are pending suggestions. */}
      {mergeSuggestions.length > 0 && mergeCursor < mergeSuggestions.length && (
        <MergeSuggestionStrip />
      )}

      {/* Measurement readout — distance + bearing.  Lives top-centre so it
          doesn't fight the merge-suggestion footer or the draw banner. */}
      {(measurement || (mode === 'measure' && measureAnchor && measureLive)) && (
        <MeasurementReadout
          live={
            measurement
              ? null
              : (() => {
                  const a = measureAnchor!;
                  const b = measureLive!;
                  const [ax, ay] = pxToWorld(a[0], a[1]);
                  const [bx, by] = pxToWorld(b[0], b[1]);
                  return { ax, ay, bx, by };
                })()
          }
          locked={measurement}
          onClear={() => { setMeasurement(null); setMeasureAnchor(null); }}
        />
      )}
    </div>
  );
}

/** Top-centre toast showing distance + bearing for the active measurement. */
function MeasurementReadout({
  live, locked, onClear,
}: {
  live: { ax: number; ay: number; bx: number; by: number } | null;
  locked: { ax: number; ay: number; bx: number; by: number } | null;
  onClear: () => void;
}) {
  const m = locked ?? live!;
  const dx = m.bx - m.ax;
  const dy = m.by - m.ay;
  const dist_m = Math.hypot(dx, dy);
  // World Y grows north (per RasterAffine); bearing = compass-clockwise from +Y.
  const bearing_deg = ((Math.atan2(dx, dy) * 180) / Math.PI + 360) % 360;
  return (
    <div className="absolute top-14 left-1/2 -translate-x-1/2 pointer-events-auto">
      <div
        className={`rounded-lg backdrop-blur-sm border px-4 py-2 text-xs shadow-lg flex items-center gap-3 ${
          locked
            ? 'bg-orange-950/90 border-orange-500 text-orange-100'
            : 'bg-orange-900/80 border-orange-700 text-orange-200'
        }`}
      >
        <span className="font-mono text-base font-semibold text-amber-200">
          {dist_m < 1
            ? `${(dist_m * 100).toFixed(1)} cm`
            : `${dist_m.toFixed(3)} m`}
        </span>
        <span className="text-[11px] text-orange-300">
          Δx={dx.toFixed(2)} Δy={dy.toFixed(2)} · bearing {bearing_deg.toFixed(1)}°
        </span>
        {locked && (
          <button
            onClick={onClear}
            className="text-orange-200 hover:text-white text-[10px] underline"
          >
            clear (Esc)
          </button>
        )}
      </div>
    </div>
  );
}

/** Footer that walks the operator through queued merge suggestions. */
function MergeSuggestionStrip() {
  const mergeSuggestions = useEditorStore((s) => s.mergeSuggestions);
  const mergeCursor = useEditorStore((s) => s.mergeCursor);
  const accept = useEditorStore((s) => s.acceptCurrentMergeSuggestion);
  const skip = useEditorStore((s) => s.skipCurrentMergeSuggestion);
  const clearAll = useEditorStore((s) => s.clearMergeSuggestions);
  const current = mergeSuggestions[mergeCursor];
  if (!current) return null;

  return (
    <div className="absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-auto">
      <div className="rounded-xl bg-purple-950/90 backdrop-blur-sm border border-purple-700 px-4 py-2.5 text-xs shadow-xl flex items-center gap-3 text-purple-100">
        <span className="text-purple-300 font-mono text-[10px]">
          {mergeCursor + 1}/{mergeSuggestions.length}
        </span>
        <span className="text-purple-200">
          Near-duplicate · <span className="font-mono text-amber-300">{(current.perp_distance_m * 100).toFixed(1)} cm</span> apart
          · <span className="font-mono text-amber-300">{current.angle_diff_deg.toFixed(1)}°</span>
        </span>
        <div className="flex gap-1">
          <button
            onClick={accept}
            className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-[11px] font-semibold"
          >
            Merge
          </button>
          <button
            onClick={skip}
            className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-200 text-[11px]"
          >
            Skip
          </button>
          <button
            onClick={clearAll}
            className="px-3 py-1 rounded text-purple-300 hover:text-purple-100 text-[11px] underline"
          >
            Dismiss all
          </button>
        </div>
      </div>
    </div>
  );
}

/** Colour of a ghost candidate based on why the pipeline dropped it. */
function ghostStroke(droppedBy: string): string {
  switch (droppedBy) {
    case 'short':     return '#facc15';   // amber — under min-length cutoff
    case 'manhattan': return '#a78bfa';   // violet — off-axis
    case 'merged':    return '#6ee7b7';   // mint — already represented by a kept segment
    default:          return '#94a3b8';
  }
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
 * Returns its pixel position or null if nothing is within radius.  Pass
 * ``excludeId === null`` to consider all active segments (used by draw-mode
 * where the new segment doesn't exist yet).
 */
function nearestEndpoint(
  col: number,
  row: number,
  segments: EditorSegment[],
  excludeId: string | null,
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

/**
 * Project (col, row) onto the body (not endpoints) of every active segment
 * except ``excludeId``.  Returns the closest projected point on any wall
 * within ``radius``.  Used when the operator drags an endpoint near a wall
 * — pulls onto the wall body so corners click together cleanly.
 */
function nearestSegmentBody(
  col: number,
  row: number,
  segments: EditorSegment[],
  excludeId: string | null,
  worldToPx: (x: number, y: number) => [number, number],
  radius: number,
): [number, number] | null {
  let best: { d: number; pt: [number, number] } | null = null;
  for (const seg of segments) {
    if (seg.id === excludeId || seg.status !== 'active') continue;
    const [x1, y1] = worldToPx(seg.x1, seg.y1);
    const [x2, y2] = worldToPx(seg.x2, seg.y2);
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    if (lenSq < 1e-9) continue;
    let t = ((col - x1) * dx + (row - y1) * dy) / lenSq;
    // Restrict to the interior of the segment — endpoints are handled by
    // nearestEndpoint to keep them visually first-class.
    if (t <= 0 || t >= 1) continue;
    const px = x1 + t * dx;
    const py = y1 + t * dy;
    const d = Math.hypot(col - px, row - py);
    if (d < radius && (!best || d < best.d)) best = { d, pt: [px, py] };
  }
  return best?.pt ?? null;
}

/**
 * Compute the snapped position of the draw cursor.
 *
 * Snap priority (first match wins):
 *   1. Existing segment endpoint within SNAP_RADIUS_PX
 *   2. Existing segment body within BODY_SNAP_RADIUS_PX
 *   3. Manhattan axis relative to the draw anchor (within AXIS_SNAP_TOLERANCE_DEG)
 *
 * The first call (when ``anchor`` is null) only does endpoint+body — there's
 * no axis to snap to yet.  Subsequent calls always re-evaluate against the
 * fresh cursor position so the snap target updates as the user drags.
 */
function snapDrawPoint(
  col: number,
  row: number,
  segments: EditorSegment[],
  anchor: [number, number] | null,
  worldToPx: (x: number, y: number) => [number, number],
  scale: number,
): [number, number] {
  // 1. Endpoint snap.
  const ep = nearestEndpoint(col, row, segments, null, worldToPx, SNAP_RADIUS_PX / scale);
  if (ep) return ep;

  // 2. Body snap.
  const body = nearestSegmentBody(col, row, segments, null, worldToPx, BODY_SNAP_RADIUS_PX / scale);
  if (body) return body;

  // 3. Axis snap from the anchor (horizontal / vertical only).  In SVG-pixel
  //    space rows grow downward, but a "horizontal" line is still equal-row;
  //    treat the snap as zeroing whichever axis delta is small.
  if (anchor) {
    const dx = col - anchor[0];
    const dy = row - anchor[1];
    if (dx === 0 && dy === 0) return [col, row];
    const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
    // Distance to nearest cardinal: 0, 90, 180, -90.
    const cardinals = [0, 90, 180, -180, -90];
    let bestDelta = Infinity;
    let bestCard = 0;
    for (const c of cardinals) {
      const d = Math.abs(angle - c);
      if (d < bestDelta) {
        bestDelta = d;
        bestCard = c;
      }
    }
    if (bestDelta <= AXIS_SNAP_TOLERANCE_DEG) {
      const len = Math.hypot(dx, dy);
      const rad = (bestCard * Math.PI) / 180;
      return [anchor[0] + Math.cos(rad) * len, anchor[1] + Math.sin(rad) * len];
    }
  }

  return [col, row];
}
