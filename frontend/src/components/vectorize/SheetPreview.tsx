/**
 * Inline SVG floor plan viewer with pan/zoom (Phase 4.6).
 *
 * Fetches the same SVG the backend writes to sheet.svg and renders it inline
 * on a white canvas — crisp at any zoom, never rasterized through <img>.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { vectorizeApi } from '../../api/vectorize';

interface Props {
  jobId: string;
  /** Bump after overrides save to reload the SVG. */
  sheetVersion?: number;
}

const MIN_ZOOM = 0.05;
const MAX_ZOOM = 8;

function parseViewBox(svgText: string): { w: number; h: number } | null {
  const match = svgText.match(/viewBox=["']([^"']+)["']/);
  if (!match) return null;
  const parts = match[1].trim().split(/\s+/).map(Number);
  if (parts.length !== 4 || parts.some((n) => !Number.isFinite(n))) return null;
  return { w: parts[2], h: parts[3] };
}

export function SheetPreview({ jobId, sheetVersion = 0 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgHostRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);

  const [svgContent, setSvgContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  const fitToView = useCallback(() => {
    const container = containerRef.current;
    const host = svgHostRef.current;
    if (!container || !host || !svgContent) return;
    const vb = parseViewBox(svgContent);
    const svg = host.querySelector('svg');
    if (!vb || !svg) return;
    const pad = 32;
    const availW = container.clientWidth - pad * 2;
    const availH = container.clientHeight - pad * 2;
    const scale = Math.min(availW / vb.w, availH / vb.h, 1.5);
    setZoom(scale);
    setPan({
      x: (container.clientWidth - vb.w * scale) / 2,
      y: (container.clientHeight - vb.h * scale) / 2,
    });
    svg.style.display = 'block';
    svg.style.width = `${vb.w}mm`;
    svg.style.height = `${vb.h}mm`;
  }, [svgContent]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const text = await vectorizeApi.fetchSheetSvg(jobId, sheetVersion);
        if (!cancelled) {
          setSvgContent(text);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Sheet unavailable');
          setSvgContent(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [jobId, sheetVersion]);

  useEffect(() => {
    if (!svgContent || !svgHostRef.current) return;
    svgHostRef.current.innerHTML = svgContent;
    fitToView();
  }, [svgContent, fitToView]);

  useEffect(() => {
    const onResize = () => fitToView();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [fitToView]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    setZoom((z) => {
      const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z * factor));
      const ratio = next / z;
      setPan((p) => ({
        x: cx - (cx - p.x) * ratio,
        y: cy - (cy - p.y) * ratio,
      }));
      return next;
    });
  }, []);

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return;
    dragRef.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }, [pan.x, pan.y]);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    setPan({
      x: drag.panX + (e.clientX - drag.x),
      y: drag.panY + (e.clientY - drag.y),
    });
  }, []);

  const handlePointerUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-2 p-8 text-center bg-white">
        <p className="text-sm font-medium text-gray-700">No floor plan sheet</p>
        <p className="text-xs text-gray-500 max-w-sm">
          This job completed before sheet rendering existed, or no rooms were detected.
        </p>
        <p className="text-xs text-rose-600">{error}</p>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full bg-white">
      <div
        ref={containerRef}
        className="h-full w-full overflow-hidden cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      >
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
            Loading sheet…
          </div>
        )}
        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: '0 0',
          }}
        >
          <div ref={svgHostRef} />
        </div>
      </div>

      <div className="absolute bottom-3 left-3 flex items-center gap-2 text-xs text-gray-600 bg-white/95 border border-gray-200 rounded-md px-2 py-1 shadow-sm">
        <button
          type="button"
          onClick={() => setZoom((z) => Math.min(MAX_ZOOM, z * 1.2))}
          className="w-6 h-6 rounded border border-gray-200 hover:bg-gray-50"
          aria-label="Zoom in"
        >
          +
        </button>
        <span className="w-12 text-center tabular-nums">{Math.round(zoom * 100)}%</span>
        <button
          type="button"
          onClick={() => setZoom((z) => Math.max(MIN_ZOOM, z / 1.2))}
          className="w-6 h-6 rounded border border-gray-200 hover:bg-gray-50"
          aria-label="Zoom out"
        >
          −
        </button>
        <button
          type="button"
          onClick={fitToView}
          className="px-2 py-0.5 rounded border border-gray-200 hover:bg-gray-50"
        >
          Fit
        </button>
      </div>
    </div>
  );
}
