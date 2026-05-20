/**
 * Image viewer for Vectorize results.
 *
 * Phase 1 keeps this simple: a pan/zoom image surface that shows one of three
 * artifacts (clean overlay / raw overlay / raster).  In Phase 3 this becomes
 * the interactive editor (select lines, drag endpoints, etc.).
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useVectorizeStore } from '../../state/vectorizeStore';
import { vectorizeApi } from '../../api/vectorize';

export function VectorizeViewer() {
  const { jobId, job, phase, overlayMode, setOverlayMode } = useVectorizeStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);

  // Pick image URL based on selected mode
  const imageUrl = jobId && job
    ? overlayMode === 'clean' && job.has_overlay
      ? vectorizeApi.overlayUrl(jobId)
      : overlayMode === 'raw' && job.has_overlay
        ? vectorizeApi.overlayRawUrl(jobId)
        : job.has_raster
          ? vectorizeApi.rasterUrl(jobId)
          : null
    : null;

  // When the image loads, fit it to the container.
  const fitToContainer = useCallback(() => {
    if (!naturalSize || !containerRef.current) return;
    const c = containerRef.current.getBoundingClientRect();
    const pad = 40;
    const scale = Math.min(
      (c.width - pad) / naturalSize.w,
      (c.height - pad) / naturalSize.h,
    );
    setTransform({
      x: (c.width - naturalSize.w * scale) / 2,
      y: (c.height - naturalSize.h * scale) / 2,
      scale,
    });
  }, [naturalSize]);

  useEffect(() => { fitToContainer(); }, [fitToContainer, imageUrl]);

  // Mouse wheel = zoom toward cursor
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handle = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const dz = -e.deltaY * 0.001;
      const newScale = Math.max(0.05, Math.min(20, transform.scale * (1 + dz)));
      // Keep the world point under the cursor fixed across the zoom.
      const ratio = newScale / transform.scale;
      setTransform({
        x: cx - (cx - transform.x) * ratio,
        y: cy - (cy - transform.y) * ratio,
        scale: newScale,
      });
    };
    el.addEventListener('wheel', handle, { passive: false });
    return () => el.removeEventListener('wheel', handle);
  }, [transform]);

  return (
    <div className="relative h-full w-full overflow-hidden bg-gray-950">
      {/* Image container with pan/zoom */}
      <div
        ref={containerRef}
        className="absolute inset-0 cursor-grab"
        style={{ cursor: dragging ? 'grabbing' : 'grab' }}
        onMouseDown={(e) => {
          setDragging(true);
          dragStart.current = { x: e.clientX, y: e.clientY, tx: transform.x, ty: transform.y };
        }}
        onMouseMove={(e) => {
          // Snapshot the drag origin into locals BEFORE invoking setState — React
          // can defer the updater, by which time onMouseUp may have nulled
          // dragStart.current (the non-null assertion would then explode).
          const start = dragStart.current;
          if (!dragging || !start) return;
          const dx = e.clientX - start.x;
          const dy = e.clientY - start.y;
          setTransform((t) => ({ ...t, x: start.tx + dx, y: start.ty + dy }));
        }}
        onMouseUp={() => { setDragging(false); dragStart.current = null; }}
        onMouseLeave={() => { setDragging(false); dragStart.current = null; }}
      >
        {imageUrl ? (
          <img
            src={imageUrl}
            alt="vectorize result"
            draggable={false}
            onLoad={(e) => {
              const im = e.currentTarget;
              setNaturalSize({ w: im.naturalWidth, h: im.naturalHeight });
            }}
            style={{
              position: 'absolute',
              top: 0, left: 0,
              transformOrigin: '0 0',
              transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
              imageRendering: transform.scale > 4 ? 'pixelated' : 'auto',
              userSelect: 'none',
              pointerEvents: 'none',
              maxWidth: 'none',
            }}
          />
        ) : (
          <EmptyState phase={phase} />
        )}
      </div>

      {/* Top-right view controls — overlay mode + fit */}
      {imageUrl && (
        <div className="absolute top-3 right-3 flex flex-col gap-2 items-end">
          <div className="rounded-lg bg-gray-900/80 backdrop-blur-sm border border-gray-700 p-1 flex gap-1">
            <ViewModeBtn
              active={overlayMode === 'clean'}
              onClick={() => setOverlayMode('clean')}
              disabled={!job?.has_overlay}
              label="Clean"
              hint="Detected lines (red) after regularization"
            />
            <ViewModeBtn
              active={overlayMode === 'raw'}
              onClick={() => setOverlayMode('raw')}
              disabled={!job?.has_overlay}
              label="Raw"
              hint="Raw detector output (orange)"
            />
            <ViewModeBtn
              active={overlayMode === 'raster'}
              onClick={() => setOverlayMode('raster')}
              disabled={!job?.has_raster}
              label="Raster"
              hint="The cleaned raster (no overlays)"
            />
          </div>
          <button
            onClick={fitToContainer}
            className="px-3 py-1.5 rounded-lg bg-gray-900/80 hover:bg-gray-800 border border-gray-700 text-xs text-gray-300 backdrop-blur-sm transition-colors"
            title="Fit image to viewport"
          >
            ⊞ Fit
          </button>
          <div className="px-3 py-1 rounded-md bg-gray-900/80 backdrop-blur-sm border border-gray-700 text-[11px] text-gray-400 font-mono">
            {(transform.scale * 100).toFixed(0)}%
          </div>
        </div>
      )}

      {/* Bottom-left scale hint */}
      {imageUrl && job?.metrics && (
        <div className="absolute bottom-3 left-3 rounded-lg bg-gray-900/80 backdrop-blur-sm border border-gray-700 px-3 py-1.5 text-[11px] text-gray-400">
          {job.metrics.raster_world_width_m.toFixed(1)} × {job.metrics.raster_world_height_m.toFixed(1)} m
          <span className="mx-2 text-gray-700">·</span>
          {job.metrics.segments_after_regularize} walls
        </div>
      )}
    </div>
  );
}

function ViewModeBtn({
  active, onClick, disabled, label, hint,
}: { active: boolean; onClick: () => void; disabled?: boolean; label: string; hint?: string }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={hint}
      className={`px-3 py-1 rounded text-xs font-medium transition-colors
        ${disabled
          ? 'text-gray-700 cursor-not-allowed'
          : active
            ? 'bg-emerald-600 text-white'
            : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'}`}
    >
      {label}
    </button>
  );
}

function EmptyState({ phase }: { phase: string }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600 gap-3">
      <div className="text-5xl">📐</div>
      <div className="text-sm">
        {phase === 'idle' && 'Drop a scan in the side panel to start'}
        {(phase === 'uploading' || phase === 'processing') && 'Running pipeline…'}
        {phase === 'failed' && 'Pipeline failed — see side panel for details'}
        {phase === 'complete' && 'Loading result…'}
      </div>
    </div>
  );
}
