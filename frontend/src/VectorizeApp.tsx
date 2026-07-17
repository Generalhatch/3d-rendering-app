/**
 * Top-level layout for the Vectorize (scan-to-CAD) surface.
 *
 * Editor view: main area + fixed sidebar (process controls).
 * Sheet view: full-width document viewer with a slide-over deliver sidebar.
 */
import { VectorizePanel } from './components/vectorize/VectorizePanel';
import { VectorizeViewer } from './components/vectorize/VectorizeViewer';
import { EditorSurface } from './components/vectorize/editor/EditorSurface';
import { SheetPreview } from './components/vectorize/SheetPreview';
import { useVectorizeStore } from './state/vectorizeStore';
import { vectorizeApi } from './api/vectorize';

export default function VectorizeApp() {
  const phase = useVectorizeStore((s) => s.phase);
  const jobId = useVectorizeStore((s) => s.jobId);
  const viewMode = useVectorizeStore((s) => s.viewMode);
  const setViewMode = useVectorizeStore((s) => s.setViewMode);
  const sheetSidebarOpen = useVectorizeStore((s) => s.sheetSidebarOpen);
  const setSheetSidebarOpen = useVectorizeStore((s) => s.setSheetSidebarOpen);
  const setSidebarTab = useVectorizeStore((s) => s.setSidebarTab);
  const sheetVersion = useVectorizeStore((s) => s.sheetVersion);
  const isComplete = phase === 'complete' && jobId !== null;
  const isSheet = isComplete && viewMode === 'sheet';

  const openDeliverSidebar = () => {
    setSidebarTab('deliver');
    setSheetSidebarOpen(true);
  };

  return (
    <div className="relative h-full flex bg-[#f7f7f5]">
      <div className={`relative flex-1 min-w-0 flex flex-col overflow-hidden ${isSheet ? 'bg-white' : 'bg-gray-950'}`}>
        {isComplete && jobId ? (
          <>
            <div
              className={`flex-shrink-0 flex items-center justify-between px-3 h-11 border-b z-10 ${
                isSheet
                  ? 'bg-white border-gray-200 text-gray-800'
                  : 'bg-gray-900/95 border-gray-700 text-gray-200'
              }`}
            >
              <div className={`flex rounded-md overflow-hidden border text-xs font-medium ${
                isSheet ? 'border-gray-300' : 'border-gray-600'
              }`}>
                <button
                  onClick={() => setViewMode('editor')}
                  className={
                    viewMode === 'editor'
                      ? 'px-3 py-1.5 bg-gray-800 text-white'
                      : isSheet
                        ? 'px-3 py-1.5 bg-white text-gray-600 hover:bg-gray-50'
                        : 'px-3 py-1.5 text-gray-300 hover:bg-gray-800'
                  }
                >
                  Editor
                </button>
                <button
                  onClick={() => setViewMode('sheet')}
                  className={
                    viewMode === 'sheet'
                      ? 'px-3 py-1.5 bg-gray-800 text-white'
                      : isSheet
                        ? 'px-3 py-1.5 bg-white text-gray-600 hover:bg-gray-50'
                        : 'px-3 py-1.5 text-gray-300 hover:bg-gray-800'
                  }
                >
                  Sheet
                </button>
              </div>

              {isSheet && (
                <div className="flex items-center gap-2 text-xs">
                  <button
                    type="button"
                    onClick={openDeliverSidebar}
                    className="rounded-md px-3 py-1.5 border border-gray-300 bg-white hover:bg-gray-50 text-gray-700"
                  >
                    Deliver
                  </button>
                  <a
                    href={vectorizeApi.sheetPdfUrl(jobId)}
                    download
                    className="rounded-md px-3 py-1.5 border border-gray-300 bg-white hover:bg-gray-50 text-gray-700"
                  >
                    Download PDF
                  </a>
                </div>
              )}
            </div>

            <div className="flex-1 min-h-0 relative">
              {viewMode === 'sheet' ? (
                <SheetPreview jobId={jobId} sheetVersion={sheetVersion} />
              ) : (
                <EditorSurface jobId={jobId} />
              )}
            </div>
          </>
        ) : (
          <VectorizeViewer />
        )}
      </div>

      {!isSheet && (
        <div className="w-[380px] flex-shrink-0 flex flex-col border-l border-gray-800 bg-gray-900 overflow-y-auto text-gray-100">
          <VectorizePanel />
        </div>
      )}

      {isSheet && sheetSidebarOpen && (
        <>
          <button
            type="button"
            aria-label="Close sidebar"
            className="absolute inset-0 z-20 bg-black/15"
            onClick={() => setSheetSidebarOpen(false)}
          />
          <div className="absolute top-11 right-0 bottom-0 w-[360px] z-30 flex flex-col border-l border-gray-200 bg-[#f7f7f5] shadow-xl overflow-y-auto">
            <VectorizePanel variant="sheet" />
          </div>
        </>
      )}
    </div>
  );
}
