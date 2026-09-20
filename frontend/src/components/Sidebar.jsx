import React, { useState } from "react";
import {
  Layers,
  Database,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Info,
  Search,
  Crosshair,
  ScanSearch,
  ChevronRight,
} from "lucide-react";

const fmt = (n) => (n ?? 0).toLocaleString();

// Status line + colour tone for each stage of the background embedding job
function describeEmbedding(es) {
  switch (es?.state) {
    case "queued":
      return { label: "Queued...", tone: "busy" };
    case "tiling":
      return { label: `Tiling... (${fmt(es.crops_generated)} crops generated)`, tone: "busy" };
    case "embedding":
      return {
        label: `Embedding... (${fmt(es.crops_embedded)}/${fmt(es.crops_total)} crops processed)`,
        tone: "busy",
      };
    case "indexing":
      return { label: "Indexing...", tone: "busy" };
    case "ready":
      return es.tiles_count > 0
        ? { label: "Ready & Embedded", tone: "ready" }
        : { label: "Ready (no valid crops to embed)", tone: "neutral" };
    case "failed":
      return { label: "Embedding failed", tone: "error" };
    case "not_embedded":
      return { label: "Not embedded", tone: "error" };
    default:
      return { label: "Checking...", tone: "busy" };
  }
}

const TONES = {
  busy: { box: "bg-amber-50 border-amber-200", title: "text-amber-900" },
  ready: { box: "bg-emerald-50/60 border-emerald-200", title: "text-emerald-800" },
  neutral: { box: "bg-neutral-50 border-neutral-200", title: "text-neutral-800" },
  error: { box: "bg-red-50 border-red-200", title: "text-red-800" },
};

const formatBounds = (b) =>
  b && b.length === 4 ? `${b[0].toFixed(4)}, ${b[1].toFixed(4)}, ${b[2].toFixed(4)}, ${b[3].toFixed(4)}` : "—";

// Collapsed by default: scene ID + status. Expands to show the rest of the scene metadata.
function DatasetCard({ scene, embedStatus }) {
  const [open, setOpen] = useState(false);
  const { label, tone } = describeEmbedding(embedStatus);
  const styles = TONES[tone];
  const state = embedStatus?.state;
  const showBar = state === "embedding" && embedStatus.crops_total > 0;

  return (
    <div className={`border rounded-[3px] ${styles.box}`}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full flex items-start space-x-2 p-2.5 text-left"
      >
        {tone === "busy" && <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-600 shrink-0 mt-0.5" />}
        {tone === "ready" && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />}
        {tone === "error" && <AlertCircle className="w-3.5 h-3.5 text-red-600 shrink-0 mt-0.5" />}
        {tone === "neutral" && <Info className="w-3.5 h-3.5 text-neutral-500 shrink-0 mt-0.5" />}

        <div className="min-w-0 flex-1">
          <div className={`text-xs font-medium ${styles.title}`}>{label}</div>
          <div className="text-[11px] font-mono text-neutral-700 truncate mt-0.5" title={scene.scene_id}>
            {scene.scene_id}
          </div>
        </div>

        <ChevronRight
          className={`w-3.5 h-3.5 text-neutral-400 shrink-0 mt-0.5 transition-transform ${open ? "rotate-90" : ""}`}
        />
      </button>

      {showBar && (
        <div className="mx-2.5 mb-2.5 -mt-0.5 h-1 bg-amber-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-amber-500"
            style={{ width: `${Math.min((embedStatus.crops_embedded / embedStatus.crops_total) * 100, 100)}%` }}
          />
        </div>
      )}

      {/* A failure reason must stay visible even when the card is collapsed */}
      {tone === "error" && embedStatus?.error && (
        <div className="px-2.5 pb-2.5 -mt-0.5 text-[11px] text-red-700 break-words">{embedStatus.error}</div>
      )}

      {open && (
        <div className="px-2.5 pb-2.5 pt-2 border-t border-black/5 space-y-1.5 font-mono text-[11px] text-neutral-600">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-neutral-400 text-[10px] uppercase font-sans">CRS</span>
              <div className="text-neutral-800 font-medium">{scene.crs || "Unknown"}</div>
            </div>
            <div>
              <span className="text-neutral-400 text-[10px] uppercase font-sans">Sensor</span>
              <div className="text-neutral-800 font-medium">{scene.sensor || "Unknown"}</div>
            </div>
          </div>
          <div>
            <span className="text-neutral-400 text-[10px] uppercase font-sans">Bounds (WGS84)</span>
            <div className="text-neutral-800 break-words">{formatBounds(scene.bounds_wgs84)}</div>
          </div>
          {state === "ready" && embedStatus.tiles_count > 0 && (
            <div>
              <span className="text-neutral-400 text-[10px] uppercase font-sans">Embedding Crops</span>
              <div className="text-neutral-800">
                {fmt(embedStatus.tiles_count)} crops (224×224)
                {embedStatus.device ? ` · ${embedStatus.device}` : ""}
                {embedStatus.crops_per_sec ? ` · ${embedStatus.crops_per_sec} crops/s` : ""}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function Sidebar({
  ingestState,
  currentScene,
  embedStatus = null,
  catalog = null,
  catalogError = null,
  error,
  searchError = null,
  searchResults = [],
  searchQuery = "",
  isSearching = false,
  selectedTileId = null,
  onSelectResult,
}) {
  const { status, fileName } = ingestState;

  return (
    <aside className="w-80 bg-white border-l border-qgis-border flex flex-col shrink-0 select-none text-qgis-text h-full overflow-hidden">
      {/* Sidebar Header */}
      <div className="h-9 px-3 border-b border-qgis-border bg-qgis-header flex items-center justify-between shrink-0">
        <span className="font-semibold text-xs text-neutral-800 flex items-center space-x-1.5">
          <Layers className="w-3.5 h-3.5 text-neutral-600" />
          <span>Workspace</span>
        </span>
        <span className="text-[11px] font-mono text-neutral-500">v0.2.0</span>
      </div>

      <div className="flex-1 overflow-y-auto divide-y divide-qgis-border">
        {/* Dataset & Ingestion Status Section */}
        <div className="p-3 bg-neutral-50/50">
          <div className="text-[11px] font-semibold text-neutral-500 uppercase tracking-wider mb-2 flex items-center space-x-1">
            <Database className="w-3 h-3 text-neutral-400" />
            <span>Active Dataset</span>
          </div>

          {status === "ingesting" && (
            <div className="bg-amber-50 border border-amber-200 rounded-[3px] p-2.5 flex items-start space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-amber-600 mt-0.5 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium text-amber-900">Importing Scene...</div>
                <div className="text-[11px] text-amber-700 truncate font-mono mt-0.5" title={fileName}>
                  {fileName}
                </div>
              </div>
            </div>
          )}

          {status === "ready" && currentScene && (
            <DatasetCard key={currentScene.scene_id} scene={currentScene} embedStatus={embedStatus} />
          )}

          {status === "error" && (
            <div className="bg-red-50 border border-red-200 rounded-[3px] p-2.5 flex items-start space-x-2 text-xs text-red-800">
              <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
              <div className="min-w-0 flex-1">
                <div className="font-medium">Ingestion Failed</div>
                <div className="text-[11px] text-red-600 mt-0.5 break-words">{error || "Failed to process image"}</div>
              </div>
            </div>
          )}

          {status === "idle" && !catalog && (
            <div className="border border-dashed border-neutral-300 rounded-[3px] p-3 text-center text-neutral-500">
              {catalogError ? (
                <>
                  <AlertCircle className="w-4 h-4 mx-auto mb-1 text-red-500" />
                  <p className="text-xs text-red-700">{catalogError}</p>
                  <p className="text-[11px] text-neutral-400 mt-0.5">Start the backend, then reload.</p>
                </>
              ) : (
                <>
                  <Loader2 className="w-4 h-4 mx-auto mb-1 animate-spin text-neutral-400" />
                  <p className="text-xs">Loading catalog...</p>
                </>
              )}
            </div>
          )}

          {status === "idle" && catalog && (
            <div className="border border-dashed border-neutral-300 rounded-[3px] p-3 text-center text-neutral-500">
              <Info className="w-4 h-4 mx-auto mb-1 text-neutral-400" />
              <p className="text-xs">
                {catalog.has_scenes ? "No scene available to resume." : "No active scene."}
              </p>
              <p className="text-[11px] text-neutral-400 mt-0.5">
                {catalog.has_scenes
                  ? "The COG for the latest scene is missing. Re-import it via Browse."
                  : "Click Browse in toolbar to open GeoTIFF / JP2."}
              </p>
            </div>
          )}

          {catalog && (
            <div
              className="mt-2 text-[11px] font-mono text-neutral-500"
              title={catalog.data_dir ? `Catalog: ${catalog.data_dir}` : undefined}
            >
              {fmt(catalog.scene_count)} {catalog.scene_count === 1 ? "scene" : "scenes"} · {fmt(catalog.faiss_vectors)}{" "}
              vectors indexed
            </div>
          )}
        </div>

        {/* Semantic Search Results Section */}
        <div className="p-3">
          <div className="text-[11px] font-semibold text-neutral-500 uppercase tracking-wider mb-2 flex items-center justify-between">
            <span className="flex items-center space-x-1">
              <ScanSearch className="w-3 h-3 text-neutral-400" />
              <span>Semantic Search Results</span>
            </span>
            {searchResults.length > 0 && (
              <span className="bg-neutral-100 text-neutral-700 px-1.5 py-0.2 rounded text-[10px] font-mono">
                {searchResults.length}
              </span>
            )}
          </div>

          {isSearching && (
            <div className="bg-neutral-50 border border-neutral-200 rounded-[3px] p-4 text-center">
              <Loader2 className="w-5 h-5 animate-spin text-neutral-600 mx-auto mb-1.5" />
              <div className="text-xs font-medium text-neutral-700">Encoding Query via RemoteCLIP...</div>
              <div className="text-[11px] text-neutral-400 mt-0.5">Searching FAISS HNSW index</div>
            </div>
          )}

          {!isSearching && searchResults.length > 0 && (
            <div className="space-y-2">
              <div className="text-[11px] text-neutral-500 font-sans italic truncate" title={searchQuery}>
                Query: "{searchQuery}"
              </div>

              <div className="space-y-1.5">
                {searchResults.map((res, idx) => {
                  const isSelected = selectedTileId === res.tile_id;
                  const scorePercent = (res.score * 100).toFixed(1);

                  return (
                    <button
                      key={res.tile_id || idx}
                      type="button"
                      onClick={() => onSelectResult && onSelectResult(res)}
                      className={`w-full text-left p-2 rounded-[3px] border transition-all ${
                        isSelected
                          ? "bg-emerald-50/80 border-emerald-400 ring-1 ring-emerald-400 shadow-sm"
                          : "bg-white border-neutral-200 hover:border-neutral-300 hover:bg-neutral-50/80"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-1.5">
                          <span className="w-4.5 h-4.5 bg-neutral-900 text-white rounded-[2px] text-[10px] font-mono font-bold flex items-center justify-center shrink-0">
                            #{idx + 1}
                          </span>
                          <span className="text-xs font-semibold text-neutral-800">
                            Match Score: {scorePercent}%
                          </span>
                        </div>
                        <Crosshair className={`w-3.5 h-3.5 ${isSelected ? "text-emerald-600" : "text-neutral-400"}`} />
                      </div>

                      {/* Progress confidence bar */}
                      <div className="w-full bg-neutral-100 h-1.5 rounded-full mt-1.5 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            res.score >= 0.8
                              ? "bg-emerald-500"
                              : res.score >= 0.6
                              ? "bg-teal-500"
                              : "bg-amber-500"
                          }`}
                          style={{ width: `${Math.min(Math.max(res.score * 100, 0), 100)}%` }}
                        />
                      </div>

                      <div className="mt-1.5 font-mono text-[10px] text-neutral-500 space-y-0.5">
                        <div className="truncate">
                          <span className="font-sans text-neutral-400">Scene: </span>
                          <span className="text-neutral-700">{res.scene_id}</span>
                        </div>
                        <div className="truncate">
                          <span className="font-sans text-neutral-400">Tile: </span>
                          <span className="text-neutral-700">{res.tile_id}</span>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {!isSearching && searchError && (
            <div className="bg-red-50 border border-red-200 rounded-[3px] p-2.5 flex items-start space-x-2 text-xs text-red-800">
              <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
              <div className="min-w-0 flex-1">
                <div className="font-medium">Search failed</div>
                <div className="text-[11px] text-red-600 mt-0.5 break-words">{searchError}</div>
              </div>
            </div>
          )}

          {!isSearching && !searchError && searchResults.length === 0 && (
            <div className="border border-dashed border-neutral-200 rounded-[3px] p-3 text-center text-neutral-400">
              <Search className="w-4 h-4 mx-auto mb-1 text-neutral-300" />
              <p className="text-xs">No active search results.</p>
              <p className="text-[11px] text-neutral-400 mt-0.5">Enter a query in the bottom search bar.</p>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
