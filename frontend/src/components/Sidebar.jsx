import React, { useState, useEffect } from "react";
import { Layers, AlertCircle, Loader2, Search, Crosshair, ScanSearch } from "lucide-react";
import ChangeResults from "./ChangeResults";
import SectionHeader from "./SectionHeader";
import WorkspaceStatus from "./WorkspaceStatus";

export default function Sidebar({
  ingestState,
  currentScene,
  scenes = [],
  embedStatus = null,
  catalog = null,
  catalogError = null,
  error,
  pipeline = null,
  onDismissPipeline,
  // semantic results of the last search
  searchError = null,
  searchResults = [],
  searchQuery = "",
  isSearching = false,
  selectedTileId = null,
  onSelectResult,
  // change results: the unified search's change half, and the browsable list of every detected change
  search = null,
  searchCount = 0,
  changes = null,
  changeJobs = [],
  changeFilters,
  onChangeFilters,
  changeView = "all",
  onChangeView,
  selectedChangeId = null,
  onOpenChange,
}) {
  // Search results and change results collapse independently. Until the analyst toggles one, the semantic tiles
  // start collapsed when change results exist (they are the main output) and everything starts expanded otherwise.
  // A new search re-applies those defaults to its own results.
  const [semanticOverride, setSemanticOverride] = useState(null);
  const [changeOverride, setChangeOverride] = useState(null);
  useEffect(() => {
    setSemanticOverride(null);
    setChangeOverride(null);
  }, [searchCount]);

  const hasChangeResults = search?.hasSearched ? (search.results?.length ?? 0) > 0 : (changes?.total ?? 0) > 0;
  const semanticOpen = semanticOverride ?? !hasChangeResults;
  const changeOpen = changeOverride ?? true;

  return (
    <aside className="w-80 bg-white border-l border-qgis-border flex flex-col shrink-0 select-none text-qgis-text h-full overflow-hidden">
      {/* Sidebar Header */}
      <div className="h-9 px-3 border-b border-qgis-border bg-qgis-header flex items-center justify-between shrink-0">
        <span className="font-semibold text-xs text-neutral-800 flex items-center space-x-1.5">
          <Layers className="w-3.5 h-3.5 text-neutral-600" />
          <span>Workspace</span>
        </span>
        <span className="text-[11px] font-mono text-neutral-500">v0.3.0</span>
      </div>

      <div className="flex-1 overflow-y-auto divide-y divide-qgis-border">
        {/* Import pipeline, scene count and scene list: one section, one source of truth */}
        <WorkspaceStatus
          pipeline={pipeline}
          onDismissPipeline={onDismissPipeline}
          ingestState={ingestState}
          error={error}
          catalog={catalog}
          catalogError={catalogError}
          scenes={scenes}
          currentScene={currentScene}
          embedStatus={embedStatus}
        />

        {/* Semantic Results Section */}
        <div className="p-3">
          <SectionHeader
            icon={ScanSearch}
            title="Semantic Results"
            count={searchResults.length}
            open={semanticOpen}
            onToggle={() => setSemanticOverride(!semanticOpen)}
          />

          {semanticOpen && (
            <>
              {isSearching && (
                <div className="bg-neutral-50 border border-neutral-200 rounded-[3px] p-4 text-center">
                  <Loader2 className="w-5 h-5 animate-spin text-neutral-600 mx-auto mb-1.5" />
                  <div className="text-xs font-medium text-neutral-700">Encoding Query via RemoteCLIP...</div>
                  <div className="text-[11px] text-neutral-400 mt-0.5">Searching tiles and detected changes</div>
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
                              <span className="text-xs font-semibold text-neutral-800">Match Score: {scorePercent}%</span>
                            </div>
                            <Crosshair className={`w-3.5 h-3.5 ${isSelected ? "text-emerald-600" : "text-neutral-400"}`} />
                          </div>

                          {/* Progress confidence bar */}
                          <div className="w-full bg-neutral-100 h-1.5 rounded-full mt-1.5 overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                res.score >= 0.8 ? "bg-emerald-500" : res.score >= 0.6 ? "bg-teal-500" : "bg-amber-500"
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
            </>
          )}
        </div>

        {/* Change Results Section */}
        <ChangeResults
          changes={changes}
          changeJobs={changeJobs}
          filters={changeFilters}
          onFiltersChange={onChangeFilters}
          search={search}
          view={changeView}
          onViewChange={onChangeView}
          selectedChangeId={selectedChangeId}
          onOpenChange={onOpenChange}
          open={changeOpen}
          onToggle={() => setChangeOverride(!changeOpen)}
        />
      </div>
    </aside>
  );
}
