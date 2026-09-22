import React from "react";
import { AlertCircle, Loader2, Search, Crosshair, Images } from "lucide-react";
import { boundsMgrs, spacedMgrs } from "../mgrs";
import { changeLabel, DIRECTION_ICONS, directionLabel, hectares } from "./changeKinds";

/** Other changes whose after-date imagery looks like the change Find Similar started from. */
function SimilarChanges({ changes, onOpenChange, selectedChangeId = null }) {
  if (!changes.length) return null;
  return (
    <div className="mb-3" data-testid="similar-changes">
      <div className="text-[10px] uppercase font-semibold text-neutral-400 mb-1">Similar changes found at</div>
      <div className="space-y-1.5">
        {changes.map((c, idx) => {
          const Icon = DIRECTION_ICONS[c.direction || "unclassified"];
          const mgrsRef = c.mgrs || boundsMgrs(c.bounds);
          const isSelected = selectedChangeId === c.candidate_id;
          const open = () => onOpenChange && onOpenChange(c.candidate_id, "similar");
          return (
            <div
              key={c.candidate_id}
              role="button"
              tabIndex={0}
              onClick={open}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  open();
                }
              }}
              data-testid="similar-change-card"
              data-candidate-id={c.candidate_id}
              className={`p-2 rounded-[3px] border transition-all cursor-pointer ${
                isSelected
                  ? "bg-amber-50/90 border-amber-400 ring-1 ring-amber-400 shadow-sm"
                  : "border-neutral-200 bg-white hover:border-neutral-300 hover:bg-neutral-50/80"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-1.5 min-w-0">
                  <span className="bg-neutral-900 text-white rounded-[2px] text-[10px] font-mono font-bold px-1 py-px shrink-0">#{idx + 1}</span>
                  {Icon && <Icon className="w-3.5 h-3.5 text-neutral-600 shrink-0" aria-label={directionLabel(c.direction)} />}
                  <span className="text-xs font-semibold text-neutral-800 truncate">{changeLabel(c.change_type)}</span>
                </div>
                <span className="font-mono text-xs font-semibold text-neutral-800 shrink-0" title="Similarity of the after-date imagery">
                  {Math.round(c.similarity * 100)}%
                </span>
              </div>
              <div className="mt-1 font-mono text-[10px] text-neutral-500 truncate">
                {mgrsRef ? `MGRS ${spacedMgrs(mgrsRef)}` : ""}
                {c.area_px ? ` · ${hectares(c.area_px, c.area_ha)}` : ""}
              </div>
              <div className="font-mono text-[10px] text-neutral-400 truncate">
                {c.scene_a_date} → {c.scene_b_date}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * The Search tab: the tiles that match the last query (or, after Find Similar, the tiles that look like the chosen one),
 * each clickable to fly to it, with a Find Similar button and the Attribution switch for the heatmap.
 */
export default function SearchTab({
  searchError = null,
  searchErrorTitle = "Search failed",
  busyWith = "search",
  results = [],
  query = "",
  isSearching = false,
  selectedTileId = null,
  onSelectResult,
  similarTo = null,
  similarChanges = [],
  onFindSimilar,
  onOpenChange,
  selectedChangeId = null,
  onClearSearch = null,
  attribution = null, // {enabled, onToggle}: the heatmap of where in the tile the query matched
  scopeLabel = null, // "active scene" / "all scenes": what the query was run against
}) {
  const showAttribution = attribution && !similarTo; // a heatmap needs a text query to explain

  return (
    <div className="p-3" data-testid="search-tab">
      {isSearching && (
        <div className="bg-neutral-50 border border-neutral-200 rounded-[3px] p-4 text-center">
          <Loader2 className="w-5 h-5 animate-spin text-neutral-600 mx-auto mb-1.5" />
          <div className="text-xs font-medium text-neutral-700">
            {busyWith === "similar" ? "Finding similar tiles..." : "Encoding Query via RemoteCLIP..."}
          </div>
          <div className="text-[11px] text-neutral-400 mt-0.5">
            {busyWith === "similar" ? "Comparing embeddings across the archive" : "Searching tiles and detected changes"}
          </div>
        </div>
      )}

      {!isSearching && (results.length > 0 || similarChanges.length > 0) && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            {similarTo ? (
              <div className="text-[11px] text-neutral-700 font-sans font-medium truncate" title={similarTo.label} data-testid="similar-label">
                {similarTo.label}
              </div>
            ) : (
              <div className="text-[11px] text-neutral-500 font-sans italic truncate" title={query} data-testid="query-label">
                Query: "{query}"{scopeLabel ? ` · ${scopeLabel}` : ""}
              </div>
            )}
            {onClearSearch && (
              <button
                type="button"
                onClick={onClearSearch}
                title="Clear search and show all baseline data"
                className="text-[10px] text-neutral-500 hover:text-neutral-800 underline shrink-0 ml-2"
              >
                Clear Search
              </button>
            )}
          </div>

          <SimilarChanges changes={similarChanges} onOpenChange={onOpenChange} selectedChangeId={selectedChangeId} />
          {similarTo && <div className="text-[10px] uppercase font-semibold text-neutral-400">Similar tiles</div>}

          <div className="space-y-1.5">
            {results.map((res, idx) => {
              const isSelected = selectedTileId === res.tile_id;
              const scorePercent = (res.score * 100).toFixed(1);
              const mgrsRef = res.mgrs || boundsMgrs(res.bounds);
              const select = () => onSelectResult && onSelectResult(res);

              // A div, not a button: the card holds its own Find Similar button, and buttons cannot nest
              return (
                <div
                  key={res.tile_id || idx}
                  role="button"
                  tabIndex={0}
                  onClick={select}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      select();
                    }
                  }}
                  data-testid="semantic-card"
                  data-tile-id={res.tile_id}
                  className={`w-full text-left p-2 rounded-[3px] border transition-all cursor-pointer ${
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

                  <div className="w-full bg-neutral-100 h-1.5 rounded-full mt-1.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${res.score >= 0.8 ? "bg-emerald-500" : res.score >= 0.6 ? "bg-teal-500" : "bg-amber-500"}`}
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
                    {mgrsRef && (
                      <div className="truncate" title={`MGRS ${mgrsRef}`} data-testid="semantic-mgrs">
                        <span className="font-sans text-neutral-400">MGRS: </span>
                        <span className="text-neutral-700">{spacedMgrs(mgrsRef)}</span>
                      </div>
                    )}
                  </div>

                  <div className="mt-1.5 flex items-center justify-between">
                    {showAttribution ? (
                      <label
                        className="flex items-center space-x-1 text-[10px] text-neutral-600 cursor-pointer"
                        title="Overlay a heatmap of where in this tile the query matched"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <input
                          type="checkbox"
                          checked={attribution.enabled}
                          onChange={(e) => attribution.onToggle(e.target.checked)}
                          className="w-3 h-3"
                          data-testid="attribution-toggle"
                        />
                        <span>Attribution</span>
                      </label>
                    ) : (
                      <span />
                    )}
                    {onFindSimilar && (
                      <button
                        type="button"
                        title="Find similar: the 10 tiles that look most like this one"
                        aria-label="Find similar"
                        data-testid="find-similar-tile"
                        onClick={(e) => {
                          e.stopPropagation();
                          onFindSimilar({ tileId: res.tile_id });
                        }}
                        className="flex items-center space-x-1 px-1.5 py-0.5 text-[10px] text-neutral-600 border border-neutral-200 rounded-[3px] bg-white hover:bg-neutral-100"
                      >
                        <Images className="w-3 h-3" />
                        <span>Find Similar</span>
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {!isSearching && searchError && (
        <div className="bg-red-50 border border-red-200 rounded-[3px] p-2.5 flex items-start space-x-2 text-xs text-red-800">
          <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
          <div className="min-w-0 flex-1">
            <div className="font-medium" data-testid="search-error-title">
              {searchErrorTitle}
            </div>
            <div className="text-[11px] text-red-600 mt-0.5 break-words">{searchError}</div>
          </div>
        </div>
      )}

      {!isSearching && !searchError && results.length === 0 && similarChanges.length === 0 && (
        <div className="border border-dashed border-neutral-200 rounded-[3px] p-3 text-center text-neutral-400">
          <Search className="w-4 h-4 mx-auto mb-1 text-neutral-300" />
          <p className="text-xs">No active search results.</p>
          <p className="text-[11px] text-neutral-400 mt-0.5">Enter a query in the bottom search bar.</p>
        </div>
      )}
    </div>
  );
}
