import React, { useState, useEffect } from "react";
import { Search, Loader2, X, Layers } from "lucide-react";

const SUGGESTIONS = [
  "structures near river",
  "airports",
  "big roads and highways",
  "urban buildings and roads",
];

// disabledReason: when set, the whole bar is greyed out and the reason replaces the placeholder
export default function SearchBar({
  onSearch,
  onClear = null,
  isSearching = false,
  disabledReason = null,
  searchAllScenes = false,
  onScopeChange = null,
  sceneLabel = null,
  activeQuery = "",
  hasActiveSearch = false,
}) {
  const [query, setQuery] = useState(activeQuery);
  const inactive = Boolean(disabledReason) || isSearching;

  useEffect(() => {
    setQuery(activeQuery || "");
  }, [activeQuery]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed && !inactive) {
      onSearch(trimmed);
    }
  };

  const handleSuggestion = (text) => {
    setQuery(text);
    onSearch(text);
  };

  const handleClear = () => {
    setQuery("");
    if (onClear) {
      onClear();
    }
  };

  return (
    <div className="h-9 bg-qgis-bg border-t border-qgis-border px-2 flex items-center space-x-2 select-none shrink-0 font-sans text-[13px] text-qgis-text">
      <form onSubmit={handleSubmit} className="flex items-center space-x-2 w-full max-w-md shrink-0">
        <div className="relative flex-1 flex items-center">
          <span className={`absolute left-2 flex items-center pointer-events-none ${disabledReason ? "text-neutral-300" : "text-neutral-500"}`}>
            {isSearching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
          </span>

          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={inactive}
            placeholder={disabledReason || "Search imagery by meaning, e.g. airports or big roads"}
            title={disabledReason || undefined}
            className="w-full h-7 pl-7 pr-7 bg-white border border-qgis-border rounded-[3px] text-[13px] text-qgis-text placeholder-neutral-400 focus:outline-none focus:border-neutral-500 disabled:bg-neutral-100 disabled:text-neutral-400 disabled:cursor-not-allowed"
          />

          {(query || hasActiveSearch) && !isSearching && (
            <button
              type="button"
              onClick={handleClear}
              title="Clear search"
              className="absolute right-1.5 p-0.5 text-neutral-400 hover:text-neutral-600"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>

        <button
          type="submit"
          disabled={!query.trim() || inactive}
          className="h-7 px-2.5 bg-white hover:bg-qgis-hover text-qgis-text border border-qgis-border rounded-[3px] text-xs font-medium transition-colors disabled:opacity-50 disabled:hover:bg-white shrink-0"
        >
          Search
        </button>

        {(hasActiveSearch || query) && !inactive && onClear && (
          <button
            type="button"
            onClick={handleClear}
            title="Clear search and show all baseline data"
            className="h-7 px-2 bg-neutral-100 hover:bg-neutral-200 text-neutral-700 border border-qgis-border rounded-[3px] text-xs font-medium transition-colors shrink-0"
          >
            Clear
          </button>
        )}
      </form>

      {onScopeChange && (
        <div
          role="group"
          aria-label="Search scope"
          className="flex items-center h-7 border border-qgis-border rounded-[3px] overflow-hidden shrink-0 text-[11px]"
          title={searchAllScenes ? "Searching every imported scene" : `Searching only the scene on the map${sceneLabel ? ` (${sceneLabel})` : ""}`}
        >
          <Layers className="w-3 h-3 mx-1.5 text-neutral-500" />
          {[
            [false, "Active scene only"],
            [true, "All scenes"],
          ].map(([all, label]) => (
            <button
              key={label}
              type="button"
              onClick={() => onScopeChange(all)}
              aria-pressed={searchAllScenes === all}
              data-testid={all ? "scope-all" : "scope-active"}
              className={`h-full px-2 whitespace-nowrap transition-colors ${
                searchAllScenes === all ? "bg-neutral-800 text-white" : "bg-white text-neutral-700 hover:bg-qgis-hover"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      <div className="h-4 w-px bg-qgis-border shrink-0" />

      {/* Quick prompts, kept inline so the bar stays a single row */}
      <div className="flex items-center space-x-1.5 min-w-0 overflow-x-auto scrollbar-none">
        {SUGGESTIONS.map((sug) => (
          <button
            key={sug}
            type="button"
            onClick={() => handleSuggestion(sug)}
            disabled={inactive}
            className="px-2 h-5 bg-neutral-200 hover:bg-neutral-300 text-neutral-700 rounded-[3px] text-[11px] whitespace-nowrap shrink-0 transition-colors disabled:opacity-50 disabled:hover:bg-neutral-200"
          >
            {sug}
          </button>
        ))}
      </div>
    </div>
  );
}
