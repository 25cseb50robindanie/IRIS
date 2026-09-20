import React, { useState } from "react";
import { Search, Loader2, X } from "lucide-react";

const SUGGESTIONS = [
  "structures near river",
  "dense forest and vegetation",
  "airport runway and aircraft",
  "urban buildings and roads",
];

// disabledReason: when set, the whole bar is greyed out and the reason replaces the placeholder
export default function SearchBar({ onSearch, isSearching, disabledReason = null }) {
  const [query, setQuery] = useState("");
  const inactive = Boolean(disabledReason) || isSearching;

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
            placeholder={disabledReason || "Search imagery by meaning, e.g. dense forest"}
            title={disabledReason || undefined}
            className="w-full h-7 pl-7 pr-7 bg-white border border-qgis-border rounded-[3px] text-[13px] text-qgis-text placeholder-neutral-400 focus:outline-none focus:border-neutral-500 disabled:bg-neutral-100 disabled:text-neutral-400 disabled:cursor-not-allowed"
          />

          {query && !isSearching && (
            <button
              type="button"
              onClick={() => setQuery("")}
              title="Clear"
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
      </form>

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
