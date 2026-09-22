import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Layers, Info } from "lucide-react";
import ChangeResults, { applyMatchFilters } from "./ChangeResults";
import OverviewTab from "./OverviewTab";
import SearchTab from "./SearchTab";

export const TABS = [
  ["overview", "Overview"],
  ["search", "Search"],
  ["changes", "Changes"],
];

const APP_VERSION = "0.3.0";
const fmt = (n) => (n ?? 0).toLocaleString();

/**
 * "IRIS v0.3.0 — ..." popover behind the (i) button: the archive in one sentence. Closes on a click elsewhere or Escape.
 */
function InfoButton({ sceneCount, vectorCount, candidateCount }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (root.current && !root.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="relative" ref={root}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-label="About IRIS"
        aria-expanded={open}
        data-testid="info-button"
        className={`p-0.5 rounded-full text-neutral-500 hover:text-neutral-800 hover:bg-qgis-hover ${open ? "text-neutral-800 bg-qgis-hover" : ""}`}
      >
        <Info className="w-3.5 h-3.5" />
      </button>
      {open && (
        <div
          role="dialog"
          data-testid="info-popover"
          className="absolute right-0 top-full mt-1 z-30 w-64 p-2.5 bg-white border border-neutral-300 rounded-[3px] shadow-lg text-[11px] leading-snug text-neutral-700 normal-case"
        >
          IRIS v{APP_VERSION} — Semantic Retrieval &amp; Change Analysis. {fmt(sceneCount)} scene{sceneCount === 1 ? "" : "s"} indexed,{" "}
          {fmt(vectorCount)} embedding{vectorCount === 1 ? "" : "s"}, {fmt(candidateCount)} change candidate{candidateCount === 1 ? "" : "s"}.
        </div>
      )}
    </div>
  );
}

export default function Sidebar({
  activeTab = "overview",
  onTabChange,
  scrollTarget = null, // {id, n}: bring this change candidate into view in the Changes tab
  // Overview
  ingestState,
  currentScene,
  scenes = [],
  embedStatus = null,
  catalog = null,
  catalogError = null,
  error,
  pipeline = null,
  onDismissPipeline,
  onSelectScene,
  watch = null, // the watchlist and its alerts (useWatchlist)
  focusedWatchId = null,
  onFocusWatch,
  onOpenAlert,
  onDownloadReport,
  reportBusy = false,
  // Search
  searchError = null,
  searchErrorTitle = "Search failed",
  busyWith = "search",
  searchResults = [],
  searchQuery = "",
  isSearching = false,
  selectedTileId = null,
  onSelectResult,
  similarTo = null,
  similarChanges = [],
  onFindSimilar,
  attribution = null,
  // Changes: the unified search's change half, and the browsable list of every detected change
  search = null,
  changes = null,
  changeJobs = [],
  changeFilters,
  onChangeFilters,
  changeView = "all",
  onChangeView,
  selectedChangeId = null,
  onOpenChange,
  onClearSearch = null,
  ablation = null,
  onNotify = null,
  searchScope = null, // {all, label}: what the last search covered
}) {
  // Each tab remembers where it was scrolled to. The position is recorded as the analyst scrolls and put back when the
  // tab comes back, before it is painted.
  const bodyRef = useRef(null);
  const scrollPositions = useRef({ overview: 0, search: 0, changes: 0 });
  const recordScroll = (e) => {
    scrollPositions.current[activeTab] = e.currentTarget.scrollTop;
  };
  useLayoutEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = scrollPositions.current[activeTab] || 0;
  }, [activeTab]);

  // A change candidate was clicked somewhere else: once the Changes tab is showing, scroll to its card
  useEffect(() => {
    if (!scrollTarget || activeTab !== "changes") return undefined;
    const frame = requestAnimationFrame(() => {
      const card = bodyRef.current?.querySelector(`[data-testid="change-card"][data-candidate-id="${scrollTarget.id}"]`);
      if (card) card.scrollIntoView({ block: "center", behavior: "smooth" });
    });
    return () => cancelAnimationFrame(frame);
  }, [scrollTarget]); // eslint-disable-line react-hooks/exhaustive-deps

  // "Changes (23)": how many the tab shows, which is the matches of the last query when it has some
  const inSearch = changeView === "search" && search?.hasSearched;
  let changeCount = 0;
  if (ablation?.on) changeCount = ablation.stats?.ablation_count ?? 0;
  else if (inSearch) changeCount = search.status === "found" ? applyMatchFilters(search.results, changeFilters).length : 0;
  else changeCount = changes?.matching ?? changes?.total ?? 0;

  const badges = { overview: 0, search: searchResults.length, changes: changeCount };
  const unseenAlerts = watch?.unacknowledged ?? 0; // a detection on a watched location the analyst has not looked at

  return (
    <aside className="w-80 bg-white border-l border-qgis-border flex flex-col shrink-0 select-none text-qgis-text h-full overflow-hidden">
      <div className="h-9 px-3 border-b border-qgis-border bg-qgis-header flex items-center justify-between shrink-0">
        <span className="font-semibold text-xs text-neutral-800 flex items-center space-x-1.5">
          <Layers className="w-3.5 h-3.5 text-neutral-600" />
          <span>Workspace</span>
        </span>
        <InfoButton sceneCount={catalog?.scene_count ?? scenes.length} vectorCount={catalog?.faiss_vectors ?? 0} candidateCount={changes?.total ?? 0} />
      </div>

      <div role="tablist" className="flex border-b border-qgis-border bg-qgis-bg shrink-0">
        {TABS.map(([id, label]) => {
          const active = activeTab === id;
          const count = badges[id];
          const raw = id === "changes" && ablation?.on;
          return (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={active}
              data-testid={`tab-${id}`}
              onClick={() => onTabChange && onTabChange(id)}
              className={`flex-1 h-8 text-[11px] font-semibold border-b-2 transition-colors whitespace-nowrap ${
                active
                  ? `bg-white text-neutral-900 ${raw ? "border-red-600" : "border-neutral-800"}`
                  : "border-transparent text-neutral-500 hover:text-neutral-800 hover:bg-qgis-hover"
              }`}
            >
              {label}
              {id === "overview" && unseenAlerts > 0 && (
                <span
                  className="ml-1 inline-flex items-center justify-center min-w-[16px] h-4 px-1 rounded-full bg-red-600 text-white text-[10px] font-mono"
                  title={`${unseenAlerts} new detection${unseenAlerts === 1 ? "" : "s"} on watched locations`}
                  data-testid="tab-overview-alerts"
                >
                  {unseenAlerts}
                </span>
              )}
              {count > 0 && (
                <span className={`ml-1 font-mono ${raw ? "text-red-700" : active ? "text-neutral-600" : "text-neutral-400"}`} data-testid={`tab-${id}-count`}>
                  ({fmt(count)})
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div ref={bodyRef} onScroll={recordScroll} className="flex-1 overflow-y-auto" data-testid="tab-body">
        {activeTab === "overview" && (
          <OverviewTab
            pipeline={pipeline}
            onDismissPipeline={onDismissPipeline}
            ingestState={ingestState}
            error={error}
            catalog={catalog}
            catalogError={catalogError}
            scenes={scenes}
            currentScene={currentScene}
            embedStatus={embedStatus}
            changeCount={changes?.total ?? 0}
            onSelectScene={onSelectScene}
            watch={watch}
            focusedWatchId={focusedWatchId}
            onFocusWatch={onFocusWatch}
            onOpenAlert={onOpenAlert}
            onDownloadReport={onDownloadReport}
            reportBusy={reportBusy}
          />
        )}

        {activeTab === "search" && (
          <SearchTab
            searchError={searchError}
            searchErrorTitle={searchErrorTitle}
            busyWith={busyWith}
            results={searchResults}
            query={searchQuery}
            isSearching={isSearching}
            selectedTileId={selectedTileId}
            onSelectResult={onSelectResult}
            similarTo={similarTo}
            similarChanges={similarChanges}
            onFindSimilar={onFindSimilar}
            onOpenChange={onOpenChange}
            selectedChangeId={selectedChangeId}
            onClearSearch={onClearSearch}
            attribution={attribution}
            scopeLabel={searchScope?.label}
          />
        )}

        {activeTab === "changes" && (
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
            onFindSimilar={onFindSimilar}
            ablation={ablation}
            onNotify={onNotify}
            onClearSearch={onClearSearch}
          />
        )}
      </div>
    </aside>
  );
}
