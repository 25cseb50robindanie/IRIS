import React, { useState } from "react";
import { ChevronRight, Loader2, AlertCircle, Layers, FileDown } from "lucide-react";
import PipelineProgress from "./PipelineProgress";
import WatchlistSection from "./WatchlistSection";

const fmt = (n) => (n ?? 0).toLocaleString();

// Embedding progress of the scene on the map, for when no import pipeline is running (e.g. after a reload)
function describeEmbedding(es) {
  switch (es?.state) {
    case "queued":
      return { label: "Embedding queued", tone: "busy" };
    case "tiling":
      return { label: `Tiling... (${fmt(es.crops_generated)} crops)`, tone: "busy" };
    case "embedding":
      return { label: `Embedding... (${fmt(es.crops_embedded)}/${fmt(es.crops_total)} crops)`, tone: "busy" };
    case "indexing":
      return { label: "Indexing...", tone: "busy" };
    case "failed":
      return { label: "Embedding failed", tone: "error", detail: es.error };
    case "not_embedded":
      return { label: "Not embedded", tone: "error", detail: es.error };
    default:
      return null; // "ready", or not known yet: nothing to warn about
  }
}

function Stat({ label, value, testId }) {
  return (
    <div className="flex-1 min-w-0 border border-neutral-200 rounded-[3px] bg-white px-2 py-1.5 text-center">
      <div className="font-mono text-sm font-semibold text-neutral-800" data-testid={testId}>
        {value}
      </div>
      <div className="text-[10px] uppercase tracking-wider text-neutral-400">{label}</div>
    </div>
  );
}

/**
 * The Overview tab: what has been imported and what an import is doing, in one place. The three figures are the
 * archive at a glance; while an import runs its steps are shown; the scene list underneath is collapsible.
 */
export default function OverviewTab({
  pipeline,
  onDismissPipeline,
  ingestState,
  error,
  catalog,
  catalogError,
  scenes = [],
  currentScene,
  embedStatus,
  changeCount = 0,
  onSelectScene,
  watch = null,
  focusedWatchId = null,
  onFocusWatch,
  onOpenAlert,
  onDownloadReport,
  reportBusy = false,
}) {
  const importing = ingestState.status === "ingesting" || pipeline?.state === "running";
  const [scenesOpen, setScenesOpen] = useState(true);

  let status = null; // one line of what is going on, when something is
  if (importing) {
    status = { text: `Importing ${pipeline?.name || ingestState.fileName || "scene"}...`, tone: "busy" };
  } else if (!catalog) {
    status = { text: catalogError || "Loading catalog...", tone: catalogError ? "error" : "busy" };
  } else if (pipeline?.state === "failed") {
    status = { text: pipeline.message || "Import failed", tone: "error" };
  } else if (ingestState.status === "error") {
    status = { text: `Import failed — ${error || "could not process the image"}`, tone: "error" };
  } else if (catalog.scene_count === 0) {
    status = { text: "No scenes imported yet. Use Browse to add one.", tone: "neutral" };
  }

  // A problem with the scene on the map stays visible whatever else is on screen
  const embed = !importing && currentScene ? describeEmbedding(embedStatus) : null;
  const rows = scenes.length ? scenes : currentScene ? [currentScene] : [];
  const color = { busy: "text-amber-800", error: "text-red-700", neutral: "text-neutral-600" };

  return (
    <div className="p-3 space-y-3" data-testid="overview-tab">
      <div className="flex space-x-2">
        <Stat label="Scenes" value={fmt(catalog?.scene_count)} testId="overview-scenes" />
        <Stat label="Vectors" value={fmt(catalog?.faiss_vectors)} testId="overview-vectors" />
        <Stat label="Changes" value={fmt(changeCount)} testId="overview-changes" />
      </div>

      {watch && <WatchlistSection watch={watch} focusedId={focusedWatchId} onFocusLocation={onFocusWatch} onOpenAlert={onOpenAlert} />}

      {status && (
        <div className={`flex items-start space-x-1.5 text-[11px] ${color[status.tone]}`} data-testid="overview-status">
          {status.tone === "busy" && <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0 mt-px" />}
          {status.tone === "error" && <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-px" />}
          <span className="min-w-0 break-words font-mono">{status.text}</span>
        </div>
      )}

      {embed && (
        <div className={`text-[11px] ${embed.tone === "error" ? "text-red-700" : "text-amber-800"}`}>
          {embed.label}
          {embed.detail ? `: ${embed.detail}` : ""}
        </div>
      )}

      {pipeline && <PipelineProgress pipeline={pipeline} onDismiss={onDismissPipeline} />}

      {rows.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setScenesOpen((o) => !o)}
            aria-expanded={scenesOpen}
            className="w-full text-left flex items-center justify-between text-[11px] font-semibold text-neutral-500 hover:text-neutral-700 uppercase tracking-wider mb-1.5"
          >
            <span className="flex items-center space-x-1">
              <ChevronRight className={`w-3 h-3 text-neutral-400 transition-transform ${scenesOpen ? "rotate-90" : ""}`} />
              <Layers className="w-3 h-3 text-neutral-400" />
              <span>Scenes</span>
            </span>
            <span className="bg-neutral-100 text-neutral-700 px-1.5 rounded text-[10px] font-mono">{rows.length}</span>
          </button>

          {scenesOpen && (
            <ul className="border border-neutral-200 rounded-[3px] bg-white divide-y divide-neutral-100" data-testid="scene-list">
              {rows.map((s) => {
                const showing = currentScene?.scene_id === s.scene_id;
                const canShow = onSelectScene && s.cog_available !== false && !showing;
                return (
                  <li key={s.scene_id}>
                    <button
                      type="button"
                      disabled={!canShow}
                      onClick={() => onSelectScene(s.scene_id)}
                      title={canShow ? "Show this scene on the map" : undefined}
                      className={`w-full text-left px-2.5 py-1.5 ${showing ? "bg-neutral-100" : canShow ? "hover:bg-neutral-50" : ""} disabled:cursor-default`}
                    >
                      <div className="flex items-center justify-between space-x-2">
                        <span className="text-xs font-semibold text-neutral-800">
                          {s.acquisition_date} · {s.sensor}
                        </span>
                        {showing && <span className="text-[10px] text-neutral-500">showing</span>}
                      </div>
                      <div className="font-mono text-[10px] text-neutral-500 truncate" title={s.scene_id}>
                        {s.scene_id}
                      </div>
                      <div className="font-mono text-[10px] text-neutral-500">
                        {s.crs || "unknown CRS"}
                        {s.tiles_count != null ? ` · ${fmt(s.tiles_count)} tile${s.tiles_count === 1 ? "" : "s"}` : ""}
                        {s.cloud_pct != null ? ` · cloud ${s.cloud_pct.toFixed(0)}%` : ""}
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}

      {onDownloadReport && (
        <button
          type="button"
          onClick={onDownloadReport}
          disabled={reportBusy}
          data-testid="download-report"
          title="Build times, storage, index size, query latency and hardware, as JSON"
          className="w-full flex items-center justify-center space-x-1.5 px-2 py-1.5 border border-neutral-300 rounded-[3px] bg-white text-[11px] font-medium text-neutral-700 hover:bg-neutral-100 disabled:opacity-60"
        >
          {reportBusy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FileDown className="w-3.5 h-3.5" />}
          <span>{reportBusy ? "Preparing report..." : "Download Report"}</span>
        </button>
      )}
    </div>
  );
}
