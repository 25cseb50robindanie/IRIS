import React, { useState, useEffect } from "react";
import { Database, ChevronRight, Loader2, AlertCircle } from "lucide-react";
import PipelineProgress from "./PipelineProgress";

const fmt = (n) => (n ?? 0).toLocaleString();
const plural = (n, word) => `${fmt(n)} ${word}${n === 1 ? "" : "s"}`;

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

/**
 * The single source of truth for what has been imported and what an import is doing. While an import runs it shows
 * the pipeline steps; afterwards it collapses to one line ("2 scenes · 4,802 vectors indexed") that expands into the
 * scene list.
 */
export default function WorkspaceStatus({
  pipeline,
  onDismissPipeline,
  ingestState,
  error,
  catalog,
  catalogError,
  scenes = [],
  currentScene,
  embedStatus,
}) {
  const importing = ingestState.status === "ingesting" || pipeline?.state === "running";
  const [override, setOverride] = useState(null);
  // A new import always opens the section again, whatever the analyst did with the previous one
  useEffect(() => {
    if (importing) setOverride(null);
  }, [importing]);
  const open = override ?? importing;

  let summary;
  let tone = "neutral";
  if (importing) {
    summary = `Importing ${pipeline?.name || ingestState.fileName || "scene"}...`;
    tone = "busy";
  } else if (!catalog) {
    summary = catalogError || "Loading catalog...";
    tone = catalogError ? "error" : "busy";
  } else if (pipeline?.state === "failed") {
    summary = pipeline.message || "Import failed";
    tone = "error";
  } else if (ingestState.status === "error") {
    summary = `Import failed — ${error || "could not process the image"}`;
    tone = "error";
  } else if (catalog.scene_count === 0) {
    summary = "No scenes imported yet. Use Browse to add one.";
  } else {
    summary = `${plural(catalog.scene_count, "scene")} · ${fmt(catalog.faiss_vectors)} vectors indexed`;
  }

  // A problem with the scene on the map stays visible even when the section is collapsed
  const embed = !importing && currentScene ? describeEmbedding(embedStatus) : null;

  const rows = scenes.length ? scenes : currentScene ? [currentScene] : [];
  const color = { busy: "text-amber-800", error: "text-red-700", neutral: "text-neutral-700" }[tone];

  return (
    <div className="p-3 bg-neutral-50/50">
      <button
        type="button"
        onClick={() => setOverride(!open)}
        aria-expanded={open}
        className="w-full text-left flex items-center justify-between text-[11px] font-semibold text-neutral-500 hover:text-neutral-700 uppercase tracking-wider"
      >
        <span className="flex items-center space-x-1">
          <ChevronRight className={`w-3 h-3 text-neutral-400 transition-transform ${open ? "rotate-90" : ""}`} />
          <Database className="w-3 h-3 text-neutral-400" />
          <span>Workspace</span>
        </span>
      </button>

      <div className={`mt-1.5 flex items-start space-x-1.5 text-xs font-medium ${color}`}>
        {tone === "busy" && <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0 mt-px" />}
        {tone === "error" && <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-px" />}
        <span className="min-w-0 break-words font-mono text-[11px] font-normal">{summary}</span>
      </div>

      {embed && (
        <div className={`mt-1 text-[11px] ${embed.tone === "error" ? "text-red-700" : "text-amber-800"}`}>
          {embed.label}
          {embed.detail ? `: ${embed.detail}` : ""}
        </div>
      )}

      {open && (
        <div className="mt-2">
          {pipeline && <PipelineProgress pipeline={pipeline} onDismiss={onDismissPipeline} />}

          {rows.length > 0 && (
            <ul className="border border-neutral-200 rounded-[3px] bg-white divide-y divide-neutral-100">
              {rows.map((s) => {
                const showing = currentScene?.scene_id === s.scene_id;
                return (
                  <li key={s.scene_id} className={`px-2.5 py-1.5 ${showing ? "bg-neutral-100" : ""}`}>
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
                      {s.tiles_count != null ? ` · ${plural(s.tiles_count, "tile")}` : ""}
                      {s.cloud_pct != null ? ` · cloud ${s.cloud_pct.toFixed(0)}%` : ""}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
