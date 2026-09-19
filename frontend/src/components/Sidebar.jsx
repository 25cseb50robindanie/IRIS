import React from "react";
import { Layers, Database, CheckCircle2, AlertCircle, Loader2, Info } from "lucide-react";

export default function Sidebar({ ingestState, currentScene, error }) {
  const { status, fileName } = ingestState;

  return (
    <aside className="w-72 bg-white border-l border-qgis-border flex flex-col shrink-0 select-none text-qgis-text">
      {/* Sidebar Header */}
      <div className="h-9 px-3 border-b border-qgis-border bg-qgis-header flex items-center justify-between">
        <span className="font-semibold text-xs text-neutral-800 flex items-center space-x-1.5">
          <Layers className="w-3.5 h-3.5 text-neutral-600" />
          <span>Workspace</span>
        </span>
        <span className="text-[11px] font-mono text-neutral-500">v0.1.0</span>
      </div>

      {/* Dataset & Ingestion Status Section */}
      <div className="p-3 border-b border-qgis-border bg-neutral-50/50">
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
          <div className="bg-emerald-50/60 border border-emerald-200 rounded-[3px] p-2.5">
            <div className="flex items-center space-x-1.5 text-emerald-800 font-medium text-xs">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              <span className="truncate">Ready</span>
            </div>
            
            <div className="mt-2 space-y-1.5 font-mono text-[11px] text-neutral-600">
              <div className="flex flex-col">
                <span className="text-neutral-400 text-[10px] uppercase font-sans">Scene ID</span>
                <span className="text-neutral-900 font-medium truncate" title={currentScene.scene_id}>
                  {currentScene.scene_id}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1 border-t border-emerald-100">
                <div>
                  <span className="text-neutral-400 text-[10px] uppercase font-sans">CRS</span>
                  <div className="text-neutral-800 font-medium">{currentScene.crs || "Unknown"}</div>
                </div>
                <div>
                  <span className="text-neutral-400 text-[10px] uppercase font-sans">Sensor</span>
                  <div className="text-neutral-800 font-medium">{currentScene.sensor || "Sentinel-2"}</div>
                </div>
              </div>

              {currentScene.bounds && (
                <div className="pt-1 border-t border-emerald-100">
                  <span className="text-neutral-400 text-[10px] uppercase font-sans">Bounds (Native)</span>
                  <div className="text-neutral-700 text-[10px] truncate">
                    [{currentScene.bounds.map((b) => (typeof b === "number" ? b.toFixed(1) : b)).join(", ")}]
                  </div>
                </div>
              )}
            </div>
          </div>
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

        {status === "idle" && (
          <div className="border border-dashed border-neutral-300 rounded-[3px] p-3 text-center text-neutral-500">
            <Info className="w-4 h-4 mx-auto mb-1 text-neutral-400" />
            <p className="text-xs">No active scene.</p>
            <p className="text-[11px] text-neutral-400 mt-0.5">Click Browse in toolbar to open GeoTIFF / JP2.</p>
          </div>
        )}
      </div>

      {/* Placeholder for future pipeline visibility / analysis stages */}
      <div className="flex-1 p-3 flex flex-col justify-center items-center text-center text-neutral-400">
        <span className="text-xs">Pipeline & Semantic Search</span>
        <span className="text-[11px] text-neutral-400 mt-1">Available in subsequent milestones</span>
      </div>
    </aside>
  );
}
