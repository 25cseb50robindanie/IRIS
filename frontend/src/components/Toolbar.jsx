import React from "react";
import { FolderOpen, ZoomIn, ZoomOut, Maximize2, Layers, Loader2 } from "lucide-react";

export default function Toolbar({
  onBrowse,
  isIngesting,
  onZoomIn,
  onZoomOut,
  onFitBounds,
  hasActiveLayer,
}) {
  return (
    <div className="h-9 bg-qgis-bg border-b border-qgis-border px-2 flex items-center justify-between select-none shrink-0">
      <div className="flex items-center space-x-1">
        {/* Browse / Import Button */}
        <button
          onClick={onBrowse}
          disabled={isIngesting}
          title="Browse Satellite Imagery (GeoTIFF / JP2)"
          className="flex items-center space-x-1.5 px-2.5 py-1 bg-white hover:bg-qgis-hover disabled:opacity-50 text-qgis-text border border-qgis-border rounded-[3px] text-xs font-medium transition-colors shadow-xs"
        >
          {isIngesting ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin text-neutral-600" />
          ) : (
            <FolderOpen className="w-3.5 h-3.5 text-neutral-700" />
          )}
          <span>{isIngesting ? "Importing..." : "Browse"}</span>
        </button>

        <div className="h-4 w-px bg-qgis-border mx-1" />

        {/* Map Navigation Controls */}
        <button
          onClick={onZoomIn}
          title="Zoom In"
          className="p-1 hover:bg-qgis-hover active:bg-neutral-300 text-neutral-700 rounded-[3px] transition-colors"
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={onZoomOut}
          title="Zoom Out"
          className="p-1 hover:bg-qgis-hover active:bg-neutral-300 text-neutral-700 rounded-[3px] transition-colors"
        >
          <ZoomOut className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={onFitBounds}
          disabled={!hasActiveLayer}
          title="Fit Full Extent"
          className="p-1 hover:bg-qgis-hover active:bg-neutral-300 disabled:opacity-40 text-neutral-700 rounded-[3px] transition-colors"
        >
          <Maximize2 className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="flex items-center space-x-2 text-xs text-neutral-500 font-mono">
        <span className="flex items-center space-x-1">
          <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
          <span className="text-neutral-700 text-[11px]">IRIS Local Engine</span>
        </span>
      </div>
    </div>
  );
}
