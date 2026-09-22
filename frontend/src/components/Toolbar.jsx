import React, { useEffect, useRef, useState } from "react";
import { FolderOpen, File, ChevronDown, ZoomIn, ZoomOut, Maximize2, Layers, Loader2 } from "lucide-react";
import SceneMenu from "./SceneMenu";
import irisLogo from "../assets/logo.png";

export default function Toolbar({
  onImportFile,
  onImportFolder,
  isIngesting,
  onZoomIn,
  onZoomOut,
  onFitBounds,
  hasActiveLayer,
  scenes = [],
  currentSceneId = null,
  onSelectScene,
  onDeleteScene,
  onOpenScenes,
}) {
  const [browseOpen, setBrowseOpen] = useState(false);
  const browseRoot = useRef(null);

  // Close the Import dropdown on an outside click or Escape
  useEffect(() => {
    if (!browseOpen) return undefined;
    const onDown = (e) => {
      if (browseRoot.current && !browseRoot.current.contains(e.target)) setBrowseOpen(false);
    };
    const onKey = (e) => e.key === "Escape" && setBrowseOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [browseOpen]);

  return (
    <div className="h-9 bg-qgis-bg border-b border-qgis-border px-2 flex items-center justify-between select-none shrink-0">
      <div className="flex items-center space-x-1.5">
        <img src={irisLogo} alt="IRIS Logo" className="w-5 h-5 rounded-full mr-1 object-contain shrink-0 select-none pointer-events-none" />
        {/* Import dropdown: a single GeoTIFF/JP2 file, or a .SAFE / Landsat folder */}
        <div className="relative" ref={browseRoot}>
          <button
            type="button"
            onClick={() => setBrowseOpen((o) => !o)}
            disabled={isIngesting}
            title="Import Satellite Imagery (GeoTIFF / JP2 / .SAFE folder)"
            aria-haspopup="menu"
            aria-expanded={browseOpen}
            className="flex items-center space-x-1.5 px-2.5 py-1 bg-white hover:bg-qgis-hover disabled:opacity-50 text-qgis-text border border-qgis-border rounded-[3px] text-xs font-medium transition-colors shadow-xs"
          >
            {isIngesting ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-neutral-600" />
            ) : (
              <FolderOpen className="w-3.5 h-3.5 text-neutral-700" />
            )}
            <span>{isIngesting ? "Importing..." : "Browse"}</span>
            {!isIngesting && <ChevronDown className="w-3 h-3 text-neutral-500" />}
          </button>

          {browseOpen && (
            <div
              role="menu"
              className="absolute left-0 top-full mt-1 w-48 z-40 bg-white border border-qgis-border rounded-[3px] shadow-lg py-1"
            >
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setBrowseOpen(false);
                  onImportFile();
                }}
                className="w-full text-left px-2.5 py-1.5 flex items-center space-x-2 text-xs hover:bg-neutral-100"
              >
                <File className="w-3.5 h-3.5 text-neutral-600 shrink-0" />
                <span>Import File</span>
              </button>
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setBrowseOpen(false);
                  onImportFolder();
                }}
                className="w-full text-left px-2.5 py-1.5 flex items-center space-x-2 text-xs hover:bg-neutral-100"
              >
                <FolderOpen className="w-3.5 h-3.5 text-neutral-600 shrink-0" />
                <span>Import Folder</span>
              </button>
            </div>
          )}
        </div>

        {/* Imported scenes: switch the map to one, or delete it */}
        <SceneMenu
          scenes={scenes}
          currentSceneId={currentSceneId}
          onSelect={onSelectScene}
          onDelete={onDeleteScene}
          onOpen={onOpenScenes}
          disabled={isIngesting}
        />

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
    </div>
  );
}
