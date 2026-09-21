import React from "react";
import { spacedMgrs, toMgrs } from "../mgrs";

export default function StatusBar({ mousePos, activeCrs }) {
  const { lng, lat, zoom, bearing } = mousePos || {};

  // Format coordinates cleanly
  const formatCoord = () => {
    if (lng === undefined || lat === undefined) return "Coordinate: —";
    const lngDir = lng >= 0 ? "E" : "W";
    const latDir = lat >= 0 ? "N" : "S";
    return `Coordinate: ${Math.abs(lng).toFixed(5)}° ${lngDir}, ${Math.abs(lat).toFixed(5)}° ${latDir}`;
  };

  // MGRS of the point under the cursor (10-digit; blank where MGRS is undefined, e.g. the poles)
  const mgrsRef = lng === undefined || lat === undefined ? null : toMgrs(lat, lng);

  // Derive approximate map scale from zoom
  const calculateScale = (z) => {
    if (z === undefined) return "1:1,000,000";
    const scaleValue = Math.round(591657550 / Math.pow(2, z));
    return `1:${scaleValue.toLocaleString()}`;
  };

  return (
    <footer className="h-6 bg-qgis-status border-t border-[#D0D0D0] px-3 flex items-center justify-between text-[11px] font-mono select-none shrink-0 text-neutral-700">
      {/* Left side: Coordinates & Scale */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-1.5 min-w-[210px]">
          <span className="text-neutral-800">{formatCoord()}</span>
        </div>

        <div className="h-3 w-px bg-neutral-300" />

        <div className="flex items-center space-x-1 min-w-[150px]" data-testid="status-mgrs">
          <span className="text-neutral-500 font-sans">MGRS</span>
          <span className="text-neutral-800">{mgrsRef ? spacedMgrs(mgrsRef) : "—"}</span>
        </div>

        <div className="h-3 w-px bg-neutral-300" />

        <div className="flex items-center space-x-1">
          <span className="text-neutral-500 font-sans">Scale</span>
          <span className="text-neutral-800 font-medium">{calculateScale(zoom)}</span>
        </div>

        <div className="h-3 w-px bg-neutral-300" />

        <div className="flex items-center space-x-1">
          <span className="text-neutral-500 font-sans">Magnifier</span>
          <span className="text-neutral-800">100%</span>
        </div>

        <div className="h-3 w-px bg-neutral-300" />

        <div className="flex items-center space-x-1">
          <span className="text-neutral-500 font-sans">Rotation</span>
          <span className="text-neutral-800">{(bearing || 0).toFixed(1)}°</span>
        </div>
      </div>

      {/* Right side: Render Checkbox & Active CRS */}
      <div className="flex items-center space-x-4">
        <label className="flex items-center space-x-1.5 cursor-default">
          <input
            type="checkbox"
            checked
            readOnly
            className="w-3 h-3 rounded-[2px] border-neutral-400 text-neutral-800 focus:ring-0"
          />
          <span className="text-neutral-700 font-sans">Render</span>
        </label>

        <div className="h-3 w-px bg-neutral-300" />

        <div className="flex items-center space-x-1">
          <span className="text-neutral-500 font-sans">CRS</span>
          <span className="px-1.5 py-0.2 bg-neutral-200 border border-neutral-300 rounded-[2px] font-semibold text-neutral-800 text-[10px]">
            {activeCrs || "EPSG:3857"}
          </span>
        </div>
      </div>
    </footer>
  );
}
