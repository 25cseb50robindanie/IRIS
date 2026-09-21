import React, { useEffect, useRef } from "react";
import { Eye, Loader2 } from "lucide-react";
import { formatLatLon, spacedMgrs, toMgrs } from "../mgrs";

/**
 * The menu that opens on a right-click (or long press) on the map. `menu` is {x, y, lng, lat}: where in the map's box it
 * opened, and the ground under the pointer. Closes on Escape or a click anywhere else.
 */
export default function MapContextMenu({ menu, onWatch, onClose, busy = false }) {
  const root = useRef(null);

  useEffect(() => {
    if (!menu) return undefined;
    const away = (e) => root.current && !root.current.contains(e.target) && onClose();
    const key = (e) => e.key === "Escape" && onClose();
    document.addEventListener("mousedown", away);
    document.addEventListener("touchstart", away);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("touchstart", away);
      document.removeEventListener("keydown", key);
    };
  }, [menu, onClose]);

  if (!menu) return null;
  const mgrsRef = toMgrs(menu.lat, menu.lng);
  return (
    <div
      ref={root}
      role="menu"
      data-testid="map-context-menu"
      style={{ left: menu.x, top: menu.y }}
      className="absolute z-30 w-56 bg-white border border-neutral-300 rounded-[3px] shadow-lg py-1 text-xs normal-case"
    >
      <div className="px-2.5 py-1 border-b border-neutral-100 font-mono text-[10px] text-neutral-500">
        <div>{formatLatLon(menu.lat, menu.lng)}</div>
        {mgrsRef && <div>MGRS {spacedMgrs(mgrsRef)}</div>}
      </div>
      <button
        type="button"
        role="menuitem"
        disabled={busy}
        onClick={() => onWatch(menu)}
        data-testid="watch-this-location"
        className="w-full flex items-center space-x-2 px-2.5 py-1.5 text-left hover:bg-neutral-100 disabled:opacity-50"
      >
        {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin text-neutral-600" /> : <Eye className="w-3.5 h-3.5 text-neutral-600" />}
        <span className="font-medium text-neutral-800">Watch this location</span>
      </button>
    </div>
  );
}
