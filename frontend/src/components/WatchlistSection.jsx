import React from "react";
import { Eye, Trash2, BellRing, Loader2, Check } from "lucide-react";
import { boundsMgrs, spacedMgrs } from "../mgrs";
import { changeLabel, hectares } from "./changeKinds";

const pct = (v) => `${Math.round(v * 100)}%`;
const MAX_ALERTS_SHOWN = 4; // per location; the rest are behind "more"

/**
 * The watchlist: places the analyst pinned on the map. The backend checks every finished analysis against them, so a new
 * detection on one of them is waiting here (and as a badge on the Overview tab) whether or not the analyst was looking.
 */
export default function WatchlistSection({ watch, focusedId = null, onFocusLocation, onOpenAlert }) {
  const { locations, alerts, unacknowledged, busy } = watch;

  return (
    <div data-testid="watchlist-section">
      <div className="flex items-center justify-between mb-1.5">
        <span className="flex items-center space-x-1 text-[11px] font-semibold text-neutral-500 uppercase tracking-wider">
          <Eye className="w-3 h-3 text-neutral-400" />
          <span>Watchlist</span>
          {locations.length > 0 && <span className="bg-neutral-100 text-neutral-700 px-1.5 rounded text-[10px] font-mono">{locations.length}</span>}
          {unacknowledged > 0 && (
            <span className="flex items-center space-x-0.5 bg-red-600 text-white px-1.5 rounded text-[10px] font-mono" data-testid="watch-unseen">
              <BellRing className="w-2.5 h-2.5" />
              <span>{unacknowledged} new</span>
            </span>
          )}
        </span>
        {unacknowledged > 0 && (
          <button
            type="button"
            onClick={watch.acknowledgeAll}
            disabled={busy}
            data-testid="watch-ack-all"
            className="flex items-center space-x-1 text-[10px] text-neutral-600 hover:text-neutral-900 underline disabled:opacity-50"
          >
            {busy ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
            <span>Mark all seen</span>
          </button>
        )}
      </div>

      {locations.length === 0 ? (
        <div className="border border-dashed border-neutral-200 rounded-[3px] p-2.5 text-[11px] text-neutral-500 leading-snug">
          Nothing watched yet. Right-click a place on the map and choose <span className="font-semibold">Watch this location</span>: when a
          new analysis finds a change there, it is flagged here.
        </div>
      ) : (
        <ul className="space-y-1.5">
          {locations.map((loc) => {
            const mine = alerts.filter((a) => a.watchlist_id === loc.id);
            const open = mine.filter((a) => !a.acknowledged_at);
            const shown = [...open, ...mine.filter((a) => a.acknowledged_at)].slice(0, MAX_ALERTS_SHOWN);
            const mgrsRef = boundsMgrs(loc.bounds);
            return (
              <li
                key={loc.id}
                data-testid="watch-item"
                data-watch-id={loc.id}
                className={`border rounded-[3px] bg-white ${
                  open.length ? "border-red-400 ring-1 ring-red-300" : focusedId === loc.id ? "border-blue-400 ring-1 ring-blue-300" : "border-neutral-200"
                }`}
              >
                <div className="flex items-start justify-between px-2 py-1.5">
                  <button type="button" onClick={() => onFocusLocation && onFocusLocation(loc)} className="min-w-0 text-left" title="Show on the map">
                    <div className="text-xs font-semibold text-neutral-800 truncate">{loc.name}</div>
                    <div className="font-mono text-[10px] text-neutral-500 truncate">
                      {mgrsRef ? `MGRS ${spacedMgrs(mgrsRef)} · ` : ""}alert at ≥ {pct(loc.confidence_threshold)}
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick={() => watch.remove(loc.id)}
                    disabled={busy}
                    aria-label={`Stop watching ${loc.name}`}
                    title="Stop watching"
                    data-testid="watch-remove"
                    className="p-0.5 text-neutral-400 hover:text-red-600 disabled:opacity-40 shrink-0"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {shown.length > 0 && (
                  <ul className="border-t border-neutral-100 divide-y divide-neutral-100">
                    {shown.map((a) => (
                      <li key={a.alert_id}>
                        <button
                          type="button"
                          onClick={() => onOpenAlert && onOpenAlert(a)}
                          data-testid="watch-alert"
                          data-alert-id={a.alert_id}
                          data-unseen={a.acknowledged_at ? "false" : "true"}
                          className={`w-full text-left px-2 py-1 hover:bg-neutral-50 ${a.acknowledged_at ? "" : "bg-red-50"}`}
                        >
                          <div className="flex items-center justify-between text-[11px]">
                            <span className={`font-semibold ${a.acknowledged_at ? "text-neutral-700" : "text-red-800"}`}>
                              {!a.acknowledged_at && <span className="mr-1 text-[9px] bg-red-600 text-white px-1 rounded">NEW</span>}
                              {changeLabel(a.change_type)}
                            </span>
                            <span className="font-mono text-neutral-700">{pct(a.confidence)}</span>
                          </div>
                          <div className="font-mono text-[10px] text-neutral-500 truncate">
                            {a.scene_a_date} → {a.scene_b_date}
                            {a.area_px ? ` · ${hectares(a.area_px, a.area_ha)}` : ""}
                          </div>
                        </button>
                      </li>
                    ))}
                    {mine.length > shown.length && <li className="px-2 py-1 text-[10px] text-neutral-500">{mine.length - shown.length} more</li>}
                  </ul>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
