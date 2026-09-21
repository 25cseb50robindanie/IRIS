import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE } from "./api";

const POLL_MS = 4000;

async function detail(res, fallback) {
  const err = await res.json().catch(() => ({}));
  const d = err.detail;
  return typeof d === "string" ? d : Array.isArray(d) && d[0]?.msg ? d[0].msg : `${fallback} (HTTP ${res.status})`;
}

/**
 * The analyst's watchlist and the alerts the backend raised on it. The backend does the watching (it checks every completed
 * job); this only shows what it found, so it polls, and tells `notify` when an alert appears that it has not shown before.
 *
 * Returns {locations, alerts, unacknowledged, add, remove, acknowledge, acknowledgeAll, busy}. `add` takes
 * {name?, center: [lon, lat], radius_m?, confidence_threshold?} or {bounds}.
 */
export default function useWatchlist(notify) {
  const [locations, setLocations] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [busy, setBusy] = useState(false);
  const seen = useRef(null); // alert ids already known; null until the first response, so old alerts do not pop up on load
  const notifyRef = useRef(notify);
  notifyRef.current = notify;

  const refresh = useCallback(async () => {
    const [w, a] = await Promise.all([fetch(`${API_BASE}/api/watchlist`), fetch(`${API_BASE}/api/watchlist/alerts`)]);
    if (!w.ok || !a.ok) throw new Error("Watchlist unavailable");
    const locs = await w.json();
    const alertBody = await a.json();
    setLocations((prev) => (JSON.stringify(prev) === JSON.stringify(locs) ? prev : locs));
    setAlerts((prev) => (JSON.stringify(prev) === JSON.stringify(alertBody.alerts) ? prev : alertBody.alerts));

    const ids = new Set(alertBody.alerts.map((x) => x.alert_id));
    if (seen.current !== null) {
      const fresh = alertBody.alerts.filter((x) => !seen.current.has(x.alert_id) && !x.acknowledged_at);
      if (fresh.length) {
        const first = fresh[0];
        notifyRef.current?.({
          kind: "alert",
          title: `Watchlist: ${fresh.length === 1 ? "new detection" : `${fresh.length} new detections`} at ${first.watch_name}`,
          message: `${first.change_type || "change"} · confidence ${Math.round(first.confidence * 100)}%`,
        });
      }
    }
    seen.current = ids;
  }, []);

  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      if (cancelled || document.hidden) return;
      refresh().catch(() => {}); // the app's connection monitor reports an unreachable backend; this just tries again
    };
    tick();
    const timer = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [refresh]);

  // Every action either succeeds or tells the analyst why not
  const act = useCallback(
    async (label, request, success) => {
      setBusy(true);
      try {
        const res = await request();
        if (!res.ok) throw new Error(await detail(res, label));
        await refresh().catch(() => {});
        if (success) notifyRef.current?.({ kind: "success", message: success });
        return true;
      } catch (err) {
        notifyRef.current?.({ kind: "error", title: label, message: err instanceof TypeError ? "Could not reach the IRIS backend." : err.message });
        return false;
      } finally {
        setBusy(false);
      }
    },
    [refresh]
  );

  const add = useCallback(
    (body) =>
      act(
        "Could not add the watched location",
        () => fetch(`${API_BASE}/api/watchlist`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
        "Watching this location. New detections here will raise an alert."
      ),
    [act]
  );
  const remove = useCallback(
    (id) => act("Could not remove the watched location", () => fetch(`${API_BASE}/api/watchlist/${id}`, { method: "DELETE" }), "Removed from the watchlist."),
    [act]
  );
  const acknowledge = useCallback(
    (alertId) => act("Could not mark the alert as seen", () => fetch(`${API_BASE}/api/watchlist/alerts/${alertId}/acknowledge`, { method: "POST" })),
    [act]
  );
  const acknowledgeAll = useCallback(
    () => act("Could not mark the alerts as seen", () => fetch(`${API_BASE}/api/watchlist/alerts/acknowledge`, { method: "POST" })),
    [act]
  );

  return { locations, alerts, unacknowledged: alerts.filter((a) => !a.acknowledged_at).length, add, remove, acknowledge, acknowledgeAll, busy };
}
