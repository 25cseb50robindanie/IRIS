import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE } from "./api";

const POLL_MS = 2000;
const LIST_LIMIT = 1000; // raw detections fetched for the map and the list, largest first

const pairQuery = (pair) => new URLSearchParams({ scene_a_id: pair.a, scene_b_id: pair.b }).toString();

/**
 * The ablation of one analysed pair: how many raw detections there are with every suppression stage off, against how
 * many the full pipeline reports, and (while `enabled`) the raw detections themselves.
 *
 * `pair` is {a, b} (scene ids) or null. A pair analysed before ablation existed reports "not_run"; switching it on
 * then starts the run and follows it until it has finished.
 */
export default function useAblation(pair, enabled) {
  const [stats, setStats] = useState(null);
  const [list, setList] = useState(null); // {total, shown, candidates}
  const [error, setError] = useState(null);
  const startedFor = useRef(null); // the pair a run was already requested for, so it is requested once

  const key = pair ? `${pair.a}|${pair.b}` : null;

  const fetchStats = useCallback(async () => {
    const res = await fetch(`${API_BASE}/api/changes/ablation/stats?${pairQuery(pair)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setStats(data);
    return data;
  }, [pair]); // eslint-disable-line react-hooks/exhaustive-deps

  // A different pair is a different question
  useEffect(() => {
    setStats(null);
    setList(null);
    setError(null);
    startedFor.current = null;
  }, [key]);

  // Statistics, followed while a run is in progress; a run is started when the analyst asks for a pair that has none
  useEffect(() => {
    if (!key) return undefined;
    let cancelled = false;
    let timer = null;

    const tick = async () => {
      try {
        const data = await fetchStats();
        if (cancelled) return;
        setError(null);
        if (data.status === "not_run" && enabled && startedFor.current !== key) {
          startedFor.current = key;
          const started = await fetch(`${API_BASE}/api/changes/ablation/run?${pairQuery(pair)}`, { method: "POST" });
          if (!started.ok && started.status !== 409) throw new Error(`HTTP ${started.status}`);
          if (!cancelled) timer = setTimeout(tick, POLL_MS);
        } else if (data.status === "running") {
          timer = setTimeout(tick, POLL_MS);
        }
      } catch (err) {
        if (!cancelled) setError("Could not read the ablation results.");
      }
    };
    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [key, enabled]); // eslint-disable-line react-hooks/exhaustive-deps

  // The raw detections, only while they are wanted and the run has finished
  const done = stats?.status === "completed";
  const count = stats?.ablation_count;
  useEffect(() => {
    if (!key || !enabled || !done) return undefined;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/changes/ablation/candidates?${pairQuery(pair)}&limit=${LIST_LIMIT}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!cancelled) setList(data);
      } catch (err) {
        if (!cancelled) setError("Could not load the raw detections.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [key, enabled, done, count]); // eslint-disable-line react-hooks/exhaustive-deps

  return { stats, list, error };
}
