import React, { useEffect, useRef, useState } from "react";
import { Download, FileJson, Gauge, Loader2 } from "lucide-react";
import { API_BASE } from "../api";
import { downloadFromApi } from "../download";

/**
 * "Export" beside the Change Results header. GeoJSON: every change candidate with its provenance. Evaluation manifest:
 * build times, storage, index size, query latency and hardware.
 */
export default function ExportMenu({ candidateCount = 0, reviewCounts = null, onError = null }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const root = useRef(null);

  // Close on a click anywhere else or on Escape
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (root.current && !root.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const run = async (key, path, fallbackName) => {
    setBusy(key);
    setError(null);
    try {
      await downloadFromApi(`${API_BASE}${path}`, fallbackName);
      setOpen(false);
    } catch (err) {
      const message = err instanceof TypeError ? "Could not reach the IRIS backend." : err.message || "Export failed";
      setError(message);
      if (onError) onError(message);
    } finally {
      setBusy(null);
    }
  };

  // Every candidate carries its analyst_decision, so "Export All" can be filtered again outside IRIS; the other two narrow the file
  const n = (k) => reviewCounts?.[k];
  const plural = (v, word) => `${v} ${word}${v === 1 ? "" : "s"}`;
  const items = [
    {
      key: "geojson",
      icon: FileJson,
      title: "Export All (GeoJSON)",
      hint: candidateCount ? `${plural(candidateCount, "candidate")}: confirmed, rejected and pending, each marked` : "No candidates to export",
      disabled: candidateCount === 0,
      path: "/api/export/changes",
      name: "iris_changes.geojson",
    },
    {
      key: "geojson-confirmed",
      icon: FileJson,
      title: "Export Confirmed Only",
      hint: n("confirmed") != null ? `${plural(n("confirmed"), "confirmed candidate")}` : "Candidates you confirmed",
      disabled: n("confirmed") === 0 || candidateCount === 0,
      path: "/api/export/changes?decision=confirmed",
      name: "iris_changes_confirmed.geojson",
    },
    {
      key: "geojson-pending",
      icon: FileJson,
      title: "Export Pending Only",
      hint: n("pending") != null ? `${plural(n("pending"), "candidate")} not yet reviewed` : "Candidates not yet reviewed",
      disabled: n("pending") === 0 || candidateCount === 0,
      path: "/api/export/changes?decision=pending",
      name: "iris_changes_pending.geojson",
    },
    {
      key: "manifest",
      icon: Gauge,
      title: "Evaluation manifest (JSON)",
      hint: "Build times, storage, index, latency, hardware",
      disabled: false,
      path: "/api/eval/manifest",
      name: "iris_eval_manifest.json",
    },
  ];

  return (
    <div className="relative shrink-0" ref={root}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        data-testid="export-button"
        className="flex items-center space-x-1 px-1.5 py-0.5 text-[11px] text-neutral-700 border border-neutral-300 rounded-[3px] bg-white hover:bg-neutral-100"
      >
        <Download className="w-3 h-3" />
        <span>Export</span>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full mt-1 w-60 z-30 bg-white border border-neutral-300 rounded-[3px] shadow-md py-1 normal-case tracking-normal"
        >
          {items.map(({ key, icon: Icon, title, hint, disabled, path, name }) => (
            <button
              key={key}
              type="button"
              role="menuitem"
              disabled={disabled || busy !== null}
              onClick={() => run(key, path, name)}
              data-testid={`export-${key}`}
              className="w-full text-left px-2.5 py-1.5 flex items-start space-x-2 hover:bg-neutral-100 disabled:opacity-50 disabled:hover:bg-transparent"
            >
              {busy === key ? <Loader2 className="w-3.5 h-3.5 mt-0.5 animate-spin text-neutral-600 shrink-0" /> : <Icon className="w-3.5 h-3.5 mt-0.5 text-neutral-600 shrink-0" />}
              <span className="min-w-0">
                <span className="block text-[11px] font-semibold text-neutral-800">{title}</span>
                <span className="block text-[10px] text-neutral-500 leading-snug">{hint}</span>
              </span>
            </button>
          ))}
          {error && <div className="px-2.5 py-1 text-[10px] text-red-700 break-words">{error}</div>}
        </div>
      )}
    </div>
  );
}
