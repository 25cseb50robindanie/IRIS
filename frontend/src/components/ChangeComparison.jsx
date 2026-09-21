import React, { useEffect, useRef, useState } from "react";
import { X, Check, Ban, Loader2, AlertCircle, ChevronRight } from "lucide-react";
import { boundsMgrs, formatLatLon, spacedMgrs } from "../mgrs";
import CompareMap from "./CompareMap";

import { changeLabel, describeDirection, DIRECTION_ICONS, hectares } from "./changeKinds";

const TERM_LABELS = {
  alignment_quality: "Alignment quality",
  cluster_distance: "Cluster distance",
  terrain_flatness: "Terrain flatness",
  valid_coverage: "Valid coverage",
};

const pct = (v) => `${(v * 100).toFixed(0)}%`;

const STATUS_STYLE = {
  pending: "bg-neutral-200 text-neutral-700",
  confirmed: "bg-emerald-100 text-emerald-800",
  rejected: "bg-red-100 text-red-800",
};

/**
 * Before/after view for one change candidate: scene A (older) on the left, scene B (newer) on the right, the
 * candidate's bounding box drawn on both, and the analyst's Confirm / Reject decision underneath.
 */
export default function ChangeComparison({ detail, loading, error, onClose, onReview }) {
  const maps = useRef({ left: null, right: null });
  const syncing = useRef(false);
  const [submitting, setSubmitting] = useState(false);
  const [reviewError, setReviewError] = useState(null);
  const [traceOpen, setTraceOpen] = useState(false); // "Processing Details": the full decision trace

  // Keep the two maps moving together: whichever one the analyst drags drives the other
  const link = (from, to) => {
    from.on("move", () => {
      if (syncing.current) return;
      syncing.current = true;
      to.jumpTo({
        center: from.getCenter(),
        zoom: from.getZoom(),
        bearing: from.getBearing(),
        pitch: from.getPitch(),
      });
      syncing.current = false;
    });
  };

  const registerMap = (side) => (map) => {
    maps.current[side] = map;
    if (maps.current.left && maps.current.right) {
      link(maps.current.left, maps.current.right);
      link(maps.current.right, maps.current.left);
    }
  };

  // A new candidate builds new maps
  useEffect(() => {
    maps.current = { left: null, right: null };
    setReviewError(null);
  }, [detail?.candidate_id]);

  const decide = async (decision) => {
    setSubmitting(true);
    setReviewError(null);
    try {
      await onReview(detail.candidate_id, decision);
    } catch (err) {
      setReviewError(err.message || "Could not save the decision");
    } finally {
      setSubmitting(false);
    }
  };

  const header = (
    <div className="h-9 px-3 flex items-center justify-between bg-qgis-header border-b border-qgis-border shrink-0">
      <span className="text-xs font-semibold text-neutral-800">Change comparison</span>
      <button
        type="button"
        onClick={onClose}
        className="flex items-center space-x-1 px-2 py-0.5 text-xs text-neutral-700 hover:bg-qgis-hover rounded-[3px]"
      >
        <X className="w-3.5 h-3.5" />
        <span>Close</span>
      </button>
    </div>
  );

  if (loading || error || !detail) {
    return (
      <div className="absolute inset-0 z-20 flex flex-col bg-white">
        {header}
        <div className="flex-1 flex items-center justify-center text-xs text-neutral-500">
          {error ? (
            <span className="flex items-center space-x-1.5 text-red-700">
              <AlertCircle className="w-4 h-4" />
              <span>{error}</span>
            </span>
          ) : (
            <span className="flex items-center space-x-1.5">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Loading candidate...</span>
            </span>
          )}
        </div>
      </div>
    );
  }

  const { scene_a: a, scene_b: b, bounds } = detail;
  // The window the view opens on: 4x the box, clamped to the scene by the backend. The orange outline stays on the
  // blob's own bounds, so the change is seen in its surroundings.
  const focus = detail.display_bounds || [
    bounds[0] - (bounds[2] - bounds[0]) * 1.5,
    bounds[1] - (bounds[3] - bounds[1]) * 1.5,
    bounds[2] + (bounds[2] - bounds[0]) * 1.5,
    bounds[3] + (bounds[3] - bounds[1]) * 1.5,
  ];
  const breakdown = detail.confidence_breakdown;
  const [lon, lat] = detail.centroid || [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2];
  const mgrsRef = detail.mgrs || boundsMgrs(bounds);
  const traceLines = detail.processing_details || [];
  const DirectionIcon = DIRECTION_ICONS[detail.direction || "unclassified"];

  return (
    <div className="absolute inset-0 z-20 flex flex-col bg-white">
      {header}

      {/* Two synchronised panes */}
      <div className="flex-1 flex min-h-0">
        {[
          { side: "left", label: "Before", scene: a },
          { side: "right", label: "After", scene: b },
        ].map(({ side, label, scene }) => (
          <div key={side} className={`relative flex-1 min-w-0 ${side === "left" ? "border-r border-white" : ""}`}>
            <CompareMap
              key={`${detail.candidate_id}-${side}`}
              cogUrl={scene.cog_url}
              box={bounds}
              focusBounds={focus}
              onReady={registerMap(side)}
            />
            <div className="absolute top-2 left-2 px-2 py-1 bg-black/70 text-white rounded-[3px] text-[11px] leading-tight pointer-events-none">
              <div className="font-semibold">{label}</div>
              <div className="font-mono">{scene.acquisition_date}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Details + decision */}
      <div className="shrink-0 max-h-[45vh] overflow-y-auto border-t border-qgis-border bg-qgis-bg px-4 py-3 flex flex-wrap items-start gap-x-8 gap-y-3 text-xs text-qgis-text select-none">
        <div className="min-w-[210px] max-w-[340px]">
          <div className="flex items-center space-x-2">
            {DirectionIcon && <DirectionIcon className="w-4 h-4 text-neutral-700" aria-hidden="true" />}
            <span className="text-sm font-semibold">{changeLabel(detail.change_type)}</span>
            <span className={`px-1.5 py-0.5 rounded-[3px] text-[10px] font-medium ${STATUS_STYLE[detail.review_status]}`}>
              {detail.review_status}
            </span>
          </div>
          <div className="mt-1 text-neutral-600">
            Confidence <span className="font-mono font-semibold text-neutral-900">{pct(detail.confidence)}</span>
            {detail.mean_dndvi != null && (
              <span className="ml-2 font-mono text-neutral-500">ΔNDVI {detail.mean_dndvi.toFixed(2)}</span>
            )}
            {detail.area_px ? (
              <span className="ml-2 font-mono text-neutral-500" data-testid="detail-area">
                {hectares(detail.area_px)}
              </span>
            ) : null}
          </div>
          <div className="mt-1 text-neutral-700 leading-snug" data-testid="direction-text">
            {describeDirection(detail)}
          </div>
          <div className="mt-1 font-mono text-[11px] text-neutral-600" data-testid="detail-location">
            {formatLatLon(lat, lon)}
            {mgrsRef ? ` · MGRS ${spacedMgrs(mgrsRef)}` : ""}
          </div>
          {detail.sub_blobs > 1 && (
            <div className="mt-0.5 text-[10px] text-neutral-500">Merged from {detail.sub_blobs} nearby change blobs</div>
          )}
          <div className="mt-2 font-mono text-[11px] text-neutral-500 space-y-0.5">
            <div className="truncate max-w-[300px]" title={a.scene_id}>
              A {a.acquisition_date} · {a.scene_id}
            </div>
            <div className="truncate max-w-[300px]" title={b.scene_id}>
              B {b.acquisition_date} · {b.scene_id}
            </div>
          </div>
        </div>

        <div className="min-w-[230px]">
          <div className="text-[10px] uppercase font-semibold text-neutral-400 mb-1">Confidence breakdown</div>
          <div className="space-y-1">
            {Object.entries(breakdown).map(([key, term]) => (
              <div key={key} className="flex items-center space-x-2">
                <span className="w-28 text-neutral-600">{TERM_LABELS[key]}</span>
                <div className="flex-1 h-1.5 bg-neutral-200 rounded-full overflow-hidden">
                  <div className="h-full bg-neutral-600" style={{ width: pct(term.value) }} />
                </div>
                <span className="w-9 text-right font-mono text-neutral-700">{pct(term.value)}</span>
              </div>
            ))}
          </div>
          {detail.terrain_is_placeholder && (
            <div className="mt-1 text-[10px] text-neutral-400">Terrain is a placeholder (no DEM loaded yet).</div>
          )}

          {traceLines.length > 0 && (
            <div className="mt-2">
              <button
                type="button"
                onClick={() => setTraceOpen((o) => !o)}
                aria-expanded={traceOpen}
                data-testid="processing-toggle"
                className="flex items-center space-x-1 text-[11px] font-semibold text-neutral-600 hover:text-neutral-900"
              >
                <ChevronRight className={`w-3 h-3 text-neutral-400 transition-transform ${traceOpen ? "rotate-90" : ""}`} />
                <span>Processing Details</span>
              </button>
              {traceOpen && (
                <dl className="mt-1 space-y-1 max-w-[440px]" data-testid="processing-details">
                  {traceLines.map((line) => (
                    <div key={line.key} className="flex text-[11px] leading-snug" data-testid={`trace-${line.key}`}>
                      <dt className="w-24 shrink-0 text-neutral-500">{line.label}</dt>
                      <dd className="text-neutral-800 min-w-0 break-words">{line.text}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          )}
        </div>

        <div className="ml-auto flex flex-col items-end space-y-1.5">
          <div className="flex items-center space-x-2">
            <button
              type="button"
              disabled={submitting}
              onClick={() => decide("rejected")}
              className="flex items-center space-x-1.5 px-3 py-1.5 bg-white hover:bg-red-50 text-red-800 border border-qgis-border rounded-[3px] text-xs font-medium disabled:opacity-50"
            >
              <Ban className="w-3.5 h-3.5" />
              <span>Reject</span>
            </button>
            <button
              type="button"
              disabled={submitting}
              onClick={() => decide("confirmed")}
              className="flex items-center space-x-1.5 px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 text-white rounded-[3px] text-xs font-medium disabled:opacity-50"
            >
              {submitting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
              <span>Confirm</span>
            </button>
          </div>
          {reviewError && <div className="text-[11px] text-red-700">{reviewError}</div>}
          {detail.reviews?.length > 0 && (
            <div className="text-[10px] text-neutral-400 font-mono">
              Last: {detail.reviews[detail.reviews.length - 1].decision} by{" "}
              {detail.reviews[detail.reviews.length - 1].analyst_id} at{" "}
              {detail.reviews[detail.reviews.length - 1].reviewed_at}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
