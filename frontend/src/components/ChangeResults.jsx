import React, { useMemo, useState } from "react";
import { AlertTriangle, Loader2, AlertCircle, Check, Ban, Copy, Download, ShieldCheck, Images } from "lucide-react";
import { downloadJson } from "../download";
import { boundsMgrs, spacedMgrs } from "../mgrs";
import ExportMenu from "./ExportMenu";
import { allOn, changeLabel, DIRECTION_ICONS, DIRECTION_OPTIONS, directionLabel, hectares, SEASONALITY, TYPE_OPTIONS } from "./changeKinds";

export const DEFAULT_FILTERS = {
  minConfidence: 0.3,
  types: allOn(TYPE_OPTIONS),
  directions: allOn(DIRECTION_OPTIONS),
  sort: "confidence",
  hideSeasonal: true, // leave out drops the persistence filter calls seasonal
  jobId: null,
};

/** The search-view matches that pass the analyst's filters (the browse view is filtered by the server). */
export function applyMatchFilters(results, filters) {
  return (results || []).filter(
    (c) =>
      c.confidence >= filters.minConfidence &&
      filters.types[c.change_type || "unclassified"] !== false &&
      filters.directions[c.direction || "unclassified"] !== false &&
      !(filters.hideSeasonal && c.seasonality_status === "seasonal") &&
      (filters.jobId == null || c.job_id === filters.jobId)
  );
}

// Pair jobs that are not (yet) a result the analyst can browse
const JOB_STATES = {
  queued: { text: "Queued", tone: "busy" },
  processing: { text: "Analysing", tone: "busy" },
  insufficient_evidence: { text: "Insufficient evidence", tone: "warn" },
  alignment_failed: { text: "Alignment failed", tone: "warn" },
  grid_mismatch: { text: "Grid mismatch", tone: "warn" },
  failed: { text: "Analysis failed", tone: "error" },
};

const PHASE_TEXT = {
  masking: "quality masking",
  grid_check: "grid check",
  alignment: "alignment",
  radiometry: "normalisation",
  detection: "detection",
  grouping: "grouping",
  scoring: "scoring",
  direction: "direction classification",
  merging: "merging",
  seasonality: "seasonal filter",
};

const ABLATION_CARD_CAP = 200; // cards drawn in the list; the map shows every raw detection that was fetched

const REVIEW_STYLE = {
  pending: "bg-neutral-100 text-neutral-600",
  confirmed: "bg-emerald-100 text-emerald-800",
  rejected: "bg-red-100 text-red-800",
};

// The whole card reflects the analyst's decision: confirmed reads green, rejected reads red and struck through
const CARD_STYLE = {
  pending: "bg-white border-neutral-200 hover:border-neutral-300 hover:bg-neutral-50",
  confirmed: "bg-emerald-50 border-emerald-300 hover:border-emerald-400",
  rejected: "bg-red-50/60 border-red-200 hover:border-red-300",
};
const TITLE_STYLE = {
  pending: "text-neutral-800",
  confirmed: "text-emerald-800",
  rejected: "text-red-700 line-through",
};

const centre = (b) => {
  const lon = (b[0] + b[2]) / 2;
  const lat = (b[1] + b[3]) / 2;
  return `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}`;
};

const pct = (v) => `${Math.round(v * 100)}%`;

function JobLine({ job }) {
  const state = JOB_STATES[job.status];
  const pair = job.scene_a_date && job.scene_b_date ? `${job.scene_a_date} → ${job.scene_b_date}` : `${job.scene_a_id} → ${job.scene_b_id}`;
  const busy = state?.tone === "busy";
  return (
    <div className="flex items-start space-x-1.5 text-[11px]">
      {busy ? (
        <Loader2 className="w-3 h-3 animate-spin text-amber-600 shrink-0 mt-0.5" />
      ) : (
        <AlertCircle className={`w-3 h-3 shrink-0 mt-0.5 ${state?.tone === "error" ? "text-red-600" : "text-amber-600"}`} />
      )}
      <div className="min-w-0 flex-1">
        <div className="font-mono text-neutral-500 truncate" title={`${job.scene_a_id} → ${job.scene_b_id}`}>
          {pair}
        </div>
        <div className="text-neutral-700">
          {busy ? `${state.text}${job.phase && PHASE_TEXT[job.phase] ? ` (${PHASE_TEXT[job.phase]})` : ""}...` : state?.text || job.status}
        </div>
        {job.error && <div className="text-neutral-500 break-words">{job.error}</div>}
      </div>
    </div>
  );
}

function Chip({ active, onClick, children, title }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={`px-2 py-0.5 rounded-[3px] text-[11px] font-mono border transition-colors ${
        active ? "bg-neutral-800 text-white border-neutral-800" : "bg-neutral-100 text-neutral-700 border-neutral-200 hover:bg-neutral-200"
      }`}
    >
      {children}
    </button>
  );
}

/** Every analysed date pair with its candidate count; clicking one filters the list to that pair. */
function PairChips({ pairs, selectedJobId, onSelect }) {
  const done = pairs.filter((p) => p.status === "completed");
  if (!done.length) return null;
  return (
    <div className="flex flex-wrap gap-1 mb-2" role="group" aria-label="Date pairs">
      {done.length > 1 && (
        <Chip active={selectedJobId == null} onClick={() => onSelect(null)}>
          All pairs
        </Chip>
      )}
      {done.map((p) => (
        <Chip
          key={p.job_id}
          active={selectedJobId === p.job_id}
          onClick={() => onSelect(selectedJobId === p.job_id ? null : p.job_id)}
          title={`${p.scene_a_id} → ${p.scene_b_id}`}
        >
          {p.scene_a_date} → {p.scene_b_date} ({p.candidates})
        </Chip>
      ))}
    </div>
  );
}

function ChangeFilters({ filters, onChange, showMatchSort }) {
  // At least one box of each group stays ticked: an empty group would be an empty list with no explanation
  const setGroup = (group, key, on) => {
    const next = { ...filters[group], [key]: on };
    if (!Object.values(next).some(Boolean)) return;
    onChange({ ...filters, [group]: next });
  };
  const groups = [
    ["types", "Type", TYPE_OPTIONS],
    ["directions", "Direction", DIRECTION_OPTIONS],
  ];
  return (
    <div className="mb-2 p-2 border border-neutral-200 rounded-[3px] bg-neutral-50 space-y-1.5 text-[11px] text-neutral-700">
      <label className="flex items-center space-x-2">
        <span className="w-24 shrink-0">Confidence ≥ {filters.minConfidence.toFixed(2)}</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={filters.minConfidence}
          onChange={(e) => onChange({ ...filters, minConfidence: parseFloat(e.target.value) })}
          className="flex-1 accent-neutral-700"
          aria-label="Minimum confidence"
        />
      </label>
      {groups.map(([group, title, options]) => (
        <div key={group} className="flex flex-wrap gap-x-3 gap-y-0.5" role="group" aria-label={`${title} filter`}>
          <span className="w-full text-[10px] uppercase font-semibold text-neutral-400">{title}</span>
          {options.map(([key, label]) => (
            <label key={key} className="flex items-center space-x-1 cursor-pointer">
              <input
                type="checkbox"
                checked={filters[group][key] !== false}
                onChange={(e) => setGroup(group, key, e.target.checked)}
                className="w-3 h-3"
                data-testid={`${group}-${key}`}
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
      ))}
      <label className="flex items-center space-x-1.5 cursor-pointer" title="Vegetation drops that match what this season usually looks like here">
        <input
          type="checkbox"
          checked={filters.hideSeasonal}
          onChange={(e) => onChange({ ...filters, hideSeasonal: e.target.checked })}
          className="w-3 h-3"
          data-testid="hide-seasonal"
        />
        <span>Hide seasonal</span>
      </label>
      <label className="flex items-center space-x-2">
        <span className="w-24 shrink-0">Sort by</span>
        <select
          value={filters.sort}
          onChange={(e) => onChange({ ...filters, sort: e.target.value })}
          className="flex-1 h-6 bg-white border border-qgis-border rounded-[3px] text-[11px] px-1"
          aria-label="Sort"
        >
          {showMatchSort && <option value="match">Best match</option>}
          <option value="confidence">Confidence</option>
          <option value="area">Area (largest first)</option>
          <option value="date">Date</option>
        </select>
      </label>
    </div>
  );
}

function ChangeCard({ c, rank, selected, onOpen, onFindSimilar, showMatch }) {
  const status = c.review_status || "pending";
  const DirectionIcon = DIRECTION_ICONS[c.direction || "unclassified"];
  const mgrsRef = c.mgrs || boundsMgrs(c.bounds);
  const season = SEASONALITY[c.seasonality_status];
  const open = () => onOpen && onOpen(c.candidate_id, "list");
  // A div, not a button: the card holds its own Find Similar button, and buttons cannot nest
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={open}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      }}
      data-testid="change-card"
      data-candidate-id={c.candidate_id}
      data-review-status={status}
      className={`w-full text-left p-2 rounded-[3px] border transition-colors cursor-pointer ${
        selected ? `border-neutral-500 ${status === "pending" ? "bg-neutral-100" : CARD_STYLE[status].split(" ")[0]}` : CARD_STYLE[status]
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-1.5 min-w-0">
          <span className="bg-neutral-900 text-white rounded-[2px] text-[10px] font-mono font-bold px-1 py-px shrink-0">#{rank}</span>
          {DirectionIcon && (
            <span title={directionLabel(c.direction)} data-testid="direction-icon" data-direction={c.direction || "unclassified"} className="shrink-0">
              <DirectionIcon className="w-3.5 h-3.5 text-neutral-600" aria-label={directionLabel(c.direction)} />
            </span>
          )}
          <span className={`text-xs font-semibold truncate ${TITLE_STYLE[status]}`} data-testid="change-title">
            {changeLabel(c.change_type)}
          </span>
          {season && (
            <span
              className={`px-1.5 py-px rounded-[3px] text-[10px] font-medium shrink-0 ${season.style}`}
              title={season.hint}
              data-testid="seasonality-badge"
              data-seasonality={c.seasonality_status}
            >
              {season.label}
            </span>
          )}
        </div>
        <span className="font-mono text-xs font-semibold text-neutral-800 shrink-0" title="Detection confidence">
          {pct(c.confidence)}
        </span>
      </div>

      <div className="mt-1 flex items-center justify-between">
        <span className="font-mono text-[10px] text-neutral-500 truncate" title="Centre of the detection">
          {centre(c.bounds)}
        </span>
        <span
          className={`flex items-center space-x-0.5 px-1.5 py-px rounded-[3px] text-[10px] font-medium shrink-0 ${REVIEW_STYLE[status]}`}
        >
          {status === "confirmed" && <Check className="w-2.5 h-2.5" />}
          {status === "rejected" && <Ban className="w-2.5 h-2.5" />}
          <span>{status}</span>
        </span>
      </div>

      {mgrsRef && (
        <div className="mt-0.5 font-mono text-[10px] text-neutral-500 truncate" title={`MGRS ${mgrsRef}`} data-testid="change-mgrs">
          MGRS {spacedMgrs(mgrsRef)}
        </div>
      )}

      <div className="mt-0.5 font-mono text-[10px] text-neutral-400 truncate">
        {c.scene_a_date} → {c.scene_b_date}
        {c.mean_dndvi != null ? ` · ΔNDVI ${c.mean_dndvi.toFixed(2)}` : ""}
        {c.area_px ? ` · ${hectares(c.area_px, c.area_ha)}` : ""}
        {c.sub_blobs > 1 ? ` · ${c.sub_blobs} blobs merged` : ""}
      </div>

      {showMatch && (
        <div
          className={`mt-1 flex items-center space-x-1 text-[10px] ${c.matches_query ? "text-emerald-700" : "text-neutral-500"}`}
          title={`After-image similarity ${c.semantic_similarity?.toFixed(3)}; combined score ${c.combined_score?.toFixed(2)}`}
        >
          {c.matches_query && <Check className="w-2.5 h-2.5" />}
          <span>
            After-image {c.matches_query ? "matches" : "does not match"} the query · {pct(c.semantic_match_score)}
          </span>
        </div>
      )}
      {showMatch && c.direction_boost > 0 && (
        <div className="mt-0.5 text-[10px] text-neutral-500" title="The query's wording hints at this direction">
          +{c.direction_boost.toFixed(1)} score: query implies {directionLabel(c.direction).toLowerCase()}
        </div>
      )}

      {onFindSimilar && (
        <div className="mt-1.5 flex justify-end">
          <button
            type="button"
            title="Find similar sites: tiles that look like this change's after-image"
            aria-label="Find similar"
            data-testid="find-similar-change"
            onClick={(e) => {
              e.stopPropagation();
              onFindSimilar({ candidateId: c.candidate_id });
            }}
            className="flex items-center space-x-1 px-1.5 py-0.5 text-[10px] text-neutral-600 border border-neutral-200 rounded-[3px] bg-white hover:bg-neutral-100"
          >
            <Images className="w-3 h-3" />
            <span>Find Similar</span>
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * "Nothing changed" as evidence, not as an empty list: it says when the area was observed and how completely, so the
 * analyst can hand it on. Copy puts the statement on the clipboard; Export saves the same facts as JSON.
 */
function NegativeEvidenceCard({ evidence, query, sceneId }) {
  const [copied, setCopied] = useState(null);
  const coverage = evidence.validCoverage != null ? pct(evidence.validCoverage) : "unknown";
  const t = evidence.thresholds || {};
  const statement = `This area was observed on ${evidence.datesCompared} with ${coverage} valid coverage. No significant change was detected. This is a confirmed negative — the area was analysed, not just unexamined.`;

  // The exported statement is the analyst's to hand on: what was observed, how completely, and what was asked
  const exportedStatement = `This area was observed on the listed dates with ${coverage} valid coverage. ${
    query ? "No changes matching the query were detected above the confidence threshold." : "No changes were detected above the confidence threshold."
  }`;
  const mgrsRef = evidence.mgrs || boundsMgrs(evidence.areaBounds);

  const record = () => ({
    type: "negative_evidence",
    generated_at: new Date().toISOString(),
    query: query || null,
    area_bounds: evidence.areaBounds || null,
    mgrs: mgrsRef || null,
    dates_analysed: evidence.datesAnalysed || [],
    valid_coverage: evidence.validCoverage,
    // An absence can only be as certain as the area was observed, so the two are the same number
    confidence_in_absence: evidence.validCoverage,
    statement: exportedStatement,
    scene_id: sceneId || null,
    pairs: evidence.pairs,
    method: {
      minimum_mapping_unit_px: t.minimum_mapping_unit_px ?? null,
      minimum_confidence: t.minimum_confidence ?? null,
    },
  });

  const copy = async () => {
    const r = record();
    const text = [
      "NEGATIVE EVIDENCE — no significant change detected",
      r.scene_id && `Scene: ${r.scene_id}`,
      r.mgrs && `MGRS: ${r.mgrs}`,
      `Observed: ${(r.dates_analysed || []).join(", ") || evidence.datesCompared}`,
      `Valid coverage: ${coverage}`,
      ...(r.pairs || []).map((p) => `  Pair ${p.dates}${p.valid_coverage != null ? ` (coverage ${pct(p.valid_coverage)})` : ""}`),
      r.method.minimum_mapping_unit_px != null && `Method: minimum mapping unit ${r.method.minimum_mapping_unit_px} px, minimum confidence ${r.method.minimum_confidence}`,
      r.query && `Query: "${r.query}"`,
      "",
      r.statement,
      `Generated: ${r.generated_at}`,
    ]
      .filter((l) => l !== false && l !== null && l !== undefined)
      .join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied("Copied");
    } catch (err) {
      setCopied("Copy failed");
    }
    setTimeout(() => setCopied(null), 2000);
  };

  const exportJson = () => downloadJson(`negative-evidence-${new Date().toISOString().slice(0, 10)}.json`, record());

  return (
    <div className="border border-emerald-200 bg-emerald-50/60 rounded-[3px] p-2.5" data-testid="negative-evidence">
      <div className="flex items-center space-x-1.5 text-xs font-semibold text-emerald-900">
        <ShieldCheck className="w-3.5 h-3.5 text-emerald-700" />
        <span>Confirmed negative</span>
      </div>
      <p className="mt-1 text-[11px] leading-snug text-emerald-950">{statement}</p>
      {mgrsRef && (
        <div className="mt-1 font-mono text-[10px] text-emerald-900" data-testid="negative-mgrs">
          Area centre MGRS {spacedMgrs(mgrsRef)}
        </div>
      )}
      <div className="mt-2 flex items-center space-x-2">
        <button
          type="button"
          onClick={copy}
          className="flex items-center space-x-1 px-2 py-1 bg-white border border-emerald-200 rounded-[3px] text-[11px] text-emerald-900 hover:bg-emerald-50"
        >
          <Copy className="w-3 h-3" />
          <span>{copied || "Copy"}</span>
        </button>
        <button
          type="button"
          onClick={exportJson}
          className="flex items-center space-x-1 px-2 py-1 bg-white border border-emerald-200 rounded-[3px] text-[11px] text-emerald-900 hover:bg-emerald-50"
        >
          <Download className="w-3 h-3" />
          <span>Export</span>
        </button>
      </div>
    </div>
  );
}

const Note = ({ children }) => (
  <div className="border border-dashed border-neutral-300 rounded-[3px] p-3 text-center text-[11px] text-neutral-500 leading-snug">{children}</div>
);

const LinkButton = ({ onClick, children }) => (
  <button type="button" onClick={onClick} className="mt-1.5 text-[11px] text-neutral-600 underline hover:text-neutral-900">
    {children}
  </button>
);

/**
 * The Ablation switch. Off: the full pipeline's results. On: every suppression stage is off and the raw detections are
 * shown instead. Orange-red when on, so nobody mistakes the raw list for the real one.
 */
function AblationSwitch({ on, onToggle, disabled }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={disabled}
      onClick={onToggle}
      data-testid="ablation-toggle"
      title="Switch the suppression stack off to see every raw detection"
      className={`flex items-center space-x-1.5 px-2 py-1 rounded-[3px] border text-[11px] font-semibold transition-colors disabled:opacity-40 ${
        on ? "bg-red-600 border-red-700 text-white shadow-sm" : "bg-white border-neutral-300 text-neutral-700 hover:bg-neutral-100"
      }`}
    >
      <span className={`relative inline-block w-6 h-3 rounded-full ${on ? "bg-red-900/50" : "bg-neutral-300"}`}>
        <span className={`absolute top-0.5 w-2 h-2 rounded-full bg-white transition-all ${on ? "left-3.5" : "left-0.5"}`} />
      </span>
      <span>Ablation</span>
    </button>
  );
}

/** "312 raw detections against 23 with the full pipeline": the demo's single most persuasive line. */
function AblationBanner({ stats, list, error }) {
  if (error) {
    return (
      <div className="mb-2 border border-red-300 bg-red-50 rounded-[3px] p-2 text-[11px] text-red-800" data-testid="ablation-banner">
        {error}
      </div>
    );
  }
  if (!stats || stats.status === "running" || stats.status === "not_run") {
    return (
      <div
        className="mb-2 border border-red-300 bg-red-50 rounded-[3px] p-2 flex items-center space-x-1.5 text-[11px] text-red-800"
        data-testid="ablation-banner"
      >
        <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
        <span>Computing the raw detections with suppression off...</span>
      </div>
    );
  }
  if (stats.status === "failed") {
    return (
      <div className="mb-2 border border-red-300 bg-red-50 rounded-[3px] p-2 text-[11px] text-red-800" data-testid="ablation-banner">
        The ablation could not be computed for this pair.
      </div>
    );
  }
  const removes = stats.reduction_pct != null && stats.reduction_pct > 0;
  return (
    <div
      className="mb-2 border border-red-400 bg-red-50 rounded-[3px] p-2.5 text-[11px] leading-snug text-red-900"
      data-testid="ablation-banner"
    >
      <div className="flex items-start space-x-1.5">
        <AlertTriangle className="w-3.5 h-3.5 text-red-600 shrink-0 mt-px" />
        <div>
          <span className="font-bold">Suppression OFF</span> — {stats.ablation_count.toLocaleString()} raw detections vs{" "}
          {stats.full_count.toLocaleString()} with full pipeline.
          {removes && ` The suppression stack removes ${stats.reduction_pct}% of false alarms.`}
        </div>
      </div>
      {list && list.shown < stats.ablation_count && (
        <div className="mt-1 text-red-700/80">
          The {list.shown.toLocaleString()} largest are drawn and listed.
        </div>
      )}
    </div>
  );
}

function AblationCard({ c, rank }) {
  const mgrsRef = c.mgrs || boundsMgrs(c.bounds);
  return (
    <div className="p-2 rounded-[3px] border border-red-200 bg-red-50/40" data-testid="ablation-card">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-1.5">
          <span className="bg-red-700 text-white rounded-[2px] text-[10px] font-mono font-bold px-1 py-px">#{rank}</span>
          <span className="text-xs font-semibold text-red-900">Raw detection</span>
        </div>
        <span className="font-mono text-xs font-semibold text-red-900">{hectares(c.area_px, c.area_ha)}</span>
      </div>
      <div className="mt-1 font-mono text-[10px] text-neutral-500 truncate">{centre(c.bounds)}</div>
      {mgrsRef && <div className="mt-0.5 font-mono text-[10px] text-neutral-500 truncate">MGRS {spacedMgrs(mgrsRef)}</div>}
    </div>
  );
}

/**
 * The Changes tab. Before a search it lists every detected change (server-side filtered); after a search it shows the
 * changes that match the query, or says why there are none. With the ablation on it lists the raw detections instead.
 */
export default function ChangeResults({
  changes,
  changeJobs = [],
  filters,
  onFiltersChange,
  search,
  view,
  onViewChange,
  selectedChangeId,
  onOpenChange,
  onFindSimilar,
  ablation = null, // {on, onToggle, available, stats, list, error}
  onNotify = null,
}) {
  const pairs = changes?.pairs || [];
  const inSearch = view === "search" && search?.hasSearched;
  const pendingJobs = changeJobs.filter((j) => j.status !== "completed");
  const completedPairs = pairs.filter((p) => p.status === "completed");
  const ablationOn = Boolean(ablation?.on);

  // Search view: the backend returns the matching changes; the analyst's filters narrow them here
  const visibleMatches = useMemo(() => {
    const rows = applyMatchFilters(search?.results, filters);
    const sorters = {
      match: (a, b) => b.combined_score - a.combined_score,
      confidence: (a, b) => b.confidence - a.confidence,
      area: (a, b) => (b.area_px || 0) - (a.area_px || 0),
      date: (a, b) => (b.scene_b_date || "").localeCompare(a.scene_b_date || ""),
    };
    return [...rows].sort(sorters[filters.sort] || sorters.match);
  }, [search?.results, filters]);

  // Evidence for "nothing changed", from the search response or, without a search, from the analysed pairs
  const evidence = useMemo(() => {
    if (inSearch && search.status === "no_change_detected") {
      return {
        datesCompared: search.meta.dates_compared,
        validCoverage: search.meta.valid_coverage,
        pairs: search.meta.pairs,
        thresholds: search.meta.thresholds,
        areaBounds: search.meta.area_bounds,
        mgrs: search.meta.mgrs,
        datesAnalysed: search.meta.dates_analysed,
      };
    }
    // With a pair chip selected, the negative is about that pair alone
    const scoped = filters.jobId == null ? completedPairs : completedPairs.filter((p) => p.job_id === filters.jobId);
    if (!inSearch && scoped.length > 0 && changes?.total === 0) {
      const coverages = scoped.map((p) => p.valid_coverage).filter((v) => v != null);
      const starts = scoped.map((p) => p.scene_a_date).filter(Boolean).sort();
      const ends = scoped.map((p) => p.scene_b_date).filter(Boolean).sort();
      return {
        datesCompared: `${starts[0]} → ${ends[ends.length - 1]}`,
        validCoverage: coverages.length ? Math.min(...coverages) : null,
        pairs: scoped.map((p) => ({
          job_id: p.job_id,
          scene_a_id: p.scene_a_id,
          scene_b_id: p.scene_b_id,
          dates: `${p.scene_a_date} → ${p.scene_b_date}`,
          valid_coverage: p.valid_coverage,
          candidates: p.candidates,
        })),
        thresholds: changes?.thresholds,
        areaBounds: changes?.area?.bounds,
        mgrs: changes?.area?.mgrs,
        datesAnalysed: [...new Set(scoped.flatMap((p) => [p.scene_a_date, p.scene_b_date]).filter(Boolean))].sort(),
      };
    }
    return null;
  }, [inSearch, search, completedPairs, changes, filters.jobId]);

  const noComparison = inSearch ? search.status === "no_comparison_available" : pairs.length === 0 && changeJobs.length === 0;
  const showBrowse = (inSearch && search.status === "found") || (!inSearch && (changes?.total ?? 0) > 0);

  let list = null;
  let label = null;
  if (showBrowse) {
    if (inSearch) {
      list = visibleMatches;
      label = `Showing ${visibleMatches.length} of ${search.meta.matching_candidates} matching changes`;
    } else {
      list = changes.candidates;
      label = `Showing ${changes.shown} of ${changes.matching} candidates${changes.filtered_by_confidence ? " (filtered by confidence)" : ""}`;
    }
  }

  const ablationList = ablation?.list?.candidates || [];

  return (
    <div className="p-3">
      <div className="flex items-center justify-between mb-2">
        {ablation ? (
          <AblationSwitch on={ablationOn} onToggle={ablation.onToggle} disabled={!ablation.available} />
        ) : (
          <span />
        )}
        <ExportMenu
          candidateCount={changes?.total ?? 0}
          reviewCounts={changes?.review_counts}
          onError={onNotify && ((message) => onNotify({ kind: "error", title: "Export failed", message }))}
        />
      </div>

      {ablationOn ? (
        <>
          <AblationBanner stats={ablation.stats} list={ablation.list} error={ablation.error} />
          {ablation.stats?.status === "completed" && ablationList.length > 0 && (
            <>
              <div className="mb-1.5 text-[11px] text-red-800 font-mono" data-testid="ablation-count">
                Showing {Math.min(ablationList.length, ABLATION_CARD_CAP)} of {ablation.stats.ablation_count.toLocaleString()} raw detections (largest first)
              </div>
              <div className="space-y-1.5">
                {ablationList.slice(0, ABLATION_CARD_CAP).map((c, idx) => (
                  <AblationCard key={c.candidate_id} c={c} rank={idx + 1} />
                ))}
              </div>
            </>
          )}
        </>
      ) : (
        <>
          {inSearch && (
            <div className="mb-2 text-[11px] text-neutral-500 italic truncate" title={search.query}>
              Changes matching "{search.query}"
            </div>
          )}

          <PairChips pairs={pairs} selectedJobId={filters.jobId} onSelect={(jobId) => onFiltersChange({ ...filters, jobId })} />

          {pendingJobs.length > 0 && (
            <div className="mb-2 space-y-1.5 border border-neutral-200 rounded-[3px] bg-neutral-50 p-2">
              {pendingJobs.map((job) => (
                <JobLine key={job.job_id} job={job} />
              ))}
            </div>
          )}

          {noComparison && <Note>Import a second scene of this area from the same sensor to enable change analysis.</Note>}
          {noComparison && inSearch && search.meta?.note && <div className="mt-1 text-[11px] text-neutral-500">{search.meta.note}</div>}

          {evidence && <NegativeEvidenceCard evidence={evidence} query={inSearch ? search.query : null} sceneId={search?.sceneId} />}

          {inSearch && search.status === "no_match" && (
            <>
              <Note>Changes were detected in this area, but none match "{search.query}".</Note>
              <LinkButton onClick={() => onViewChange("all")}>Show all {search.meta.total_candidates} detected changes</LinkButton>
            </>
          )}

          {showBrowse && (
            <>
              <ChangeFilters filters={filters} onChange={onFiltersChange} showMatchSort={inSearch} />
              <div className="mb-1.5 text-[11px] text-neutral-500 font-mono" data-testid="change-count">
                {label}
              </div>
              {list.length > 0 ? (
                <div className="space-y-1.5">
                  {list.map((c, idx) => (
                    <ChangeCard
                      key={c.candidate_id}
                      c={c}
                      rank={idx + 1}
                      selected={selectedChangeId === c.candidate_id}
                      onOpen={onOpenChange}
                      onFindSimilar={onFindSimilar}
                      showMatch={inSearch}
                    />
                  ))}
                </div>
              ) : (
                <Note>No candidates pass the current filters.</Note>
              )}
              {inSearch && <LinkButton onClick={() => onViewChange("all")}>Show all detected changes in this area</LinkButton>}
            </>
          )}

          {!inSearch && search?.hasSearched && <LinkButton onClick={() => onViewChange("search")}>Back to results for "{search.query}"</LinkButton>}
        </>
      )}
    </div>
  );
}
