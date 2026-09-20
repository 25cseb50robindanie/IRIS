import React from "react";
import { Check, Loader2, AlertCircle, Circle, MinusCircle, X } from "lucide-react";

// A detail like "12/100" reads as a count "(12/100)"; a sentence such as "against S2A_..." reads as-is
const withDetail = (label, detail, active) => {
  const dots = active ? "..." : "";
  if (!detail) return `${label}${dots}`;
  if (detail.startsWith("against")) return `${label}${dots} ${detail}`; // "Running change detection against X"
  return active ? `${label}... (${detail})` : `${label} — ${detail}`;
};

function StepIcon({ state }) {
  switch (state) {
    case "done":
      return <Check className="w-3 h-3 text-emerald-600" />;
    case "active":
      return <Loader2 className="w-3 h-3 text-amber-600 animate-spin" />;
    case "failed":
      return <AlertCircle className="w-3 h-3 text-red-600" />;
    case "skipped":
      return <MinusCircle className="w-3 h-3 text-neutral-300" />;
    default:
      return <Circle className="w-3 h-3 text-neutral-300" />;
  }
}

const TEXT_STYLE = {
  done: "text-neutral-700",
  active: "text-neutral-900 font-medium",
  failed: "text-red-700",
  skipped: "text-neutral-400",
  pending: "text-neutral-400",
};

/**
 * Step-by-step progress of an import, from format detection to change detection. Every step is real work reported
 * by the backend; nothing here is a timer.
 */
export default function PipelineProgress({ pipeline, onDismiss }) {
  const finished = pipeline.state !== "running";
  const failed = pipeline.state === "failed";

  return (
    <div className="mb-2 border border-neutral-200 rounded-[3px] bg-white p-2.5">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[11px] font-semibold text-neutral-700 truncate" title={pipeline.name}>
          {finished ? "Import" : "Importing"} {pipeline.name && <span className="font-mono font-normal">{pipeline.name}</span>}
        </span>
        <span className="flex items-center space-x-1.5 shrink-0">
          <span className="font-mono text-[10px] text-neutral-400">{Math.round(pipeline.elapsed_s)}s</span>
          {finished && onDismiss && (
            <button
              type="button"
              onClick={onDismiss}
              title="Dismiss"
              className="text-neutral-400 hover:text-neutral-600 p-0.5"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </span>
      </div>

      <ol className="space-y-1">
        {pipeline.steps.map((step) => {
          const active = step.state === "active";
          const showBar = active && step.total > 0 && step.done != null;
          return (
            <li key={step.key} className="text-[11px] leading-snug">
              <div className="flex items-start space-x-1.5">
                <span className="mt-0.5 shrink-0">
                  <StepIcon state={step.state} />
                </span>
                <span className={`min-w-0 break-words ${TEXT_STYLE[step.state] || "text-neutral-500"}`}>
                  {withDetail(step.label, step.detail, active)}
                </span>
              </div>
              {showBar && (
                <div className="ml-[18px] mt-0.5 h-1 bg-neutral-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-amber-500"
                    style={{ width: `${Math.min((step.done / step.total) * 100, 100)}%` }}
                  />
                </div>
              )}
            </li>
          );
        })}
      </ol>

      {pipeline.message && (
        <div
          className={`mt-2 pt-1.5 border-t border-neutral-100 text-[11px] font-medium ${
            failed ? "text-red-700" : "text-emerald-800"
          }`}
        >
          {pipeline.message}
        </div>
      )}
    </div>
  );
}
