import React from "react";
import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";

const STYLE = {
  error: { box: "bg-red-50 border-red-300 text-red-900", icon: AlertCircle, iconClass: "text-red-600" },
  success: { box: "bg-emerald-50 border-emerald-300 text-emerald-900", icon: CheckCircle2, iconClass: "text-emerald-600" },
  info: { box: "bg-white border-neutral-300 text-neutral-800", icon: Info, iconClass: "text-neutral-500" },
  alert: { box: "bg-amber-50 border-amber-400 text-amber-950", icon: AlertCircle, iconClass: "text-amber-600" },
};

/**
 * Messages the analyst must not miss: a failed request, a lost connection, a watchlist hit. They stack at the bottom of the
 * window, go away by themselves (errors stay longer) and can be dismissed. `toasts` is [{id, kind, message, title?}].
 */
export default function Toaster({ toasts, onDismiss }) {
  if (!toasts.length) return null;
  return (
    <div className="fixed bottom-14 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center space-y-2 pointer-events-none" aria-live="polite">
      {toasts.map((t) => {
        const { box, icon: Icon, iconClass } = STYLE[t.kind] || STYLE.info;
        return (
          <div
            key={t.id}
            role={t.kind === "error" ? "alert" : "status"}
            data-testid="toast"
            data-kind={t.kind}
            className={`pointer-events-auto flex items-start space-x-2 max-w-md px-3 py-2 border rounded-[3px] shadow-lg text-xs ${box}`}
          >
            <Icon className={`w-4 h-4 shrink-0 mt-px ${iconClass}`} />
            <div className="min-w-0 break-words">
              {t.title && <div className="font-semibold">{t.title}</div>}
              <div className={t.title ? "text-[11px] opacity-90" : ""}>{t.message}</div>
            </div>
            <button type="button" onClick={() => onDismiss(t.id)} aria-label="Dismiss" className="shrink-0 opacity-60 hover:opacity-100">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
