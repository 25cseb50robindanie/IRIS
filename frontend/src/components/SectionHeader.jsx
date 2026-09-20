import React from "react";
import { ChevronRight } from "lucide-react";

/** Clickable header for a collapsible Workspace section; the count stays visible while collapsed. */
export default function SectionHeader({ icon: Icon, title, count = 0, open, onToggle }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      className={`w-full text-left text-[11px] font-semibold text-neutral-500 hover:text-neutral-700 uppercase tracking-wider flex items-center justify-between ${
        open ? "mb-2" : ""
      }`}
    >
      <span className="flex items-center space-x-1">
        <ChevronRight className={`w-3 h-3 text-neutral-400 transition-transform ${open ? "rotate-90" : ""}`} />
        <Icon className="w-3 h-3 text-neutral-400" />
        <span>{title}</span>
      </span>
      {count > 0 && (
        <span className="bg-neutral-100 text-neutral-700 px-1.5 py-0.2 rounded text-[10px] font-mono">{count}</span>
      )}
    </button>
  );
}
