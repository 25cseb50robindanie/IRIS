import React, { useState } from "react";
import { X, Loader2 } from "lucide-react";

/**
 * Dev-mode-only fallback for browser sessions where window.iris (the native Electron dialog) isn't wired up:
 * lets the analyst paste a path instead of picking one from an OS dialog.
 */
export default function ImportPathModal({ mode, onCancel, onSubmit, onBrowse, busy = false }) {
  const [value, setValue] = useState("");

  const submit = (e) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (trimmed) onSubmit(trimmed);
  };

  const handlePick = async () => {
    if (onBrowse) {
      const picked = await onBrowse();
      if (picked) setValue(picked);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onMouseDown={(e) => e.target === e.currentTarget && onCancel()}
    >
      <form
        onSubmit={submit}
        className="w-[28rem] max-w-[90vw] bg-white border border-qgis-border rounded-[3px] shadow-lg p-4"
      >
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-semibold text-neutral-800">
            {mode === "folder" ? "Import Folder" : "Import File"}
          </h2>
          <button type="button" onClick={onCancel} className="text-neutral-400 hover:text-neutral-700">
            <X className="w-4 h-4" />
          </button>
        </div>

        <label className="block text-[11px] text-neutral-600 mb-1">Paste or select the full folder or file path</label>
        <div className="flex items-center space-x-1.5">
          <input
            autoFocus
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={mode === "folder" ? "/path/to/S2A_MSIL2A_....SAFE" : "/path/to/image.tif"}
            className="flex-1 px-2.5 py-1.5 text-xs border border-qgis-border rounded-[3px] focus:outline-none focus:ring-1 focus:ring-neutral-400"
          />
          {onBrowse && (
            <button
              type="button"
              onClick={handlePick}
              disabled={busy}
              className="px-2.5 py-1.5 bg-neutral-100 hover:bg-neutral-200 border border-qgis-border rounded-[3px] text-xs font-medium text-neutral-700 disabled:opacity-50 shrink-0"
            >
              Browse...
            </button>
          )}
        </div>
        <p className="mt-1 text-[10px] text-neutral-400">
          Click Browse to open the Windows dialog, or paste a path directly.
        </p>

        <div className="mt-3 flex items-center justify-end space-x-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="px-2.5 py-1 bg-white border border-qgis-border rounded-[3px] text-xs text-neutral-700 hover:bg-neutral-50 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={busy || !value.trim()}
            className="flex items-center space-x-1 px-2.5 py-1 bg-neutral-800 hover:bg-neutral-900 text-white rounded-[3px] text-xs font-medium disabled:opacity-50"
          >
            {busy && <Loader2 className="w-3 h-3 animate-spin" />}
            <span>Import</span>
          </button>
        </div>
      </form>
    </div>
  );
}
