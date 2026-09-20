import React, { useEffect, useRef, useState } from "react";
import { ChevronDown, Trash2, Loader2, AlertCircle } from "lucide-react";

const label = (s) => `${s.acquisition_date} · ${s.sensor}`;

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** What deleting a scene takes with it, spelled out before the analyst confirms. */
function consequences(scene) {
  const parts = [plural(scene.tiles_count, "embedded tile")];
  if (scene.change_candidates > 0) {
    const reviewed = scene.reviews > 0 ? ` (${plural(scene.reviews, "analyst decision")} included)` : "";
    parts.push(`${plural(scene.change_candidates, "change candidate")}${reviewed}`);
  }
  return `This removes ${parts.join(" and ")}, plus its image files. Your original source files are not touched.`;
}

/**
 * Toolbar dropdown of every imported scene: click one to show it on the map, or use the trash icon to delete it
 * (with an explicit confirmation of what goes with it).
 */
export default function SceneMenu({ scenes = [], currentSceneId, onSelect, onDelete, onOpen, disabled }) {
  const [open, setOpen] = useState(false);
  const [confirmId, setConfirmId] = useState(null);
  const [busyId, setBusyId] = useState(null);
  const [rowError, setRowError] = useState(null);
  const [notice, setNotice] = useState(null);
  const root = useRef(null);

  // Close on an outside click or Escape
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (root.current && !root.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const toggle = () => {
    if (!open && onOpen) onOpen();
    setOpen((o) => !o);
    setConfirmId(null);
    setRowError(null);
    setNotice(null);
  };

  const current = scenes.find((s) => s.scene_id === currentSceneId);

  const confirmDelete = async (scene) => {
    setBusyId(scene.scene_id);
    setRowError(null);
    try {
      const result = await onDelete(scene.scene_id);
      setConfirmId(null);
      if (result?.files_failed?.length) {
        setNotice(`Deleted, but these files could not be removed: ${result.files_failed.join(", ")}`);
      }
    } catch (err) {
      setRowError({ id: scene.scene_id, text: err.message || "Could not delete the scene" });
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="relative" ref={root}>
      <button
        type="button"
        onClick={toggle}
        disabled={disabled}
        title="Imported scenes"
        aria-expanded={open}
        className="flex items-center space-x-1.5 px-2.5 py-1 bg-white hover:bg-qgis-hover disabled:opacity-50 text-qgis-text border border-qgis-border rounded-[3px] text-xs font-medium transition-colors max-w-[230px]"
      >
        <span className="truncate">{current ? label(current) : scenes.length ? "Select scene" : "No scenes"}</span>
        <span className="font-mono text-[10px] text-neutral-500 shrink-0">{scenes.length}</span>
        <ChevronDown className="w-3 h-3 text-neutral-500 shrink-0" />
      </button>

      {open && (
        <div className="absolute left-0 top-full mt-1 w-[26rem] max-h-[70vh] overflow-y-auto bg-white border border-qgis-border rounded-[3px] shadow-lg z-40">
          {scenes.length === 0 ? (
            <div className="p-4 text-center text-xs text-neutral-500">No scenes imported yet. Use Browse to import one.</div>
          ) : (
            <ul className="divide-y divide-qgis-border">
              {scenes.map((scene) => {
                const active = scene.scene_id === currentSceneId;
                const confirming = confirmId === scene.scene_id;
                const busy = busyId === scene.scene_id;
                return (
                  <li key={scene.scene_id} className={active ? "bg-neutral-100" : ""}>
                    <div className="flex items-stretch">
                      <button
                        type="button"
                        disabled={!scene.cog_available || busy}
                        onClick={() => {
                          onSelect(scene.scene_id);
                          setOpen(false);
                        }}
                        className="flex-1 min-w-0 text-left px-3 py-2 hover:bg-neutral-50 disabled:cursor-not-allowed disabled:hover:bg-transparent"
                      >
                        <div className="flex items-center justify-between space-x-2">
                          <span className={`text-xs font-semibold ${scene.cog_available ? "text-neutral-800" : "text-neutral-400"}`}>
                            {label(scene)}
                          </span>
                          <span className="font-mono text-[10px] text-neutral-500 shrink-0">
                            {scene.cloud_pct != null ? `cloud ${scene.cloud_pct.toFixed(0)}% · ` : ""}
                            {plural(scene.tiles_count, "tile")}
                          </span>
                        </div>
                        <div className="font-mono text-[10px] text-neutral-500 truncate" title={scene.scene_id}>
                          {scene.scene_id}
                        </div>
                        {!scene.cog_available && (
                          <div className="text-[10px] text-amber-700">Image file missing; it can only be deleted</div>
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setConfirmId(confirming ? null : scene.scene_id);
                          setRowError(null);
                        }}
                        disabled={busy}
                        title="Delete this scene"
                        className="px-2.5 text-neutral-400 hover:text-red-700 hover:bg-red-50 disabled:opacity-50"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    {confirming && (
                      <div className="px-3 py-2 bg-red-50 border-t border-red-100 text-[11px] text-red-900">
                        <div className="font-medium">Delete {label(scene)}?</div>
                        <div className="mt-0.5">{consequences(scene)}</div>
                        {rowError?.id === scene.scene_id && (
                          <div className="mt-1 flex items-start space-x-1 text-red-700">
                            <AlertCircle className="w-3 h-3 mt-0.5 shrink-0" />
                            <span>{rowError.text}</span>
                          </div>
                        )}
                        <div className="mt-2 flex items-center space-x-2">
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() => confirmDelete(scene)}
                            className="flex items-center space-x-1 px-2.5 py-1 bg-red-700 hover:bg-red-800 text-white rounded-[3px] font-medium disabled:opacity-60"
                          >
                            {busy && <Loader2 className="w-3 h-3 animate-spin" />}
                            <span>{busy ? "Deleting..." : "Delete"}</span>
                          </button>
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() => setConfirmId(null)}
                            className="px-2.5 py-1 bg-white border border-qgis-border rounded-[3px] text-neutral-700 hover:bg-neutral-50"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
          {notice && <div className="px-3 py-2 text-[11px] text-amber-800 bg-amber-50 border-t border-amber-100">{notice}</div>}
        </div>
      )}
    </div>
  );
}
