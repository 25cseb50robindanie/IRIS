import { ArrowUpRight, ArrowDownRight, ArrowUp, ArrowDown, Minus } from "lucide-react";

// What a change is called (Phase 4b). Candidates stored before direction classification arrive as "unclassified".
export const TYPE_OPTIONS = [
  ["clearance", "Clearance"],
  ["construction", "Construction"],
  ["revegetation", "Revegetation"],
  ["urban_expansion", "Urban expansion"],
  ["water_loss", "Water loss"],
  ["vegetation_retreat", "Vegetation retreat"],
  ["unclassified", "Unclassified"],
];

export const DIRECTION_OPTIONS = [
  ["appearance", "Appearance"],
  ["disappearance", "Disappearance"],
  ["expansion", "Expansion"],
  ["contraction", "Contraction"],
  ["unclassified", "Unclassified"],
];

const TYPE_LABELS = Object.fromEntries(TYPE_OPTIONS);
const DIRECTION_LABELS = Object.fromEntries(DIRECTION_OPTIONS);

// Appearance ↗, disappearance ↘, expansion ⬆, contraction ⬇
export const DIRECTION_ICONS = {
  appearance: ArrowUpRight,
  disappearance: ArrowDownRight,
  expansion: ArrowUp,
  contraction: ArrowDown,
  unclassified: Minus,
};

// The seasonal persistence filter's verdict on a vegetation drop: only candidates it applied to carry one
export const SEASONALITY = {
  seasonal: { label: "Seasonal", style: "bg-neutral-200 text-neutral-700", hint: "This drop matches what this season usually looks like here" },
  anomalous: { label: "Anomalous", style: "bg-red-100 text-red-800", hint: "This drop is beyond the normal variation for this season" },
  unverified: { label: "Unverified", style: "bg-yellow-100 text-yellow-800", hint: "Too few earlier same-season scenes to check whether this is seasonal" },
};

export const changeLabel =(type) => (type && TYPE_LABELS[type] ? `${TYPE_LABELS[type]}` : "Unclassified change");
export const directionLabel = (direction) => DIRECTION_LABELS[direction] || "Unclassified";

export const allOn = (options) => Object.fromEntries(options.map(([key]) => [key, true]));

/** "Direction: Appearance — Construction. Evidence: ΔNDVI -0.31, dominant class changed from vegetation to bare." */
export function describeDirection(detail) {
  const direction = directionLabel(detail.direction);
  const type = detail.change_type && detail.change_type !== "unclassified" ? ` — ${changeLabel(detail.change_type)}` : "";
  const ev = detail.direction_evidence;
  if (!ev) {
    return `Direction: ${direction}${type}. No direction evidence was recorded (analysed before direction classification).`;
  }
  const parts = [];
  if (ev.delta_ndvi != null) parts.push(`ΔNDVI ${ev.delta_ndvi.toFixed(2)}`);
  parts.push(
    ev.dominant_a === ev.dominant_b
      ? `dominant class stayed ${ev.dominant_a}`
      : `dominant class changed from ${ev.dominant_a} to ${ev.dominant_b}`
  );
  if (detail.direction === "unclassified") parts.push(ev.rule);
  return `Direction: ${direction}${type}. Evidence: ${parts.join(", ")}.`;
}

// Area in hectares. `ha` comes from the raster's pixel size (10 m Sentinel-2, 30 m Landsat); a candidate stored before it was
// recorded has none and was always 10 m, i.e. 100 m² a pixel. One decimal always, so 247.3 ha reads "247.3 ha".
export const hectares = (px, ha = null) =>
  `${(ha ?? px / 100).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} ha`;
