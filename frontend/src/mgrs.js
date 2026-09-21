import { forward } from "mgrs";

// 10-digit grid reference (1 m). Same precision as the backend (backend/mgrs_ref.py), so a point reads the same everywhere.
export const MGRS_ACCURACY = 5;

const inRange = (lat, lon) => Number.isFinite(lat) && Number.isFinite(lon) && lat >= -80 && lat <= 84 && lon >= -180 && lon <= 180;

/** MGRS reference of a WGS84 point, or null when it cannot be expressed (polar regions, bad values). */
export function toMgrs(lat, lon) {
  if (!inRange(lat, lon)) return null;
  try {
    return forward([lon, lat], MGRS_ACCURACY);
  } catch (err) {
    return null;
  }
}

/** MGRS reference of the centre of a [min_lon, min_lat, max_lon, max_lat] box. */
export function boundsMgrs(b) {
  if (!b || b.length !== 4) return null;
  return toMgrs((b[1] + b[3]) / 2, (b[0] + b[2]) / 2);
}

/** "43RCL 03332 98814": the reference in its printed grouping (zone+band+square, easting, northing). */
export function spacedMgrs(ref) {
  const m = /^(\d{1,2}[C-X][A-Z]{2})(\d+)$/.exec(ref || "");
  if (!m) return ref || "";
  const digits = m[2];
  const half = digits.length / 2;
  return `${m[1]} ${digits.slice(0, half)} ${digits.slice(half)}`;
}

export const formatLatLon = (lat, lon) =>
  `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}`;
