/** Save a Blob as a file through the browser (no network: the data is already in memory). */
export function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const downloadJson = (filename, value, type = "application/json") =>
  downloadBlob(filename, new Blob([JSON.stringify(value, null, 2)], { type }));

const filenameFrom = (res, fallback) => /filename="?([^";]+)"?/.exec(res.headers.get("content-disposition") || "")?.[1] || fallback;

/** GET a backend document and save it; throws with the backend's message when it fails. */
export async function downloadFromApi(url, fallbackName) {
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Download failed (HTTP ${res.status})`);
  }
  downloadBlob(filenameFrom(res, fallbackName), await res.blob());
}
