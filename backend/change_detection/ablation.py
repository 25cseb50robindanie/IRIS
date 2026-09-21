"""IRIS change detection — ablation: the same pair with every suppression stage switched off.

The full pipeline earns its keep by what it removes. This runs the shortest route from two images to a list of changes,
so the analyst can see what the suppression stack is suppressing:

    raw NIR subtraction -> absolute value -> K-Means K=2 -> morphological cleanup -> connected components -> MMU filter

What is deliberately NOT done here: no SCL / cloud masking (every pixel is valid), no PIF radiometric normalisation
(raw values), no alignment, no merging of nearby blobs, no scoring, no direction, no seasonal filter. The difference
image, K-means, morphology and minimum mapping unit are the same code paths and parameters as the full pipeline, so the
two lists differ only by the suppression. The same "change centroid is below the minimum magnitude" sanity gate is kept
(K-means always splits two clusters, even on identical images).

The result is stored in `ablation_candidates`: the largest MAX_ABLATION_STORED raw detections with their outlines, and the
true total in the job's details, so a count shown to the analyst is never a capped one.
"""

import gc
import json
import logging
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as window_bounds
from scipy import ndimage
from sklearn.cluster import KMeans

from catalog import ablation as store
from catalog.database import get_scene, init_connection, init_schema
from change_detection import params
from change_detection.alignment import regrid_to_reference
from change_detection.grouping import attach_geometry
from change_detection.scoring import pixel_area_ha
from change_detection.rasters import SceneReader, channel, dn_scale, grid_signature, grids_match, strips
from mgrs_ref import to_mgrs

logger = logging.getLogger("iris.change.ablation")

STAGING_DIR = Path("data/staging")
# An outline is traced in a window around the blob; beyond this many pixels the box stands in for it (memory)
MAX_TRACE_WINDOW_PX = 4_000_000
# Traced outlines follow every pixel step, and the biggest raw blobs have tens of thousands of vertices and thousands of
# holes (the top 200 on the real pair were 10 MB). They are only drawn on a map, so their rings are simplified to this
# tolerance (2 px, 20 m at 10 m/px) and their holes are left out.
OUTLINE_TOLERANCE_DEG = 0.0002
OUTLINE_MAX_VERTICES = 1000  # a still bigger outline is simplified harder until it fits (a percolated blob spans kilometres)


def _simplify_ring(ring: List[List[float]], tolerance: float) -> List[List[float]]:
    """Douglas-Peucker on a closed ring; the ring is returned unchanged if it would collapse below a triangle."""
    pts = np.asarray(ring, dtype=np.float32).reshape(-1, 1, 2)
    kept = cv2.approxPolyDP(pts, tolerance, True).reshape(-1, 2)
    if len(kept) < 3:
        return ring
    out = [[float(x), float(y)] for x, y in kept]
    return out + [out[0]]


def simplify_outline(
    geometry_json: str, tolerance: float = OUTLINE_TOLERANCE_DEG, max_vertices: int = OUTLINE_MAX_VERTICES
) -> str:
    """A GeoJSON Polygon / MultiPolygon reduced to its simplified exterior rings: holes are dropped, exteriors are kept.

    The tolerance doubles until the outline has at most `max_vertices` vertices (or stops shrinking).
    """
    geometry = json.loads(geometry_json)
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    simplified = [[_simplify_ring(rings[0], tolerance)] for rings in polygons]
    for _ in range(12):
        if sum(len(poly[0]) for poly in simplified) <= max_vertices:
            break
        tolerance *= 2.0
        simplified = [[_simplify_ring(rings[0], tolerance)] for rings in polygons]
    geometry["coordinates"] = simplified[0] if geometry["type"] == "Polygon" else simplified
    return json.dumps(geometry, separators=(",", ":"))


def _analysis_path(scene: Dict[str, Any]) -> Path:
    path = Path(scene["analysis_cog_path"] or scene["cog_path"] or "")
    if not path.is_file():
        raise ValueError(f"Analysis raster for {scene['scene_id']} is missing on disk")
    return path


def _detect(path_a: Path, path_b: Path, work_dir: Path) -> Dict[str, Any]:
    """Raw difference -> K-means -> cleanup -> components. Returns the labelled blobs and figures for the trace."""
    with SceneReader(path_a) as reader_a, SceneReader(path_b) as reader_b:
        height, width = reader_a.height, reader_a.width
        scale = dn_scale(reader_a.dtype)
        abs_diff = np.lib.format.open_memmap(str(work_dir / "abs_diff.npy"), mode="w+", dtype=np.float32, shape=(height, width))

        rng = np.random.default_rng(params.KMEANS_SEED)
        rate = min(1.0, params.KMEANS_SAMPLE_MAX / max(height * width, 1))
        samples: List[np.ndarray] = []
        for row0, rows in strips(height):
            diff = np.abs(channel(reader_b.strip(row0, rows)) - channel(reader_a.strip(row0, rows)))  # raw: no masks, no PIF
            abs_diff[row0 : row0 + rows] = diff
            flat = diff.ravel()
            samples.append(flat[rng.random(flat.size) < rate] if rate < 1.0 else flat)
        sample = np.concatenate(samples)

    out: Dict[str, Any] = {"labels": None, "objects": [], "areas": np.zeros(1, dtype=np.int64), "keep": np.zeros(1, dtype=bool)}
    trace: Dict[str, Any] = {"kmeans_sample": int(sample.size)}
    out["trace"] = trace
    if sample.size < 2 or float(np.ptp(sample)) < 1e-9:
        trace["result"] = "the difference image has no variation"
        return out

    km = KMeans(n_clusters=2, n_init=10, random_state=params.KMEANS_SEED).fit(sample.reshape(-1, 1).astype(np.float64))
    centers = km.cluster_centers_.ravel()
    c_high, c_low = float(centers.max()), float(centers.min())
    threshold = (c_high + c_low) / 2.0
    trace.update({"kmeans_centroids": [round(c_low, 4), round(c_high, 4)], "kmeans_threshold": round(threshold, 4)})
    if c_high < params.MIN_CHANGE_MAGNITUDE * scale:
        trace["result"] = "change centroid below the minimum change magnitude"
        return out

    change = np.zeros((height, width), dtype=np.uint8)
    for row0, rows in strips(height):
        change[row0 : row0 + rows] = np.asarray(abs_diff[row0 : row0 + rows]) > threshold
    del abs_diff
    kernel = np.ones((params.MORPH_KERNEL_SIZE, params.MORPH_KERNEL_SIZE), dtype=np.uint8)
    change = cv2.morphologyEx(change, cv2.MORPH_OPEN, kernel)
    change = cv2.morphologyEx(change, cv2.MORPH_CLOSE, kernel)

    labels, n = ndimage.label(change)
    labels = labels.astype(np.int32, copy=False)
    del change
    areas = np.zeros(n + 1, dtype=np.int64)
    for row0, rows in strips(height):
        areas += np.bincount(labels[row0 : row0 + rows].ravel(), minlength=n + 1)
    keep = areas >= params.MIN_BLOB_PIXELS
    keep[0] = False
    trace.update({"components_total": int(n), "components_kept": int(keep.sum()), "min_blob_pixels": params.MIN_BLOB_PIXELS})
    out.update({"labels": labels, "objects": ndimage.find_objects(labels) if n else [], "areas": areas, "keep": keep})
    return out


def run_ablation(scene_a_id: str, scene_b_id: str) -> Dict[str, Any]:
    """Run the suppression-free pipeline for a pair whose full analysis has completed, and store the result.

    Returns {"job_id", "found", "stored"}. The pair may be given in either order (the older scene is A). Raises ValueError
    when the pair has no completed change-detection job; any other failure is recorded on the job as status "failed".
    """
    conn = init_connection()
    try:
        init_schema(conn)
        job = store.find_job(conn, scene_a_id, scene_b_id)
        if job is None or job["status"] != "completed":
            raise ValueError("Ablation needs a pair whose change detection has completed")
        scene_a, scene_b = get_scene(conn, job["scene_a_id"]), get_scene(conn, job["scene_b_id"])
        if scene_a is None or scene_b is None:
            raise ValueError("Both scenes must be in the catalog")
        store.mark_status(conn, job["job_id"], store.RUNNING)
    finally:
        conn.close()

    job_id = job["job_id"]
    started = time.monotonic()
    work_dir = STAGING_DIR / f"ablation_{job_id}"
    det: Dict[str, Any] = {}
    try:
        path_a, path_b = _analysis_path(scene_a), _analysis_path(scene_b)
        work_dir.mkdir(parents=True, exist_ok=True)
        if not grids_match(grid_signature(path_a), grid_signature(path_b)):
            # No suppression does not mean no geometry: the two rasters must at least share a pixel grid to be subtracted
            regridded = work_dir / "b_on_a_grid.tif"
            regrid_to_reference(path_b, path_a, regridded, Resampling.bilinear)
            path_b = regridded

        det = _detect(path_a, path_b, work_dir)
        with rasterio.open(str(path_a)) as src:
            transform, crs = src.transform, (src.crs.to_string() if src.crs else None)

        pixel_ha = pixel_area_ha(transform, crs)
        kept = np.flatnonzero(det["keep"])
        found = int(kept.size)
        order = kept[np.argsort(-det["areas"][kept], kind="stable")][: params.MAX_ABLATION_STORED]

        rows: List[Dict[str, Any]] = []
        for label in order:
            sl = det["objects"][label - 1]
            row0, row1, col0, col1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            left, bottom, right, top = window_bounds(Window(col0, row0, col1 - col0, row1 - row0), transform)
            native = [float(left), float(bottom), float(right), float(top)]
            wgs84 = native
            if crs:
                try:
                    wgs84 = [float(v) for v in transform_bounds(crs, "EPSG:4326", *native)]
                except Exception:
                    logger.warning("Could not reproject ablation bounds to WGS84; storing native bounds")
            lon, lat = (wgs84[0] + wgs84[2]) / 2.0, (wgs84[1] + wgs84[3]) / 2.0
            row: Dict[str, Any] = {
                "min_x": native[0], "min_y": native[1], "max_x": native[2], "max_y": native[3],
                "min_lon": wgs84[0], "min_lat": wgs84[1], "max_lon": wgs84[2], "max_lat": wgs84[3],
                "area_px": int(det["areas"][label]),
                "area_ha": None if pixel_ha is None else round(int(det["areas"][label]) * pixel_ha, 4),
                "centroid_lon": round(lon, 6), "centroid_lat": round(lat, 6), "mgrs_ref": to_mgrs(lat, lon),
                "geometry": None,
            }
            if (row1 - row0) * (col1 - col0) <= MAX_TRACE_WINDOW_PX:
                row["blob_labels"] = [int(label)]  # attach_geometry traces the outline and the pixel centroid
            rows.append(row)
        attach_geometry(SimpleNamespace(labels=det["labels"], objects=det["objects"]), rows, transform, crs)
        for row in rows:
            if row["geometry"]:
                row["geometry"] = simplify_outline(row["geometry"])

        summary = {
            "found": found,
            "stored": len(rows),
            "suppression": "off",
            "seconds": round(time.monotonic() - started, 2),
            **det["trace"],
        }
        conn = init_connection()
        try:
            store.save_run(conn, job_id, job["scene_a_id"], job["scene_b_id"], rows, summary)
        finally:
            conn.close()
        logger.info("Ablation %s -> %s: %d raw detections (%d stored)", job["scene_a_id"], job["scene_b_id"], found, len(rows))
        return {"job_id": job_id, "found": found, "stored": len(rows)}
    except Exception as e:
        logger.exception("Ablation for job %d failed", job_id)
        conn = init_connection()
        try:
            store.mark_status(conn, job_id, store.FAILED, error=f"{type(e).__name__}")
        finally:
            conn.close()
        raise
    finally:
        det = {}  # release the memory-mapped labels before deleting the staging files (Windows keeps them open)
        gc.collect()
        shutil.rmtree(work_dir, ignore_errors=True)
