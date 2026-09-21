"""IRIS change detection — seasonal persistence filter.

A vegetation drop is not always an event: fields are harvested and dry seasons come round every year. For each
vegetation-related candidate whose |mean dNDVI| exceeds NDVI_DROP_SIG, the same ground is looked up in earlier years of
the same season (same sensor, within SEASONAL_WINDOW_DAYS of the after-scene's date in its year):

  * fewer than SEASONAL_MIN_PRIORS clear prior observations  -> "unverified"  (confidence x 0.85: the check could not be made)
  * NDVI history available, current NDVI within SEASONAL_SIGMAS std devs of it -> "seasonal"   (confidence x 0.5)
  * the current drop exceeds that variation                                    -> "anomalous"  (confidence unchanged)

"Current NDVI" is the mean NDVI of the changed pixels in the after scene, and the "drop" is how far it sits below the
mean NDVI of the same pixels in the prior same-season scenes. NDVI is read from each scene's analysis COG at the
candidate's outline, so the check compares like with like. A prior counts as clear when at least SEASONAL_MIN_VALID of
those pixels are valid (SCL classes and data mask). Candidates the check does not apply to keep `seasonality_status` NULL.
"""

import json
import logging
import math
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds

from catalog import changes as store
from catalog.database import get_scene, init_connection, init_schema
from change_detection import params
from change_detection.detection import _parse_offsets
from change_detection.direction import to_reflectance
from change_detection.rasters import dn_scale

logger = logging.getLogger("iris.change.seasonal")

SEASONAL, ANOMALOUS, UNVERIFIED = "seasonal", "anomalous", "unverified"
STATUSES = (SEASONAL, ANOMALOUS, UNVERIFIED)


def _parse_date(value: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(value or "")
    except ValueError:
        return None


def same_season(prior: date, current: date) -> bool:
    """True when `prior` is from an earlier year and within SEASONAL_WINDOW_DAYS of `current`'s day of the year."""
    if prior.year >= current.year:
        return False
    try:
        anchor = prior.replace(year=current.year)
    except ValueError:  # 29 February in a year that has none
        anchor = prior.replace(year=current.year, day=28)
    gap = abs((anchor - current).days)
    return min(gap, 365 - gap) <= params.SEASONAL_WINDOW_DAYS


def eligible(candidate: Dict[str, Any]) -> bool:
    """Vegetation-related change with a strong enough NDVI move for the check to apply."""
    dndvi = candidate.get("mean_dndvi")
    return (
        candidate.get("change_type") in params.SEASONAL_TYPES
        and dndvi is not None
        and abs(dndvi) > params.NDVI_DROP_SIG
    )


def _prior_pool(conn: Any, scene_b: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Same-sensor scenes from earlier years of the same season, with their WGS84 footprint."""
    current = _parse_date(scene_b["acquisition_date"])
    if current is None:
        return []
    rows = conn.execute(
        """
        SELECT s.scene_id, s.acquisition_date, s.analysis_cog_path, s.scl_path,
               f.min_lon, f.min_lat, f.max_lon, f.max_lat
        FROM scenes s JOIN scene_footprints f ON f.id = s.rowid
        WHERE s.sensor = ? AND s.scene_id != ?
        """,
        (scene_b["sensor"], scene_b["scene_id"]),
    ).fetchall()
    keys = ["scene_id", "acquisition_date", "analysis_cog_path", "scl_path", "min_lon", "min_lat", "max_lon", "max_lat"]
    pool = [dict(zip(keys, r)) for r in rows]
    return [p for p in pool if (d := _parse_date(p["acquisition_date"])) is not None and same_season(d, current)]


class _Sampler:
    """Mean NDVI of a candidate's pixels in a scene's analysis COG. Each dataset is opened once and closed by close()."""

    def __init__(self) -> None:
        self._open: Dict[str, Any] = {}
        self._lut = np.zeros(256, dtype=bool)
        self._lut[list(params.SCL_VALID_CLASSES)] = True

    def _dataset(self, path: str) -> Any:
        if path not in self._open:
            self._open[path] = rasterio.open(path)
        return self._open[path]

    def close(self) -> None:
        for ds in self._open.values():
            ds.close()
        self._open.clear()

    def mean_ndvi(
        self, scene: Dict[str, Any], geometry: Optional[Dict[str, Any]], bbox: Sequence[float]
    ) -> Optional[float]:
        """Mean NDVI over the candidate's pixels, or None when the scene was not clear there (or cannot be read)."""
        if not scene.get("analysis_cog_path"):
            return None
        try:
            return self._mean_ndvi(scene, geometry, bbox)
        except Exception as e:  # one unreadable prior must not abort the whole job
            logger.warning("Could not sample NDVI from %s: %s", scene.get("scene_id"), type(e).__name__)
            return None

    def _mean_ndvi(self, scene: Dict[str, Any], geometry: Optional[Dict[str, Any]], bbox: Sequence[float]) -> Optional[float]:
        src = self._dataset(scene["analysis_cog_path"])
        if src.count < 4 or src.crs is None:
            return None
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        raw = from_bounds(left, bottom, right, top, transform=src.transform)
        col0, row0 = max(0, math.floor(raw.col_off)), max(0, math.floor(raw.row_off))
        col1 = min(src.width, math.ceil(raw.col_off + raw.width))
        row1 = min(src.height, math.ceil(raw.row_off + raw.height))
        if col1 <= col0 or row1 <= row0:
            return None  # the candidate lies outside this scene
        window = Window(col0, row0, col1 - col0, row1 - row0)

        step = max(1, math.ceil(max(window.width, window.height) / params.SEASONAL_MAX_SAMPLE_PX))
        out_h, out_w = max(1, math.ceil(window.height / step)), max(1, math.ceil(window.width / step))
        strip = src.read(window=window, out_shape=(src.count, out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
        valid = (src.read_masks(window=window, out_shape=(src.count, out_h, out_w)) > 0).all(axis=0)
        if scene.get("scl_path"):
            scl = self._dataset(scene["scl_path"]).read(1, window=window, out_shape=(out_h, out_w), resampling=Resampling.nearest)
            valid &= self._lut[scl]

        inside = np.ones((out_h, out_w), dtype=bool)
        if geometry:
            transform = src.window_transform(window) * Affine.scale(window.width / out_w, window.height / out_h)
            shape = transform_geom("EPSG:4326", src.crs, geometry)
            traced = features.geometry_mask([shape], out_shape=(out_h, out_w), transform=transform, invert=True)
            if traced.any():
                inside = traced
        if valid[inside].mean() < params.SEASONAL_MIN_VALID:
            return None

        refl = to_reflectance(strip, _parse_offsets(src.tags(), src.count), dn_scale(src.dtypes[0]))
        red, nir = refl[2], refl[3]
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = np.where(nir + red > 1e-6, (nir - red) / (nir + red), np.nan)
        use = inside & valid & np.isfinite(ndvi)
        return float(ndvi[use].mean()) if use.any() else None


def _mark(candidate: Dict[str, Any], status: str, factor: float, evidence: Dict[str, Any]) -> None:
    evidence["confidence_before"] = round(float(candidate["confidence"]), 4)
    evidence["confidence_factor"] = factor  # stored confidence = the four weighted terms x this factor
    candidate["seasonality_status"] = status
    candidate["confidence"] = min(1.0, float(candidate["confidence"]) * factor)
    ev = candidate.get("direction_evidence")
    if isinstance(ev, dict):  # a candidate with no evidence dict is not eligible in practice (it is unclassified)
        ev["seasonality"] = evidence


def apply_seasonality(candidates: List[Dict[str, Any]], scene_b: Dict[str, Any]) -> Dict[str, Any]:
    """Run the persistence filter over `candidates` (rows as the pipeline builds them), in place. Returns the trace.

    `scene_b` is the after scene as the catalog row: its date and sensor pick the priors, its raster gives the current NDVI.
    """
    trace: Dict[str, Any] = {"eligible": 0, "seasonal": 0, "anomalous": 0, "unverified": 0, "prior_scenes": 0}
    todo = [c for c in candidates if eligible(c) and not c.get("seasonality_status")]
    trace["eligible"] = len(todo)
    if not todo:
        return trace

    conn = init_connection()
    try:
        pool = _prior_pool(conn, scene_b)
    finally:
        conn.close()
    trace["prior_scenes"] = len(pool)

    sampler = _Sampler()
    try:
        for c in todo:
            lon = c["centroid_lon"] if c.get("centroid_lon") is not None else (c["min_lon"] + c["max_lon"]) / 2.0
            lat = c["centroid_lat"] if c.get("centroid_lat") is not None else (c["min_lat"] + c["max_lat"]) / 2.0
            here = [p for p in pool if p["min_lon"] <= lon <= p["max_lon"] and p["min_lat"] <= lat <= p["max_lat"]]
            evidence: Dict[str, Any] = {"priors_covering": len(here), "priors_clear": 0}

            history: List[float] = []
            current: Optional[float] = None
            if len(here) >= params.SEASONAL_MIN_PRIORS:  # otherwise no amount of reading could reach the minimum
                geometry = json.loads(c["geometry"]) if c.get("geometry") else None
                bbox = [c["min_lon"], c["min_lat"], c["max_lon"], c["max_lat"]]
                history = [v for v in (sampler.mean_ndvi(p, geometry, bbox) for p in here) if v is not None]
                evidence["priors_clear"] = len(history)
                if len(history) >= params.SEASONAL_MIN_PRIORS:
                    current = sampler.mean_ndvi(scene_b, geometry, bbox)

            if current is None:
                _mark(c, UNVERIFIED, params.UNVERIFIED_CONFIDENCE_FACTOR, evidence)
                trace["unverified"] += 1
                continue

            mean = float(np.mean(history))
            std = max(float(np.std(history, ddof=1)), params.SEASONAL_MIN_STD)
            drop = mean - current  # how far below the usual NDVI for this season the after scene sits
            evidence.update(
                {
                    "prior_mean_ndvi": round(mean, 3),
                    "prior_std_ndvi": round(std, 3),
                    "current_ndvi": round(current, 3),
                    "drop_in_std": round(drop / std, 2),
                }
            )
            if drop <= params.SEASONAL_SIGMAS * std:
                _mark(c, SEASONAL, params.SEASONAL_CONFIDENCE_FACTOR, evidence)
                trace["seasonal"] += 1
            else:
                _mark(c, ANOMALOUS, 1.0, evidence)
                trace["anomalous"] += 1
    finally:
        sampler.close()

    candidates.sort(key=lambda r: (-r["confidence"], -r["area_px"]))
    logger.info(
        "Seasonal filter: %d checked (%d seasonal, %d anomalous, %d unverified) against %d prior scenes",
        trace["eligible"], trace["seasonal"], trace["anomalous"], trace["unverified"], trace["prior_scenes"],
    )
    return trace


def annotate_job(job_id: int) -> Dict[str, Any]:
    """Apply the filter to the stored candidates of a finished job that have not been checked yet.

    For catalogs analysed before the filter existed: it avoids re-running the whole pair. Candidates that already carry
    a status are left alone, so calling it twice never multiplies a confidence twice.
    """
    conn = init_connection()
    try:
        init_schema(conn)
        job = next((j for j in store.list_jobs(conn) if j["job_id"] == job_id), None)
        if job is None:
            raise ValueError(f"Unknown job: {job_id}")
        scene_b = get_scene(conn, job["scene_b_id"])
        if scene_b is None:
            raise ValueError(f"Scene {job['scene_b_id']} is no longer in the catalog")
        rows = conn.execute(
            "SELECT candidate_id, change_type, mean_dndvi, confidence, area_px, min_lon, min_lat, max_lon, max_lat, "
            "centroid_lon, centroid_lat, geometry, direction_evidence FROM change_candidates "
            "WHERE job_id = ? AND seasonality_status IS NULL",
            (job_id,),
        ).fetchall()
        keys = ["candidate_id", "change_type", "mean_dndvi", "confidence", "area_px", "min_lon", "min_lat", "max_lon",
                "max_lat", "centroid_lon", "centroid_lat", "geometry", "direction_evidence"]
        cands = [dict(zip(keys, r)) for r in rows]
        for c in cands:
            c["direction_evidence"] = json.loads(c["direction_evidence"]) if c["direction_evidence"] else None
    finally:
        conn.close()

    trace = apply_seasonality(cands, scene_b)

    conn = init_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for c in cands:
                if c.get("seasonality_status"):
                    conn.execute(
                        "UPDATE change_candidates SET seasonality_status = ?, confidence = ?, direction_evidence = ? "
                        "WHERE candidate_id = ?",
                        (
                            c["seasonality_status"],
                            c["confidence"],
                            json.dumps(c["direction_evidence"]) if c["direction_evidence"] else None,
                            c["candidate_id"],
                        ),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()
    return trace
