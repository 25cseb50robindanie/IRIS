"""IRIS change detection — Phase 5: per-candidate scoring.

    confidence = 0.35 * alignment_quality + 0.35 * cluster_distance_percentile
               + 0.15 * terrain_flatness  + 0.15 * valid_coverage

All four terms are in [0, 1]. Weights are starting values pending calibration against OSCD (see params.py).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as window_bounds

from change_detection import params
from change_detection.detection import Detection
from change_detection.rasters import strips

logger = logging.getLogger("iris.change.scoring")


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def score_candidates(
    det: Detection,
    mutual: np.ndarray,
    alignment_quality: float,
    transform: Affine,
    crs: Optional[str],
    scene_a_id: str,
    scene_b_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Score every surviving blob. Returns (candidate rows ready for the catalog, trace figures)."""
    trace: Dict[str, Any] = {"terrain": "placeholder: flat (no DEM loaded)"}
    if det.labels is None or det.keep is None or not det.keep.any():
        trace.update(
            {
                "candidates": 0,
                "candidates_found": 0,
                "dropped_insufficient_evidence": 0,
                "dropped_low_confidence": 0,
                "dropped_over_cap": 0,
            }
        )
        return [], trace

    labels, keep = det.labels, det.keep
    n = len(keep) - 1
    height = labels.shape[0]

    # Per-blob sums of the cluster distance (distance from the no-change centroid) and of signed dNDVI,
    # accumulated strip by strip so no full-size float array is ever built.
    sum_d = np.zeros(n + 1)
    sum_dn = np.zeros(n + 1)
    cnt_dn = np.zeros(n + 1)
    for row0, rows in strips(height):
        lab = labels[row0 : row0 + rows]
        sel = keep[lab]  # pixels that belong to a surviving blob
        if not sel.any():
            continue
        dist = np.asarray(det.abs_diff[row0 : row0 + rows])[sel].astype(np.float64) - det.c_low
        lab_sel = lab[sel]
        sum_d += np.bincount(lab_sel, weights=dist, minlength=n + 1)
        if det.dndvi is not None:
            dn = np.asarray(det.dndvi[row0 : row0 + rows])[sel]
            ok = np.isfinite(dn)
            sum_dn += np.bincount(lab_sel[ok], weights=dn[ok].astype(np.float64), minlength=n + 1)
            cnt_dn += np.bincount(lab_sel[ok], minlength=n + 1)

    # NormClusterDist: each blob's mean distance from the no-change centroid, ranked against the OTHER BLOBS of this
    # pair (0.0 = weakest change, 1.0 = strongest). Ranking against pixels instead lets a few large blobs own most of
    # the pixels, which pins every blob's mean near the top. A tied or lone blob gets the mid-rank, never 1.0.
    kept_labels = np.flatnonzero(keep)
    blob_means = sum_d[kept_labels] / det.areas[kept_labels]
    sorted_means = np.sort(blob_means)

    rows_out: List[Dict[str, Any]] = []
    dropped_coverage = 0
    dropped_confidence = 0
    for label in kept_labels:
        sl = det.objects[label - 1]
        area = int(det.areas[label])
        mean_d = sum_d[label] / area
        below = np.searchsorted(sorted_means, mean_d, side="left")
        at_or_below = np.searchsorted(sorted_means, mean_d, side="right")
        cluster_pct = _clamp01(((below + at_or_below) / 2.0) / len(sorted_means))
        coverage = float(mutual[sl].mean())
        if coverage < params.MIN_CANDIDATE_VALID_COVERAGE:
            dropped_coverage += 1  # "Insufficient Evidence": a strong-looking score on a sliver of valid data
            continue

        mean_dndvi = float(sum_dn[label] / cnt_dn[label]) if cnt_dn[label] > 0 else None
        w = params.CONFIDENCE_WEIGHTS
        confidence = _clamp01(
            w["alignment_quality"] * _clamp01(alignment_quality)
            + w["cluster_distance"] * cluster_pct
            + w["terrain_flatness"] * params.TERRAIN_FLATNESS_PLACEHOLDER
            + w["valid_coverage"] * coverage
        )

        if confidence < params.MIN_STORED_CONFIDENCE:
            dropped_confidence += 1
            continue

        row0, row1, col0, col1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        left, bottom, right, top = window_bounds(Window(col0, row0, col1 - col0, row1 - row0), transform)
        native = [float(left), float(bottom), float(right), float(top)]
        wgs84 = native
        if crs:
            try:
                wgs84 = [float(v) for v in transform_bounds(crs, "EPSG:4326", *native)]
            except Exception:
                logger.warning("Could not reproject candidate bounds to WGS84; storing native bounds")

        rows_out.append(
            {
                "scene_a_id": scene_a_id,
                "scene_b_id": scene_b_id,
                "min_x": native[0],
                "min_y": native[1],
                "max_x": native[2],
                "max_y": native[3],
                "min_lon": wgs84[0],
                "min_lat": wgs84[1],
                "max_lon": wgs84[2],
                "max_lat": wgs84[3],
                "change_type": "unclassified",  # named by Phase 4b (direction.classify_candidates)
                "direction": None,
                "direction_evidence": None,
                "blob_label": int(label),  # lets Phase 4b find this blob's pixels; removed before storage
                "confidence": confidence,
                "norm_rmse": None,
                "norm_cluster_dist": cluster_pct,
                "terrain_flatness": params.TERRAIN_FLATNESS_PLACEHOLDER,
                "valid_coverage": coverage,
                "alignment_quality": _clamp01(alignment_quality),
                "area_px": area,
                "mean_dndvi": mean_dndvi,
            }
        )

    rows_out.sort(key=lambda r: (-r["confidence"], -r["area_px"]))
    found = len(rows_out) + dropped_confidence  # blobs that passed the MMU and coverage floor
    over_cap = max(0, len(rows_out) - params.MAX_STORED_CANDIDATES)
    rows_out = rows_out[: params.MAX_STORED_CANDIDATES]
    trace.update(
        {
            "candidates": len(rows_out),
            "candidates_found": found,
            "dropped_insufficient_evidence": dropped_coverage,
            "dropped_low_confidence": dropped_confidence,
            "dropped_over_cap": over_cap,
            "confidence_weights": params.CONFIDENCE_WEIGHTS,
            "min_stored_confidence": params.MIN_STORED_CONFIDENCE,
        }
    )
    logger.info(
        "Scored %d candidates (%d below confidence %.2f, %d on thin coverage, %d over cap)",
        len(rows_out),
        dropped_confidence,
        params.MIN_STORED_CONFIDENCE,
        dropped_coverage,
        over_cap,
    )
    return rows_out, trace
