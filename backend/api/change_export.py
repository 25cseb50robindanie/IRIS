"""IRIS export API — the change candidates as GeoJSON, each with the provenance needed to trust and reproduce it.

One feature per change candidate. The geometry is the outline of the changed pixels (WGS84), traced when the detection
was made; a candidate stored before outlines were kept falls back to its bounding box and says so in `geometry_source`.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response

from api.provenance import processing_block
from catalog import changes as store
from catalog.database import init_connection, init_schema

logger = logging.getLogger("iris.api.export")
router = APIRouter(prefix="/api/export", tags=["export"])

GEOJSON_CRS = {"type": "name", "properties": {"name": "EPSG:4326"}}


def _box_polygon(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> Dict[str, Any]:
    ring = [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]
    return {"type": "Polygon", "coordinates": [ring]}


def _latest_review(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    return reviews[-1] if reviews else {}


def _feature(cand: Dict[str, Any], geometry_json: Optional[str], details: Dict[str, Any], reviews: List[Dict[str, Any]], checksums: Dict[str, Optional[str]]) -> Dict[str, Any]:
    if geometry_json:
        geometry, source = json.loads(geometry_json), "changed_pixel_outline"
    else:
        geometry = _box_polygon(cand["min_lon"], cand["min_lat"], cand["max_lon"], cand["max_lat"])
        source = "bounding_box"
    review = _latest_review(reviews)
    return {
        "type": "Feature",
        "id": cand["candidate_id"],
        "geometry": geometry,
        "properties": {
            "candidate_id": cand["candidate_id"],
            "direction": cand["direction"],
            "change_type": cand["change_type"],
            "confidence": round(cand["confidence"], 4),
            "confidence_breakdown": {
                "alignment_quality": cand["alignment_quality"],
                "cluster_distance_percentile": cand["norm_cluster_dist"],
                "terrain_flatness": cand["terrain_flatness"],
                "valid_coverage": cand["valid_coverage"],
            },
            "mgrs": cand["mgrs_ref"],
            "scene_a_id": cand["scene_a_id"],
            "scene_b_id": cand["scene_b_id"],
            "date_a": cand["scene_a_date"],
            "date_b": cand["scene_b_date"],
            "sensor": cand["sensor"],
            "delta_ndvi": None if cand["mean_dndvi"] is None else round(cand["mean_dndvi"], 4),
            "analyst_decision": review.get("decision", "pending"),
            "analyst_id": review.get("analyst_id"),
            "reviewed_at": review.get("reviewed_at"),
            "processing": processing_block(details),
            # Beyond the core record: enough to check the number and to reproduce the input
            "area_px": cand["area_px"],
            "area_ha": None if cand["area_px"] is None else round(cand["area_px"] / 100.0, 2),  # 10 m pixels
            "sub_blobs": cand["sub_blobs"],
            "direction_evidence": cand["direction_evidence"],
            "geometry_source": source,
            "raw_checksum_a": checksums.get(cand["scene_a_id"]),
            "raw_checksum_b": checksums.get(cand["scene_b_id"]),
            "analyst_notes": review.get("notes"),
            "review_count": len(reviews),
        },
    }


@router.get("/changes")
def export_changes(
    job_id: Optional[int] = Query(default=None, description="Only this pair's candidates"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
) -> Response:
    """GeoJSON FeatureCollection of change candidates (all of them by default), as a download."""
    conn = init_connection()
    try:
        init_schema(conn)
        jobs = {j["job_id"]: j for j in store.list_jobs(conn)}
        cands = store.query_candidates(conn, job_ids=None if job_id is None else [job_id], min_confidence=min_confidence)
        geometry = dict(conn.execute("SELECT candidate_id, geometry FROM change_candidates").fetchall())
        checksums = dict(conn.execute("SELECT scene_id, raw_checksum FROM scenes").fetchall())
        reviews: Dict[int, List[Dict[str, Any]]] = {c["candidate_id"]: store.get_reviews(conn, c["candidate_id"]) for c in cands}
    finally:
        conn.close()

    features = [
        _feature(c, geometry.get(c["candidate_id"]), (jobs.get(c["job_id"]) or {}).get("details") or {}, reviews[c["candidate_id"]], checksums)
        for c in cands
    ]
    now = datetime.now(timezone.utc)
    body = {
        "type": "FeatureCollection",
        "name": "iris_change_candidates",
        "crs": GEOJSON_CRS,
        "generated_at": now.isoformat(timespec="seconds"),
        "feature_count": len(features),
        "features": features,
    }
    logger.info("Exported %d change candidates as GeoJSON", len(features))
    return Response(
        content=json.dumps(body),
        media_type="application/geo+json",
        headers={"Content-Disposition": f'attachment; filename="iris_changes_{now:%Y%m%d}.geojson"'},
    )
