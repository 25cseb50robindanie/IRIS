"""IRIS change-aware search — which detected changes match what the analyst asked for.

Semantic search answers "which tiles look like this?" at one date. This answers "which *changes* look like this?":
a change candidate is a match when (1) it overlaps one of the semantic result tiles, and (2) the AFTER-date crop
covering it also matches the text query. Two subsystems, two questions: match strength and change status are
reported separately and only combined for ranking (architecture.md, "Semantic vs. change discrepancy policy").
"""

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

from api.provenance import analysed_area
from catalog import changes as store_changes
from change_detection import params
from change_detection.direction import direction_hints
from catalog.database import find_overlapping_scenes, list_scene_tiles
from embedding.index import FaissVectorStore

logger = logging.getLogger("iris.api.change_search")

# An after-crop "matches" when it scores in the top quarter of that scene's tiles for the query. Absolute CLIP cosines
# sit near 0.25 whatever the query, so a fixed similarity cut-off would be arbitrary; the rank within the scene is not.
SEMANTIC_MATCH_PERCENTILE = 0.75
CHANGE_RESULTS_LIMIT = 50
SEMANTIC_WEIGHT = 0.5
CONFIDENCE_WEIGHT = 0.5

FOUND = "found"
NO_MATCH = "no_match"
NO_CHANGE = "no_change_detected"
NO_COMPARISON = "no_comparison_available"


def _intersects(a: List[float], b: List[float]) -> bool:
    """WGS84 [min_lon, min_lat, max_lon, max_lat] boxes overlap."""
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _job_coverage(job: Dict[str, Any]) -> Optional[float]:
    d = job.get("details") or {}
    value = d.get("mutual_coverage_after_alignment", d.get("mutual_coverage"))
    return float(value) if value is not None else None


def analyse_changes(
    conn,
    store: FaissVectorStore,
    query_vector: np.ndarray,
    scene_id: Optional[str],
    semantic_tiles: List[Dict[str, Any]],
    query_text: str = "",
) -> Dict[str, Any]:
    """Return {"change_status", "change_results", "change_meta"} for one query.

    `semantic_tiles` are the semantic result tiles (each with WGS84 "bounds"). Runs inside the caller's hold on
    the vector store lock, so the ids read here cannot be renumbered by a scene deletion mid-request.
    """
    # Scenes that share this one's footprint (and sensor): the changes between any two of them are relevant here
    if scene_id:
        scope: Set[str] = {scene_id} | {o["scene_id"] for o in find_overlapping_scenes(conn, scene_id)}
    else:
        scope = {r[0] for r in conn.execute("SELECT scene_id FROM scenes").fetchall()}
    dates = dict(conn.execute("SELECT scene_id, acquisition_date FROM scenes").fetchall())

    jobs = [j for j in store_changes.list_jobs(conn) if j["scene_a_id"] in scope and j["scene_b_id"] in scope]
    completed = [j for j in jobs if j["status"] == store_changes.JOB_COMPLETED]

    meta: Dict[str, Any] = {
        "dates_compared": None,
        "valid_coverage": None,
        "total_candidates": 0,
        "matching_candidates": 0,
        "pairs": [],
        "note": None,
        "match_percentile": SEMANTIC_MATCH_PERCENTILE,
        "thresholds": {
            "minimum_mapping_unit_px": params.MIN_BLOB_PIXELS,
            "minimum_confidence": params.MIN_STORED_CONFIDENCE,
        },
    }

    if not completed:
        if jobs:
            worst = jobs[0]
            meta["note"] = (
                f"Change analysis for {dates.get(worst['scene_a_id'], '?')} → {dates.get(worst['scene_b_id'], '?')} "
                f"did not complete ({worst['status'].replace('_', ' ')})."
            )
        return {"change_status": NO_COMPARISON, "change_results": [], "change_meta": meta}

    candidates = store_changes.query_candidates(conn, job_ids=[j["job_id"] for j in completed])
    per_job: Dict[int, int] = {}
    for c in candidates:
        per_job[c["job_id"]] = per_job.get(c["job_id"], 0) + 1

    coverages = [c for c in (_job_coverage(j) for j in completed) if c is not None]
    starts = [dates.get(j["scene_a_id"]) for j in completed if dates.get(j["scene_a_id"])]
    ends = [dates.get(j["scene_b_id"]) for j in completed if dates.get(j["scene_b_id"])]
    meta.update(
        {
            "dates_compared": f"{min(starts)} → {max(ends)}" if starts and ends else None,
            # the least-observed pair bounds what can be claimed, so a negative is only as strong as its worst pair
            "valid_coverage": round(min(coverages), 4) if coverages else None,
            "total_candidates": len(candidates),
            "pairs": [
                {
                    "job_id": j["job_id"],
                    "scene_a_id": j["scene_a_id"],
                    "scene_b_id": j["scene_b_id"],
                    "dates": f"{dates.get(j['scene_a_id'], '?')} → {dates.get(j['scene_b_id'], '?')}",
                    "valid_coverage": _job_coverage(j),
                    "candidates": per_job.get(j["job_id"], 0),
                }
                for j in completed
            ],
        }
    )

    area = analysed_area(conn, completed)
    meta["area_bounds"], meta["mgrs"] = area["bounds"], area["mgrs"]
    meta["dates_analysed"] = sorted(
        {d for j in completed for d in (dates.get(j["scene_a_id"]), dates.get(j["scene_b_id"])) if d}
    )

    if not candidates:
        return {"change_status": NO_CHANGE, "change_results": [], "change_meta": meta}

    # (1) candidates that overlap a semantic result tile
    overlapping = [c for c in candidates if any(_intersects([c["min_lon"], c["min_lat"], c["max_lon"], c["max_lat"]], t["bounds"]) for t in semantic_tiles)]

    # (2) score the query against the AFTER-date crops covering each of them
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    scenes_cache: Dict[str, Dict[str, Any]] = {}

    def scene_scores(after_id: str) -> Dict[str, Any]:
        if after_id not in scenes_cache:
            tiles = list_scene_tiles(conn, after_id)
            if tiles:
                sims = store.vectors_for([t["faiss_id"] for t in tiles]) @ q
                boxes = np.array([t["bounds"] for t in tiles], dtype=np.float64)
                scenes_cache[after_id] = {"tiles": tiles, "sims": sims, "sorted": np.sort(sims), "boxes": boxes}
            else:
                scenes_cache[after_id] = {"tiles": [], "sims": np.empty(0), "sorted": np.empty(0), "boxes": np.empty((0, 4))}
        return scenes_cache[after_id]

    # Words like "new" or "cleared" say which direction the analyst is after. Simple keyword matching, not NLP.
    hints = direction_hints(query_text)
    meta["direction_hints"] = hints

    results: List[Dict[str, Any]] = []
    unscored = 0
    for c in overlapping:
        sc = scene_scores(c["scene_b_id"])
        if not sc["tiles"]:
            unscored += 1
            continue
        b = sc["boxes"]
        hit = (b[:, 0] <= c["max_lon"]) & (b[:, 2] >= c["min_lon"]) & (b[:, 1] <= c["max_lat"]) & (b[:, 3] >= c["min_lat"])
        if not hit.any():
            unscored += 1  # its after-crop was dropped at tiling (mostly no-data), so there is nothing to score
            continue
        best = int(np.flatnonzero(hit)[np.argmax(sc["sims"][hit])])
        similarity = float(sc["sims"][best])
        percentile = float(np.searchsorted(sc["sorted"], similarity, side="right")) / len(sc["sorted"])
        boost = params.DIRECTION_SEARCH_BOOST if c["direction"] in hints else 0.0
        results.append(
            {
                "candidate_id": c["candidate_id"],
                "job_id": c["job_id"],
                "scene_a_id": c["scene_a_id"],
                "scene_b_id": c["scene_b_id"],
                "scene_a_date": c["scene_a_date"],
                "scene_b_date": c["scene_b_date"],
                "bounds": [c["min_lon"], c["min_lat"], c["max_lon"], c["max_lat"]],
                "change_type": c["change_type"],
                "direction": c["direction"],
                "confidence": c["confidence"],
                "mean_dndvi": c["mean_dndvi"],
                "area_px": c["area_px"],
                "area_ha": c["area_ha"],
                "sub_blobs": c["sub_blobs"],
                "mgrs": c["mgrs_ref"],
                "seasonality_status": c["seasonality_status"],
                "review_status": c["review_status"],
                "semantic_match_score": round(percentile, 4),
                "semantic_similarity": round(similarity, 4),
                "matches_query": percentile >= SEMANTIC_MATCH_PERCENTILE,
                "direction_boost": boost,
                "evidence": c.get("direction_evidence"),
                # A direction hint in the query adds to the score, so it can lift a candidate above 1.0
                "combined_score": round(SEMANTIC_WEIGHT * percentile + CONFIDENCE_WEIGHT * c["confidence"] + boost, 4),
                "after_tile_id": sc["tiles"][best]["tile_id"],
            }
        )

    matching = sorted((r for r in results if r["matches_query"]), key=lambda r: -r["combined_score"])
    meta["matching_candidates"] = len(matching)
    meta["unscored_candidates"] = unscored
    if not matching:
        return {"change_status": NO_MATCH, "change_results": [], "change_meta": meta}
    return {"change_status": FOUND, "change_results": matching[:CHANGE_RESULTS_LIMIT], "change_meta": meta}
