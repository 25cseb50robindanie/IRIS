"""IRIS Find Similar — image-to-image discovery (§2.2.4): one find leads to many.

The seed is a tile (by tile id or FAISS id), or a change candidate (its after-date tile is used). The seed's own
embedding, taken from the FAISS index, is the query, so no model runs and the answer is the tiles that look most like it,
in every scene of the archive, the seed itself excluded.

A change candidate also gets `similar_changes`: the OTHER detected changes whose after-date imagery looks like its own.
That is the tiles' similarity again, read at the changes: each change is scored by its best-matching after-date tile.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator

import instrumentation
from api.search import MIN_FETCH, OVERFETCH_FACTOR, SearchResultItem, _semantic_item
from catalog import changes as store_changes
from catalog.database import get_tiles_by_faiss_ids, init_connection, init_schema, list_scene_tiles
from embedding.index import FaissVectorStore, get_vector_store

logger = logging.getLogger("iris.api.similar")
router = APIRouter(prefix="/api", tags=["search"])

TILE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,300}$")  # ASCII on purpose: pydantic's regex engine caps the size of \w{1,300}
DEFAULT_SIMILAR = 10


class SimilarRequest(BaseModel):
    """Exactly one seed: a tile (by id or FAISS id), or a change candidate (its after-date tile is used)."""
    tile_id: Optional[str] = Field(default=None, pattern=TILE_ID_RE.pattern)
    faiss_id: Optional[int] = Field(default=None, ge=0)
    candidate_id: Optional[int] = Field(default=None, ge=1)
    top_k: int = Field(default=DEFAULT_SIMILAR, ge=1, le=100)

    @model_validator(mode="after")
    def _one_seed(self) -> "SimilarRequest":
        if sum(v is not None for v in (self.tile_id, self.faiss_id, self.candidate_id)) != 1:
            raise ValueError("Give exactly one of tile_id, faiss_id or candidate_id")
        return self


class SimilarSeed(BaseModel):
    tile_id: str
    scene_id: str
    bounds: List[float]
    mgrs: Optional[str] = None
    candidate_id: Optional[int] = None  # set when the seed was chosen through a change candidate


class SimilarChange(BaseModel):
    """A detected change whose after-date imagery resembles the seed change's."""
    candidate_id: int
    job_id: int
    scene_a_id: str
    scene_b_id: str
    scene_a_date: Optional[str] = None
    scene_b_date: Optional[str] = None
    bounds: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    change_type: Optional[str] = None
    direction: Optional[str] = None
    confidence: float
    area_px: Optional[int] = None
    area_ha: Optional[float] = None
    mgrs: Optional[str] = None
    review_status: str
    seasonality_status: Optional[str] = None
    similarity: float  # cosine similarity of its best-matching after-date tile to the seed
    after_tile_id: str


class SimilarResponse(BaseModel):
    similar_to: SimilarSeed
    label: str  # "Similar to <tile_id>", or "Similar to change #<id>" for a change seed
    semantic_results: List[SearchResultItem]
    similar_changes: List[SimilarChange] = []  # only for a change seed: other changes that look alike


def _candidate_seed_tile(conn: Any, candidate_id: int) -> str:
    """The after-date (scene B) tile that covers the candidate's centre, else the one overlapping it most."""
    cand = store_changes.get_candidate(conn, candidate_id)
    if cand is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown change candidate: {candidate_id}")
    lon = cand["centroid_lon"] if cand["centroid_lon"] is not None else (cand["min_lon"] + cand["max_lon"]) / 2.0
    lat = cand["centroid_lat"] if cand["centroid_lat"] is not None else (cand["min_lat"] + cand["max_lat"]) / 2.0

    best_id, best_key = None, None
    for t in list_scene_tiles(conn, cand["scene_b_id"]):
        b = t["bounds"]
        ix = max(0.0, min(b[2], cand["max_lon"]) - max(b[0], cand["min_lon"]))
        iy = max(0.0, min(b[3], cand["max_lat"]) - max(b[1], cand["min_lat"]))
        contains = b[0] <= lon <= b[2] and b[1] <= lat <= b[3]
        key = (contains, ix * iy)
        if (ix * iy > 0 or contains) and (best_key is None or key > best_key):
            best_id, best_key = t["tile_id"], key
    if best_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No embedded tile covers this change (its area was dropped at tiling), so there is nothing to compare with.",
        )
    return best_id


def _similar_changes(
    conn: Any, store: FaissVectorStore, seed_vector: np.ndarray, seed_candidate_id: int, seed_tile_id: str, top_k: int
) -> List[SimilarChange]:
    """Other detected changes ranked by how much their after-date imagery looks like the seed's.

    A change is scored by its best-matching after-date tile (the tiles that overlap it). Changes that sit in the seed's own
    tile are left out: they match trivially, being the same picture. Caller holds the vector store lock.
    """
    completed = [j["job_id"] for j in store_changes.list_jobs(conn) if j["status"] == store_changes.JOB_COMPLETED]
    others = [c for c in store_changes.query_candidates(conn, job_ids=completed) if c["candidate_id"] != seed_candidate_id]

    per_scene: Dict[str, Dict[str, Any]] = {}
    scored: List[SimilarChange] = []
    for c in others:
        if c["scene_b_id"] not in per_scene:
            tiles = list_scene_tiles(conn, c["scene_b_id"])
            per_scene[c["scene_b_id"]] = {
                "tiles": tiles,
                "sims": store.vectors_for([t["faiss_id"] for t in tiles]) @ seed_vector if tiles else np.empty(0),
                "boxes": np.array([t["bounds"] for t in tiles], dtype=np.float64) if tiles else np.empty((0, 4)),
            }
        scene = per_scene[c["scene_b_id"]]
        if not scene["tiles"]:
            continue
        b = scene["boxes"]
        hit = (b[:, 0] <= c["max_lon"]) & (b[:, 2] >= c["min_lon"]) & (b[:, 1] <= c["max_lat"]) & (b[:, 3] >= c["min_lat"])
        if not hit.any():
            continue  # its after-date tile was dropped at tiling (mostly no data): nothing to compare
        best = int(np.flatnonzero(hit)[np.argmax(scene["sims"][hit])])
        tile_id = scene["tiles"][best]["tile_id"]
        if tile_id == seed_tile_id:
            continue
        scored.append(
            SimilarChange(
                candidate_id=c["candidate_id"],
                job_id=c["job_id"],
                scene_a_id=c["scene_a_id"],
                scene_b_id=c["scene_b_id"],
                scene_a_date=c["scene_a_date"],
                scene_b_date=c["scene_b_date"],
                bounds=[c["min_lon"], c["min_lat"], c["max_lon"], c["max_lat"]],
                change_type=c["change_type"],
                direction=c["direction"],
                confidence=c["confidence"],
                area_px=c["area_px"],
                area_ha=c["area_ha"],
                mgrs=c["mgrs_ref"],
                review_status=c["review_status"],
                seasonality_status=c["seasonality_status"],
                similarity=round(float(scene["sims"][best]), 4),
                after_tile_id=tile_id,
            )
        )
    scored.sort(key=lambda r: r.similarity, reverse=True)
    return scored[:top_k]


@router.post("/search/similar", response_model=SimilarResponse, status_code=status.HTTP_200_OK)
def find_similar(payload: SimilarRequest) -> SimilarResponse:
    """The tiles most similar to a seed tile, by cosine similarity of their embeddings, across the whole archive."""
    started = time.perf_counter()
    timer = instrumentation.PipelineTimer()
    store = get_vector_store()

    conn = init_connection()
    try:
        init_schema(conn)
        # The index and the catalog are joined by faiss_id: hold the store lock so a scene deletion cannot renumber them
        with store.lock:
            if payload.faiss_id is not None:
                found = get_tiles_by_faiss_ids(conn, [payload.faiss_id]).get(payload.faiss_id)
                if found is None or payload.faiss_id >= store.total_vectors:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown vector: {payload.faiss_id}")
                tile_id = found["tile_id"]
            else:
                tile_id = payload.tile_id or _candidate_seed_tile(conn, payload.candidate_id)  # type: ignore[arg-type]
            row = conn.execute("SELECT faiss_id FROM tiles WHERE tile_id = ?", (tile_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown tile: {tile_id}")
            if row[0] is None:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This tile has no embedding to compare with.")
            seed_fid = int(row[0])

            with timer.time_stage("vector_lookup"):
                seed_vector = store.vectors_for([seed_fid])[0]
            fetch_k = max((payload.top_k + 1) * OVERFETCH_FACTOR, MIN_FETCH)
            with timer.time_stage("faiss_search"):
                scores_2d, indices_2d = store.search(seed_vector, top_k=fetch_k)
            ranked = [
                (int(i), float(s))
                for i, s in zip(indices_2d[0], scores_2d[0])
                if i >= 0 and int(i) != seed_fid  # the seed is trivially its own best match
            ]
            tiles_map: Dict[int, Dict[str, Any]] = get_tiles_by_faiss_ids(conn, [seed_fid] + [fid for fid, _ in ranked])
            seed_meta = tiles_map[seed_fid]
            results = [_semantic_item(tiles_map[fid], score) for fid, score in ranked if fid in tiles_map]
            with timer.time_stage("change_matching"):
                similar_changes = (
                    _similar_changes(conn, store, seed_vector, payload.candidate_id, seed_meta["tile_id"], payload.top_k)
                    if payload.candidate_id is not None
                    else []
                )
    finally:
        conn.close()

    results.sort(key=lambda r: r.score, reverse=True)
    results = results[: payload.top_k]
    total = time.perf_counter() - started
    instrumentation.record_query(instrumentation.QUERY_SIMILAR, total, {**timer.to_dict(), "total": round(total * 1000.0, 1)})
    return SimilarResponse(
        similar_to=SimilarSeed(
            tile_id=seed_meta["tile_id"],
            scene_id=seed_meta["scene_id"],
            bounds=seed_meta["bounds_wgs84"],
            mgrs=seed_meta.get("mgrs"),
            candidate_id=payload.candidate_id,
        ),
        label=f"Similar to change #{payload.candidate_id}" if payload.candidate_id is not None else f"Similar to {seed_meta['tile_id']}",
        semantic_results=results,
        similar_changes=similar_changes,
    )
