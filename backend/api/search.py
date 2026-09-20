"""IRIS Search API Router — one query, two answers: matching tiles, and matching changes.

  a) semantic_results — RemoteCLIP text embedding against the FAISS index, top-k tiles by cosine similarity,
     scoped to one scene when `scene_id` is given (the scene shown on the map).
  b) change_results   — the detected changes that overlap those tiles AND whose after-date crop also matches the
     query (see change_search.py), with a status saying why there are none when there are none.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from api.change_search import analyse_changes
from catalog.database import get_scene, get_tiles_by_faiss_ids, init_connection, init_schema, list_scene_tiles
from embedding.embedder import RemoteCLIPEmbedder
from embedding.index import get_vector_store
from ingestion.loader import io_path

logger = logging.getLogger("iris.api.search")
router = APIRouter(prefix="/api", tags=["search"])

# Re-importing a scene leaves its old vectors in the HNSW graph (FAISS can't delete them) but
# their catalog rows point at the new ones, so fetch extra hits to still fill top_k after the join.
OVERFETCH_FACTOR = 5
MIN_FETCH = 50
SCENE_ID_RE = re.compile(r"^[\w\-.]{1,200}$")


class SearchRequest(BaseModel):
    """Payload for a search query."""
    query: str = Field(..., min_length=1, max_length=500, description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of semantic results to return")
    scene_id: Optional[str] = Field(
        default=None,
        pattern=SCENE_ID_RE.pattern,
        description="Restrict results to this scene (the one shown on the map); omit to search every scene",
    )


class SearchResultItem(BaseModel):
    """Individual ranked semantic result (a tile)."""
    tile_id: str
    scene_id: str
    score: float
    bounds: List[float]  # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326 for map fitting
    bounds_native: Optional[List[float]] = None
    crop_url: Optional[str] = None


class ChangeResultItem(BaseModel):
    """A detected change that matches the query."""
    candidate_id: int
    job_id: int
    scene_a_id: str
    scene_b_id: str
    scene_a_date: Optional[str] = None
    scene_b_date: Optional[str] = None
    bounds: List[float]
    change_type: Optional[str] = None
    direction: Optional[str] = None
    confidence: float
    mean_dndvi: Optional[float] = None
    area_px: Optional[int] = None
    review_status: str
    semantic_match_score: float  # rank (0-1) of the after-crop's similarity among all tiles of the after scene
    semantic_similarity: float  # raw cosine similarity of that crop to the query
    matches_query: bool
    combined_score: float  # 0.5 * semantic_match_score + 0.5 * confidence + direction_boost
    direction_boost: float = 0.0  # added when the query's words hint at this candidate's direction
    evidence: Optional[Dict[str, Any]] = None  # direction evidence (dominant class at each date, rule)
    after_tile_id: Optional[str] = None


class PairMeta(BaseModel):
    job_id: int
    scene_a_id: Optional[str] = None
    scene_b_id: Optional[str] = None
    dates: str
    valid_coverage: Optional[float] = None
    candidates: int


class ChangeMeta(BaseModel):
    dates_compared: Optional[str] = None  # widest span across the analysed pairs, e.g. "2022-06-23 → 2026-08-21"
    valid_coverage: Optional[float] = None  # the least-observed analysed pair (a negative is only as strong as this)
    total_candidates: int = 0
    matching_candidates: int = 0
    pairs: List[PairMeta] = []
    note: Optional[str] = None
    match_percentile: Optional[float] = None
    unscored_candidates: int = 0
    thresholds: Dict[str, Any] = {}  # what 'no significant change' was measured against
    direction_hints: List[str] = []  # directions the query's wording asked for ("new" -> appearance, ...)


class SearchResponse(BaseModel):
    """Both result sets for one query."""
    query: str
    scene_id: Optional[str] = None
    semantic_results: List[SearchResultItem]
    change_results: List[ChangeResultItem]
    change_status: str  # found | no_match | no_change_detected | no_comparison_available
    change_meta: ChangeMeta


def _semantic_item(tile_meta: Dict[str, Any], score: float) -> SearchResultItem:
    # Check if local crop image exists for UI preview
    crop_file = Path(f"data/crops/{tile_meta['scene_id']}/{tile_meta['tile_id']}.png")
    crop_url = f"file:///{crop_file.resolve().as_posix()}" if io_path(crop_file).exists() else None
    return SearchResultItem(
        tile_id=tile_meta["tile_id"],
        scene_id=tile_meta["scene_id"],
        score=round(score, 4),
        bounds=tile_meta["bounds_wgs84"],
        bounds_native=tile_meta["bounds_native"],
        crop_url=crop_url,
    )


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def search_imagery(payload: SearchRequest) -> SearchResponse:
    """Semantic search plus change-aware search for one query."""
    query_str = payload.query.strip()
    if not query_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty",
        )

    # 1. Load FAISS index; an empty index is a state the analyst can fix, not a server error
    vector_store = get_vector_store()
    if vector_store.total_vectors == 0:
        logger.info("Search requested but FAISS vector index is currently empty")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No imagery has been indexed yet. Import a scene and wait for embedding to finish, then search again.",
        )

    # 2. Compute text query embedding via RemoteCLIP
    try:
        embedder = RemoteCLIPEmbedder.get_instance()
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Failed to load RemoteCLIP model: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RemoteCLIP model could not be loaded: {e}",
        )

    try:
        query_vector = embedder.embed_text(query_str)
    except Exception as e:
        logger.exception("Failed to encode search query with RemoteCLIP: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode the search query: {e}",
        )

    # The index and the catalog are joined by faiss_id, so every read below happens while no scene deletion can
    # rebuild the index and renumber those ids in between.
    conn = init_connection()
    try:
        init_schema(conn)
        with vector_store.lock:
            semantic: List[SearchResultItem] = []

            if payload.scene_id:
                if get_scene(conn, payload.scene_id) is None:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown scene: {payload.scene_id}")
                # 3a. Scene-scoped: the scene's tiles are the allowlist, scanned exactly
                tiles = list_scene_tiles(conn, payload.scene_id)
                scores, ids = vector_store.search_subset(query_vector, [t["faiss_id"] for t in tiles], payload.top_k)
                tiles_map = get_tiles_by_faiss_ids(conn, [int(i) for i in ids])
                ranked = [(int(i), float(s)) for i, s in zip(ids, scores)]
            else:
                # 3b. Unscoped: cosine nearest neighbours over the whole index (over-fetched, see OVERFETCH_FACTOR)
                fetch_k = max(payload.top_k * OVERFETCH_FACTOR, MIN_FETCH)
                scores_2d, indices_2d = vector_store.search(query_vector, top_k=fetch_k)
                ranked = []
                if len(indices_2d) and len(indices_2d[0]):
                    ranked = [(int(i), float(s)) for i, s in zip(indices_2d[0], scores_2d[0]) if i >= 0]
                tiles_map = get_tiles_by_faiss_ids(conn, [fid for fid, _ in ranked])

            for fid, score in ranked:
                if fid in tiles_map:
                    semantic.append(_semantic_item(tiles_map[fid], score))
            semantic.sort(key=lambda r: r.score, reverse=True)
            semantic = semantic[: payload.top_k]

            # 4. Change-aware search over the same tiles
            change = analyse_changes(
                conn,
                vector_store,
                query_vector,
                payload.scene_id,
                [{"bounds": r.bounds} for r in semantic],
                query_str,
            )
    finally:
        conn.close()

    return SearchResponse(
        query=query_str,
        scene_id=payload.scene_id,
        semantic_results=semantic,
        change_results=[ChangeResultItem(**r) for r in change["change_results"]],
        change_status=change["change_status"],
        change_meta=ChangeMeta(**change["change_meta"]),
    )
