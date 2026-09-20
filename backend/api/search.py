"""IRIS Search API Router — Semantic retrieval over satellite imagery embeddings."""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from catalog.database import get_tiles_by_faiss_ids, init_connection
from embedding.embedder import RemoteCLIPEmbedder
from embedding.index import get_vector_store

logger = logging.getLogger("iris.api.search")
router = APIRouter(prefix="/api", tags=["search"])

# Re-importing a scene leaves its old vectors in the HNSW graph (FAISS can't delete them) but
# their catalog rows point at the new ones, so fetch extra hits to still fill top_k after the join.
OVERFETCH_FACTOR = 5
MIN_FETCH = 50


class SearchRequest(BaseModel):
    """Payload for semantic search query."""
    query: str = Field(..., min_length=1, max_length=500, description="Natural language semantic search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")


class SearchResultItem(BaseModel):
    """Individual ranked search result."""
    tile_id: str
    scene_id: str
    score: float
    bounds: List[float]  # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326 for map fitting
    bounds_native: Optional[List[float]] = None
    crop_url: Optional[str] = None


class SearchResponse(BaseModel):
    """Response containing ranked search results."""
    query: str
    total_results: int
    results: List[SearchResultItem]


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def search_imagery(payload: SearchRequest) -> SearchResponse:
    """Perform semantic search using RemoteCLIP text embeddings against the FAISS index."""
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

    # 3. Cosine nearest neighbours (inner product on L2-normalized vectors)
    fetch_k = max(payload.top_k * OVERFETCH_FACTOR, MIN_FETCH)
    scores, indices = vector_store.search(query_vector, top_k=fetch_k)

    if len(indices) == 0 or len(indices[0]) == 0:
        return SearchResponse(query=query_str, total_results=0, results=[])

    matched_faiss_ids = [int(idx) for idx in indices[0] if idx >= 0]
    matched_scores = [float(s) for idx, s in zip(indices[0], scores[0]) if idx >= 0]

    # 4. Lookup metadata in SQLite catalog
    conn = init_connection()
    try:
        tiles_map = get_tiles_by_faiss_ids(conn, matched_faiss_ids)
    finally:
        conn.close()

    # 5. Build ranked results
    results: List[SearchResultItem] = []
    for fid, score in zip(matched_faiss_ids, matched_scores):
        if fid not in tiles_map:
            continue
        
        tile_meta = tiles_map[fid]
        
        # Check if local crop image exists for UI preview
        crop_file = Path(f"data/crops/{tile_meta['scene_id']}/{tile_meta['tile_id']}.png")
        crop_url = f"file:///{crop_file.resolve().as_posix()}" if crop_file.exists() else None

        results.append(
            SearchResultItem(
                tile_id=tile_meta["tile_id"],
                scene_id=tile_meta["scene_id"],
                score=round(score, 4),
                bounds=tile_meta["bounds_wgs84"],
                bounds_native=tile_meta["bounds_native"],
                crop_url=crop_url,
            )
        )

    # Sort descending by similarity score and trim the over-fetch back to what was asked for
    results.sort(key=lambda x: x.score, reverse=True)
    results = results[: payload.top_k]

    return SearchResponse(
        query=query_str,
        total_results=len(results),
        results=results,
    )
