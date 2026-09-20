"""IRIS Catalog Status API — lets the frontend resume from existing catalog state on startup."""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel

from catalog.database import count_tiles, init_connection, init_schema, list_scenes_newest_first
from embedding.index import get_vector_store
from ingestion.loader import read_raster_bounds

logger = logging.getLogger("iris.api.catalog_status")
router = APIRouter(prefix="/api", tags=["catalog"])


class LatestScene(BaseModel):
    """The most recently imported scene whose COG is still on disk (same shape as an ingest response)."""
    scene_id: str
    sensor: str
    acquisition_date: str
    crs: str
    cog_path: str
    cog_url: str
    bounds: List[float]  # [minx, miny, maxx, maxy] in native CRS
    bounds_wgs84: List[float]  # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
    tiles_count: int


class CatalogStatusResponse(BaseModel):
    """Snapshot of what is already imported and indexed."""
    has_scenes: bool
    scene_count: int
    tiles_count: int  # embedded tile rows in the catalog
    faiss_vectors: int  # vectors in the FAISS index; search needs this to be > 0
    latest_scene: Optional[LatestScene] = None
    data_dir: str  # absolute path the catalog was read from (data paths are relative to the server's cwd)


@router.get("/catalog/status", response_model=CatalogStatusResponse, status_code=status.HTTP_200_OK)
def catalog_status() -> CatalogStatusResponse:
    """Report whether scenes exist, the latest scene to resume on, and how many vectors are indexed."""
    conn = init_connection()
    try:
        init_schema(conn)
        scenes = list_scenes_newest_first(conn)
        tiles_total = count_tiles(conn)
    finally:
        conn.close()

    latest: Optional[LatestScene] = None
    for scene in scenes:
        cog_file = Path(scene["cog_path"] or "")
        if not scene["cog_path"] or not cog_file.is_file():
            logger.warning("Scene %s has no COG on disk at %r; skipping for resume", scene["scene_id"], scene["cog_path"])
            continue
        try:
            bounds, bounds_wgs84 = read_raster_bounds(cog_file)
        except Exception:
            logger.exception("Could not read bounds of %s; skipping for resume", cog_file)
            continue
        latest = LatestScene(
            **scene,
            cog_url=f"file:///{cog_file.resolve().as_posix()}",
            bounds=bounds,
            bounds_wgs84=bounds_wgs84,
        )
        break

    return CatalogStatusResponse(
        has_scenes=bool(scenes),
        scene_count=len(scenes),
        tiles_count=tiles_total,
        faiss_vectors=get_vector_store().total_vectors,
        latest_scene=latest,
        data_dir=str(Path("data").resolve()),
    )
