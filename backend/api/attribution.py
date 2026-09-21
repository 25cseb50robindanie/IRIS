"""IRIS Attribution API Router — where in a search-result tile the query matched (see embedding/attribution.py)."""

import logging
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from rasterio.warp import transform as warp_points

from catalog.database import init_connection, init_schema
from embedding.attribution import attribution_heatmap
from ingestion.loader import io_path

logger = logging.getLogger("iris.api.attribution")
router = APIRouter(prefix="/api", tags=["search"])

TILE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,300}$")  # ASCII on purpose: pydantic's regex engine caps the size of \w{1,300}
CROPS_ROOT = Path("data/crops")


class AttributionRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="The text query the tile was returned for")
    tile_id: str = Field(..., pattern=TILE_ID_RE.pattern)


class AttributionResponse(BaseModel):
    tile_id: str
    heatmap_base64: str  # RGBA PNG, 224x224, transparent where the match is weak
    bounds_wgs84: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    corners_wgs84: Optional[List[List[float]]] = None  # [lon, lat] of the top-left, top-right, bottom-right, bottom-left corners
    method: str
    query_conditioned: bool  # False when the raw attention was used because the query's gradient carried no signal


def _corners(crs: Optional[str], native: List[float]) -> Optional[List[List[float]]]:
    """The tile's four corners in WGS84. A UTM tile is slightly rotated in lon/lat, so its box alone would not fit it."""
    if not crs:
        return None
    min_x, min_y, max_x, max_y = native
    try:
        lons, lats = warp_points(crs, "EPSG:4326", [min_x, max_x, max_x, min_x], [max_y, max_y, min_y, min_y])
    except Exception:
        return None
    return [[float(lo), float(la)] for lo, la in zip(lons, lats)]


@router.post("/search/attribution", response_model=AttributionResponse, status_code=status.HTTP_200_OK)
def tile_attribution(payload: AttributionRequest) -> AttributionResponse:
    """A heatmap of the patches of a tile that made it match the query."""
    conn = init_connection()
    try:
        init_schema(conn)
        row = conn.execute(
            "SELECT t.scene_id, t.min_x, t.min_y, t.max_x, t.max_y, t.min_lon, t.min_lat, t.max_lon, t.max_lat, s.crs "
            "FROM tiles t JOIN scenes s ON s.scene_id = t.scene_id WHERE t.tile_id = ?",
            (payload.tile_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown tile: {payload.tile_id}")

    scene_id = row[0]
    crop = CROPS_ROOT / scene_id / f"{payload.tile_id}.png"  # both parts come from the catalog / the ASCII pattern above
    if not io_path(crop).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The image crop for this tile is no longer on disk.")

    try:
        heatmap, method = attribution_heatmap(crop, payload.query.strip())
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception:
        logger.exception("Attribution failed for tile %s", payload.tile_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="The attribution heatmap could not be computed.")

    native = [row[1], row[2], row[3], row[4]]
    bounds = [row[5], row[6], row[7], row[8]] if row[5] is not None else native
    return AttributionResponse(
        tile_id=payload.tile_id,
        heatmap_base64=heatmap,
        bounds_wgs84=bounds,
        corners_wgs84=_corners(row[9], native) if row[5] is not None else None,
        method=method,
        query_conditioned=method == "gradient_weighted_last_layer_attention",
    )
