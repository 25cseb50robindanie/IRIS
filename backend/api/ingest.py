"""IRIS Ingestion API Router."""

import logging
from pathlib import Path
from typing import List, Optional
from urllib.request import url2pathname

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from catalog.database import init_connection, init_schema
from ingestion.loader import convert_to_cog, inspect_raster

logger = logging.getLogger("iris.api.ingest")
router = APIRouter(prefix="/api", tags=["ingestion"])


class IngestRequest(BaseModel):
    """Payload for scene ingestion."""
    file_path: str = Field(..., description="Absolute path to satellite image file on local filesystem")


class IngestResponse(BaseModel):
    """Response returned upon successful scene ingestion."""
    scene_id: str
    cog_path: str
    cog_url: str
    sensor: str
    acquisition_date: str
    crs: str
    bounds: List[float]  # [minx, miny, maxx, maxy] in native CRS
    bounds_wgs84: List[float]  # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_scene(payload: IngestRequest) -> IngestResponse:
    """Ingest a satellite scene, detect capabilities, convert to COG, and index in catalog."""
    raw_path_str = payload.file_path.strip().strip('"').strip("'")
    if not raw_path_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file_path cannot be empty",
        )

    if raw_path_str.lower().startswith("file:"):
        # Accept file:///C:/... URIs (e.g. pasted from a browser); url2pathname also unquotes
        raw_path_str = url2pathname(raw_path_str[len("file:") :].removeprefix("//"))

    file_path = Path(raw_path_str).resolve()
    if not file_path.exists() or not file_path.is_file():
        logger.warning("Ingest file not found: raw=%r resolved=%s", payload.file_path, file_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}",
        )

    # 1. Inspect raster and capability detection
    try:
        metadata = inspect_raster(file_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path.name}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to inspect satellite image: {e}",
        )
    except Exception as e:
        logger.exception("Unexpected error during capability detection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inspecting raster capabilities",
        )

    # 2. Convert to COG
    try:
        cog_path = convert_to_cog(file_path, metadata)
    except Exception as e:
        logger.exception("Failed to convert image to COG")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to convert image to Cloud-Optimized GeoTIFF: {e}",
        )

    # Relative path representation for catalog and portability
    relative_cog_path = f"data/cogs/{cog_path.name}"
    # Raw (unencoded) absolute file:/// URI; the client encodes it once when used as a query value
    cog_url = f"file:///{Path(cog_path).resolve().as_posix()}"

    # 3. Write record to SQLite catalog
    try:
        conn = init_connection()
        init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO scenes (
                scene_id,
                sensor,
                acquisition_date,
                crs,
                file_path,
                cog_path,
                cloud_pct,
                raw_checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scene_id) DO UPDATE SET
                sensor=excluded.sensor,
                acquisition_date=excluded.acquisition_date,
                crs=excluded.crs,
                file_path=excluded.file_path,
                cog_path=excluded.cog_path,
                cloud_pct=excluded.cloud_pct,
                raw_checksum=excluded.raw_checksum;
            """,
            (
                metadata.scene_id,
                metadata.sensor,
                metadata.acquisition_date,
                metadata.crs,
                str(file_path),
                relative_cog_path,
                None,
                metadata.raw_checksum,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception("Failed to update catalog database")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record scene metadata in catalog database",
        )

    return IngestResponse(
        scene_id=metadata.scene_id,
        cog_path=relative_cog_path,
        cog_url=cog_url,
        sensor=metadata.sensor,
        acquisition_date=metadata.acquisition_date,
        crs=metadata.crs,
        bounds=metadata.bounds,
        bounds_wgs84=metadata.bounds_wgs84,
    )
