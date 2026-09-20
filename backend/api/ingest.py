"""IRIS Ingestion API Router — Ingest satellite scenes, convert to COG, then embed crops in the background."""

import logging
import threading
from pathlib import Path
from typing import List, Optional
from urllib.request import url2pathname

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from catalog.database import get_scene_tile_count, init_connection, init_schema, insert_tiles_batch
from embedding import progress as stages
from embedding.crop import extract_crops
from embedding.embedder import CHECKPOINT_MISSING_MSG, RemoteCLIPEmbedder, find_checkpoint
from embedding.index import get_vector_store
from embedding.progress import embedding_progress
from ingestion.loader import convert_to_cog, inspect_raster

logger = logging.getLogger("iris.api.ingest")
router = APIRouter(prefix="/api", tags=["ingestion"])

EMBED_BATCH_SIZE = 32
# Ingestion is serialized: one embedding job at a time keeps memory bounded and FAISS ids consistent
_EMBED_LOCK = threading.Lock()


class IngestRequest(BaseModel):
    """Payload for scene ingestion."""
    file_path: str = Field(..., description="Absolute path to satellite image file on local filesystem")


class IngestResponse(BaseModel):
    """Response returned once the scene is cataloged and viewable; embedding continues in the background."""
    scene_id: str
    cog_path: str
    cog_url: str
    sensor: str
    acquisition_date: str
    crs: str
    bounds: List[float]  # [minx, miny, maxx, maxy] in native CRS
    bounds_wgs84: List[float]  # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
    tiles_count: int = 0
    embedding_status: str = stages.QUEUED  # poll GET /api/status/{scene_id} for progress


class EmbeddingStatusResponse(BaseModel):
    """Progress of a scene's background embedding job."""
    scene_id: str
    state: str  # queued | tiling | embedding | indexing | ready | failed | not_embedded
    crops_generated: int = 0
    crops_total: int = 0
    crops_embedded: int = 0
    tiles_count: int = 0
    device: Optional[str] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0
    crops_per_sec: Optional[float] = None


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_scene(payload: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """Ingest a satellite scene and convert it to COG so it is viewable at once.

    Crop extraction, RemoteCLIP embedding and FAISS indexing continue as a background job;
    poll GET /api/status/{scene_id} for progress.
    """
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
    cog_url = f"file:///{Path(cog_path).resolve().as_posix()}"

    # 3. Record the scene in the catalog so it is browsable before embedding finishes
    try:
        conn = init_connection()
        try:
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
                    raw_checksum=excluded.raw_checksum,
                    created_at=strftime('%Y-%m-%d %H:%M:%f', 'now');  -- re-import becomes the most recent scene
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
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to update catalog database")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record scene metadata in catalog database",
        )

    # 4. Embedding continues in the background so the map can load immediately
    embedding_progress.start(metadata.scene_id)
    background_tasks.add_task(_run_embedding_job, cog_path, metadata.scene_id)

    return IngestResponse(
        scene_id=metadata.scene_id,
        cog_path=relative_cog_path,
        cog_url=cog_url,
        sensor=metadata.sensor,
        acquisition_date=metadata.acquisition_date,
        crs=metadata.crs,
        bounds=metadata.bounds,
        bounds_wgs84=metadata.bounds_wgs84,
        tiles_count=0,
        embedding_status=stages.QUEUED,
    )


@router.get("/status/{scene_id}", response_model=EmbeddingStatusResponse)
def embedding_status(scene_id: str) -> EmbeddingStatusResponse:
    """Report tiling / embedding / indexing progress for a scene."""
    snap = embedding_progress.snapshot(scene_id)
    if snap is not None:
        return EmbeddingStatusResponse(**snap)

    # Not tracked in this process (e.g. the server restarted): fall back to what the catalog says
    conn = init_connection()
    try:
        init_schema(conn)
        tile_count = get_scene_tile_count(conn, scene_id)
    finally:
        conn.close()

    if tile_count is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown scene: {scene_id}")
    if tile_count > 0:
        return EmbeddingStatusResponse(scene_id=scene_id, state=stages.READY, tiles_count=tile_count)
    return EmbeddingStatusResponse(
        scene_id=scene_id,
        state="not_embedded",
        error="No embeddings recorded for this scene; re-import it to embed.",
    )


def _run_embedding_job(cog_path: Path, scene_id: str) -> None:
    """Background task: run the embedding pipeline, recording any failure in the progress tracker."""
    with _EMBED_LOCK:
        try:
            _embed_scene(cog_path, scene_id)
        except Exception as e:
            logger.exception("Embedding failed for scene %s", scene_id)
            message = str(e) if isinstance(e, FileNotFoundError) else f"{type(e).__name__}: {e}"
            embedding_progress.fail(scene_id, message)


def _embed_scene(cog_path: Path, scene_id: str) -> None:
    """Crop -> RemoteCLIP -> FAISS -> catalog tiles, reporting progress at each stage."""
    if find_checkpoint() is None:
        # Fail before doing any tiling work
        logger.error(CHECKPOINT_MISSING_MSG)
        raise FileNotFoundError(CHECKPOINT_MISSING_MSG)

    # Tiling: 224x224 crops, skipping any that are >50% nodata/black
    embedding_progress.update(scene_id, state=stages.TILING)
    crops = extract_crops(
        cog_path,
        scene_id,
        on_progress=lambda n: embedding_progress.update(scene_id, crops_generated=n),
    )
    embedding_progress.update(scene_id, crops_generated=len(crops), crops_total=len(crops))
    if not crops:
        logger.warning("Scene %s produced no valid crops; nothing to embed", scene_id)
        embedding_progress.update(scene_id, state=stages.READY, tiles_count=0)
        return

    # Embedding: batches of 32, on GPU when available
    embedder = RemoteCLIPEmbedder.get_instance()
    embedding_progress.update(scene_id, state=stages.EMBEDDING, device=str(embedder.device))
    embeddings = embedder.embed_images(
        [c.file_path for c in crops],
        batch_size=EMBED_BATCH_SIZE,
        on_batch=lambda done, _total: embedding_progress.update(scene_id, crops_embedded=done),
    )

    # An unusable crop must be absent from the index, never represented by a fabricated vector
    finite = np.isfinite(embeddings).all(axis=1)
    if not finite.all():
        logger.warning("Dropping %d crops with non-finite embeddings in scene %s", int((~finite).sum()), scene_id)
        crops = [c for c, ok in zip(crops, finite) if ok]
        embeddings = embeddings[finite]

    # Indexing: FAISS first, then the catalog rows that map faiss_id -> tile bounds
    embedding_progress.update(scene_id, state=stages.INDEXING)
    faiss_ids = get_vector_store().add_vectors(embeddings)

    tiles_records = [
        {
            "tile_id": crop.tile_id,
            "scene_id": crop.scene_id,
            "faiss_id": fid,
            "min_x": crop.bounds_native[0],
            "min_y": crop.bounds_native[1],
            "max_x": crop.bounds_native[2],
            "max_y": crop.bounds_native[3],
            "min_lon": crop.bounds_wgs84[0],
            "min_lat": crop.bounds_wgs84[1],
            "max_lon": crop.bounds_wgs84[2],
            "max_lat": crop.bounds_wgs84[3],
            "cloud_pct": None,
            "valid_pixel_frac": crop.valid_pixel_frac,
        }
        for crop, fid in zip(crops, faiss_ids)
    ]

    conn = init_connection()
    try:
        init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        insert_tiles_batch(conn, tiles_records)
        conn.commit()
    finally:
        conn.close()

    embedding_progress.update(scene_id, state=stages.READY, tiles_count=len(tiles_records))
    logger.info("Scene %s embedded: %d tiles indexed", scene_id, len(tiles_records))
