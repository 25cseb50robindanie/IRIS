"""IRIS Ingestion API Router — Ingest satellite scenes, convert to COG, then embed crops in the background."""

import logging
import threading
from pathlib import Path
from typing import List, Optional
from urllib.request import url2pathname

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from catalog.database import get_scene_tile_count, init_connection, init_schema, insert_tiles_batch, upsert_scene
from api.pipeline_status import PIPELINE_ID_RE, SAFE_STEPS, SINGLE_FILE_STEPS, new_pipeline_id, pipeline_tracker
from change_detection.trigger import Reporter, schedule_change_detection
import instrumentation
from embedding import progress as stages
from embedding.crop import extract_crops
from embedding.embedder import CHECKPOINT_MISSING_MSG, RemoteCLIPEmbedder, find_checkpoint
from embedding.index import get_vector_store
from embedding.progress import embedding_progress
from ingestion.loader import convert_to_cog, inspect_raster, io_path
from ingestion.safe import build_sentinel2_cogs, inspect_safe_product, open_safe_product
from mgrs_ref import bounds_centre_mgrs

logger = logging.getLogger("iris.api.ingest")
router = APIRouter(prefix="/api", tags=["ingestion"])

COG_DIR = "data/cogs"
EMBED_BATCH_SIZE = 32
# Ingestion is serialized: one embedding job at a time keeps memory bounded and FAISS ids consistent
EMBED_LOCK = threading.Lock()
_EMBED_LOCK = EMBED_LOCK  # scene deletion takes the same lock, so it cannot race a running job


class IngestRequest(BaseModel):
    """Payload for scene ingestion."""
    file_path: str = Field(
        ...,
        description="Absolute path to a satellite image file, or to a Sentinel-2 L2A .SAFE folder",
    )
    pipeline_id: Optional[str] = Field(
        default=None,
        pattern=PIPELINE_ID_RE.pattern,
        description="Client-chosen id to poll GET /api/pipeline/{id} for step-by-step progress during the import",
    )


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
    product_type: str = "single_file"  # "sentinel2_safe" when a .SAFE folder was imported
    analysis_cog_path: Optional[str] = None  # 4-band B02/B03/B04/B08 COG (SAFE only)
    scl_path: Optional[str] = None  # SCL resampled to 10 m (SAFE only)
    cloud_pct: Optional[float] = None  # % cloud/shadow/cirrus from the SCL (SAFE only)
    pipeline_id: Optional[str] = None  # poll GET /api/pipeline/{id} until its state is done or failed


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


def _stage_reporter(pid: str):
    """Adapt build_sentinel2_cogs' (stage, done, total) callback to pipeline steps."""

    def report(stage: str, done: int, total: int) -> None:
        if done >= total:
            pipeline_tracker.complete(pid, stage)
        elif done == 0:
            pipeline_tracker.begin(pid, stage)
        else:
            pipeline_tracker.progress(pid, stage, done, total, f"{100 * done // total}%")

    return report


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_scene(payload: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """Ingest a satellite scene and convert it to COG so it is viewable at once.

    Crop extraction, RemoteCLIP embedding, FAISS indexing and change detection continue as a background job;
    poll GET /api/pipeline/{pipeline_id} for step-by-step progress from the moment this request starts.
    """
    pid = payload.pipeline_id or new_pipeline_id()
    pipeline_tracker.start(pid, Path(payload.file_path.strip().strip("\"'")).name)
    try:
        return _ingest(payload, background_tasks, pid)
    except HTTPException as e:
        pipeline_tracker.fail_active(pid, str(e.detail))
        raise
    except Exception:
        pipeline_tracker.fail_active(pid, "Unexpected error during import")
        raise


def _ingest(payload: IngestRequest, background_tasks: BackgroundTasks, pid: str) -> IngestResponse:
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
    probe = io_path(file_path)  # Python's stat fails beyond Windows' 260-character limit without this
    if not probe.exists() or not (probe.is_file() or probe.is_dir()):
        logger.warning("Ingest path not found: raw=%r resolved=%s", payload.file_path, file_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}",
        )

    # A folder must be a Sentinel-2 L2A product (.SAFE, or a folder containing one); anything else is a
    # single raster file and takes the original flow.
    product = None
    if probe.is_dir():
        try:
            product = open_safe_product(file_path)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid Sentinel-2 product: {e}",
            )
        if product is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This folder does not contain a Sentinel-2 L2A product (MTD_MSIL2A.xml not found)",
            )

    # 1. Inspect raster and capability detection
    try:
        metadata = inspect_safe_product(product) if product else inspect_raster(file_path)
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
    except Exception:
        logger.exception("Unexpected error during capability detection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inspecting raster capabilities",
        )

    pipeline_tracker.complete(pid, "detect", "Sentinel-2 L2A detected" if product else f"{metadata.driver} raster detected")
    pipeline_tracker.plan(pid, SAFE_STEPS if product else SINGLE_FILE_STEPS)

    # 2. Convert to COG(s): display + analysis + SCL for a .SAFE product, a single COG otherwise
    analysis_rel: Optional[str] = None
    scl_rel: Optional[str] = None
    cloud_pct: Optional[float] = None
    try:
        if product:
            outputs = build_sentinel2_cogs(product, on_stage=_stage_reporter(pid))
            cog_path = outputs.display_cog
            analysis_rel = f"{COG_DIR}/{outputs.analysis_cog.name}"
            scl_rel = f"{COG_DIR}/{outputs.scl_cog.name}"
            cloud_pct = outputs.cloud_pct
        else:
            pipeline_tracker.begin(pid, "cog")
            cog_path = convert_to_cog(file_path, metadata)
            pipeline_tracker.complete(pid, "cog")
    except ValueError as e:
        if not product:
            raise
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid Sentinel-2 product: {e}",
        )
    except Exception as e:
        logger.exception("Failed to convert image to COG")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to convert image to Cloud-Optimized GeoTIFF: {e}",
        )

    # Relative path representation for catalog and portability
    relative_cog_path = f"{COG_DIR}/{cog_path.name}"
    cog_url = f"file:///{Path(cog_path).resolve().as_posix()}"

    # 3. Record the scene in the catalog so it is browsable before embedding finishes
    try:
        conn = init_connection()
        try:
            init_schema(conn)
            conn.execute("BEGIN IMMEDIATE")
            upsert_scene(
                conn,
                {
                    "scene_id": metadata.scene_id,
                    "sensor": metadata.sensor,
                    "acquisition_date": metadata.acquisition_date,
                    "crs": metadata.crs,
                    "file_path": str(file_path),
                    "cog_path": relative_cog_path,
                    "analysis_cog_path": analysis_rel,
                    "scl_path": scl_rel,
                    "cloud_pct": cloud_pct,
                    "raw_checksum": metadata.raw_checksum,
                },
                bounds_wgs84=metadata.bounds_wgs84,
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

    # 4. Embedding (then change detection) continues in the background so the map can load immediately
    embedding_progress.start(metadata.scene_id)
    pipeline_tracker.set_scene(pid, metadata.scene_id)
    background_tasks.add_task(_run_embedding_job, cog_path, metadata.scene_id, pid)

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
        product_type="sentinel2_safe" if product else "single_file",
        analysis_cog_path=analysis_rel,
        scl_path=scl_rel,
        cloud_pct=cloud_pct,
        pipeline_id=pid,
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


class _PipelineReporter(Reporter):
    """Feeds change-detection progress into the import's pipeline record."""

    def __init__(self, pid: Optional[str]) -> None:
        self.pid = pid
        self.candidates = 0
        self.lines: List[str] = []

    def overlap_check_started(self) -> None:
        pipeline_tracker.begin(self.pid, "overlap")

    def pairs_found(self, scene_id: str, pairs) -> None:
        if not pairs:
            pipeline_tracker.complete(self.pid, "overlap", "No overlapping scene found")
            pipeline_tracker.skip(
                self.pid, "change", "No overlapping scene yet. Import a second scene of the same area to enable change detection"
            )
        else:
            pipeline_tracker.complete(self.pid, "overlap", f"{len(pairs)} overlapping scene{'s' if len(pairs) != 1 else ''}")

    @staticmethod
    def _other(scene_id: str, a_id: str, b_id: str) -> str:
        return a_id if b_id == scene_id else b_id

    def pair_started(self, scene_id: str, a_id: str, b_id: str) -> None:
        pipeline_tracker.begin(self.pid, "change", f"against {self._other(scene_id, a_id, b_id)}")

    def pair_finished(self, scene_id: str, a_id: str, b_id: str, result) -> None:
        other = self._other(scene_id, a_id, b_id)
        n = result.get("candidates")
        if result.get("status") == "completed" and isinstance(n, int):
            self.candidates += n
            self.lines.append(f"against {other}: {n} candidate{'s' if n != 1 else ''}")
        elif result.get("status") == "failed":
            pipeline_tracker.fail_step(self.pid, "change", f"against {other}: analysis failed")
            return
        else:
            self.lines.append(f"against {other}: {str(result.get('status', 'done')).replace('_', ' ')}")
        pipeline_tracker.complete(self.pid, "change", "; ".join(self.lines))


def _run_embedding_job(cog_path: Path, scene_id: str, pid: Optional[str] = None) -> None:
    """Background task: embed the scene, then pair it and run change detection, reporting each step."""
    with _EMBED_LOCK:
        tiles = 0
        embed_error: Optional[str] = None
        try:
            tiles = _embed_scene(cog_path, scene_id, pid)
        except Exception as e:
            logger.exception("Embedding failed for scene %s", scene_id)
            embed_error = str(e) if isinstance(e, FileNotFoundError) else f"{type(e).__name__}: {e}"
            embedding_progress.fail(scene_id, embed_error)
            # The pipeline carries on to change detection; if nothing was running yet (e.g. no checkpoint), the
            # failure belongs to the embedding step
            if not pipeline_tracker.fail_active(pid, embed_error, end=False):
                pipeline_tracker.fail_step(pid, "embed", embed_error)

        # Pair the scene with overlapping same-sensor scenes and run change detection. It never needs the
        # embeddings, so a failed embedding does not block it, and its failure must not undo the embedding.
        reporter = _PipelineReporter(pid)
        try:
            schedule_change_detection(scene_id, reporter)
        except Exception:
            logger.exception("Change detection trigger failed for scene %s", scene_id)
            pipeline_tracker.fail_active(pid, "Change detection could not run", end=False)

        n = reporter.candidates
        found = f"{n} change candidate{'s' if n != 1 else ''} found"
        if embed_error:
            pipeline_tracker.finish(pid, f"Completed with errors — embedding failed; {found}")
        else:
            pipeline_tracker.finish(pid, f"Complete — {tiles} tile{'s' if tiles != 1 else ''} indexed, {found}")
        snap = pipeline_tracker.snapshot(pid or "")
        instrumentation.record_ingestion(
            scene_id,
            snap["name"] if snap else "",
            pipeline_tracker.timings(pid),
            pipeline_tracker.elapsed(pid),
            "completed_with_errors" if embed_error else "completed",
            tiles,
        )


def _embed_scene(cog_path: Path, scene_id: str, pid: Optional[str] = None) -> int:
    """Crop -> RemoteCLIP -> FAISS -> catalog tiles, reporting progress at each stage. Returns tiles indexed."""
    if find_checkpoint() is None:
        # Fail before doing any tiling work
        logger.error(CHECKPOINT_MISSING_MSG)
        raise FileNotFoundError(CHECKPOINT_MISSING_MSG)

    # Tiling: 224x224 crops, skipping any that are >50% nodata/black
    embedding_progress.update(scene_id, state=stages.TILING)
    pipeline_tracker.begin(pid, "crops")
    crops = extract_crops(
        cog_path,
        scene_id,
        on_progress=lambda n: embedding_progress.update(scene_id, crops_generated=n),
        on_scan=lambda done, total: pipeline_tracker.progress(pid, "crops", done, total, f"{done}/{total}"),
    )
    embedding_progress.update(scene_id, crops_generated=len(crops), crops_total=len(crops))
    pipeline_tracker.complete(pid, "crops", f"{len(crops)} crops")
    if not crops:
        logger.warning("Scene %s produced no valid crops; nothing to embed", scene_id)
        embedding_progress.update(scene_id, state=stages.READY, tiles_count=0)
        pipeline_tracker.skip(pid, "embed", "No valid crops to embed")
        pipeline_tracker.skip(pid, "index", "Nothing to index")
        return 0

    # Embedding: batches of 32, on GPU when available
    pipeline_tracker.begin(pid, "embed", "loading model")
    embedder = RemoteCLIPEmbedder.get_instance()
    embedding_progress.update(scene_id, state=stages.EMBEDDING, device=str(embedder.device))

    def on_batch(done: int, total: int) -> None:
        embedding_progress.update(scene_id, crops_embedded=done)
        pipeline_tracker.progress(pid, "embed", done, total, f"{done}/{total} crops")

    embeddings = embedder.embed_images([c.file_path for c in crops], batch_size=EMBED_BATCH_SIZE, on_batch=on_batch)
    pipeline_tracker.complete(pid, "embed", f"{len(crops)} crops on {embedder.device}")

    # An unusable crop must be absent from the index, never represented by a fabricated vector
    finite = np.isfinite(embeddings).all(axis=1)
    if not finite.all():
        logger.warning("Dropping %d crops with non-finite embeddings in scene %s", int((~finite).sum()), scene_id)
        crops = [c for c, ok in zip(crops, finite) if ok]
        embeddings = embeddings[finite]

    # Indexing: FAISS first, then the catalog rows that map faiss_id -> tile bounds
    embedding_progress.update(scene_id, state=stages.INDEXING)
    pipeline_tracker.begin(pid, "index")
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
            "mgrs_ref": bounds_centre_mgrs(crop.bounds_wgs84),  # 10-digit MGRS of the tile centroid
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
    pipeline_tracker.complete(pid, "index", f"{len(tiles_records)} tiles")
    logger.info("Scene %s embedded: %d tiles indexed", scene_id, len(tiles_records))
    return len(tiles_records)
