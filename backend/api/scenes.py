"""IRIS Scenes API Router — list imported scenes and delete one with everything that depends on it."""

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from api.ingest import EMBED_LOCK
from catalog.database import (
    delete_scene_rows,
    get_scene,
    init_connection,
    init_schema,
    list_scene_summaries,
    list_tiles_except_scene,
    remap_tile_faiss_ids,
)
from embedding.index import get_vector_store
from embedding.progress import TERMINAL_STATES, embedding_progress
from ingestion.loader import io_path, read_raster_bounds

logger = logging.getLogger("iris.api.scenes")
router = APIRouter(prefix="/api", tags=["scenes"])

COGS_ROOT = Path("data/cogs")
CROPS_ROOT = Path("data/crops")
SCENE_ID_RE = re.compile(r"^[\w\-.]{1,200}$")


class SceneInfo(BaseModel):
    """An imported scene, in the shape the map needs to display it."""
    scene_id: str
    sensor: str
    acquisition_date: str
    crs: str
    cog_path: str
    cog_url: str
    cog_available: bool  # the COG file still exists, so the scene can be shown
    bounds: Optional[List[float]] = None
    bounds_wgs84: Optional[List[float]] = None
    cloud_pct: Optional[float] = None
    tiles_count: int
    has_analysis: bool
    change_candidates: int  # candidates that involve this scene (they are deleted with it)
    reviews: int  # analyst decisions on those candidates (deleted with them)


class SceneList(BaseModel):
    scenes: List[SceneInfo]


class DeleteResult(BaseModel):
    scene_id: str
    vectors_removed: int
    vectors_remaining: int
    tiles: int
    change_candidates: int
    reviews: int
    jobs: int
    files_removed: int
    files_failed: List[str]  # names of files that could not be deleted (e.g. still open); safe to delete by hand


@router.get("/scenes", response_model=SceneList)
def list_scenes() -> SceneList:
    """All imported scenes, most recent acquisition first."""
    conn = init_connection()
    try:
        init_schema(conn)
        rows = list_scene_summaries(conn)
    finally:
        conn.close()

    scenes: List[SceneInfo] = []
    for r in rows:
        cog = Path(r["cog_path"] or "")
        available = bool(r["cog_path"]) and io_path(cog).is_file()
        bounds = bounds_wgs84 = None
        if available:
            try:
                bounds, bounds_wgs84 = read_raster_bounds(cog)
            except Exception:
                logger.warning("Could not read bounds of %s", cog.name)
        scenes.append(
            SceneInfo(
                scene_id=r["scene_id"],
                sensor=r["sensor"],
                acquisition_date=r["acquisition_date"],
                crs=r["crs"],
                cog_path=r["cog_path"] or "",
                cog_url=f"file:///{cog.resolve().as_posix()}" if r["cog_path"] else "",
                cog_available=available,
                bounds=bounds,
                bounds_wgs84=bounds_wgs84,
                cloud_pct=r["cloud_pct"],
                tiles_count=r["tiles_count"],
                has_analysis=bool(r["analysis_cog_path"]),
                change_candidates=r["change_candidates"],
                reviews=r["reviews"],
            )
        )
    return SceneList(scenes=scenes)


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _delete_files(scene: Dict[str, Optional[str]]) -> Dict[str, object]:
    """Delete the scene's COGs and crop folder. Only paths inside data/cogs and data/crops are ever touched."""
    removed = 0
    failed: List[str] = []
    for key in ("cog_path", "analysis_cog_path", "scl_path"):
        value = scene.get(key)
        if not value:
            continue
        path = Path(value)
        if not _inside(path, COGS_ROOT):
            logger.warning("Refusing to delete %s: it is outside data/cogs", path.name)
            failed.append(path.name)
            continue
        target = io_path(path)
        if not target.exists():
            continue
        try:
            target.unlink()
            removed += 1
        except OSError:
            logger.warning("Could not delete %s", path.name)
            failed.append(path.name)

    crops = CROPS_ROOT / str(scene["scene_id"])
    if _inside(crops, CROPS_ROOT) and io_path(crops).is_dir():
        try:
            shutil.rmtree(io_path(crops))
            removed += 1
        except OSError:
            logger.warning("Could not delete the crop folder for %s", scene["scene_id"])
            failed.append(f"{scene['scene_id']} crops")
    return {"removed": removed, "failed": failed}


@router.delete("/scenes/{scene_id}", response_model=DeleteResult)
def delete_scene(scene_id: str) -> DeleteResult:
    """Remove a scene: its vectors, catalog rows (tiles, change candidates, reviews, jobs) and files on disk."""
    if not SCENE_ID_RE.match(scene_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid scene id")

    # An import runs its embedding and change detection under this lock; deleting under it would pull data out
    # from beneath the job. Refuse instead of waiting.
    if not EMBED_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An import is still processing. Wait for it to finish, then delete the scene.",
        )
    try:
        progress = embedding_progress.snapshot(scene_id)
        if progress is not None and progress["state"] not in TERMINAL_STATES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This scene is still being processed. Wait for it to finish, then delete it.",
            )

        conn = init_connection()
        try:
            init_schema(conn)
            scene = get_scene(conn, scene_id)
            if scene is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown scene: {scene_id}")

            kept = list_tiles_except_scene(conn, scene_id)
            store = get_vector_store()
            with store.lock:
                before = store.total_vectors
                new_index, tmp = store.prepare_rebuild([fid for _, fid in kept])
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    counts = delete_scene_rows(conn, scene_id)
                    remap_tile_faiss_ids(conn, kept)
                    # File replace inside the transaction: if it fails, the catalog rolls back and nothing changed
                    store.commit_rebuild(new_index, tmp)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    store.discard_rebuild(tmp)
                    logger.exception("Deleting scene %s failed; catalog rolled back", scene_id)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Could not delete the scene; nothing was changed.",
                    )
                after = store.total_vectors
        finally:
            conn.close()

        embedding_progress.forget(scene_id)
        files = _delete_files(scene)
    finally:
        EMBED_LOCK.release()

    logger.info("Deleted scene %s: %s, vectors %d -> %d", scene_id, counts, before, after)
    return DeleteResult(
        scene_id=scene_id,
        vectors_removed=before - after,
        vectors_remaining=after,
        tiles=counts["tiles"],
        change_candidates=counts["change_candidates"],
        reviews=counts["reviews"],
        jobs=counts["jobs"],
        files_removed=int(files["removed"]),
        files_failed=list(files["failed"]),  # type: ignore[arg-type]
    )
