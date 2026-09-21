"""IRIS Ablation API Router — what the suppression stack removes, for one analysed pair.

The ablation is the same pair differenced with every suppression stage off (change_detection/ablation.py). It runs
automatically after a pair's analysis; for a pair analysed before that existed, POST /changes/ablation/run starts it.
"""

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel

from api.ingest import EMBED_LOCK
from catalog import ablation as store
from catalog.database import init_connection, init_schema
from change_detection.ablation import run_ablation

logger = logging.getLogger("iris.api.ablation")
router = APIRouter(prefix="/api", tags=["change"])

SCENE_ID_PATTERN = re.compile(r"^[\w\-.]{1,200}$").pattern
MAX_LIST = 5000

# Jobs whose ablation was started by this process and has not finished. A recorded "running" with nothing in flight and
# nothing holding the analysis lock is a run that was cut short (the backend restarted), and reads as "not run".
_in_flight: set = set()
_in_flight_lock = threading.Lock()


class AblationStats(BaseModel):
    scene_a_id: str
    scene_b_id: str
    status: str  # completed | running | failed | not_run
    full_count: int  # detections the full pipeline reports for the pair (stored, plus any only the storage cap dropped)
    ablation_count: int  # raw detections with suppression off (the true total, not the stored subset)
    reduction_pct: Optional[float] = None  # share of the raw detections the full pipeline does not report
    stored: int = 0  # raw detections kept for display (largest first)


class AblationCandidate(BaseModel):
    candidate_id: int
    bounds: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    area_px: int
    area_ha: Optional[float] = None
    mgrs: Optional[str] = None
    centroid: Optional[List[float]] = None
    geometry: Optional[Dict[str, Any]] = None  # outline of the changed pixels (EPSG:4326); None where only the box is known
    is_ablation: bool = True


class AblationList(BaseModel):
    scene_a_id: str
    scene_b_id: str
    total: int  # raw detections found
    shown: int
    candidates: List[AblationCandidate]


def _effective_status(job: Dict[str, Any], summary: Dict[str, Any]) -> str:
    recorded = summary["status"]
    if recorded == store.RUNNING:
        with _in_flight_lock:
            active = job["job_id"] in _in_flight
        return store.RUNNING if (active or EMBED_LOCK.locked()) else store.NOT_RUN
    return recorded


def _pair_job(conn: Any, scene_a_id: str, scene_b_id: str) -> Dict[str, Any]:
    job = store.find_job(conn, scene_a_id, scene_b_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="These two scenes have not been analysed as a pair")
    return job


@router.get("/changes/ablation/stats", response_model=AblationStats)
def ablation_stats(
    scene_a_id: str = Query(..., pattern=SCENE_ID_PATTERN), scene_b_id: str = Query(..., pattern=SCENE_ID_PATTERN)
) -> AblationStats:
    """Detections with the full pipeline against raw detections with suppression off."""
    conn = init_connection()
    try:
        init_schema(conn)
        job = _pair_job(conn, scene_a_id, scene_b_id)
        summary = store.get_summary(job)
        full = store.count_full(conn, job)
    finally:
        conn.close()

    state = _effective_status(job, summary)
    done = state == store.COMPLETED
    raw = int(summary.get("found", 0)) if done else 0
    return AblationStats(
        scene_a_id=job["scene_a_id"],
        scene_b_id=job["scene_b_id"],
        status=state,
        full_count=full,
        ablation_count=raw,
        reduction_pct=round((1.0 - full / raw) * 100.0, 1) if done and raw > 0 else None,
        stored=int(summary.get("stored", 0)) if done else 0,
    )


@router.get("/changes/ablation/candidates", response_model=AblationList)
def ablation_candidates(
    scene_a_id: str = Query(..., pattern=SCENE_ID_PATTERN),
    scene_b_id: str = Query(..., pattern=SCENE_ID_PATTERN),
    limit: int = Query(default=1000, ge=1, le=MAX_LIST, description="Largest raw detections to return"),
) -> AblationList:
    """The raw detections of a pair (suppression off), largest first."""
    conn = init_connection()
    try:
        init_schema(conn)
        job = _pair_job(conn, scene_a_id, scene_b_id)
        summary = store.get_summary(job)
        rows = store.list_ablation(conn, job["job_id"], limit)
    finally:
        conn.close()

    return AblationList(
        scene_a_id=job["scene_a_id"],
        scene_b_id=job["scene_b_id"],
        total=int(summary.get("found", 0)) if summary["status"] == store.COMPLETED else 0,
        shown=len(rows),
        candidates=[
            AblationCandidate(
                candidate_id=r["candidate_id"],
                bounds=[r["min_lon"], r["min_lat"], r["max_lon"], r["max_lat"]],
                area_px=r["area_px"] or 0,
                area_ha=r["area_ha"],
                mgrs=r["mgrs_ref"],
                centroid=[r["centroid_lon"], r["centroid_lat"]] if r["centroid_lon"] is not None else None,
                geometry=json.loads(r["geometry"]) if r["geometry"] else None,
            )
            for r in rows
        ],
    )


def _run_in_background(job_id: int, scene_a_id: str, scene_b_id: str) -> None:
    try:
        with EMBED_LOCK:  # one heavy raster job at a time, as for imports and scene deletion
            run_ablation(scene_a_id, scene_b_id)
    except Exception:
        logger.warning("Ablation for %s -> %s did not complete", scene_a_id, scene_b_id)  # recorded on the job
    finally:
        with _in_flight_lock:
            _in_flight.discard(job_id)


@router.post("/changes/ablation/run", response_model=AblationStats, status_code=status.HTTP_202_ACCEPTED)
def start_ablation(
    background: BackgroundTasks,
    scene_a_id: str = Query(..., pattern=SCENE_ID_PATTERN),
    scene_b_id: str = Query(..., pattern=SCENE_ID_PATTERN),
) -> AblationStats:
    """Start the ablation for a pair analysed before it ran automatically (or re-run one). Poll /ablation/stats."""
    registered = False
    conn = init_connection()
    try:
        init_schema(conn)
        job = _pair_job(conn, scene_a_id, scene_b_id)
        if job["status"] != "completed":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This pair's change detection has not completed")
        with _in_flight_lock:
            if job["job_id"] in _in_flight:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The ablation for this pair is already running")
            _in_flight.add(job["job_id"])
            registered = True
        store.mark_status(conn, job["job_id"], store.RUNNING)
        full = store.count_full(conn, job)
    except Exception:
        if registered:
            with _in_flight_lock:
                _in_flight.discard(job["job_id"])
        raise
    finally:
        conn.close()

    background.add_task(_run_in_background, job["job_id"], job["scene_a_id"], job["scene_b_id"])
    return AblationStats(
        scene_a_id=job["scene_a_id"], scene_b_id=job["scene_b_id"], status=store.RUNNING, full_count=full, ablation_count=0
    )
