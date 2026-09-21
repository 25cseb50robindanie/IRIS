"""IRIS Catalog — ablation runs: the raw detections of a pair with every suppression stage switched off.

Same rules as database.py: every caller opens its own connection and takes the write lock up front. The run's outcome
(status, the true number of raw detections, what was stored) lives in the pair's job `details` under "ablation", so it
travels with the rest of the decision trace.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional

from catalog import changes as jobs

# Outcome of a run, as stored in job.details["ablation"]["status"]
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
NOT_RUN = "not_run"  # a pair analysed before ablation existed, or one whose job has not completed

_COLUMNS = (
    "candidate_id job_id scene_a_id scene_b_id min_x min_y max_x max_y min_lon min_lat max_lon max_lat area_px area_ha "
    "centroid_lon centroid_lat mgrs_ref geometry"
).split()


def find_job(conn: sqlite3.Connection, scene_a_id: str, scene_b_id: str) -> Optional[Dict[str, Any]]:
    """The change-detection job of a pair, whichever order the scenes are given in."""
    return jobs.get_job_for_pair(conn, scene_a_id, scene_b_id) or jobs.get_job_for_pair(conn, scene_b_id, scene_a_id)


def _set_details(conn: sqlite3.Connection, job_id: int, ablation: Dict[str, Any]) -> None:
    """Merge the "ablation" block into the job's details. Call inside the caller's write transaction."""
    row = conn.execute("SELECT details FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    details = json.loads(row[0]) if row and row[0] else {}
    details["ablation"] = ablation
    conn.execute("UPDATE jobs SET details = ? WHERE job_id = ?", (json.dumps(details), job_id))


def mark_status(conn: sqlite3.Connection, job_id: int, status: str, **extra: Any) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        _set_details(conn, job_id, {"status": status, **extra})
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def save_run(
    conn: sqlite3.Connection, job_id: int, scene_a_id: str, scene_b_id: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]
) -> None:
    """Replace the pair's ablation candidates and record the run's summary, in one transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM ablation_candidates WHERE job_id = ?", (job_id,))
        conn.executemany(
            """
            INSERT INTO ablation_candidates (
                job_id, scene_a_id, scene_b_id, min_x, min_y, max_x, max_y,
                min_lon, min_lat, max_lon, max_lat, area_px, area_ha, sub_blobs,
                centroid_lon, centroid_lat, mgrs_ref, geometry, is_ablation
            ) VALUES (
                :job_id, :scene_a_id, :scene_b_id, :min_x, :min_y, :max_x, :max_y,
                :min_lon, :min_lat, :max_lon, :max_lat, :area_px, :area_ha, 1,
                :centroid_lon, :centroid_lat, :mgrs_ref, :geometry, 1
            )
            """,
            [{**r, "job_id": job_id, "scene_a_id": scene_a_id, "scene_b_id": scene_b_id} for r in rows],
        )
        _set_details(conn, job_id, {"status": COMPLETED, **summary})
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_summary(job: Dict[str, Any]) -> Dict[str, Any]:
    """{"status", "found", "stored", ...} of a job's ablation run; status is NOT_RUN when there has been none."""
    block = (job.get("details") or {}).get("ablation")
    if not block:
        return {"status": NOT_RUN}
    return block


def count_full(conn: sqlite3.Connection, job: Dict[str, Any]) -> int:
    """Detections the full pipeline reports for this job: those stored, plus any the storage cap alone left out.

    The cap (MAX_STORED_CANDIDATES) is a limit on what is kept, not a suppression stage, so counting only what survived it
    would credit the suppression stack with more than it removed.
    """
    stored = int(conn.execute("SELECT COUNT(*) FROM change_candidates WHERE job_id = ?", (job["job_id"],)).fetchone()[0])
    over_cap = int(((job.get("details") or {}).get("scoring") or {}).get("dropped_over_cap") or 0)
    return stored + over_cap


def list_ablation(conn: sqlite3.Connection, job_id: int, limit: int) -> List[Dict[str, Any]]:
    """The pair's stored raw detections, largest first."""
    rows = conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM ablation_candidates WHERE job_id = ? "
        "ORDER BY area_px DESC, candidate_id ASC LIMIT ?",
        (job_id, limit),
    ).fetchall()
    out = [dict(zip(_COLUMNS, r)) for r in rows]
    for r in out:
        if r["area_ha"] is None and r["area_px"]:
            r["area_ha"] = r["area_px"] / 100.0  # stored before pixel size was recorded: 10 m assumed
    return out
