"""IRIS Catalog — change-detection jobs, candidates and analyst reviews.

Same rules as database.py: every caller opens its own connection (init_connection) and takes the
write lock up front with BEGIN IMMEDIATE.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional

from mgrs_ref import bounds_centre_mgrs

# Job states: queued -> processing -> completed, or one of the terminal outcomes below
JOB_QUEUED = "queued"
JOB_PROCESSING = "processing"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"
JOB_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
JOB_ALIGNMENT_FAILED = "alignment_failed"
JOB_GRID_MISMATCH = "grid_mismatch"

# A finished job is never re-run: the same pair must not be processed twice (idempotency)
JOB_FINAL_STATES = frozenset(
    {JOB_COMPLETED, JOB_INSUFFICIENT_EVIDENCE, JOB_ALIGNMENT_FAILED, JOB_GRID_MISMATCH}
)

REVIEW_DECISIONS = ("confirmed", "rejected")

# Change types a candidate can carry today. Candidates stored before direction classification existed carry the
# old names (vegetation_loss, vegetation_gain); they are presented, and filtered, as "unclassified".
REFINED_TYPES = (
    "clearance", "construction", "revegetation", "urban_expansion", "water_loss", "vegetation_retreat", "unclassified",
)
_TYPE_SQL = (
    "CASE WHEN c.change_type IN (" + ",".join(f"'{t}'" for t in REFINED_TYPES) + ") THEN c.change_type ELSE 'unclassified' END"
)
_DIRECTION_SQL = "COALESCE(c.direction, 'unclassified')"

_JOB_COLUMNS = "job_id, scene_a_id, scene_b_id, status, created_at, started_at, completed_at, error_message, details"


def _job_row(row: tuple) -> Dict[str, Any]:
    keys = [c.strip() for c in _JOB_COLUMNS.split(",")]
    job = dict(zip(keys, row))
    job["details"] = json.loads(job["details"]) if job["details"] else {}
    return job


def get_job_for_pair(conn: sqlite3.Connection, scene_a_id: str, scene_b_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        f"SELECT {_JOB_COLUMNS} FROM jobs WHERE job_type='change_detection' AND scene_a_id=? AND scene_b_id=?",
        (scene_a_id, scene_b_id),
    ).fetchone()
    return _job_row(row) if row else None


def create_job(conn: sqlite3.Connection, scene_a_id: str, scene_b_id: str) -> Dict[str, Any]:
    """Enqueue the ordered pair (A older, B newer). An existing pair is left untouched (UNIQUE + DO NOTHING)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """
            INSERT INTO jobs (job_type, scene_a_id, scene_b_id, status)
            VALUES ('change_detection', ?, ?, 'queued')
            ON CONFLICT(job_type, scene_a_id, scene_b_id) DO NOTHING
            """,
            (scene_a_id, scene_b_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    job = get_job_for_pair(conn, scene_a_id, scene_b_id)
    assert job is not None
    return job


def list_jobs(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        f"SELECT {_JOB_COLUMNS} FROM jobs WHERE job_type='change_detection' ORDER BY job_id DESC"
    ).fetchall()
    return [_job_row(r) for r in rows]


def mark_job_processing(conn: sqlite3.Connection, job_id: int) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "UPDATE jobs SET status='processing', started_at=datetime('now'), completed_at=NULL, error_message=NULL "
            "WHERE job_id=?",
            (job_id,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def update_job_details(conn: sqlite3.Connection, job_id: int, details: Dict[str, Any]) -> None:
    """Persist progress/provenance while a job is still running (current phase, timings so far)."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("UPDATE jobs SET details=? WHERE job_id=?", (json.dumps(details), job_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def finish_job(
    conn: sqlite3.Connection,
    job_id: int,
    status: str,
    details: Dict[str, Any],
    error: Optional[str] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Record a job outcome. Candidates and the status change commit in one transaction so they can't disagree."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM change_candidates WHERE job_id = ?", (job_id,))
        for cand in candidates or []:
            conn.execute(
                """
                INSERT INTO change_candidates (
                    job_id, scene_a_id, scene_b_id,
                    min_x, min_y, max_x, max_y, min_lon, min_lat, max_lon, max_lat,
                    change_type, direction, confidence,
                    norm_rmse, norm_cluster_dist, terrain_flatness, valid_coverage,
                    alignment_quality, area_px, mean_dndvi, direction_evidence,
                    sub_blobs, centroid_lon, centroid_lat, mgrs_ref, geometry
                ) VALUES (
                    :job_id, :scene_a_id, :scene_b_id,
                    :min_x, :min_y, :max_x, :max_y, :min_lon, :min_lat, :max_lon, :max_lat,
                    :change_type, :direction, :confidence,
                    :norm_rmse, :norm_cluster_dist, :terrain_flatness, :valid_coverage,
                    :alignment_quality, :area_px, :mean_dndvi, :direction_evidence,
                    :sub_blobs, :centroid_lon, :centroid_lat, :mgrs_ref, :geometry
                )
                """,
                {
                    "sub_blobs": 1,
                    "centroid_lon": None,
                    "centroid_lat": None,
                    "mgrs_ref": None,
                    "geometry": None,
                    **cand,
                    "job_id": job_id,
                    "direction_evidence": json.dumps(cand["direction_evidence"]) if cand.get("direction_evidence") else None,
                },
            )
        conn.execute(
            "UPDATE jobs SET status=?, completed_at=datetime('now'), error_message=?, details=? WHERE job_id=?",
            (status, error, json.dumps(details), job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


_CANDIDATE_SELECT = """
    SELECT c.candidate_id, c.job_id, c.scene_a_id, c.scene_b_id,
           c.min_x, c.min_y, c.max_x, c.max_y, c.min_lon, c.min_lat, c.max_lon, c.max_lat,
           c.change_type, c.direction, c.confidence,
           c.norm_cluster_dist, c.terrain_flatness, c.valid_coverage, c.alignment_quality,
           c.area_px, c.mean_dndvi, c.created_at, c.direction_evidence,
           c.sub_blobs, c.centroid_lon, c.centroid_lat, c.mgrs_ref,
           sa.acquisition_date, sb.acquisition_date, sa.sensor,
           (SELECT r.decision FROM reviews r WHERE r.candidate_id = c.candidate_id
              ORDER BY r.review_id DESC LIMIT 1)
    FROM change_candidates c
    LEFT JOIN scenes sa ON sa.scene_id = c.scene_a_id
    LEFT JOIN scenes sb ON sb.scene_id = c.scene_b_id
"""

_CANDIDATE_KEYS = (
    "candidate_id job_id scene_a_id scene_b_id min_x min_y max_x max_y min_lon min_lat max_lon max_lat "
    "change_type direction confidence norm_cluster_dist terrain_flatness valid_coverage alignment_quality "
    "area_px mean_dndvi created_at direction_evidence sub_blobs centroid_lon centroid_lat mgrs_ref "
    "scene_a_date scene_b_date sensor review_status"
).split()


def _candidate_row(row: tuple) -> Dict[str, Any]:
    cand = dict(zip(_CANDIDATE_KEYS, row))
    cand["review_status"] = cand["review_status"] or "pending"
    cand["direction_evidence"] = json.loads(cand["direction_evidence"]) if cand["direction_evidence"] else None
    if cand["change_type"] not in REFINED_TYPES:
        cand["change_type"] = "unclassified"  # stored before direction classification: the old names no longer exist
    cand["direction"] = cand["direction"] or "unclassified"
    cand["sub_blobs"] = cand["sub_blobs"] or 1
    if not cand["mgrs_ref"]:
        # Stored before MGRS geocoding, or the centroid was not traced: the centre of the bounding box stands in
        cand["mgrs_ref"] = bounds_centre_mgrs([cand["min_lon"], cand["min_lat"], cand["max_lon"], cand["max_lat"]])
    return cand


def list_candidates(conn: sqlite3.Connection, limit: int = 1000) -> List[Dict[str, Any]]:
    rows = conn.execute(
        _CANDIDATE_SELECT + " ORDER BY c.confidence DESC, c.candidate_id ASC LIMIT ?", (limit,)
    ).fetchall()
    return [_candidate_row(r) for r in rows]


def count_candidates(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM change_candidates").fetchone()[0])


def get_candidate(conn: sqlite3.Connection, candidate_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(_CANDIDATE_SELECT + " WHERE c.candidate_id = ?", (candidate_id,)).fetchone()
    return _candidate_row(row) if row else None


def get_reviews(conn: sqlite3.Connection, candidate_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT review_id, decision, analyst_id, notes, reviewed_at FROM reviews "
        "WHERE candidate_id = ? ORDER BY review_id ASC",
        (candidate_id,),
    ).fetchall()
    return [
        {"review_id": r[0], "decision": r[1], "analyst_id": r[2], "notes": r[3], "reviewed_at": r[4]} for r in rows
    ]


def add_review(
    conn: sqlite3.Connection,
    candidate_id: int,
    decision: str,
    analyst_id: Optional[str],
    notes: Optional[str],
) -> None:
    """Append an analyst decision to the audit trail (earlier decisions are kept; the latest one wins)."""
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"decision must be one of {REVIEW_DECISIONS}")
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO reviews (candidate_id, decision, analyst_id, notes) VALUES (?, ?, ?, ?)",
            (candidate_id, decision, analyst_id, notes),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def query_candidates(
    conn: sqlite3.Connection,
    job_ids: Optional[List[int]] = None,
    min_confidence: float = 0.0,
    types: Optional[List[str]] = None,
    directions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Candidates of the given jobs (all jobs when None), strongest first, optionally filtered."""
    where: List[str] = []
    args: List[Any] = []
    if job_ids is not None:
        if not job_ids:
            return []
        where.append(f"c.job_id IN ({','.join('?' for _ in job_ids)})")
        args.extend(job_ids)
    if min_confidence > 0:
        where.append("c.confidence >= ?")
        args.append(min_confidence)
    if types:
        where.append(f"{_TYPE_SQL} IN ({','.join('?' for _ in types)})")
        args.extend(types)
    if directions:
        where.append(f"{_DIRECTION_SQL} IN ({','.join('?' for _ in directions)})")
        args.extend(directions)
    sql = _CANDIDATE_SELECT + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY c.confidence DESC, c.area_px DESC, c.candidate_id ASC"
    return [_candidate_row(r) for r in conn.execute(sql, args).fetchall()]
