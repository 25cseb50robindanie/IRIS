"""IRIS Catalog — the analyst's watchlist: locations to keep an eye on, and the alerts raised when a new detection lands there.

A watched location is a WGS84 box (the columns are called min_x/min_y/max_x/max_y as the brief has them, but hold lon/lat) with
a confidence threshold. Every time a change-detection job completes, its new detections are checked against every location:
a detection whose box overlaps a watched box and whose confidence reaches that location's threshold raises an alert. The check
runs inside finish_job's own transaction (see catalog/changes.py), so no completed job can be missed, and a job that is
re-run replaces its alerts along with its candidates.

Same rules as database.py: every caller opens its own connection and takes the write lock up front with BEGIN IMMEDIATE.
"""

import math
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence

from mgrs_ref import to_mgrs

MAX_WATCHED = 500  # a watchlist is a short list of places that matter, not a second catalog
METRES_PER_DEGREE_LAT = 111_320.0


def box_around(lon: float, lat: float, radius_m: float) -> List[float]:
    """[min_lon, min_lat, max_lon, max_lat] of the square that circumscribes a circle of `radius_m` metres around a point."""
    dlat = radius_m / METRES_PER_DEGREE_LAT
    dlon = radius_m / (METRES_PER_DEGREE_LAT * max(math.cos(math.radians(lat)), 0.01))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def _spaced(mgrs: Optional[str]) -> Optional[str]:
    """43RCL0139196701 -> "43RCL 01391 96701", the grouping the map and the status bar print."""
    m = re.fullmatch(r"(\d{1,2}[C-X][A-Z]{2})(\d+)", mgrs or "")
    if not m or len(m.group(2)) % 2:
        return mgrs
    half = len(m.group(2)) // 2
    return f"{m.group(1)} {m.group(2)[:half]} {m.group(2)[half:]}"


def add_location(
    conn: sqlite3.Connection, name: Optional[str], bounds: Sequence[float], confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """Add a watched location; the name defaults to its MGRS reference. Raises ValueError when the list is full."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0] >= MAX_WATCHED:
            raise ValueError(f"The watchlist is full ({MAX_WATCHED} locations); remove one first")
        label = (name or "").strip() or f"Watch {_spaced(to_mgrs((bounds[1] + bounds[3]) / 2.0, (bounds[0] + bounds[2]) / 2.0)) or 'location'}"
        cur = conn.execute(
            "INSERT INTO watchlist (name, min_x, min_y, max_x, max_y, confidence_threshold) VALUES (?, ?, ?, ?, ?, ?)",
            (label, bounds[0], bounds[1], bounds[2], bounds[3], confidence_threshold),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return get_location(conn, int(cur.lastrowid))  # type: ignore[return-value]


def _location(row: tuple) -> Dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "bounds": [row[2], row[3], row[4], row[5]],
        "confidence_threshold": row[6],
        "created_at": row[7],
        "alerts": row[8],
        "unacknowledged_alerts": row[9],
    }


_LOCATION_SELECT = """
    SELECT w.id, w.name, w.min_x, w.min_y, w.max_x, w.max_y, w.confidence_threshold, w.created_at,
           (SELECT COUNT(*) FROM watchlist_alerts a WHERE a.watchlist_id = w.id),
           (SELECT COUNT(*) FROM watchlist_alerts a WHERE a.watchlist_id = w.id AND a.acknowledged_at IS NULL)
    FROM watchlist w
"""


def get_location(conn: sqlite3.Connection, watch_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(_LOCATION_SELECT + " WHERE w.id = ?", (watch_id,)).fetchone()
    return _location(row) if row else None


def list_locations(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    return [_location(r) for r in conn.execute(_LOCATION_SELECT + " ORDER BY w.id").fetchall()]


def delete_location(conn: sqlite3.Connection, watch_id: int) -> bool:
    """Remove a location and its alerts. True when it existed."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM watchlist_alerts WHERE watchlist_id = ?", (watch_id,))
        removed = conn.execute("DELETE FROM watchlist WHERE id = ?", (watch_id,)).rowcount > 0
        conn.commit()
        return removed
    except Exception:
        conn.rollback()
        raise


def raise_alerts_for_job(conn: sqlite3.Connection, job_id: int) -> int:
    """Alerts for the candidates of `job_id` that overlap a watched box and reach its threshold. Returns how many were raised.

    Call inside the caller's write transaction, after the job's candidates are in place. Earlier alerts of the same job (it is
    being re-run) are replaced.
    """
    conn.execute("DELETE FROM watchlist_alerts WHERE job_id = ?", (job_id,))
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO watchlist_alerts (watchlist_id, candidate_id, job_id, confidence)
        SELECT w.id, c.candidate_id, c.job_id, c.confidence
        FROM change_candidates c
        JOIN watchlist w
          ON c.min_lon <= w.max_x AND c.max_lon >= w.min_x AND c.min_lat <= w.max_y AND c.max_lat >= w.min_y
        WHERE c.job_id = ? AND c.confidence >= w.confidence_threshold
        """,
        (job_id,),
    )
    return int(cur.rowcount)


def list_alerts(conn: sqlite3.Connection, unacknowledged_only: bool = False) -> List[Dict[str, Any]]:
    """Alerts, newest first, each with the watched location and enough of the candidate to show it."""
    rows = conn.execute(
        f"""
        SELECT a.alert_id, a.watchlist_id, w.name, a.candidate_id, a.job_id, a.confidence, a.created_at, a.acknowledged_at,
               c.min_lon, c.min_lat, c.max_lon, c.max_lat, c.change_type, c.direction, c.area_px, c.area_ha, c.mgrs_ref,
               sa.acquisition_date, sb.acquisition_date,
               (SELECT r.decision FROM reviews r WHERE r.candidate_id = c.candidate_id ORDER BY r.review_id DESC LIMIT 1)
        FROM watchlist_alerts a
        JOIN watchlist w ON w.id = a.watchlist_id
        JOIN change_candidates c ON c.candidate_id = a.candidate_id
        LEFT JOIN scenes sa ON sa.scene_id = c.scene_a_id
        LEFT JOIN scenes sb ON sb.scene_id = c.scene_b_id
        {"WHERE a.acknowledged_at IS NULL" if unacknowledged_only else ""}
        ORDER BY a.alert_id DESC
        """
    ).fetchall()
    keys = (
        "alert_id watchlist_id watch_name candidate_id job_id confidence created_at acknowledged_at "
        "min_lon min_lat max_lon max_lat change_type direction area_px area_ha mgrs scene_a_date scene_b_date review_status"
    ).split()
    out = []
    for row in rows:
        a = dict(zip(keys, row))
        if a["area_ha"] is None and a["area_px"]:
            a["area_ha"] = a["area_px"] / 100.0  # stored before pixel size was recorded: 10 m assumed
        a["review_status"] = a["review_status"] or "pending"
        out.append(a)
    return out


def acknowledge(conn: sqlite3.Connection, alert_id: Optional[int] = None) -> int:
    """Mark one alert (or, with None, every open alert) as seen. Returns how many changed."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if alert_id is None:
            n = conn.execute("UPDATE watchlist_alerts SET acknowledged_at = datetime('now') WHERE acknowledged_at IS NULL").rowcount
        else:
            n = conn.execute(
                "UPDATE watchlist_alerts SET acknowledged_at = datetime('now') WHERE alert_id = ? AND acknowledged_at IS NULL",
                (alert_id,),
            ).rowcount
        conn.commit()
        return int(n)
    except Exception:
        conn.rollback()
        raise
