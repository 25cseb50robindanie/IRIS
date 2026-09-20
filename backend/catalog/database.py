"""IRIS Catalog — SQLite with WAL, R-Tree, and job state machine.

IMPORTANT: every process/worker must call init_connection() on its OWN
connection AFTER being forked. Never inherit a connection across fork().
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = Path("data/iris_catalog.db")


def init_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a new connection with the required pragmas.
    
    Call this INSIDE each worker process, never in the parent.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist. Idempotent."""
    conn.executescript("""
        -- Scenes: one row per imported satellite scene
        CREATE TABLE IF NOT EXISTS scenes (
            scene_id        TEXT PRIMARY KEY,
            sensor          TEXT NOT NULL,
            acquisition_date TEXT NOT NULL,
            crs             TEXT NOT NULL,
            file_path       TEXT NOT NULL,
            cog_path        TEXT,
            cloud_pct       REAL,
            raw_checksum    TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        -- Tiles: one row per analysis tile / embedding crop
        CREATE TABLE IF NOT EXISTS tiles (
            tile_id         TEXT PRIMARY KEY,
            scene_id        TEXT NOT NULL REFERENCES scenes(scene_id),
            faiss_id        INTEGER,
            min_x           REAL NOT NULL,
            min_y           REAL NOT NULL,
            max_x           REAL NOT NULL,
            max_y           REAL NOT NULL,
            min_lon         REAL,
            min_lat         REAL,
            max_lon         REAL,
            max_lat         REAL,
            cloud_pct       REAL,
            valid_pixel_frac REAL,
            FOREIGN KEY (scene_id) REFERENCES scenes(scene_id)
        );

        -- R-Tree spatial index on tile bounding boxes
        CREATE VIRTUAL TABLE IF NOT EXISTS tiles_rtree USING rtree(
            id,
            min_x, max_x,
            min_y, max_y
        );

        -- Index on faiss_id for fast vector search joins
        CREATE INDEX IF NOT EXISTS idx_tiles_faiss_id ON tiles(faiss_id);

        -- Job state machine for background processing
        CREATE TABLE IF NOT EXISTS jobs (
            job_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_type        TEXT NOT NULL,  -- 'change_detection', 'embedding'
            scene_a_id      TEXT,
            scene_b_id      TEXT,
            status          TEXT NOT NULL DEFAULT 'queued',
                            -- queued -> processing -> completed | failed
            created_at      TEXT DEFAULT (datetime('now')),
            started_at      TEXT,
            completed_at    TEXT,
            error_message   TEXT,
            -- Prevent duplicate jobs for the same pair
            UNIQUE(job_type, scene_a_id, scene_b_id)
        );

        -- Change candidates: precomputed results
        CREATE TABLE IF NOT EXISTS change_candidates (
            candidate_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          INTEGER NOT NULL REFERENCES jobs(job_id),
            scene_a_id      TEXT NOT NULL,
            scene_b_id      TEXT NOT NULL,
            min_x           REAL NOT NULL,
            min_y           REAL NOT NULL,
            max_x           REAL NOT NULL,
            max_y           REAL NOT NULL,
            change_type     TEXT,  -- construction, vegetation_clearance, water_change, linear_feature
            direction       TEXT,  -- appearance, disappearance, expansion, contraction
            confidence      REAL NOT NULL,
            norm_rmse       REAL,
            norm_cluster_dist REAL,
            terrain_flatness REAL,
            valid_coverage  REAL,
            earliest_date   TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        -- Analyst review decisions (audit trail)
        CREATE TABLE IF NOT EXISTS reviews (
            review_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id    INTEGER NOT NULL REFERENCES change_candidates(candidate_id),
            decision        TEXT NOT NULL,  -- 'confirmed', 'rejected'
            analyst_id      TEXT,
            notes           TEXT,
            reviewed_at     TEXT DEFAULT (datetime('now'))
        );
    """)

    # Schema migration for existing databases: check and add missing columns
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(tiles)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    
    for col in ["min_lon", "min_lat", "max_lon", "max_lat"]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE tiles ADD COLUMN {col} REAL")

    conn.commit()


def insert_tiles_batch(conn: sqlite3.Connection, tiles_data: List[Dict[str, Any]]) -> None:
    """Batch insert tile records into SQLite catalog."""
    if not tiles_data:
        return

    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO tiles (
            tile_id,
            scene_id,
            faiss_id,
            min_x,
            min_y,
            max_x,
            max_y,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            cloud_pct,
            valid_pixel_frac
        ) VALUES (
            :tile_id,
            :scene_id,
            :faiss_id,
            :min_x,
            :min_y,
            :max_x,
            :max_y,
            :min_lon,
            :min_lat,
            :max_lon,
            :max_lat,
            :cloud_pct,
            :valid_pixel_frac
        )
        ON CONFLICT(tile_id) DO UPDATE SET
            faiss_id=excluded.faiss_id,
            min_x=excluded.min_x,
            min_y=excluded.min_y,
            max_x=excluded.max_x,
            max_y=excluded.max_y,
            min_lon=excluded.min_lon,
            min_lat=excluded.min_lat,
            max_lon=excluded.max_lon,
            max_lat=excluded.max_lat,
            valid_pixel_frac=excluded.valid_pixel_frac;
        """,
        tiles_data,
    )


def get_tiles_by_faiss_ids(conn: sqlite3.Connection, faiss_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Lookup tile metadata by a list of faiss_ids."""
    if not faiss_ids:
        return {}

    placeholders = ",".join("?" for _ in faiss_ids)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT 
            tile_id, scene_id, faiss_id,
            min_x, min_y, max_x, max_y,
            min_lon, min_lat, max_lon, max_lat,
            valid_pixel_frac
        FROM tiles
        WHERE faiss_id IN ({placeholders})
        """,
        faiss_ids,
    )

    results: Dict[int, Dict[str, Any]] = {}
    for row in cursor.fetchall():
        fid = row[2]
        results[fid] = {
            "tile_id": row[0],
            "scene_id": row[1],
            "faiss_id": fid,
            "bounds_native": [row[3], row[4], row[5], row[6]],
            "bounds_wgs84": [row[7], row[8], row[9], row[10]] if row[7] is not None else [row[3], row[4], row[5], row[6]],
            "valid_pixel_frac": row[11],
        }
    return results


def get_scene_tile_count(conn: sqlite3.Connection, scene_id: str) -> Optional[int]:
    """Number of embedded tiles for a scene, or None if the scene isn't in the catalog."""
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM scenes WHERE scene_id = ?", (scene_id,))
    if cursor.fetchone() is None:
        return None
    cursor.execute("SELECT COUNT(*) FROM tiles WHERE scene_id = ?", (scene_id,))
    return int(cursor.fetchone()[0])


def list_scenes_newest_first(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """All scenes, most recently imported first, each with its embedded-tile count."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            s.scene_id, s.sensor, s.acquisition_date, s.crs, s.cog_path,
            (SELECT COUNT(*) FROM tiles t WHERE t.scene_id = s.scene_id) AS tiles_count
        FROM scenes s
        ORDER BY s.created_at DESC, s.rowid DESC
        """
    )
    return [
        {
            "scene_id": row[0],
            "sensor": row[1],
            "acquisition_date": row[2],
            "crs": row[3],
            "cog_path": row[4],
            "tiles_count": int(row[5]),
        }
        for row in cursor.fetchall()
    ]


def count_tiles(conn: sqlite3.Connection) -> int:
    """Total embedded tile rows across all scenes."""
    return int(conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0])
