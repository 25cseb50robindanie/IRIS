"""IRIS Catalog — SQLite with WAL, R-Tree, and job state machine.

IMPORTANT: every process/worker must call init_connection() on its OWN
connection AFTER being forked. Never inherit a connection across fork().
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/iris_catalog.db")


def init_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a new connection with the required pragmas.
    
    Call this INSIDE each worker process, never in the parent.
    """
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

        -- Tiles: one row per analysis tile
        CREATE TABLE IF NOT EXISTS tiles (
            tile_id         TEXT PRIMARY KEY,
            scene_id        TEXT NOT NULL REFERENCES scenes(scene_id),
            faiss_id        INTEGER,
            min_x           REAL NOT NULL,
            min_y           REAL NOT NULL,
            max_x           REAL NOT NULL,
            max_y           REAL NOT NULL,
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
    conn.commit()
