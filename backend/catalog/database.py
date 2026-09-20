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
            analysis_cog_path TEXT,  -- 4-band B02/B03/B04/B08 COG used by change detection (Sentinel-2 SAFE only)
            scl_path        TEXT,    -- SCL raster resampled to 10m with nearest-neighbour (Sentinel-2 SAFE only)
            cloud_pct       REAL,
            raw_checksum    TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        -- Scene footprints in WGS84 for overlap lookups when pairing scenes for change detection.
        -- id is scenes.rowid. (Tile bounds are in per-scene native CRS, so they cannot be compared across scenes.)
        CREATE VIRTUAL TABLE IF NOT EXISTS scene_footprints USING rtree(
            id,
            min_lon, max_lon,
            min_lat, max_lat
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
            details         TEXT,  -- JSON: phase timings, alignment method, thresholds (decision trace)
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
            min_lon         REAL,
            min_lat         REAL,
            max_lon         REAL,
            max_lat         REAL,
            alignment_quality REAL,  -- ECC rho, or ORB inlier ratio when ECC failed
            area_px         INTEGER,
            mean_dndvi      REAL,    -- signed mean NDVI(B) - NDVI(A) inside the blob; NULL without a NIR band
            direction_evidence TEXT, -- JSON: dominant classes at each date, class shares, rule that fired
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

    _ensure_columns(cursor, "scenes", {"analysis_cog_path": "TEXT", "scl_path": "TEXT"})
    _ensure_columns(cursor, "jobs", {"details": "TEXT"})
    _ensure_columns(
        cursor,
        "change_candidates",
        {
            "min_lon": "REAL",
            "min_lat": "REAL",
            "max_lon": "REAL",
            "max_lat": "REAL",
            "alignment_quality": "REAL",
            "area_px": "INTEGER",
            "mean_dndvi": "REAL",
            "direction_evidence": "TEXT",
        },
    )

    conn.commit()


def _ensure_columns(cursor: sqlite3.Cursor, table: str, columns: Dict[str, str]) -> None:
    """Add any missing columns to an existing table (idempotent schema migration)."""
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    for name, col_type in columns.items():
        if name not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")


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


SCENE_COLUMNS = (
    "scene_id, sensor, acquisition_date, crs, file_path, cog_path, analysis_cog_path, scl_path, cloud_pct, raw_checksum"
)


def upsert_scene(conn: sqlite3.Connection, scene: Dict[str, Any], bounds_wgs84: Optional[List[float]] = None) -> None:
    """Insert or update a scene row and its WGS84 footprint. Call inside the caller's write transaction.

    Re-importing a scene makes it the most recent one (created_at is refreshed at millisecond precision).
    """
    conn.execute(
        """
        INSERT INTO scenes (
            scene_id, sensor, acquisition_date, crs, file_path, cog_path,
            analysis_cog_path, scl_path, cloud_pct, raw_checksum
        ) VALUES (
            :scene_id, :sensor, :acquisition_date, :crs, :file_path, :cog_path,
            :analysis_cog_path, :scl_path, :cloud_pct, :raw_checksum
        )
        ON CONFLICT(scene_id) DO UPDATE SET
            sensor=excluded.sensor,
            acquisition_date=excluded.acquisition_date,
            crs=excluded.crs,
            file_path=excluded.file_path,
            cog_path=excluded.cog_path,
            analysis_cog_path=excluded.analysis_cog_path,
            scl_path=excluded.scl_path,
            cloud_pct=excluded.cloud_pct,
            raw_checksum=excluded.raw_checksum,
            created_at=strftime('%Y-%m-%d %H:%M:%f', 'now');
        """,
        scene,
    )
    if bounds_wgs84 and len(bounds_wgs84) == 4:
        rowid = conn.execute("SELECT rowid FROM scenes WHERE scene_id = ?", (scene["scene_id"],)).fetchone()[0]
        conn.execute("DELETE FROM scene_footprints WHERE id = ?", (rowid,))
        min_lon, min_lat, max_lon, max_lat = bounds_wgs84
        conn.execute(
            "INSERT INTO scene_footprints (id, min_lon, max_lon, min_lat, max_lat) VALUES (?, ?, ?, ?, ?)",
            (rowid, min_lon, max_lon, min_lat, max_lat),
        )


def get_scene(conn: sqlite3.Connection, scene_id: str) -> Optional[Dict[str, Any]]:
    """One scene row as a dict, or None."""
    row = conn.execute(f"SELECT {SCENE_COLUMNS} FROM scenes WHERE scene_id = ?", (scene_id,)).fetchone()
    if row is None:
        return None
    return dict(zip([c.strip() for c in SCENE_COLUMNS.split(",")], row))


def find_overlapping_scenes(conn: sqlite3.Connection, scene_id: str) -> List[Dict[str, Any]]:
    """Other scenes from the same sensor whose WGS84 footprint intersects this scene's (R-Tree lookup).

    Each result carries the overlap as a fraction of the smaller footprint's area (bounding-box approximation).
    """
    cursor = conn.execute(
        """
        SELECT s2.scene_id, s2.sensor, s2.acquisition_date,
               f1.min_lon, f1.max_lon, f1.min_lat, f1.max_lat,
               f2.min_lon, f2.max_lon, f2.min_lat, f2.max_lat
        FROM scenes s1
        JOIN scene_footprints f1 ON f1.id = s1.rowid
        JOIN scene_footprints f2
             ON f2.max_lon >= f1.min_lon AND f2.min_lon <= f1.max_lon
            AND f2.max_lat >= f1.min_lat AND f2.min_lat <= f1.max_lat
        JOIN scenes s2 ON s2.rowid = f2.id
        WHERE s1.scene_id = ? AND s2.scene_id != s1.scene_id AND s2.sensor = s1.sensor
        """,
        (scene_id,),
    )
    results = []
    for row in cursor.fetchall():
        a_minx, a_maxx, a_miny, a_maxy, b_minx, b_maxx, b_miny, b_maxy = row[3:]
        ix = max(0.0, min(a_maxx, b_maxx) - max(a_minx, b_minx))
        iy = max(0.0, min(a_maxy, b_maxy) - max(a_miny, b_miny))
        smaller = min((a_maxx - a_minx) * (a_maxy - a_miny), (b_maxx - b_minx) * (b_maxy - b_miny))
        results.append(
            {
                "scene_id": row[0],
                "sensor": row[1],
                "acquisition_date": row[2],
                "overlap_fraction": (ix * iy / smaller) if smaller > 0 else 0.0,
            }
        )
    return results


def list_scene_summaries(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Every scene with what depends on it, most recent acquisition first (for the scene switcher)."""
    cursor = conn.execute(
        """
        SELECT s.scene_id, s.sensor, s.acquisition_date, s.crs, s.cog_path, s.analysis_cog_path, s.scl_path,
               s.cloud_pct, s.created_at,
               (SELECT COUNT(*) FROM tiles t WHERE t.scene_id = s.scene_id),
               (SELECT COUNT(*) FROM change_candidates c WHERE c.scene_a_id = s.scene_id OR c.scene_b_id = s.scene_id),
               (SELECT COUNT(*) FROM reviews r JOIN change_candidates c ON c.candidate_id = r.candidate_id
                 WHERE c.scene_a_id = s.scene_id OR c.scene_b_id = s.scene_id)
        FROM scenes s
        ORDER BY s.acquisition_date DESC, s.created_at DESC, s.rowid DESC
        """
    )
    keys = [
        "scene_id", "sensor", "acquisition_date", "crs", "cog_path", "analysis_cog_path", "scl_path",
        "cloud_pct", "created_at", "tiles_count", "change_candidates", "reviews",
    ]
    return [dict(zip(keys, row)) for row in cursor.fetchall()]


def list_tiles_except_scene(conn: sqlite3.Connection, scene_id: str) -> List[Tuple[str, int]]:
    """(tile_id, faiss_id) of every embedded tile outside `scene_id`, in faiss_id order."""
    rows = conn.execute(
        "SELECT tile_id, faiss_id FROM tiles WHERE faiss_id IS NOT NULL AND scene_id != ? ORDER BY faiss_id",
        (scene_id,),
    ).fetchall()
    return [(r[0], int(r[1])) for r in rows]


def delete_scene_rows(conn: sqlite3.Connection, scene_id: str) -> Dict[str, int]:
    """Delete a scene and everything that hangs off it. Call inside the caller's write transaction.

    Foreign keys run child-first: reviews -> change candidates -> jobs -> tiles -> footprint -> scene. Analyst
    reviews of a deleted scene's candidates go with them (their candidate row cannot exist without the scene).
    """
    counts: Dict[str, int] = {}
    cand_ids = [
        r[0]
        for r in conn.execute(
            "SELECT candidate_id FROM change_candidates WHERE scene_a_id = ? OR scene_b_id = ?", (scene_id, scene_id)
        ).fetchall()
    ]
    counts["reviews"] = counts["change_candidates"] = 0
    for i in range(0, len(cand_ids), 500):  # stay well under SQLite's variable limit
        chunk = cand_ids[i : i + 500]
        marks = ",".join("?" for _ in chunk)
        counts["reviews"] += conn.execute(f"DELETE FROM reviews WHERE candidate_id IN ({marks})", chunk).rowcount
        counts["change_candidates"] += conn.execute(
            f"DELETE FROM change_candidates WHERE candidate_id IN ({marks})", chunk
        ).rowcount
    counts["jobs"] = conn.execute(
        "DELETE FROM jobs WHERE scene_a_id = ? OR scene_b_id = ?", (scene_id, scene_id)
    ).rowcount
    counts["tiles"] = conn.execute("DELETE FROM tiles WHERE scene_id = ?", (scene_id,)).rowcount
    row = conn.execute("SELECT rowid FROM scenes WHERE scene_id = ?", (scene_id,)).fetchone()
    if row is not None:
        conn.execute("DELETE FROM scene_footprints WHERE id = ?", (row[0],))
    counts["scenes"] = conn.execute("DELETE FROM scenes WHERE scene_id = ?", (scene_id,)).rowcount
    return counts


def remap_tile_faiss_ids(conn: sqlite3.Connection, kept: List[Tuple[str, int]]) -> None:
    """After the index is rebuilt from the kept vectors, kept[i] lives at faiss_id i."""
    conn.executemany("UPDATE tiles SET faiss_id = ? WHERE tile_id = ?", [(new, tile_id) for new, (tile_id, _) in enumerate(kept)])


def list_scene_tiles(conn: sqlite3.Connection, scene_id: str) -> List[Dict[str, Any]]:
    """Every embedded tile of a scene with its WGS84 bounds: the allowlist for scene-scoped search."""
    rows = conn.execute(
        "SELECT tile_id, faiss_id, min_lon, min_lat, max_lon, max_lat FROM tiles "
        "WHERE scene_id = ? AND faiss_id IS NOT NULL AND min_lon IS NOT NULL ORDER BY faiss_id",
        (scene_id,),
    ).fetchall()
    return [
        {"tile_id": r[0], "faiss_id": int(r[1]), "bounds": [r[2], r[3], r[4], r[5]]} for r in rows
    ]
