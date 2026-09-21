"""Tests for the ablation: the same pair with every suppression stage off, and the endpoints that report it."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import ablation as store
from catalog.database import delete_scene_rows, init_connection, init_schema
from change_detection import params, run_change_detection
from change_detection.ablation import run_ablation, simplify_outline
from change_detection.trigger import run_ablations, schedule_change_detection
from main import app
from s2_factory import Scenario, ingest_without_embedding, make_safe

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
CLEARED = (200, 260, 300, 380)  # a real vegetation -> bare change
# Four separate 60 x 60 clouds over scene B: the full pipeline masks them, a raw subtraction reads each as a change
CLOUDS = [(20, 80, 20, 80), (20, 80, 140, 200), (100, 160, 20, 80), (100, 160, 140, 200)]


@pytest.fixture(autouse=True)
def no_large_area_merge(monkeypatch):
    monkeypatch.setattr(params, "LARGE_AREA_MIN_BLOBS", 10**6)
    monkeypatch.setattr(params, "LARGE_AREA_MIN_PIXELS", 10**12)


def cloudy_pair(tmp_path, run_full=True):
    """A real change plus four clouds in the newer scene, and a radiometric difference between the dates."""
    a = ingest_without_embedding(
        make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1, landcover=[(*CLEARED, "vegetation")]))
    )
    b_scn = Scenario(date="2024-02-14", noise_seed=2, gain=1.05, offset=20, landcover=[(*CLEARED, "bare")], clouds=CLOUDS)
    b = ingest_without_embedding(make_safe(tmp_path / "src", NAME_B, b_scn))
    if run_full:
        run_change_detection(a, b)
    return a, b


def test_ablation_reports_more_raw_detections_than_the_full_pipeline(tmp_path):
    a, b = cloudy_pair(tmp_path)
    client = TestClient(app)

    # Before it has run, the pair reports "not run" and no raw count
    early = client.get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert early["status"] == "not_run" and early["ablation_count"] == 0 and early["reduction_pct"] is None
    assert early["full_count"] == 1  # the cloud-masked pipeline reports the one real change

    result = run_ablation(a, b)
    assert result["found"] >= len(CLOUDS)

    stats = client.get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert stats["status"] == "completed"
    assert stats["full_count"] == 1
    assert stats["ablation_count"] == result["found"] > stats["full_count"]
    assert stats["reduction_pct"] == pytest.approx((1 - 1 / stats["ablation_count"]) * 100, abs=0.06)
    assert (stats["scene_a_id"], stats["scene_b_id"]) == (a, b)
    # either order finds the pair
    swapped = client.get("/api/changes/ablation/stats", params={"scene_a_id": b, "scene_b_id": a}).json()
    assert swapped["ablation_count"] == stats["ablation_count"]


def test_raw_detections_include_what_the_suppression_stack_removes(tmp_path):
    a, b = cloudy_pair(tmp_path)
    run_ablation(a, b)
    client = TestClient(app)
    body = client.get("/api/changes/ablation/candidates", params={"scene_a_id": a, "scene_b_id": b}).json()

    assert body["total"] == body["shown"] == len(body["candidates"]) >= len(CLOUDS)
    areas = [c["area_px"] for c in body["candidates"]]
    assert areas == sorted(areas, reverse=True)  # largest first
    # each 60 x 60 (3,600 px) cloud is one raw blob, untouched by masking or merging; where the landscape under a cloud
    # was already bright the raw difference is smaller, so a blob can lose a few percent of its pixels
    clouds = [a for a in areas if a > 3000]
    assert len(clouds) == len(CLOUDS) and max(clouds) <= 3600
    for c in body["candidates"]:
        assert c["is_ablation"] is True and c["area_px"] >= params.MIN_BLOB_PIXELS  # the MMU still applies
        assert len(c["bounds"]) == 4 and c["bounds"][0] < c["bounds"][2]
        assert c["geometry"]["type"] in ("Polygon", "MultiPolygon")
        assert c["mgrs"] and len(c["centroid"]) == 2

    assert client.get("/api/changes/ablation/candidates", params={"scene_a_id": a, "scene_b_id": b, "limit": 2}).json()["shown"] == 2


def test_full_count_adds_back_what_only_the_storage_cap_dropped(tmp_path):
    a, b = cloudy_pair(tmp_path)
    run_ablation(a, b)
    conn = init_connection()
    try:
        job = store.find_job(conn, a, b)
        details = job["details"]
        details["scoring"]["dropped_over_cap"] = 6  # the cap, not the suppression stack, left six detections out
        conn.execute("UPDATE jobs SET details = ? WHERE job_id = ?", (json.dumps(details), job["job_id"]))
        conn.commit()
    finally:
        conn.close()
    stats = TestClient(app).get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert stats["full_count"] == 1 + 6
    assert stats["reduction_pct"] == pytest.approx((1 - 7 / stats["ablation_count"]) * 100, abs=0.06)


def test_outlines_are_simplified_but_keep_their_shape():
    # a 100 x 100 unit square walked in 1-unit steps (a pixel staircase), with a jagged 1-unit notch on one side
    ring = [[float(x), 0.0] for x in range(0, 101)] + [[100.0, float(y)] for y in range(1, 101)]
    ring += [[float(x), 100.0] for x in range(99, -1, -1)] + [[0.0, float(y)] for y in range(99, -1, -1)]
    geometry = json.dumps({"type": "Polygon", "coordinates": [ring]})
    simple = json.loads(simplify_outline(geometry, tolerance=2.0))
    assert simple["type"] == "Polygon"
    exterior = simple["coordinates"][0]
    assert len(exterior) < 12 < len(ring) and exterior[0] == exterior[-1]
    xs, ys = [p[0] for p in exterior], [p[1] for p in exterior]
    assert (min(xs), max(xs), min(ys), max(ys)) == (0.0, 100.0, 0.0, 100.0)  # the corners survive

    tiny = json.dumps({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]})
    assert json.loads(simplify_outline(tiny, tolerance=5.0))["coordinates"][0][0] == [0, 0]  # never collapsed away
    multi = json.dumps({"type": "MultiPolygon", "coordinates": [[ring], [ring]]})
    assert json.loads(simplify_outline(multi, tolerance=2.0))["type"] == "MultiPolygon"

    # a wiggly ring far over the vertex budget is simplified harder rather than served in full
    wiggle = [[float(i), 5.0 * ((i // 2) % 2)] for i in range(0, 4000)] + [[3999.0, -10.0], [0.0, -10.0], [0.0, 0.0]]
    capped = json.loads(simplify_outline(json.dumps({"type": "Polygon", "coordinates": [wiggle]}), tolerance=0.5, max_vertices=200))
    assert len(capped["coordinates"][0]) <= 200 and capped["coordinates"][0][0] == capped["coordinates"][0][-1]

    hole = [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0], [40.0, 40.0]]
    holed = json.dumps({"type": "Polygon", "coordinates": [ring, hole]})
    assert len(json.loads(simplify_outline(holed, tolerance=2.0))["coordinates"]) == 1  # holes are left out of a map outline


def test_ablation_rows_are_flagged_and_do_not_touch_the_full_results(tmp_path):
    a, b = cloudy_pair(tmp_path)
    client = TestClient(app)
    before = client.get("/api/changes").json()["candidates"]
    run_ablation(a, b)
    assert client.get("/api/changes").json()["candidates"] == before

    conn = init_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM ablation_candidates WHERE is_ablation = 1").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM ablation_candidates WHERE is_ablation != 1").fetchone()[0] == 0
    finally:
        conn.close()


def test_running_it_again_replaces_the_stored_rows(tmp_path):
    a, b = cloudy_pair(tmp_path)
    first = run_ablation(a, b)
    second = run_ablation(a, b)
    assert first["found"] == second["found"]
    conn = init_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM ablation_candidates").fetchone()[0] == second["stored"]
    finally:
        conn.close()


def test_the_stored_subset_is_capped_but_the_total_is_not(tmp_path, monkeypatch):
    a, b = cloudy_pair(tmp_path)
    monkeypatch.setattr(params, "MAX_ABLATION_STORED", 2)
    result = run_ablation(a, b)
    assert result["stored"] == 2 and result["found"] >= len(CLOUDS)
    stats = TestClient(app).get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert stats["ablation_count"] == result["found"] and stats["stored"] == 2


def test_ablation_needs_a_completed_pair(tmp_path):
    a, b = cloudy_pair(tmp_path, run_full=False)
    with pytest.raises(ValueError):
        run_ablation(a, b)
    client = TestClient(app)
    assert client.get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).status_code == 404
    assert client.post("/api/changes/ablation/run", params={"scene_a_id": a, "scene_b_id": b}).status_code == 404


def test_endpoints_validate_their_input(tmp_path):
    client = TestClient(app)
    assert client.get("/api/changes/ablation/stats").status_code == 422
    assert client.get("/api/changes/ablation/stats", params={"scene_a_id": "../../x/y", "scene_b_id": "b"}).status_code == 422
    assert client.get("/api/changes/ablation/candidates", params={"scene_a_id": "a", "scene_b_id": "b", "limit": 0}).status_code == 422
    assert client.get("/api/changes/ablation/stats", params={"scene_a_id": "a", "scene_b_id": "b"}).status_code == 404


def test_post_run_starts_the_ablation_for_an_already_analysed_pair(tmp_path):
    a, b = cloudy_pair(tmp_path)
    client = TestClient(app)
    resp = client.post("/api/changes/ablation/run", params={"scene_a_id": a, "scene_b_id": b})
    assert resp.status_code == 202 and resp.json()["status"] == "running"
    # the background task has finished by the time the test client returns
    stats = client.get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert stats["status"] == "completed" and stats["ablation_count"] > stats["full_count"]


def test_a_stale_running_record_reads_as_not_run(tmp_path):
    a, b = cloudy_pair(tmp_path)
    conn = init_connection()
    try:
        job = store.find_job(conn, a, b)
        store.mark_status(conn, job["job_id"], store.RUNNING)
    finally:
        conn.close()
    stats = TestClient(app).get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert stats["status"] == "not_run"  # nothing is running: the process that started it is gone


def test_a_failed_run_is_recorded_and_reported(tmp_path):
    a, b = cloudy_pair(tmp_path)
    conn = init_connection()
    try:
        conn.execute("UPDATE scenes SET analysis_cog_path = 'data/cogs/missing.tif' WHERE scene_id = ?", (b,))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ValueError):
        run_ablation(a, b)
    stats = TestClient(app).get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert stats["status"] == "failed" and stats["ablation_count"] == 0


def test_it_runs_automatically_after_a_pair_completes(tmp_path):
    a = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1, landcover=[(*CLEARED, "vegetation")])))
    b = ingest_without_embedding(make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, landcover=[(*CLEARED, "bare")])))
    results = schedule_change_detection(b)
    assert [(r["scene_a_id"], r["scene_b_id"], r["status"]) for r in results] == [(a, b, "completed")]
    assert run_ablations(results) == 1
    stats = TestClient(app).get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert stats["status"] == "completed"

    # a pair that was not just analysed (an earlier, finished job) is left alone
    assert run_ablations(schedule_change_detection(b)) == 0


# The layout of an ablation_candidates table found in a real catalog, made by other code: no is_ablation column, a NOT NULL
# confidence with a default, and its own indexes. CREATE TABLE IF NOT EXISTS keeps such a table as it is.
FOREIGN_ABLATION_TABLE = """
CREATE TABLE ablation_candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER REFERENCES jobs(job_id),
    scene_a_id TEXT NOT NULL, scene_b_id TEXT NOT NULL,
    min_x REAL NOT NULL, min_y REAL NOT NULL, max_x REAL NOT NULL, max_y REAL NOT NULL,
    change_type TEXT DEFAULT 'unclassified', direction TEXT DEFAULT 'unclassified', confidence REAL NOT NULL DEFAULT 0.5,
    norm_rmse REAL, norm_cluster_dist REAL, terrain_flatness REAL, valid_coverage REAL, earliest_date TEXT,
    min_lon REAL, min_lat REAL, max_lon REAL, max_lat REAL, alignment_quality REAL, area_px INTEGER, mean_dndvi REAL,
    direction_evidence TEXT, sub_blobs INTEGER DEFAULT 1, centroid_lon REAL, centroid_lat REAL, mgrs_ref TEXT, geometry TEXT,
    created_at TEXT DEFAULT (datetime('now')), ablation_run BOOLEAN DEFAULT TRUE
);
CREATE INDEX idx_ablation_candidates_pair ON ablation_candidates(scene_a_id, scene_b_id);
"""


def test_an_existing_table_with_another_layout_is_adopted(tmp_path):
    conn = init_connection()
    try:
        conn.executescript(FOREIGN_ABLATION_TABLE)
    finally:
        conn.close()
    a, b = cloudy_pair(tmp_path)  # the catalog is initialised (and migrated) while scenes are ingested
    result = run_ablation(a, b)
    assert result["found"] >= len(CLOUDS)

    stats = TestClient(app).get("/api/changes/ablation/stats", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert stats["status"] == "completed" and stats["ablation_count"] == result["found"]
    body = TestClient(app).get("/api/changes/ablation/candidates", params={"scene_a_id": a, "scene_b_id": b}).json()
    assert body["shown"] == result["stored"] and all(c["is_ablation"] for c in body["candidates"])
    conn = init_connection()
    try:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(ablation_candidates)")}
        assert {"is_ablation", "ablation_run", "seasonality_status"} <= columns  # theirs kept, ours added
    finally:
        conn.close()


def test_deleting_a_scene_removes_its_ablation_rows(tmp_path):
    a, b = cloudy_pair(tmp_path)
    run_ablation(a, b)
    conn = init_connection()
    try:
        init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        counts = delete_scene_rows(conn, b)
        conn.commit()
        assert counts["ablation_candidates"] > 0
        assert conn.execute("SELECT COUNT(*) FROM ablation_candidates").fetchone()[0] == 0
    finally:
        conn.close()
