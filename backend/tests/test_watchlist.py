"""Tests for the watchlist: pinned locations, and the alerts raised when a completed job's detection lands on one."""

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rasterio.warp import transform as warp_points

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog import changes as store
from catalog import watchlist as wl
from catalog.database import delete_scene_rows, init_connection, init_schema
from change_detection import run_change_detection
from main import app
from s2_factory import Scenario, ingest_without_embedding, make_safe

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
CLEARED = (200, 260, 300, 380)  # rows, columns of the change, 10 m pixels from the scene origin


def where_the_change_will_be():
    """(lon, lat) of the centre of the CLEARED box: scenes start at UTM (300000, 3100000), zone 43N."""
    x, y = 300000.0 + 10.0 * 340, 3100000.0 - 10.0 * 230
    lons, lats = warp_points("EPSG:32643", "EPSG:4326", [x], [y])
    return float(lons[0]), float(lats[0])


def scenes(tmp_path):
    a = ingest_without_embedding(make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1, landcover=[(*CLEARED, "vegetation")])))
    b = ingest_without_embedding(make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, landcover=[(*CLEARED, "bare")])))
    return a, b


def watch(client, **body):
    resp = client.post("/api/watchlist", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---- adding, listing, removing ----------------------------------------------------------------------------------------------------


def test_a_location_is_added_from_a_centre_and_radius():
    client = TestClient(app)
    lon, lat = 73.0, 28.0
    w = watch(client, name="Airstrip", center=[lon, lat], radius_m=1000, confidence_threshold=0.6)
    assert (w["name"], w["confidence_threshold"], w["alerts"], w["unacknowledged_alerts"]) == ("Airstrip", 0.6, 0, 0)
    min_lon, min_lat, max_lon, max_lat = w["bounds"]
    assert (min_lat + max_lat) / 2 == pytest.approx(lat) and (min_lon + max_lon) / 2 == pytest.approx(lon)
    assert (max_lat - min_lat) * 111_320 / 2 == pytest.approx(1000, rel=1e-3)  # half the height is the radius
    assert (max_lon - min_lon) > (max_lat - min_lat)  # a degree of longitude is shorter at 28 N, so the box is wider in degrees


def test_a_location_is_added_from_bounds_and_named_by_default(tmp_path):
    client = TestClient(app)
    w = watch(client, bounds=[73.0, 28.0, 73.01, 28.01])
    assert w["bounds"] == [73.0, 28.0, 73.01, 28.01] and w["confidence_threshold"] == 0.5
    assert re.fullmatch(r"Watch 43R[A-Z]{2} \d{5} \d{5}", w["name"])  # the MGRS reference of its centre, as the map prints it
    assert [x["id"] for x in client.get("/api/watchlist").json()] == [w["id"]]


def test_bad_requests_are_rejected(tmp_path):
    client = TestClient(app)
    post = lambda **b: client.post("/api/watchlist", json=b).status_code  # noqa: E731
    assert post() == 422  # neither bounds nor centre
    assert post(bounds=[73, 28, 73.1, 28.1], center=[73, 28]) == 422  # both
    assert post(bounds=[73.1, 28, 73, 28.1]) == 422  # min above max
    assert post(bounds=[73, 28, 200, 28.1]) == 422  # off the map
    assert post(center=[73, 95]) == 422
    assert post(center=[73, 28], radius_m=0) == 422 and post(center=[73, 28], radius_m=10**6) == 422
    assert post(center=[73, 28], confidence_threshold=1.5) == 422 and post(center=[73, 28], confidence_threshold=-0.1) == 422
    assert post(center=[73, 28], name="x" * 101) == 422
    assert client.get("/api/watchlist").json() == []


def test_a_location_is_removed_and_a_second_removal_is_a_404():
    client = TestClient(app)
    w = watch(client, center=[73, 28])
    assert client.delete(f"/api/watchlist/{w['id']}").status_code == 204
    assert client.get("/api/watchlist").json() == []
    assert client.delete(f"/api/watchlist/{w['id']}").status_code == 404


def test_the_watchlist_has_a_size_limit(monkeypatch):
    monkeypatch.setattr(wl, "MAX_WATCHED", 2)
    client = TestClient(app)
    watch(client, center=[73, 28])
    watch(client, center=[74, 28])
    resp = client.post("/api/watchlist", json={"center": [75, 28]})
    assert resp.status_code == 409 and "full" in resp.json()["detail"]


# ---- alerts ----------------------------------------------------------------------------------------------------------------------------


def test_a_detection_on_a_watched_location_raises_an_alert_when_the_job_completes(tmp_path):
    a, b = scenes(tmp_path)
    client = TestClient(app)
    lon, lat = where_the_change_will_be()
    w = watch(client, name="Field 12", center=[lon, lat], radius_m=300, confidence_threshold=0.5)
    assert client.get("/api/watchlist/alerts").json() == {"total": 0, "unacknowledged": 0, "alerts": []}

    assert run_change_detection(a, b)["status"] == "completed"  # nobody is looking at the page

    body = client.get("/api/watchlist/alerts").json()
    assert (body["total"], body["unacknowledged"]) == (1, 1)
    (alert,) = body["alerts"]
    cand = client.get("/api/changes").json()["candidates"][0]
    assert (alert["watchlist_id"], alert["watch_name"], alert["candidate_id"]) == (w["id"], "Field 12", cand["candidate_id"])
    assert alert["confidence"] == pytest.approx(cand["confidence"]) and alert["confidence"] >= 0.5
    assert alert["bounds"] == cand["bounds"] and alert["mgrs"] == cand["mgrs"]
    assert (alert["change_type"], alert["direction"], alert["review_status"]) == ("clearance", "disappearance", "pending")
    assert (alert["scene_a_date"], alert["scene_b_date"]) == ("2024-01-10", "2024-02-14")
    assert alert["area_ha"] == pytest.approx(cand["area_ha"]) and alert["acknowledged_at"] is None

    listed = client.get("/api/watchlist").json()[0]
    assert (listed["alerts"], listed["unacknowledged_alerts"]) == (1, 1)


def test_no_alert_when_the_detection_is_elsewhere_or_below_the_threshold(tmp_path):
    a, b = scenes(tmp_path)
    client = TestClient(app)
    lon, lat = where_the_change_will_be()
    watch(client, name="Far away", center=[lon + 0.5, lat], radius_m=500)  # 50 km east
    watch(client, name="Demanding", center=[lon, lat], radius_m=300, confidence_threshold=0.999)
    run_change_detection(a, b)
    assert client.get("/api/changes").json()["total"] == 1  # there is a detection...
    assert client.get("/api/watchlist/alerts").json()["total"] == 0  # ...but neither location is hit by it


def test_the_threshold_is_inclusive_and_each_overlapping_location_raises_its_own_alert(tmp_path):
    a, b = scenes(tmp_path)
    client = TestClient(app)
    lon, lat = where_the_change_will_be()
    # the exact confidence is unknown until the job has run: watch with 0.5, then compare with a threshold set to it
    first = watch(client, name="Wide", bounds=[lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01], confidence_threshold=0.3)
    second = watch(client, name="Small", center=[lon, lat], radius_m=50, confidence_threshold=0.0)
    run_change_detection(a, b)
    alerts = client.get("/api/watchlist/alerts").json()["alerts"]
    assert sorted(x["watch_name"] for x in alerts) == ["Small", "Wide"]
    assert {x["watchlist_id"] for x in alerts} == {first["id"], second["id"]}
    confidence = alerts[0]["confidence"]

    conn = init_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE watchlist SET confidence_threshold = ? WHERE id = ?", (confidence, first["id"]))  # exactly equal
        job_id = store.list_jobs(conn)[0]["job_id"]
        wl.raise_alerts_for_job(conn, job_id)
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM watchlist_alerts WHERE watchlist_id = ?", (first["id"],)).fetchone()[0] == 1
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE watchlist SET confidence_threshold = ? WHERE id = ?", (confidence + 1e-6, first["id"]))  # just above
        wl.raise_alerts_for_job(conn, job_id)
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM watchlist_alerts WHERE watchlist_id = ?", (first["id"],)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM watchlist_alerts").fetchone()[0] == 1  # checking again never duplicates
    finally:
        conn.close()


def test_a_location_added_after_the_job_does_not_alert_on_what_was_already_found(tmp_path):
    a, b = scenes(tmp_path)
    client = TestClient(app)
    run_change_detection(a, b)
    lon, lat = where_the_change_will_be()
    watch(client, center=[lon, lat], radius_m=300)
    assert client.get("/api/watchlist/alerts").json()["total"] == 0  # alerts are for NEW detections, raised by new jobs


def test_alerts_can_be_acknowledged_one_at_a_time_or_all_at_once(tmp_path):
    a, b = scenes(tmp_path)
    client = TestClient(app)
    lon, lat = where_the_change_will_be()
    watch(client, name="One", center=[lon, lat], radius_m=300)
    watch(client, name="Two", center=[lon, lat], radius_m=400)
    run_change_detection(a, b)
    alerts = client.get("/api/watchlist/alerts").json()["alerts"]
    assert len(alerts) == 2

    assert client.post(f"/api/watchlist/alerts/{alerts[0]['alert_id']}/acknowledge").json() == {"acknowledged": 1}
    assert client.post(f"/api/watchlist/alerts/{alerts[0]['alert_id']}/acknowledge").json() == {"acknowledged": 0}  # already seen
    body = client.get("/api/watchlist/alerts").json()
    assert (body["total"], body["unacknowledged"]) == (2, 1)
    only_open = client.get("/api/watchlist/alerts", params={"unacknowledged_only": "true"}).json()["alerts"]
    assert [x["alert_id"] for x in only_open] == [alerts[1]["alert_id"]]

    assert client.post("/api/watchlist/alerts/acknowledge").json() == {"acknowledged": 1}
    assert client.get("/api/watchlist/alerts").json()["unacknowledged"] == 0
    assert client.post("/api/watchlist/alerts/424242/acknowledge").status_code == 404


def test_removing_a_location_removes_its_alerts(tmp_path):
    a, b = scenes(tmp_path)
    client = TestClient(app)
    lon, lat = where_the_change_will_be()
    keep = watch(client, name="Keep", center=[lon, lat], radius_m=300)
    drop = watch(client, name="Drop", center=[lon, lat], radius_m=300)
    run_change_detection(a, b)
    assert client.get("/api/watchlist/alerts").json()["total"] == 2
    assert client.delete(f"/api/watchlist/{drop['id']}").status_code == 204
    alerts = client.get("/api/watchlist/alerts").json()["alerts"]
    assert [x["watchlist_id"] for x in alerts] == [keep["id"]]


def test_a_job_that_is_run_again_replaces_its_alerts_instead_of_tripping_over_them(tmp_path):
    a, b = scenes(tmp_path)
    client = TestClient(app)
    lon, lat = where_the_change_will_be()
    watch(client, center=[lon, lat], radius_m=300)
    run_change_detection(a, b)
    conn = init_connection()
    try:
        job = store.list_jobs(conn)[0]
        # a re-run deletes the old candidates, which the alert points at; the alert must go first
        store.finish_job(conn, job["job_id"], store.JOB_COMPLETED, job["details"], candidates=[])
        assert conn.execute("SELECT COUNT(*) FROM watchlist_alerts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM change_candidates").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_job_that_did_not_complete_raises_nothing(tmp_path):
    a, b = scenes(tmp_path)
    client = TestClient(app)
    lon, lat = where_the_change_will_be()
    watch(client, center=[lon, lat], radius_m=300)
    conn = init_connection()
    try:
        init_schema(conn)
        job = store.create_job(conn, a, b)
        store.finish_job(conn, job["job_id"], store.JOB_FAILED, {"phase": "done"}, error="boom")
    finally:
        conn.close()
    assert client.get("/api/watchlist/alerts").json()["total"] == 0


def test_deleting_a_scene_removes_the_alerts_on_its_detections(tmp_path):
    a, b = scenes(tmp_path)
    client = TestClient(app)
    lon, lat = where_the_change_will_be()
    watch(client, center=[lon, lat], radius_m=300)
    run_change_detection(a, b)
    conn = init_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        counts = delete_scene_rows(conn, b)
        conn.commit()
        assert counts["watchlist_alerts"] == 1
    finally:
        conn.close()
    assert client.get("/api/watchlist/alerts").json()["total"] == 0
    assert len(client.get("/api/watchlist").json()) == 1  # the location itself stays: it is the analyst's, not the scene's
