"""Tests for Part D: the change API (list, detail, jobs, review) end to end through POST /api/ingest."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import app
from s2_factory import Scenario, make_safe

NAME_A = "S2A_MSIL2A_20240110T050649_N0510_R019_T43PHM_20240110T090000"
NAME_B = "S2A_MSIL2A_20240214T050649_N0510_R019_T43PHM_20240214T090000"
CLEARED = (200, 260, 300, 380)  # vegetation on the first date, bare ground on the second


@pytest.fixture
def client_with_changes(tmp_path):
    """Ingest two overlapping same-sensor scenes through the API; the second one triggers change detection."""
    client = TestClient(app)
    a = make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1, landcover=[(*CLEARED, "vegetation")]))
    b = make_safe(
        tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, gain=1.05, landcover=[(*CLEARED, "bare")])
    )
    assert client.post("/api/ingest", json={"file_path": str(a)}).status_code == 200
    assert client.post("/api/ingest", json={"file_path": str(b)}).status_code == 200
    return client


def test_ingest_triggers_change_detection_and_lists_candidates(client_with_changes):
    client = client_with_changes

    listing = client.get("/api/changes")
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 1
    cand = body["candidates"][0]
    assert cand["scene_a_id"] == NAME_A and cand["scene_b_id"] == NAME_B
    assert cand["scene_a_date"] == "2024-01-10" and cand["scene_b_date"] == "2024-02-14"
    assert cand["review_status"] == "pending"
    assert cand["direction"] == "disappearance" and cand["change_type"] == "clearance"
    min_lon, min_lat, max_lon, max_lat = cand["bounds"]  # WGS84, ready for MapLibre
    assert 72 < min_lon < max_lon < 74 and 27 < min_lat < max_lat < 29
    assert 0 < cand["confidence"] <= 1


def test_candidate_detail_has_scenes_breakdown_and_trace(client_with_changes):
    client = client_with_changes
    cid = client.get("/api/changes").json()["candidates"][0]["candidate_id"]

    resp = client.get(f"/api/changes/{cid}")
    assert resp.status_code == 200
    d = resp.json()

    assert d["scene_a"]["scene_id"] == NAME_A and d["scene_b"]["scene_id"] == NAME_B
    assert d["scene_a"]["acquisition_date"] < d["scene_b"]["acquisition_date"]
    for scene in (d["scene_a"], d["scene_b"]):
        assert scene["cog_url"].startswith("file:///") and scene["bounds_wgs84"]
        assert Path(scene["cog_url"][len("file:///"):]).is_file()

    breakdown = d["confidence_breakdown"]
    assert set(breakdown) == {"alignment_quality", "cluster_distance", "terrain_flatness", "valid_coverage"}
    # the four terms sum to the score, times whatever the seasonal filter applied (a clearance with no history: x0.85)
    assert d["seasonality_status"] == "unverified" and d["confidence_factor"] == 0.85
    assert sum(t["contribution"] for t in breakdown.values()) * d["confidence_factor"] == pytest.approx(d["confidence"], abs=1e-6)
    assert breakdown["alignment_quality"]["weight"] == 0.35 and breakdown["terrain_flatness"]["weight"] == 0.15
    assert d["terrain_is_placeholder"] is True

    # both scenes carry tile T43PHM, so alignment is skipped by definition
    assert d["decision_trace"]["alignment"]["method"] == "same_tile"
    assert d["bounds_native"][0] < d["bounds_native"][2]
    assert d["reviews"] == []


def test_unknown_candidate_is_404(client_with_changes):
    assert client_with_changes.get("/api/changes/999999").status_code == 404
    assert client_with_changes.post("/api/changes/999999/review", json={"decision": "confirmed"}).status_code == 404


def test_confirm_reject_are_persisted_with_an_audit_trail(client_with_changes):
    client = client_with_changes
    cid = client.get("/api/changes").json()["candidates"][0]["candidate_id"]

    resp = client.post(f"/api/changes/{cid}/review", json={"decision": "confirmed", "notes": "  new clearing  "})
    assert resp.status_code == 200
    assert resp.json()["review_status"] == "confirmed"

    resp = client.post(f"/api/changes/{cid}/review", json={"decision": "rejected"})
    assert resp.json()["review_status"] == "rejected"  # the latest decision wins...

    detail = client.get(f"/api/changes/{cid}").json()
    assert [r["decision"] for r in detail["reviews"]] == ["confirmed", "rejected"]  # ...but both are kept
    assert detail["reviews"][0]["notes"] == "new clearing"
    assert detail["reviews"][0]["analyst_id"]  # local operator identifier
    assert detail["reviews"][0]["reviewed_at"]
    assert client.get("/api/changes").json()["candidates"][0]["review_status"] == "rejected"


def test_review_decision_is_validated(client_with_changes):
    client = client_with_changes
    cid = client.get("/api/changes").json()["candidates"][0]["candidate_id"]
    assert client.post(f"/api/changes/{cid}/review", json={"decision": "maybe"}).status_code == 422
    assert client.post(f"/api/changes/{cid}/review", json={}).status_code == 422
    assert client.post(f"/api/changes/{cid}/review", json={"decision": "confirmed", "notes": "x" * 1001}).status_code == 422
    assert client.get(f"/api/changes/{cid}").json()["reviews"] == []


def test_jobs_endpoint_reports_outcomes_so_no_result_is_not_mistaken_for_no_change(tmp_path):
    client = TestClient(app)
    a = make_safe(tmp_path / "src", NAME_A, Scenario(date="2024-01-10", noise_seed=1))
    b = make_safe(tmp_path / "src", NAME_B, Scenario(date="2024-02-14", noise_seed=2, nodata_rows=400))
    client.post("/api/ingest", json={"file_path": str(a)})
    client.post("/api/ingest", json={"file_path": str(b)})

    assert client.get("/api/changes").json()["candidates"] == []
    jobs = client.get("/api/changes/jobs").json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "insufficient_evidence"
    assert "30%" in jobs[0]["error"]
    assert jobs[0]["candidates"] == 0


def test_empty_catalog_has_no_changes():
    client = TestClient(app)
    body = client.get("/api/changes").json()
    assert (body["total"], body["candidates"], body["pairs"]) == (0, [], [])
    assert client.get("/api/changes/jobs").json() == {"jobs": []}


def test_jobs_carry_the_acquisition_dates_of_each_pair(client_with_changes):
    job = client_with_changes.get("/api/changes/jobs").json()["jobs"][0]
    assert (job["scene_a_date"], job["scene_b_date"]) == ("2024-01-10", "2024-02-14")
